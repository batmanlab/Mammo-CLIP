import logging
import numpy as np
import os
import torch
from scipy.special import softmax
from sklearn import metrics
from sklearn.metrics import auc, precision_recall_curve
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from typing import Dict, List, Union

from .data import DataModule
from .model import build_model

log = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, config: Dict, ckpt_path):
        super(Evaluator).__init__()

        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # load ckpt config
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.ckpt_config = ckpt["config"]

        # load dataset
        self.datamodule = DataModule(
            data_config=self.config["data_test"],
            dataloader_config=self.config["dataloader"],
            tokenizer_config=self.ckpt_config["tokenizer"] if "tokenizer" in self.ckpt_config else None,
            transform_config=self.config["transform"] if "transform" in self.config else self.ckpt_config["transform"],
            mean=self.config["base"]["mean"],
            std=self.config["base"]["std"],
            image_encoder_type=self.ckpt_config["model"]["image_encoder"]["model_type"],
            cur_fold=self.config["base"]["fold"]
        )

        self.test_dataloader_dict = self.datamodule.valid_dataloader()

        assert len(self.test_dataloader_dict) > 0

        # load model
        self.model = build_model(
            model_config=self.ckpt_config["model"],
            loss_config=self.ckpt_config["loss"],
            tokenizer=self.datamodule.tokenizer
        )
        self.model = self.model.to(self.device)
        log.info(self.model)

    def evaluate_img_text_retieval(self, checkpoint, test_dataset_name, save_path=save_path):
        print(xxx)

    def evaluate_clip(self, checkpoint, test_dataset_name, file_name=None, save_path=None):
        log.info(f"<<<<<==============================================================================>>>>>")
        log.info(f"Load model {checkpoint}")
        ckpt = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(ckpt["model"], strict=False)
        self.model.eval()
        dataloader = self.test_dataloader_dict[test_dataset_name]

        idx = 0
        image_embeddings = []
        text_embeddings = []
        texts = []
        label_names = []
        labels = []
        for batch in tqdm(dataloader):
            if self.ckpt_config["model"]["image_encoder"]["model_type"].lower() == "swin":
                batch["images"] = batch["images"].squeeze(1).permute(0, 3, 1, 2)

            idx += 1
            img_emb = self.encode_image(batch["images"])
            image_embeddings.append(img_emb)
            if "texts" in batch:
                texts += batch["texts"]
            if "text_tokens" in batch:
                text_emb = self.encode_text(batch["text_tokens"])
                text_embeddings.append(text_emb)
            if "label_names" in batch:
                label_names.extend(batch["label_names"])
            if "labels" in batch:
                labels.extend(batch["labels"].numpy())

            if idx == 20:
                break
        image_embeddings = np.concatenate(image_embeddings, axis=0)
        file_name = file_name.split("/")[-1]
        file_name = "img-emb-" + file_name.split(".")[0] + ".npy"
        np.save(os.path.join(save_path, file_name), image_embeddings)
        log.info(f"image_embeddings shape: {image_embeddings.shape}")
        class_list = [self.config["data_test"][test_dataset_name]["label_col"]]
        prompts = self.config["data_test"][test_dataset_name]["label_text"][class_list[0]]

        results = {}
        if test_dataset_name in {"rsna"}:
            results["zeroshot_binary"] = self.zeroshot_binary(image_embeddings, label_names, class_list, prompts,
                                                              labels)

        log.info("Results")
        log.info(results["zeroshot_binary"])
        log.info(f"<<<<<==============================================================================>>>>>")
        return results

    def encode_image(self, image: torch.Tensor):
        with torch.no_grad():
            img_emb = self.model.encode_image(image.to(self.device))
            img_emb = self.model.image_projection(img_emb) if self.model.projection else img_emb
            img_emb = img_emb / torch.norm(img_emb, dim=1, keepdim=True)
        return img_emb.detach().cpu().numpy()

    def encode_text(self, text_token: Union[str, List[str], Dict, torch.Tensor]):
        if isinstance(text_token, str) or isinstance(text_token, list):
            text_token = self.datamodule.tokenizer(
                text_token, padding="longest", truncation=True, return_tensors="pt",
                max_length=self.ckpt_config["base"]["text_max_length"]
            )

        with torch.no_grad():
            text_emb = self.model.encode_text(text_token.to(self.device))
            text_emb = self.model.text_projection(text_emb) if self.model.projection else text_emb
            text_emb = text_emb / torch.norm(text_emb, dim=1, keepdim=True)
        return text_emb.detach().cpu().numpy()

    def zeroshot_binary(self, image_embeddings: np.ndarray, label_names: list, class_list: list, prompts: list, labels):
        log.info("Evaluating zero-shot binary classification")
        if type(label_names[0]) is not list:
            label_names = [[label] for label in label_names]

        result = {}
        for class_name in class_list:
            text_embeddings = self.encode_text(prompts)
            similarities = metrics.pairwise.cosine_similarity(image_embeddings, text_embeddings)
            similarities = softmax(similarities, axis=1)

            y_true = labels

            result[class_name] = {}
            fpr, tpr, thresholds = metrics.roc_curve(y_true, similarities[:, 1])
            result[class_name]["AUROC"] = metrics.auc(fpr, tpr)
            result[class_name]["Accuracy"] = metrics.accuracy_score(y_true, np.argmax(similarities, axis=1))
            result[class_name]["F1"] = metrics.f1_score(y_true, np.argmax(similarities, axis=1))

            y_pred = similarities[:, 1]
            y_true = np.array(y_true)
            positive_indices = np.where(y_true == 1)[0]
            predicted_pred = y_pred[positive_indices]
            predicted_pred = np.where(predicted_pred >= 0.5, 1, 0)
            predicted_true = y_true[positive_indices]

            accuracy_for_positives = np.sum(predicted_pred == predicted_true) / predicted_true.shape[0]

            pF = pfbeta_binarized(gt=y_true, pred=y_pred)
            prauc = pr_auc(gt=y_true, pred=y_pred)
            aucroc = auroc(gt=y_true, pred=y_pred)

            result[class_name]["ACC_POSITIVES"] = accuracy_for_positives
            result[class_name]["pF"] = pF
            result[class_name]["prauc"] = prauc

        return classification_score(result)


def classification_score(result: dict):
    for key, value in result.items():
        for sub_key, sub_value in value.items():
            if isinstance(sub_value, np.float32):
                value[sub_key] = float(sub_value)
    auroc = np.mean([value["AUROC"] for value in result.values()])
    f1 = np.mean([value["F1"] for value in result.values()])
    acc = np.mean([value["Accuracy"] for value in result.values()])
    acc_pos = np.mean([value["ACC_POSITIVES"] for value in result.values()])
    pF = np.mean([value["pF"] for value in result.values()])
    prauc = np.mean([value["prauc"] for value in result.values()])

    result["ACC_POSITIVES(Avg)"] = acc_pos
    result["pF(Avg)"] = pF
    result["prauc(Avg)"] = prauc
    result["AUROC(Avg)"] = auroc
    result["F1(Avg)"] = f1
    result["Accuracy(Avg)"] = acc

    s = "\n".join(f"{k}: {v}" for k, v in result.items())
    log.info(s)

    return result


def multiclass_classification(preds: np.ndarray, labels: np.ndarray, class_list: list):
    log.info("evaluate multi-class classification")
    preds_args = np.argmax(preds, axis=1)

    class_dict = {class_name: {"total_num": 0, "correct_num": 0} for class_name in class_list}
    for idx, class_name in enumerate(class_list):
        class_dict[class_name]["total_num"] = labels[:, idx].sum()
        class_dict[class_name]["correct_num"] = (labels[:, idx] * (preds_args == idx)).sum()

    total_num = len(labels)
    correct_num = sum([v["correct_num"] for v in class_dict.values()])

    result = {k: v["correct_num"] / v["total_num"] for k, v in class_dict.items()}
    result["Accuracy(Macro)"] = np.mean(list(result.values()))
    result["Accuracy(Micro)"] = correct_num / total_num  # same with macro due to same total_num
    s = " / ".join([f"{c}: {v:.3f}" for c, v in result.items()])
    log.info(s)

    return result


def pfbeta_binarized(gt, pred):
    positives = pred[gt == 1]
    scores = []
    for th in positives:
        binarized = (pred >= th).astype('int')
        score = pfbeta(gt, binarized, 1)
        scores.append(score)

    return np.max(scores)


def auroc(gt, pred):
    return roc_auc_score(gt, pred)


def pfbeta(gt, pred, beta):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(gt)):
        prediction = min(max(pred[idx], 0), 1)
        if (gt[idx]):
            y_true_count += 1
            ctp += prediction
            # cfp += 1 - prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0


def pr_auc(gt, pred, get_all=False):
    precision, recall, _ = precision_recall_curve(gt, pred)
    score = auc(recall, precision)
    if get_all:
        return score, precision, recall
    else:
        return score
