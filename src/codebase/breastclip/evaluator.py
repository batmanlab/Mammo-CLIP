import logging
from typing import Dict, List, Union

import numpy as np
import torch
from scipy.special import softmax
from sklearn import metrics
from sklearn.metrics import auc, precision_recall_curve, accuracy_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

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
        self.clip_image_encoder = None
        if self.ckpt_config["model"]["image_encoder"]["model_type"] == "swin":
            self.clip_image_encoder = self.ckpt_config["model"]["image_encoder"]["model_type"]
        elif (
                self.ckpt_config["model"]["image_encoder"]["name"] == "resnet101" or
                self.ckpt_config["model"]["image_encoder"]["name"] == "resnet152" or
                self.ckpt_config["model"]["image_encoder"]["name"] == "tf_efficientnet_b5_ns-detect" or
                self.ckpt_config["model"]["image_encoder"]["name"] == "tf_efficientnetv2-detect"
        ):
            self.clip_image_encoder = self.ckpt_config["model"]["image_encoder"]["name"]
        # load dataset
        self.datamodule = DataModule(
            data_config=self.config["data_test"],
            dataloader_config=self.config["dataloader"],
            tokenizer_config=self.ckpt_config["tokenizer"] if "tokenizer" in self.ckpt_config else None,
            transform_config=self.config["transform"] if "transform" in self.config else self.ckpt_config["transform"],
            mean=self.config["base"]["mean"],
            std=self.config["base"]["std"],
            image_encoder_type=self.clip_image_encoder,
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

    def get_embeddings(self, checkpoint, test_dataset_name):
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
        mass = []
        calc = []
        density = []
        cancer = []
        for batch in tqdm(dataloader):
            if (
                    self.clip_image_encoder == "swin" or
                    self.clip_image_encoder == "resnet101" or
                    self.clip_image_encoder == "resnet152" or
                    self.clip_image_encoder == "tf_efficientnet_b5_ns-detect" or
                    self.clip_image_encoder == "tf_efficientnetv2-detect"
            ):
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
            if "mass" in batch:
                mass.extend(batch["mass"].numpy())
            if "calc" in batch:
                calc.extend(batch["calc"].numpy())
            if "density" in batch:
                density.extend(batch["density"].numpy())
            if "cancer" in batch:
                cancer.extend(batch["cancer"].numpy())

        image_embeddings = np.concatenate(image_embeddings, axis=0)
        if len(text_embeddings) > 0:
            text_embeddings = np.concatenate(text_embeddings, axis=0)

        return {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
            "texts": texts,
            "label_names": label_names,
            "labels": labels,
            "mass": mass,
            "calc": calc,
            "density": density,
            "cancer": cancer
        }

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

    def eval_zeroshot(self, checkpoint, test_dataset_name, zs_prompts, save_path):
        emb = self.get_embeddings(checkpoint, test_dataset_name)

        print(f"Load model {checkpoint}")
        ckpt = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(ckpt["model"], strict=False)
        self.model.eval()
        image_embeddings = emb["image_embeddings"]

        print("image_embeddings.shape", image_embeddings.shape)

        print("evaluate zero shot clip")
        results = {}
        log.info(zs_prompts)
        for label_text in zs_prompts:
            print(f"Evaluating {label_text}")
            label_prompts = list(zs_prompts[label_text])
            label_tokens = self.datamodule.tokenizer(
                label_prompts, padding="longest", truncation=True, return_tensors="pt",
                max_length=256
            )
            with torch.no_grad():
                text_emb = self.model.encode_text(label_tokens.to(self.device))
                text_emb = self.model.text_projection(text_emb) if self.model.projection else text_emb
                text_emb = text_emb / torch.norm(text_emb, dim=1, keepdim=True)
                text_emb = text_emb.detach().cpu().numpy()

            similarities = softmax(metrics.pairwise.cosine_similarity(image_embeddings, text_emb), axis=1)
            print(text_emb.shape, image_embeddings.shape, similarities.shape)
            if label_text.lower() == "suspicious_calcification":
                fpr, tpr, thresholds = metrics.roc_curve(emb["calc"], similarities[:, 1])
                auroc = metrics.auc(fpr, tpr)
                results[label_text] = auroc
            elif label_text.lower()== "mass":
                fpr, tpr, thresholds = metrics.roc_curve(emb["mass"], similarities[:, 1])
                auroc = metrics.auc(fpr, tpr)
                results[label_text] = auroc
            elif label_text.lower() == "density":
                predictions = np.argmax(similarities, axis=1)
                accuracy = accuracy_score(emb["density"], predictions)
                results[label_text] = accuracy
            elif label_text.lower() == "cancer" or label_text.lower() == "malignancy":
                fpr, tpr, thresholds = metrics.roc_curve(emb["cancer"], similarities[:, 1])
                auroc = metrics.auc(fpr, tpr)
                results[label_text] = auroc

        print(test_dataset_name)
        print(results)
        return results


    def eval_img_text_retrieval(self, checkpoint, test_dataset_name, save_path):
        emb = self.get_embeddings(checkpoint, test_dataset_name)
        image_embeddings = emb["image_embeddings"]
        text_embeddings = emb["text_embeddings"]
        text_list = emb["texts"]

        log.info("image_embeddings.shape", image_embeddings.shape)
        log.info("text_embeddings.shape", text_embeddings.shape)
        log.info(len(text_list))

        log.info("evaluate image text retrieval")

        identical_text_set = []

        idx2label = {}
        identical_indexes = []
        for i, text in enumerate(text_list):
            if text not in identical_text_set:
                identical_text_set.append(text)
                identical_indexes.append(i)
                idx2label[i] = len(identical_text_set) - 1
            else:
                idx2label[i] = identical_text_set.index(text)

        identical_text_embedding = text_embeddings[identical_indexes]

        num_samples = image_embeddings.shape[0]
        n_text = len(identical_text_set)

        similarities = metrics.pairwise.cosine_similarity(image_embeddings, identical_text_embedding)  # n x m
        recall_dict = {1: 0, 5: 0, 10: 0, 15: 0}
        mean_rank = 0
        for idx in range(num_samples):
            label = idx2label[idx]
            similarity = similarities[idx]
            similarity_args = similarity.argsort()

            # rank of the paired text
            rank = n_text - np.argwhere(similarity_args == label).ravel()[0]
            mean_rank += rank

            for k in recall_dict:
                if rank <= k:
                    recall_dict[k] += 1

        # results
        log.info(
            "\n".join([f"Recall@{k}: {v / num_samples:.3f}" for k, v in
                       recall_dict.items()]) + f"\nmean rank: {mean_rank / num_samples:.3f}"
        )
        result = {}
        result.update({f"Recall@{k}": v / num_samples for k, v in recall_dict.items()})
        result.update({"MeanRank": mean_rank / num_samples})
        return {
            "retrieval_i2t": result
        }


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
