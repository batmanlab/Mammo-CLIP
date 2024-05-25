import logging
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from .data import DataModule
from .model import build_model

log = logging.getLogger(__name__)


def save_representations(cfg: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"cfg: {cfg}")
    fold = cfg["classifier"]["fold"]
    arch = cfg["classifier"]["clf_arch"]

    clf_checkpoint = cfg["classifier"]["clf_chkpt"].format(fold)
    clf = build_model(model_config=cfg["classifier"], loss_config=cfg["loss"], tokenizer=None)
    clf = clf.to(device)
    ckpt = torch.load(clf_checkpoint, map_location="cpu")
    clf.load_state_dict(ckpt["model"], strict=True)
    clf.eval()

    log.info(f"================== fold: {fold}  =======================")
    log.info(f"clf_checkpoint: {clf_checkpoint}")
    log.info("clf is loaded")

    save_path = Path(cfg["base"]["output"]["save_path"]) / f"fold{fold}"
    os.makedirs(save_path, exist_ok=True)

    datamodule = DataModule(
        data_config=cfg["data_aligner"],
        dataloader_config=cfg["dataloader"],
        tokenizer_config=cfg["tokenizer"] if "tokenizer" in cfg else None,
        transform_config=cfg["transform"] if "transform" in cfg else cfg["transform"],
        mean=cfg["base"]["mean"],
        std=cfg["base"]["std"],
        image_encoder_type=cfg["model"]["image_encoder"]["model_type"],
        cur_fold=fold
    )
    train_dataloader, train_sampler = datamodule.train_dataloader(distributed=False)
    test_dataloader_dict = datamodule.valid_dataloader()
    valid_dataloader = test_dataloader_dict["rsna"]
    log.info(f"train_loader: {len(train_dataloader)}")
    log.info(f"valid_dataloader: {len(valid_dataloader)}")

    clip = build_model(model_config=cfg["model"], loss_config=cfg["loss"], tokenizer=datamodule.tokenizer)
    clip = clip.to(device)
    clf_checkpoint_breast_clip = cfg["base"]["breast_clip_chkpt"]
    ckpt = torch.load(clf_checkpoint_breast_clip, map_location="cpu")
    clip.load_state_dict(ckpt["model"], strict=True)
    clip.eval()

    log.info(clip)
    log.info(f"clf_checkpoint: {clf_checkpoint_breast_clip}")
    log.info("clip is loaded")

    image_encoder_type = cfg["model"]["image_encoder"]["model_type"]
    save_features(valid_dataloader, device, "valid", image_encoder_type, clip, clf, save_path, arch)
    save_features(train_dataloader, device, "train", image_encoder_type, clip, clf, save_path, arch)


def save_features(loader, device, mode, image_encoder_type, clip, clf, save_path, arch):
    all_reps_classifier = []
    all_reps_clip = []
    all_image_paths = []
    out_put_GT = torch.FloatTensor()
    out_put_predict = torch.FloatTensor()
    out_put_age = torch.FloatTensor()
    out_put_calc_0_1 = torch.FloatTensor()
    out_put_calc_0_15 = torch.FloatTensor()
    out_put_calc_0_25 = torch.FloatTensor()
    out_put_mass_0_1 = torch.FloatTensor()
    out_put_mass_0_15 = torch.FloatTensor()
    out_put_mass_0_2 = torch.FloatTensor()
    out_put_clip = torch.FloatTensor()
    out_put_scar = torch.FloatTensor()
    out_put_mark = torch.FloatTensor()
    out_put_mole = torch.FloatTensor()
    out_put_fold = torch.FloatTensor()
    print(f"First saving {mode} representations..")
    with torch.no_grad():
        with tqdm(total=len(loader)) as t:
            for batch_id, batch in enumerate(loader):
                if image_encoder_type.lower() == "swin":
                    if "images_clip" in batch:
                        batch["images_clip"] = batch["images_clip"].squeeze(1).permute(0, 3, 1, 2)

                image_reps_clip = clip.encode_image_normalized(batch["images_clip"].to(device))
                image_reps_clf, _ = clf(batch["images_clf"].to(device))

                image_paths = batch["img_paths"]
                targets = batch["labels"]
                pred = batch["preds"]
                age = batch["age"]
                calc_0_1 = batch["calc_0_1"]
                calc_0_15 = batch["calc_0_1"]
                calc_0_25 = batch["calc_0_1"]
                mass_0_1 = batch["mass_0_1"]
                mass_0_15 = batch["mass_0_15"]
                mass_0_2 = batch["mass_0_2"]
                biopsy_clip = batch["clip"]
                scar = batch["scar"]
                mark = batch["mark"]
                mole = batch["mole"]
                fold = batch["folds"]
                reps_clip = [x.detach().cpu().numpy() for x in image_reps_clip]
                reps_classifier = [x.detach().cpu().numpy() for x in image_reps_clf]

                all_reps_clip.extend(reps_clip)
                all_reps_classifier.extend(reps_classifier)
                all_image_paths.extend(image_paths)

                out_put_predict = torch.cat((out_put_predict, pred), dim=0)
                out_put_GT = torch.cat((out_put_GT, targets), dim=0)
                out_put_age = torch.cat((out_put_age, age), dim=0)
                out_put_calc_0_1 = torch.cat((out_put_calc_0_1, calc_0_1), dim=0)
                out_put_calc_0_15 = torch.cat((out_put_calc_0_15, calc_0_15), dim=0)
                out_put_calc_0_25 = torch.cat((out_put_calc_0_25, calc_0_25), dim=0)
                out_put_mass_0_1 = torch.cat((out_put_mass_0_1, mass_0_1), dim=0)
                out_put_mass_0_15 = torch.cat((out_put_mass_0_15, mass_0_15), dim=0)
                out_put_mass_0_2 = torch.cat((out_put_mass_0_2, mass_0_2), dim=0)
                out_put_clip = torch.cat((out_put_clip, biopsy_clip), dim=0)
                out_put_scar = torch.cat((out_put_scar, scar), dim=0)
                out_put_mark = torch.cat((out_put_mark, mark), dim=0)
                out_put_mole = torch.cat((out_put_mole, mole), dim=0)
                out_put_fold = torch.cat((out_put_fold, fold), dim=0)

                t.set_postfix(epoch='{0}'.format(batch_id))
                t.update()

        all_reps_classifier = np.stack(all_reps_classifier)
        all_reps_clip = np.stack(all_reps_clip)
        print(
            f"Classifier {mode} embedding shape: {all_reps_classifier.shape}, "
            f"Clip {mode} embedding shape: {all_reps_clip.shape} "
        )
        np.save(save_path / f"{mode}_classifier_{arch}_embeddings.npy", all_reps_classifier)
        np.save(save_path / f"{mode}_clip.npy", all_reps_clip)

        save_path = save_path / "ground_truths"
        os.makedirs(save_path, exist_ok=True)
        torch.save(out_put_GT, save_path / f"{mode}_GT.pth.tar")
        torch.save(out_put_predict, save_path / f"{mode}_predictions.pth.tar")
        torch.save(out_put_age, save_path / f"{mode}_age.pth.tar")
        torch.save(out_put_calc_0_1, save_path / f"{mode}_calcification_0_1.pth.tar")
        torch.save(out_put_calc_0_15, save_path / f"{mode}_calcification_0_15.pth.tar")
        torch.save(out_put_calc_0_25, save_path / f"{mode}_calcification_0_25.pth.tar")
        torch.save(out_put_mass_0_1, save_path / f"{mode}_mass_0_1.pth.tar")
        torch.save(out_put_mass_0_15, save_path / f"{mode}_mass_0_15.pth.tar")
        torch.save(out_put_mass_0_2, save_path / f"{mode}_mass_0_2.pth.tar")
        torch.save(out_put_clip, save_path / f"{mode}_clip.pth.tar")
        torch.save(out_put_scar, save_path / f"{mode}_scar.pth.tar")
        torch.save(out_put_mark, save_path / f"{mode}_mark.pth.tar")
        torch.save(out_put_mole, save_path / f"{mode}_{arch}_mole.pth.tar")

        print("Outputs are saved at:")
        print(save_path)
