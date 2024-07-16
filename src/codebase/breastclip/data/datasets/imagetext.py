import ast
import json
import logging
import random
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from breastclip.data.data_utils import load_transform
from breastclip.prompts.prompts import generate_report_from_labels
from nltk import tokenize
from torch.utils.data.dataset import Dataset

log = logging.getLogger(__name__)


class ImageTextDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            split: str,
            df: pd.DataFrame,
            dataset: str,
            data_dir: str,
            image_dir: str,
            text_max_length: int = 256,
            loss_config: Dict = None,
            transform_config: Dict = None,
            mean=0,
            std=0,
            image_encoder_type="swin",
            **kwargs
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.root_dir = Path(data_dir)
        self.img_dir = image_dir
        self.dataset = dataset
        self.text_max_length = text_max_length
        self.loss_config = {k: v for k, v in loss_config.items()}
        self.tfms = load_transform(split=split, transform_config=transform_config)
        self.split = split
        self.mean = mean
        self.std = std
        self.image_encoder_type = image_encoder_type
        self.image_aug_other_image = True
        self.image_view_aug = True
        self.has_backtranslated = hasattr(self.df, "text_augment")
        with open(
                "/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/codebase/breastclip/data/datasets/prompts.json"
        ) as f:
            self.prompt_json = json.load(f)

        log.info(f"split: {split} transform")
        log.info(self.tfms)

    def __len__(self):
        return len(self.df)

    def _get_img_path(self, study_id, image_id):
        if self.dataset.lower() == "upmc":
            return self.root_dir / self.img_dir / f"Patient_{study_id}" / image_id
        elif self.dataset.lower() == "vindr":
            return self.root_dir / self.img_dir / f"{str(study_id)}" / image_id

    def __getitem__(self, index):
        view_list = None
        selected_image_indices = {}
        if hasattr(self.df, "CC"):
            try:
                view_list = ast.literal_eval(self.df["view"][index])
            except Exception:
                view_list = [self.df["view"][index]]

            if len(view_list) >= 2:
                view_list = np.random.choice(view_list, size=2, replace=False)
                image_path_list = []
                for view in view_list:
                    try:
                        image_paths = ast.literal_eval(self.df[view][index])
                    except Exception:
                        image_paths = [self.df[view][index]]
                    if image_paths:
                        chosen_index = np.random.choice(len(image_paths))
                        image_path = image_paths[chosen_index]
                        image_path_list.append(image_path)
                        selected_image_indices[view] = chosen_index

            else:
                if len(view_list) == 1:
                    tag = view_list[0]
                else:
                    tag = "image"

                try:
                    image_path_list = ast.literal_eval(self.df[tag][index])
                except Exception:
                    image_path_list = [self.df[tag][index]]

                if self.split == "train":
                    if self.image_aug_other_image and len(image_path_list) > 1:
                        image_path_list = np.random.choice(image_path_list, size=2, replace=False)
                    else:
                        image_path_list = np.random.choice(image_path_list, size=1)
        else:
            try:
                image_path_list = ast.literal_eval(self.df["image"][index])
            except Exception:
                image_path_list = [self.df["image"][index]]

        study_id = str(self.df.iloc[index]['patient_id'])
        img_path = self._get_img_path(study_id, image_path_list[0])
        if (
                self.image_encoder_type == "swin" or
                self.image_encoder_type == "resnet101" or
                self.image_encoder_type == "resnet152" or
                self.image_encoder_type == "tf_efficientnet_b5_ns-detect" or
                self.image_encoder_type == "tf_efficientnetv2-detect"
        ):
            image_original = Image.open(img_path).convert('RGB')
            image_original = np.array(image_original)
        else:
            image_original = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if self.tfms:
            augmented = self.tfms(image=image_original)
            image = augmented['image']
        else:
            image = image_original
        image = image.astype('float32')
        image -= image.min()
        image /= image.max()
        image = torch.tensor((image - self.mean) / self.std, dtype=torch.float32)
        image = image.unsqueeze(0)

        if self.image_view_aug:
            if len(image_path_list) > 1:
                img_path = self._get_img_path(study_id, image_path_list[1])
                if (
                        self.image_encoder_type == "swin" or
                        self.image_encoder_type == "resnet101" or
                        self.image_encoder_type == "resnet152" or
                        self.image_encoder_type == "tf_efficientnet_b5_ns-detect" or
                        self.image_encoder_type == "tf_efficientnetv2-detect"
                ):
                    image_original = Image.open(img_path).convert('RGB')
                    image_original = np.array(image_original)
                else:
                    image_original = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if self.tfms:
                augmented = self.tfms(image=image_original)
                image_view = augmented['image']
            else:
                image_view = image_original
            image_view = image_view.astype('float32')
            image_view -= image_view.min()
            image_view /= image_view.max()
            image_view = torch.tensor((image_view - self.mean) / self.std, dtype=torch.float32)
            image_view = image_view.unsqueeze(0)

        # Get from image-text dataset
        if hasattr(self.df, "text"):
            try:
                text_list = ast.literal_eval(self.df["text"][index])
            except Exception:
                text_list = self.df["text"][index]

            if self.has_backtranslated:
                try:
                    text_aug_list = ast.literal_eval(self.df["text_augment"][index])
                except Exception:
                    text_aug_list = self.df["text_augment"][index]

            if len(text_list) >= 2:
                indexes = np.random.randint(len(text_list), size=2)  # Multiple section
                text = text_aug_list[indexes[0]] if random.random() < 0.5 and self.has_backtranslated else text_list[
                    indexes[0]]
                text2 = text_aug_list[indexes[1]] if random.random() < 0.5 and self.has_backtranslated else text_list[
                    indexes[1]]

            else:
                if random.random() < 0.5:
                    text = text_list[0]
                    text2 = text_aug_list[0] if self.has_backtranslated else text_list[0]
                else:
                    text = text_aug_list[0] if self.has_backtranslated else text_list[0]
                    text2 = text_list[0]

            if self.split == "train":  # Text shuffle augment
                for _text in [text, text2]:
                    _text_list = tokenize.sent_tokenize(_text, language="english")
                    random.shuffle(_text_list)
                    _text = " ".join(_text_list)

        # Get from image-label dataset.
        elif hasattr(self.df, "CC_FINDING"):
            if self.dataset.lower() == "vindr":
                # Image and view column of upmc_vindr should be in order of ["CC", "MLO"]
                # CC_FINDING, MLO_FINDING are in order:
                # [[+ve right findings], [+ve left findings], [-ve right findings], [-ve left findings]]
                cc, mlo = view_list
                cc_findings = ast.literal_eval(self.df[f"{cc}_FINDING"][index])
                mlo_findings = ast.literal_eval(self.df[f"{mlo}_FINDING"][index])
                text = generate_report_from_labels(cc_findings, self.prompt_json, deterministic=(self.split != "train"))
                text2 = generate_report_from_labels(mlo_findings, self.prompt_json,
                                                    deterministic=(self.split != "train"))
        else:
            raise AttributeError("There is no report column in DataFrame.")

        out = {"image": image, "image_view": image_view, "text": text, "text2": text2}
        return out

    def collate_fn(self, instances: List):
        images = torch.stack([ins["image"] for ins in instances], dim=0)
        texts = list([ins["text"] for ins in instances])
        text_tokens = self.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt",
                                     max_length=self.text_max_length)

        texts2 = list([ins["text2"] for ins in instances])
        text_tokens2 = self.tokenizer(texts2, padding="max_length", truncation=True, return_tensors="pt",
                                      max_length=self.text_max_length)
        image_views = torch.stack([ins["image_view"] for ins in instances], dim=0)

        batch = {
            "images": images,
            "image_views": image_views,
            "texts": texts,
            "texts2": texts2,
            "text_tokens": text_tokens,
            "text_tokens2": text_tokens2,
        }

        return batch
