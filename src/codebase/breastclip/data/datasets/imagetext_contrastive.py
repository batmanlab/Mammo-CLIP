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
from torch.utils.data.dataset import Dataset

log = logging.getLogger(__name__)


class ImageTextDataset_contrastive(Dataset):
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
            convirt_mode=True,
            **kwargs
    ):
        self.dataframe = df[["patient_id", "image_id", "FINDINGS", "IMPRESSION", "REPORT", "BIRADS_numeric", "fold"]]
        self.tokenizer = tokenizer
        self.root_dir = Path(data_dir)
        self.img_dir = image_dir
        self.dataset = dataset
        self.text_max_length = text_max_length
        self.loss_config = {k: v for k, v in loss_config.items()}
        self.tfms = load_transform(split=split, transform_config=transform_config)
        self.mean = mean
        self.std = std
        self.image_encoder_type = image_encoder_type
        self.convirt_mode = convirt_mode

        log.info(f"split: {split} transform")
        log.info(self.tfms)

    def __len__(self):
        return len(self.dataframe)

    def _get_img_path(self, study_id, image_id):
        if self.dataset.lower() == 'upmc':
            return self.root_dir / self.img_dir / f'Patient_{study_id}' / image_id
        else:
            return self.root_dir / self.img_dir / f'{str(study_id)}' / image_id

    def __getitem__(self, idx):
        study_id = str(self.dataframe.iloc[idx]['patient_id'])
        image_id = self.dataframe.iloc[idx]['image_id']
        img_path = self._get_img_path(study_id, image_id)

        if (
                self.image_encoder_type == "swin" or
                self.image_encoder_type == "resnet101" or
                self.image_encoder_type == "resnet152" or
                self.image_encoder_type == "tf_efficientnet_b5_ns-detect" or
                self.image_encoder_type == "tf_efficientnetv2-detect"
        ):
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if self.tfms:
            augmented = self.tfms(image=img)
            img = augmented['image']
        img = img.astype('float32')
        img -= img.min()
        img /= img.max()
        img = torch.tensor((img - self.mean) / self.std, dtype=torch.float32)
        img = img.unsqueeze(0)

        content = self.dataframe.iloc[idx]['REPORT']
        if self.convirt_mode:
            content = content.replace("\n", " ")
            ls_text = content.split(".")
            if '' in ls_text:
                ls_text.remove('')

            text = random.choice(ls_text)
        else:
            text = content
        label = torch.tensor(self.dataframe.iloc[idx]['BIRADS_numeric'], dtype=torch.long)
        results = {"image": img, "text": text, "label": label}
        return results

    def collate_fn(self, instances: List):
        images = torch.stack([ins["image"] for ins in instances], dim=0)
        texts = list([ins["text"] for ins in instances])
        text_tokens = self.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt",
                                     max_length=self.text_max_length)
        labels = torch.stack([ins["label"] for ins in instances], dim=0)

        batch = {
            "images": images,
            "texts": texts,
            "labels": labels,
            "text_tokens": text_tokens,
        }

        return batch
