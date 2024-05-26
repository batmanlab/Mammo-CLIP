import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from PIL import Image
from breastclip.data.data_utils import load_transform
from torch.utils.data.dataset import Dataset

log = logging.getLogger(__name__)


class ImageClassificationZSDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            split: str,
            df,
            dataset: str,
            data_dir: str,
            image_dir: str,
            text_max_length: int = 256,
            transform_config: Dict = None,
            mean=0,
            std=0,
            image_encoder_type="swin",
            **kwargs
    ):
        log.info(f"Loading Image classification dataset: [{split}]")
        self.df = df
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.root_dir = Path(data_dir)
        self.img_dir = image_dir
        self.dataset = dataset
        self.transform = load_transform(split=split, transform_config=transform_config)

        log.info(f"split: {split} transform")
        log.info(f"dataset: {self.dataset}")
        log.info(self.transform)

        self.mean = mean
        self.std = std
        self.image_encoder_type = image_encoder_type

    def __len__(self):
        return len(self.df)

    def _get_img_path(self, study_id, image_id):
        if self.dataset.lower() == 'upmc':
            return self.root_dir / self.img_dir / f'Patient_{study_id}' / image_id
        else:
            return self.root_dir / self.img_dir / f'{str(study_id)}' / image_id

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        study_id = str(self.df.iloc[idx]['patient_id'])
        image_id = self.df.iloc[idx]['image_id']
        img_path = str(self._get_img_path(study_id, image_id))
        if not img_path.endswith(".png"):
            img_path += ".png"

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

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        img = img.astype('float32')
        img -= img.min()
        img /= img.max()
        img = torch.tensor((img - self.mean) / self.std, dtype=torch.float32)
        img = img.unsqueeze(0)
        if self.dataset.lower() == 'vindr':
            mass = torch.tensor(data["Mass"], dtype=torch.long)
            calc = torch.tensor(data["Suspicious_Calcification"], dtype=torch.long)
            density = torch.tensor(data["density"], dtype=torch.long)
            return {
                'images': img,
                'mass': mass,
                'calc': calc,
                'density': density
            }
        elif self.dataset.lower() == 'rsna':
            cancer = torch.tensor(data["cancer"], dtype=torch.long)
            return {
                'images': img,
                'malignancy': cancer,
                'cancer': cancer
            }
