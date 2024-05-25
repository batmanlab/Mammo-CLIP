import cv2
import logging
import numpy as np
import pandas as pd
import random
import torch
from PIL import Image
from breastclip.data.data_utils import load_transform
from pathlib import Path
from torch.utils.data.dataset import Dataset
from typing import Dict, List

log = logging.getLogger(__name__)

import cv2
import logging
import numpy as np
import torch
from PIL import Image
from breastclip.data.data_utils import load_transform
from pathlib import Path
from torch.utils.data.dataset import Dataset
from typing import Dict, List

log = logging.getLogger(__name__)


class ImageTextDataset_Retrieval(Dataset):
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
            label_col: str = 'text',
            image_encoder_type="swin",
            **kwargs
    ):
        log.info(f"Loading Image Text retrieval dataset: [{split}]")
        self.df = df
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.root_dir = Path(data_dir)
        self.img_dir = image_dir
        self.dataset = dataset
        self.transform = load_transform(split=split, transform_config=transform_config)

        log.info(f"split: {split} transform")
        log.info(self.transform)

        self.mean = mean
        self.std = std
        self.label_col = label_col
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

        text_label = data[self.label_col]
        return {
            'image': img,
            'text_label': text_label,
            'img_path': img_path
        }

    def collate_fn(self, instances: List):
        images = torch.stack([ins["image"] for ins in instances], dim=0)
        texts = list([ins["text_label"] for ins in instances])
        text_tokens = self.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt",
                                     max_length=self.text_max_length)
        img_paths = list([ins['img_path'] for ins in instances])
        return {
            "images": images,
            "texts": texts,
            "text_tokens": text_tokens,
            "img_paths": img_paths
        }
