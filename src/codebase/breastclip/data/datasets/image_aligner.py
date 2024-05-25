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


class ImageAligenerDataset(Dataset):
    def __init__(
            self,
            split: str,
            df,
            dataset: str,
            data_dir: str,
            image_dir: str,
            transform_config: Dict = None,
            mean=0,
            std=0,
            label_col: str = 'cancer',
            image_encoder_type="swin",
            **kwargs
    ):
        log.info(f"Loading Image classification dataset: [{split}]")
        self.df = df
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

    def __getitem__(self, idx):
        data = self.df.iloc[idx]

        img_path = str(
            self.root_dir / self.img_dir / str(self.df.iloc[idx]['patient_id']) / str(self.df.iloc[idx]['image_id']))

        if self.image_encoder_type == "swin":
            img_clip = Image.open(img_path).convert('RGB')
            img_clip = np.array(img_clip)
        else:
            img_clip = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=img_clip)
            img_clip = augmented['image']

        img_clip = img_clip.astype('float32')
        img_clip -= img_clip.min()
        img_clip /= img_clip.max()
        img_clip = torch.tensor((img_clip - self.mean) / self.std, dtype=torch.float32)
        img_clip = img_clip.unsqueeze(0)

        img_clf = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=img_clf)
            img_clf = augmented['image']

        img_clf = img_clf.astype('float32')
        img_clf -= img_clf.min()
        img_clf /= img_clf.max()
        img_clf = torch.tensor((img_clf - self.mean) / self.std, dtype=torch.float32)
        img_clf = img_clf.unsqueeze(0)

        label = torch.tensor(data[self.label_col], dtype=torch.long)
        pred = torch.tensor(data["predictions_aucroc_weighted_BCE_y_bal_dataloader_n"], dtype=torch.long)
        age = torch.tensor(data["age"], dtype=torch.long)
        calc_0_1 = torch.tensor(data["Suspicious_Calcification_th_0.1"], dtype=torch.long)
        calc_0_15 = torch.tensor(data["Suspicious_Calcification_th_0.15"], dtype=torch.long)
        calc_0_25 = torch.tensor(data["Suspicious_Calcification_th_0.25"], dtype=torch.long)
        mass_0_1 = torch.tensor(data["Mass_th_0.1"], dtype=torch.long)
        mass_0_15 = torch.tensor(data["Mass_th_0.15"], dtype=torch.long)
        mass_0_2 = torch.tensor(data["Mass_th_0.2"], dtype=torch.long)
        clip = torch.tensor(data["CLIP_V1_bin"], dtype=torch.long)
        scar = torch.tensor(data["SCAR_V1_bin"], dtype=torch.long)
        mark = torch.tensor(data["MARK_V1_bin"], dtype=torch.long)
        mole = torch.tensor(data["MOLE_V1_bin"], dtype=torch.long)
        fold = torch.tensor(data["fold"], dtype=torch.long)
        return {
            'image_clip': img_clip,
            'image_clf': img_clf,
            'img_path': img_path,
            'label': label,
            'pred': pred,
            'age': age,
            'calc_0_1': calc_0_1,
            'calc_0_15': calc_0_15,
            'calc_0_25': calc_0_25,
            'mass_0_1': mass_0_1,
            'mass_0_15': mass_0_15,
            'mass_0_2': mass_0_2,
            'clip': clip,
            'scar': scar,
            'mark': mark,
            'mole': mole,
            'fold': fold
        }

    def collate_fn(self, instances: List):
        images_clip = torch.stack([ins["image_clip"] for ins in instances], dim=0)
        images_clf = torch.stack([ins["image_clf"] for ins in instances], dim=0)
        labels = torch.stack([ins["label"] for ins in instances], dim=0)
        preds = torch.stack([ins["pred"] for ins in instances], dim=0)
        age = torch.stack([ins["age"] for ins in instances], dim=0)
        calc_0_1 = torch.stack([ins["calc_0_1"] for ins in instances], dim=0)
        calc_0_15 = torch.stack([ins["calc_0_15"] for ins in instances], dim=0)
        calc_0_25 = torch.stack([ins["calc_0_25"] for ins in instances], dim=0)
        mass_0_1 = torch.stack([ins["mass_0_1"] for ins in instances], dim=0)
        mass_0_15 = torch.stack([ins["mass_0_15"] for ins in instances], dim=0)
        mass_0_2 = torch.stack([ins["mass_0_2"] for ins in instances], dim=0)
        clip = torch.stack([ins["clip"] for ins in instances], dim=0)
        scar = torch.stack([ins["scar"] for ins in instances], dim=0)
        mark = torch.stack([ins["mark"] for ins in instances], dim=0)
        mole = torch.stack([ins["mole"] for ins in instances], dim=0)
        folds = torch.stack([ins["fold"] for ins in instances], dim=0)
        img_paths = list([ins['img_path'] for ins in instances])

        return {
            "images_clip": images_clip,
            "images_clf": images_clf,
            "img_paths": img_paths,
            "labels": labels,
            "preds": preds,
            "age": age,
            "calc_0_1": calc_0_1,
            "calc_0_15": calc_0_15,
            "calc_0_25": calc_0_25,
            "mass_0_1": mass_0_1,
            "mass_0_15": mass_0_15,
            "mass_0_2": mass_0_2,
            "clip": clip,
            "scar": scar,
            "mark": mark,
            "mole": mole,
            "folds": folds
        }
