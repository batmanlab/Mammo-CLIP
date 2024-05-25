import logging
from pathlib import Path
from typing import Dict, List

import cv2
import nltk
import numpy as np
import pandas as pd
import torch
from PIL import Image
from breastclip.data.data_utils import load_transform
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
        self.dataframe = df[["patient_id", "image_id", "laterality", "view", "text1", "text_aug", "fold"]]
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
        self.study_laterality_groups = {}

        log.info(f"split: {split} transform")
        log.info(self.tfms)

        # Group by 'STUDY_ID' and 'laterality'
        grouped = df.groupby(['patient_id', 'laterality'])
        for (study_id, laterality), group_indices in grouped.groups.items():
            group = df.loc[group_indices]
            text1 = self._split_report_into_segment(group["text1"].tolist()[0])
            text2 = self._split_report_into_segment(group["text_aug"].tolist()[0])
            self.study_laterality_groups[(study_id, laterality)] = {
                "text1": text1,
                "text2": text2,
                'indices': group_indices
            }

    def __len__(self):
        return len(self.study_laterality_groups)

    def _get_img_path(self, study_id, image_id):
        if self.dataset.lower() == 'upmc':
            return self.root_dir / self.img_dir / f'Patient_{study_id}' / image_id
        else:
            return self.root_dir / self.img_dir / f'{str(study_id)}' / image_id

    def _split_report_into_segment(self, report):
        if pd.isnull(report):
            return []
        else:
            report = report.replace("\n", " ")
            reports = report.split(". ")
            study_sent = []
            for sent in reports:
                if len(sent) == 0:
                    continue
                tokens = nltk.wordpunct_tokenize(sent.lower())

                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)

                if len(included_tokens) > 3:  # only include relative long sentences
                    study_sent.append(" ".join(included_tokens))
            combined_text = ". ".join(study_sent)
            return combined_text

    def __getitem__(self, idx):
        group_key = list(self.study_laterality_groups.keys())[idx]
        group_info = self.study_laterality_groups[group_key]
        group_indices = group_info['indices']
        group = self.dataframe.loc[group_indices]
        available_positions = group['view'].unique()

        if len(available_positions) == 2:
            view_group = group[group['view'] == "CC"]
            sample = view_group.sample(n=1, random_state=1)
            study_id = sample['patient_id'].values[0]
            laterality = sample['laterality'].values[0]
            image_id = sample['image_id'].values[0]

            img_path = self._get_img_path(study_id, image_id)
            if self.image_encoder_type == "swin":
                img1 = Image.open(img_path).convert('RGB')
                img1 = np.array(img1)
            else:
                img1 = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            if self.tfms:
                augmented = self.tfms(image=img1)
                img1 = augmented['image']

            img1 = img1.astype('float32')
            img1 -= img1.min()
            img1 /= img1.max()
            img1 = torch.tensor((img1 - self.mean) / self.std, dtype=torch.float32)
            img1 = img1.unsqueeze(0)

            view_group = group[group['view'] == "MLO"]
            sample = view_group.sample(n=1, random_state=1)
            study_id = sample['patient_id'].values[0]
            laterality = sample['laterality'].values[0]
            image_id = sample['image_id'].values[0]

            img_path = self._get_img_path(study_id, image_id)
            if self.image_encoder_type == "swin":
                img2 = Image.open(img_path).convert('RGB')
                img2 = np.array(img2)
            else:
                img2 = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if self.tfms:
                augmented = self.tfms(image=img2)
                img2 = augmented['image']

            img2 = img2.astype('float32')
            img2 -= img2.min()
            img2 /= img2.max()
            img2 = torch.tensor((img2 - self.mean) / self.std, dtype=torch.float32)
            img2 = img2.unsqueeze(0)
        else:
            view_group = group[group['view'] == available_positions[0]]
            samples = view_group.sample(n=2, random_state=1, replace=True)
            images = []
            for _, sample in samples.iterrows():
                study_id = sample['patient_id']
                laterality = sample['laterality']
                image_id = sample['image_id']

                img_path = self._get_img_path(study_id, image_id)
                if self.image_encoder_type == "swin":
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
                images.append(img)

            img1 = images[0]
            img2 = images[1]

        text1 = group_info['text1']
        text2 = group_info['text2']

        results = {"image": img1, "image_view": img2, "text": text1, "text2": text2}
        return results

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
