import logging
from pathlib import Path
from typing import Dict

import pandas as pd
from breastclip import util
from torch.utils.data import ConcatDataset, DataLoader

# DDP
from torch.utils.data.distributed import DistributedSampler

#
from .data_utils import load_tokenizer
from .datasets import load_dataset

log = logging.getLogger(__name__)


class DataModule:
    def __init__(
            self,
            data_config: Dict,
            dataloader_config: Dict = None,
            tokenizer_config: Dict = None,
            loss_config: Dict = None,
            transform_config: Dict = None,
            cur_fold: int = 0,
            mean: float = 0,
            std: float = 0,
            image_encoder_type: str = "swin"
    ):
        self.train_sampler = None
        dtype_options = {
            "patient_id": str,
            "image_id": str,
            "laterality": str,
            "view": str,
            "text1": str,
            "text_aug": str,
            "fold": int
        }

        self.data_config = data_config
        self.dataloader_config = dataloader_config
        self.tokenizer_config = tokenizer_config
        self.loss_config = loss_config
        self.tokenizer = load_tokenizer(**self.tokenizer_config) if self.tokenizer_config is not None else None
        self.datasets = {"train": [], "valid": [], "test": []}
        self.image_encoder_type = image_encoder_type

        self.train_loader = None
        self.valid_loader_dict = None
        self.test_loader = None

        for dataset in data_config:
            df = pd.read_csv(Path(
                data_config[dataset]["data_dir"]) / data_config[dataset]["data_path"], dtype=dtype_options)
            df = df.fillna(0)
            if data_config[dataset]["name"].lower() == "vindr":
                train_df = df[df['split'] == "training"].reset_index(drop=True)
                valid_df = df[df['split'] == "test"].reset_index(drop=True)
            else:
                train_df = df[df['fold'] != cur_fold].reset_index(drop=True)
                valid_df = df[df['fold'] == cur_fold].reset_index(drop=True)

            train_dataset = load_dataset(
                df=train_df,
                split="train",
                dataset=data_config[dataset]["name"],
                data_dir=data_config[dataset]["data_dir"],
                image_dir=data_config[dataset]["img_dir"],
                data_type=data_config[dataset]["data_type"],
                tokenizer=self.tokenizer,
                transform_config=transform_config,
                loss_config=self.loss_config,
                text_max_length=data_config[dataset]["text_max_length"],
                mean=mean,
                std=std,
                image_encoder_type=self.image_encoder_type,
                label_col=data_config[dataset]["label_col"] if "label_col" in data_config[dataset] else None,
                label_text=data_config[dataset]["label_text"] if "label_text" in data_config[dataset] else None,
            )
            valid_dataset = load_dataset(
                df=valid_df,
                split="valid",
                dataset=data_config[dataset]["name"],
                data_dir=data_config[dataset]["data_dir"],
                image_dir=data_config[dataset]["img_dir"],
                data_type=data_config[dataset]["data_type"],
                tokenizer=self.tokenizer,
                transform_config=transform_config,
                loss_config=self.loss_config,
                text_max_length=data_config[dataset]["text_max_length"],
                mean=mean,
                std=std,
                image_encoder_type=self.image_encoder_type,
                label_col=data_config[dataset]["label_col"] if "label_col" in data_config[dataset] else None,
                label_text=data_config[dataset]["label_text"] if "label_text" in data_config[dataset] else None,
            )

            self.datasets["train"].append(train_dataset)
            self.datasets["valid"].append(valid_dataset)

            log.info(f"Loading fold: {cur_fold}")
            log.info(f"train_df length: {train_df.shape}")
            log.info(f"valid_df length: {valid_df.shape}")
            log.info(f"Length of train_dataset: {len(train_dataset)}")
            log.info(f"Length of valid_dataset: {len(valid_dataset)}")

            log.info(f"Dataset: {dataset} is loaded")

    def train_dataloader(self, distributed):
        assert self.dataloader_config is not None

        if self.train_loader is None:
            dataset = ConcatDataset(self.datasets["train"])
            shuffle = self.dataloader_config["train"]["shuffle"]

            # DDP
            if distributed:
                self.dataloader_config["train"]["shuffle"] = False  # Disable shuffle if using DistributedSampler

                # Create DistributedSampler
                self.train_sampler = DistributedSampler(dataset=dataset, num_replicas=util.GlobalEnv.get().world_size,
                                                        rank=util.GlobalEnv.get().world_rank)

            else:
                self.train_sampler = None

            self.train_loader = DataLoader(
                dataset,
                collate_fn=getattr(self.datasets["train"][0], "collate_fn", None),
                sampler=self.train_sampler,
                **self.dataloader_config["train"],
            )

        return self.train_loader, self.train_sampler

    def valid_dataloader(self, distributed=False):
        assert self.dataloader_config is not None

        if self.valid_loader_dict is None:
            self.valid_loader_dict = dict()

            for val_dataset in self.datasets["valid"]:
                # DDP
                if distributed:
                    sampler = DistributedSampler(dataset=val_dataset, num_replicas=util.GlobalEnv.get().world_size,
                                                 rank=util.GlobalEnv.get().world_rank)
                else:
                    sampler = None
                    # sampler.set_epoch(0)   # This ensures shuffling will be the same across epochs.

                dataloader = DataLoader(
                    val_dataset, collate_fn=getattr(val_dataset, "collate_fn", None), sampler=sampler,
                    **self.dataloader_config["valid"]
                )
                self.valid_loader_dict[val_dataset.dataset] = dataloader

        return self.valid_loader_dict
