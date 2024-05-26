import numpy as np
import pandas as pd
import pickle
import torch
import torchvision
import torchvision.transforms
from albumentations import *
from imgaug import augmenters as iaa
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset_concepts import MammoDataset_concept, MammoDataset_concept_detection, \
    collater_for_concept_detection, MammoDataset, collator_mammo_dataset_w_concepts


class center_crop(object):
    def crop_center(self, img):
        _, y, x = img.shape
        crop_size = np.min([y, x])
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)
        return img[:, starty:starty + crop_size, startx:startx + crop_size]

    def __call__(self, img):
        return self.crop_center(img)


class normalize(object):
    def normalize_(self, img, maxval=255):
        img = (img) / (maxval)
        return img

    def __call__(self, img):
        return self.normalize_(img)


def get_transforms(args):
    if (args.dataset.lower() == "rsna" or args.dataset.lower() == "vindr") and args.model_type.lower() == "classifier":
        if args.img_size[0] == 1520 and args.img_size[1] == 912:
            return Compose([
                HorizontalFlip(),
                VerticalFlip(),
                Affine(rotate=20, translate_percent=0.1, scale=[0.8, 1.2], shear=20),
                ElasticTransform(alpha=args.alpha, sigma=args.sigma)
            ], p=args.p)
        else:
            return Compose([
                Resize(width=int(args.img_size[0]), height=int(args.img_size[1])),
                HorizontalFlip(),
                VerticalFlip(),
                Affine(rotate=20, translate_percent=0.1, scale=[0.8, 1.2], shear=20),
                ElasticTransform(alpha=args.alpha, sigma=args.sigma)
            ], p=args.p
            )
    elif args.dataset.lower() == "vindr" and args.model_type.lower() == "concept-detector":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        train_affine_trans = iaa.Sequential([
            iaa.Resize({'height': args.resize, 'width': args.resize}),
            iaa.Fliplr(0.5),  # HorizontalFlip
            iaa.Flipud(0.5),  # VerticalFlip
            iaa.Affine(rotate=(-20, 20), translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, scale=(0.8, 1.2),
                       shear=(-20, 20)),
            iaa.ElasticTransformation(alpha=args.alpha, sigma=args.sigma)
        ])

        test_affine_trans = iaa.Sequential([
            iaa.Resize({'height': args.resize, 'width': args.resize}),
            iaa.CropToFixedSize(width=args.resize, height=args.resize)  # Adjust width and height as needed
        ])

        return transform, train_affine_trans, test_affine_trans


def get_dataloader_concept_classifier(args, train=True):
    valid_dataset = MammoDataset_concept(args=args, df=args.valid_folds, dataset=args.dataset)
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
        drop_last=False,
    )
    if train:
        train_dataset = MammoDataset_concept(
            args=args, df=args.train_folds, dataset=args.dataset, transform=get_transforms(args)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
            drop_last=True,
        )
        return train_loader, valid_loader
    else:
        return valid_loader


def get_dataloader_concept_detector(args, train=True):
    transform, train_affine_trans, test_affine_trans = get_transforms(args)
    valid_dataset = MammoDataset_concept_detection(
        args=args, df=args.valid_folds, iaa_transform=test_affine_trans, transform=transform
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
        drop_last=False, collate_fn=collater_for_concept_detection
    )
    if train:
        train_dataset = MammoDataset_concept_detection(
            args=args, df=args.train_folds, iaa_transform=train_affine_trans, transform=transform
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
            drop_last=True, collate_fn=collater_for_concept_detection
        )
        return train_loader, valid_loader, valid_dataset
    else:
        return valid_loader, valid_dataset


def get_dataloader_RSNA(args):
    train_tfm = None
    val_tfm = None
    if args.arch.lower() == "swin_tiny_custom_norm" or args.arch.lower() == "swin_base_custom_norm":
        color_jitter_transform = torchvision.transforms.ColorJitter(
            brightness=0.1,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )
        normalize_transform = torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        train_tfm = torchvision.transforms.Compose([
            color_jitter_transform,
            torchvision.transforms.ToTensor(),
            normalize_transform
        ])
        val_tfm = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize_transform
        ])
    elif args.arch.lower() == "swin_tiny_custom" or args.arch.lower() == "swin_base_custom":
        train_tfm = Compose([
            ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.1, p=1),
        ])
    else:
        train_tfm = get_transforms(args)

    train_dataset = MammoDataset(args=args, df=args.train_folds, transform=train_tfm)
    valid_dataset = MammoDataset(args=args, df=args.valid_folds, transform=val_tfm)

    if args.balanced_dataloader == "y":
        weight_path = args.output_path / f"random_sampler_weights_fold{str(args.cur_fold)}.pkl"
        if weight_path.exists():
            weights = pickle.load(open(weight_path, "rb"))
        else:
            weight_for_positive_class = args.sampler_weights[f"fold{str(args.cur_fold)}"]["pos_wt"]
            weight_for_negative_class = args.sampler_weights[f"fold{str(args.cur_fold)}"]["neg_wt"]
            args.train_folds["weights_random_sampler"] = args.train_folds.apply(
                lambda row: weight_for_positive_class if row["cancer"] == 1 else weight_for_negative_class, axis=1
            )
            weights = args.train_folds["weights_random_sampler"].values
            pickle.dump(weights, open(args.output_path / f"random_sampler_weights_fold{args.cur_fold}.pkl", "wb"))

        weights = weights.tolist()
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
            drop_last=True, collate_fn=collator_mammo_dataset_w_concepts, sampler=sampler
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
            drop_last=True, collate_fn=collator_mammo_dataset_w_concepts
        )

    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
        drop_last=False, collate_fn=collator_mammo_dataset_w_concepts
    )

    return train_loader, valid_loader


def get_dataset(args, is_train_mode=True, is_classifier=True, train=True):
    if args.dataset.lower() == "rsna" and args.model_type.lower() == "classifier":
        return get_dataloader_RSNA(args)
    elif (
            args.dataset.lower() == "vindr" or args.dataset.lower() == "rsna"
    ) and args.model_type.lower() == 'concept-classifier':
        return get_dataloader_concept_classifier(args, train=train)
    elif args.dataset.lower() == "vindr" and args.model_type.lower() == 'concept-detector':
        return get_dataloader_concept_detector(args, train=train)
