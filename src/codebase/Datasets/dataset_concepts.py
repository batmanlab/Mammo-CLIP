from collections import defaultdict

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from torch.utils.data import Dataset


class MammoDataset(Dataset):
    def __init__(self, args, df, transform=None):
        self.args = args
        self.df = df
        self.dir_path = args.data_dir / args.img_dir
        self.dataset = args.dataset
        self.transform = transform
        self.image_encoder_type = args.image_encoder_type
        self.label = args.label

        print(transform)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        img_path = self.dir_path / str(self.df.iloc[idx]['patient_id']) / str(self.df.iloc[idx]['image_id'])
        if self.dataset.lower() == "rsna":
            img_path = f'{img_path}.png'
        if (
                self.args.arch.lower() == "upmc_breast_clip_det_b5_period_n_ft" or
                self.args.arch.lower() == "upmc_vindr_breast_clip_det_b5_period_n_ft" or
                self.args.arch.lower() == "upmc_breast_clip_det_b5_period_n_lp" or
                self.args.arch.lower() == "upmc_vindr_breast_clip_det_b5_period_n_lp" or
                self.args.arch.lower() == "upmc_breast_clip_det_b2_period_n_ft" or
                self.args.arch.lower() == "upmc_vindr_breast_clip_det_b2_period_n_ft" or
                self.args.arch.lower() == "upmc_breast_clip_det_b2_period_n_lp" or
                self.args.arch.lower() == "upmc_vindr_breast_clip_det_b2_period_n_lp"
        ):
            img = Image.open(img_path).convert('RGB')
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if self.transform and (
                self.args.arch.lower() == "swin_tiny_custom_norm" or
                self.args.arch.lower() == "swin_base_custom_norm"):
            img = self.transform(img)
        elif self.transform:
            img = np.array(img)
            augmented = self.transform(image=img)
            img = augmented['image']

            img = img.astype('float32')
            img -= img.min()
            img /= img.max()
            img = torch.tensor((img - self.args.mean) / self.args.std, dtype=torch.float32)
        else:
            img = np.array(img)
            img = img.astype('float32')
            img -= img.min()
            img /= img.max()
            img = torch.tensor((img - self.args.mean) / self.args.std, dtype=torch.float32)

        return {
            'x': img.unsqueeze(0),
            'y': torch.tensor(data[self.label], dtype=torch.long),
            'img_path': str(img_path)
        }


def collator_mammo_dataset_w_concepts(batch):
    return {
        'x': torch.stack([item['x'] for item in batch]),
        'y': torch.from_numpy(np.array([item["y"] for item in batch], dtype=np.float32)),
        'img_path': [item['img_path'] for item in batch]
    }


def collator_mammo_datasett(batch):
    return {
        'x': torch.stack([item['x'] for item in batch]),
        'y': torch.from_numpy(np.array([item["y"] for item in batch], dtype=np.float32)),
        'img_path': [item['img_path'] for item in batch],
    }


def collator_mammo_dataset_concept(batch):
    return {
        'x': torch.stack([item['x'] for item in batch]),
        'y': torch.from_numpy(np.array([item["y"] for item in batch], dtype=np.float32)),
        'img_path': [item['img_path'] for item in batch],
        'boxes': torch.stack([item['boxes'] for item in batch])
    }


class MammoDataset_concept_detection(Dataset):
    def __init__(self, args, df, iaa_transform=None, transform=None):
        self.args = args
        self.dir_path = args.data_dir / args.img_dir
        self.annotations = df
        self.dataset = args.dataset
        self.labels_list = args.concepts
        self.iaa_transform = iaa_transform
        self.transform = transform
        self.mean = args.mean
        self.std = args.std
        self.image_dict = self._generate_image_dict()

    def _generate_image_dict(self):
        image_dict = defaultdict(lambda: {"boxes": [], "labels": []})

        for idx, row in self.annotations.iterrows():
            if "study_id" in row:
                study_id = row["study_id"]
            else:
                study_id = row["patient_id"]
            image_id = row["image_id"]
            boxes = row[["resized_xmin", "resized_ymin", "resized_xmax", "resized_ymax"]].values.tolist()
            labels = [label.strip() for label in row["finding_categories"].strip("[]").split(",")]
            for label in labels:
                label = label.strip("''")

                if label == 'No Finding':
                    boxes = [0, 0, 0, 0]

                if label in self.labels_list:
                    index = self.labels_list.index(label)
                    image_dict[(study_id, image_id)]["boxes"].append(boxes + [index])
                    image_dict[(study_id, image_id)]["labels"].append(index)

        return image_dict

    def __len__(self):
        return len(self.image_dict)

    def __getitem__(self, idx):
        return self.get_items_for_vindr(idx)

    def get_items_for_vindr(self, idx):
        study_id, image_id = list(self.image_dict.keys())[idx]
        boxes = self.image_dict[(study_id, image_id)]["boxes"]
        labels = self.image_dict[(study_id, image_id)]["labels"]

        path = None
        if self.dataset.lower() == 'vindr' and not image_id.endswith(".png"):
            path = f"{self.dir_path}/{study_id}/{image_id}.png"
        elif self.dataset.lower() == 'vindr' and image_id.endswith(".png"):
            path = f"{self.dir_path}/{study_id}/{image_id}"

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = Image.fromarray(image).convert('RGB')
        image = np.array(image)
        if self.iaa_transform:
            bb_box = []
            for bb in boxes:
                bb_box.append(BoundingBox(x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3]))
            bbs_on_image = BoundingBoxesOnImage(bb_box, shape=image.shape)
            image, boxes = self.iaa_transform(
                image=image,
                bounding_boxes=[bbs_on_image]
            )
        if self.transform:
            image = self.transform(image)
        image = image.to(torch.float32)

        image -= image.min()
        image /= image.max()
        image = torch.tensor((image - self.mean) / self.std, dtype=torch.float32)
        bb_final = []
        for idx, bb in enumerate(boxes[0]):
            bb_final.append([bb.x1, bb.y1, bb.x2, bb.y2, labels[idx]])

        target = {
            "boxes": torch.tensor(bb_final),
            "labels": labels,
        }
        return {
            "image": image,
            "target": target,
            "study_id": study_id,
            "image_id": image_id,
            "img_path": path
        }


def collater_for_concept_detection(data):
    image = [s["image"] for s in data]
    res_bbox_tensor = [s["target"]["boxes"] for s in data]
    image_path = [s['img_path'] for s in data]

    max_num_annots = max(annot.shape[0] for annot in res_bbox_tensor)
    if max_num_annots > 0:
        annot_padded = torch.ones((len(res_bbox_tensor), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(res_bbox_tensor):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(res_bbox_tensor), 1, 5)) * -1

    return {
        "image": torch.stack(image),
        "res_bbox_tensor": annot_padded,
        "image_path": image_path
    }


class MammoDataset_concept(Dataset):
    def __init__(self, args, df, dataset, transform=None, windowing=False):
        self.df = df
        self.dir_path = args.data_dir / args.img_dir
        self.dataset = dataset
        self.target_dataset = args.target_dataset
        self.transform = transform
        self.args = args

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = None
        study_id = None
        laterality = self.df.iloc[idx]['laterality']
        if self.dataset.lower() == 'upmc' and self.target_dataset.lower() == 'upmc':
            study_id = str(self.df.iloc[idx]['STUDY_ID'])
            img_path = self.dir_path / f'Patient_{study_id}' / self.df.iloc[idx]['IMAGE_ID']

        elif self.dataset.lower() == 'upmc' and self.target_dataset.lower() == 'rsna':
            study_id = self.df.iloc[idx]['STUDY_ID']
            img_path = self.dir_path / str(study_id) / str(self.df.iloc[idx]['IMAGE_ID'])

        elif self.dataset.lower() == 'vindr':
            study_id = str(self.df.iloc[idx]['study_id'])
            img_path = self.dir_path / f'{study_id}' / self.df.iloc[idx]['image_id']
            img_path = f'{img_path}.png'

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        img = img.astype('float32')
        img -= img.min()
        img /= img.max()
        img = torch.tensor((img - self.args.mean) / self.args.std, dtype=torch.float32)

        y = None
        if self.dataset.lower() == 'upmc' and self.target_dataset.lower() == 'rsna':
            y = torch.tensor(self.df.iloc[idx]['cancer'], dtype=torch.long)
        elif self.args.model_type.lower() == 'concept-classifier' and self.args.concept.lower() == 'clip_v1':
            y = self.df.iloc[idx]['CLIP_V1']
        elif self.args.model_type.lower() == 'concept-classifier' and self.args.concept.lower() == 'mark_v1':
            y = self.df.iloc[idx]['MARK_V1']
        elif self.args.model_type.lower() == 'concept-classifier' and self.args.concept.lower() == 'mole_v1':
            y = self.df.iloc[idx]['MOLE_V1']
        elif self.args.model_type.lower() == 'concept-classifier' and self.args.concept.lower() == 'scar_v1':
            y = self.df.iloc[idx]['SCAR_V1']
        elif self.args.model_type.lower() == 'concept-classifier' and self.args.concept.lower() == 'architectural_distortion':
            y = self.df.iloc[idx]['Architectural_Distortion']
        elif self.args.model_type.lower() == 'concept-classifier' and self.args.concept.lower() == 'asymmetry':
            y = self.df.iloc[idx]['Asymmetry']
        elif self.args.model_type.lower() == 'concept-classifier' and self.args.concept.lower() == 'focal_asymmetry':
            y = self.df.iloc[idx]['Focal_Asymmetry']
        elif self.args.model_type.lower() == 'concept-classifier' and self.args.concept.lower() == 'global_asymmetry':
            y = self.df.iloc[idx]['Global_Asymmetry']
        elif self.args.model_type.lower() == 'concept-classifier' and self.args.concept.lower() == 'mass':
            y = self.df.iloc[idx]['Mass']
        elif self.args.model_type.lower() == 'concept-classifier' and self.args.concept.lower() == 'nipple_retraction':
            y = self.df.iloc[idx]['Nipple_Retraction']
        elif self.args.model_type.lower() == 'concept-classifier' and self.args.concept.lower() == 'skin_retraction':
            y = self.df.iloc[idx]['Skin_Retraction']
        elif self.args.model_type.lower() == 'concept-classifier' and self.args.concept.lower() == 'skin_thickening':
            y = self.df.iloc[idx]['Skin_Thickening']
        elif self.args.model_type.lower() == 'concept-classifier' and self.args.concept.lower() == 'suspicious_calcification':
            y = self.df.iloc[idx]['Suspicious_Calcification']
        elif self.args.model_type.lower() == 'concept-classifier' and self.args.concept.lower() == 'suspicious_lymph_node':
            y = self.df.iloc[idx]['Suspicious_Lymph_Node']

        if self.target_dataset.lower() == 'rsna':
            return {
                'x': img.unsqueeze(0),
                'y': y,
                'img_path': str(img_path),
                'study_id': study_id,
                'laterality': laterality
            }
        elif self.dataset.lower == "vindr":
            boxes = [
                self.df.iloc[idx]["resized_xmin"],
                self.df.iloc[idx]["resized_ymin"],
                self.df.iloc[idx]["resized_xmax"],
                self.df.iloc[idx]["resized_ymax"]
            ]
            return {
                'x': img.unsqueeze(0),
                'y': y.astype(np.float32),
                'img_path': str(img_path),
                'boxes': torch.tensor(boxes)
            }
        else:
            return {
                'x': img.unsqueeze(0),
                'y': y.astype(np.float32),
                'img_path': str(img_path),
                'boxes': torch.tensor([0, 0, 0, 0])
            }


def plot_image_with_boxes(image, boxes):
    fig, ax = plt.subplots(1)
    image = image[0].numpy()
    ax.imshow(image, cmap=plt.cm.bone)
    for box in boxes:
        xmin, ymin, xmax, ymax, _ = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()
