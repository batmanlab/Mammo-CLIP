import os
import sys

from IPython.core.display import display

import os
import dicomsdl
import numpy as np
import pandas as pd

import torch
import timm
import nvidia.dali.types as types

# from nvidia.dali.plugin.pytorch import feed_ndarray, to_torch_type

# we need to patch DALI for Int16 support

to_torch_type = {
    types.DALIDataType.FLOAT: torch.float32,
    types.DALIDataType.FLOAT64: torch.float64,
    types.DALIDataType.FLOAT16: torch.float16,
    types.DALIDataType.UINT8: torch.uint8,
    types.DALIDataType.INT8: torch.int8,
    types.DALIDataType.UINT16: torch.int16,
    types.DALIDataType.INT16: torch.int16,
    types.DALIDataType.INT32: torch.int32,
    types.DALIDataType.INT64: torch.int64
}


def np_CountUpContinuingOnes(b_arr):
    # indice continuing zeros from left side.
    # ex: [0,1,1,0,1,0,0,1,1,1,0] -> [0,0,0,3,3,5,6,6,6,6,10]
    left = np.arange(len(b_arr))
    left[b_arr > 0] = 0
    left = np.maximum.accumulate(left)

    # from right side.
    # ex: [0,1,1,0,1,0,0,1,1,1,0] -> [0,3,3,3,5,5,6,10,10,10,10]
    rev_arr = b_arr[::-1]
    right = np.arange(len(rev_arr))
    right[rev_arr > 0] = 0
    right = np.maximum.accumulate(right)
    right = len(rev_arr) - 1 - right[::-1]

    return right - left - 1


def adjust_bounding_box(original_coords, left_crop, top_crop):
    x1, y1, x2, y2 = original_coords

    x1_new = x1 - left_crop
    y1_new = y1 - top_crop
    x2_new = x2 - left_crop
    y2_new = y2 - top_crop

    return x1_new, y1_new, x2_new, y2_new


def ExtractBreast(img, true_bounding_box):
    img_copy = img.copy()
    img = np.where(img <= 40, 0, img)  # To detect backgrounds easily
    height, _ = img.shape

    # whether each col is non-constant or not
    y_a = height // 2 + int(height * 0.4)
    y_b = height // 2 - int(height * 0.4)
    b_arr = img[y_b:y_a].std(axis=0) != 0
    continuing_ones = np_CountUpContinuingOnes(b_arr)
    # longest should be the breast
    col_ind = np.where(continuing_ones == continuing_ones.max())[0]
    img = img[:, col_ind]

    # whether each row is non-constant or not
    _, width = img.shape
    x_a = width // 2 + int(width * 0.4)
    x_b = width // 2 - int(width * 0.4)
    b_arr = img[:, x_b:x_a].std(axis=1) != 0
    continuing_ones = np_CountUpContinuingOnes(b_arr)
    # longest should be the breast
    row_ind = np.where(continuing_ones == continuing_ones.max())[0]
    adjusted_coords = adjust_bounding_box(true_bounding_box, col_ind[0], row_ind[0])
    return img_copy[row_ind][:, col_ind], adjusted_coords


def save_imgs(in_path, original_bbox, SIZE=(912, 1520)):
    dicom = dicomsdl.open(in_path)
    data = dicom.pixelData()
    data = data[5:-5, 5:-5]
    if dicom.getPixelDataInfo()['PhotometricInterpretation'] == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    adjusted_bbox = (
        max(0, original_bbox[0] - 5),  # Adjust top, making sure it's not less than 0
        max(0, original_bbox[1] - 5),  # Adjust left, making sure it's not less than 0
        min(data.shape[0], original_bbox[2] - 5),  # Adjust bottom, considering the new image shape
        min(data.shape[1], original_bbox[3] - 5),  # Adjust right, considering the new image shape
    )

    extracted_breast, adjusted_boxes = ExtractBreast(data, adjusted_bbox)
    scale_x = SIZE[0] / extracted_breast.shape[1]
    scale_y = SIZE[1] / extracted_breast.shape[0]
    # img = cv2.resize(extracted_breast, SIZE, interpolation=cv2.INTER_AREA)
    resized_xmin = adjusted_boxes[0]
    resized_ymin = adjusted_boxes[1]
    resized_xmax = adjusted_boxes[2]
    resized_ymax = adjusted_boxes[3]
    resized_width = resized_xmax - resized_xmin
    resized_height = resized_ymax - resized_ymin
    print(resized_xmin, resized_ymin, resized_width, resized_height)

    resized_xmin = (resized_xmin * scale_x)
    resized_ymin = (resized_ymin * scale_y)
    resized_xmax = (resized_xmax * scale_x)
    resized_ymax = (resized_ymax * scale_y)

    # Create a Rectangle patch for the resized bounding box
    resized_width = resized_xmax - resized_xmin
    resized_height = resized_ymax - resized_ymin

    print(resized_xmin, resized_ymin, resized_width, resized_height)
    return resized_xmin, resized_ymin, resized_xmax, resized_ymax


###############################
### source url: https://www.kaggle.com/code/masato114/rsna-generate-train-images/notebook
###############################
print('torch version:', torch.__version__)
print('timm version:', timm.__version__)
##############################
####### Load data ################
##############################

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

train_df = pd.read_csv(
    '/restricted/projectnb/batmanlab/shawn24/PhD/RSNA_Breast_Imaging/Dataset/External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/finding_annotations.csv')
train_df = train_df.head(2255)
display(train_df.head(5))
display(train_df.shape)

phase = 'train'  # 'test'

if phase == 'train':
    df = train_df.reset_index(drop=True)

IMG_PATH = f"/restricted/projectnb/batmanlab/shawn24/PhD/RSNA_Breast_Imaging/Dataset/External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images"
# test_images = glob.glob(f"{IMG_PATH}*/*.dcm")

print("Number of images :", len(df))

_SIZE = (912, 1520)
resized_xmin_arr = []
resized_ymin_arr = []
resized_xmax_arr = []
resized_ymax_arr = []

for index, row in df.iterrows():
    study_id = row["study_id"]
    image_id = row["image_id"]
    original_bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
    _in_path = os.path.join(IMG_PATH, study_id, f"{image_id}.dicom")
    print(_in_path)
    print(f"=====================>>>>> {index} <<<<<=====================")
    resized_xmin, resized_ymin, resized_xmax, resized_ymax = save_imgs(
        in_path=_in_path, original_bbox=original_bbox, SIZE=_SIZE
    )
    resized_xmin_arr.append(resized_xmin)
    resized_ymin_arr.append(resized_ymin)
    resized_xmax_arr.append(resized_xmax)
    resized_ymax_arr.append(resized_ymax)

df['resized_xmin'] = resized_xmin_arr
df['resized_ymin'] = resized_ymin_arr
df['resized_xmax'] = resized_xmax_arr
df['resized_ymax'] = resized_ymax_arr

df.to_csv(
    '/restricted/projectnb/batmanlab/shawn24/PhD/RSNA_Breast_Imaging/Dataset/External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection.csv',
    index=False
)
