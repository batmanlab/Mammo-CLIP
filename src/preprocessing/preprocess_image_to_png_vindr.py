import os
import sys

from IPython.core.display import display

sys.path.append(os.path.abspath("/restricted/projectnb/batmanlab/shawn24/PhD/RSNA_Breast_Imaging"))
import os
import dicomsdl
import argparse
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

import torch
import timm
import nvidia.dali.types as types
import cv2

### source url: https://www.kaggle.com/code/masato114/rsna-generate-train-images/notebook
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


def ExtractBreast(img):
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

    return img_copy[row_ind][:, col_ind]


def save_imgs(in_path, out_path, SIZE=(912, 1520)):
    dicom = dicomsdl.open(in_path)
    data = dicom.pixelData()
    data = data[5:-5, 5:-5]
    if dicom.getPixelDataInfo()['PhotometricInterpretation'] == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    img = ExtractBreast(data)
    img = cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, img)

    print(out_path)


def main():
    parser = argparse.ArgumentParser(description="Process image configurations for VinDr Breast Imaging.")
    parser.add_argument('--phase', type=str, default='train', help='Phase of processing, e.g., "test"')
    parser.add_argument('--width', type=int, default=912, help='The width of the image')
    parser.add_argument('--height', type=int, default=1520, help='The height of the image')
    parser.add_argument('--base_folder', type=str,
                        default='/restricted/projectnb/batmanlab/shawn24/PhD/RSNA_Breast_Imaging/Dataset/External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0',
                        help='Base folder for dataset and outputs')

    args = parser.parse_args()
    save_folder = os.path.join(args.base_folder, f"mammo_clip/images_png/")
    img_path = os.path.join(args.base_folder, f"images")
    j2k_folder = os.path.join(args.base_folder, "tmp/j2k/")
    df = pd.read_csv(os.path.join(args.base_folder, f"breast-level_annotations.csv"))
    SIZE = (args.width, args.height)

    if args.phase == 'train':
        df = df.reset_index(drop=True)
    else:
        df = df

    print('df:', df.shape)
    print('torch version:', torch.__version__)
    print('timm version:', timm.__version__)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    os.makedirs(save_folder, exist_ok=True)
    for patient_id in df['study_id'].unique():
        os.makedirs(os.path.join(save_folder, str(patient_id)), exist_ok=True)

    if len(df) > 100:
        N_CHUNKS = 4
    else:
        N_CHUNKS = 1

    CHUNKS = [(len(df) / N_CHUNKS * k, len(df) / N_CHUNKS * (k + 1)) for k in range(N_CHUNKS)]
    CHUNKS = np.array(CHUNKS).astype(int)

    for chunk in tqdm(CHUNKS):
        for patient_id, img_id in zip(df.iloc[chunk[0]: chunk[1]]['study_id'].values,
                                      df.iloc[chunk[0]: chunk[1]]['image_id'].values):
            _in_path = os.path.join(img_path, patient_id, f"{img_id}.dicom")
            _out_path = os.path.join(save_folder, patient_id, f"{img_id}.png")

            print(_in_path, _out_path)
            save_imgs(in_path=_in_path, out_path=_out_path, SIZE=SIZE)


if __name__ == "__main__":
    main()
