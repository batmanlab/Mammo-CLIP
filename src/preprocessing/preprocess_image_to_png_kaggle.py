from IPython.core.display import display

import glob

import os
from joblib import Parallel, delayed
import shutil
import cv2
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

import ctypes
import pydicom
import dicomsdl
import argparse
import torch
import timm
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from nvidia.dali.types import DALIDataType
from pydicom.filebase import DicomBytesIO
from nvidia.dali.backend import TensorGPU, TensorListGPU

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


def feed_ndarray(dali_tensor, arr, cuda_stream=None):
    """
    Copy contents of DALI tensor to PyTorch's Tensor.

    Parameters
    ----------
    `dali_tensor` : nvidia.dali.backend.TensorCPU or nvidia.dali.backend.TensorGPU
                    Tensor from which to copy
    `arr` : torch.Tensor
            Destination of the copy
    `cuda_stream` : torch.cuda.Stream, cudaStream_t or any value that can be cast to cudaStream_t.
                    CUDA stream to be used for the copy
                    (if not provided, an internal user stream will be selected)
                    In most cases, using pytorch's current stream is expected (for example,
                    if we are copying to a tensor allocated with torch.zeros(...))
    """
    dali_type = to_torch_type[dali_tensor.dtype]

    assert dali_type == arr.dtype, ("The element type of DALI Tensor/TensorList"
                                    " doesn't match the element type of the target PyTorch Tensor: "
                                    "{} vs {}".format(dali_type, arr.dtype))
    assert dali_tensor.shape() == list(arr.size()), \
        ("Shapes do not match: DALI tensor has size {0}, but PyTorch Tensor has size {1}".
         format(dali_tensor.shape(), list(arr.size())))
    cuda_stream = types._raw_cuda_stream(cuda_stream)

    # turn raw int to a c void pointer
    c_type_pointer = ctypes.c_void_p(arr.data_ptr())
    if isinstance(dali_tensor, (TensorGPU, TensorListGPU)):
        stream = None if cuda_stream is None else ctypes.c_void_p(cuda_stream)
        dali_tensor.copy_to_external(c_type_pointer, stream, non_blocking=True)
    else:
        dali_tensor.copy_to_external(c_type_pointer)
    return arr


##############################
####### Utils ################
##############################
def convert_dicom_to_j2k(IMG_PATH, patient_id, img_id, save_folder=""):
    patient = patient_id
    image = img_id
    file = IMG_PATH + f"{patient}/{image}.dcm"
    dcmfile = pydicom.dcmread(file)

    if dcmfile.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.90':
        with open(file, 'rb') as fp:
            raw = DicomBytesIO(fp.read())
            ds = pydicom.dcmread(raw)
        offset = ds.PixelData.find(b"\x00\x00\x00\x0C")  # <---- the jpeg2000 header info we're looking for
        hackedbitstream = bytearray()
        hackedbitstream.extend(ds.PixelData[offset:])
        with open(save_folder + f"{patient}_{image}.jp2", "wb") as binary_file:
            binary_file.write(hackedbitstream)


@pipeline_def
def j2k_decode_pipeline(j2kfiles):
    jpegs, _ = fn.readers.file(files=j2kfiles)
    images = fn.experimental.decoders.image(jpegs, device='mixed', output_type=types.ANY_DATA,
                                            dtype=DALIDataType.UINT16)
    return images


def normalised_to_8bit(image, photometric_interpretation):
    if photometric_interpretation == 'MONOCHROME1':
        image = image.max() - image
    xmin = image.min()
    xmax = image.max()
    norm = np.empty_like(image, dtype=np.uint8)
    dicomsdl.util.convert_to_uint8(image, norm, xmin, xmax)

    return norm


# Count up the continuing "1"
# Ex. [0,1,1,0,1,0,0,1,1,1,0] -> [-1,2,2,-1,1,-1,-1,3,3,3,-1]
# Numpy
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


def np_ExtractBreast(img):
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


# https://www.kaggle.com/code/raddar/convert-dicom-to-np-array-the-correct-way/notebook
def save_array(IMG_PATH, SIZE, SAVE_FOLDER, patient_id, img_id, voi_lut=False, fix_monochrome=True):
    path = IMG_PATH + f"{patient_id}/{img_id}.dcm"

    dicom = pydicom.dcmread(path)
    if dicom.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.90':  # ALREADY PROCESSED
        return

    dicom = dicomsdl.open(path)
    data = dicom.pixelData()
    data = data[5:-5, 5:-5]
    if fix_monochrome and dicom.getPixelDataInfo()['PhotometricInterpretation'] == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    data = np_ExtractBreast(data)
    data = cv2.resize(data, SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(SAVE_FOLDER + f"{patient_id}/{img_id}.png", data)


# 1. Since background in a image has constant values in a row- and col-wise direction,
# the breast of interest can be extracted by detecting non-constant elements.
# 2. Even if some objects other than breasts were seen in a image,
# because they are usually smaller, we can extract the breast by picking up the larger one.
# Heuristic way may not be always perfect....

# Count up the continuing "1"
# ex. [0,1,1,0,1,0,0,1,1,1,0] -> [-1,2,2,-1,1,-1,-1,3,3,3,-1]
# Pytroch
def torch_CountUpContinuingOnes(b_arr):
    # indice continuing zeros from left side.
    # [0,1,1,0,1,0,0,1,1,1,0] -> [0,0,0,3,3,5,6,6,6,6,10]
    left = torch.arange(len(b_arr))
    left[b_arr > 0] = 0
    left = torch.cummax(left, dim=-1)[0]

    # from right side.
    # [0,1,1,0,1,0,0,1,1,1,0] -> [0,3,3,3,5,5,6,10,10,10,10]
    rev_arr = torch.flip(b_arr, [-1])
    right = torch.arange(len(rev_arr))
    right[rev_arr > 0] = 0
    right = torch.cummax(right, dim=-1)[0]
    right = len(rev_arr) - 1 - torch.flip(right, [-1])

    return right - left - 1


def torch_ExtractBreast(img_ori):
    img = torch.where(img_ori <= 40, torch.zeros_like(img_ori), img_ori)  # To detect backgrounds easily
    height, _ = img.shape

    # whether each col is non-constant or not
    y_a = height // 2 + int(height * 0.4)
    y_b = height // 2 - int(height * 0.4)
    b_arr = img[y_b:y_a].to(torch.float32).std(dim=0) != 0
    continuing_ones = torch_CountUpContinuingOnes(b_arr)
    # longest should be the breast
    col_ind = torch.where(continuing_ones == continuing_ones.max())[0]
    img = img[:, col_ind]

    # whether each row is non-constant or not
    _, width = img.shape
    x_a = width // 2 + int(width * 0.4)
    x_b = width // 2 - int(width * 0.4)
    b_arr = img[:, x_b:x_a].to(torch.float32).std(axis=1) != 0
    continuing_ones = torch_CountUpContinuingOnes(b_arr)
    # longest should be the breast
    row_ind = torch.where(continuing_ones == continuing_ones.max())[0]

    return img_ori[row_ind][:, col_ind]


def convert_dicom_to_png(SIZE, IMG_PATH, SAVE_FOLDER, J2K_FOLDER, df):
    print("Number of images :", len(df))

    os.makedirs(SAVE_FOLDER, exist_ok=True)
    for patient_id in df['patient_id'].unique():
        os.makedirs(SAVE_FOLDER + str(patient_id), exist_ok=True)

    if len(df) > 100:
        N_CHUNKS = 4
    else:
        N_CHUNKS = 1

    CHUNKS = [(len(df) / N_CHUNKS * k, len(df) / N_CHUNKS * (k + 1)) for k in range(N_CHUNKS)]
    CHUNKS = np.array(CHUNKS).astype(int)

    if torch.cuda.is_available():
        for chunk in tqdm(CHUNKS):
            os.makedirs(J2K_FOLDER, exist_ok=True)

            _ = Parallel(n_jobs=2)(
                delayed(convert_dicom_to_j2k)(IMG_PATH, patient_id, img_id, save_folder=J2K_FOLDER)
                for patient_id, img_id in
                zip(df.iloc[chunk[0]: chunk[1]]['patient_id'].values, df.iloc[chunk[0]: chunk[1]]['image_id'].values)
            )

            j2kfiles = glob.glob(J2K_FOLDER + "*.jp2")
            print(f"Length of j2kfiles: {len(j2kfiles)}")
            if not len(j2kfiles):
                continue

            pipe = j2k_decode_pipeline(j2kfiles, batch_size=1, num_threads=2, device_id=0, debug=True)
            pipe.build()

            for f in j2kfiles:
                patient, image = f.split('/')[-1][:-4].split('_')
                dicom = pydicom.dcmread(IMG_PATH + f"{patient}/{image}.dcm")

                out = pipe.run()

                # Dali -> Torch
                img = out[0][0]
                img_torch = torch.empty(img.shape(), dtype=torch.int16, device="cuda")
                print(f"{patient}/{image}")
                feed_ndarray(img, img_torch, cuda_stream=torch.cuda.current_stream(device=0))
                img = img_torch.float()
                img = img.reshape(img.shape[0], img.shape[1])

                # read_mammography()
                img = img[5:-5, 5:-5]
                if dicom.PhotometricInterpretation == "MONOCHROME1":
                    img = img.max() - img
                min_, max_ = img.min(), img.max()
                img = (img - min_) / (max_ - min_)
                img = (img * 255)

                # extract the breast of interest
                img = torch_ExtractBreast(img)

                # Back to CPU + SAVE
                img = img.cpu().numpy().astype(np.uint8)
                img = cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA)

                #             fig, axes = plt.subplots(figsize=(5, 5), ncols=1, nrows=1)
                #             axes.imshow(img, cmap='bone')

                cv2.imwrite(SAVE_FOLDER + f"{patient}/{image}.png", img)
                print(SAVE_FOLDER + f"{patient}/{image}.png")

            shutil.rmtree(J2K_FOLDER)

        #     process remaining on cpu
        results = Parallel(n_jobs=2)(
            delayed(save_array)(IMG_PATH, SIZE, SAVE_FOLDER, patient_id, img_id)
            for patient_id, img_id in tqdm(zip(df['patient_id'].values, df['image_id'].values), total=len(df))
        )


def main():
    parser = argparse.ArgumentParser(description="Process image configurations for RSNA Breast Imaging.")
    parser.add_argument('--phase', type=str, default='train', help='Phase of processing, e.g., "test"')
    parser.add_argument('--width', type=int, default=912, help='The width of the image')
    parser.add_argument('--height', type=int, default=1520, help='The height of the image')
    parser.add_argument('--base_folder', type=str,
                        default='/restricted/projectnb/batmanlab/shawn24/PhD/RSNA_Breast_Imaging/Dataset/RSNA_Cancer_Detection',
                        help='Base folder for dataset and outputs')

    args = parser.parse_args()

    save_folder = os.path.join(args.base_folder, f"mammo_clip/{args.phase}_images_png/")
    img_path = os.path.join(args.base_folder, f"{args.phase}_images/")
    j2k_folder = os.path.join(args.base_folder, "tmp/j2k/")
    df = pd.read_csv(os.path.join(args.base_folder, f"{args.phase}.csv"))
    SIZE = (args.width, args.height)

    if args.phase == 'train':
        df = df.reset_index(drop=True)
    else:
        df = df

    print('torch version:', torch.__version__)
    print('timm version:', timm.__version__)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    convert_dicom_to_png(SIZE, img_path, save_folder, j2k_folder, df)


if __name__ == "__main__":
    main()