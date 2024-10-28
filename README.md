# Mammo-CLIP: A Vision Language Foundation Model to Enhance Data Efficiency and Robustness in Mammography

[![Project](https://img.shields.io/badge/Project%20page-lightgreen)](https://shantanu-ai.github.io/projects/MICCAI-2024-Mammo-CLIP/)
[![Paper](https://img.shields.io/badge/Paper-9cf)](https://papers.miccai.org/miccai-2024/paper/0926_paper.pdf)
[![Hugging Face](https://img.shields.io/badge/Checkpoints-Hugging%20Face-yellow)](https://huggingface.co/shawn24/Mammo-CLIP/tree/main/Pre-trained-checkpoints/)
[![Pre-training Checkpoints](https://img.shields.io/badge/Checkpoints-Google%20Drive-blue)](https://github.com/batmanlab/Mammo-CLIP/blob/main/README.md#mammo-clip-checkpoints)
[![VinDr png data](https://img.shields.io/badge/VinDr%20Mammogram%20png%20images-lightblue)](https://www.kaggle.com/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png)
[![Poster](https://img.shields.io/badge/Poster-orange)](https://github.com/shantanu-ai/shantanu-ai.github.io/tree/main/projects/MICCAI-2024-Mammo-CLIP/static/data/Mammo-CLIP-MICCAI-24-poster-v1.pdf)
![](https://visitor-badge.laobi.icu/badge?page_id=batmanlab.Mammo-CLIP&right_color=%23FFA500)

[Shantanu Ghosh<sup>1</sup>](https://shantanu-ai.github.io/)
, [Clare B. Poynton<sup>2</sup>](https://www.bumc.bu.edu/camed/profile/clare-poynton/)
, [Shyam Visweswaran<sup>3</sup>](https://www.thevislab.com/lab/doku.php),
[Kayhan Batmanghelich<sup>1</sup>](https://www.batman-lab.com/)
<br/>
<sup>1</sup>BU ECE, <sup>2</sup> BUMC, <sup>3</sup> Pitt DBMI <br/>

#### ⚠️ WARNING: Look for `/restricted/projectnb/batmanlab/shawn24/PhD` and replace it with your own path. E.g, `.src/codebase/breastclip/data/datasets/imagetext.py`, change the json path

#### ⚠️ WARNING: There is a plethora of pre-processing settings available for RSNA and VinDr Mammo datasets. We recommend using the pre-processing discussed in the following sections. We are not responsible for any discrepancies in the results due to different pre-processing settings. If you use the VinDr png dataset uploaded in kaggle, it is fully pre-processed. Else you can use the pre-processing scripts provided in the following sections.

#### ⚠️ WARNING: If you find the `punkt_tab` error, run the following command in the python environment:

```python
import nltk

nltk.download('punkt_tab')
```

## FAQ

After going through the instruction, it is recommended to visit the following queries logged in the issues:

* [Issue-2](https://github.com/batmanlab/Mammo-CLIP/issues/2) for any further clarification on pretraining.
* [Issue-10](https://github.com/batmanlab/Mammo-CLIP/issues/10) for getting the embeddings.
* [Issue-6](https://github.com/batmanlab/Mammo-CLIP/issues/6) for further clarification on the downstream tasks and
  corresponding datasets.
* [Issue-13](https://github.com/batmanlab/Mammo-CLIP/issues/13) for setting up the baselines.
* [Issue-9](https://github.com/batmanlab/Mammo-CLIP/issues/9) for problems related to BioClinincalBert from Hugging
  Face.

If we hear more queries, we may add a separate FAQs in the future.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Data Download](#data-download)
3. [Pre-processing Images](#pre-processing-images)
4. [Data Preparation for Pretraining](#data-preparation-for-pretraining)
5. [Data Preparation for Downstream Evaluation Tasks](#data-preparation-for-downstream-evaluation-tasks)
6. [Final Dataset Directory Structures](#final-dataset-directory-structures)
7. [Mammo-CLIP checkpoints](#mammo-clip-checkpoints)
8. [Pretraining Mammo-CLIP](#pretraining-mammo-clip)
9. [Creating classifiers and detectors for downstream evaluations](#creating-classifiers-and-detectors-for-downstream-evaluations)
10. [Evaluation](#evaluation)
11. [Tutorial Notebooks](#tutorial-notebooks)
12. [Additional Scripts](#additional-scripts)
13. [Mammo-FActOR](#mammo-factor)
14. [Citation](#citation)
15. [License and Copyright](#license-and-copyright)
16. [Contact](#contact)
17. [Acknowledgements](#acknowledgements)
18. [Contributing](#contributing)

## Environment Setup

Use [environment.yml](https://github.com/batmanlab/Mammo-CLIP/blob/main/environment.yml) to setup the environment.

```bash
git clone git@github.com:batmanlab/Mammo-CLIP.git
cd Mammo-CLIP
conda env create --name Mammo-CLIP -f environment.yml
conda activate Mammo-CLIP
```

Mammo-CLIP is implemented with following specification:

* Python version: 3.8.18
* PyTorch version: 2.2.2
* CUDA version: 11.8

## Data Download

Download the original versions VinDr and RSNA from the links for downstream evaluations:

- [RSNA](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)
- [VinDr](https://vindr.ai/datasets/mammo)

For the PNG images converted from the original Dicom images, as mentioned in the preprocessing steps in the paper, refer
to the following links:

- [VinDr](https://www.kaggle.com/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png)

To preprocess the dicom images directly, follow the instructions in the next section. If you downloaded the PNG images,
skip the preprocessing steps.

## Pre-processing images

### Convert to png: RSNA

```bash
python ./src/preprocessing/preprocess_image_to_png_kaggle.py \
  --phase="test" \
  --base_folder="/restricted/projectnb/batmanlab/shawn24/PhD/RSNA_Breast_Imaging/Dataset/RSNA_Cancer_Detection"
```

### convert to png: VinDr

```bash
python ./src/preprocessing/preprocess_image_to_png_vindr.py \
  --phase="test" \
  --base_folder="/restricted/projectnb/batmanlab/shawn24/PhD/RSNA_Breast_Imaging/Dataset/External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
```

## Data preparation for pretraining

### Image-text dataset

1. Our image-text dataset is an in-house dataset from UPMC. You can have your own image+text dataset where images are 2D
   mammograms and texts are radiology reports. If you have access to such dataset, follow the setup here. Extract the `IMPRESSION` and `FINDINGS` sections from the
   report and create a csv. The sample
   csv: [upmc_dicom_consolidated_final_folds_BIRADS_num_1_report.csv](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/data_csv/upmc_dicom_consolidated_final_folds_BIRADS_num_1_report.csv)

2. Note the `FINDINGS` and `IMPRESSION` columns are
   used to generate the text for the image. The `HISTORY`, `FINDINGS` and `IMPRESSION` columns contains templated text
   due to privacy.

3. Next run the following command to augment the text with `upmc_dicom_consolidated_final_folds_BIRADS_num_1_report.csv`
   file:

```bash
# input: upmc_dicom_consolidated_final_folds_BIRADS_num_1_report.csv
# output: clip_pretrain_100.csv

python ./src/codebase/augment_text.py \
  --dataset-path="/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/codebase/data_csv" \
  --csv-path="upmc_dicom_consolidated_final_folds_BIRADS_num_1_report.csv" \
  --dataset="upmc" 
```

The `augment_text.py` script will do the preprocessing for the `FINDINGS` and `IMPRESSION` columns by converting them to
lower case and removing punctuations. Look at the `_split_report_into_segment_concat` function in the script for more
details.

4. The csv file of the final image-text dataset should have the following format:

| index | patient_id | laterality              | image                                                   | view                                                                         | CC                                                              | MLO                                                              | text                           | text_augment                                       |
|-------|------------|-------------------------|---------------------------------------------------------|------------------------------------------------------------------------------|-----------------------------------------------------------------|------------------------------------------------------------------|--------------------------------|----------------------------------------------------|
| 0     | patient_id | laterality ('R' or 'L') | List of all image_paths for patient_id-laterality combo | List of views for patient_id-laterality combo (only 'CC' and 'MLO' are used) | List of image paths for CC view for patient_id-laterality combo | List of image paths for MLO view for patient_id-laterality combo | List of [findings, impression] | List of [augmented findings, augmented impression] |

5. The final sample csv file as the output of `step3` is
   here: [clip_pretrain_100.csv](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/data_csv/clip_pretrain_100.csv). `clip_pretrain_100.csv`
   is used for pretraining the image-text variant of Mammo-CLIP.

### Image-label dataset

We use VinDr dataset as image-label dataset though it can be expanded to any such datasets. Make sure that every patient
should have atleast one CC and MLO image per laterality. So if you are planning to use it in the pre-training setup, use
the
following notebook to preprocess the VinDr dataset:

```bash
./src/codebase/notebooks/preprocess-clip/VinDr.ipynb
```

When you download the VinDr dataset, you will get these two csv
files: [breast-level_annotations.csv](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/data_csv/breast-level_annotations.csv)
and [finding_annotations.csv](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/data_csv/finding_annotations.csv)
. We preprocess the `finding_annotations.csv` file to
get [vindr_detection_v1_folds.csv](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/data_csv/vindr_detection_v1_folds.csv)
. [VinDr.ipynb](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/notebooks/preprocess-clip/VinDr.ipynb)
notebook requires vindr_detection_v1_folds.csv file as input and
generate [clip_vindr_final.csv](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/data_csv/clip_vindr_final.csv)
file.

**Both [clip_pretrain_100.csv](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/data_csv/clip_pretrain_100.csv)
and [clip_vindr_final.csv](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/data_csv/clip_vindr_final.csv)
files are used for pretraining the image-text and
image-label variant of Mammo-CLIP.**

The csv file of the final image-label (VinDr) dataset should have the following format:

| index | patient_id | laterality              | image                                                   | view                                                                         | CC                                                                                | MLO                                                                                  | CC_FINDING                                                               | MLO_FINDING                                                               |
|-------|------------|-------------------------|---------------------------------------------------------|------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|--------------------------------------------------------------------------|---------------------------------------------------------------------------|
| 0     | patient_id | laterality ('R' or 'L') | List of all image_paths for patient_id-laterality combo | List of views for patient_id-laterality combo (only 'CC' and 'MLO' are used) | List of image paths for CC view for patient_id-laterality combo, e.g, [CC_img ..] | List of image paths for MLO view for patient_id-laterality combo, e.g, [MLO_img .. ] | Findings per image per laterality for CC view (see below for the format) | Findings per image per laterality for MLO view (see below for the format) |

**Explanation for CC_FINDING and MLO_FINDING Columns:**
In the above table, for the row, CC_FINDING can be expanded
as:

```
[
  [+ve findings for CC_img if laterality of CC_img is R],
  [+ve findings for CC_img if laterality of CC_img is L],
  [-ve findings for CC_img if laterality of CC_img is R],
  [-ve findings for CC_img if laterality of CC_img is L],
]
```

As VinDr contains a single image per patient-laterality combo, we did n

Similarly, in the above table, for the row, MLO_FINDING can be expanded
as:

```
[
  [+ve findings for MLO_img if laterality of MLO_img is R],
  [+ve findings for MLO_img if laterality of MLO_img is L],
  [-ve findings for MLO_img if laterality of MLO_img is R],
  [-ve findings for MLO_img if laterality of MLO_img is L],
]
```

## Data preparation for downstream evaluation tasks

Use the following csv files as metadata for the downstream tasks (classification, detection, zero-shot):

| Dataset | CSV                                                                                                                                  |
|---------|--------------------------------------------------------------------------------------------------------------------------------------|
| VinDr   | [vindr_detection_v1_folds.csv](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/data_csv/vindr_detection_v1_folds.csv) |
| RSNA    | [train_folds.csv](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/data_csv/train_folds.csv)                           | 

For detection/localization tasks, we have included the coordinates of the resized bounding boxes of VinDr in the above
csv file. Somebody interested in resizing the bounding boxes by themselves, run the following command
with [finding_annotations.csv](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/data_csv/finding_annotations.csv)
file as input:

```bash
python ./src/preprocessing/preprocess_VinDr_detector.py
````

## Final dataset directory structures

### Image+Text pretraining dataset

```bash
.
├── list_tree_files.sh
├── upmc_dicom_consolidated_final_folds_BIRADS_num_1_report.csv
├── clip_pretrain_100.csv
└── DICOM/images_png_CC_MLO/
    ├── Patient_100/
    │   ├── 1.png
    │   ├── 2.png
    └── Patient_200/
        ├── 3.png
        ├── 4.png
        ├── 53.png
        ├── 6.png
        └── 7.png
        

```

### VinDr

```bash
.
├── breast-level_annotations.csv
├── finding_annotations.csv
├── vindr_detection_v1_folds.csv 
├── clip_vindr_final.csv
└── images_png/
    ├── c7811f4575c1229ad4a7606de49ea68f/
    │   ├── 9eb4650a2b630e44074c403f6127c5a1.png
    │   ├── cc3fdc5d733a671f3000e20838e192d9.png
    │   ├── 181fd193d3b785dc9faafdaa8e1695fc.png
    │   └── 55eb5ea616abacd225e584ffc8be57da.png
    └── a1dd219b28806fc295fac20ceb147870/
        ├── 887cdcc99ebed66bd062ada6c8210152.png
        ├── 36f2921a2ac19eba7420c591c4c07ae4.png
        ├── 12dc17dfd9d30ea7c0c1ccb33a505085.png
        └── e22e4f297b4c82279e7b78a98417a6cd.png
```

### RSNA

```bash
.
├── train_folds.csv
├── train_images_png/
    ├── 59549/
    │   ├── 1154694388.png
    │   ├── 1192817932.png
    │   ├── 1979035704.png
    │   ├── 2022274082.png
    │   ├── 431013616.png
    │   ├── 457600713.png
    │   ├── 78005871.png
    │   └── 856162422.png
    └── 28242/
        ├── 1966298736.png
        ├── 233201459.png
        ├── 349787619.png
        └── 98615814.png


```

## Mammo-CLIP checkpoints

Following are the pre-training checkpoints of Mammo-CLIP:

| Model architecture | Checkpoints (Google  drive)                                                                            | Checkpoints (Hugging Face)                                                                                                 |
|--------------------|--------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| Best performance   | [Efficient-Net B5](https://drive.google.com/file/d/1c14IwqxkMRFD78BEhNA17n3b6C21fuQ1/view?usp=sharing) | [Efficient-Net B5](https://huggingface.co/shawn24/Mammo-CLIP/blob/main/Pre-trained-checkpoints/b5-model-best-epoch-7.tar)  |
| Lightweight        | [Efficient-Net B2](https://drive.google.com/file/d/1dNqicN0_Oeo4T4920eljxDX0x0htFgAc/view?usp=sharing) | [Efficient-Net B2](https://huggingface.co/shawn24/Mammo-CLIP/blob/main/Pre-trained-checkpoints/b2-model-best-epoch-10.tar) |

We have also uploaded the downstream checkpoints for classification and localization (both linear probe and finetuning)
with the image encoder of Efficient-Net B5 Mammo-CLIP for fold
0 [here](https://huggingface.co/shawn24/Mammo-CLIP/tree/main/Downstream-checkpoints).

## Pretraining Mammo-CLIP

For pretraining Efficient-Net B5 Mammo-CLIP with a single GPU, use the following command:

```bash
python ./src/codebase/train.py --config-name pre_train_b5_clip.yaml
```

For pretraining Efficient-Net B5 Mammo-CLIP with a 4 GPUs using pytorch-ddp, use the following command:

```bash
torchrun --nproc_per_node=4 ./src/codebase/train.py --config-name pre_train_b5_clip.yaml
```

All the `yaml` files for the config are
found [here](https://github.com/batmanlab/Mammo-CLIP/tree/main/src/codebase/configs).

* Use
  [pre_train_b5_clip.yaml](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/configs/pre_train_b5_clip.yaml)
  for pre-training image-text variant of Efficient-Net B5 Mammo-CLIP
* Use
  [pre_train_b2_clip.yaml](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/configs/pre_train_b2_clip.yaml)
  for pre-training image-text variant of Efficient-Net B2 Mammo-CLIP
* Use
  [pre_train_b5_w_vindr_clip.yaml](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/configs/pre_train_b5_w_vindr_clip.yaml)
  for pre-training image-text + image-label variant of Efficient-Net B5 Mammo-CLIP

## Creating classifiers and detectors for downstream evaluations

* For creating classifiers for downstream evaluations using the image encoder of Mammo-CLIP, use the
  class `BreastClipClassifier`
  in [breast-clip-classifier.py](https://github.com/batmanlab/Mammo-CLIP/blob/c9cc232368eaf0a6d55f1bea04490d9136362466/src/codebase/Classifiers/models/breast_clip_classifier.py#L6)
  file.
* For creating detectors for downstream evaluations using the image encoder of Mammo-CLIP, use the
  function `RetinaNet_efficientnet`
  in [detector_model.py](https://github.com/batmanlab/Mammo-CLIP/blob/c9cc232368eaf0a6d55f1bea04490d9136362466/src/codebase/Detectors/retinanet/detector_model.py#L357)
  file.

## Evaluation

### Zero-shot evaluation of Mammo-CLIP

```bash
FOLD=0
CKPT="b2-model-best-epoch-10.tar"
DIR="./Mammo-CLIP/src/codebase/outputs/upmc_clip/b2_detector_period_n"
FULL_CKPT="$DIR/checkpoints/fold_$FOLD/$CKPT"

python ./src/codebase/eval_zero_shot_clip.py \
  --config-name zs_clip.yaml hydra.run.dir=$DIR model.clip_check_point=$FULL_CKPT
```

Adjust the `CKPT` and `DIR` variables according to your setup.

### Linear probe vision encoder Mammo-CLIP on target classification task

```bash
python ./src/codebase/train_classifier.py \
  --data-dir '/restricted/projectnb/batmanlab/shawn24/PhD/RSNA_Breast_Imaging/Dataset' \
  --img-dir 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images_png' \
  --csv-file 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection_v1_folds.csv' \
  --clip_chk_pt_path "/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/codebase/outputs/upmc_clip/b5_detector_period_n/checkpoints/fold_0/b5-model-best-epoch-7.tar" \
  --data_frac 1.0 \
  --dataset 'ViNDr' \
  --arch 'upmc_breast_clip_det_b5_period_n_lp' \
  --label "Mass" \
  --epochs 30 \
  --batch-size 8 \
  --num-workers 0 \
  --print-freq 10000 \
  --log-freq 500 \
  --running-interactive 'n' \
  --n_folds 1 \
  --lr 5.0e-5 \
  --weighted-BCE 'y' \
  --balanced-dataloader 'n' 
```

* `data-dir`: root directory of the dataset
* `img-dir`: directory containing images, absolute path: `data-dir/img-dir`
* `csv-file`: csv file containing image paths and labels, absolute path: `data-dir/csv-file`
* `clip_chk_pt_path`: path to the checkpoint of the pre-trained Mammo-CLIP model
* `dataset`: dataset name, e.g., `ViNDr` or `RSNA`
* `data_frac`: fraction of the dataset to use for training, e.g., `1.0`, `0.5` etc
* `arch`: architecture of the model, e.g., `upmc_breast_clip_det_b5_period_n_lp` for Efficient-Net B5
  or `upmc_breast_clip_det_b2_period_n_lp` for Efficient-Net B2, pretrained on UPMC dataset.
  Also, `upmc_vindr_breast_clip_det_b5_period_n_lp` for Efficient-Net B5
  or `upmc_vindr_breast_clip_det_b2_period_n_lp` for Efficient-Net B2, pretrained on UPMC and VinDr datasets.
* `label`: target label for classification, e.g., `Mass`, `Suspicious_Calcification`or `density` for ViNDr
  dataset; `cancer` for RSNA dataset
* `running-interactive`: running on interactive mode. In this mode,the training will be done using 100 samples for
  sanity check

### Finetune vision encoder Mammo-CLIP on target classification task

```bash
python ./src/codebase/train_classifier.py \
  --data-dir '/restricted/projectnb/batmanlab/shawn24/PhD/RSNA_Breast_Imaging/Dataset' \
  --img-dir 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images_png' \
  --csv-file 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection_v1_folds.csv' \
  --clip_chk_pt_path "/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/codebase/outputs/upmc_clip/b5_detector_period_n/checkpoints/fold_0/b5-model-best-epoch-7.tar" \
  --data_frac 1.0 \
  --dataset 'ViNDr' \
  --arch 'upmc_breast_clip_det_b5_period_n_ft' \
  --label "Mass" \
  --epochs 30 \
  --batch-size 8 \
  --num-workers 0 \
  --print-freq 10000 \
  --log-freq 500 \
  --running-interactive 'n' \
  --n_folds 1 \
  --lr 5.0e-5 \
  --weighted-BCE 'y' \
  --balanced-dataloader 'n'
 ```

* `data-dir`: root directory of the dataset
* `img-dir`: directory containing images, absolute path: `data-dir/img-dir`
* `csv-file`: csv file containing image paths and labels, absolute path: `data-dir/csv-file`
* `clip_chk_pt_path`: path to the checkpoint of the pre-trained Mammo-CLIP model
* `dataset`: dataset name, e.g., `ViNDr` or `RSNA`
* `data_frac`: fraction of the dataset to use for training, e.g., `1.0`, `0.5` etc
* `arch`: `arch`: architecture of the model, e.g., `upmc_breast_clip_det_b5_period_n_ft` for Efficient-Net B5
  or `upmc_breast_clip_det_b2_period_n_ft` for Efficient-Net B2, pretrained on UPMC dataset.
  Also, `upmc_vindr_breast_clip_det_b5_period_n_ft` for Efficient-Net B5
  or `upmc_vindr_breast_clip_det_b2_period_n_ft` for Efficient-Net B2, pretrained on UPMC and VinDr datasets.
* `label`: target label for classification, e.g., `Mass`, `Suspicious_Calcification`or `density` for ViNDr
  dataset; `cancer` for RSNA dataset
* `running-interactive`: running on interactive mode. In this mode,the training will be done using 100 samples for
  sanity check

### Linear probe vision encoder Mammo-CLIP on target detection task

```bash
python ./src/codebase/train_detector.py \
  --data-dir '/restricted/projectnb/batmanlab/shawn24/PhD/RSNA_Breast_Imaging/Dataset' \
  --img-dir 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images_png' \
  --csv-file 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection_v1_folds.csv' \
  --clip_chk_pt_path "/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/codebase/outputs/upmc_clip/b5_detector_period_n/checkpoints/fold_0/b5-model-best-epoch-7.tar" \
  --dataset 'ViNDr' \
  --arch 'clip_b5_upmc' \
  --epochs 120 \
  --batch-size 7 \
  --freeze_backbone "y" \
  --data_frac 1.0 \
  --concepts 'Mass' \
  --print-freq 5000 \
  --log-freq 300 \
  --running-interactive 'n' \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --score-threshold 0.2
 ```

* `data-dir`: root directory of the dataset
* `img-dir`: directory containing images, absolute path: `data-dir/img-dir`
* `csv-file`: csv file containing image paths and labels, absolute path: `data-dir/csv-file`
* `clip_chk_pt_path`: path to the checkpoint of the pre-trained Mammo-CLIP model
* `dataset`: dataset name, e.g., `ViNDr`
* `data_frac`: fraction of the dataset to use for training, e.g., `1.0`, `0.5` etc
* `arch`: architecture of the model, e.g., `clip_b5_upmc` for Efficient-Net B5 or `clip_b2_upmc` for Efficient-Net B2,
  pretrained on UPMC dataset. Similarly, `clip_b5_upmc_vindr` for Efficient-Net B5 or `clip_b2_upmc_vindr` for
  Efficient-Net B2,
  pretrained on UPMC and VinDr datasets.
* `concepts`: target label for classification, e.g., `Mass`, `Suspicious Calcification` for ViNDr dataset
* `running-interactive`: running on interactive mode. In this mode,the training will be done using 100 samples for
  sanity check
* `freeze_backbone`: freeze the backbone of the model, for linear probe, set to `y`

### Finetune vision encoder Mammo-CLIP on target detection task

```bash
python ./src/codebase/train_detector.py \
  --data-dir '/restricted/projectnb/batmanlab/shawn24/PhD/RSNA_Breast_Imaging/Dataset' \
  --img-dir 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images_png' \
  --csv-file 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection_v1_folds.csv' \
  --clip_chk_pt_path "/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/codebase/outputs/upmc_clip/b5_detector_period_n/checkpoints/fold_0/b5-model-best-epoch-7.tar" \
  --dataset 'ViNDr' \
  --arch 'clip_b5_upmc' \
  --epochs 120 \
  --batch-size 7 \
  --freeze_backbone "n" \
  --data_frac 1.0 \
  --concepts 'Mass' \
  --print-freq 5000 \
  --log-freq 300 \
  --running-interactive 'n' \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --score-threshold 0.2
 ```

* `data-dir`: root directory of the dataset
* `img-dir`: directory containing images, absolute path: `data-dir/img-dir`
* `csv-file`: csv file containing image paths and labels, absolute path: `data-dir/csv-file`
* `clip_chk_pt_path`: path to the checkpoint of the pre-trained Mammo-CLIP model
* `dataset`: dataset name, e.g., `ViNDr`
* `data_frac`: fraction of the dataset to use for training, e.g., `1.0`, `0.5` etc
* `arch`: architecture of the model, e.g., `clip_b5_upmc` for Efficient-Net B5 or `clip_b2_upmc` for Efficient-Net B2,
  pretrained on UPMC dataset. Similarly, `clip_b5_upmc_vindr` for Efficient-Net B5 or `clip_b2_upmc_vindr` for
  Efficient-Net B2,
  pretrained on UPMC and VinDr datasets.
* `concepts`: target label for classification, e.g., `Mass`, `Suspicious Calcification` for ViNDr dataset
* `running-interactive`: running on interactive mode. In this mode,the training will be done using 100 samples for
  sanity check
* `freeze_backbone`: freeze the backbone of the model, for finetune, set to `n`

## Tutorial Notebooks

* For a quick look at setting up the downstream classifier, follow the
  notebook: [Downstream_classifier_tutorial.ipynb](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/notebooks/Tutorial/Downstream_classifier_tutorial.ipynb)
* For a quick look at downloading the image embeddings from the vision encoder of Mammo-CLIP, follow the
  notebook: [Get_Embedding_Vision_encoder_Mammo_CLIP_tutorial.ipynb](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/notebooks/Tutorial/Get_Embedding_Vision_encoder_Mammo_CLIP_tutorial.ipynb)

## Additional scripts

For all the training scripts, we add them in
the [scripts](https://github.com/batmanlab/Mammo-CLIP/tree/main/src/scripts) directory:

| Scripts                                                                                                                              | Purpose                                                           |
|--------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| [pretrain_mammo_clip_b5.sh](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/scripts/pretrain_mammo_clip_b5.sh)                 | Pretrain Mammo-CLIP b5 with image+text data                       |
| [pretrain_mammo_clip_b5_ddp.sh](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/scripts/pretrain_mammo_clip_b5_ddp.sh)         | Pretrain Mammo-CLIP b5 with image+text data using multiple GPUs   |
| [pretrain_mammo_clip_b2.sh](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/scripts/pretrain_mammo_clip_b2.sh)                 | Pretrain Mammo-CLIP b2 with image+text data                       |
| [pretrain_mammo_clip_b2_ddp.sh](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/scripts/pretrain_mammo_clip_b2_ddp.sh)         | Pretrain Mammo-CLIP b2 with image+text data using multiple GPUs   |
| [pretrain_mammo_clip_w_vindr_b5.sh](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/scripts/pretrain_mammo_clip_w_vindr_b5.sh) | Pretrain Mammo-CLIP b5 with image+text data and image+label data  |
| [classifier_fine_tune_b5.sh](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/scripts/classifier_fine_tune_b5.sh)               | Evaluate Mammo-CLIP b5 on fine tuning tasks for classification    |
| [classifier_fine_tune_b2.sh](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/scripts/classifier_fine_tune_b2.sh)               | Evaluate Mammo-CLIP b2 on fine tuning tasks for classification    |
| [classifier_linear_probe_b5.sh](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/scripts/classifier_linear_probe_b5.sh)         | Evaluate Mammo-CLIP b5 on linear probing tasks for classification |
| [classifier_linear_probe_b2.sh](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/scripts/classifier_linear_probe_b2.sh)         | Evaluate Mammo-CLIP b2 on linear probing tasks for classification |
| [detector_fine_tune_b5.sh](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/scripts/detector_fine_tune_b5.sh)                   | Evaluate Mammo-CLIP b5 on fine tuning tasks for detection         |
| [detector_fine_tune_b2.sh](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/scripts/detector_fine_tune_b2.sh)                   | Evaluate Mammo-CLIP b2 on fine tuning tasks for detection         |
| [detector_linear_probe_b5.sh](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/scripts/detector_linear_probe_b5.sh)             | Evaluate Mammo-CLIP b5 on linear probing tasks for detection      |
| [detector_linear_probe_b2.sh](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/scripts/detector_linear_probe_b2.sh)             | Evaluate Mammo-CLIP b2 on linear probing tasks for detection      |

## Mammo-FActOR

For training Mammo-FActOR, refer to the
following [notebook](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/notebooks/Mammo-Factor/Mammo-Factor.ipynb).

## Citation

```bibtex
@InProceedings{10.1007/978-3-031-72390-2_59,
author="Ghosh, Shantanu
and Poynton, Clare B.
and Visweswaran, Shyam
and Batmanghelich, Kayhan",
editor="Linguraru, Marius George
and Dou, Qi
and Feragen, Aasa
and Giannarou, Stamatia
and Glocker, Ben
and Lekadir, Karim
and Schnabel, Julia A.",
title="Mammo-CLIP: A Vision Language Foundation Model to Enhance Data Efficiency and Robustness in Mammography",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="632--642",
abstract="The lack of large and diverse training data on Computer-Aided Diagnosis (CAD) in breast cancer detection has been one of the concerns that impedes the adoption of the system. Recently, pre-training with large-scale image text datasets via Vision-Language models (VLM) (e.g., CLIP) partially addresses the issue of robustness and data efficiency in computer vision (CV). This paper proposes Mammo-CLIP, the first VLM pre-trained on a substantial amount of screening mammogram-report pairs, addressing the challenges of dataset diversity and size. Our experiments on two public datasets demonstrate strong performance in classifying and localizing various mammographic attributes crucial for breast cancer detection, showcasing data efficiency and robustness similar to CLIP in CV. We also propose Mammo-FActOR, a novel feature attribution method, to provide spatial interpretation of representation with sentence-level granularity within mammography reports. Code is available publicly: https://github.com/batmanlab/Mammo-CLIP.",
isbn="978-3-031-72390-2"
}
```

## License and copyright

Licensed under the Creative Commons Attribution 4.0 International

Copyright © [Batman Lab](https://www.batman-lab.com/), 2024

## Contact

For any queries, contact [Shantanu Ghosh](https://shantanu-ai.github.io/) (email: **shawn24@bu.edu**)

## Acknowledgements

Special thanks to Boston University Masters
students [Abhishek Varshney](https://www.linkedin.com/in/abhishek-varshney-a75748159/) & [Akshat Gurbuxani](https://www.linkedin.com/in/akshatgurbuxani/)
for enabling multi-GPU support to Mammo-CLIP.

## Contributing

Did you try Mammo-CLIP on other datasets containing 2D-Mammograms and want to report the results? Feel free to send
a [pull request](https://github.com/shantanu-ai/deep-learning-resources/pulls).
