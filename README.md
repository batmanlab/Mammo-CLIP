# Mammo-CLIP: A Vision Language Foundation Model to Enhance Data Efficiency and Robustness in Mammography

### [[Paper](https://arxiv.org/pdf/2405.12255)] [[Hugging Face](https://huggingface.co/shawn24/Mammo-CLIP/)] [[Pre-training Checkpoints](https://github.com/batmanlab/Mammo-CLIP/blob/main/README.md#mammo-clip-checkpoints)] [[VinDr png data](https://www.kaggle.com/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png)]

[Shantanu Ghosh<sup>1</sup>](https://shantanu-ai.github.io/)
, [Clare B. Poynton<sup>2</sup>](https://www.bumc.bu.edu/camed/profile/clare-poynton/)
, [Shyam Visweswaran<sup>3</sup>](https://www.thevislab.com/lab/doku.php),
[Kayhan Batmanghelich<sup>1</sup>](https://www.batman-lab.com/)
<br/>
<sup>1</sup>BU ECE, <sup>2</sup> BUMC, <sup>3</sup> Pitt DBMI <br/>

## FAQ

After going through the instruction, it is recommended to visit
this [link](https://github.com/batmanlab/Mammo-CLIP/issues/2) for any further clarification on pretraining. If we hear
more queries, we may add a separate FAQs in the future.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Data Download](#data-download)
3. [Pre-processing Images](#pre-processing-images)
4. [Data Preparation for Pretraining](#data-preparation-for-pretraining)
5. [Data Preparation for Downstream Evaluation Tasks](#data-preparation-for-downstream-evaluation-tasks)
6. [Mammo-CLIP checkpoints](#mammo-clip-checkpoints)
7. [Pretraining Mammo-CLIP](#pretraining-mammo-clip)
8. [Creating classifiers and detectors](#creating-classifiers-and-detectors)
9. [Evaluation](#evaluation)
10. [Additional Scripts](#additional-scripts)
11. [Citation](#citation)
12. [License and Copyright](#license-and-copyright)
13. [Contact](#contact)

## Environment Setup

Use [environment.yml](https://github.com/batmanlab/Mammo-CLIP/blob/main/environment.yml) to setup the environment.

```bash
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
- [VinDr](vindr.ai/datasets/mammo)

For the PNG images converted from the original Dicom images, as mentioned in the preprocessing steps in the paper, refer to the following links:
  - RSNA (WIP)
  - [VinDr](https://www.kaggle.com/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png)

To preprocess the dicom images directly, follow the instructions in the next section. If you downloaded the PNG images, skip the preprocessing steps.

## Pre-processing images

### Convert to png: RSNA

```bash
python ./src/preprocessing/preprocess_image_to_png_kaggle.py \
  --phase="test" \
  --base_folder="/ocean/projects/asc170022p/shg121/PhD/RSNA_Breast_Imaging/Dataset/RSNA_Cancer_Detection"
```

### convert to png: VinDr

```bash
python ./src/preprocessing/preprocess_image_to_png_vindr.py \
  --phase="test" \
  --base_folder="/ocean/projects/asc170022p/shg121/PhD/RSNA_Breast_Imaging/Dataset/External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
```

## Data preparation for pretraining

### Image-text dataset

1. Our image-text dataset is an in-house dataset from UPMC. The sample
   csv: [upmc_dicom_consolidated_final_folds_BIRADS_num_1_report.csv](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/data_csv/upmc_dicom_consolidated_final_folds_BIRADS_num_1_report.csv)

2. Note the `HISTORY`, `FINDINGS`, and `IMPRESSION` columns in the csv file. The `FINDINGS` and `IMPRESSION` columns are
   used to generate the text for the image. The `HISTORY`, `FINDINGS` and `IMPRESSION` columns contains templated text
   due to privacy.
3. Next run the following command to augment the text with `upmc_dicom_consolidated_final_folds_BIRADS_num_1_report.csv`
   file:

```bash
# input: upmc_dicom_consolidated_final_folds_BIRADS_num_1_report.csv
# output: clip_pretrain_100.csv

python ./src/codebase/augment_text.py \
  --dataset-path="/ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/codebase/data_csv" \
  --csv-path="upmc_dicom_consolidated_final_folds_BIRADS_num_1_report.csv" \
  --dataset="upmc" 
```

4. The csv file of the final image-text dataset should have the following format:

| index | patient_id | laterality              | image                                                   | view                                                                         | CC                                                              | MLO                                                              | text                            | text_augment                                       |
|-------|------------|-------------------------|---------------------------------------------------------|------------------------------------------------------------------------------|-----------------------------------------------------------------|------------------------------------------------------------------|---------------------------------|----------------------------------------------------|
| 0     | patient_id | laterality ('R' or 'L') | List of all image_paths for patient_id-laterality combo | List of views for patient_id-laterality combo (only 'CC' and 'MLO' are used) | List of image paths for CC view for patient_id-laterality combo | List of image paths for MLO view for patient_id-laterality combo | List of [findings, impression]  | List of [augmented findings, augmented impression] |

5. The final sample csv file as the output of `step3` is
   here: [clip_pretrain_100.csv](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/data_csv/clip_pretrain_100.csv)

### Image-label dataset

We use VinDr dataset as image-label dataset. So if you are planning to use it in the pre-training setup, use the
following notebook to preprocess the VinDr dataset:

```bash
.src/codebase/notebooks/preprocess-clip/VinDr.ipynb
```

When you download the VinDr dataset, you will get these two csv
files: [breast-level_annotations.csv](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/data_csv/breast-level_annotations.csv)
and [finding_annotations.csv](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/data_csv/finding_annotations.csv)
. We preprocess the `finding_annotations.csv` file to
get [vindr_detection_v1_folds.csv](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/data_csv/vindr_detection_v1_folds.csv)
. `VinDr.ipynb` notebook requires vindr_detection_v1_folds.csv file as input and
generate [clip_vindr_final.csv](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/data_csv/clip_vindr_final.csv)
file.

## Data preparation for downstream evaluation tasks

Use the following csv files as metadata for the downstream tasks (classification, detection, zero-shot):

| Dataset | CSV                                                                                                                                  |
|---------|--------------------------------------------------------------------------------------------------------------------------------------|
| VinDr   | [vindr_detection_v1_folds.csv](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/data_csv/vindr_detection_v1_folds.csv) |
| RSNA    | [train_folds.csv](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/data_csv/train_folds.csv)                           | 

For detection/localization tasks, we have included the coordinates of the resized bounding boxes of VinDr in the above csv file. Somebody interested in resizing the bounding boxes by themselves, refer to this [code](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/preprocessing/preprocess_VinDr_detector.py).

## Mammo-CLIP checkpoints
Following are the pre-training checkpoints of Mammo-CLIP:
| Model architecture | Checkpoints (Google drive)                                                                             | Checkpoints (Hugging Face)                                                                                                  |
|--------------------|--------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Best performance   | [Efficient-Net B5](https://drive.google.com/file/d/1c14IwqxkMRFD78BEhNA17n3b6C21fuQ1/view?usp=sharing) | [Efficient-Net B5](https://huggingface.co/shawn24/Mammo-CLIP/blob/main/Pre-trained-checkpoints/b5-model-best-epoch-7.tar)   |
| Lightweight        | [Efficient-Net B2](https://drive.google.com/file/d/1dNqicN0_Oeo4T4920eljxDX0x0htFgAc/view?usp=sharing) | [Efficient-Net B2](https://huggingface.co/shawn24/Mammo-CLIP/blob/main/Pre-trained-checkpoints/b2-model-best-epoch-10.tar)  |

We have also uploaded the downstream checkpoints for classification and localization (both linear probe and finetuning) with the image encoder of Efficient-Net B5 Mammo-CLIP for fold 0 [here](https://huggingface.co/shawn24/Mammo-CLIP/tree/main/Downstream-checkpoints).

## Warning

Look for `/ocean/projects/asc170022p/shg121/PhD` and replace it with your own path.

## Pretraining Mammo-CLIP

```bash
python ./Mammo-CLIP/src/codebase/train.py --config-name pre_train_b5_clip.yaml
```

* Use
  [pre_train_b5_clip.yaml](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/configs/pre_train_b5_clip.yaml)
  for pre-training image-text variant of Efficient-Net B5 Mammo-CLIP
* Use
  [pre_train_b2_clip.yaml](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/configs/pre_train_b2_clip.yaml)
  for pre-training image-text variant of Efficient-Net B2 Mammo-CLIP

## Creating classifiers and detectors

* For creating classifiers for downstream evaluations using the image encoder of Mammo-CLIP, use the class `BreastClipClassifier`
  in [breast-clip-classifier.py](https://github.com/batmanlab/Mammo-CLIP/blob/c9cc232368eaf0a6d55f1bea04490d9136362466/src/codebase/Classifiers/models/breast_clip_classifier.py#L6)
  file.
* For creating detectors for downstream evaluations using the image encoder of Mammo-CLIP, use the function `RetinaNet_efficientnet`
  in [detector_mode.py](https://github.com/batmanlab/Mammo-CLIP/blob/c9cc232368eaf0a6d55f1bea04490d9136362466/src/codebase/Detectors/retinanet/detector_model.py#L357)
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
  --data-dir '/ocean/projects/asc170022p/shg121/PhD/RSNA_Breast_Imaging/Dataset' \
  --img-dir 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images_png' \
  --csv-file 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection_v1_folds.csv' \
  --clip_chk_pt_path "/ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/codebase/outputs/upmc_clip/b5_detector_period_n/checkpoints/fold_0/b5-model-best-epoch-7.tar" \
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
* `arch`: architecture of the model, e.g., `upmc_breast_clip_det_b5_period_n_lp` for B5
  or `upmc_breast_clip_det_b2_period_n_lp` for B2
* `label`: target label for classification, e.g., `Mass`, `Suspicious_Calcification`or `density` for ViNDr
  dataset; `cancer` for RSNA dataset
* `running-interactive`: running on interactive mode. In this mode,the training will be done using 100 samples for
  sanity check

### Finetune vision encoder Mammo-CLIP on target classification task

```bash
python ./src/codebase/train_classifier.py \
  --data-dir '/ocean/projects/asc170022p/shg121/PhD/RSNA_Breast_Imaging/Dataset' \
  --img-dir 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images_png' \
  --csv-file 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection_v1_folds.csv' \
  --clip_chk_pt_path "/ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/codebase/outputs/upmc_clip/b5_detector_period_n/checkpoints/fold_0/b5-model-best-epoch-7.tar" \
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
* `arch`: architecture of the model, e.g., `upmc_breast_clip_det_b5_period_n_ft` for B5
  or `upmc_breast_clip_det_b2_period_n_ft` for B2
* `label`: target label for classification, e.g., `Mass`, `Suspicious_Calcification`or `density` for ViNDr
  dataset; `cancer` for RSNA dataset
* `running-interactive`: running on interactive mode. In this mode,the training will be done using 100 samples for
  sanity check

### Linear probe vision encoder Mammo-CLIP on target detection task

```bash
python ./src/codebase/train_detector.py \
  --data-dir '/ocean/projects/asc170022p/shg121/PhD/RSNA_Breast_Imaging/Dataset' \
  --img-dir 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images_png' \
  --csv-file 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection_v1_folds.csv' \
  --clip_chk_pt_path "/ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/codebase/outputs/upmc_clip/b5_detector_period_n/checkpoints/fold_0/b5-model-best-epoch-7.tar" \
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
* `arch`: architecture of the model, e.g., `clip_b5_upmc` for B5 or `clip_b2_upmc` for B2
* `concepts`: target label for classification, e.g., `Mass`, `Suspicious Calcification` for ViNDr dataset
* `running-interactive`: running on interactive mode. In this mode,the training will be done using 100 samples for
  sanity check
* `freeze_backbone`: freeze the backbone of the model, for linear probe, set to `y`

### Finetune vision encoder Mammo-CLIP on target detection task

```bash
python ./src/codebase/train_detector.py \
  --data-dir '/ocean/projects/asc170022p/shg121/PhD/RSNA_Breast_Imaging/Dataset' \
  --img-dir 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images_png' \
  --csv-file 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection_v1_folds.csv' \
  --clip_chk_pt_path "/ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/codebase/outputs/upmc_clip/b5_detector_period_n/checkpoints/fold_0/b5-model-best-epoch-7.tar" \
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
* `arch`: architecture of the model, e.g., `clip_b5_upmc` for B5 or `clip_b2_upmc` for B2
* `concepts`: target label for classification, e.g., `Mass`, `Suspicious Calcification` for ViNDr dataset
* `running-interactive`: running on interactive mode. In this mode,the training will be done using 100 samples for
  sanity check
* `freeze_backbone`: freeze the backbone of the model, for finetune, set to `n`

## Additional scripts

For all the training scripts, we add them in
the [scripts](https://github.com/batmanlab/Mammo-CLIP/tree/main/src/scripts) directory:

| Scripts                                                                                                                      | Purpose                                                           |
|------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| [pretrain_mammo_clip_b5.sh](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/scripts/pretrain_mammo_clip_b5.sh)         | Pretrain Mammo-CLIP b5                                            |
| [pretrain_mammo_clip_b2.sh](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/scripts/pretrain_mammo_clip_b2.sh)         | Pretrain Mammo-CLIP b2                                            |
| [classifier_fine_tune_b5.sh](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/scripts/classifier_fine_tune_b5.sh)       | Evaluate Mammo-CLIP b5 on fine tuning tasks for classification    |
| [classifier_fine_tune_b2.sh](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/scripts/classifier_fine_tune_b2.sh)       | Evaluate Mammo-CLIP b2 on fine tuning tasks for classification    |
| [classifier_linear_probe_b5.sh](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/scripts/classifier_linear_probe_b5.sh) | Evaluate Mammo-CLIP b5 on linear probing tasks for classification |
| [classifier_linear_probe_b2.sh](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/scripts/classifier_linear_probe_b2.sh) | Evaluate Mammo-CLIP b2 on linear probing tasks for classification |
| [detector_fine_tune_b5.sh](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/scripts/detector_fine_tune_b5.sh)           | Evaluate Mammo-CLIP b5 on fine tuning tasks for detection         |
| [detector_fine_tune_b2.sh](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/scripts/detector_fine_tune_b2.sh)           | Evaluate Mammo-CLIP b2 on fine tuning tasks for detection         |
| [detector_linear_probe_b5.sh](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/scripts/detector_linear_probe_b5.sh)     | Evaluate Mammo-CLIP b5 on linear probing tasks for detection      |
| [detector_linear_probe_b2.sh](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/scripts/detector_linear_probe_b2.sh)     | Evaluate Mammo-CLIP b2 on linear probing tasks for detection      |

## Citation

```bibtex
@article{ghosh2024mammo,
  title={Mammo-CLIP: A Vision Language Foundation Model to Enhance Data Efficiency and Robustness in Mammography},
  author={Ghosh, Shantanu and Poynton, Clare B and Visweswaran, Shyam and Batmanghelich, Kayhan},
  journal={arXiv preprint arXiv:2405.12255},
  year={2024}
}
```

## License and copyright

Licensed under the Creative Commons Attribution 4.0 International

Copyright © [Batman Lab](https://www.batman-lab.com/), 2024

## Contact

For any queries, contact: **shawn24@bu.edu**
