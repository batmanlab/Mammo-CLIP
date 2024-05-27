## Work in progress

## Environment setup

```bash
conda env create --name Mammo-CLIP -f environment.yml
conda activate Mammo-CLIP
```

## Data Instructions

Download the VinDr and RSNA from the links for downstream evaluations:

- [RSNA](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)
- [VinDr](vindr.ai/datasets/mammo)

## Png conversion RSNA

```bash
python /ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/preprocessing/preprocess_image_to_png_kaggle.py \
  --phase="test" \
  --base_folder="/ocean/projects/asc170022p/shg121/PhD/RSNA_Breast_Imaging/Dataset/RSNA_Cancer_Detection"
```

## Png conversion VinDr

```bash
python /ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/preprocessing/preprocess_image_to_png_vindr.py \
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

python /ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/codebase/augment_text.py \
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
./notebooks/preprocess-clip/VinDr.ipynb
```

When you download the VinDr dataset, you will get these two csv
files: [breast-level_annotations.csv](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/data_csv/breast-level_annotations.csv)
and [finding_annotations.csv](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/data_csv/finding_annotations.csv)
. We preprocess the `finding_annotations.csv` file to
get [vindr_detection_v1_folds.csv](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/data_csv/vindr_detection_v1_folds.csv)
. `VinDr.ipynb` notebook requires vindr_detection_v1_folds.csv file as input and
generate [clip_vindr_final.csv](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/data_csv/clip_vindr_final.csv)
file.

## Data preparation for downstream tasks
| Dataset   | CSV                                                                                                                                   |
|-----------|---------------------------------------------------------------------------------------------------------------------------------------|
| **VinDr** | [vindr_detection_v1_folds.csv](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/codebase/data_csv/vindr_detection_v1_folds.csv)  |
| **RSNA**  | [train_folds.csv]()

## Pretrain b5

```bash
python /ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/codebase/train.py --config-name pre_train_b5_clip.yaml
```

## Pretrain lightweight b2

```bash
python /ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/codebase/train.py --config-name pre_train_b2_clip.yaml
```

## Zero-shot evaluation of Mammo-CLIP

```bash
FOLD=0
CKPT="b2-model-best-epoch-10.tar"
DIR="/ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/codebase/outputs/upmc_clip/b2_detector_period_n"
FULL_CKPT="$DIR/checkpoints/fold_$FOLD/$CKPT"

python /ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/codebase/eval_zero_shot_clip.py \
  --config-name zs_clip.yaml hydra.run.dir=$DIR model.clip_check_point=$FULL_CKPT
```

## Linear probe vision encoder Mammo-CLIP on target classification task

```bash
python /ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/codebase/train_classifier.py \
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

## Finetune vision encoder Mammo-CLIP on target classification task

```bash
python /ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/codebase/train_classifier.py \
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

## Linear probe vision encoder Mammo-CLIP on target classification task

```bash
python /ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/codebase/train_detector.py \
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

## Finetune vision encoder Mammo-CLIP on target classification task

```bash
python /ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/codebase/train_detector.py \
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

## Checkpoint

[B5](https://drive.google.com/file/d/1c14IwqxkMRFD78BEhNA17n3b6C21fuQ1/view?usp=sharing)

[B2](https://drive.google.com/file/d/1dNqicN0_Oeo4T4920eljxDX0x0htFgAc/view?usp=sharing)

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

Licensed under the [MIT License](LICENSE)

Copyright © [Batman Lab](https://www.batman-lab.com/), 2024

## Contact

For any queries, contact: **shawn24@bu.edu**