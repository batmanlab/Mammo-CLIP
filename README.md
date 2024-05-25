## To be updated soon


## Data Instructions

Download the VinDr and RSNA from the links for downstram evaluations:

- [RSNA](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)
- [VinDr](vindr.ai/datasets/mammo)


## Png conversion Kaggle RSNA
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

## Back translation image-text dataset (UPMC)
#### Data annonymized but have similar format
#### input: upmc_dicom_consolidated_final_folds_BIRADS_num_1_report.csv
#### output: upmc_breast_clip_without_period_lower_case.csv
```bash
# input: upmc_dicom_consolidated_final_folds_BIRADS_num_1_report.csv
# output: upmc_breast_clip_without_period_lower_case.csv

python /ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/codebase/augment_text.py \
  --dataset-path="/ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/codebase/data_csv" \
  --csv-path="upmc_dicom_consolidated_final_folds_BIRADS_num_1_report.csv" \
  --dataset="upmc" 
```