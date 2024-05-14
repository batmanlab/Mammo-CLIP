## To be updated soon


## Data Instructions

Download the VinDr and RSNA from the links for downstram evaluations:

- [RSNA](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)
- [VinDr](vindr.ai/datasets/mammo)


### Datasets

| Dataset | Model | CSV Path |
|---------|-------|----------|
| RSNA w/ (UPMC + VinDr) concepts | For classification breast cancer in RSNA and measuring the bias | `/ocean/projects/asc170022p/shg121/PhD/RSNA_Breast_Imaging/Dataset/RSNA_Cancer_Detection/rsna_w_upmc_concepts_breast_clip.csv` |
| RSNA | For classification using clip and zs. Same as row 1 w/o the concept annotation | `/ocean/projects/asc170022p/shared/Projects/breast-imaging/RSNA_Breast_Imaging/Dataset/RSNA_Cancer_Detection/train_folds_birads.csv` |
| VinDr | For classification and object detection of the concepts | `/ocean/projects/asc170022p/shared/Projects/breast-imaging/RSNA_Breast_Imaging/Dataset/External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection_v1_folds.csv` |
| UPMC | For classification of clip/mole/mark/scar | `/ocean/projects/asc170022p/shared/Projects/breast-imaging/RSNA_Breast_Imaging/Dataset/External/UPMC/upmc_dicom_consolidated_final_folds_clip_mark_mole_scar_v1.csv` |
| UPMC | Pre-training clip | `/ocean/projects/asc170022p/shg121/PhD/RSNA_Breast_Imaging/Dataset/External/UPMC/upmc_breast_clip_without_period_lower_case.csv` |
| VinDr | Pre-training clip | `/ocean/projects/asc170022p/shg121/PhD/RSNA_Breast_Imaging/Dataset/` |
