#!/bin/sh
#SBATCH --output=/ocean/projects/asc170022p/shg121/PhD/Breast-CLIP/src/psc_logs/breast-clip/misc/back_translation_wo_period_%j.out

pwd
hostname
date

CURRENT=$(date +"%Y-%m-%d_%T")
echo $CURRENT

slurm_output_train1=/ocean/projects/asc170022p/shg121/PhD/Breast-CLIP/src/psc_logs/breast-clip/misc/back_translation_wo_period_$CURRENT.out

source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh

conda activate breast_clip_rtx_6000

python /ocean/projects/asc170022p/shg121/PhD/Breast-CLIP/src/codebase/back_translation.py --csv-path "upmc_breast_clip_without_period.csv">$slurm_output_train1

python /ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/codebase/augment_text.py \
  --dataset-path="/ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/codebase/data_csv" \
  --csv-path="upmc_dicom_consolidated_final_folds_BIRADS_num_1_report.csv" \
  --dataset="upmc"