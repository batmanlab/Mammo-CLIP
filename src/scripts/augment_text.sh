#!/bin/sh
#SBATCH --output=/ocean/projects/asc170022p/shg121/PhD/Breast-CLIP/src/psc_logs/breast-clip/misc/back_translation_wo_period_%j.out

pwd
hostname
date

CURRENT=$(date +"%Y-%m-%d_%T")
FOLD=0
CKPT="model-epoch-6.tar"
CKPT_PATH="/ocean/projects/asc170022p/shg121/PhD/Breast-CLIP/src/codebase/outputs/upmc_rsna_w_upmc_concepts_vindr_w_upmc_concepts/swin/checkpoints/fold_$FOLD/$CKPT"
MODEL_TYPE="swin"

echo $CURRENT
echo $FOLD
echo $CKPT
echo $CKPT_PATH
echo $MODEL_TYPE

slurm_output_train1=/ocean/projects/asc170022p/shg121/PhD/Breast-CLIP/src/psc_logs/breast-clip/misc/back_translation_wo_period_$CURRENT.out

echo "Swin transformer breast-clip"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh

conda activate breast_clip_rtx_6000

#python /ocean/projects/asc170022p/shg121/PhD/Breast-CLIP/src/codebase/augment_text.py --col_name "FINDINGS">$slurm_output_train1
#python /ocean/projects/asc170022p/shg121/PhD/Breast-CLIP/src/codebase/back_translation.py --csv-path "upmc_breast_clip_without_period.csv">$slurm_output_train1

python /ocean/projects/asc170022p/shg121/PhD/Breast-CLIP/src/codebase/back_translation.py --csv-path "upmc_breast_clip_without_period.csv">$slurm_output_train1

python /ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/codebase/augment_text.py \
  --dataset-path="/ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/codebase/data_csv" \
  --csv-path="upmc_dicom_consolidated_final_folds_BIRADS_num_1_report.csv" \
  --dataset="upmc"