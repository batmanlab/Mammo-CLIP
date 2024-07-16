#!/bin/sh
#SBATCH --output=/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/psc_logs/b2_det_ft_%j.out

pwd
hostname
date

CURRENT=$(date +"%Y-%m-%d_%T")
echo $CURRENT

slurm_output_train_mass=/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/psc_logs/clip_train/b2_det_ft_mass_$CURRENT.out
slurm_output_train_calc=/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/psc_logs/clip_train/b2_det_ft_calc_$CURRENT.out

echo "Mammo-clip b2"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh

conda activate breast_clip_rtx_6000

# Mass
python ./src/codebase/train_detector.py \
  --img-dir 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images_png' \
  --csv-file 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection_v1_folds.csv' \
  --clip_chk_pt_path "/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/codebase/outputs/upmc_clip/b2_detector_period_n/checkpoints/fold_0/b2-model-best-epoch-7.tar" \
  --dataset 'ViNDr' \
  --arch 'clip_b2_upmc' \
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
  --score-threshold 0.2 >$slurm_output_train_mass


# Suspicious Calcification
python ./src/codebase/train_detector.py \
  --img-dir 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images_png' \
  --csv-file 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection_v1_folds.csv' \
  --clip_chk_pt_path "/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/codebase/outputs/upmc_clip/b2_detector_period_n/checkpoints/fold_0/b2-model-best-epoch-7.tar" \
  --dataset 'ViNDr' \
  --arch 'clip_b2_upmc' \
  --epochs 120 \
  --batch-size 7 \
  --freeze_backbone "n" \
  --data_frac 1.0 \
  --concepts 'Suspicious Calcification' \
  --print-freq 5000 \
  --log-freq 300 \
  --running-interactive 'n' \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --score-threshold 0.2 >$slurm_output_train_calc
