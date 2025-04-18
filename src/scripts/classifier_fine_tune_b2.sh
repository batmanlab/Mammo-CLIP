#!/bin/sh
#SBATCH --output=/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/psc_logs/b2_cls_ft_%j.out

pwd
hostname
date

CURRENT=$(date +"%Y-%m-%d_%T")
echo $CURRENT

slurm_output_train_mass=/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/psc_logs/b2_cls_ft_mass_$CURRENT.out
slurm_output_train_calc=/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/psc_logs/b2_cls_ft_calc_$CURRENT.out
slurm_output_train_density=/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/psc_logs/b2_cls_ft_density_$CURRENT.out
slurm_output_train_cancer=/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/psc_logs/b2_cls_ft_cancer_$CURRENT.out

echo "Mammo-clip b2"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh

conda activate breast_clip_rtx_6000

# Mass (VinDr)
python ./src/codebase/train_classifier.py \
  --img-dir 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images_png' \
  --csv-file 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection_v1_folds.csv' \
  --clip_chk_pt_path "/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/codebase/outputs/upmc_clip/b2_detector_period_n/checkpoints/fold_0/b2-model-best-epoch-10.tar" \
  --data_frac 1.0 \
  --dataset 'ViNDr' \
  --arch 'upmc_breast_clip_det_b2_period_n_ft' \
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
  --balanced-dataloader 'n' >$slurm_output_train_mass


# Suspicious_Calcification (VinDr)
python ./src/codebase/train_classifier.py \
  --img-dir 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images_png' \
  --csv-file 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection_v1_folds.csv' \
  --clip_chk_pt_path "/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/codebase/outputs/upmc_clip/b2_detector_period_n/checkpoints/fold_0/b2-model-best-epoch-10.tar" \
  --data_frac 1.0 \
  --dataset 'ViNDr' \
  --arch 'upmc_breast_clip_det_b2_period_n_ft' \
  --label "Suspicious_Calcification" \
  --epochs 30 \
  --batch-size 8 \
  --num-workers 0 \
  --print-freq 10000 \
  --log-freq 500 \
  --running-interactive 'n' \
  --n_folds 1 \
  --lr 5.0e-5 \
  --weighted-BCE 'y' \
  --balanced-dataloader 'n'>$slurm_output_train_calc


# Density (VinDr)
python ./src/codebase/train_classifier.py \
  --img-dir 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images_png' \
  --csv-file 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection_v1_folds.csv' \
  --clip_chk_pt_path "/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/codebase/outputs/upmc_clip/b2_detector_period_n/checkpoints/fold_0/b2-model-best-epoch-10.tar" \
  --data_frac 1.0 \
  --dataset 'ViNDr' \
  --arch 'upmc_breast_clip_det_b2_period_n_ft' \
  --label "density" \
  --epochs 30 \
  --batch-size 8 \
  --num-workers 0 \
  --print-freq 10000 \
  --log-freq 500 \
  --running-interactive 'n' \
  --n_folds 1 \
  --lr 5.0e-5 \
  --weighted-BCE 'y' \
  --balanced-dataloader 'n'>$slurm_output_train_density


# Cancer (RSNA)
python ./src/codebase/train_classifier.py \
  --img-dir 'RSNA_Cancer_Detection/train_images_png' \
  --csv-file 'RSNA_Cancer_Detection/train_folds.csv' \
  --clip_chk_pt_path "/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/codebase/outputs/upmc_clip/b2_detector_period_n/checkpoints/fold_0/b2-model-best-epoch-10.tar" \
  --dataset 'RSNA' \
  --data_frac 1.0 \
  --label "cancer" \
  --n_folds 1 \
  --lr 5e-5 \
  --weight-decay 1e-4 \
  --warmup-epochs 1 \
  --arch 'upmc_breast_clip_det_b2_period_n_ft' \
  --epochs 20 \
  --batch-size 6 \
  --num-workers 0 \
  --print-freq 10000 \
  --log-freq 500 \
  --running-interactive 'n' \
  --lr 5.0e-5 \
  --weighted-BCE 'y' \
  --balanced-dataloader 'n'>$slurm_output_train_cancer


python ./src/codebase/train_classifier.py \
  --img-dir 'RSNA_Cancer_Detection/train_images_png' \
  --csv-file 'RSNA_Cancer_Detection/train_folds.csv' \
  --clip_chk_pt_path "/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/codebase/outputs/upmc_clip/b2_detector_period_n/checkpoints/fold_0/b2-model-best-epoch-10.tar" \
  --dataset 'RSNA' \
  --data_frac 1.0 \
  --label "cancer" \
  --n_folds 1 \
  --lr 5e-5 \
  --weight-decay 1e-4 \
  --warmup-epochs 1 \
  --arch 'upmc_breast_clip_det_b2_period_n_ft' \
  --epochs 20 \
  --batch-size 6 \
  --num-workers 0 \
  --print-freq 10000 \
  --log-freq 500 \
  --running-interactive 'y' \
  --lr 5.0e-5 \
  --weighted-BCE 'y' \
  --balanced-dataloader 'n'








