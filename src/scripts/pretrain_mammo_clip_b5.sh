#!/bin/sh
#SBATCH --output=/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/psc_logs/clip_train/b5_det_%j.out

pwd
hostname
date

CURRENT=$(date +"%Y-%m-%d_%T")
echo $CURRENT

slurm_output_train1=/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/psc_logs/clip_train/b5_det_$CURRENT.out

echo "Pretrain Mammo-clip b5"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh

conda activate breast_clip_rtx_6000

python ./src/codebase/train.py --config-name pre_train_b5_clip.yaml >$slurm_output_train1
