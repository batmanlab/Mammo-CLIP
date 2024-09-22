#!/bin/bash -l

#$ -N b5_det_upmc_ddp        # Give job a name
#$ -o /restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/scc_logs/pretrain/fold0_b5_det_$JOB_ID_$JOB_NAME.out       # File name for the stdout output of the job.

#$ -P batmanlab     # Specify the SCC project name you want to use
#$ -l h_rt=48:00:00  # Specify the hard time limit for the job
#$ -pe omp 16        # Number of cores
#$ -l gpus=4        # Number of GPUs
#$ -l gpu_c=8.0     # GPU Compute capacity

#$ -j y            # Merge the error and output streams into a single file
#$ -m bea          # The batch system sends an email to you. The possible values are – when the job begins (b), ends (e), is aborted (a), is suspended (s),
                   # or never (n) – default.

pwd
hostname
date

CURRENT=$(date +"%Y-%m-%d_%T")
slurm_output_train1=/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/scc_logs/pretrain/fold0_b5_det_$CURRENT_$JOB_NAME.out

echo $CURRENT
echo  Mammo-clip

# Load your environment and run the job
module load miniconda/23.1.0
module load python3/3.8
conda activate breast_clip_rtx_6000

torchrun --nproc_per_node=4 /restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/codebase/train.py \
 --config-name pre_train_b5_clip.yaml >$slurm_output_train1






