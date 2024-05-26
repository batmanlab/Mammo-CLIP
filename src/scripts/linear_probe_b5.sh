# Mass
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


# Suspicious_Calcification
python /ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/codebase/train_classifier.py \
  --img-dir 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images_png' \
  --csv-file 'External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection_v1_folds.csv' \
  --clip_chk_pt_path "/ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/codebase/outputs/upmc_clip/b5_detector_period_n/checkpoints/fold_0/b5-model-best-epoch-7.tar" \
  --data_frac 1.0 \
  --dataset 'ViNDr' \
  --arch 'upmc_breast_clip_det_b5_period_n_lp' \
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
  --balanced-dataloader 'n'
















