import collections
import cv2
import gc
import os
import pandas as pd
import pickle
import random
import time
import torch
import warnings
from pathlib import Path
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from Datasets.dataset_utils import get_dataset, get_transforms
from Detectors.detectors_utils import _get_detections, _get_annotations, compute_overlap, _compute_ap
from Detectors.retinanet.detector_model import RetinaNet_efficientnet
from utils import seed_all, timeSince

warnings.filterwarnings("ignore")
import numpy as np


def do_experiements(args, device):
    args.data_dir = Path(args.data_dir)
    args.df = pd.read_csv(args.data_dir / args.csv_file)
    print(args.data_dir / args.csv_file)
    args.df = args.df.fillna(0)
    print(f"df shape: {args.df.shape}")

    if args.dataset.lower() == 'vindr':
        args.df = args.df.head(2254)
    args.model_base_name = args.arch
    seed_all(args.seed)
    args.cur_fold = 0
    args.train_folds = args.df[args.df['split'] == "training"].reset_index(drop=True)
    args.valid_folds = args.df[args.df['split'] == "test"].reset_index(drop=True)
    print(f'train: {args.train_folds.shape}', f'valid: {args.valid_folds.shape}')
    print(args.valid_folds.columns)
    if args.running_interactive:
        # test on small subsets of data on interactive mode
        args.train_folds = args.train_folds.head(100)
        args.valid_folds = args.valid_folds.head(n=1000)
        # print(args.valid_folds[args.valid_folds["Architectural_Distortion"] == 1].shape)

    train_loop(args, device)

    print(f'Checkpoints saved at: {args.chk_pt_path}')
    print(f'Outputs saved at: {args.output_path}')


def train_loop(args, device):
    print(f'\n================== fold: {args.cur_fold} training ======================')
    if args.data_frac < 1.0:
        args.train_folds = args.train_folds.sample(frac=args.data_frac, random_state=1, ignore_index=True)
    train_loader, valid_loader, valid_dataset = get_dataset(args)
    print(f'train_loader: {len(train_loader)}', f'valid_loader: {len(valid_dataset)}')

    # Create the model
    retinanet = RetinaNet_efficientnet(
        num_classes=len(args.concepts), model_type=args.arch, focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma, clip_chk_pt=args.clip_chk_pt_path, freeze_backbone=args.freeze_backbone,
    )

    print(retinanet)
    retinanet.to(device)
    retinanet.training = True

    optimizer = Adam(retinanet.parameters(), lr=args.lr)
    logger = SummaryWriter(args.tb_logs_path / f'fold{args.cur_fold}')

    best_MAP = 0.

    loss_hist = collections.deque(maxlen=500)
    cls_loss_hist = collections.deque(maxlen=500)
    reg_loss_hist = collections.deque(maxlen=500)
    retinanet.train()
    # retinanet.module.freeze_bn()

    for epoch_num in range(args.epochs):
        start_time = time.time()
        retinanet.train()
        # retinanet.module.freeze_bn()

        start = time.time()
        epoch_loss = []
        epoch_cls_loss = []
        epoch_reg_loss = []
        for iter_num, data in enumerate(train_loader):
            try:
                optimizer.zero_grad()
                img = data["image"].to(device)
                bbox = data["res_bbox_tensor"].to(device)
                batch_size = img.size(0)
                classification_loss, regression_loss = retinanet([img, bbox])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss
                if bool(loss == 0):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()
                loss_hist.append(float(loss))
                cls_loss_hist.append(float(classification_loss))
                reg_loss_hist.append(float(regression_loss))

                epoch_loss.append(float(loss))
                epoch_cls_loss.append(float(classification_loss))
                epoch_reg_loss.append(float(regression_loss))
                if iter_num % args.print_freq == 0 or iter_num == (len(train_loader) - 1):
                    print('Epoch: [{0}][{1}/{2}] '
                          'Elapsed {remain:s} '
                          'Classification loss: {cls_loss:1.5f} '
                          'Regression loss: {reg_loss:1.5f} '
                          'Running loss: {run_loss:1.5f} '.format(
                        epoch_num + 1, iter_num, len(train_loader),
                        remain=timeSince(start, float(iter_num + 1) / len(train_loader)),
                        cls_loss=float(classification_loss),
                        reg_loss=float(regression_loss),
                        run_loss=np.mean(loss_hist)
                    ))

                if iter_num % args.log_freq == 0 or iter_num == (len(train_loader) - 1):
                    index = iter_num + len(train_loader) * epoch_num
                    logger.add_scalar('train/iter_loss', np.mean(loss_hist), index)
                    logger.add_scalar('train/iter_cls_loss', np.mean(cls_loss_hist), index)
                    logger.add_scalar('train/iter_reg_loss', np.mean(reg_loss_hist), index)

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        _, MAP = evaluate(valid_dataset, logger=logger, epoch_num=epoch_num, concepts=args.concepts,
                          retinanet=retinanet, score_threshold=args.score_threshold)
        logger.add_scalar('train/epoch_loss', np.mean(np.array(epoch_loss)), epoch_num)
        logger.add_scalar('train/epoch_cls_loss', np.mean(np.array(epoch_cls_loss)), epoch_num)
        logger.add_scalar('train/epoch_reg_loss', np.mean(np.array(epoch_reg_loss)), epoch_num)
        logger.add_scalar('valid/MAP', MAP, epoch_num)

        elapsed = time.time() - start_time
        print(
            f'Epoch {epoch_num + 1} - avg_train_loss: {np.mean(np.array(epoch_loss)):.4f},  '
            f'time: {elapsed:.0f}s, MAP: {MAP:.4f}'
        )

        torch.save(
            {
                'state_dict': retinanet.state_dict(),
                'MAP': MAP
            },
            args.chk_pt_path / f'{args.model_base_name}_seed_{args.seed}_fold_{args.cur_fold}_epoch_{epoch_num}.pth'
        )

        if best_MAP < MAP:
            best_MAP = MAP
            print(f'Epoch {epoch_num + 1} - Save Best aucroc: {best_MAP:.4f} Model')
            torch.save(
                {
                    'state_dict': retinanet.state_dict(),
                    'best_MAP': best_MAP
                },
                args.chk_pt_path / f'{args.model_base_name}_seed_{args.seed}_'
                                   f'fold_{args.cur_fold}_best_auroc.pth'
            )
    print(f'[Fold{args.cur_fold}] Best MAP: {best_MAP:.4f}')

    torch.cuda.empty_cache()
    gc.collect()


def evaluate(val_dataset, concepts, logger, epoch_num, retinanet, iou_threshold=0.5, score_threshold=0.05,
             max_detections=100):
    """Evaluate a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet           : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a
            detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """

    # gather all detections and annotations

    num_classes = len(concepts)
    all_detections = _get_detections(
        val_dataset,
        retinanet,
        num_classes=num_classes,
        score_threshold=score_threshold,
        max_detections=max_detections,
    )
    all_annotations = _get_annotations(val_dataset)

    average_precisions = {}
    for label in range(num_classes):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(val_dataset)):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if (
                        max_overlap >= iou_threshold
                        and assigned_annotation not in detected_annotations
                ):
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(
            true_positives + false_positives, np.finfo(np.float64).eps
        )

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    print("\nmAP:")
    MAP = 0
    mAPs = []
    _range = range(1, num_classes) if 'No Finding' in concepts else range(0, num_classes)
    for label in _range:
        label_name = concepts[label]
        MAP = average_precisions[label][0]
        mAPs.append(MAP)
        logger.add_scalar(f'valid/{label_name}_mAP', MAP, epoch_num)
        print("{}: {}".format(label_name, average_precisions[label][0]))

    return average_precisions, np.mean(np.array(mAPs))
