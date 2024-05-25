import logging
import os
import shutil
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from . import util
from .data import DataModule
from .loss import build_loss
from .model import build_model

log = logging.getLogger(__name__)


def run_validation(local_rank, cfg: Dict):
    if "tokenizer" in cfg:
        assert (
                cfg["tokenizer"]["pretrained_model_name_or_path"] == cfg["model"]["text_encoder"]["name"]
        ), "tokenizer should be same to text_encoder"

    distributed = local_rank != -1
    log.info(f"local_rank: {local_rank}")
    log.info(f"distributed: {distributed}")

    if distributed:
        ngpus_per_node = torch.cuda.device_count()
        print(f"ngpus_per_node: {ngpus_per_node}")
        dist.init_process_group(backend="nccl")
        print(f"local_rank: {local_rank}")
        print(f"device_count: {torch.cuda.device_count()}")
        # Check if local_rank is within the valid range of CUDA devices
        assert local_rank < torch.cuda.device_count(), f"Invalid local_rank: {local_rank}"

        # Set the CUDA device based on local_rank
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cur_fold = cfg["base"]["fold"]
    check_pt_dir = Path(cfg["base"]["output"]["checkpoint"])
    tensorboard_dir = Path(cfg["base"]["output"]["tensorboard"])
    clip_image_encoder = cfg["model"]["image_encoder"]["model_type"]

    tensorboard_path_train = tensorboard_dir / f"fold_{cur_fold}/train"
    tensorboard_path_valid = tensorboard_dir / f"fold_{cur_fold}/valid"
    check_pt_path = check_pt_dir / f"fold_{cur_fold}"

    log.info(f"cur_fold: {cur_fold}")
    log.info(f"tensorboard_path_train: {tensorboard_path_train}")
    log.info(f"tensorboard_path_valid: {tensorboard_path_valid}")
    log.info(f"check_pt_path: {check_pt_path}")
    log.info(f"DistEnv: {util.GlobalEnv.get()}")
    log.info(f"{device}: Load datasets")

    print("=====================>>> Creating datasets <<<=====================")
    datamodule = DataModule(
        data_config=cfg["data_train"],
        dataloader_config=cfg["dataloader"],
        tokenizer_config=cfg["tokenizer"] if "tokenizer" in cfg else None,
        loss_config=cfg["loss"],
        transform_config=cfg["transform"],
        mean=cfg["base"]["mean"],
        std=cfg["base"]["std"],
        image_encoder_type=clip_image_encoder,
        cur_fold=cur_fold
    )

    valid_dataloaders = datamodule.valid_dataloader(distributed=distributed)

    log.info(f"{device}: Build the model")
    model = build_model(cfg["model"], cfg["loss"], datamodule.tokenizer)
    model = model.to(device)

    if distributed:
        model = DDP(model, device_ids=[device], find_unused_parameters=True)
    if util.GlobalEnv.get().master:
        log.info(f"{device}: Model info:\n{model}")

    log.info(f"{device}: Build the loss function")
    loss_func = build_loss(cfg["loss"])

    if local_rank < 1:
        import nltk

        log.info("Download nltk module")
        nltk.download("punkt")

    # tensorboard
    util.GlobalEnv.get().summary_writer.valid = util.DistSummaryWriter(tensorboard_path_valid)
    util.GlobalEnv.get().summary_writer.global_step = 0

    total_epochs = 5
    # validate
    if util.GlobalEnv.get().master:
        os.makedirs(check_pt_path, exist_ok=True)

        log.info(f"{device}: Validating the model")
        # training

        best_loss = 9e9
        epoch_resume = 0
        for epoch in range(epoch_resume, total_epochs):
            filename = check_pt_path / "model"
            chkpt_path = f"{filename}-epoch-{epoch + 1}.tar"
            ckpt = torch.load(chkpt_path, map_location=device)
            model.load_state_dict(ckpt["model"], strict=False)
            log.info(f"Epoch {epoch + 1}, {chkpt_path} is loaded")

            val_loss_dict_per_dataset = validate(
                model, device, loss_func, valid_dataloaders, epoch, total_epochs, local_rank, cfg["base"]["amp"],
                cfg["model"]["image_encoder"]["model_type"]
            )

            # tensorboard
            avg_val_loss_per_loss = {"total": 0.0}
            for loss_key in loss_func.loss_list:
                avg_val_loss_per_loss[loss_key.name] = 0.0

            for data_name, loss_dict in val_loss_dict_per_dataset.items():
                for loss_key, v in loss_dict.items():
                    util.GlobalEnv.get().summary_writer.valid.add_scalar(f"loss_per_epoch/{loss_key}/{data_name}", v,
                                                                         epoch + 1)
                    avg_val_loss_per_loss[loss_key] += v

            for loss_key in avg_val_loss_per_loss:
                avg_val_loss_per_loss[loss_key] /= len(valid_dataloaders)
                util.GlobalEnv.get().summary_writer.valid.add_scalar(f"loss_per_epoch/{loss_key}",
                                                                     avg_val_loss_per_loss[loss_key], epoch + 1)

            log.info(f"Epoch {epoch + 1}, validation loss: {avg_val_loss_per_loss[cfg['base']['loss_best']]}")
            if avg_val_loss_per_loss[cfg["base"]["loss_best"]] < best_loss:
                shutil.copyfile(chkpt_path, f"{filename}-best.tar")
                log.info(f"{filename}-best.tar saved")
                best_loss = avg_val_loss_per_loss[cfg["base"]["loss_best"]]

        util.GlobalEnv.get().summary_writer.valid.close()
        log.info(f"{device}: Validation has been completed")


def validate(model, device, loss_func, dataloader_dict, epoch, total_epochs, local_rank, amp, image_encoder,
             print_step=10):
    model.eval()
    loss_dict_per_dataset = dict()
    with torch.no_grad():
        for data_name, dataloader in dataloader_dict.items():
            avg_loss_dict = {"total": 0.0}
            for loss_key in loss_func.loss_list:
                avg_loss_dict[loss_key.name] = 0.0

            if util.GlobalEnv.get().local_rank < 1:
                progress_iter = tqdm(enumerate(dataloader), desc=f"[{epoch:03d}/{total_epochs:03d} epoch valid]",
                                     total=len(dataloader))
            else:
                progress_iter = enumerate(dataloader)

            for idx, batch in progress_iter:
                if image_encoder.lower() == "swin":
                    batch["images"] = batch["images"].squeeze(1).permute(0, 3, 1, 2)
                    batch["image_views"] = batch["image_views"].squeeze(1).permute(0, 3, 1, 2)
                if amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch, device)
                        loss_dict = loss_func(**outputs, is_train=False)
                else:
                    outputs = model(batch, device)
                    loss_dict = loss_func(**outputs, is_train=False)

                if util.GlobalEnv.get().world_size > 1:
                    for loss_key in loss_dict:
                        dist.all_reduce(loss_dict[loss_key], dist.ReduceOp.SUM)
                        loss_dict[loss_key] = loss_dict[loss_key] / util.GlobalEnv.get().world_size

                for loss_key in loss_dict:
                    avg_loss_dict[loss_key] += loss_dict[loss_key]

                if (idx % print_step == 0 or idx == len(dataloader) - 1) and local_rank < 1:
                    progress_iter.set_postfix(
                        {
                            "loss": f'{avg_loss_dict["total"]:.6f}',
                            "CUDA-Mem(%)": torch.cuda.memory_usage(device),
                            "CUDA-Util(%)": torch.cuda.utilization(device),
                        }
                    )

            for loss_key in avg_loss_dict:
                avg_loss_dict[loss_key] = avg_loss_dict[loss_key] / len(dataloader)

            loss_dict_per_dataset[data_name] = avg_loss_dict
    return loss_dict_per_dataset
