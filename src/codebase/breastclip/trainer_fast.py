import gc
import logging
import math
import os
import pickle
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from typing import Dict

from . import util
from .data import DataModule
from .loss import build_loss
from .model import build_model
from .optimizer import build_optimizer
from .scheduler import build_scheduler

log = logging.getLogger(__name__)


def run_fast(local_rank, cfg: Dict):
    if "tokenizer" in cfg:
        assert (
                cfg["tokenizer"]["pretrained_model_name_or_path"] == cfg["model"]["text_encoder"]["name"]
        ), "tokenizer should be same to text_encoder"

    distributed = local_rank != -1
    log.info(f"Running without validation loop")
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

    if "args_path" in cfg["base"]["output"]:
        pickle.dump(cfg, open(os.path.join(cfg["base"]["output"]["args_path"], "cfg.pkl"), "wb"))

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

    train_dataloader, train_sampler = datamodule.train_dataloader(distributed=distributed)

    log.info(f"{device}: Build the model")
    model = build_model(cfg["model"], cfg["loss"], datamodule.tokenizer)
    model = model.to(device)

    if "resume_training" in cfg["base"] and cfg["base"]["resume_training"]:
        filename = check_pt_path / "model"
        model_idx = cfg["base"]["epoch_to_start"]
        chkpt_path = f"{filename}-epoch-{model_idx}.tar"
        log.info(f"Loading checkpoint: {chkpt_path}")
        ckpt = torch.load(chkpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)

    if distributed:
        model = DDP(model, device_ids=[device], find_unused_parameters=True)
    if util.GlobalEnv.get().master:
        log.info(f"{device}: Model info:\n{model}")

    log.info(f"{device}: Build the loss function")
    loss_func = build_loss(cfg["loss"])

    log.info(f"{device}: Build the optimizer")
    optimizer = build_optimizer(model, cfg["optimizer"])

    log.info(f"{device}: Build the LR scheulder")
    if "total_epochs" in cfg["scheduler"]["config"]:
        # with open_dict(cfg):
        cfg["scheduler"]["config"]["total_steps"] = len(train_dataloader) * cfg["scheduler"]["config"]["total_epochs"]
    if "warmup_epochs" in cfg["scheduler"]["config"]:
        # with open_dict(cfg):
        if isinstance(cfg["scheduler"]["config"]["warmup_epochs"], int):
            cfg["scheduler"]["config"]["warmup_steps"] = len(train_dataloader) * cfg["scheduler"]["config"][
                "warmup_epochs"]
        elif isinstance(cfg["scheduler"]["config"]["warmup_epochs"], float):
            cfg["scheduler"]["config"]["warmup_steps"] = cfg["scheduler"]["config"]["warmup_epochs"]

    scheduler = build_scheduler(optimizer, cfg["scheduler"])
    scaler = torch.cuda.amp.GradScaler() if cfg["base"]["amp"] else None

    if local_rank < 1:
        import nltk

        log.info("Download nltk module")
        nltk.download("punkt")

    # train
    if "total_epoch" in cfg["scheduler"]:
        total_epochs = cfg["scheduler"]["total_epoch"]
        cfg["scheduler"]["config"]["total_steps"] = total_epochs * len(train_dataloader)
    else:
        total_epochs = math.ceil(cfg["scheduler"]["config"]["total_steps"] / len(train_dataloader))

    # tensorboard
    util.GlobalEnv.get().summary_writer.train = util.DistSummaryWriter(tensorboard_path_train)
    util.GlobalEnv.get().summary_writer.valid = util.DistSummaryWriter(tensorboard_path_valid)
    util.GlobalEnv.get().summary_writer.global_step = 0
    util.GlobalEnv.get().summary_writer.train.add_text(
        "hyperparams/config", "\n".join(["\t" + line for line in OmegaConf.to_yaml(cfg).splitlines()]), 0
    )
    if util.GlobalEnv.get().master:
        os.makedirs(check_pt_path, exist_ok=True)

        log.info(f"{device}: Training the model")
        # training

        best_loss = 9e9
        if "epoch_to_start" in cfg["base"]:
            epoch_resume = cfg["base"]["epoch_to_start"]
        else:
            epoch_resume = 0

        log.info(f"Training is started from epoch: {epoch_resume}")
        for epoch in range(epoch_resume, total_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_loss_dict = train(
                model,
                device,
                loss_func,
                optimizer,
                scheduler,
                train_dataloader,
                epoch,
                total_epochs,
                scaler,
                cfg["scheduler"]["config"]["total_steps"],
                cfg["model"]["image_encoder"]["model_type"]
            )

            # tensorboard
            for k, v in train_loss_dict.items():
                util.GlobalEnv.get().summary_writer.train.add_scalar(f"loss_per_epoch/{k}", v, epoch + 1)

            if util.GlobalEnv.get().master:
                # checkpoint
                filename = check_pt_path / "model"
                checkpoint = f"{filename}-epoch-{epoch + 1}.tar"
                model_state_dict = model.state_dict() if local_rank == -1 else model.module.state_dict()
                torch.save(
                    {
                        "model": model_state_dict,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "config": cfg,
                        "epoch": epoch + 1,
                        "train_loss": train_loss_dict["total"],
                    },
                    checkpoint,
                )
                log.info(f"Epoch {epoch + 1}, last-model saved")

        util.GlobalEnv.get().summary_writer.train.close()
        util.GlobalEnv.get().summary_writer.valid.close()
        log.info(f"{device}: Training has been completed")


def train(model, device, loss_func, optimizer, scheduler, dataloader, epoch, total_epochs, scaler, total_step,
          image_encoder, print_step=30):
    model.train()
    if util.GlobalEnv.get().local_rank < 1:
        progress_iter = tqdm(enumerate(dataloader), desc=f"[{epoch:03d}/{total_epochs:03d} epoch train]",
                             total=len(dataloader))
    else:
        progress_iter = enumerate(dataloader)

    avg_loss_dict = {"total": 0.0}
    for k in loss_func.loss_list:
        avg_loss_dict[k.name] = 0.0

    for idx, batch in progress_iter:
        optimizer.zero_grad(set_to_none=True)
        if image_encoder.lower() == "swin":
            batch["images"] = batch["images"].squeeze(1).permute(0, 3, 1, 2)
            batch["image_views"] = batch["image_views"].squeeze(1).permute(0, 3, 1, 2)

        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(batch, device)
                loss_dict = loss_func(**outputs, is_train=True)
            total_loss = loss_dict["total"]
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch, device)
            loss_dict = loss_func(**outputs, is_train=True)
            total_loss = loss_dict["total"]
            total_loss.backward()
            optimizer.step()

        scheduler.step()
        util.GlobalEnv.get().summary_writer.global_step = scheduler._step_count

        loss_dict = {key: value.detach().cpu() for key, value in loss_dict.items()}
        for k in loss_dict:
            avg_loss_dict[k] += loss_dict[k]

        total_loss = total_loss.detach().cpu()
        if idx % print_step == 0 and util.GlobalEnv.get().local_rank < 1:
            for k, lr in enumerate(scheduler.get_last_lr()):
                util.GlobalEnv.get().summary_writer.train.add_scalar(f"hyperparam/lr-{k}", lr,
                                                                     scheduler._step_count)
            util.GlobalEnv.get().summary_writer.train.add_scalar("loss", total_loss, scheduler._step_count)

            for k in loss_dict:
                util.GlobalEnv.get().summary_writer.train.add_scalar(f"loss/{k}", loss_dict[k],
                                                                     scheduler._step_count)

            progress_iter.set_postfix(
                {
                    "lr": [f"{v:.8f}" for v in scheduler.get_last_lr()],
                    "loss": f"{total_loss:.6f}",
                    "CUDA-Mem": f"{torch.cuda.memory_usage(device)}%",
                    "CUDA-Util": f"{torch.cuda.utilization(device)}%",
                }
            )
        if total_step == scheduler._step_count:
            break

    for k in avg_loss_dict:
        avg_loss_dict[k] = avg_loss_dict[k] / len(dataloader)

    torch.cuda.empty_cache()
    gc.collect()
    return avg_loss_dict
