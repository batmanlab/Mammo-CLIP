import logging
import os
import argparse
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from breastclip import convert_dictconfig_to_dict, run, run_ddp, seed_everything

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train_vit_upmc_rsna_vindr")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = -1

    if local_rank < 1:
        log.info(f"Configurations:\n{OmegaConf.to_yaml(cfg)}")

    seed_everything(cfg.base.seed)
    torch.backends.cudnn.benchmark = True

    cfg = convert_dictconfig_to_dict(cfg)
    if "LOCAL_RANK" in os.environ:
        run_ddp(local_rank, cfg)
    else:
        run(local_rank, cfg)


if __name__ == "__main__":
    main()