from typing import Dict

from .breast_clip import BreastClip
from .breast_clip_contrastive import BreastClip_contrastive
from .classification import Classification
from .combined_loss import CombinedLoss


def build_loss(all_loss_config: Dict) -> CombinedLoss:
    loss_list = []

    for loss_config in all_loss_config:
        cfg = all_loss_config[loss_config]
        if cfg["loss_ratio"] == 0.0:
            continue
        if loss_config == "classification":
            loss = Classification(**cfg)
        elif loss_config == "breast_clip":
            loss = BreastClip(**cfg)
        elif loss_config == "breast_clip_contrastive":
            loss = BreastClip_contrastive(**cfg)
        else:
            raise KeyError(f"Unknown loss: {loss_config}")

        loss_list.append(loss)

    total_loss = CombinedLoss(loss_list)
    return total_loss
