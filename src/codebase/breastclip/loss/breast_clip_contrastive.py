import torch
import torch.nn as nn
from breastclip import util
from torch.nn import functional as F

all_gather_func = util.DistAutogradAllGatherFunction(partial=False)


def all_gather(tensor):
    world_size = util.GlobalEnv.get().world_size
    if world_size > 1:
        tensor_list = all_gather_func.apply(tensor)
        all_tensor = torch.cat(tensor_list, 0)
    else:
        all_tensor = tensor
    return all_tensor


class BreastClip_contrastive(nn.Module):
    def __init__(self, label_smoothing=0.0, i2i_weight=0.0, t2t_weight=0.0, loss_ratio=1.0):
        super(BreastClip_contrastive, self).__init__()
        self.name = "contrastive"
        self.label_smoothing = label_smoothing
        self.loss_ratio = loss_ratio
        self.i2i_weight = i2i_weight
        self.t2t_weight = t2t_weight

    def forward(self, image_embeddings, text_embeddings, labels, logit_scale, is_train, **kwargs):
        world_rank = util.GlobalEnv.get().world_rank
        batch_size = labels.size(0)

        all_image_embeddings = all_gather(image_embeddings)
        all_text_embeddings = all_gather(text_embeddings)

        with torch.no_grad():
            labels = labels + (world_rank * batch_size)

        loss_i2t = 0
        loss_t2i = 0

        # I1 - T1
        logits_per_image = logit_scale * image_embeddings @ all_text_embeddings.T
        logits_per_text = logit_scale * text_embeddings @ all_image_embeddings.T

        label_smoothing = self.label_smoothing if is_train else 0.0
        loss_i2t += F.cross_entropy(logits_per_image, labels, label_smoothing=label_smoothing)
        loss_t2i += F.cross_entropy(logits_per_text, labels, label_smoothing=label_smoothing)

        if is_train:
            util.GlobalEnv.get().summary_writer.train.add_scalar(
                "loss/contrastive/steps_i2t", loss_i2t, util.GlobalEnv.get().summary_writer.global_step
            )
            util.GlobalEnv.get().summary_writer.train.add_scalar(
                "loss/contrastive/steps_t2i", loss_t2i, util.GlobalEnv.get().summary_writer.global_step
            )

        # contrastive loss
        loss = (0.75 * loss_i2t + 0.25 * loss_t2i)  # shape: (batch_size,)
        return loss.mean()
