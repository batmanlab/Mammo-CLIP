import logging

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

log = logging.getLogger(__name__)


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class Single_layer_network(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True):
        super(Single_layer_network, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size, bias=bias)

    def forward(self, x):
        out = self.linear(x)
        return out


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        if isinstance(self.p, int):
            # If self.p is an integer
            return (
                    self.__class__.__name__
                    + "("
                    + "p="
                    + "{:.4f}".format(self.p)
                    + ", "
                    + "eps="
                    + str(self.eps)
                    + ")"
            )
        else:
            # If self.p is a PyTorch tensor
            return (
                    self.__class__.__name__
                    + "("
                    + "p="
                    + "{:.4f}".format(self.p.data.tolist()[0])
                    + ", "
                    + "eps="
                    + str(self.eps)
                    + ")"
            )


class MammoEfficientNet(nn.Module):
    def __init__(self, cfg, *, in_chans=1, p=3, p_trainable=False, eps=1e-6):
        super(MammoEfficientNet, self).__init__()
        name = cfg["clf_arch"]
        pretrained = cfg["pretrained"]
        get_features = cfg["get_features"]

        model = timm.create_model(name, pretrained=pretrained, in_chans=in_chans)
        clsf = model.default_cfg['classifier']
        n_features = model._modules[clsf].in_features
        model._modules[clsf] = nn.Identity()

        self.fc = nn.Linear(n_features, 1)
        self.model = model
        self.get_features = get_features
        self.pool = nn.Sequential(GeM(p=p, eps=eps, p_trainable=p_trainable), nn.Flatten())

    def forward(self, x):
        # x = self.model(x)
        x = self.model.forward_features(x)
        x = self.pool(x)
        logits = self.fc(x)
        if self.get_features:
            return x, logits
        else:
            return logits
