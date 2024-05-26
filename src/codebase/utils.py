import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_device(args):
    return "cuda" if args.device == "cuda" else 'cpu'


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_Paths(args):
    chk_pt_path = Path(f"{args.checkpoints}/{args.dataset}/{args.model_type}/{args.arch}/{args.root}")
    output_path = Path(f"{args.output_path}/{args.dataset}/zz/{args.model_type}/{args.arch}/{args.root}")
    tb_logs_path = Path(f"{args.tensorboard_path}/{args.dataset}/{args.model_type}/{args.arch}/{args.root}")

    return chk_pt_path, output_path, tb_logs_path


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
