import argparse
import os

import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import datasets
import models
import utils
from statistics import mean
import torch
import torch.distributed as dist

torch.distributed.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

# config로 불러와야 하는 내용
# scoring
# - zeroshot proxy
# - search space
# sam-adapter
# - dataset
# - sam pretrained model