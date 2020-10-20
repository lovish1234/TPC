# BVH & RL, April 2020
# Test trained CVAE & TPC and measure diversity of samples for uncertainty estimation of specific video files
# This file contains common code (imports etc.)

import os
import ntpath
import shutil
import pickle
import sys
import time
import re
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path

# visualization
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
plt.switch_backend('agg')

sys.path.append('../../tpc')
sys.path.append('../../vae-dpc')
sys.path.append('../../utils')
sys.path.append('../../vis/epic')
sys.path.append('../tpc')
sys.path.append('../vae-dpc')
sys.path.append('../utils')
sys.path.append('../vis/epic')
from model_visualize import *
from cvae_model import *
from vrnn_model import *
from dataset_epic import *
from dataset_other import *
from epic_data import *

# backbone
from resnet_2d3d import neq_load_customized

# data augmentation methods
from augmentation import *

# collect results
from utils import AverageMeter, save_checkpoint, denorm, calc_topk_accuracy

import torch
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms

# saving log, calculating accuracy etc.
import torchvision.utils as vutils

# torch.backends.cudnn.benchmark = True # TODO: what is this for?
