import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import pickle
import sys
from heapq import nlargest
from IPython.display import HTML, Video
from PIL import Image
from skvideo.io import FFmpegWriter
from torchvision import datasets, models, transforms
sys.path.append('../../tpc-backbone/')
sys.path.append('../../tpc/')
sys.path.append('../../vae-dpc/')
sys.path.append('../../utils/')
sys.path.append('../../utils/diversity_meas')
sys.path.append('../epic/')
sys.path.append('../')
from div_vae import *
from div_tpc import *
from dataset_3d import *
from diversity import *
from epic_data import *
from utils import *
from tqdm import tqdm
import random

# parameters
num_videos = 100  # number of videos to sample from epic-kitchens
n_frames = 20 * 6 * 5
video_info = pd.read_csv('epic_videos.csv')
tpc_path = '/proj/vondrick/ruoshi/github/TPC/tpc/log_tmp/phi_capacity/epic-128_r34_dpc-rnn_bs64_lr0.001_seq8_pred3_len5_ds6_train-all_distance_L2_distance-type_uncertain_margin_10.0/model/model_best_epoch129.pth.tar'
cvae_path = '/proj/vondrick/basile/Uncertainty/TPC/vae-dpc/log_conv/epic-128_r34_dpc-cvae_bs64_lr0.001_seq8_pred3_len5_ds6_kl64.0_train-all/model/model_best_epoch300.pth.tar'

# calculate and store uncertainty
for i in tqdm(range(num_videos)):

    # sample and generate video
    # sample a video based-on length distribution
    row = video_info.sample(weights=video_info['prob']).iloc[0]
    video_id = row['video_id']
    video_info = pd.read_csv('epic_videos.csv')
    # avoid short clip at the end
    start_frame = int(random.random() * (row['length'] - n_frames - 1))
    start_uncertainty = start_frame + 5 * 6 * 5
    convert_epic_video(video_id, start_frame, 'videos/%s_%d_%d.mp4' % (video_id, start_frame,
                                                                       start_uncertainty), n_frames=n_frames, insert_pause=-1, show_vid=False)

    # measure tpc uncertainty
    video_path = 'videos/%s_%d_%d.mp4' % (video_id,
                                          start_frame, start_uncertainty)
    results_dir = 'tpc_visuals/%s_%d_%d' % (video_id,
                                            start_frame, start_uncertainty)
    results_tpc = measure_tpc_uncertainty_video(
        tpc_path, video_path, results_dir, gpus_str='7,6', batch_size=64)
    torch.cuda.empty_cache()

    # measure cvae uncertainty

    video_path = 'videos/%s_%d_%d.mp4' % (video_id,
                                          start_frame, start_uncertainty)
    results_dir = 'cvae_visuals/%s_%d_%d' % (
        video_id, start_frame, start_uncertainty)
    results_cvae = measure_cvae_uncertainty_video(
        cvae_path, video_path, results_dir, gpus_str='7,6', batch_size=64)
    torch.cuda.empty_cache()
