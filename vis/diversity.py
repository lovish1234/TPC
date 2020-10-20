# April 2020
# Visualize diversity for uncertainty estimation over time of video sequences
# TODO: this file is kind of outdated, since this functionality is taken over by Lovish instead of Basile now

import os
import ntpath
import shutil
import pickle
import sys
import time
import re
import argparse
import numpy as np
from tqdm import tqdm

# visualization
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
import seaborn as sns
sns.set()
from scipy.signal import find_peaks
# from matplotlib import rcParams
# rcParams.update({'figure.autolayout': True})

plt.switch_backend('agg')

sys.path.append('../vae-dpc')
sys.path.append('../tpc')
sys.path.append('../utils')
sys.path.append('../utils/diversity_meas')
from cvae_model import *
from vrnn_model import *
from dataset_3d import *
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

torch.backends.cudnn.benchmark = True



def enrich_uncertain_results(video_fp, results, to_dir, format='vrnn', results_from_path=False, plot_metrics=None):
    '''
    Visualization method after processing all sequences of a single video file.
    Saves (1) original video (2) results object (3) graph plots (4) annotated video to the given folder.

    NOTE: results is expected to be a dictionary following a certain format, example key-value pairs:
    {('bla.mp4', 1234): [[4.6277223  2.4677615  2.546618  ]
                         [0.3095014  0.14996651 0.15751362]
                         [0.9995848  0.9991826  0.9988366 ]
                         [0.99996156 0.99993753 0.99990815]],
     ('bla.mp4', 1236): ..., ...}.
    Here, the horizontal index is time step, and the vertical index is distance metric:
    If format = 'cvae':   (L2 / L2 after pool / cosine / cosine after pool / norm L2 / norm L2 after pool / repeat for hidden states)
    If format = 'vrnn':   (same as above)
    If format = 'tpc':    (radius)
    If format = 'motion': (global optical flow magnitude / global optical flow entropy)

    results_from_path: if True, results is a file path to be imported using pickle, otherwise object itself.
    plot_metrics: List of indices of uncertainty metrics to plot (legend labels hardcoded).
    '''

    if results_from_path:
        with open(results, 'rb') as f:
            results = pickle.load(results)

    if not(os.path.exists(to_dir)):
        os.makedirs(to_dir)

    fn_video = ntpath.basename(video_fp)
    fp_video_copy = os.path.join(to_dir, fn_video)
    fp_results = os.path.join(to_dir, format + '_results.p')
    fp_graph = os.path.join(to_dir, format + '_graph.png')
    fp_video_annot = os.path.join(
        to_dir, fn_video[:-4] + '_annot_' + format + fn_video[-4:])

    # Original video
    shutil.copy2(video_fp, fp_video_copy)

    # Results object (NOTE: assume this already exists, do not overwrite anything to avoid risk)
    # with open(fp_results, 'wb') as f:
    #     pickle.dump(results, f)

    # Graph with thumbnails
    # plot_uncertainity for plotting multiple uncertaininty metrics together
    # plot_uncertainty(results, video_fp, fp_graph,
    #                  format=format, metrics=plot_metrics)
    # plot_uncertainty_embed for plotting images embedded along with extremums of the metrics
    plot_uncertainty_embed(results, video_fp, fp_graph,
                     format=format, metrics=plot_metrics)

    # Annotated video
    annotate_video(results, video_fp, fp_video_annot)


def plot_uncertainty(results, video_fp, dst_fp, stride=3, format='vrnn', metrics=None, time_steps=None):
    '''
    Stores a graph of the uncertainty results to the specified destination file.
    stride: apply spacing between data points *on top of* any start_stride that might have been used earlier.
    metrics: list of values in [0, 12] (VAE) or [0, 3] (motion) inclusive, selecting distance metrics (see labels below).
    time_steps: list of values in [0, 2] (VAE only) inclusive, selecting future time steps.
    '''

    # Specify labels and default set of metrics by format
    if format == 'cvae' or 'vrnn' in format:
        labels = ['Pred L2 dist', 'Pred pooled L2 dist', 'Pred norm L2 dist', 'Pred pooled norm L2 dist',
                  'Pred cosine dist', 'Pred pooled cosine dist',
                  'Context L2 dist', 'Context pooled L2 dist', 'Context norm L2 dist', 'Context pooled norm L2 dist',
                  'Context cosine dist', 'Context pooled cosine dist',
                  'Verb entropy', 'Noun entropy']
        # if cosine distance, apply affine transform
        to_magnify = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]
        if metrics is None:
            metrics = [3, 9, 12, 13]
            time_steps = [3, 3, 3, 3]
        if time_steps is None:
            time_steps = [3] * len(metrics)
        assert(len(metrics) == len(time_steps))

    elif format == 'motion':
        labels = ['Magnitude', 'Angle', 'X', 'Y']
        if metrics is None:
            # by default take the magnitude of motion
            metrics = [0]

    # Gather relevant information from dictionary
    if format == 'tpc':
        x = []
        y1 = []
        for kv in list(results.items())[::stride]:
            frame = kv[0][1]
            x.append(frame + 5 * 5 * 6)
            # Radius looking 0.0 to 0.5 seconds into the future
            radius = kv[1]
            y1.append(radius)

    elif format == 'cvae' or 'vrnn' in format:
        x = []  # list
        ys = []  # list of lists
        for j in range(len(metrics)):
            ys.append([])

        # Loop over all selected frames
        for kv in list(results.items())[::stride]:
            frame = kv[0][1]
            x.append(frame + 5 * 5 * 6)

            # Gather data points (different metrics and/or time steps) for this frame
            for j, (metric, time_step) in enumerate(zip(metrics, time_steps)):
                if metric < 12:
                    cur_y = kv[1]['mean_var'][metric]
                else:
                    # 12 or above => select class type
                    cur_y = (kv[1]['action_entropy'][metric - 12] - 4) / 3
                if time_step < len(cur_y):
                    cur_y = cur_y[time_step]
                else:
                    cur_y = cur_y.mean()
                if to_magnify[metric]:
                    cur_y = (1 - cur_y) * 9e3
                ys[j].append(cur_y)

    elif format == 'motion':
        x = []  # list
        ys = []  # list of lists
        for j in range(len(metrics)):
            ys.append([])

        # Loop over all selected frames
        for kv in list(results.items())[::stride]:
            frame = kv[0][1]
            # no offset because motion can be measured everywhere
            x.append(frame)
            # Gather data points (different metrics) for this frame
            for j, metric in enumerate(metrics):
                cur_y = kv[1][metric]
                ys[j].append(cur_y)

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(nrows=6, ncols=12)

    # Show thumbnails above graph
    frames = read_all_frames(video_fp)
    n_frames = frames.shape[0]
    fps = 60
    for i in range(6):
        ax = fig.add_subplot(gs[:2, i * 2:i * 2 + 2])
        frame_idx = int(i / 6 * n_frames + n_frames / 12)
        ax.imshow(frames[frame_idx])
        ax.axis('off')

    # Plot results graph
    x = np.array(x)
    ax = fig.add_subplot(gs[2:, :])

    if format == 'cvae' or 'vrnn' in format:
        for j, y in enumerate(ys):
            cur_time_step = time_steps[j]
            if cur_time_step >= 3:
                cur_time_step = 'all'
            ax.plot(x / fps, y, label=labels[metrics[j]] + ' t=' + str(cur_time_step))
    elif format == 'motion':
        for j, y in enumerate(ys):
            ax.plot(x / fps, y, label=labels[metrics[j]])
    elif format == 'tpc':
        ax.plot(x / fps, y1, label='radius')

    ax.grid(True)
    ax.legend()
    ax.set_xlim([0, n_frames / fps])
    ax.set_xlabel('Time [s]')

    if format == 'cvae' or 'vrnn' in format:
        ax.set_ylabel('Mean variance / entropy')
    elif format == 'tpc':
        ax.set_ylabel('TPC radius')
    elif format == 'motion':
        ax.set_ylabel('Camera motion')

    if format != 'motion':
        ax.axvline(x=2.5, color='gray', dashes=[3, 3])
        ax.axvline(x=n_frames / fps - 1.5, color='gray', dashes=[3, 3])

    ax.set_title(list(results.keys())[0][0])

    plt.savefig(dst_fp, dpi=384, bbox_inches = "tight")

    return fig

def smooth(signal, filter_size=3, metrics=[0]):

    # Here
    y_smooth_all = []
    for i in range(len(metrics)):
        filter_instance = np.ones(filter_size)/filter_size
        y_smooth = np.convolve(signal[i], filter_instance, mode='same')
        y_smooth_all.append(y_smooth)
    return y_smooth_all


def plot_uncertainty_embed(results, video_fp, dst_fp, stride=3, format='motion', metrics=None, time_steps=None):
    '''
    *** Currently works for optical flow only***
    Stores a graph of the uncertainty results to the specified destination file.
    stride: apply spacing between data points *on top of* any start_stride that might have been used earlier.
    metrics: list of values in [0, 12] (VAE) or [0, 3] (motion) inclusive, selecting distance metrics (see labels below).
    time_steps: list of values in [0, 2] (VAE only) inclusive, selecting future time steps.
    '''
    allowed_formats = ['motion', 'vrnn']


    if format not in allowed_formats:
        print("The format %s is currently not supported"% (format))
        return None

    if format=='motion':
        # Specify labels and default set of metrics by format
        labels = ['Magnitude', 'Angle', 'X', 'Y']
        if metrics is None:
            # by default take the magnitude of motion
            metrics = [0]

        x = []  # list
        ys = []  # list of lists
        for j in range(len(metrics)):
            ys.append([])


        stored_stride = list(results.items())[1][0][1] - list(results.items())[0][0][1]

        # Loop over all selected frames
        for kv in list(results.items())[::stride]:
            frame = kv[0][1]
            # no offset because motion can be measured everywhere
            x.append(frame)
            # Gather data points (different metrics) for this frame
            for j, metric in enumerate(metrics):
                cur_y = kv[1][metric]
                ys[j].append(cur_y)

    else:

        labels = ['Pred L2 dist', 'Pred pooled L2 dist', 'Pred norm L2 dist', 'Pred pooled norm L2 dist',
                  'Pred cosine dist', 'Pred pooled cosine dist',
                  'Context L2 dist', 'Context pooled L2 dist', 'Context norm L2 dist', 'Context pooled norm L2 dist',
                  'Context cosine dist', 'Context pooled cosine dist',
                  'Verb entropy', 'Noun entropy']
        # if cosine distance, apply affine transform
        to_magnify = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]
        if metrics is None:
            metrics = [9]
            #metrics = [3, 9, 12, 13]
            time_steps = [3]
            #time_steps = [3, 3, 3, 3]
        if time_steps is None:
            time_steps = [3] * len(metrics)
        assert(len(metrics) == len(time_steps))

        x = []  # list
        ys = []  # list of lists
        for j in range(len(metrics)):
            ys.append([])

        stored_stride = list(results.items())[1][0][1] - list(results.items())[0][0][1]
        # Loop over all selected frames
        for kv in list(results.items())[::stride]:
            frame = kv[0][1]
            x.append(frame + 5 * 5 * 6)

            # Gather data points (different metrics and/or time steps) for this frame
            for j, (metric, time_step) in enumerate(zip(metrics, time_steps)):
                if metric < 12:
                    cur_y = kv[1]['mean_var'][metric]
                else:
                    # 12 or above => select class type
                    cur_y = (kv[1]['action_entropy'][metric - 12] - 4) / 3
                if time_step < len(cur_y):
                    cur_y = cur_y[time_step]
                else:
                    cur_y = cur_y.mean()
                if to_magnify[metric]:
                    cur_y = (1 - cur_y) * 9e3
                ys[j].append(cur_y)

    ys = smooth(ys, filter_size=5, metrics=metrics)
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)

    # Embed images above maxima
    frames = read_all_frames(video_fp)
    n_frames = frames.shape[0]
    fps = 60

    # Plot results graph
    x = np.array(x)
    for j, y in enumerate(ys):
        ax.plot(x / fps, y, label=labels[metrics[j]])

    if format=='motion':
        # Display the maximax
        peaks, _ = find_peaks(y, height=3, prominence=5)
    elif format=='vrnn':
        peaks, _ = find_peaks(y, distance=10)

    ax.plot(x[peaks]/fps, y[peaks], "x")
    y_limit = ax.get_ylim()[1]
    for i in range(len(peaks)):

        arr_img = frames[peaks[i]*stride*stored_stride]

        imagebox = OffsetImage(arr_img, zoom=0.3)
        imagebox.image.axes = ax
        xy = [x[peaks[i]]/fps, y[peaks[i]]]

        ab = AnnotationBbox(imagebox, xy,
                            xybox=(x[peaks[i]]/fps, 1.2*y_limit),
                            xycoords='data',
                            boxcoords="data",
                            pad=0.5,
                            arrowprops=dict(
                                color='black',
                                arrowstyle="->")
                            )
        ax.add_artist(ab)

    ax.grid(True)
    ax.legend()
    ax.set_xlim([0, n_frames / fps])

    if format=='motion':
        ax.set_ylabel('Camera motion')
        ax.set_title(list(results.keys())[0][0])

    elif format=='vrnn':
        ax.set_ylabel('Mean variance / entropy')
        ax.set_title(list(results.keys())[0][0])
        ax.axvline(x=2.5, color='gray', dashes=[3, 3])
        ax.axvline(x=n_frames / fps - 1.5, color='gray', dashes=[3, 3])

    # pad_inches necessary otherwise image frames get cropped out
    plt.savefig(dst_fp, dpi=384, bbox_inches='tight', pad_inches=2.5)


    return fig

def annotate_video(results, video_fp, dst_fp, moving_avg=12, pause_max_uncert=True):
    '''
    Generates and stores a copy of the video with border colors indicating uncertainty of the immediate future.
    Red = uncertain, green = certain, blue = maximal uncertainty.
    moving_avg: Smoothen signal by storing the average of the given number of latest data points.
    pause_max_uncert: Visually indicate the point of maximal diversity.
    '''
    # TODO
    pass
