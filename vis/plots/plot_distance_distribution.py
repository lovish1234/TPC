import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import matplotlib as pyplot
import argparse
import glob
import torch


def get_class(dataset='block_toy'):
    # construct dictionary of class name
    if dataset == 'ucf11':
        classInd_dir = '/proj/vondrick/lovish/datasets/ucf11/classInd.txt'
    elif dataset == 'ucf101':
        classInd_dir = '/proj/vondrick/lovish/datasets/ucf101/splits_classification/classInd.txt'
    elif dataset == 'block_toy' or dataset == 'block_toy_imagenet':
        classInd_dir = '/proj/vondrick/lovish/data/block_toy/classInd.txt'
    class_name = {}
    with open(classInd_dir) as f:
        for line in f:
            (key, val) = line.split()
            class_name[int(key)] = val
    return class_name


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='block_toy', type=str)
parser.add_argument('--embed-dir', default='/proj/vondrick/lovish/DPC/dpc/log_tmp/block_toy_imagenet_3-128_r10_dpc-rnn_bs64_lr0.001_seq6_pred2_len5_ds1_train-all/neighbours_ImageNet/00200/', type=str)
parser.add_argument('--score-type', default='L2', type=str)

args = parser.parse_args()

distance_dir = os.path.join(args.embed_dir, '%s_distance.pt' % args.score_type)
radius_dir = os.path.join(args.embed_dir, 'radius.pt')
target_dir = os.path.join(args.embed_dir, 'target_embed.pt')

distance = torch.load(distance_dir).numpy()
radius = torch.load(radius_dir).numpy()
target = torch.load(target_dir).numpy()
target = target + 1  # block_toy class name starts from -1 to 9


class_name = get_class(args.dataset)
path = os.path.join(args.embed_dir, 'distances_distribution')
try:
    os.mkdir(path)
except:
    print('directory existed')
    exit()

for i in class_name.keys():
    pos_idx = np.where(target == i)[0]
    neg_idx = np.where(target != i)[0]
    distance_class = distance[pos_idx[0]]
    # print(distance_class)
    # print(pos_idx)
    # print(neg_idx)
    pos_dist = distance_class[pos_idx]
    neg_dist = distance_class[neg_idx]

    bins = np.linspace(0, 5, 50)
    plt.hist(pos_dist, bins, alpha=0.5, label='positive')
    plt.hist(neg_dist, bins, alpha=0.5, label='negative')
    plt.legend(loc='upper right')
    filename = os.path.join(
        args.embed_dir, 'distances_distribution', 'class%d.png' % i)
    plt.savefig(filename)
    plt.clf()
