import os
import argparse
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils import data
import torch.nn as nn
from torchvision import datasets, models, transforms
import torchvision.utils as vutils
from dataset_visualize import *

parser = argparse.ArgumentParser()

# directory of the input embeddings
parser.add_argument('--embed-dir', default='', type=str)
# distance (score function)
parser.add_argument('--epoch', default=100,
                    type=int, help='which iteration of model to visualize')
# distance (score function)
parser.add_argument('--distance', default='cosine',
                    type=str, help='can be cosine or L2')
# number of nearest neighbors to retrieve
parser.add_argument('--K', default=10,
                    type=int, help='number of nearest neighbors')
parser.add_argument('--dataset', default='ucf11', type=str)


# according to DPC paper
parser.add_argument('--num_seq', default=5, type=int)
parser.add_argument('--seq_len', default=5, type=int)
parser.add_argument('--ds', default=3, type=int,
                    help='frame downsampling rate')
# note that the test split is fixed by default - as per DPC
parser.add_argument('--split', default=1, type=int)
parser.add_argument('--img_dim', default=256, type=int)
parser.add_argument('--mode', default='test', type=str)
parser.add_argument(
    '--class-names', default='/proj/vondrick/lovish/datasets/ucf11/classInd.txt', type=str)


def main():
    global args
    args = parser.parse_args()

    if args.distance == 'cosine':
        distance_f = nn.CosineSimilarity(dim=0, eps=1e-6)

    transform = transforms.Compose([
        # centercrop
        CenterCrop(size=224),
        # RandomSizedCrop(consistent=True, size=224, p=0.0),
        Scale(size=(args.img_dim, args.img_dim)),
        ToTensor(),
        # Normalize()
    ])

    # construct dictionary of class name
    if args.dataset == 'ucf11':
        classInd_dir = '/proj/vondrick/lovish/datasets/ucf11/classInd.txt'
    elif args.dataset == 'ucf101':
        classInd_dir = '/proj/vondrick/lovish/datasets/ucf101/splits_classification/classInd.txt'
    elif args.dataset == 'block_toy':
        classInd_dir = '/proj/vondrick/lovish/data/block_toy/classInd.txt'
    class_name = {}
    with open(classInd_dir) as f:
        for line in f:
            (key, val) = line.split()
            class_name[int(key)] = val
    print(class_name)

    if args.dataset == 'ucf101':
        dataset = UCF101_3d(mode=args.mode,
                            transform=transform,
                            seq_len=args.seq_len,
                            num_seq=args.num_seq,
                            downsample=args.ds,
                            which_split=args.split)
    elif args.dataset == 'hmdb51':
        dataset = HMDB51_3d(mode=args.mode,
                            transform=transform,
                            seq_len=args.seq_len,
                            num_seq=args.num_seq,
                            downsample=args.ds,
                            which_split=args.split)
    elif args.dataset == 'ucf11':
        # no split here
        dataset = UCF11_3d(mode=args.mode,
                           transform=transform,
                           num_seq=args.num_seq,
                           seq_len=args.seq_len,
                           downsample=args.ds)
    elif args.dataset == 'block_toy':
        # no split here
        dataset = block_toy(mode=args.mode,
                            transform=transform,
                            num_seq=args.num_seq,
                            seq_len=args.seq_len,
                            downsample=args.ds)
    else:
        raise ValueError('dataset not supported')

    # print(dataset[300][0].shape)
    # print(dataset[300][1])
    # print(dataset[300][2])

    # classes = list(map(int, args.classes.split(",")))
    # print(classes)

    distance_path = os.path.join(args.embed_dir, "%0.5d" % args.epoch)
    if args.distance == 'cosine':
        adj = torch.load(os.path.join(
            distance_path, 'cosine_distance.pt')).numpy()
    elif args.distance == 'L2':
        adj = torch.load(os.path.join(distance_path, 'L2_distance.pt')).numpy()

    labels = torch.load(os.path.join(distance_path, 'target_embed.pt')).numpy()
    # images = torch.load(os.path.join(args.embed_dir, 'input_embed.pt')).numpy() # input embeddings

    save_dir = os.path.join(distance_path, 'nearest_%d' % args.K)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # print(images.shape, labels.shape, adj.shape)
    # images = images * 0.15 + 0.45
    for i in class_name.keys():
        np.random.seed(0)
        try:
            # sample 1 image from action class 'i'
            index_origin = np.random.choice(np.where(labels == i - 1)[0], 1)[0]
        except Exception:
            break
        print(index_origin)
        if args.distance == 'L2':
            # query K nearest neighbors from adjacency matrix
            indices_quiried = adj[index_origin].argsort()[1:args.K]
        elif args.distance == 'cosine':
            indices_quiried = adj[index_origin].argsort()[-args.K:-1][::-1]
        # print(adj[0])
        # print(adj[1])
        # print(indices_quiried)
        # print("********")
        # print(indices_quiried)
        image_origin = dataset[index_origin][0].permute(
            1, 0, 2, 3, 4)[:, 2, 2, :, :].numpy()
        # directory for class being quiried
        class_dir = os.path.join(save_dir, class_name[i])
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        plt.imsave(os.path.join(class_dir, 'origin_class_%s.jpg' %
                                class_name[i]), np.transpose(image_origin, (1, 2, 0)))
        images_quiried_labels = labels[indices_quiried]
        # print(len(indices_quiried))
        # print(images_quiried_labels)
        # print(indices_quiried)
        # print(images_quiried_labels)
        for j in range(len(indices_quiried)):
            num = indices_quiried[j]
            image_quiried = dataset[num][0].permute(
                1, 0, 2, 3, 4)[:, 2, 2, :, :].numpy()
            label = dataset[num][1].numpy()
            plt.imsave(os.path.join(class_dir, 'quiried_num_%d_class_%s.jpg' % (
                j, class_name[images_quiried_labels[j] + 1])), np.transpose(image_quiried, (1, 2, 0)))


if __name__ == '__main__':
    main()
