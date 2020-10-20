# Inspired by : https://github.com/Singularity42/Sync-DRAW/blob/master/dataset/mnist_caption_single.py


import argparse

import numpy as np
from random import randint
import datetime
import scipy
import math


# from scipy.misc import toimage
import numpy as np
from random import randint
import random

# loading the mnist dataset
import gzip
import pickle

# saving the gif
from PIL import Image
import os
from moviepy.editor import ImageSequenceClip


# set the random seed
np.random.seed(np.random.randint(1 << 30))


def create_dataset():
    '''
    Used for creating a dummy dataset split
    '''
    numbers = np.random.permutation(10)

    dataset = np.zeros((2, 10), dtype=np.int)
    dataset[0, :] = numbers
    dataset[1, :] = 10 + numbers

    train = []
    val = []

    count = 0
    for i in range(10):
        dummy = count % 2
        val.append(dataset[dummy, i])
        train.append(dataset[1 - dummy, i])
        count = count + 1
    return np.array(train), np.array(val)  # ,np.array(test)


def GetRandomTrajectory(batch_size, motion):

    length = args.seq_length

    # image is 64x64, digit is 28x28 MNIST = 36
    canvas_size = args.image_size - args.digit_size

    # visualize and see
    # the starting point of the two numbers
    y = np.random.rand(args.batch_size)
    x = np.random.rand(args.batch_size)

    start_y = np.zeros((length, args.batch_size))  # 20x128 # 10x1
    start_x = np.zeros((length, args.batch_size))  # 20x128 # 10x1

    # get the velocity
    # theta = * 2 * np.pi
    if motion == 0:
        # 90 deg motion or vertical motion
        theta = np.ones(args.batch_size) * 0.5 * np.pi
    else:
        # 0 deg motion or horizontal motion
        theta = np.ones(args.batch_size) * 0 * np.pi

    # velocity in x and y directions
    v_y = 2 * np.sin(theta)
    v_x = 2 * np.cos(theta)

    for i in range(length):
        y += v_y * args.step_length
        x += v_x * args.step_length

        # Do not bounce off the edges
        for j in range(args.batch_size):
            if x[j] <= 0:

                x[j] = 0

                # bounce if hit the left wall
                v_x[j] = -v_x[j]

            if x[j] >= 1.0:

                x[j] = 1.0

                # bounce if hit the right wall
                v_x[j] = -v_x[j]
            if y[j] <= 0:

                y[j] = 0

                # bounce if hit the bottom wall
                v_y[j] = -v_y[j]

            if y[j] >= 1.0:

                y[j] = 1.0

                # bounce if hit the top wall
                v_y[j] = -v_y[j]

        start_y[i, :] = y
        start_x[i, :] = x

    # scale to the size of the canvas.
    start_y = (canvas_size * start_y).astype(np.int32)
    start_x = (canvas_size * start_x).astype(np.int32)

    return start_y.T, start_x.T


def Overlap(a, b):
    return np.maximum(a, b)

# function to render the final gif
def create_gif(digit_imgs, motion):

    # get an array of random numbers for indices
    # motion horizontal or vertical

    # gets random trajectory for the whole batch
    start_y, start_x = GetRandomTrajectory(args.batch_size, motion)
    #print (start_x.shape, start_y.shape)

    # define black pixels
    # 10x1x64x64
    gifs = np.zeros((args.batch_size, args.seq_length,
                     args.image_size, args.image_size), dtype=np.float32)
    #print (gifs.shape)

    for i in range(args.batch_size):

        for j in range(args.seq_length):

            #print (i,j)

            # use to select the position of image in a batch
            top = start_y[i, j]
            left = start_x[i, j]
            bottom = top + args.digit_size
            right = left + args.digit_size

            gifs[i, j, top:bottom, left:right] = digit_imgs[i, :, :]

    return gifs


def gif(filename, array, fps=10, scale=1.0):
    """Creates a gif given a stack of images using moviepy

    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e

    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)

    Parameters
    ----------
    filename : string
        The filename of the gif to write to

    array : array_like
        A numpy array that contains a sequence of images

    fps : int
        frames per second (default: 10)

    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension

    # print (array.shape)
    # start here
    # img = Image.fromarray(array[0], 'L')
    # img.save('x.png')
    # img.show()

    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    array = (array * 255.0).astype(np.uint8)
    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # number of images in one gif sequence
    parser.add_argument('--seq_length', default=20, type=int)

    # size of the canvas on which the digits float
    parser.add_argument('--image_size', default=64, type=int)

    # number of gifs in a batch
    parser.add_argument('--batch_size', default=10, type=int)

    # number of digits to be introduced on one canvas
    parser.add_argument('--num_digits', default=1, type=int)

    # the magnitude of movement between two consecutive frames
    parser.add_argument('--step_length', default=0.1, type=float)

    # size of the digits on the canvas
    parser.add_argument('--digit_size', default=28, type=int)

    global args
    args = parser.parse_args()

    # extract train and val data
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

    train_data, train_labels = train_set
    val_data, val_labels = valid_set

    train_data = train_data.reshape(-1, 28, 28)
    val_data = val_data.reshape(-1, 28, 28)

    data = np.concatenate((train_data, val_data), axis=0)

    # concatenate train and val data and labels

    labels = np.concatenate((train_labels, val_labels), axis=0)
    digits = data[:args.batch_size, :, :]

    gifs = create_gif(digits, 0)
    for i in range(gifs.shape[0]):
        filename = '/proj/vondrick/lovish/datasets/mnist/vertical/vertical_' + \
            str(i + 1) + '.gif'
        gif(filename, gifs[i], fps=10, scale=1.0)

    gifs = create_gif(digits, 1)
    for i in range(gifs.shape[0]):
        filename = '/proj/vondrick/lovish/datasets/mnist/horizontal/gif_' + \
            str(i + 1) + '.gif'
        gif(filename, gifs[i], fps=10, scale=1.0)

    # with h5py.File('mnist_single_gif.h5','w') as hf:

    #     hf.create_dataset('mnist_gif_train', data=data_train)
    #     hf.create_dataset('mnist_captions_train', data=captions_train)
    #     hf.create_dataset('mnist_count_train', data=count_train)
    #     hf.create_dataset('mnist_dataset_train', data=train)

    #     hf.create_dataset('mnist_gif_val', data=data_val)
    #     hf.create_dataset('mnist_captions_val', data=captions_val)
    #     hf.create_dataset('mnist_count_val', data=count_val)
    #     hf.create_dataset('mnist_dataset_val', data=val)
