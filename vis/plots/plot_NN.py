import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import argparse
import glob

parser = argparse.ArgumentParser()

parser.add_argument(
    '--class-names', default='/proj/vondrick/lovish/datasets/block_toy/classInd.txt', type=str)
parser.add_argument('--image-dir', default='/proj/vondrick/lovish/DPC/dpc/log_tmp/block_toy_imagenet_3-128_r10_dpc-rnn_bs64_lr0.001_seq6_pred2_len5_ds1_train-all/neighbours_ImageNet/00200/nearest_10', type=str)

args = parser.parse_args()

root_dir = args.image_dir

plt.rcParams.update({'font.size': 0})

w = 10
h = 10
fig = plt.figure(figsize=(20, 20))
columns = 11
rows = 10
for i in range(10):
    sub_dir = os.path.join(root_dir, str(i))
    j = 0
    filenames = [img for img in glob.glob("%s/*.jpg" % sub_dir)]
    filenames.sort()  # ADD THIS LINE
    for filename in filenames:
        print(filename)
        j += 1
        img_dir = os.path.join(sub_dir, filename)
        img = mpimg.imread(img_dir)
        fig.add_subplot(rows, columns, i * columns + j)
        plt.imshow(img)
results_dir = os.path.join(root_dir, 'merged.png')
plt.savefig(results_dir)
