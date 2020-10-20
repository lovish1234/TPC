import csv
import pandas as pd
import argparse
import os


def create_cls_file(path, filename):
    dirs = os.listdir(path)
    file_handle = open(filename, 'w')
    count = 0

    for files in dirs:
        file_handle.write("%d  %s \n" % (count, files))
        count += 1

    file_handle.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--folder', default='/proj/vondrick/lovish/datasets/Kinetics400/videos/val_split', type=str)
    parser.add_argument(
        '--filename', default='/proj/vondrick/lovish/datasets/Kinetics400/classInd.txt', type=str)

    args = parser.parse_args()
    create_cls_file(args.folder, args.filename)
