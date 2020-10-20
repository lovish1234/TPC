import os
import csv
import glob
from tqdm import tqdm
import pandas as pd
from joblib import delayed, Parallel
import argparse


def write_list(data_list, path, ):
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for row in data_list:
            if row:
                writer.writerow(row)
    print('split saved to %s' % path)


def main_UCF101(f_root, splits_root, csv_root='../data/ucf101/'):
    '''generate training/testing split, count number of available frames, save in csv'''
    if not os.path.exists(csv_root):
        os.makedirs(csv_root)
    for which_split in [1, 2, 3]:

        train_set = []
        test_set = []
        train_split_file = os.path.join(
            splits_root, 'trainlist%02d.txt' % which_split)
        with open(train_split_file, 'r') as f:
            for line in f:
                vpath = os.path.join(f_root, line.split(' ')[0][0:-4]) + '/'

                train_set.append(
                    [vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])

        test_split_file = os.path.join(
            splits_root, 'testlist%02d.txt' % which_split)
        with open(test_split_file, 'r') as f:
            for line in f:
                vpath = os.path.join(f_root, line.rstrip()[0:-4]) + '/'
                test_set.append(
                    [vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])

        write_list(train_set, os.path.join(
            csv_root, 'train_split%02d.csv' % which_split))
        write_list(test_set, os.path.join(
            csv_root, 'test_split%02d.csv' % which_split))


def main_UCF11(f_root, splits_root, csv_root='../data/ucf11/'):
    '''generate training/testing split, count number of available frames, save in csv'''
    if not os.path.exists(csv_root):
        os.makedirs(csv_root)
    train_set = []
    test_set = []
    train_split_file = os.path.join(splits_root, 'train_map.csv')
    with open(train_split_file, 'r') as f:
        for line in f:
            # print(line)
            vpath = os.path.join(f_root, line.split(',')[0][0:-4]) + '/'
            train_set.append(
                [vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])

    test_split_file = os.path.join(splits_root, 'test_map.csv')
    with open(test_split_file, 'r') as f:
        for line in f:
            vpath = os.path.join(f_root, line.split(',')[0][0:-4]) + '/'
            test_set.append(
                [vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])

    write_list(train_set, os.path.join(csv_root, 'train_split.csv'))
    write_list(test_set, os.path.join(csv_root, 'test_split.csv'))


def main_block(f_root, splits_root, csv_root='../data/block_toy/'):
    '''generate training/testing split, count number of available frames, save in csv'''
    if not os.path.exists(csv_root):
        os.makedirs(csv_root)
    train_set = []
    test_set = []
    train_split_file = os.path.join(splits_root, 'train_map.csv')
    # print('here')
    with open(train_split_file, 'r') as f:
        for line in f:
            # print(line)
            vpath = os.path.join(f_root, line.split(',')[0][0:-4]) + '/'
            train_set.append(
                [vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])

    test_split_file = os.path.join(splits_root, 'test_map.csv')
    with open(test_split_file, 'r') as f:
        for line in f:
            vpath = os.path.join(f_root, line.split(',')[0][0:-4]) + '/'
            test_set.append(
                [vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])

    write_list(train_set, os.path.join(csv_root, 'train_split.csv'))
    write_list(test_set, os.path.join(csv_root, 'test_split.csv'))


def main_HMDB51(f_root, splits_root, csv_root='../data/hmdb51/'):
    '''generate training/testing split, count number of available frames, save in csv'''
    if not os.path.exists(csv_root):
        os.makedirs(csv_root)
    for which_split in [1, 2, 3]:
        print('Here')
        train_set = []
        test_set = []
        split_files = sorted(glob.glob(os.path.join(
            splits_root, '*_test_split%d.txt' % which_split)))
        assert len(split_files) == 51
        for split_file in split_files:
            action_name = os.path.basename(split_file)[0:-16]
            with open(split_file, 'r') as f:
                for line in f:
                    video_name = line.split(' ')[0]
                    _type = line.split(' ')[1]
                    vpath = os.path.join(
                        f_root, action_name, video_name[0:-4]) + '/'
                    if _type == '1':
                        #print (vpath)
                        #print (len(glob.glob(os.path.join(vpath, '*.jpg'))))
                        train_set.append(
                            [vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])
                    elif _type == '2':
                        #print (len(glob.glob(os.path.join(vpath, '*.jpg'))))
                        test_set.append(
                            [vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])

        write_list(train_set, os.path.join(
            csv_root, 'train_split%02d.csv' % which_split))
        write_list(test_set, os.path.join(
            csv_root, 'test_split%02d.csv' % which_split))

### For Kinetics ###


def get_split(root, split_path, mode):
    print('processing %s split ...' % mode)
    print('checking %s' % root)
    split_list = []

    #print ('split_path '+split_path)
    #print ('root '+root)

    split_content = pd.read_csv(split_path).iloc[:, 0:4]
    split_list = Parallel(n_jobs=64)(delayed(check_exists)(row, root)
                                     for i, row in tqdm(split_content.iterrows(), total=len(split_content)))
    return split_list


def check_exists(row, root):

    if args.dataset == 'kinetics-partial':
        dirname = '_'.join(['25fps' + row['youtube_id'], '%06d' %
                            row['time_start'], '%06d' % row['time_end']])
    elif args.dataset == 'kinetics':
        dirname = '_'.join([row['youtube_id'], '%06d' %
                            row['time_start'], '%06d' % row['time_end']])

    full_dirname = os.path.join(root, row['label'], dirname)
    #print (full_dirname)

    if os.path.exists(full_dirname):
        #print ("Yes")
        n_frames = len(glob.glob(os.path.join(full_dirname, '*.jpg')))
        return [full_dirname, n_frames]
    else:
        #print ("No")
        return None


def main_Kinetics400(mode, k400_path, f_root, csv_root='/proj/vondrick/lovish/data/kinetics'):
    train_split_path = os.path.join(csv_root, 'kinetics_train.csv')
    val_split_path = os.path.join(csv_root, 'kinetics_val.csv')
    test_split_path = os.path.join(csv_root, 'kinetics_test.csv')

    if not os.path.exists(csv_root):
        os.makedirs(csv_root)
    if mode == 'train':
        train_split = get_split(os.path.join(
            f_root, 'train_split'), train_split_path, 'train')
        write_list(train_split, os.path.join(csv_root, 'train_split.csv'))
    elif mode == 'val':
        val_split = get_split(os.path.join(
            f_root, 'val_split'), val_split_path, 'val')
        print(val_split)
        write_list(val_split, os.path.join(csv_root, 'val_split.csv'))
    elif mode == 'test':
        test_split = get_split(f_root, test_split_path, 'test')
        write_list(test_split, os.path.join(csv_root, 'test_split.csv'))
    else:
        raise IOError('wrong mode')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='hmdb51', type=str)
    args = parser.parse_args()

    # f_root is the frame path
    # splits root is the path where train and test splits are available
    # filename and number of frames
    # edit 'your_path' here:
    # main_block(f_root='/proj/vondrick/lovish/datasets/block_toy/block_toy_1/frame', 
    #             splits_root='/proj/vondrick/lovish/datasets/block_toy/block_toy_1/splits_classification',
    #             csv_root='/proj/vondrick/lovish/data/block_toy/block_toy_1/')
    # main_block(f_root='/proj/vondrick/lovish/datasets/block_toy/block_toy_2/frame', 
    #             splits_root='/proj/vondrick/lovish/datasets/block_toy/block_toy_2/splits_classification',
    #             csv_root='/proj/vondrick/lovish/data/block_toy/block_toy_2/')
    # main_block(f_root='/proj/vondrick/lovish/datasets/block_toy/block_toy_3/frame', 
    #             splits_root='/proj/vondrick/lovish/datasets/block_toy/block_toy_3/splits_classification',
    #             csv_root='/proj/vondrick/lovish/data/block_toy/block_toy_3/')
    # main_block(f_root='/proj/vondrick/lovish/datasets/block_toy/block_toy_imagenet_1/frame', 
    #             splits_root='/proj/vondrick/lovish/datasets/block_toy/block_toy_imagenet_1/splits_classification',
    #             csv_root='/proj/vondrick/lovish/data/block_toy/block_toy_imagenet_1/')
    # main_block(f_root='/proj/vondrick/lovish/datasets/block_toy/block_toy_imagenet_2/frame', 
    #             splits_root='/proj/vondrick/lovish/datasets/block_toy/block_toy_imagenet_2/splits_classification',
    #             csv_root='/proj/vondrick/lovish/data/block_toy/block_toy_imagenet_2/')
    # main_block(f_root='/proj/vondrick/lovish/datasets/block_toy/block_toy_imagenet_3/frame', 
    #             splits_root='/proj/vondrick/lovish/datasets/block_toy/block_toy_imagenet_3/splits_classification',
    #             csv_root='/proj/vondrick/lovish/data/block_toy/block_toy_imagenet_3/')

    main_block(f_root='/proj/vondrick/lovish/datasets/block_toy/block_toy_combined_test/frame', 
                splits_root='/proj/vondrick/lovish/datasets/block_toy/block_toy_combined_test/splits_classification',
                csv_root='/proj/vondrick/lovish/data/block_toy/block_toy_combined_test/')
    
    # main_UCF101(f_root='/proj/vondrick/lovish/datasets/ucf101/frame',
    #             splits_root='/proj/vondrick/lovish/datasets/ucf101/splits_classification')
    # main_UCF11(f_root='/proj/vondrick/lovish/datasets/ucf11/frame', 
    #             splits_root='/proj/vondrick/lovish/datasets/ucf11/splits_classification',
    #             csv_root='/proj/vondrick/lovish/data/ucf11')
    # main_block(f_root='/proj/vondrick/lovish/datasets/block_toy/frame', 
    #             splits_root='/proj/vondrick/lovish/datasets/block_toy/splits_classification',
    #             csv_root='/proj/vondrick/lovish/data/block_toy')
    # main_block(f_root='/proj/vondrick/lovish/datasets/block_toy_stochastic/frame', 
    #             splits_root='/proj/vondrick/lovish/datasets/block_toy_stochastic/splits_classification',
    #             csv_root='/proj/vondrick/lovish/data/block_toy_stochastic')
    # main_block(f_root='/proj/vondrick/lovish/datasets/block_toy_cosine/frame', 
    #             splits_root='/proj/vondrick/lovish/datasets/block_toy_cosine/splits_classification',
    #             csv_root='/proj/vondrick/lovish/data/block_toy_cosine')
    # main_HMDB51(f_root='/proj/vondrick/lovish/datasets/hmdb51/frame',
    #             splits_root='/proj/vondrick/lovish/datasets/hmdb51/split/testTrainMulti_7030_splits')
    # main_Kinetics400(mode='train',  # train or val or test
    #                  k400_path='/proj/vondrick/lovish/datasets/kinetics-partial',
    #                  f_root='/proj/vondrick/lovish/datasets/kinetics-partial/frame')
    # main_Kinetics400(mode='val',  # train or val or test
    #                  k400_path='/proj/vondrick/lovish/datasets/kinetics-partial',
    #                  f_root='/proj/vondrick/lovish/datasets/kinetics-partial/frame')
    # main_Kinetics400(mode='train',  # train or val or test
    #                  k400_path='/proj/vondrick/lovish/datasets/kinetics',
    #                  f_root='/proj/vondrick/lovish/datasets/kinetics/frame')
    # main_Kinetics400(mode='val',  # train or val or test
    #                  k400_path='/proj/vondrick/lovish/datasets/kinetics',
    #                  f_root='/proj/vondrick/lovish/datasets/kinetics/frame')

    # main_Kinetics400(mode='train', # train or val or test
    #                  k400_path='your_path/Kinetics',
    #                  f_root='your_path/Kinetics400_256/frame',
    #                  csv_root='../data/kinetics400_256')
