# May 2020
# Merge all datasets in this file
# Everything except Epic Kitchens & block toy

import cv2
import torch
from torch.utils import data
import os
import sys
import ast
from PIL import Image
from multiprocessing import Manager

import pandas as pd
import numpy as np

from augmentation import *
from tqdm import tqdm
from joblib import Parallel, delayed
from gulpio import GulpDirectory


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            result = img.convert('RGB')
            return result


class SingleVideoDataset(data.Dataset):
    ''' Dataset that iterates over a whole video file once.
    One returned item of this dataset represents a sequence of 8 blocks,
    and we keep returning items covering temporally shifted sequences within the video file. '''

    def __init__(self, file_path, transform=None, start_stride=3, num_seq=8, seq_len=5, downsample=3):
        ''' start_stride: Frame start number increment within the full video for every sequence.
        Example: if start_stride=2 and downsample=3, we first process frames (0, 3, 6, ..., 117) for item index = 0
        and then (2, 5, 8, ..., 119) for item index = 1 and so on. '''
        self.file_path = file_path
        self.transform = transform
        self.start_stride = start_stride
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample

        # Load video frames into memory
        self.frames = []  # PIL images to match other datasets
        cap = cv2.VideoCapture(file_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == False:
                break
            frame = frame[:, :, ::-1]  # BGR to RGB
            assert(frame.ndim == 3)
            assert(frame.shape[2] == 3)
            frame = Image.fromarray(frame)  # convert to PIL image
            self.frames.append(frame)
        cap.release()

        # self.frames = np.array(self.frames)
        # self.frame_count = self.frames.shape[0]
        self.frame_count = len(self.frames)
        manager = Manager()
        self.frames = manager.list(self.frames)

        # TODO review correctness of formula (there might be one-off mistake):
        self.num_items = (self.frame_count - num_seq *
                          seq_len * downsample) // start_stride

    def idx_sampler(self, start_idx):
        '''Sample a sequence of frame indices from a video. '''
        if self.frame_count - self.num_seq * self.seq_len * self.downsample <= 0:
            return None
        # Get 1D list of frame numbers representing START of every block
        seq_idx = np.expand_dims(
            np.arange(self.num_seq), -1) * self.downsample * self.seq_len + start_idx
        # Get 2D list of frame numbers of ALL frames within every block
        seq_idx_block = seq_idx + \
            np.expand_dims(np.arange(self.seq_len), 0) * self.downsample
        # Return list of frame indices to be used
        return seq_idx_block

    def __getitem__(self, index):
        start_idx = index * self.start_stride
        idx_block = self.idx_sampler(start_idx)

#         print(index, start_idx)
#         print(idx_block[0], idx_block[-1])
#         print(len(self.frames))

        # Decode list of frame numbers
        # dimensions = (block number, frame within block)
        assert(idx_block.shape == (self.num_seq, self.seq_len))
        # most significant = block number
        idx_block = idx_block.reshape(self.num_seq * self.seq_len)

        # Read all frames as list of numpy images
        seq = [self.frames[i] for i in idx_block]

        # print(len(seq), seq[0].shape)

        # Transform sequence of images (NOT the same for all frames!)
        t_seq = self.transform(seq)

        # Convert list of tensors into altogether tensor
        num_crop = None
        try:
            (C, H, W) = t_seq[0].size()
            t_seq = torch.stack(t_seq, 0)
        except:
            (C, H, W) = t_seq[0][0].size()
            tmp = [torch.stack(i, 0) for i in t_seq]
            assert len(tmp) == 5
            num_crop = 5
            t_seq = torch.stack(tmp, 1)

        # View as e.g. (8, 5, 3, 128, 128) and then transpose to (8, 3, 5, 128, 128)
        # Final dimensionality: block => color => frame => height => width
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1, 2)

        # NEW: return all useful information in a dictionary
        result = {'t_seq': t_seq, 'idx_block': idx_block,
                  'vpath': self.file_path}
        return result

    def __len__(self):
        return self.num_items


class Kinetics400_full_3d(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq=5,
                 downsample=3,
                 epsilon=5,
                 unit_test=False,
                 big=False,
                 drive='ssd'):
        self.mode = mode
        self.transform = transform
        self.drive = drive

        # number of frames in one pack
        self.seq_len = seq_len

        # number of sequences of frames
        self.num_seq = num_seq

        # choose the nth frame
        self.downsample = downsample

        # no use in the code
        self.epsilon = epsilon

        # check what this does
        self.unit_test = unit_test

        # which dataset to use
        if big:
            print('Using Kinetics400 full data (256x256)')
        else:
            print('Using Kinetics400 partial data (150x150)')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}

        # where to get the action_file
        action_file = os.path.join(
            '/proj/vondrick/lovish/data/kinetics-partial', 'classInd.txt')

        # read as a python dataframes
        action_df = pd.read_csv(action_file, sep='  ', header=None)

        # action name to id's and vice versa
        for _, row in action_df.iterrows():
            act_id, act_name = row

            # ids already start from 0.
            act_id = int(act_id) - 1  # let id start from 0
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # splits
        # change the directories
        if big:
            if mode == 'train':
                split = '/proj/vondrick/lovish/data/kinetics400_256/train_split_%s.csv' % self.drive
                video_info = pd.read_csv(split, header=None)
            elif (mode == 'val') or (mode == 'test'):
                split = '/proj/vondrick/lovish/data/kinetics400_256/val_split_%s.csv' % self.drive
                video_info = pd.read_csv(split, header=None)
            else:
                raise ValueError('wrong mode')
        else:  # small
            if mode == 'train':
                split = '/proj/vondrick/lovish/data/kinetics-partial/train_split_%s.csv' % self.drive
                video_info = pd.read_csv(split, header=None)
            elif (mode == 'val') or (mode == 'test'):
                split = '/proj/vondrick/lovish/data/kinetics-partial/val_split_%s.csv' % self.drive
                video_info = pd.read_csv(split, header=None)
            else:
                raise ValueError('wrong mode')

        # drop videos which are smaller than what we can afford
        drop_idx = []
        print('filter out too short videos ...')
        for idx, row in tqdm(video_info.iterrows(), total=len(video_info)):
            vpath, vlen = row

            # if number of frames less than 150 or 5 sec.
            if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val':
            self.video_info = self.video_info.sample(
                frac=0.3, random_state=666)
        if self.unit_test:
            self.video_info = self.video_info.sample(32, random_state=666)
        # shuffle not necessary because use RandomSampler

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''

        # if video too short return None
        if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
            return [None]

        n = 1
        # choose a start frame
        start_idx = np.random.choice(
            range(vlen - self.num_seq * self.seq_len * self.downsample), n)

        # get a sequence of frames to start with
        seq_idx = np.expand_dims(
            np.arange(self.num_seq), -1) * self.downsample * self.seq_len + start_idx

        # indices of consecutive frames
        seq_idx_block = seq_idx + \
            np.expand_dims(np.arange(self.seq_len), 0) * self.downsample
        return [seq_idx_block, vpath]

    def __getitem__(self, index):

        # get the path and number of frames in the video
        vpath, vlen = self.video_info.iloc[index]

        # sample the starting frame randomly
        items = self.idx_sampler(vlen, vpath)

        if items is None:
            print(vpath)

        # takes vpath, returns vpath
        idx_block, vpath = items

        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq * self.seq_len)

        # load all the frames needed to complete the sequence
        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i + 1)))
               for i in idx_block]
        t_seq = self.transform(seq)  # apply same transform

        # each part is a RGB image
        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1, 2)

        try:
            vname = vpath.split('/')[-3]
            vid = self.encode_action(vname)
        except:
            vname = vpath.split('/')[-2]
            vid = self.encode_action(vname)
        label = torch.LongTensor([vid])

        # NEW: return all useful information in a dictionary
        result = {'t_seq': t_seq, 'label': label,
                  'idx_block': idx_block, 'vpath': vpath}
        return result

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]


class UCF101_3d(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq=5,
                 downsample=3,
                 epsilon=5,
                 which_split=1,
                 drive='ssd'):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.which_split = which_split
        self.drive = drive

        # splits
        if mode == 'train':
            split = '/proj/vondrick/lovish/data/ucf101/train_split%02d_split_%s.csv' % (
                self.which_split, self.drive)
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):  # use val for test
            split = '/proj/vondrick/lovish/data/ucf101/test_split%02d_split_%s.csv' % (
                self.which_split, self.drive)
            video_info = pd.read_csv(split, header=None)
        else:
            raise ValueError('wrong mode')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}
        action_file = os.path.join(
            '/proj/vondrick/lovish/data/ucf101', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row

            # NOTE: dataset_3d_lc.py contained this line:
            act_id = int(act_id) - 1  # let id start from 0

            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val':
            self.video_info = self.video_info.sample(frac=0.3)
        # shuffle not required due to external sampler

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
            return [None]
        n = 1

        # From LC:
        if self.mode == 'test':
            # all possible frames with downsampling
            seq_idx_block = np.arange(0, vlen, self.downsample)
            return [seq_idx_block, vpath]

        start_idx = np.random.choice(
            range(vlen - self.num_seq * self.seq_len * self.downsample), n)
        seq_idx = np.expand_dims(
            np.arange(self.num_seq), -1) * self.downsample * self.seq_len + start_idx
        seq_idx_block = seq_idx + \
            np.expand_dims(np.arange(self.seq_len), 0) * self.downsample
        return [seq_idx_block, vpath]

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        items = self.idx_sampler(vlen, vpath)
        if items is None:
            print(vpath)

        idx_block, vpath = items
        if self.mode != 'test':
            assert idx_block.shape == (self.num_seq, self.seq_len)
            idx_block = idx_block.reshape(self.num_seq * self.seq_len)

        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i + 1)))
               for i in idx_block]
        t_seq = self.transform(seq)  # apply same transform

        # From LC:
        num_crop = None
        try:
            (C, H, W) = t_seq[0].size()
            t_seq = torch.stack(t_seq, 0)
        except:
            (C, H, W) = t_seq[0][0].size()
            tmp = [torch.stack(i, 0) for i in t_seq]
            assert len(tmp) == 5
            num_crop = 5
            t_seq = torch.stack(tmp, 1)
        if self.mode == 'test':
            # return all available clips, but cut into length = num_seq
            SL = t_seq.size(0)
            clips = []
            i = 0
            while i + self.seq_len <= SL:
                clips.append(t_seq[i:i + self.seq_len, :])
                # i += self.seq_len//2
                i += self.seq_len
            if num_crop:
                # half overlap:
                clips = [torch.stack(clips[i:i + self.num_seq], 0).permute(2, 0, 3, 1, 4, 5)
                         for i in range(0, len(clips) + 1 - self.num_seq, self.num_seq // 2)]
                NC = len(clips)
                t_seq = torch.stack(clips, 0).view(
                    NC * num_crop, self.num_seq, C, self.seq_len, H, W)
            else:
                # half overlap:
                clips = [torch.stack(clips[i:i + self.num_seq], 0).transpose(1, 2)
                         for i in range(0, len(clips) + 1 - self.num_seq, self.num_seq // 2)]
                t_seq = torch.stack(clips, 0)
        else:
            t_seq = t_seq.view(self.num_seq, self.seq_len,
                               C, H, W).transpose(1, 2)

        # Collect label
        try:
            vname = vpath.split('/')[-3]
            vid = self.encode_action(vname)
        except:
            vname = vpath.split('/')[-2]
            vid = self.encode_action(vname)
        label = torch.LongTensor([vid])

        # NEW: return all useful information in a dictionary
        result = {'t_seq': t_seq, 'label': label,
                  'idx_block': idx_block, 'vpath': vpath}
        return result

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return action code'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]


class UCF11_3d(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq=5,
                 downsample=3,
                 epsilon=5,
                 which_split=1,
                 drive='ssd'):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.which_split = which_split
        self.drive = drive

        # splits
        if mode == 'train':
            split = '/proj/vondrick/lovish/data/ucf11/train_split_%s.csv' % self.drive
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):  # use val for test
            split = '/proj/vondrick/lovish/data/ucf11/test_split_%s.csv' % self.drive
            video_info = pd.read_csv(split, header=None)
        else:
            raise ValueError('wrong mode')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}
        action_file = os.path.join(
            '/proj/vondrick/lovish/data/ucf11', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id)  # let id start from 0
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        # if mode == 'val': self.video_info = self.video_info.sample(frac=0.3)
        # shuffle not required due to external sampler

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
            return [None]
        n = 1

        if self.mode == 'test':
            # all possible frames with downsampling
            seq_idx_block = np.arange(0, vlen, self.downsample)
            return [seq_idx_block, vpath]

        start_idx = np.random.choice(
            range(vlen - self.num_seq * self.seq_len * self.downsample), n)
        seq_idx = np.expand_dims(
            np.arange(self.num_seq), -1) * self.downsample * self.seq_len + start_idx
        seq_idx_block = seq_idx + \
            np.expand_dims(np.arange(self.seq_len), 0) * self.downsample
        return [seq_idx_block, vpath]

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        items = self.idx_sampler(vlen, vpath)
        if items is None:
            print(vpath)

        idx_block, vpath = items
        if self.mode != 'test':
            assert idx_block.shape == (self.num_seq, self.seq_len)
            idx_block = idx_block.reshape(self.num_seq * self.seq_len)

        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i + 1)))
               for i in idx_block]
        t_seq = self.transform(seq)  # apply same transform

        num_crop = None
        try:
            (C, H, W) = t_seq[0].size()
            t_seq = torch.stack(t_seq, 0)
        except:
            (C, H, W) = t_seq[0][0].size()
            tmp = [torch.stack(i, 0) for i in t_seq]
            assert len(tmp) == 5
            num_crop = 5
            t_seq = torch.stack(tmp, 1)

        if self.mode == 'test':
            # return all available clips, but cut into length = num_seq
            SL = t_seq.size(0)
            clips = []
            i = 0
            while i + self.seq_len <= SL:
                clips.append(t_seq[i:i + self.seq_len, :])
                # i += self.seq_len//2
                i += self.seq_len
            if num_crop:
                # half overlap:
                clips = [torch.stack(clips[i:i + self.num_seq], 0).permute(2, 0, 3, 1, 4, 5)
                         for i in range(0, len(clips) + 1 - self.num_seq, self.num_seq // 2)]
                NC = len(clips)
                t_seq = torch.stack(clips, 0).view(
                    NC * num_crop, self.num_seq, C, self.seq_len, H, W)
            else:
                # half overlap:
                clips = [torch.stack(clips[i:i + self.num_seq], 0).transpose(1, 2)
                         for i in range(0, len(clips) + 1 - self.num_seq, self.num_seq // 2)]
                t_seq = torch.stack(clips, 0)
        else:
            t_seq = t_seq.view(self.num_seq, self.seq_len,
                               C, H, W).transpose(1, 2)

        try:
            vname = vpath.split('/')[-3]
            vid = self.encode_action(vname)
        except:
            vname = vpath.split('/')[-2]
            vid = self.encode_action(vname)
        label = torch.LongTensor([vid])

        # NEW: return all useful information in a dictionary
        result = {'t_seq': t_seq, 'label': label,
                  'idx_block': idx_block, 'vpath': vpath}
        return result

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return action code'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]


class HMDB51_3d(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq=1,
                 downsample=1,
                 epsilon=5,
                 which_split=1):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.which_split = which_split

        # splits
        if mode == 'train':
            split = '/proj/vondrick/lovish/data/hmdb51/train_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):
            # use test for val, temporary
            split = '/proj/vondrick/lovish/data/hmdb51/test_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
        else:
            raise ValueError('wrong mode')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}

        action_file = os.path.join(
            '/proj/vondrick/lovish/data/hmdb51', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep='  ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id) - 1  # let id start from 0
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val':
            self.video_info = self.video_info.sample(frac=0.3)
        # shuffle not required

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
            return [None]
        n = 1
        if self.mode == 'test':
            # all possible frames with downsampling
            seq_idx_block = np.arange(0, vlen, self.downsample)
            return [seq_idx_block, vpath]
        start_idx = np.random.choice(
            range(vlen - self.num_seq * self.seq_len * self.downsample), n)
        seq_idx = np.expand_dims(
            np.arange(self.num_seq), -1) * self.downsample * self.seq_len + start_idx
        seq_idx_block = seq_idx + \
            np.expand_dims(np.arange(self.seq_len), 0) * self.downsample
        return [seq_idx_block, vpath]

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        items = self.idx_sampler(vlen, vpath)
        if items is None:
            print(vpath)

        idx_block, vpath = items
        if self.mode != 'test':
            assert idx_block.shape == (self.num_seq, self.seq_len)
            idx_block = idx_block.reshape(self.num_seq * self.seq_len)

        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i + 1)))
               for i in idx_block]
        t_seq = self.transform(seq)  # apply same transform

        num_crop = None
        try:
            (C, H, W) = t_seq[0].size()
            t_seq = torch.stack(t_seq, 0)
        except:
            (C, H, W) = t_seq[0][0].size()
            tmp = [torch.stack(i, 0) for i in t_seq]
            assert len(tmp) == 5
            num_crop = 5
            t_seq = torch.stack(tmp, 1)
        # print(t_seq.size())
        # import ipdb; ipdb.set_trace()
        if self.mode == 'test':
            # return all available clips, but cut into length = num_seq
            SL = t_seq.size(0)
            clips = []
            i = 0
            while i + self.seq_len <= SL:
                clips.append(t_seq[i:i + self.seq_len, :])
                # i += self.seq_len//2
                i += self.seq_len
            if num_crop:
                # half overlap:
                clips = [torch.stack(clips[i:i + self.num_seq], 0).permute(2, 0, 3, 1, 4, 5)
                         for i in range(0, len(clips) + 1 - self.num_seq, self.num_seq // 2)]
                NC = len(clips)
                t_seq = torch.stack(clips, 0).view(
                    NC * num_crop, self.num_seq, C, self.seq_len, H, W)
            else:
                # half overlap:
                clips = [torch.stack(clips[i:i + self.num_seq], 0).transpose(1, 2)
                         for i in range(0, len(clips) + 1 - self.num_seq, 3 * self.num_seq // 4)]
                t_seq = torch.stack(clips, 0)
        else:
            t_seq = t_seq.view(self.num_seq, self.seq_len,
                               C, H, W).transpose(1, 2)

        try:
            vname = vpath.split('/')[-3]
            vid = self.encode_action(vname)
        except:
            vname = vpath.split('/')[-2]
            vid = self.encode_action(vname)

        label = torch.LongTensor([vid])

        # NEW: return all useful information in a dictionary
        result = {'t_seq': t_seq, 'label': label,
                  'idx_block': idx_block, 'vpath': vpath}
        return result

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]
