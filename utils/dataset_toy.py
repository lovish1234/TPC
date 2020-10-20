# May 2020
# Merge all datasets in this file
# Block toy only

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


class block_toy(data.Dataset):
    '''
    mode: train or test split
    seq_len: number of frames in a video block
    num_seq: number of video block
    downsample: temporal downsample rate of frames
    num_sample: number of 'sequence of video blocks' sampled from one video
    drive: where the data is located
    num: which block toy tier to use
    '''

    def __init__(self,
                 num=1,
                 mode='train',
                 transform=None,
                 seq_len=1,
                 num_seq=2,
                 downsample=1,
                 drive='ssd',
                 num_sample=5):
        self.mode = mode
        self.num = num
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.drive = drive
        self.num_sample = num_sample

        # splits
        if mode == 'train':
            split = '/proj/vondrick/lovish/data/block_toy/block_toy_' + \
                str(num) + '/train_split_%s.csv' % self.drive
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):
            # use test for val, temporary
            split = '/proj/vondrick/lovish/data/block_toy/block_toy_' + \
                str(num) + '/test_split_%s.csv' % self.drive
            video_info = pd.read_csv(split, header=None)
        else:
            raise ValueError('wrong mode')

        # get action list [here just the values of y]
        self.action_dict_encode = {}
        self.action_dict_decode = {}

        action_file = os.path.join(
            '/proj/vondrick/lovish/data/block_toy', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id)  # let id start from 0
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # for k,v in self.action_dict_encode.items():
        #   print (k,v, type(k), type(v))
        # filter out too short videos
        # although we just require 2 sequences of length 1 here
        # practically no sequence will be dropped as there
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        # [part of the original code] why do we need it ?
        if mode == 'val':
            self.video_info = self.video_info.sample(frac=1.0)

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
            return [None]
        n = 1
        # if self.mode == 'test':
        #     # all possible frames with downsampling
        #     seq_idx_block = np.arange(0, vlen, self.downsample)
        #     return [seq_idx_block, vpath]

        start_idx = np.random.choice(
            range(1, vlen - self.num_seq * self.seq_len * self.downsample), n)
        #print ("start_idx:", start_idx)
        seq_idx = np.expand_dims(
            np.arange(self.num_seq), -1) * self.downsample * self.seq_len + start_idx
        #print ("seq_idx:", seq_idx)
        seq_idx_block = seq_idx + \
            np.expand_dims(np.arange(self.seq_len), 0) * self.downsample
        #print ("seq_idx_block:", seq_idx_block)
        return [seq_idx_block, vpath]

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index // self.num_sample]
        items = self.idx_sampler(vlen, vpath)
        if items is None:
            print(vpath)

        idx_block, vpath = items
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq * self.seq_len)

        #print ("idx_block, vpath: ", idx_block, vpath)
        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i + 1)))
               for i in idx_block]

        # do we need it here
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
        t_seq = t_seq.view(self.num_seq, self.seq_len,
                           C, H, W).transpose(1, 2)

        #print (vpath, vpath.split('/'))
        try:
            #print ('try', vpath, vpath.split('/'))
            vname = vpath.split('/')[-2]
            #print (vname)
            vid = self.encode_action(int(vname))
        except:
            #print ('except', vpath)
            vname = vpath.split('/')[-3]
            #print (vname)
            vid = self.encode_action(int(vname))

        label = torch.LongTensor([vid])

        # OLD: return sequence only
        # return t_seq
        # NEW: return all useful information in a dictionary
        result = {'t_seq': t_seq, 'idx_block': idx_block, 'vpath': vpath}
        return result

    def __len__(self):
        return len(self.video_info) * self.num_sample

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]


class block_toy_imagenet(data.Dataset):
    '''
    mode: train or test split
    seq_len: number of frames in a video block
    num_seq: number of video block
    downsample: temporal downsample rate of frames
    num_sample: number of 'sequence of video blocks' sampled from one video
    drive: where the data is located
    num: which block toy tier to use
    '''

    def __init__(self,
                 mode='train',
                 num=1,
                 transform=None,
                 seq_len=1,
                 num_seq=2,
                 downsample=1,
                 drive='ssd',
                 num_sample=5):
        self.mode = mode
        self.num = num
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.drive = drive
        self.num_sample = num_sample  # number of sequences sampled from one video

        # splits
        if mode == 'train':
            split = '/proj/vondrick/lovish/data/block_toy/block_toy_imagenet_' + \
                num + '/train_split_%s.csv' % self.drive
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):
            # use test for val, temporary
            split = '/proj/vondrick/lovish/data/block_toy/block_toy_imagenet_' + \
                num + '/test_split_%s.csv' % self.drive
            video_info = pd.read_csv(split, header=None)
        else:
            raise ValueError('wrong mode')

        # get action list [here just the values of y]
        self.action_dict_encode = {}
        self.action_dict_decode = {}

        action_file = os.path.join(
            '/proj/vondrick/lovish/data/block_toy', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id) - 1  # let id start from 0
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # for k,v in self.action_dict_encode.items():
        #   print (k,v, type(k), type(v))
        # filter out too short videos
        # although we just require 2 sequences of length 1 here
        # practically no sequence will be dropped as there
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        # [part of the original code] why do we need it ?
        if mode == 'val':
            self.video_info = self.video_info.sample(frac=1.0)

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
            return [None]
        n = 1
        # if self.mode == 'test':
        #     # all possible frames with downsampling
        #     seq_idx_block = np.arange(0, vlen, self.downsample)
        #     return [seq_idx_block, vpath]

        start_idx = np.random.choice(
            range(vlen - self.num_seq * self.seq_len * self.downsample), n)
        #print ("start_idx:", start_idx)
        seq_idx = np.expand_dims(
            np.arange(self.num_seq), -1) * self.downsample * self.seq_len + start_idx
        #print ("seq_idx:", seq_idx)
        seq_idx_block = seq_idx + \
            np.expand_dims(np.arange(self.seq_len), 0) * self.downsample
        #print ("seq_idx_block:", seq_idx_block)
        return [seq_idx_block, vpath]

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index // self.num_sample]
        items = self.idx_sampler(vlen, vpath)
        if items is None:
            print(vpath)

        idx_block, vpath = items
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq * self.seq_len)

        #print ("idx_block, vpath: ", idx_block, vpath)
        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i + 1)))
               for i in idx_block]

        # do we need it here
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
        t_seq = t_seq.view(self.num_seq, self.seq_len,
                           C, H, W).transpose(1, 2)

        #print (vpath, vpath.split('/'))
        try:
            #print ('try', vpath, vpath.split('/'))
            vname = vpath.split('/')[-2]
            #print (vname)
            vid = self.encode_action(int(vname))
        except:
            #print ('except', vpath)
            vname = vpath.split('/')[-3]
            #print (vname)
            vid = self.encode_action(int(vname))

        label = torch.LongTensor([vid])

        # OLD: return sequence only
        # return t_seq
        # NEW: return all useful information in a dictionary
        result = {'t_seq': t_seq, 'idx_block': idx_block, 'vpath': vpath}
        return result

    def __len__(self):
        return len(self.video_info) * self.num_sample

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]
