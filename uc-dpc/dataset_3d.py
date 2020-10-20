# get dataset in 3-d format

#### OBSOLETE DO NOT USE ####

import torch
from torch.utils import data
from torchvision import transforms
import os
import sys

import pandas as pd
import numpy as np
sys.path.append('../utils')

# augmentation tools
from augmentation import *

# bookkeeping tools
from tqdm import tqdm

# parallalization tools
from joblib import Parallel, delayed


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class epic(data.Dataset):
    def __init__(self,
                 num=1,
                 mode='train',
                 transform=None,
                 seq_len=1,
                 num_seq=2,
                 downsample=1):
        self.mode = mode
        self.num = num
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample

        # splits
        if mode == 'train':
            split = '/proj/vondrick/lovish/data/epic-kitchen' + '/train_split.csv'
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):
            # use test for val, temporary
            split = '/proj/vondrick/lovish/data/epic-kitchen' + '/test_split.csv'
            video_info = pd.read_csv(split, header=None)
        else:
            raise ValueError('wrong mode')

        # get action list [here just the values of y]
        self.action_dict_encode = {}
        self.action_dict_decode = {}

        action_file = os.path.join(
            '/proj/vondrick/datasets/epic-kitchens/data/annotations', 'EPIC_verb_classes.csv')
        action_df = pd.read_csv(action_file, sep=',', header=0)
        for _, row in action_df.iterrows():
            act_id, act_name = row[0], row[1]
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
            self.video_info = self.video_info.sample(frac=0.2)

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
        vpath, vlen = self.video_info.iloc[int(index / 20)]
        items = self.idx_sampler(vlen, vpath)
        if items is None:
            print(vpath)

        idx_block, vpath = items
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq * self.seq_len)

        #print ("idx_block, vpath: ", idx_block, vpath)
        seq = [pil_loader(os.path.join(vpath, 'frame_%010d.jpg' % (i + 1)))
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

        return t_seq

    def __len__(self):
        return len(self.video_info) * 20

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]        

class block_toy(data.Dataset):
    def __init__(self,
                 num=1,
                 mode='train',
                 transform=None,
                 seq_len=1,
                 num_seq=2,
                 downsample=1):
        self.mode = mode
        self.num = num
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample

        # splits
        if mode == 'train':
            split = '/proj/vondrick/lovish/data/block_toy/block_toy_' + num + '/train_split.csv'
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):
            # use test for val, temporary
            split = '/proj/vondrick/lovish/data/block_toy/block_toy_' + num + '/test_split.csv'
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
        vpath, vlen = self.video_info.iloc[index]
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

        return t_seq

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]


class block_toy_imagenet(data.Dataset):
    def __init__(self,
                 mode='train',
                 num=1,
                 transform=None,
                 seq_len=1,
                 num_seq=2,
                 downsample=1):
        self.mode = mode
        self.num = num
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample

        # splits
        if mode == 'train':
            split = '/proj/vondrick/lovish/data/block_toy/block_toy_imagenet_'+ num +'/train_split.csv'
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):
            # use test for val, temporary
            split = '/proj/vondrick/lovish/data/block_toy/block_toy_imagenet_'+ num +'/test_split.csv'
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
        vpath, vlen = self.video_info.iloc[index]
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

        return t_seq

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]


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
                 return_label=False):
        self.mode = mode
        self.transform = transform

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

        # return video label along with frames
        self.return_label = return_label

        # which dataset to use
        if big:
            print('Using Kinetics400 full data (256x256)')
        else:
            print('Using Kinetics400 full data (150x150)')

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
                split = '/proj/vondrick/lovish/data/kinetics400_256/train_split.csv'
                video_info = pd.read_csv(split, header=None)
            elif (mode == 'val') or (mode == 'test'):
                split = '/proj/vondrick/lovish/data/kinetics400_256/val_split.csv'
                video_info = pd.read_csv(split, header=None)
            else:
                raise ValueError('wrong mode')
        else:  # small
            if mode == 'train':
                split = '/proj/vondrick/lovish/data/kinetics-partial/train_split.csv'
                video_info = pd.read_csv(split, header=None)
            elif (mode == 'val') or (mode == 'test'):
                split = '/proj/vondrick/lovish/data/kinetics-partial/val_split.csv'
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

        if self.return_label:
            try:
                vname = vpath.split('/')[-3]
                vid = self.encode_action(vname)
            except:
                vname = vpath.split('/')[-2]
                vid = self.encode_action(vname)

            label = torch.LongTensor([vid])
            return t_seq, label

        return t_seq

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
                 return_label=False):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq

        self.downsample = downsample
        self.epsilon = epsilon

        self.which_split = which_split
        self.return_label = return_label

        # splits
        if mode == 'train':
            split = '/proj/vondrick/lovish/data/ucf101/train_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):  # use val for test
            split = '/proj/vondrick/lovish/data/ucf101/test_split%02d.csv' % self.which_split
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
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq * self.seq_len)

        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i + 1)))
               for i in idx_block]
        t_seq = self.transform(seq)  # apply same transform

        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1, 2)

        if self.return_label:
            try:
                vname = vpath.split('/')[-3]
                vid = self.encode_action(vname)
            except:
                vname = vpath.split('/')[-2]
                vid = self.encode_action(vname)
            label = torch.LongTensor([vid])
            return t_seq, label

        return t_seq

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
                 return_label=False):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq

        self.downsample = downsample
        self.epsilon = epsilon

        self.which_split = which_split
        self.return_label = return_label

        # splits
        if mode == 'train':
            split = '/proj/vondrick/lovish/data/ucf11/train_split.csv'
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):  # use val for test
            split = '/proj/vondrick/lovish/data/ucf11/test_split.csv'
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
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq * self.seq_len)

        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i + 1)))
               for i in idx_block]
        t_seq = self.transform(seq)  # apply same transform

        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1, 2)

        if self.return_label:
            try:
                vname = vpath.split('/')[-3]
                vid = self.encode_action(vname)
            except:
                vname = vpath.split('/')[-2]
                vid = self.encode_action(vname)
            label = torch.LongTensor([vid])
            return t_seq, label

        return t_seq

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return action code'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]
