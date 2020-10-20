# Difference with tpc/dataset_3d: sample method = action based

#############################################################################
#### OBSOLETE DO NOT USE, see utils/dataset_epic.py and dataset_other.py ####
#############################################################################

import torch
from torch.utils import data
from torchvision import transforms
import os
import sys
import time
import pickle
import csv
import glob
import pandas as pd
import numpy as np
import cv2
import ast

sys.path.append('../utils')
from augmentation import *
from tqdm import tqdm
from joblib import Parallel, delayed

from gulpio import GulpDirectory
from epic_kitchens.dataset.epic_dataset import EpicVideoDataset, EpicVideoFlowDataset, GulpVideoSegment


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class epic_antiact(data.Dataset):
    '''
    Action anticipation Epic Kitchens dataset.
    '''

    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=1,
                 num_seq=2,
                 downsample=1,
                 class_type='verb'):
        print('WARNING: using obsolete dataset class')
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.class_type = class_type

        self.train_set = '/proj/vondrick/datasets/epic-kitchens/data/annotations/train_val/EPIC_train_action_labels.csv'
        self.val_set = '/proj/vondrick/datasets/epic-kitchens/data/annotations/train_val/EPIC_val_action_labels.csv'

        self.verb_classes = '/proj/vondrick/datasets/epic-kitchens/data/annotations/EPIC_verb_classes.csv'
        self.noun_classes = '/proj/vondrick/datasets/epic-kitchens/data/annotations/EPIC_noun_classes.csv'

        self.raw_data_path = '/proj/vondrick/datasets/epic-kitchens/data/raw/rgb'
        self.test_seen_set = '/proj/vondrick/datasets/epic-kitchens/data/annotations/EPIC_test_s1_timestamps.csv'
        self.test_unseen_set = '/proj/vondrick/datasets/epic-kitchens/data/annotations/EPIC_test_s2_timestamps.csv'

        train_df = pd.read_csv(self.train_set)
        val_df = pd.read_csv(self.val_set)
        self.verbs_df = pd.read_csv(self.verb_classes)
        self.nouns_df = pd.read_csv(self.noun_classes)
        verbs_map = {}
        for index, row in self.verbs_df.iterrows():
            verbs_map[row['class_key']] = ast.literal_eval(row['verbs'])
        nouns_map = {}
        for index, row in self.nouns_df.iterrows():
            nouns_map[row['class_key']] = ast.literal_eval(row['nouns'])
        train_df_replaced = train_df.copy()
        val_df_replaced = val_df.copy()
        for key in verbs_map.keys():
            train_df_replaced['verb'] = train_df_replaced['verb'].replace(
                verbs_map[key], key)
            val_df_replaced['verb'] = val_df_replaced['verb'].replace(
                verbs_map[key], key)
        for key in nouns_map.keys():
            train_df_replaced['noun'] = train_df_replaced['noun'].replace(
                nouns_map[key], key)
            val_df_replaced['noun'] = val_df_replaced['noun'].replace(
                nouns_map[key], key)

        actions_train = train_df_replaced
        actions_val = val_df_replaced
        test_seen_time = pd.read_csv(self.test_seen_set)
        test_unseen_time = pd.read_csv(self.test_unseen_set)

        # splits
        if mode == 'train':
            video_info = actions_train
        elif mode == 'val':
            video_info = actions_val
        elif mode == 'test_seen':
            video_info = test_seen_time
        elif mode == 'test_unseen':
            video_info = test_unseen_time
        else:
            raise ValueError('wrong mode')

        drop_idx = []
        for idx, row in video_info.iterrows():
            if row['start_frame'] < num_seq * seq_len * downsample + 60:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

    def idx_sampler(self, row):
        if row['start_frame'] < self.num_seq * self.seq_len * self.downsample:
            return [None]
        start_idx = row['start_frame'] - self.num_seq * \
            self.seq_len * self.downsample + (self.downsample - 1) - 60
        #print ("start_idx:", start_idx)
        seq_idx = np.expand_dims(
            np.arange(self.num_seq), -1) * self.downsample * self.seq_len + start_idx
        #print ("seq_idx:", seq_idx)
        seq_idx_block = seq_idx + \
            np.expand_dims(np.arange(self.seq_len), 0) * self.downsample
        return seq_idx_block

    def __getitem__(self, index):
        row = self.video_info.iloc[index]
        vpath = os.path.join(self.raw_data_path,
                             row['participant_id'], row['video_id'])
        idx_block = self.idx_sampler(row)
        if idx_block is None:
            print(vpath)
#         print(idx_block)
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq * self.seq_len)

        #print ("idx_block, vpath: ", idx_block, vpath)
        seq = [pil_loader(os.path.join(vpath, 'frame_%010d.jpg' % i))
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
        if self.mode in ['train', 'val']:
            if self.class_type == 'verb':
                label = row['verb_class']
            elif self.class_type == 'noun':
                label = row['noun_class']
            return t_seq, label
        elif self.mode in ['test_seen', 'test_unseen']:
            return t_seq
        else:
            raise ValueError('wrong mode')

    def __len__(self):
        return len(self.video_info)


class epic_gulp(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=6,
                 num_seq=5,
                 downsample=3,
                 class_type='verb+noun'):
        print('WARNING: using obsolete dataset class')
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.class_type = class_type
        gulp_root = '/local/vondrick/epic-kitchens/gulp'

        print(os.path.join(gulp_root, 'rgb_train', self.class_type))
        self.EpicDataset = EpicVideoDataset(
            os.path.join(gulp_root, 'rgb_train'), self.class_type)
        dataset = list(self.EpicDataset)
        rgb = []
        for i in range(len(dataset)):
            # remove segments that are too short
            if dataset[i].num_frames > self.seq_len * self.num_seq * self.downsample:
                rgb.append(dataset[i])
        del dataset
        train_idx = random.sample(
            range(1, len(rgb)), int(float(len(rgb)) * 0.8))
        rgb_train = []
        rgb_val = []
        for i in range(len(rgb)):
            if i in train_idx:
                rgb_train.append(rgb[i])
            else:
                rgb_val.append(rgb[i])
        if self.mode == 'train':
            self.video_info = rgb_train
        elif self.mode in ['val']:
            self.video_info = rgb_val

    def idx_sampler(self, index):
        vlen = self.video_info[index].num_frames
        if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
            return None
        n = 1
        start_idx = np.random.choice(
            range(vlen - self.num_seq * self.seq_len * self.downsample), n)
#             print ("start_idx:", start_idx)
        seq_idx = np.expand_dims(
            np.arange(self.num_seq), -1) * self.downsample * self.seq_len + start_idx
#             print ("seq_idx:", seq_idx)
        seq_idx_block = seq_idx + \
            np.expand_dims(np.arange(self.seq_len), 0) * self.downsample
#             print ("seq_idx_block:", seq_idx_block)
        return seq_idx_block

    def __getitem__(self, index):

        idx_block = self.idx_sampler(index)
#             print(idx_block)
#             print(index)
#             print(len(self.video_info))
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq * self.seq_len)

        #print ("idx_block, ", idx_block)
        segment = self.EpicDataset.load_frames(self.video_info[index])
        seq = [segment[i] for i in idx_block]

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

        action = torch.LongTensor([self.video_info[index].verb_class])
        noun = torch.LongTensor([self.video_info[index].noun_class])

        return t_seq, action, noun

    def __len__(self):
        return len(self.video_info)


class UCF101_3d(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq=1,
                 downsample=3,
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
            split = '/proj/vondrick/lovish/data/ucf101/train_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):
            # use test for val, temporary
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

        return t_seq, label

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]


class UCF11_3d(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq=1,
                 downsample=3,
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
            split = '/proj/vondrick/lovish/data/ucf11/train_split.csv'
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):
            # use test for val, temporary
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

        return t_seq, label

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
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

        return t_seq, label

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]
