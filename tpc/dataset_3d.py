# get dataset in 3-d format

#############################################################################
#### OBSOLETE DO NOT USE, see utils/dataset_epic.py and dataset_other.py ####
#############################################################################

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
sys.path.append('../utils')

# augmentation tools
from augmentation import *

# bookkeeping tools
from tqdm import tqdm

# parallalization tools
from joblib import Parallel, delayed


from gulpio import GulpDirectory
from epic_kitchens.dataset.epic_dataset import EpicVideoDataset, EpicVideoFlowDataset, GulpVideoSegment


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            result = img.convert('RGB')
            return result


class epic_gulp(data.Dataset):
    '''
    Epic Kitchens dataset via the gulp adapter.
    Loads clips uniformly sampled within an action segment.
    '''

    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=5,
                 num_seq=8,
                 downsample=6,
                 class_type='verb+noun',
                 drive='ssd'):
        print('-- WARNING! -- using obsolete dataset class, see utils/dataset_epic.py and dataset_other.py instead')
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.class_type = class_type
        if drive == 'ssd':
            gulp_root = '/local/vondrick/epic-kitchens/gulp'
        else:
            gulp_root = '/proj/vondrick/datasets/epic-kitchens/data/processed/gulp'

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

        # OLD: return sequence only
        # return t_seq, action, noun
        # NEW: return all useful information in a dictionary
        result = {'t_seq': t_seq, 'idx_block': idx_block, 'vpath': 'TODO, idk how to retrieve',
                  'action': action, 'noun': noun}
        return result

    def __len__(self):
        return len(self.video_info)


class epic_action(data.Dataset):
    '''Dataset that load all video sequences conditioned on verbs and nouns
    provided as input.
    Train and val splits are splitted manually from EPIC default training set
    Test seen and test unseen are original EPIC dataset
    For train and val, return the labels of both verb and noun
    For test_seen and test_unseen, returns only video sequence, vpath, and index'''

    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=1,
                 num_seq=2,
                 downsample=1,
                 verb='all',
                 noun='all',
                 drive='ssd'):
        print('-- WARNING! -- using obsolete dataset class, see utils/dataset_epic.py and dataset_other.py instead')
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.verb = verb  # specified verb to be retrieved, when 'all', load all actions
        self.noun = noun  # specified noun to be retrieved, when 'all', load all actions
        self.drive = drive

        self.train_all = '/proj/vondrick/datasets/epic-kitchens/data/annotations/EPIC_train_action_labels.csv'
        self.train_set = '/proj/vondrick/datasets/epic-kitchens/data/annotations/train_val/EPIC_train_action_labels.csv'
        self.val_set = '/proj/vondrick/datasets/epic-kitchens/data/annotations/train_val/EPIC_val_action_labels.csv'
        self.verb_classes = '/proj/vondrick/datasets/epic-kitchens/data/annotations/EPIC_verb_classes.csv'
        self.noun_classes = '/proj/vondrick/datasets/epic-kitchens/data/annotations/EPIC_noun_classes.csv'
        if self.drive == 'hdd':
            self.raw_data_path = '/proj/vondrick/datasets/epic-kitchens/data/raw/rgb'
        elif self.drive == 'ssd':
            self.raw_data_path = '/local/vondrick/epic-kitchens/raw/rgb'
        self.test_seen_set = '/proj/vondrick/datasets/epic-kitchens/data/annotations/EPIC_test_s1_timestamps.csv'
        self.test_unseen_set = '/proj/vondrick/datasets/epic-kitchens/data/annotations/EPIC_test_s2_timestamps.csv'

        train_df = pd.read_csv(self.train_all)
        val_df = pd.read_csv(self.val_set)
        self.verbs_df = pd.read_csv(self.verb_classes)
        self.nouns_df = pd.read_csv(self.noun_classes)

        # construct a map to replace redundant verb
        verbs_map = {}
        for index, row in self.verbs_df.iterrows():
            verbs_map[row['class_key']] = ast.literal_eval(row['verbs'])

        # construct a map to replace redundant noun
        nouns_map = {}
        for index, row in self.nouns_df.iterrows():
            nouns_map[row['class_key']] = ast.literal_eval(row['nouns'])

        # replace redundant verbs and nouns
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

        # drop unqualified sequences (not enough frames to be loaded)
        drop_idx = []
        for idx, row in video_info.iterrows():
            if row['stop_frame'] < num_seq * seq_len * downsample + 1:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        # conditioned on given verb or noun
        if verb != 'all':
            self.video_info = self.video_info.loc[self.video_info['verb'] == self.verb]
        if noun != 'all':
            self.video_info = self.video_info.loc[self.video_info['noun'] == self.noun]

    def idx_sampler(self, row):
        ''' sample and expand index block'''
#         if row['start_frame'] < self.num_seq * self.seq_len * self.downsample:
#             return [None]
        start_idx = row['stop_frame'] - self.num_seq * \
            self.seq_len * self.downsample
        seq_idx = np.expand_dims(
            np.arange(self.num_seq), -1) * self.downsample * self.seq_len + start_idx
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
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq * self.seq_len)

        seq = [pil_loader(os.path.join(vpath, 'frame_%010d.jpg' % i))
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
        t_seq = t_seq.view(self.num_seq, self.seq_len,
                           C, H, W).transpose(1, 2)
        if self.mode in ['train', 'val']:
            label_verb = row['verb_class']
            label_noun = row['noun_class']
            return {'t_seq': t_seq, 'vpath': vpath, 'idx_block': idx_block, 'verb': label_verb, 'noun': label_noun}
        elif self.mode in ['test_seen', 'test_unseen']:
            return {'t_seq': t_seq, 'vpath': vpath, 'idx_block': idx_block}
        else:
            raise ValueError('wrong mode')

    def __len__(self):
        return len(self.video_info)


class SingleVideoDataset(data.Dataset):
    ''' Dataset that iterates over a whole video file once.
    One returned item of this dataset represents a sequence of 8 blocks,
    and we keep returning items covering temporally shifted sequences within the video file. '''

    def __init__(self, file_path, transform=None, start_stride=3, num_seq=8, seq_len=5, downsample=3):
        ''' start_stride: Frame start number increment within the full video for every sequence.
        Example: if start_stride=2 and downsample=3, we first process frames (0, 3, 6, ..., 117) for item index = 0
        and then (2, 5, 8, ..., 119) for item index = 1 and so on. '''
        print('-- WARNING! -- using obsolete dataset class, see utils/dataset_epic.py and dataset_other.py instead')
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


class epic(data.Dataset):
    '''
    mode: train or test split
    seq_len: number of frames in a video block
    num_seq: number of video block
    downsample: temporal downsample rate of frames
    num_sample: number of 'sequence of video blocks' sampled from one video
    '''

    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=5,
                 num_seq=8,
                 downsample=6,
                 drive='ssd',
                 num_sample=5000):
        print('-- WARNING! -- using obsolete dataset class, see utils/dataset_epic.py and dataset_other.py instead')
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.drive = drive
        # number of sequences to sample in total from train_split
        self.num_sample = num_sample

        # splits
        if mode == 'train':
            split = '/proj/vondrick/lovish/data/epic-kitchen' + \
                '/train_split_%s.csv' % self.drive
            video_info = pd.read_csv(split)
        elif mode == 'val':
            # use test for val, temporary
            split = '/proj/vondrick/lovish/data/epic-kitchen' + \
                '/val_split_%s.csv' % self.drive
            video_info = pd.read_csv(split)
            self.num_sample = int(self.num_sample * 0.2)
        elif mode == 'test':
            # use test for val, temporary
            split = '/proj/vondrick/lovish/data/epic-kitchen' + \
                '/test_split_%s.csv' % self.drive
            video_info = pd.read_csv(split)
            self.num_sample = int(self.num_sample * 0.2)
        else:
            raise ValueError('wrong mode')

        # construct map to decode and encode action
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

        # drop videos that are too short
        drop_idx = []
        for idx, row in video_info.iterrows():
            # vpath, vlen = row
            vlen = row['n_frames']
            if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        # calculate probability for each video
        total_frame = sum(self.video_info['n_frames'])
        prob = []
        for index, row in self.video_info.iterrows():
            prob.append(row['n_frames'] / total_frame)
        self.video_info['prob'] = prob

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
            return [None]
        n = 1

        # Get uniformly random start frame number
        start_idx = np.random.choice(
            range(1, vlen - self.num_seq * self.seq_len * self.downsample), n)

        # Get 1D list of frame numbers representing START of every block
        seq_idx = np.expand_dims(
            np.arange(self.num_seq), -1) * self.downsample * self.seq_len + start_idx

        # Get 2D list of frame numbers of ALL frames within every block
        seq_idx_block = seq_idx + \
            np.expand_dims(np.arange(self.seq_len), 0) * self.downsample

        # Return video file path & list of used frame indices
        return [seq_idx_block, vpath]

    def __getitem__(self, index):
        # Get video path and associated total number of frames
        # for train, resample sequences each time
        # for val and test, set deterministic random_state
        if self.mode == 'train':
            row = self.video_info.sample(
                weights=self.video_info['prob']).iloc[0]
        elif self.mode in ['val', 'test']:
            row = self.video_info.sample(
                weights=self.video_info['prob'], random_state=index).iloc[0]
        vpath, vlen = row['path'], row['n_frames']

        # Sample uniformly random sequence of frames within this video
        items = self.idx_sampler(vlen, vpath)
        if items is None:
            print(vpath)
            raise Exception(
                'Too short video! (shouldn\'t happen because we filtered them out)')

        # Decode list of frame numbers
        idx_block, vpath = items
        # dimensions = (block number, frame within block)
        assert(idx_block.shape == (self.num_seq, self.seq_len))
        # most significant = block number
        idx_block = idx_block.reshape(self.num_seq * self.seq_len)

        # Read all frames as images
        seq = [pil_loader(os.path.join(vpath, 'frame_%010d.jpg' % (i + 1)))
               for i in idx_block]

        # Transform sequence of images (NOT the same for all frames!)
        t_seq = self.transform(seq)

        # What the heck are we doing here? lol
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
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1, 2)

        # OLD: return sequence only
        # return t_seq
        # NEW: return all useful information in a dictionary
        result = {'t_seq': t_seq, 'idx_block': idx_block, 'vpath': vpath}
        return result

    def __len__(self):
        return self.num_sample

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]


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
        print('-- WARNING! -- using obsolete dataset class, see utils/dataset_epic.py and dataset_other.py instead')
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
        print('-- WARNING! -- using obsolete dataset class, see utils/dataset_epic.py and dataset_other.py instead')
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
                 return_label=False,
                 drive='hdd'):
        print('-- WARNING! -- using obsolete dataset class, see utils/dataset_epic.py and dataset_other.py instead')
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

        if self.return_label:
            try:
                vname = vpath.split('/')[-3]
                vid = self.encode_action(vname)
            except:
                vname = vpath.split('/')[-2]
                vid = self.encode_action(vname)

            label = torch.LongTensor([vid])

            # OLD: return sequence only
            # return t_seq, label
            # NEW: return all useful information in a dictionary
            result = {'t_seq': t_seq, 'label': label,
                      'idx_block': idx_block, 'vpath': vpath}
            return result

        # OLD: return sequence only
        # return t_seq
        # NEW: return all useful information in a dictionary
        result = {'t_seq': t_seq, 'idx_block': idx_block, 'vpath': vpath}
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
                 return_label=False,
                 drive='hdd'):
        print('-- WARNING! -- using obsolete dataset class, see utils/dataset_epic.py and dataset_other.py instead')
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq

        self.downsample = downsample
        self.epsilon = epsilon

        self.which_split = which_split
        self.return_label = return_label

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

            # OLD: return sequence only
            # return t_seq, label
            # NEW: return all useful information in a dictionary
            result = {'t_seq': t_seq, 'label': label,
                      'idx_block': idx_block, 'vpath': vpath}
            return result

        # OLD: return sequence only
        # return t_seq
        # NEW: return all useful information in a dictionary
        result = {'t_seq': t_seq, 'idx_block': idx_block, 'vpath': vpath}
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
                 return_label=False,
                 drive='hdd'):
        print('-- WARNING! -- using obsolete dataset class, see utils/dataset_epic.py and dataset_other.py instead')
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq

        self.downsample = downsample
        self.epsilon = epsilon

        self.which_split = which_split
        self.return_label = return_label
        self.drive = drive

        # splits
        if mode == 'train':
            split = '/proj/vondrick/lovish/data/ucf11/train_split.csv' % self.drive
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):  # use val for test
            split = '/proj/vondrick/lovish/data/ucf11/test_split.csv' % self.drive
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

            # OLD: return sequence only
            # return t_seq, label
            # NEW: return all useful information in a dictionary
            result = {'t_seq': t_seq, 'label': label,
                      'idx_block': idx_block, 'vpath': vpath}
            return result

        # OLD: return sequence only
        # return t_seq
        # NEW: return all useful information in a dictionary
        result = {'t_seq': t_seq, 'idx_block': idx_block, 'vpath': vpath}
        return result

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return action code'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]
