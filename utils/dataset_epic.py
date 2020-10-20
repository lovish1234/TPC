# May 2020
# Merge all datasets in this file
# Epic Kitchens only

import cv2
import torch
from torch.utils import data
from torchvision import transforms
import os
import time
import pickle
import random
import sys
import ast
from PIL import Image
from multiprocessing import Manager
import time

import pandas as pd
import numpy as np

from augmentation import *
from tqdm import tqdm
from joblib import Parallel, delayed
from gulpio import GulpDirectory
from epic_kitchens.dataset.epic_dataset import EpicVideoDataset, EpicVideoFlowDataset, GulpVideoSegment
from utils import *


def pil_loader(path):
#     im = Image.open(path)
#     return im
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            result = img.convert('RGB')
            return result


class epic_unlabeled(data.Dataset):
    '''
    Epic Kitchens dataset that uniformly randomly extracts segments throughout all video files.
    If needed, corresponding action labels are also given for every video block,
    but this may be empty because not every frame is labeled.
    This class is functionally identical to tpc/dataset_3d.py/epic().
    '''

    def __init__(self, mode='train', transform=None, seq_len=5, num_seq=8,
                 downsample=6, num_clips=10000, train_val_split=0.2, drive='ssd',
                 retrieve_actions=False, present_matching=False,
                 pred_step=3):
        '''
        mode: train / val / test
        seq_len: Number of frames in a video block.
        num_seq: Number of video blocks in a clip / sequence.
        downsample: Temporal sampling rate of frames. The effective new FPS becomes old FPS / downsample.
        num_clips: Total number of video sequences available in this dataset instance for training.
        retrieve_actions: Also add action over time to the returned dictionary.
        present_matching: Whether the model is doing present matching
        '''
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.present_matching = present_matching
        # for present matching, sample num_seq of sequences, but check existence for num_seq + pred_step
        if present_matching:
            self.seq_frames = seq_len * (pred_step + num_seq) * downsample
        else:
            # number of frames in one complete sequence
            self.seq_frames = seq_len * num_seq * downsample
        self.num_clips = num_clips
        self.train_val_split = train_val_split
        self.drive = drive
        self.retrieve_actions = retrieve_actions

        # Gather video info for train / val / test
        if mode == 'train':
            split = '/proj/vondrick/lovish/data/epic-kitchen/train_split_%s.csv' % self.drive
            video_info = pd.read_csv(split)
        elif mode == 'val':
            split = '/proj/vondrick/lovish/data/epic-kitchen/val_split_%s.csv' % self.drive
            video_info = pd.read_csv(split)
            self.num_clips = int(self.num_clips * train_val_split)
        elif mode == 'test':
            split = '/proj/vondrick/lovish/data/epic-kitchen/test_split_%s.csv' % self.drive
            video_info = pd.read_csv(split)
            self.num_clips = int(self.num_clips * train_val_split)
        else:
            raise ValueError('Unknown dataset mode: ' + mode)

        # Load dictionary to map video frames to actions
        if self.retrieve_actions:
            print('Loading action from frame dictionary...')
            with open('/proj/vondrick/ruoshi/github/TPC/tpc/action_from_frame.p', 'rb') as fp:
                self.map_action_frame = pickle.load(fp)

        # Drop videos that are too short
        drop_idx = []
        for idx, row in video_info.iterrows():
            vlen = row['n_frames']
            if vlen - self.seq_frames <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        # Detect repeated videos
        try:
            assert self.video_info['path'].nunique() == len(self.video_info)
        except AssertionError as e:
            e.args += ('There are repeated videos in dataframe loaded')

        self.last_random_state = 1949

    def index_sampler(self, vlen):
        ''' Chooses and returns all frame indices for a single sequence. '''
        if vlen - self.seq_frames <= 0:
            return None

        # Get uniformly random start frame number
        start_idx = np.random.randint(0, vlen - self.seq_frames)

        # Get 1D list of frame numbers representing START of every block
        seq_idx = np.expand_dims(
            np.arange(self.num_seq), -1) * self.downsample * self.seq_len + start_idx

        # Get 2D list of frame numbers of ALL frames within every block
        seq_idx_block = seq_idx + \
            np.expand_dims(np.arange(self.seq_len), 0) * self.downsample

        # Return list of used frame indices
        return seq_idx_block

    def __getitem__(self, index):

        ### time ###
        time_start = time.time()

        if self.mode == 'train':
            # For train, sample fully random sequences each time
            # Force randomness (this might help especially with num_workers > 0)
            random_state = index * 1830 + \
                int((time.time() * 123456) % 321654) + self.last_random_state
            np.random.seed(random_state)
            self.last_random_state = random_state % 654321
        elif self.mode in ['val', 'test']:
            # For val and test, set deterministic random_state
            random_state = index

        # Get video path and associated total number of frames
        row = self.video_info.sample(
            weights=self.video_info['n_frames'], random_state=random_state).iloc[0]
        vpath, vlen = row['path'], row['n_frames']
        
        # Sample uniformly random sequence of frames within this video
        idx_block = self.index_sampler(vlen)
        if idx_block is None:
            print(vpath, vlen)
            raise Exception(
                'Too short video! (shouldn\'t happen because we filtered them out)')

        # Decode list of frame numbers
        # dimensions = (block number, frame within block)
        assert(idx_block.shape == (self.num_seq, self.seq_len))
        # most significant = block number
        idx_block = idx_block.reshape(self.num_seq * self.seq_len)

        ### time ###
        time_2 = time.time()
        time_prepare = time.time() - time_start

        # Read all frames as images
        seq = [pil_loader(os.path.join(vpath, 'frame_%010d.jpg' % (i + 1)))
               for i in idx_block]

        ### time ###
        time_3 = time.time()
        time_load_img = time.time() - time_2

        # Transform sequence of images (NOT necessarily the same for all frames!)
        t_seq = self.transform(seq)

        ### time ###
        time_4 = time.time()
        time_transform = time.time() - time_3

        # What the heck are we doing here? lol
        try:
            (C, H, W) = t_seq[0].size()
            t_seq = torch.stack(t_seq, 0)
        except:
            (C, H, W) = t_seq[0][0].size()
            tmp = [torch.stack(i, 0) for i in t_seq]
            assert len(tmp) == 5
            t_seq = torch.stack(tmp, 1)

        # View as e.g. (8, 5, 3, 128, 128) and then transpose to (8, 3, 5, 128, 128)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1, 2)

        # NEW: return all useful information in a dictionary
        result = {'t_seq': t_seq, 'idx_block': idx_block, 'vpath': vpath}

        # Append action over time (start of each block) if specified
        if self.retrieve_actions:
            actions = []
            for frame_idx in idx_block[::self.seq_len]:
                cur_action = get_action_from_frame(
                    self.map_action_frame, vpath, frame_idx)
                if not('None' in cur_action):
                    actions.append(cur_action)
                else:
                    actions.append('')
            # result['actions'] = np.array(actions)
            result['actions'] = actions
            
        time_end = time.time()
#         total_time = time_end - time_start
#         print('prepare time: %0.3f s, %0.3f %%' % (time_prepare, time_prepare / total_time * 100))
#         print('load img time: %0.3f s, %0.3f %%' % (time_load_img, time_load_img / total_time * 100))
#         print('transform time: %0.3f s, %0.3f %%' % (time_transform, time_transform / total_time * 100))
#         print('loading image from %s, frame No. %d - %d\n' % (vpath, idx_block[0], idx_block[-1]))
        return result

    def __len__(self):
        return self.num_clips


class epic_present_matching(data.Dataset):
    '''
    Epic Kitchens dataset that load sequences based on present matching results on c_t.
    Similar video sequences are put into one batch as a form of hard negative mining.

    We first load a preprocessed adjacency matrix dist_adj which include the nearest
    neighbors for each video sequences

    Order of dataset:
    for an index i, we take No. dist_adj[i // num_ngbrs][i % num_ngbrs] video sequence
    As a result, the number of sets of neighbors in a batch will be batch_size/num_ngbrs

    NOTE: must use sequential data sampler in dataloader
    '''

    def __init__(self, epoch, freq=10, mode='train', transform=None, seq_len=5, num_seq=8,
                 downsample=6, num_clips=10000, train_val_split=0.2, drive='ssd',
                 retrieve_actions=False, NN_path='', num_ngbrs=4):
        '''
        epoch: current epoch number
        freq: frequency of negative mining (every 10 epochs by default)
        mode: train / val / test
        seq_len: Number of frames in a video block.
        num_seq: Number of video blocks in a clip / sequence.
        downsample: Temporal sampling rate of frames. The effective new FPS becomes old FPS / downsample.
        num_clips: Total number of video sequences available in this dataset instance for training.
        retrieve_actions: Also add action over time to the returned dictionary.
        NN_path: path to present matching results, used to load nearest neighbors in the same batch
        num_ngbrs: Number of neighbors to put in one batch
        '''
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        # number of frames in one complete sequence
        self.seq_frames = seq_len * num_seq * downsample
        if mode in ['val', 'test']:
            self.num_clips = int(num_clips * train_val_split)
        else:
            self.num_clips = num_clips
        self.train_val_split = train_val_split
        self.drive = drive
        self.retrieve_actions = retrieve_actions
        self.NN_path = NN_path
        self.num_ngbrs = num_ngbrs
        '''
        E.g. for epoch=26, freq=10, offset = (26 % 10) * (10000/4) = 15000
        dataset will retrieve data from row 15000 in dist_adj
        '''
        if mode in ['val', 'test']:
            self.offset = 0
        else:
            self.offset = int((epoch % freq) * (num_clips / num_ngbrs))

        if mode not in ['train', 'val', 'test']:
            raise ValueError('Unknown dataset mode: ' + mode)
        
        # Load present matching data
        distance_path = os.path.join(NN_path, 'argmax_100.pt')
        vpath_path = os.path.join(NN_path, 'vpath.p')
        idx_block_path = os.path.join(NN_path, 'idx_block.p')
        # nearest 100 neighbors of each video sequences
        dist_adj = torch.load(distance_path)

        # video path of each video sequences corresponding to dist_adj
        with open(vpath_path, "rb") as fp:
            vpath_list = pickle.load(fp)
        # idx_block of each video sequences corresponding to dist_adj
        with open(idx_block_path, "rb") as fp:
            idx_block_list = pickle.load(fp)
        self.vpath_list = vpath_list
        '''
        dist_adj stores the nearest neighbors information for each sequences
        In each row, first index point to the quiried sequence, the following
        indices point to its nearest neighbors by order of cosine distance
        '''
        self.dist_adj = dist_adj

        self.idx_block_list = idx_block_list

        # Load dictionary to map video frames to actions
        if self.retrieve_actions:
            print('Loading action from frame dictionary...')
            with open('/proj/vondrick/ruoshi/github/TPC/tpc/action_from_frame.p', 'rb') as fp:
                self.map_action_frame = pickle.load(fp)

    def index_sampler(self, vlen=None, start_idx=None):
        ''' Chooses and returns all frame indices for a single sequence. 
        If start index is provided, expand to get index block'''
        if vlen is not None and vlen - self.seq_frames <= 0:
            return None

        if start_idx is None:
            # Get uniformly random start frame number
            start_idx = np.random.randint(0, vlen - self.seq_frames)

        # Get 1D list of frame numbers representing START of every block
        seq_idx = np.expand_dims(
            np.arange(self.num_seq), -1) * self.downsample * self.seq_len + start_idx

        # Get 2D list of frame numbers of ALL frames within every block
        seq_idx_block = seq_idx + \
            np.expand_dims(np.arange(self.seq_len), 0) * self.downsample

        # Return list of used frame indices
        return seq_idx_block

    def __getitem__(self, index):

        #         ### time ###
        #         time_start = time.time()
        index += self.offset  # change starting point in dist_adj
        row_index = index // self.num_ngbrs
        col_index = index % self.num_ngbrs
        
        # get corresponding video path and idx_block according to distance matrix
        video_index = self.dist_adj[row_index][col_index].item()
        vpath = self.vpath_list[video_index]
        first_frame = self.idx_block_list[video_index][0].item()
        idx_block = self.index_sampler(start_idx=first_frame)


        # Decode list of frame numbers
        # dimensions = (block number, frame within block)
        assert(idx_block.shape == (self.num_seq, self.seq_len))
        # most significant = block number
        idx_block = idx_block.reshape(self.num_seq * self.seq_len)

        # Read all frames as images
        seq = [pil_loader(os.path.join(vpath, 'frame_%010d.jpg' % (i + 1)))
               for i in idx_block]

        # Transform sequence of images (NOT necessarily the same for all frames!)
        t_seq = self.transform(seq)

        # What the heck are we doing here? lol
        try:
            (C, H, W) = t_seq[0].size()
            t_seq = torch.stack(t_seq, 0)
        except:
            (C, H, W) = t_seq[0][0].size()
            tmp = [torch.stack(i, 0) for i in t_seq]
            assert len(tmp) == 5
            t_seq = torch.stack(tmp, 1)

        # View as e.g. (8, 5, 3, 128, 128) and then transpose to (8, 3, 5, 128, 128)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1, 2)

        # NEW: return all useful information in a dictionary
        result = {'t_seq': t_seq, 'idx_block': idx_block, 'vpath': vpath}

        # Append action over time (start of each block) if specified
        if self.retrieve_actions:
            actions = []
            for frame_idx in idx_block[::self.seq_len]:
                cur_action = get_action_from_frame(
                    self.map_action_frame, vpath, frame_idx)
                if not('None' in cur_action):
                    actions.append(cur_action)
                else:
                    actions.append('')
            # result['actions'] = np.array(actions)
            result['actions'] = actions
        return result

    def __len__(self):
        return self.num_clips


class epic_action_based(data.Dataset):
    '''
    Epic Kitchens dataset via the gulp adapter,
    that samples video sequences around labeled actions, either conditional or not.
    If needed, only a subset of verbs or nouns can be considered.
    The returned clips are either uniformly sampled within an action segment,
    or can have specific starting or ending points aligned with the action duration
    in different ways, as specified.
    '''

    def __init__(self, mode='train', transform=None, seq_len=5, num_seq=8,
                 downsample=6, class_type='both', train_val_split=0.2,
                 verb_subset=None, noun_subset=None, participant_subset=None,
                 drive='ssd', sample_method='within', sample_offset=0,
                 label_fraction=1.0):
        '''
        mode: train / val / test_seen / test_unseen.
        seq_len: Number of frames in a video block.
        num_seq: Number of video blocks in a clip / sequence.
        downsample: Temporal sampling rate of frames. The effective new FPS becomes old FPS / downsample.
        class_type: verb / noun / both.
        label_fraction: < 1.0 means use fewer labels for training.

        verb_subset: List of verb strings to condition on, if not None.
        noun_subset: List of noun strings to condition on, if not None.
        participant_subset: List of participants to condition on, if not None.
        Examples: verb_subset = ['take', 'open', 'close'], participant_subset = ['P16', 'P11'].

        sample_method: within / match_start / match_end / before
          - within = uniformly randomly sample sequence fully within an action label segment (e.g. for action classification)
          - match_start = START of sequence matches START of action label segment
          - match_end = END of sequence matches END of action label segment (e.g. for future uncertainty ranking)
          - before = END of sequence matches START of action label segment (e.g. for action anticipation)
        (NOTE: 'within' discards too short segments, all other methods do not.)

        sample_offset: Number of video blocks to shift sequence sampling by.
        Example 1: if (sample_method == 'match_start', sample_offset == -2),
        then video sequence starts already 2 blocks before the current action starts.
        Example 2: if (sample_method == 'match_end', sample_offset == 3, pred_step == 3),
        then all warmup video blocks represent the current action in progress, but all predicted blocks represent another, unknown action.
        Example 3: if (sample_method == 'before', sample_offset == 1, pred_step == 3),
        then only the LAST predicted block represents the current action, all preceding blocks represent something else.
        '''
        if not(class_type in ['verb', 'noun']):
            class_type = 'verb+noun'
            # print('=> class_type is now set to both a.k.a. verb+noun')

        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.block_frames = seq_len * downsample  # number of frames in one video block
        # number of frames in one complete sequence
        self.seq_frames = seq_len * num_seq * downsample
        self.class_type = class_type
        self.train_val_split = train_val_split
        self.verb_subset = verb_subset
        self.noun_subset = noun_subset
        self.participant_subset = participant_subset
        self.drive = drive
        self.sample_method = sample_method
        self.sample_offset = sample_offset
        self.label_fraction = label_fraction

        # Verify arguments
        if not(mode in ['train', 'val', 'test_seen', 'test_unseen']):
            raise ValueError('Unknown dataset mode: ' + mode)

        # Specify paths, both gulp (= main) and jpg (= backup)
        # JPG path format arguments:
        if drive == 'ssd':
            gulp_root = '/proj/vondrick/datasets/epic-kitchens/data/processed/gulp'
            self.jpg_path = '/local/vondrick/epic-kitchens/raw/rgb/{}/{}/frame_{:010d}.jpg'
        else:
            print('== WARNING! == using HDD instead of SSD')
            gulp_root = '/proj/vondrick/datasets/epic-kitchens/data/processed/gulp'
            self.jpg_path = '/proj/vondrick/datasets/epic-kitchens/data/raw/rgb/{}/{}/frame_{:010d}.jpg'

        # Load video info (RGB frames)
        subfolder = ('rgb_train' if mode == 'train' or mode ==
                     'val' else 'rgb_' + mode)
        full_path = os.path.join(gulp_root, subfolder)
        print('Selected dataset:', full_path, self.class_type)
        self.epic_dataset_inst = EpicVideoDataset(full_path, self.class_type)

        # Split dataset randomly into train & validation with fixed seed
        # NOTE: this split will be different for other values of train_val_split
        dataset = list(self.epic_dataset_inst)
        if mode in ['train', 'val']:
            rand_state = random.getstate()
            random.seed(8888)
            train_list = random.sample(dataset, int(
                len(dataset) * (1 - self.train_val_split)))  # without replacement
            random.setstate(rand_state)  # retain & restore random state

            if label_fraction < 1.0:
                print('== WARNING! == using just a fraction of available labels for training: ' + str(label_fraction * 100) + '%')
                used_train_len = int(label_fraction * len(train_list))
                train_list = train_list[:used_train_len] # deterministic operation because of fixed seed above

            if mode == 'train':
                dataset = train_list
            elif mode == 'val':
                train_set = set(train_list)
                val_list = []
                for item in dataset:
                    if item not in train_set:
                        val_list.append(item)
                dataset = val_list

        # Loop over segments in epic dataset and filter out videos
        rgb = []
        for i in range(len(dataset)):
            # If within, retain only sufficiently long video clips
            if sample_method == 'within' and dataset[i].num_frames <= self.seq_frames:
                continue
            # Condition on verbs
            if verb_subset is not None and not(dataset[i].verb in verb_subset):
                continue
            # Condition on nouns
            if noun_subset is not None and not(dataset[i].noun in noun_subset):
                continue
            # Condition on participants
            if participant_subset is not None and not(dataset[i].participant_id in participant_subset):
                continue
            rgb.append(dataset[i])

        self.video_info = rgb
        del dataset

        # Fix memory leak
#         manager = Manager()
#         self.video_info = manager.list(self.video_info)

    def index_sampler(self, segment_idx, vlen):
        ''' Chooses and returns all frame indices for a single sequence. '''

        # Get start frame number
        if self.sample_method == 'within':
            start_idx = np.random.choice(range(0, vlen - self.seq_frames))
        elif self.sample_method == 'match_start':
            start_idx = 0
        elif self.sample_method == 'match_end':
            start_idx = vlen - self.seq_frames
        elif self.sample_method == 'before':
            start_idx = -self.seq_frames

        # Apply offset
        start_idx += self.sample_offset * self.downsample * self.seq_len

        # Get 1D list of frame numbers representing START of every block
        seq_idx = np.expand_dims(
            np.arange(self.num_seq), -1) * self.downsample * self.seq_len + start_idx

        # Get 2D list of frame numbers of ALL frames within every block
        seq_idx_block = seq_idx + \
            np.expand_dims(np.arange(self.seq_len), 0) * self.downsample

        # Return list of used frame indices
        return seq_idx_block

    def __getitem__(self, index):
        segment_idx = index
        if self.mode == 'train':
            # For train, sample fully random subsequencies each time
            # Force randomness (this might help especially with num_workers > 0)
            np.random.seed(index + int((time.time() * 1234) % 123456))

        vlen = self.video_info[segment_idx].num_frames
        idx_block = self.index_sampler(segment_idx, vlen)
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq * self.seq_len)
        segment_start = self.video_info[segment_idx].start_frame
        # always use indices within full video file instead of action segment
        idx_block = idx_block + segment_start

        participant_id = self.video_info[segment_idx].participant_id  # 'P12'
        video_id = self.video_info[segment_idx].video_id  # 'P12_08'
        fp_first = self.jpg_path.format(
            participant_id, video_id, idx_block[0] + 1)
        fp_last = self.jpg_path.format(
            participant_id, video_id, idx_block[-1] + 1)
        if os.path.exists(fp_first) and os.path.exists(fp_last):
            seq = [pil_loader(self.jpg_path.format(
                participant_id, video_id, idx + 1)) for idx in idx_block]
        else:
            # The selected range is out of bounds even for the whole video file
            # Simply restart method to preserve the data's statistical properties
            return self.__getitem__(np.random.randint(self.__len__()))
        # discarding load_frames method with epic gulp because of inefficient dataloading
#         else:
#             # Use EpicVideoDataset
#             full_segment = self.epic_dataset_inst.load_frames(self.video_info[index])
#             seq = [full_segment[i] for i in idx_block]

        # Transform sequence of images (NOT necessarily the same for all frames!)
        t_seq = self.transform(seq)

        # What the heck are we doing here? lol
        try:
            (C, H, W) = t_seq[0].size()
            t_seq = torch.stack(t_seq, 0)
        except:
            (C, H, W) = t_seq[0][0].size()
            tmp = [torch.stack(i, 0) for i in t_seq]
            assert len(tmp) == 5
            t_seq = torch.stack(tmp, 1)

        # View as e.g. (8, 5, 3, 128, 128) and then transpose to (8, 3, 5, 128, 128)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1, 2)

        # NEW: return all useful information in a dictionary
        # NOTE: as of 05/10, idx_block is always with reference to whole video start, not segment start
        verb_class = self.video_info[segment_idx].verb_class
        noun_class = self.video_info[segment_idx].noun_class
        label = verb_class if self.class_type == 'verb' else noun_class
        result = {'t_seq': t_seq, 'idx_block': idx_block, 'label': label,
                  'participant_id': self.video_info[segment_idx].participant_id,
                  'video_id': self.video_info[segment_idx].video_id,
                  'segment_start': self.video_info[segment_idx].start_frame,
                  'segment_stop': self.video_info[segment_idx].stop_frame,
                  'verb': self.video_info[segment_idx].verb,
                  'noun': self.video_info[segment_idx].noun,
                  'narration': self.video_info[segment_idx].narration,
                  'verb_class': verb_class, 'noun_class': noun_class}
        return result

    def __len__(self):
        return len(self.video_info)
