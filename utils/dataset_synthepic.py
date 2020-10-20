# May 2020
# Synthetic variant(s) of Epic Kitchens dataset a.k.a. Epic Splitchens or SynthEpic

import os, time
import pandas as pd
import numpy as np
from dataset_epic import *
from epic_data import *


class synthepic_action_pair(data.Dataset):
    '''
    Epic Kitchens variant that contains videos concatenating a sequence of two actions
    (action_A, gap, cut, gap, action_B), based on labeled segments.
    Uses frames generated by the synthepic_curate.ipynb Jupyter notebook.
    '''

    def __init__(self, mode='train', transform=None, seq_len=5, num_seq=8,
                 pred_step=3, downsample=6, drive='ssd',
                 sample_method='within', sample_offset=0,
                 exact_cuts=False):
        '''
        sample_method: within / match_cut
            - within = given a clip, sample uniformly randomly within.
            - match_cut = transition between subclips is precisely when we start predicting (t -> t+1).
        sample_offset: Number of video blocks to shift sequence sampling by (in case of match_cut only).
        '''
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.pred_step = pred_step
        self.downsample = downsample
        self.seq_frames = seq_len * num_seq * downsample # number of frames in one complete sequence
        self.drive = drive
        self.sample_method = sample_method
        self.sample_offset = sample_offset
        self.exact_cuts = exact_cuts

        # Gather video info for train / val / test. Every row of video_info is:
        # (video_root, n_frames, cut_idx, verb_A, noun_A, verb_B, noun_B)
        if exact_cuts:
            path_prefix = '/proj/vondrick/shared/uncertainty/synthepic_exact_'
        else:
            path_prefix = '/proj/vondrick/shared/uncertainty/synthepic_'
        if mode == 'train':
            split = path_prefix + 'train_split_%s.csv' % self.drive
            video_info = pd.read_csv(split)
        elif mode == 'val':
            split = path_prefix + 'val_split_%s.csv' % self.drive
            video_info = pd.read_csv(split)
        else:
            raise ValueError('Unknown dataset mode: ' + mode)

        # Drop videos that are too short
        drop_idx = []
        for idx, row in video_info.iterrows():
            vlen = row['n_frames']
            if vlen - self.seq_frames <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)
        print('SynthEPIC size:', len(self.video_info), 'sample method:', sample_method, sample_offset,
              'exact:', exact_cuts)

        self.last_random_state = 1754


    def index_sampler(self, n_frames, cut_idx):
        ''' Chooses and returns all frame indices for a single sequence. '''
        if n_frames - self.seq_frames <= 0:
            return None

        if self.sample_method == 'within':
            # Get uniformly random start frame number
            start_idx = np.random.randint(0, n_frames - self.seq_frames)
        elif self.sample_method == 'match_cut':
            start_idx = cut_idx - self.seq_len * (self.num_seq - self.pred_step) * self.downsample
            start_idx += self.sample_offset * self.seq_len * self.downsample

        # Verify clip range
        if start_idx < 0 or start_idx + self.seq_frames >= n_frames:
            return None

        # Get 1D list of frame numbers representing START of every block
        seq_idx = np.expand_dims(
            np.arange(self.num_seq), -1) * self.downsample * self.seq_len + start_idx

        # Get 2D list of frame numbers of ALL frames within every block
        seq_idx_block = seq_idx + \
            np.expand_dims(np.arange(self.seq_len), 0) * self.downsample

        # Return list of used frame indices
        return seq_idx_block


    def __getitem__(self, index):
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
        vpath, n_frames, cut_idx = row['vpath'], row['n_frames'], row['n_frames_A']
        
        # Sample uniformly random sequence of frames within this video
        idx_block = self.index_sampler(n_frames, cut_idx)
        if idx_block is None:
            # When match_cut didn't work out, simply call this method again to preserve statistical properties
            return self.__getitem__(np.random.randint(self.__len__()))

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

        # Convert list into tensor
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

        # Return all useful information in a dictionary
        result = {'t_seq': t_seq, 'idx_block': idx_block, 'vpath': vpath,
                  'n_frames': n_frames, 'cut_idx': cut_idx,
                  'n_frames_A': row['n_frames_A'], 'n_frames_B': row['n_frames_B'],
                  'video_id_A': row['video_id_A'], 'video_id_B': row['video_id_B'],
                  'src_start_A': row['src_start_A'], 'src_end_A': row['src_end_A'],
                  'src_start_B': row['src_start_B'], 'src_end_B': row['src_end_B'],
                  'verb_A': row['verb_A'], 'noun_A': row['noun_A'],
                  'verb_A_class': row['verb_A_class'], 'noun_A_class': row['noun_A_class'],
                  'verb_B': row['verb_B'], 'noun_B': row['noun_B'],
                  'verb_B_class': row['verb_B_class'], 'noun_B_class': row['noun_B_class']}
        return result


    def __len__(self):
        return len(self.video_info)