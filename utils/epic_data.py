# April 2020
# Tools related to the Epic Kitchens & SynthEpic dataset for TPC / VAE-DPC.

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from collections import *
from IPython.display import HTML, Video
from pathlib import Path
from skvideo.io import FFmpegWriter
from epic_kitchens.dataset.epic_dataset import EpicVideoDataset, EpicVideoFlowDataset, GulpVideoSegment


def convert_jpg_video(video_root, start_frame, dst_path, n_frames=-1, fps=60, insert_pause=-1,
                      show_vid=False, frame_number_display=False, ignore_if_exist=True, store_frames=[]):
    '''
    Helper for Epic and SynthEpic visualization.
    ignore_if_exist: If True, don't write any files if the .mp4 already exists. Set False if store_frames changed.
    '''

    if not(os.path.exists(dst_path)) or not(ignore_if_exist):

        if not(os.path.exists(Path(dst_path).parent)):
            os.makedirs(Path(dst_path).parent)
        if n_frames <= 0:
            n_frames = len(os.listdir(video_root))

        if store_frames!=[]:
            dst_frame_path = 'single_video_frames/'+str(video_root.split('/')[-1])+'/'+str(start_frame)
            if not(os.path.exists(dst_frame_path)):
                os.makedirs(dst_frame_path)

        # Write all frames
        writer = FFmpegWriter(dst_path, inputdict={'-r': str(fps)})
        for i in range(n_frames):
            cur_frame = start_frame + i
            file_name = 'frame_{:010d}.jpg'.format(cur_frame + 1)
            video_path = os.path.join(video_root, file_name)
            if not(os.path.exists(video_path)):
                break  # n_frames could be overestimated due to extra files

            frame = plt.imread(video_path)
            if cur_frame in store_frames:
                frame_still = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(dst_frame_path, file_name), frame_still)

            if frame_number_display:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (250, 250)
                fontScale = 1
                fontColor = (255, 255, 255)
                lineType = 2
                frame_annotated = cv2.putText(frame, str(start_frame + i),
                                            bottomLeftCornerOfText,
                                            font,
                                            fontScale,
                                            fontColor,
                                            lineType)

                frame_annotated = frame_annotated.copy()
                writer.writeFrame(frame_annotated)
            else:	
                frame = frame.copy()	
                writer.writeFrame(frame)

            # Insert pause (with orange borders) just as predictions start to happen
            if insert_pause >= 0 and i == insert_pause:
                alert_frame = frame.copy()
                alert_frame[:20, :] = [255, 128, 0]
                alert_frame[-20:, :] = [255, 128, 0]
                alert_frame[:, :20] = [255, 128, 0]
                alert_frame[:, -20:] = [255, 128, 0]
                for j in range(60):
                    writer.writeFrame(alert_frame)

        writer.close()

    if show_vid:
        # For use in jupyter notebook
        display(Video(dst_path, embed=True, width=480, height=320))


def convert_epic_video(video_id, start_frame, dst_path, n_frames=8 * 5 * 6, fps=60, insert_pause=-1,
                       show_vid=False, frame_number_display=False, ignore_if_exist=True, store_frames=[]):
    '''
    Converts a sequence in the dataset from a bunch of images to a video file.
    Example: video_id = 'P12_08', start_frame = 1234, dst_path = 'ruovish.mp4'.

    WARNING: n_frames must be at least 240 frames (= 4 seconds), preferably longer,
    otherwise no meaningful diversity results can be obtained with VAE-DPC.

    insert_pause: If >= 0, show orange borders on that frame index, indicating that future prediction starts.
    show_vid: Display output video after writing (e.g. for Jupyter notebooks).
    frame_number_display: Print frame numbers on each frame of the video.
    ignore_if_exist: If True, don't write any files if the .mp4 already exists. Set False if store_frames changed.
    '''

    kitchen_dir = video_id[:3]
    video_root = '/local/vondrick/epic-kitchens/raw/rgb/' + \
        kitchen_dir + '/' + video_id
    print(dst_path)
    print("Here")
    convert_jpg_video(video_root, start_frame, dst_path, n_frames=n_frames, fps=fps,
                      insert_pause=insert_pause, show_vid=show_vid, frame_number_display=frame_number_display,
                      ignore_if_exist=ignore_if_exist, store_frames=store_frames)


def read_all_frames(video_path):
    ''' Returns (n_frames, 256, 456, 3) numpy array with video frames. '''
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        frame = frame[:, :, ::-1]  # BGR to RGB
        assert(frame.ndim == 3)
        assert(frame.shape[2] == 3)
        frames.append(frame)
    frames = np.array(frames)
    return frames


def retrieve_action_pair_clips(verb_A, noun_A, verb_B, noun_B, mode='train'):
    '''
    Returns a list of all identifying metadata for Epic video clip sequences
    that depict an ordered sequence of actions "verb_A noun_A" -> "verb_B noun_B".
    Arguments can be either class indices or strings.
    Here, every item is: (video_id, start_frame, frames_A, frames_gap, frames_B),
    for example: ('P12_08', 1234, 200, 100, 200) where total video clip frames is 500.
    mode: train / val / test.
    WARNING: same gulp folder is used for train & val, i.e. split is NOT made here!
    '''

    # Format with participant_id, video_id, frame_number
    video_path = '/local/vondrick/epic-kitchens/raw/rgb/{}/{}/frame_{:010d}.jpg'
    gulp_subfolder = ('rgb_train' if mode ==
                      'train' or mode == 'val' else 'rgb_' + mode)
    gulp_path = '/proj/vondrick/datasets/epic-kitchens/data/processed/gulp/' + gulp_subfolder
    epic_inst = list(EpicVideoDataset(gulp_path, 'verb+noun'))
    epic_inst.sort(key=(lambda k: k.video_id +
                        '_{:010d}'.format(k.start_frame)))
    result = []

    # Loop over all pairs of action segments
    for i in range(len(epic_inst) - 1):
        segment_A = epic_inst[i]
        segment_B = epic_inst[i + 1]
        # Proceed only if same video
        if segment_A.video_id != segment_B.video_id:
            continue
        # Condition on first action
        if not(verb_A in [segment_A.verb, segment_A.verb_class]) or \
                not(noun_A in [segment_A.noun, segment_A.noun_class]):
            continue
        # Condition on second action
        if not(verb_B in [segment_B.verb, segment_B.verb_class]) or \
                not(noun_B in [segment_B.noun, segment_B.noun_class]):
            continue
        # Append clip
        cur_item = (segment_A.video_id, segment_A.start_frame, segment_A.num_frames,
                    segment_B.start_frame -
                    (segment_A.start_frame + segment_A.num_frames),
                    segment_B.num_frames)
        result.append(cur_item)

    return result


def retrieve_action_pair_clips_dict(mode='train'):
    '''
    Similar to retrieve_action_pair_clips(), but does not condition on actions,
    and returns every available pair in the dataset organized by dictionary keys instead.
    '''

    # Format with participant_id, video_id, frame_number
    video_path = '/local/vondrick/epic-kitchens/raw/rgb/{}/{}/frame_{:010d}.jpg'
    gulp_subfolder = ('rgb_train' if mode ==
                      'train' or mode == 'val' else 'rgb_' + mode)
    gulp_path = '/proj/vondrick/datasets/epic-kitchens/data/processed/gulp/' + gulp_subfolder
    epic_inst = list(EpicVideoDataset(gulp_path, 'verb+noun'))
    epic_inst.sort(key=(lambda k: k.video_id +
                        '_{:010d}'.format(k.start_frame)))
    # maps (verb_A, noun_A, verb_B, noun_B) to list of clip metadata
    result = defaultdict(list)

    # Loop over all pairs of action segments
    for i in range(len(epic_inst) - 1):
        segment_A = epic_inst[i]
        segment_B = epic_inst[i + 1]
        # Proceed only if same video
        if segment_A.video_id != segment_B.video_id:
            continue
        # Set action pair as key
        cur_key = (segment_A.verb_class, segment_A.noun_class,
                   segment_B.verb_class, segment_B.noun_class)
        # Append clip
        cur_item = (segment_A.video_id, segment_A.start_frame, segment_A.num_frames,
                    segment_B.start_frame -
                    (segment_A.start_frame + segment_A.num_frames),
                    segment_B.num_frames)
        result[cur_key].append(cur_item)

    return result


def get_action_encoder(class_type='verb'):
    ''' Returns a dictionary that allows you to call encoder['put'] to obtain 1. '''
    action_file = '/proj/vondrick/datasets/epic-kitchens/data/annotations/EPIC_' + \
        class_type + '_classes.csv'
    action_df = pd.read_csv(action_file, sep=',', header=0)
    result = dict()
    for _, row in action_df.iterrows():
        act_id, act_name = row[0], row[1]
        act_id = int(act_id)  # let id start from 0
        result[act_name] = act_id


def get_action_encoder(class_type='verb'):
    ''' Returns a dictionary that allows you to call encoder['put'] to obtain 1. '''
    action_file = '/proj/vondrick/datasets/epic-kitchens/data/annotations/EPIC_' + \
        class_type + '_classes.csv'
    action_df = pd.read_csv(action_file, sep=',', header=0)
    result = dict()
    for _, row in action_df.iterrows():
        act_id, act_name = row[0], row[1]
        act_id = int(act_id)  # let id start from 0
        result[act_name] = act_id
    return result


def get_action_decoder(class_type='verb'):
    ''' Returns a dictionary that allows you to call decoder[2] to obtain 'open'. '''
    action_file = '/proj/vondrick/datasets/epic-kitchens/data/annotations/EPIC_' + \
        class_type + '_classes.csv'
    action_df = pd.read_csv(action_file, sep=',', header=0)
    result = dict()
    for _, row in action_df.iterrows():
        act_id, act_name = row[0], row[1]
        act_id = int(act_id)  # let id start from 0
        result[act_id] = act_name
    return result
