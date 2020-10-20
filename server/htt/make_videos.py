import csv
import pickle
import os
import cv2

from skvideo.io import FFmpegWriter
from tqdm import tqdm


# try to incorporate multiprocessing
import multiprocessing
from multiprocessing import Process
from multiprocessing import JoinableQueue as Queue


def save_video(metadata_chunk):
    '''
    Load csv file containg information about data
    '''

    fps = 60
    df = metadata_chunk

    for index, row in tqdm(df.iterrows(), total=len(df.index)):

        # input directory
        input_base_directory = '/proj/vondrick/datasets/epic-kitchens/data/raw/rgb/'
        input_directory = os.path.join(
            input_base_directory, row['participant_id'], row['video_id'])

        # output directory and filename
        output_base_directory = '/proj/vondrick/datasets/epic-kitchens/data/videos_htt/rgb/'
        output_directory = os.path.join(
            output_base_directory, row['participant_id'], row['video_id'])
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        start_frame = row['start_frame'] - 210
        stop_frame = row['start_frame'] - 60
        # output_filename = os.path.join(output_directory, str(
        #     index) + '_' + row['verb'] + '_' + row['noun'] + '.mp4')
        output_filename = os.path.join(output_directory, row['video_id'] + '_' + row['verb'] + '_' + row['noun'] + '_' + str(start_frame) + '_' + str(stop_frame) + '.mp4')

        # try writing the file as mp4
        #writer = FFmpegWriter(output_filename, inputdict={'-r': str(fps), '-vcodec': 'h264', '-pix_fmt': 'yuv420p'})
        writer = FFmpegWriter(output_filename, inputdict={'-r': str(fps)})
        for i in range(start_frame, stop_frame + 1):
            frame_path = os.path.join(
                input_directory, 'frame_' + str(i).zfill(10) + '.jpg')
            frame = cv2.imread(frame_path)
            # frame here
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.writeFrame(frame)
        writer.close()
    q.task_done()


if __name__ == '__main__':

    processes = 28

    metadata_chunk = []
    with open("/proj/vondrick/datasets/epic-kitchens/data/annotations/EPIC_train_action_labels.pkl", "rb") as f:
        df = pickle.load(f)

    df = df[df['start_frame'] > 210]
    for x, df_x in df.groupby('participant_id'):
        metadata_chunk.append(df_x)

    q = Queue()
    for i in range(processes):
        q.put(i)
        multiprocessing.Process(target=save_video,
                                args=(metadata_chunk[i],)).start()
    q.join()
