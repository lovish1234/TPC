# calculate average, confusion matrix, accruacy

import torch
import numpy as np
import os
import ntpath
import sys

from datetime import datetime
import glob
import matplotlib.pyplot as plt

plt.switch_backend('agg')

from collections import deque
from tqdm import tqdm
from torchvision import transforms


def save_checkpoint(state, is_best=0, filename='models/checkpoint.pth.tar', save_every=10):

    # save after every few epochs
    if state['epoch'] % save_every == 0:
        torch.save(state, filename)

    # last_epoch_path = os.path.join(os.path.dirname(filename),
    #                                'epoch%s.pth.tar' % str(state['epoch']-gap))
    # if not keep_all:
    #     try: os.remove(last_epoch_path)
    #     except: pass

    if is_best:
        past_best = glob.glob(os.path.join(
            os.path.dirname(filename), 'model_best_*.pth.tar'))
        for i in past_best:
            try:
                os.remove(i)
            except:
                pass
        torch.save(state, os.path.join(os.path.dirname(filename),
                                       'model_best_epoch%s.pth.tar' % str(state['epoch'])))


def write_log(content, epoch, filename):
    if not os.path.exists(filename):
        log_file = open(filename, 'w')
    else:
        log_file = open(filename, 'a')
    log_file.write('## Epoch %d:\n' % epoch)
    log_file.write('time: %s\n' % str(datetime.now()))
    log_file.write(content + '\n\n')
    log_file.close()


def calc_topk_accuracy(output, target, topk=(1,)):
    '''
    Modified from: https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels, 
    calculate top-k accuracies.
    '''
    maxk = max(topk)
    batch_size = target.size(0)

    # take the indices of topk values
    values, pred = output.topk(maxk, 1, True, True)
    #print (values[0], pred[0])

    pred = pred.t()
    #print (pred, type(pred), pred.shape)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res


def calc_bottomk_accuracy(output, target, topk=(1,)):
    '''
    Modified from: https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels, 
    calculate top-k accuracies.
    '''
    maxk = max(topk)
    batch_size = target.size(0)

    # take the indices of topk values
    values, pred = output.topk(maxk, 1, False, True)
    #print (values[0], pred[0])

    pred = pred.t()
    #print (pred, type(pred), pred.shape)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res


def calc_accuracy(output, target):
    ''' output: (B, N) or list thereof; target: (B) or list thereof.
    Supports multi-type classification as well, returning joint accuracy as the last item. '''

    if isinstance(output, list) and len(output) >= 2:
        # Multiple class types
        accuracies = []
        preds = []
        for cur_out, cur_tar in zip(output, target):
            cur_tar = cur_tar.squeeze()
            _, cur_pred = torch.max(cur_out, dim=1)
            cur_acc = torch.mean((cur_pred == cur_tar).float()).item()
            accuracies.append(cur_acc)
            preds.append(cur_pred)

        # Calculate joint accuracy as well
        B = output[0].shape[0]
        class_divisions = len(output)
        both_correct = 0
        for i in range(B):
            if np.all([preds[k][i].item() == target[k][i].item() for k in range(class_divisions)]):
                both_correct += 1
        joint_acc = both_correct / B
        accuracies.append(joint_acc)
        return accuracies

    else:
        # Single class type
        if isinstance(output, list):
            output = output[0]
            target = target[0]
        target = target.squeeze()
        _, pred = torch.max(output, 1)
        return torch.mean((pred == target).float())


def calc_accuracy_binary(output, target):
    '''output, target: (B, N), output is logits, before sigmoid '''
    pred = output > 0
    acc = torch.mean((pred == target.byte()).float())
    del pred, output, target
    return acc


def denorm(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    assert len(mean) == len(std) == 3
    inv_mean = [-mean[i] / std[i] for i in range(3)]
    inv_std = [1 / i for i in std]
    return transforms.Normalize(mean=inv_mean, std=inv_std)


def get_action_from_frame(map_action_frame, vpath, frame_index):
    # frame_id = vpath.split('/')[-1] + '_' + str(frame_index)
    video_id = ntpath.basename(vpath).split('.')[0]  # remove path & extension, if any
    frame_id = video_id + '_' + str(frame_index)
    action = map_action_frame.get(frame_id, None)
    if action == None:
        return 'None' + ' ' + 'None'
    else:
        return action.get_action()


class single_action:
    x = 5

    def __init__(self, verb, noun):
        self.verb = verb
        self.noun = noun

    def get_action(self):
        return(self.verb + ' ' + self.noun)

    def get_verb(self):
        return self.verb

    def get_noun(self):
        return self.noun


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.local_history = deque([])
        self.local_avg = 0
        self.history = []
        self.dict = {}  # save all data values here
        self.save_dict = {}  # save mean and std here, for summary table

    def update(self, val, n=1, history=0, step=64):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if history:
            self.history.append(val)
        if step > 0:
            self.local_history.append(val)
            if len(self.local_history) > step:
                self.local_history.popleft()
            self.local_avg = np.average(self.local_history)

    def dict_update(self, val, key):
        if key in self.dict.keys():
            self.dict[key].append(val)
        else:
            self.dict[key] = [val]

    def __len__(self):
        return self.count


class AccuracyTable(object):
    '''compute accuracy for each class'''

    def __init__(self):
        self.dict = {}

    def update(self, pred, tar):
        pred = torch.squeeze(pred)
        tar = torch.squeeze(tar)
        for i, j in zip(pred, tar):
            i = int(i)
            j = int(j)
            if j not in self.dict.keys():
                self.dict[j] = {'count': 0, 'correct': 0}
            self.dict[j]['count'] += 1
            if i == j:
                self.dict[j]['correct'] += 1

    def print_table(self, label):
        for key in self.dict.keys():
            acc = self.dict[key]['correct'] / self.dict[key]['count']
            print('%s: %2d, accuracy: %3d/%3d = %0.6f'
                  % (label, key, self.dict[key]['correct'], self.dict[key]['count'], acc))


class ConfusionMeter(object):
    '''compute and show confusion matrix'''

    def __init__(self, num_class):
        self.num_class = num_class
        self.mat = np.zeros((num_class, num_class))
        self.precision = []
        self.recall = []

    def update(self, pred, tar):
        pred, tar = pred.cpu().numpy(), tar.cpu().numpy()
        pred = np.squeeze(pred)
        tar = np.squeeze(tar)
        for p, t in zip(pred.flat, tar.flat):
            self.mat[p][t] += 1

    def print_mat(self):
        print('Confusion Matrix: (target in columns)')
        print(self.mat)

    def plot_mat(self, path, dictionary=None, annotate=False):
        plt.figure(dpi=600)
        plt.imshow(self.mat,
                   cmap=plt.cm.jet,
                   interpolation=None,
                   extent=(0.5, np.shape(self.mat)[0] + 0.5, np.shape(self.mat)[1] + 0.5, 0.5))
        width, height = self.mat.shape
        if annotate:
            for x in range(width):
                for y in range(height):
                    plt.annotate(str(int(self.mat[x][y])), xy=(y + 1, x + 1),
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 fontsize=8)

        if dictionary is not None:
            plt.xticks([i + 1 for i in range(width)],
                       [dictionary[i] for i in range(width)],
                       rotation='vertical')
            plt.yticks([i + 1 for i in range(height)],
                       [dictionary[i] for i in range(height)])
        plt.xlabel('Ground Truth')
        plt.ylabel('Prediction')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path, format='svg')
        plt.clf()

        # for i in range(width):
        #     if np.sum(self.mat[i,:]) != 0:
        #         self.precision.append(self.mat[i,i] / np.sum(self.mat[i,:]))
        #     if np.sum(self.mat[:,i]) != 0:
        #         self.recall.append(self.mat[i,i] / np.sum(self.mat[:,i]))
        # print('Average Precision: %0.4f' % np.mean(self.precision))
        # print('Average Recall: %0.4f' % np.mean(self.recall))


class data_prefetcher:
    '''
    Open a cuda side stream to prefetch data from next iteration, this process
    is overlapping with forward pass in the model, thus accelerating dataloading
    process.

    NOTE:
    1. a cuda variable specifying cuda device and a dataloader is required as an input
    2. pin_memory in the datalaoder needs to be set to True before passed in
    '''

    def __init__(self, loader, cuda):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload(cuda)

    def preload(self, cuda):
        try:
            self.input_dict = next(self.loader)
            self.input_seq = self.input_dict['t_seq']
        except StopIteration:
            self.input_dict = None
            self.input_seq = None
            return
        with torch.cuda.stream(self.stream):
            self.input_seq = self.input_seq.to(cuda, non_blocking=True)

    def next(self, cuda):
        torch.cuda.current_stream().wait_stream(self.stream)
        input_dict = self.input_dict
        input_seq = self.input_seq
        self.preload(cuda)
        return input_dict, input_seq
