import os
import sys
import time
import re
import argparse
import numpy as np
from tqdm import tqdm

# visualization
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
plt.switch_backend('agg')

sys.path.append('../utils')
# from dataset_3d import *
from model_3d import *

# backbone
from resnet_2d3d import neq_load_customized

# data augmentation methods
from augmentation import *

# collect results
from utils import AverageMeter, save_checkpoint, denorm, calc_topk_accuracy

import torch
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms

# saving log, calculating accuracy etc.
import torchvision.utils as vutils

from dpc_util import *

# Hyperbolic neural network package
import geoopt
from present_matching import *


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--model', default='dpc-rnn', type=str)
parser.add_argument('--dataset', default='ucf101', type=str)

# frame preprocessing
parser.add_argument('--seq_len', default=5, type=int,
                    help='number of frames in each video block')
parser.add_argument('--num_seq', default=8, type=int,
                    help='number of video blocks')
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--ds', default=3, type=int,
                    help='frame downsampling rate')

# optimizer learning rate and decay
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')

# resume training
parser.add_argument('--resume', default='', type=str,
                    help='path of model to resume')
parser.add_argument('--pretrain', default='', type=str,
                    help='path of pretrained model')
parser.add_argument('--epochs', default=300, type=int,
                    help='number of total epochs to run')

# help in re-setting the lr to appropriate value
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')

# gpu setting
parser.add_argument('--gpu', default='0,1', type=str)


parser.add_argument('--print_freq', default=1, type=int,
                    help='frequency of printing output during training')
parser.add_argument('--reset_lr', action='store_true',
                    help='Reset learning rate when resume training?')
parser.add_argument('--prefix', default='tmp', type=str,
                    help='prefix of checkpoint filename')

# which layers to train [ use for fine-tuning ]
parser.add_argument('--train_what', default='all', type=str)
parser.add_argument('--img_dim', default=128, type=int)

# distance metric in the embedding space
parser.add_argument('--distance', default='dot', type=str)
parser.add_argument('--num_clips', default=10000, type=int, help='Number of sequences to sample in total from train_split')
parser.add_argument('--drive', default='ssd', type=str, help='Type of disk drive training data is located at')

# regression or DPC
parser.add_argument('--regression', default=False, action='store_true', help='If true, perform regression on prediction and ground truth embeddings, similar to http://openaccess.thecvf.com/content_cvpr_2016/papers/Vondrick_Anticipating_Visual_Representations_CVPR_2016_paper.pdf')

# Present matching parameters
parser.add_argument('--pm_start', default=1000, type=int,
                    help='Number of epochs present matching kicks in')
parser.add_argument('--pm_freq', default=10, type=int,
                    help='Frequency of present matching (every 10 epochs by default)')
parser.add_argument('--pm_num_samples', default=100000, type=int,
                    help='Number of videos sequences to sample during present matching')
parser.add_argument('--pm_num_ngbrs', default=4, type=int,
                    help='Number of nearest neighbors to put in one batch')
parser.add_argument('--pm_gap_thres', default=30, type=int,
                    help='Minimum pair-wise frame gap between two sequences in the same batch')

def main():

    # set to constant for consistant results
    torch.manual_seed(0)
    np.random.seed(0)

    global args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # ensure to disable present matching during regression
    if args.regression:
        assert args.pm_start > args.epochs
    
    global cuda
    cuda = torch.device('cuda')
    args.cuda = cuda

    ### dpc model ###
    if args.model == 'dpc-rnn':
        model = DPC_RNN(sample_size=args.img_dim,
                        num_seq=args.num_seq,
                        seq_len=args.seq_len,
                        network=args.net,
                        pred_step=args.pred_step,
                        distance=args.distance)
    else:
        raise ValueError('wrong model!')

    # parallelize the model
    model = nn.DataParallel(model)
    model = model.to(cuda)

    # loss function (change this)
    # why cross-entropy, we can take a direct cosine distance instead
    global criterion
    criterion = nn.CrossEntropyLoss()
    if args.regression:
        criterion = nn.MSELoss()

    ### optimizer ###
    # dont' think we need to use 'last' keyword during pre-training anywhere
    if args.train_what == 'last':
        for name, param in model.module.resnet.named_parameters():
            param.requires_grad = False
    else:
        pass  # train all layers

    # check if gradient flowing to appropriate layers
    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    params = model.parameters()
    if args.distance == 'poincare':
        print('Using Riemmanian Adam optimizer')
#         optimizer = geoopt.optim.RiemannianSGD(params, lr=0.01, momentum=0.9)
#         optimizer = geoopt.optim.RiemannianAdam(params, lr=args.lr, weight_decay=args.wd)
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    args.old_lr = None

    best_acc = 0
    global iteration
    iteration = 0

    ### restart training ###
    if args.resume:
        if os.path.isfile(args.resume):
            # load the old model and set the learning rate accordingly

            # get the old learning rate
            args.old_lr = float(re.search('_lr(.+?)_', args.resume).group(1))
            print("=> loading resumed checkpoint '{}'".format(args.resume))

            #
            checkpoint = torch.load(
                args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])

            # load old optimizer, start with the corresponding learning rate
            if not args.reset_lr:  # if didn't reset lr, load old optimizer
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                # reset to new learning rate
                print('==== Change lr from %f to %f ====' %
                      (args.old_lr, args.lr))
            print("=> loaded resumed checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("[Warning] no checkpoint found at '{}'".format(args.resume))

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrain))
            checkpoint = torch.load(
                args.pretrain, map_location=torch.device('cpu'))

            # neq_load_customized
            model = neq_load_customized(model, checkpoint['state_dict'])
            print("=> loaded pretrained checkpoint '{}' (epoch {})"
                  .format(args.pretrain, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    ### load data ###
    transform = get_transform(args)

    train_loader = get_data(args, transform, 'train')
    val_loader = get_data(args, transform, 'val')

    # setup tools

    # denormalize to display input images via tensorboard
    global de_normalize
    de_normalize = denorm()

    global img_path
    img_path, model_path, pm_cache = set_path(args)
    global writer_train

    # Train & validate for multiple epochs
    # book-keeping
    writer_val = SummaryWriter(os.path.join(img_path, 'val'))
    writer_val_pm = SummaryWriter(os.path.join(img_path, 'val_pm'))
    writer_train = SummaryWriter(os.path.join(img_path, 'train'))

    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):

        if epoch < args.pm_start:
            print('\n\nTraining with randomly sampled sequences from %s' %
                  args.dataset)
            train_loss, train_acc, train_accuracy_list = train(train_loader, model, optimizer, epoch, args.regression)

        if epoch >= args.pm_start:

            # change torch multithreading setting to circumvent 'open files' limit
            torch.multiprocessing.set_sharing_strategy('file_system')

            args.dataset = 'epic_present_matching'

            print('\n\nTraining with matched sequences (nearest neighbours) from %s\n\n' % args.dataset)
            if epoch % args.pm_freq == 0:
                print(' ######################################## \n Matching sequences in embedding space... \n ########################################\n\n')

                # present matching and save results for future training
                present_matching(args, model, pm_cache, epoch, 'train', dataset='epic_unlabeled')
                present_matching(args, model, pm_cache, epoch, 'val', dataset='epic_unlabeled')

            # present matching training
            pm_path = os.path.join(pm_cache, 'epoch_train_%d' % (
                epoch - epoch % args.pm_freq))  # retrieve last present matching results
            train_loader_pm = get_data(
                args, transform, mode='train', pm_path=pm_path, epoch=epoch)
            train_loss, train_acc, train_accuracy_list = train(train_loader_pm, model, optimizer, epoch)
            del train_loader_pm

            # present matching validation
            pm_path = os.path.join(pm_cache, 'epoch_val_%d' % (
                epoch - epoch % args.pm_freq))  # retrieve last present matching results
            val_loader_pm = get_data(
                args, transform, mode='val', pm_path=pm_path, epoch=epoch)

            val_loss_pm, val_acc_pm, val_accuracy_list_pm = validate(val_loader_pm, model, epoch)
            del val_loader_pm

        # normal validation
        val_loss, val_acc, val_accuracy_list = validate(val_loader, model, epoch, args.regression)

        # Train curves
        writer_train.add_scalar('global/loss', train_loss, epoch)
        writer_train.add_scalar('global/accuracy', train_acc, epoch)

        # Val curves
        writer_val.add_scalar('global/loss', val_loss, epoch)
        writer_val.add_scalar('global/accuracy', val_acc, epoch)
        
        if epoch >= args.pm_start:
            # Add present matching curves
            writer_val_pm.add_scalar('global/loss', val_loss_pm, epoch)
            writer_val_pm.add_scalar('global/accuracy', val_acc_pm, epoch)

        # Train accuracies
        writer_train.add_scalar('accuracy/top1', train_accuracy_list[0], epoch)
        writer_train.add_scalar('accuracy/top3', train_accuracy_list[1], epoch)
        writer_train.add_scalar('accuracy/top5', train_accuracy_list[2], epoch)

        # Val accuracies
        writer_val.add_scalar('accuracy/top1', val_accuracy_list[0], epoch)
        writer_val.add_scalar('accuracy/top3', val_accuracy_list[1], epoch)
        writer_val.add_scalar('accuracy/top5', val_accuracy_list[2], epoch)

        if epoch >= args.pm_start:
            # Add present matching curves
            writer_val_pm.add_scalar('accuracy/top1', val_accuracy_list_pm[0], epoch)
            writer_val_pm.add_scalar('accuracy/top3', val_accuracy_list_pm[1], epoch)
            writer_val_pm.add_scalar('accuracy/top5', val_accuracy_list_pm[2], epoch)

        # save check_point (best accuracy measured without encoder)
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({'epoch': epoch + 1,
                         'net': args.net,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'iteration': iteration},
                        is_best, filename=os.path.join(model_path, 'epoch%s.pth.tar' % str(epoch + 1)), save_every=10)

    print('Training from ep %d to ep %d finished' %
          (args.start_epoch, args.epochs))


def process_output(mask):
    '''task mask as input, compute the target for contrastive loss'''
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu'''
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg

    # [4, 3, 16, 4, 3, 16]
    # B and B2 must be same always ??
    (B, NP, SQ, B2, NS, _) = mask.size()  # [B, P, SQ, B, N, SQ]
    #print (B,B2,NS,NP,SQ)

    # mark out the positives

    # not using temporal negatives and spatial negatives differently
    target = mask == 1
    target.requires_grad = False
    return target, (B, B2, NS, NP, SQ)


def train(data_loader, model, optimizer, epoch, regression):

    # average losses
    losses = AverageMeter()

    # average accuracy
    accuracy = AverageMeter()

    # top-1, top-3 and top-5 accuracy
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]

    model.train()
    global iteration

    #print (len(data_loader))
    for idx, mbatch in enumerate(data_loader):

        tic = time.time()
        input_seq = mbatch['t_seq']
        vpath = mbatch['vpath']
        idx_block = mbatch['idx_block']
        input_seq = input_seq.to(cuda)
        B = input_seq.size(0)
        if regression:
            [pred_, gt_, score_, mask_] = model(input_seq, regression=True)
        else:
            [score_, mask_] = model(input_seq)

        # visualize the input sequence to the network
        if (iteration == 0) or (iteration == args.print_freq):
            if B > 8:
                input_seq = input_seq[0:8, :]
            writer_train.add_image('input_seq',
                                   de_normalize(vutils.make_grid(
                                       input_seq.transpose(2, 3).contiguous(
                                       ).view(-1, 3, args.img_dim, args.img_dim),
                                       nrow=args.num_seq * args.seq_len)),
                                   iteration)
        del input_seq

        # why only for idx 0 ?
        if idx == 0:
            target_, (_, B2, NS, NP, SQ) = process_output(mask_)

        # score is a 6d tensor: [B, P, SQ, B, N, SQ]
        # Number Predicted NP, Number Sequence NS
        score_flattened = score_.view(B * NP * SQ, B2 * NS * SQ)
        target_flattened = target_.view(B * NP * SQ, B2 * NS * SQ)
        #print(score_flattened.shape, target_flattened.shape)

        # added
        target_flattened = target_flattened.double()
        target_flattened = target_flattened.argmax(dim=1)
        #print(score_flattened.shape, target_flattened.shape)
        #print(score_flattened[0,:], target_flattened)

        # loss function, here cross-entropy for DPC
        if regression:
            loss = criterion(pred_, gt_) * len(pred_) # criterion = nn.MSELoss during regression
        else:
            loss = criterion(score_flattened, target_flattened)
        top1, top3, top5 = calc_topk_accuracy(
            score_flattened, target_flattened, (1, 3, 5))
        # break
        accuracy_list[0].update(top1.item(), B)
        accuracy_list[1].update(top3.item(), B)
        accuracy_list[2].update(top5.item(), B)

        losses.update(loss.item(), B)
        accuracy.update(top1.item(), B)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del loss

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.6f} ({loss.local_avg:.4f})\t'
                  'Acc: top1 {3:.4f}; top3 {4:.4f}; top5 {5:.4f} T:{6:.2f}\t'.format(
                      epoch, idx, len(data_loader), top1, top3, top5, time.time() - tic, loss=losses) + 
                  'Average %s distance: %0.6f' % (args.distance, -torch.mean(score_).item()))

            writer_train.add_scalar('local/loss', losses.val, iteration)
            writer_train.add_scalar('local/accuracy', accuracy.val, iteration)

            iteration += 1
        del score_
        
    return losses.avg, accuracy.avg, [i.avg for i in accuracy_list]


def validate(data_loader, model, epoch, regression):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    model.eval()

    with torch.no_grad():

        for idx, mbatch in tqdm(enumerate(data_loader), total=len(data_loader)):

            input_seq = mbatch['t_seq']
            vpath = mbatch['vpath']
            idx_block = mbatch['idx_block']
            input_seq = input_seq.to(cuda)
            B = input_seq.size(0)
            if regression:
                [pred_, gt_, score_, mask_] = model(input_seq, regression=True)
            else:
                [score_, mask_] = model(input_seq)
            del input_seq

            if idx == 0:
                target_, (_, B2, NS, NP, SQ) = process_output(mask_)

            # [B, P, SQ, B, N, SQ]
            score_flattened = score_.view(B * NP * SQ, B2 * NS * SQ)
            target_flattened = target_.view(B * NP * SQ, B2 * NS * SQ)

            target_flattened = target_flattened.double()
            target_flattened = target_flattened.argmax(dim=1)

            if regression:
                loss = criterion(pred_, gt_) * len(pred_) # criterion = nn.MSELoss during regression
            else:
                loss = criterion(score_flattened, target_flattened)
                
            top1, top3, top5 = calc_topk_accuracy(
                score_flattened, target_flattened, (1, 3, 5))

            losses.update(loss.item(), B)
            accuracy.update(top1.item(), B)

            accuracy_list[0].update(top1.item(), B)
            accuracy_list[1].update(top3.item(), B)
            accuracy_list[2].update(top5.item(), B)

    print('[{0}/{1}] Loss {loss.local_avg:.4f}\t'
          'Acc: top1 {2:.4f}; top3 {3:.4f}; top5 {4:.4f} \t'.format(
              epoch, args.epochs, *[i.avg for i in accuracy_list], loss=losses))
    return losses.avg, accuracy.avg, [i.avg for i in accuracy_list]



def set_path(args):
    if args.resume:
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        # TODO pm args?
        exp_path = 'log_{args.prefix}/{args.dataset}-{args.img_dim}_{0}_{args.model}_\
bs{args.batch_size}_lr{1}_seq{args.num_seq}_pred{args.pred_step}_len{args.seq_len}_ds{args.ds}_\
train-{args.train_what}{2}'.format(
            'r%s' % args.net[6::],
            args.old_lr if args.old_lr is not None else args.lr,
            '_pt=%s' % args.pretrain.replace(
                '/', '-') if args.pretrain else '',
            args=args)
        if 0 <= args.pm_start and args.pm_start <= args.epochs:
            exp_path += '_pm-start{args.pm_start}_pm-freq{args.pm_freq}_pm-ngbrs{args.pm_num_ngbrs}'.format(args=args)

    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    pm_cache = os.path.join(exp_path, 'pm_cache')
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(pm_cache):
        os.makedirs(pm_cache)
    return img_path, model_path, pm_cache


if __name__ == '__main__':
    main()