# Self-supervised pretraining using DPC with VAE

import os
import random
import sys
import time
import re
import argparse
import ntpath
import numpy as np
import pickle
from tqdm import tqdm
import copy

# visualization
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
plt.switch_backend('agg')

sys.path.append('../tpc')  # use same dataset file as TPC
sys.path.append('../utils')
sys.path.append('../utils/diversity_meas')
from div_vae import *
from cvae_model import *
from vrnn_model import *
from dataset_epic import *
from dataset_synthepic import *
from dataset_other import *
from dataset_toy import *

# backbone
from resnet_2d3d import neq_load_customized

# data augmentation methods
from augmentation import *

# collect results
from utils import AverageMeter, save_checkpoint, denorm, calc_topk_accuracy, single_action, get_action_from_frame

import torch
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
from torch import autograd  # anomaly detection
from present_matching import *
import copy

# saving log, calculating accuracy etc.
import torchvision.utils as vutils

from train_util import *

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--dataset', default='k400', type=str, help='k400 / ucf / epic_unlabeled / epic_before / epic_end')

# frame preprocessing
parser.add_argument('--seq_len', default=5, type=int,
                    help='number of frames in each video block')
parser.add_argument('--num_seq', default=8, type=int,
                    help='number of video blocks')
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--ds', default=6, type=int,
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
parser.add_argument('--prefix', default='conv', type=str,
                    help='prefix of checkpoint filename')

# which layers to train [ use for fine-tuning ]
parser.add_argument('--train_what', default='all', type=str)
parser.add_argument('--img_dim', default=128, type=int)

# number of sequences sampled from one video in epic and block_toy dataset
parser.add_argument('--num_clips', default=10000, type=int,
                    help='Number of sequences to sample in total from train_split')
parser.add_argument('--drive', default='ssd', type=str,
                    help='Type of disk drive training data is located at')
parser.add_argument('--new_augment', action='store_true',
                    help='Use data augmentation with consistent train-test distortion')

# CVAE / diversity specific parameters
parser.add_argument('--model', default='vrnn', type=str, help='cvae / vrnn / vrnn-i')
parser.add_argument('--cvae_arch', default='conv_e', type=str,
                    help='Architecture of the CVAE mapping present to future (fc / conv_a / conv_b / conv_c / conv_d / conv_e) (default: conv_e)')
parser.add_argument('--vrnn_latent_size', default=16, type=int,
                    help='Dimensionality of the VRNN probabilistic latent space')
parser.add_argument('--vrnn_kernel_size', default=1, type=int,
                    help='Kernel size of all VRNN convolutional layers (prior, enc, dec, rnn)')
parser.add_argument('--vrnn_dropout', default=0.1, type=float,
                    help='Dropout in RNN for aggregation (GRU cell) (default: 0.1)')
parser.add_argument('--vae_kl_weight', default=1.0, type=float,
                    help='Base weight factor of KL divergence term in VAE-DPC loss (default: 1.0)')
parser.add_argument('--vae_no_kl_norm', default=False, action='store_true',
                    help='If True, disable KL loss weight normalization')
parser.add_argument('--vae_encoderless_epochs', default=25, type=int,
                    help='Number of epochs to train the prediction network deterministically (DPC-like) before enabling the encoder and beta warmup (default: 100)')
parser.add_argument('--vae_inter_kl_epochs', default=25, type=int,
                    help='Number of epochs to wait after starting the encoder (and training with zero KL loss) before starting beta warmup (default: 50)')
parser.add_argument('--vae_kl_beta_warmup', default=150, type=int,
                    help='Number of epochs during which to warmup the KL loss term weight from 0% to 100% according to a cosine curve (default: 100)')
parser.add_argument('--pred_divers_wt', default=0.001, type=float,
                    help='Importance of variance (diversity) in predictions as measured by cosine distance in loss (default: 0.1)')
parser.add_argument('--pred_divers_formula', default='inv',
                    type=str, help='Use -var (neg) or 1/var (inv) (default: inv)')
parser.add_argument('--test_diversity', default=False, action='store_true',
                    help='If True, evaluate model by measuring diversity of multiple generated samples. Note: specify pretrained model path as well.')
parser.add_argument('--paths', default=20, type=int,
                    help='Future paths generated by VAE (default: 20)')
# parser.add_argument('--vrnn_context_noise', default=0.0, type=float,
#                     help='Add Gaussian noise to input of VRNN decoder for regularization (default: 0)')
# parser.add_argument('--vrnn_time_indep', default=False, action='store_true',
#                     help='If True, z does not see or influence context (i.e. VRNN becomes CVAE)')
parser.add_argument('--select_best_among', default=1, type=int,
                    help='[do not use for now] During validation and when the encoder is disabled, generate the specified number of futures in a holistic way, and judge the best performing path only (default = 1).')
parser.add_argument('--force_context_dropout', default=False, action='store_true',
                    help='If True, use different dropout mask (p = 0.1) on c_t for every path during uncertainty measurements.')

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

# Action classification custom arguments
# parser.add_argument('--diverse_actions', default=False, action='store_true', help='If True, evaluate model by measuring diversity of multiple generated future samples (labels in action space). Note: specify TEST model path as well.')
# parser.add_argument('--test_path', default='', type=str, help='Path of finetuned model for multimodal action classification')
# parser.add_argument('--class_type', default='verb', type=str) # TODO: support 'both' as well


def main():

    # Set constant random state for consistent results
    torch.manual_seed(704)
    np.random.seed(704)
    random.seed(704)

    global args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    global cuda
    cuda = torch.device('cuda')
    args.cuda = cuda

    ### dpc model ###
    if args.model == 'cvae':
        model = DPC_CVAE(img_dim=args.img_dim,
                         num_seq=args.num_seq,
                         seq_len=args.seq_len,
                         pred_step=args.pred_step,
                         network=args.net,
                         cvae_arch=args.cvae_arch)
    elif 'vrnn' in args.model:
        model = DPC_VRNN(img_dim=args.img_dim,
                         num_seq=args.num_seq,
                         seq_len=args.seq_len,
                         pred_step=args.pred_step,
                         network=args.net,
                         latent_size=args.vrnn_latent_size,
                         kernel_size=args.vrnn_kernel_size,
                         rnn_dropout=args.vrnn_dropout,
                         time_indep='-i' in args.model)
    else:
        raise ValueError('Unknown / wrong model: ' + args.model)

    # parallelize the model
    model = nn.DataParallel(model)
    model = model.to(cuda)

    # loss function (change this)
    # why cross-entropy, we can take a direct cosine distance instead
    criterion = nn.CrossEntropyLoss()

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

    ### load transform & dataset ###
    transform_train = get_transform(args, mode = 'train')
    transform_val = get_transform(args, mode = 'val')

    train_loader = get_data(args, transform_train, 'train')
    val_loader = get_data(args, transform_val, 'val')

    # setup tools

    # Initialize denormalize transform to display input images via tensorboard
    de_normalize = denorm()

    # Get paths
    global img_path
    img_path, model_path, divers_path, pm_cache = set_path(args)

    global writer_train

    if not(args.test_diversity):

        # Train & validate for multiple epochs
        # book-keeping
        writer_val_enc = SummaryWriter(os.path.join(img_path, 'val_enc'))
        writer_val_noenc = SummaryWriter(os.path.join(img_path, 'val_noenc'))
        writer_val_pm_enc = SummaryWriter(os.path.join(img_path, 'val_pm_enc'))
        writer_val_pm_noenc = SummaryWriter(os.path.join(img_path, 'val_pm_noenc'))
        writer_train = SummaryWriter(os.path.join(img_path, 'train'))
        global cur_vae_kl_weight, cur_pred_divers_wt

        ### main loop ###
        for epoch in range(args.start_epoch, args.epochs):
            # Initially train without latent space if specified
            if epoch < args.vae_encoderless_epochs:
                train_with_latent = False
                cur_pred_divers_wt = 0.0 # encouraging diversity doesn't apply right now
            else:
                train_with_latent = True
                cur_pred_divers_wt = args.pred_divers_wt # hard transition
            print('Encoder enabled for training this epoch:', train_with_latent)

            # Then train for a while with beta = 0 (not explicitly coded)

            # Beta warmup = VAE KL loss weight adjustment over time
            if train_with_latent:
                epoch_post = epoch - (args.vae_encoderless_epochs + args.vae_inter_kl_epochs)
                if epoch_post < 0:
                    cur_vae_kl_weight = 0.0
                elif 0 <= epoch_post and epoch_post < args.vae_kl_beta_warmup:
                    cur_vae_kl_weight = args.vae_kl_weight * \
                        (1.0 - np.cos(epoch_post / args.vae_kl_beta_warmup * np.pi)) / 2
                else:
                    cur_vae_kl_weight = args.vae_kl_weight
            else:
                cur_vae_kl_weight = 0.0
            print('Current VAE KL loss term weight:', cur_vae_kl_weight)

            if epoch < args.pm_start:
                print('\n\nTraining with randomly sampled sequences from %s' %
                      args.dataset)
                train_loss_dpc, train_loss_kl, train_loss_divers, train_loss, train_acc, train_accuracy_list = train(
                    train_loader, model, optimizer, epoch, criterion, writer_train, args, de_normalize,
                    cur_vae_kl_weight, cur_pred_divers_wt, do_encode=train_with_latent)

            else:

                # change torch multithreading setting to circumvent 'open files' limit
                torch.multiprocessing.set_sharing_strategy('file_system')
                
                print('\n\nTraining with matched sequences (nearest neighbours) from %s\n\n' % args.dataset)
                if epoch % args.pm_freq == 0:
                    print(' ######################################## \n Matching sequences in embedding space... \n ########################################\n\n')
                    
                    # present matching and save results for future training
                    present_matching(args, model, pm_cache, epoch, 'train')
                    present_matching(args, model, pm_cache, epoch, 'val')
    
                
                args_pm = copy.copy(args)
                args_pm.dataset = 'epic_present_matching'
            
                pm_path = os.path.join(pm_cache, 'epoch_train_%d' % (
                    epoch - epoch % args_pm.pm_freq))  # retrieve last present matching results
                train_loader_pm = get_data(
                    args_pm, transform_train, mode='train', pm_path=pm_path, epoch=epoch)
                train_loss_dpc, train_loss_kl, train_loss_divers, train_loss, train_acc, train_accuracy_list = train(
                    train_loader_pm, model, optimizer, epoch, criterion, writer_train, args, de_normalize,
                    cur_vae_kl_weight, cur_pred_divers_wt, do_encode=train_with_latent)
                del train_loader_pm

                pm_path = os.path.join(pm_cache, 'epoch_val_%d' % (
                    epoch - epoch % args_pm.pm_freq))  # retrieve last present matching results
                val_loader_pm = get_data(
                    args_pm, transform_val, mode='val', pm_path=pm_path, epoch=epoch)
                
                val_loss_dpc_pm_enc, val_loss_kl_pm_enc, val_loss_divers_pm_enc, \
                    val_loss_pm_enc, val_acc_pm_enc, val_acc_pm_enc_list = validate(
                    val_loader_pm, model, epoch, args, criterion, cur_vae_kl_weight,
                    cur_pred_divers_wt, do_encode=train_with_latent,
                    select_best_among=1 if train_with_latent else args.select_best_among) # only one path when encoding
                
                val_loss_dpc_pm_noenc, val_loss_kl_pm_noenc, val_loss_divers_pm_noenc, \
                    val_loss_pm_noenc, val_acc_pm_noenc, val_acc_pm_noenc_list = validate(
                    val_loader_pm, model, epoch, args, criterion, cur_vae_kl_weight,
                    cur_pred_divers_wt, do_encode=False, select_best_among=args.select_best_among)
                del val_loader_pm

            val_loss_dpc_enc, val_loss_kl_enc, val_loss_divers_enc, \
                val_loss_enc, val_acc_enc, val_acc_enc_list = validate(
                val_loader, model, epoch, args, criterion, cur_vae_kl_weight,
                cur_pred_divers_wt, do_encode=train_with_latent,
                select_best_among=1 if train_with_latent else args.select_best_among) # only one path when encoding

            val_loss_dpc_noenc, val_loss_kl_noenc, val_loss_divers_noenc, \
                val_loss_noenc, val_acc_noenc, val_acc_noenc_list = validate(
                val_loader, model, epoch, args, criterion, cur_vae_kl_weight,
                cur_pred_divers_wt, do_encode=False, select_best_among=args.select_best_among)
            
            # Train curves
            writer_train.add_scalar('global/loss_dpc', train_loss_dpc, epoch)
            writer_train.add_scalar('global/loss_vae_kl', train_loss_kl, epoch)
            writer_train.add_scalar('global/loss_vae_divers', train_loss_divers, epoch)
            writer_train.add_scalar('global/loss', train_loss, epoch)
            writer_train.add_scalar('global/vae_kl_weight', cur_vae_kl_weight, epoch)
            writer_train.add_scalar('global/accuracy', train_acc, epoch)

            # Val curves
            writer_val_enc.add_scalar('global/loss_dpc', val_loss_dpc_enc, epoch)
            writer_val_enc.add_scalar('global/loss_vae_kl', val_loss_kl_enc, epoch)
            writer_val_enc.add_scalar('global/loss_vae_divers', val_loss_divers_enc, epoch)
            writer_val_enc.add_scalar('global/loss', val_loss_enc, epoch)
            writer_val_enc.add_scalar('global/accuracy', val_acc_enc, epoch)
            writer_val_noenc.add_scalar('global/loss_dpc', val_loss_dpc_noenc, epoch)
            writer_val_noenc.add_scalar('global/loss_vae_kl', val_loss_kl_noenc, epoch)
            writer_val_noenc.add_scalar('global/loss_vae_divers', val_loss_divers_noenc, epoch)
            writer_val_noenc.add_scalar('global/loss', val_loss_noenc, epoch)
            writer_val_noenc.add_scalar('global/accuracy', val_acc_noenc, epoch)

            if epoch >= args.pm_start:
                # Add present matching curves
                writer_val_pm_enc.add_scalar('global/loss_dpc', val_loss_dpc_pm_enc, epoch)
                writer_val_pm_enc.add_scalar('global/loss_vae_kl', val_loss_kl_pm_enc, epoch)
                writer_val_pm_enc.add_scalar('global/loss_vae_divers', val_loss_divers_pm_enc, epoch)
                writer_val_pm_enc.add_scalar('global/loss', val_loss_pm_enc, epoch)
                writer_val_pm_enc.add_scalar('global/accuracy', val_acc_pm_enc, epoch)
                writer_val_pm_noenc.add_scalar('global/loss_dpc', val_loss_dpc_pm_noenc, epoch)
                writer_val_pm_noenc.add_scalar('global/loss_vae_kl', val_loss_kl_pm_noenc, epoch)
                writer_val_pm_noenc.add_scalar('global/loss_vae_divers', val_loss_divers_pm_noenc, epoch)
                writer_val_pm_noenc.add_scalar('global/loss', val_loss_pm_noenc, epoch)
                writer_val_pm_noenc.add_scalar('global/accuracy', val_acc_pm_noenc, epoch)

            # Train accuracies
            writer_train.add_scalar('accuracy/top1', train_accuracy_list[0], epoch)
            writer_train.add_scalar('accuracy/top3', train_accuracy_list[1], epoch)
            writer_train.add_scalar('accuracy/top5', train_accuracy_list[2], epoch)

            # Val accuracies
            writer_val_noenc.add_scalar('accuracy/top1', val_acc_noenc_list[0], epoch)
            writer_val_noenc.add_scalar('accuracy/top3', val_acc_noenc_list[1], epoch)
            writer_val_noenc.add_scalar('accuracy/top5', val_acc_noenc_list[2], epoch)
            writer_val_enc.add_scalar('accuracy/top1', val_acc_enc_list[0], epoch)
            writer_val_enc.add_scalar('accuracy/top3', val_acc_enc_list[1], epoch)
            writer_val_enc.add_scalar('accuracy/top5', val_acc_enc_list[2], epoch)
            
            if epoch >= args.pm_start:
                # Add present matching curves
                writer_val_pm_noenc.add_scalar('accuracy/top1', val_acc_pm_noenc_list[0], epoch)
                writer_val_pm_noenc.add_scalar('accuracy/top3', val_acc_pm_noenc_list[1], epoch)
                writer_val_pm_noenc.add_scalar('accuracy/top5', val_acc_pm_noenc_list[2], epoch)
                writer_val_pm_enc.add_scalar('accuracy/top1', val_acc_pm_enc_list[0], epoch)
                writer_val_pm_enc.add_scalar('accuracy/top3', val_acc_pm_enc_list[1], epoch)
                writer_val_pm_enc.add_scalar('accuracy/top5', val_acc_pm_enc_list[2], epoch)

            # save check_point (best accuracy measured without encoder)
            is_best = val_acc_noenc > best_acc
            best_acc = max(val_acc_noenc, best_acc)
            save_checkpoint({'epoch': epoch + 1,
                             'net': args.net,
                             'state_dict': model.state_dict(),
                             'best_acc': best_acc,
                             'optimizer': optimizer.state_dict(),
                             'iteration': iteration},
                            is_best, filename=os.path.join(model_path, 'epoch%s.pth.tar' % str(epoch + 1)), save_every=10)

        print('Training from ep %d to ep %d finished' %
              (args.start_epoch, args.epochs))

    else:

        # Uncertainty evaluation: full video & no color adjustments
        # NOTE: be careful with training augmentation to prevent train-test resolution / scaling discrepancy
        tf_diversity = transforms.Compose([
            RandomHorizontalFlip(consistent=True),
            CenterCrop(224),
            Scale(size=(args.img_dim, args.img_dim)),
            ToTensor(),
            Normalize()
        ])

        print('Measuring diversity of generated samples...')
        val_divers_loader = get_data(args, tf_diversity, 'val')
        results = measure_vae_uncertainty_loader(val_divers_loader, model, args.paths,
                                                 print_freq=args.print_freq, collect_actions=False,
                                                 force_context_dropout=args.force_context_dropout)
        cur_path = os.path.join(divers_path, 'epoch' + str(checkpoint['epoch']) + \
                   '_paths' + str(args.paths) + \
                   ('_ctdrop' if args.force_context_dropout else '') + '.p')
        with open(cur_path, 'wb') as f:
            pickle.dump(results, f)
        print('For future use, uncertainty evaluation results stored to ' + cur_path)


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


def loss_kl_divergence(mu, logvar, normalization=True):
    '''
    Calculates KL divergence loss term for CVAE.
    mu, logvar: from one item within minibatch, same size as latent space.
    NOTE: normalization enabled by default as of 05/27, divides by batch_size * pred_step.
    '''
    # mu and logvar are probably many-dimensional, but this code is agnostic of that
    loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    if normalization:
        loss /= (mu.shape[0] * mu.shape[1])
    return loss


def loss_kl_divergence_prior(mu_prio, mu_post, logvar_prio, logvar_post, normalization=True):
    '''
    Calculates KL divergence loss term for VRNN.
    all arguments: from one item within minibatch, same size as latent space.
    NOTE: normalization enabled by default as of 05/27, divides by batch_size * pred_step.
    '''
    std_prio = logvar_prio.mul(0.5).exp()
    std_post = logvar_post.mul(0.5).exp()
    loss = -0.5 * torch.sum(1 + logvar_post - logvar_prio - (std_post.pow(2) + (mu_post - mu_prio).pow(2)) / std_prio.pow(2))
    if normalization:
        loss /= (mu_prio.shape[0] * mu_prio.shape[1])
    return loss


def loss_pred_variance(pred_cos_sim, latent_dist, pred_divers_formula):
    '''
    Calculates diversity loss term for CVAE and VRNN.
    pred_cos_sim: cosine similarity of one pair, from one item within minibatch and one time step.
    latent_dist: L2 distance between latent vectors that generated the pair.
    As of 05/20, multiply by latent_dist.
    '''
    # pred_cos_sim is tensor with a cosine distance from a random pair
    # lower is better; typical variance range was [0.999990, 0.999998] if not encouraged (old model)
    # mean_cos_sim = pred_cos_sim.mean() # across time steps

    # TODO: normalize by latent_dist? what about mode collapse in prior?

    delta = 1.0 - pred_cos_sim
    if pred_divers_formula == 'neg':
        loss = -delta
    elif pred_divers_formula in ['div', 'inv']:
        # causes diminishing returns for larger variance
        loss = 1.0 / (delta + 1e-6)
    else:
        raise Exception('Unknown diversity loss formula: ', pred_divers_formula)

    loss *= latent_dist.detach() / 5.0 # typical L2 between 8/16-dim randn x, y

    return loss


if __name__ == '__main__':
    main()
