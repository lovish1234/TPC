# This file is for action classification / anticipation on DPC, TPC, or CVAE-DPC / VRNN

import os
import hashlib
import pickle
import random
import sys
import time
import argparse
import re
import ast
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

sys.path.append('../utils')
sys.path.append('../utils/diversity_meas')
sys.path.append('../backbone')

from dataset_epic import *
from dataset_other import *
from dataset_toy import *
from model_lc_future import *  # DPC / TPC / CVAE / VRNN future
from model_lc_present import *  # DPC / TPC / CVAE / VRNN present
from resnet_2d3d import neq_load_customized
from augmentation import *
from div_vae import *
from utils import AverageMeter, ConfusionMeter, save_checkpoint, write_log, calc_topk_accuracy, denorm, calc_accuracy

import torch
import torch.optim as optim
from torch.utils import data
import torch.nn as nn
from torchvision import datasets, models, transforms
import torchvision.utils as vutils

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='resnet34', type=str)
parser.add_argument('--dataset', default='', type=str)

# note that the test split is fixed by default
parser.add_argument('--split', default=1, type=int)

parser.add_argument('--seq_len', default=5, type=int)
# IMPORTANT: change to 5 for present classification
parser.add_argument('--num_seq', default=8, type=int)
parser.add_argument('--pred_step', default=3, type=int)

# NOTE: this changes according to the dataset
parser.add_argument('--num_class', default=101, type=int)

parser.add_argument('--dropout', default=0.5, type=float)

parser.add_argument('--ds', default=6, type=int)
parser.add_argument('--batch_size', default=4, type=int)

# optimizer
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Initial learning rate')
parser.add_argument('--wd', default=1e-3, type=float,
                    help='Weight decay factor')
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--pretrain', default='random', type=str,
                    help='Path to pretrained model (or random)')
parser.add_argument('--test', default='', type=str,
                    help='Path to fine-tuned classification model')

parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--print_freq', default=5, type=int)
parser.add_argument('--reset_lr', action='store_true',
                    help='Reset learning rate when resume training?')
parser.add_argument('--train_what', default='ft',
                    type=str, help='Train what parameters?')
parser.add_argument('--prefix', default='tmp', type=str)
parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument('--num_workers', default=16, type=int,
                    help='PyTorch multi-threading number')

parser.add_argument('--class_type', default='both',
                    type=str, help='verb / noun / both')

# CUSTOM arguments:
parser.add_argument('--model', required=True, type=str,
                    help='dpc / tpc / cvae / vrnn / vrnn-i')
parser.add_argument('--temporality', default='future', type=str,
                    help='future = action anticipation / present = action classification (according to original DPC schema) (default: future)')
parser.add_argument('--label_fraction', default=1.0, type=float, help='Fraction of labels to use for finetuning after self-supervised representation learning (default: 1.0 = all)')
parser.add_argument('--new_augment', action='store_true',
                    help='Use data augmentation with consistent train-test distortion')
parser.add_argument('--store_results', default=False, action='store_true',
                    help='If True, skip training loop, validate model with validation dataset and save (1) softmax output (2) predicted labels (verb/noun). To use this feature, provide path to pretrained model in args.resume')
parser.add_argument('--oracle_samples', default='1', type=str,
                    help='During test and validation and when the encoder is disabled, generate the specified number of futures in a holistic way, and judge the best performing item only, for every item in a batch. This follows the oracle loss principle and can be a list e.g. 1,2,3,5 (default = 1).')

# CVAE / VRNN arguments:
parser.add_argument('--cvae_arch', default='conv_e', type=str,
                    help='Architecture of the CVAE mapping present to future (fc / conv_a / conv_b / conv_c / conv_d / conv_e)')
parser.add_argument('--vae_encode_train', default=False, action='store_true',
                    help='If True, force encode ground truth + input into Gaussian parameters for z during training, which is kind of cheating')
parser.add_argument('--vrnn_latent_size', default=16, type=int,
                    help='Dimensionality of the VRNN probabilistic latent space')
parser.add_argument('--vrnn_kernel_size', default=1, type=int,
                    help='Kernel size of all VRNN convolutional layers (prior, enc, dec, rnn)')
parser.add_argument('--vrnn_dropout', default=0.1, type=float,
                    help='Dropout in RNN for aggregation (GRU cell) (default: 0.1)')
parser.add_argument('--diverse_actions', default=False, action='store_true',
                    help='If True, evaluate VAE model by measuring diversity of multiple generated future samples (labels in action space). Note: specify TEST model path as well.')
parser.add_argument('--paths', default=20, type=int,
                    help='Future samples per time step generated by VAE (default: 20).')
parser.add_argument('--force_context_dropout', default=False, action='store_true',
                    help='If True, use different dropout mask (p = 0.1) on c_t for every path during uncertainty measurements.')
# parser.add_argument('--vrnn_time_indep', default=False, action='store_true',
#                     help='If True, z does not see or influence context (i.e. VRNN becomes CVAE)')

def main():

    # Set constant random state for consistent results
    torch.manual_seed(1776)
    np.random.seed(1776)
    random.seed(1776)

    global args, cuda, class_types
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cuda = torch.device('cuda')

    # Verify arguments & set number of classes
    if not(args.model in ['dpc', 'tpc', 'cvae'] or 'vrnn' in args.model):
        raise Exception('Unknown model: ' + args.model)
    if not (args.temporality in ['present', 'future']):
        raise Exception('Unknown temporality: ' + args.temporality)

    class_types = 1
    if args.dataset == 'ucf101':
        args.num_class = 101
    elif args.dataset == 'hmdb51':
        args.num_class = 51
    elif args.dataset == 'ucf11':
        args.num_class = 11
    elif 'epic' in args.dataset:
        if args.class_type == 'verb':
            args.num_class = 125
        elif args.class_type == 'noun':
            args.num_class = 352
        elif args.class_type in ['both', 'action', 'verbnoun', 'verb+noun']:
            args.num_class = [125, 352]
            class_types = 2
        else:
            raise Exception('Unknown class type: ' + args.class_type)
    else:
        raise Exception('Unknown dataset: ' + args.dataset)

    if (args.temporality == 'present' and args.num_seq == 8) or \
            (args.temporality == 'future' and args.num_seq == 5):
        print()
        print('===> WARNING / SANITY CHECK: I think the arguments temporality=' + args.temporality +
              ' and num_seq=' + str(args.num_seq) + ' are not what you intended')
        print()

    if (args.temporality == 'present' and args.diverse_actions):
        raise ValueError(
            'It is impossible to generate diverse samples of the present')
    if (args.diverse_actions and args.store_results):
        raise ValueError(
            'Conflicting arguments: please generate either diverse actions or store detailed output results, but cannot do both')

    ### classifier model ###
    print('Temporality:', args.temporality, ' Model:', args.model)
    if args.temporality == 'present':
        # Present / action classification
        # NOTE: same model class for all models because no future prediction is involved, EXCEPT vrnn (= different rnn)
        if not('vrnn' in args.model):
            print('Initializing LC_present...')
            model = LC_present(sample_size=args.img_dim,
                               num_seq=args.num_seq,
                               seq_len=args.seq_len,
                               network=args.net,
                               num_class=args.num_class,
                               dropout=args.dropout)
        else:
            print('Initializing LC_present_VRNN...')
            model = LC_present_VRNN(sample_size=args.img_dim,
                                    num_seq=args.num_seq,
                                    seq_len=args.seq_len,
                                    network=args.net,
                                    latent_size=args.vrnn_latent_size,
                                    kernel_size=args.vrnn_kernel_size,
                                    rnn_dropout=args.vrnn_dropout,
                                    cls_dropout=args.dropout,
                                    num_class=args.num_class,
                                    time_indep='-i' in args.model)
    else:
        # Future / action anticipation => extra pred_step argument & different architectures
        if args.model == 'dpc':
            print('Initializing LC_future_DPC...')
            model = LC_future_DPC(sample_size=args.img_dim,
                                  num_seq=args.num_seq,
                                  seq_len=args.seq_len,
                                  network=args.net,
                                  num_class=args.num_class,
                                  dropout=args.dropout,
                                  pred_step=args.pred_step)
        elif args.model == 'tpc':
            print('Initializing LC_future_TPC...')
            model = LC_future_TPC(sample_size=args.img_dim,
                                  num_seq=args.num_seq,
                                  seq_len=args.seq_len,
                                  network=args.net,
                                  num_class=args.num_class,
                                  dropout=args.dropout,
                                  pred_step=args.pred_step)
        elif args.model == 'cvae':
            print('Initializing LC_future_CVAE...')
            model = LC_future_CVAE(sample_size=args.img_dim,
                                   num_seq=args.num_seq,
                                   seq_len=args.seq_len,
                                   pred_step=args.pred_step,
                                   network=args.net,
                                   cvae_arch=args.cvae_arch,
                                   num_class=args.num_class,
                                   dropout=args.dropout)
        elif 'vrnn' in args.model:
            print('Initializing LC_future_VRNN...')
            model = LC_future_VRNN(sample_size=args.img_dim,
                                   num_seq=args.num_seq,
                                   seq_len=args.seq_len,
                                   pred_step=args.pred_step,
                                   network=args.net,
                                   latent_size=args.vrnn_latent_size,
                                   kernel_size=args.vrnn_kernel_size,
                                   rnn_dropout=args.vrnn_dropout,
                                   num_class=args.num_class,
                                   cls_dropout=args.dropout,
                                   time_indep='-i' in args.model)

    model = nn.DataParallel(model)
    model = model.to(cuda)

    # loss function
    global criterion
    criterion = nn.CrossEntropyLoss()

    ### optimizer ###
    params = None
    if args.train_what == 'ft':
        print('=> finetune backbone with smaller lr')
        params = []
        for name, param in model.module.named_parameters():
            # if ('resnet' in name) or ('rnn' in name) or ('vae' in name):
            if 'enc_' in name or 'prior_' in name:
                # Completely exclude VAE encoder since no KL divergence loss exists here
                # Also exclude prior because it also influences the latent distribution
                param.requires_grad = False
                print('Zero LR:', name)
            elif 'final' in name:
                # Only final linear classification layers
                params.append({'params': param})
                print('Maintain LR:', name)
            else:
                # Everything except VAE encoder and final layers
                params.append({'params': param, 'lr': args.lr / 10})
    else:
        pass  # train all layers

    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    if params is None:
        params = model.parameters()

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    # learning rate multiplier based scheduler
    if args.dataset == 'hmdb51':
        def lr_lambda(ep): return MultiStepLR_Restart_Multiplier(
            ep, gamma=0.1, step=[150, 250, 300], repeat=1)
    elif args.dataset == 'ucf101':
        if args.img_dim == 224:
            def lr_lambda(ep): return MultiStepLR_Restart_Multiplier(
                ep, gamma=0.1, step=[300, 400, 500], repeat=1)
        else:
            def lr_lambda(ep): return MultiStepLR_Restart_Multiplier(
                ep, gamma=0.1, step=[60, 80, 100], repeat=1)
    elif args.dataset == 'ucf11':
        if args.img_dim == 224:
            def lr_lambda(ep): return MultiStepLR_Restart_Multiplier(
                ep, gamma=0.1, step=[300, 400, 500], repeat=1)
        else:
            def lr_lambda(ep): return MultiStepLR_Restart_Multiplier(
                ep, gamma=0.1, step=[60, 80, 100], repeat=1)
    elif 'epic' in args.dataset:
        if args.img_dim == 224:
            def lr_lambda(ep): return MultiStepLR_Restart_Multiplier(
                ep, gamma=0.1, step=[300, 400, 500], repeat=1)
        else:
            # def lr_lambda(ep): return MultiStepLR_Restart_Multiplier(
            #     ep, gamma=0.1, step=[60, 80, 100], repeat=1)
            def lr_lambda(ep): return MultiStepLR_Restart_Multiplier(
                ep, gamma=0.1, step=[40, 60, 80], repeat=1)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    args.old_lr = None
    best_acc = 0
    global iteration
    iteration = 0

    oracle_samples = [int(s) for s in args.oracle_samples.split(',')]

    ### restart training ###
    if args.test:

        # do data augmentation when testing on validation set
        if os.path.isfile(args.test):
            print("=> loading testing checkpoint '{}'".format(args.test))
            # checkpoint = torch.load(args.test)
            checkpoint = torch.load(args.test, map_location=torch.device('cpu'))
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                print(
                    '=> [Warning]: weight structure is not equal to test model; Use non-equal load ==')
                model = neq_load_customized(model, checkpoint['state_dict'])
            print("=> loaded testing checkpoint '{}' (epoch {})".format(
                args.test, checkpoint['epoch']))
            global num_epoch
            num_epoch = checkpoint['epoch']
        elif args.test == 'random':
            print("=> [Warning] loaded random weights")
        else:
            raise ValueError()

        # NOTE: be careful with training augmentation to prevent train-test resolution / scaling discrepancy
        # TODO: perform 'store_results' logic here
        tf_test = transforms.Compose([
            CenterCrop(224),
            Scale(size=(args.img_dim, args.img_dim)),
            ToTensor(),
            Normalize()
        ])
        tf_val = transforms.Compose([
            RandomHorizontalFlip(consistent=True),
            RandomSizedCrop(consistent=True, size=224, p=0.3),
            Scale(size=(args.img_dim, args.img_dim)),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                        hue=0.1, p=0.3, consistent=True),
            ToTensor(),
            Normalize()
        ])

        test_loader = get_data(tf_test, 'val')
        val_loader = get_data(tf_val, 'val')
        # test_loss, test_acc = test(test_loader, model)

        for cur_os in oracle_samples:

            print('Current oracle (select best among) sample count:', cur_os)

            val_with_encode = (cur_os == 1 and args.temporality == 'future') # makes no difference for present)
            if val_with_encode:
                val_loss_enc_clean, val_acc_enc_clean, val_verb_acc_enc_clean, val_noun_acc_enc_clean = \
                    validate(test_loader, model, do_encode=True, oracle_samples=1) # only one path when encoding
                print('== WITH encoder, CLEAN test transform, based on 1-sample ==')
                print('Val loss:', val_loss_enc_clean)
                print('Val joint acc:', val_acc_enc_clean)
                print('Val verb acc:', val_verb_acc_enc_clean)
                print('Val noun acc:', val_noun_acc_enc_clean)
                print()
                val_loss_enc_unc, val_acc_enc_unc, val_verb_acc_enc_unc, val_noun_acc_enc_unc = \
                    validate(val_loader, model, do_encode=True, oracle_samples=1) # only one path when encoding
                print('== WITH encoder, UNCLEAN val transform, based on 1-sample ==')
                print('Val loss:', val_loss_enc_unc)
                print('Val joint acc:', val_acc_enc_unc)
                print('Val verb acc:', val_verb_acc_enc_unc)
                print('Val noun acc:', val_noun_acc_enc_unc)
                print()

            val_loss_noenc_clean, val_acc_noenc_clean, val_verb_acc_noenc_clean, val_noun_acc_noenc_clean = \
                validate(test_loader, model, do_encode=False, oracle_samples=cur_os)
            print('== WITHOUT encoder, CLEAN test transform, based on K-sample oracle loss ==')
            print('Val loss:', val_loss_noenc_clean)
            print('Val joint acc:', val_acc_noenc_clean)
            print('Val verb acc:', val_verb_acc_noenc_clean)
            print('Val noun acc:', val_noun_acc_noenc_clean)
            print()
            val_loss_noenc_unc, val_acc_noenc_unc, val_verb_acc_noenc_unc, val_noun_acc_noenc_unc = \
                validate(val_loader, model, do_encode=False, oracle_samples=cur_os)
            print('== WITHOUT encoder, UNCLEAN val transform, based on K-sample oracle loss ==')
            print('Val loss:', val_loss_noenc_unc)
            print('Val joint acc:', val_acc_noenc_unc)
            print('Val verb acc:', val_verb_acc_noenc_unc)
            print('Val noun acc:', val_noun_acc_noenc_unc)
            print()

            if val_with_encode:
                content = 'WITH encoder, CLEAN tf, 1 sample, loss: {}\t acc_joint: {}\t acc_verb: {}\t acc_noun: {}\n'.format(
                    val_loss_enc_clean, val_acc_enc_clean, val_verb_acc_enc_clean, val_noun_acc_enc_clean) + \
                    'WITHOUT encoder, CLEAN tf, K sample, loss: {}\t acc_joint: {}\t acc_verb: {}\t acc_noun: {}\n'.format(
                    val_loss_noenc_clean, val_acc_noenc_clean, val_verb_acc_noenc_clean, val_noun_acc_noenc_clean) + \
                    'WITH encoder, UNCLEAN tf, 1 sample, loss: {}\t acc_joint: {}\t acc_verb: {}\t acc_noun: {}\n'.format(
                    val_loss_enc_unc, val_acc_enc_unc, val_verb_acc_enc_unc, val_noun_acc_enc_unc) + \
                    'WITHOUT encoder, UNCLEAN tf, K sample, loss: {}\t acc_joint: {}\t acc_verb: {}\t acc_noun: {}\n'.format(
                    val_loss_noenc_unc, val_acc_noenc_unc, val_verb_acc_noenc_unc, val_noun_acc_noenc_unc)
            else:
                content = 'WITHOUT encoder, CLEAN tf, K sample, loss: {}\t acc_joint: {}\t acc_verb: {}\t acc_noun: {}\n'.format(
                    val_loss_noenc_clean, val_acc_noenc_clean, val_verb_acc_noenc_clean, val_noun_acc_noenc_clean) + \
                    'WITHOUT encoder, UNCLEAN tf, K sample, loss: {}\t acc_joint: {}\t acc_verb: {}\t acc_noun: {}\n'.format(
                    val_loss_noenc_unc, val_acc_noenc_unc, val_verb_acc_noenc_unc, val_noun_acc_noenc_unc)

            write_log(content=content, epoch=num_epoch,
                filename=os.path.join(os.path.dirname(args.test), 'test_log_oracle{}_bs{}_gpu{}.md'.format(cur_os, args.batch_size, args.gpu)))

        sys.exit()

    else:  # not test
        torch.backends.cudnn.benchmark = True

    if args.resume:
        if os.path.isfile(args.resume):
            args.old_lr = float(re.search('_lr(.+?)_', args.resume).group(1))
            print("=> loading resumed checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            if not args.reset_lr:  # if didn't reset lr, load old optimizer
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('==== Change lr from %f to %f ====' %
                      (args.old_lr, args.lr))
            iteration = checkpoint['iteration']
            print("=> loaded resumed checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if (not args.resume) and args.pretrain:

        # for fine-tuning with random weights
        if args.pretrain == 'random':
            print('=> using random weights')

        # findtune using a self-supervised pre-trained model
        elif os.path.isfile(args.pretrain):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrain))
            checkpoint = torch.load(
                args.pretrain, map_location=torch.device('cpu'))
            model = neq_load_customized(model, checkpoint['state_dict'])
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(
                args.pretrain, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    ### load transform & data ###
    # TODO: work on new_augment
    tf_train = transforms.Compose([
        RandomHorizontalFlip(consistent=True),
        RandomSizedCrop(consistent=True, size=224, p=1.0),
        Scale(size=(args.img_dim, args.img_dim)),
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5,
                    hue=0.25, p=0.3, consistent=True),
        ToTensor(),
        Normalize()
    ])
    tf_val = transforms.Compose([
        RandomHorizontalFlip(consistent=True),
        RandomSizedCrop(consistent=True, size=224, p=0.3),
        Scale(size=(args.img_dim, args.img_dim)),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                    hue=0.1, p=0.3, consistent=True),
        ToTensor(),
        Normalize()
    ])

    train_loader = get_data(tf_train, 'train')
    val_loader = get_data(tf_val, 'val')

    # setup tools
    global de_normalize
    de_normalize = denorm()
    global img_path
    img_path, model_path, divers_path = set_path(args)
    global writer_train

    # The following code decides what exactly to do

    if args.store_results:

        print('Storing all outputs of the validation set in detail, for evaluation and comparison purposes...')
        print('== WARNING: using VALIDATION transform which is NOT as clean as test transform! ==')
        # TODO: use test() method instead (see args.test), which uses cleaner data transform
        # & supports more evaluation metrics (confusion matrix etc.)
        validate(val_loader, model, store_results=True, oracle_samples=oracle_samples[0])

    elif not(args.diverse_actions):

        print('Finetuning the pretrained model on action ' +
              ('classification' if args.temporality == 'present' else 'anticipation') + '...')
        print('Number of class types:', class_types, args.num_class)
        try:  # old version
            writer_val_enc = SummaryWriter(
                log_dir=os.path.join(img_path, 'val_enc'))
            writer_val_noenc = SummaryWriter(
                log_dir=os.path.join(img_path, 'val_noenc'))
            writer_train = SummaryWriter(
                log_dir=os.path.join(img_path, 'train'))
        except:  # v1.7
            writer_val_enc = SummaryWriter(
                logdir=os.path.join(img_path, 'val_enc'))
            writer_val_noenc = SummaryWriter(
                logdir=os.path.join(img_path, 'val_noenc'))
            writer_train = SummaryWriter(
                logdir=os.path.join(img_path, 'train'))

        ### main loop ###
        for epoch in range(args.start_epoch, args.epochs):
            # PyTorch: self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
            writer_train.add_scalar(
                'learning_rate', get_current_learn_rate(optimizer), epoch)

            if class_types > 1:
                # ==== Multiple class types ====
                train_loss, train_acc, train_verb_acc, train_noun_acc = \
                    train(train_loader, model, optimizer, epoch,
                          do_encode=args.vae_encode_train)
                val_loss_enc, val_acc_enc, val_verb_acc_enc, val_noun_acc_enc = \
                    validate(val_loader, model, do_encode=True, oracle_samples=1) # only one path when encoding
                val_loss_noenc, val_acc_noenc, val_verb_acc_noenc, val_noun_acc_noenc = \
                    validate(val_loader, model, do_encode=False, oracle_samples=oracle_samples[0])

                writer_train.add_scalar('global/loss', train_loss, epoch)
                writer_train.add_scalar(
                    'global/acc_verb', train_verb_acc, epoch)
                writer_train.add_scalar(
                    'global/acc_noun', train_noun_acc, epoch)
                writer_train.add_scalar('global/acc_joint', train_acc, epoch)
                writer_val_enc.add_scalar('global/loss', val_loss_enc, epoch)
                writer_val_enc.add_scalar(
                    'global/acc_verb', val_verb_acc_enc, epoch)
                writer_val_enc.add_scalar(
                    'global/acc_noun', val_noun_acc_enc, epoch)
                writer_val_enc.add_scalar(
                    'global/acc_joint', val_acc_enc, epoch)
                writer_val_noenc.add_scalar(
                    'global/loss', val_loss_noenc, epoch)
                writer_val_noenc.add_scalar(
                    'global/acc_verb', val_verb_acc_noenc, epoch)
                writer_val_noenc.add_scalar(
                    'global/acc_noun', val_noun_acc_noenc, epoch)
                writer_val_noenc.add_scalar(
                    'global/acc_joint', val_acc_noenc, epoch)

            else:
                # ==== Single class type ====
                train_loss, train_acc = train(
                    train_loader, model, optimizer, epoch, do_encode=args.vae_encode_train)
                val_loss_enc, val_acc_enc = validate(
                    val_loader, model, do_encode=True, oracle_samples=1) # only one path when encoding
                val_loss_noenc, val_acc_noenc = validate(
                    val_loader, model, do_encode=False, oracle_samples=oracle_samples[0])

                writer_train.add_scalar('global/loss', train_loss, epoch)
                writer_train.add_scalar('global/accuracy', train_acc, epoch)
                writer_val_enc.add_scalar('global/loss', val_loss_enc, epoch)
                writer_val_enc.add_scalar(
                    'global/accuracy', val_acc_enc, epoch)
                writer_val_noenc.add_scalar(
                    'global/loss', val_loss_noenc, epoch)
                writer_val_noenc.add_scalar(
                    'global/accuracy', val_acc_noenc, epoch)

            scheduler.step(epoch)

            # save check_point (best accuracy measured without encoder)
            is_best = val_acc_noenc > best_acc
            best_acc = max(val_acc_noenc, best_acc)

            # save the model corresponding to the best val accuracy
            save_checkpoint({
                'epoch': epoch + 1,
                'net': args.net,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'iteration': iteration
            }, is_best, filename=os.path.join(model_path, 'epoch%s.pth.tar' % str(epoch + 1)))

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
        val_divers_loader = get_data(tf_diversity, 'val')

        print('Gathering action classification outputs + contexts + predictions of diverse generated samples...')	
        results = measure_vae_uncertainty_loader(val_divers_loader, model, args.paths,
                                                 print_freq=args.print_freq, collect_actions=True,
                                                 force_context_dropout=args.force_context_dropout)
        cur_path = os.path.join(divers_path, 'epoch' + str(checkpoint['epoch']) + \
                   '_paths' + str(args.paths) + \
                   ('_ctdrop' if args.force_context_dropout else '') + '.p')
        with open(cur_path, 'wb') as f:
            pickle.dump(results, f)
        print('For future use, action label uncertainty results stored to ' + cur_path)


def get_current_learn_rate(optimizer):
    res = 0.0
    for param_group in optimizer.param_groups:
        res = np.max([param_group['lr'], res])  # to avoid returning LR / 10
    return res


def train(data_loader, model, optimizer, epoch, do_encode=False):
    '''
    Train for one epoch.
    '''
    losses = AverageMeter()
    accuracy = AverageMeter()  # also functions as joint
    acc_verb = AverageMeter()
    acc_noun = AverageMeter()
    model.train()
    global iteration, class_types, cuda

    for idx, mbatch in enumerate(data_loader):
        tic = time.time()

        input_seq = mbatch['t_seq']
        input_seq = input_seq.to(cuda)
        if class_types > 1:
            # Multiple class types
            target_verb = mbatch['verb_class'].to(cuda)
            target_noun = mbatch['noun_class'].to(cuda)
            target = [target_verb, target_noun]
        else:
            # Single class type
            target = mbatch['label'].to(cuda)

        # batch size
        B = input_seq.size(0)

        # most often: (action_output, context)
        # future VAE if encoding: (action_output, context, mus, logvars)
        if args.model == 'dpc':
            output = model(input_seq)[0]
        else:
            output = model(input_seq, do_encode=do_encode)[0]  # tensor or list of tensors

        # visualize the set of images fed as input
        if (iteration == 0) or (iteration == args.print_freq):

            # visualize and all video sequences in batch
            if B > 8:
                input_seq = input_seq[0:8, :]
            writer_train.add_image('input_seq',
                                   de_normalize(vutils.make_grid(
                                       input_seq.transpose(2, 3).contiguous(
                                       ).view(-1, 3, args.img_dim, args.img_dim),
                                       nrow=args.num_seq * args.seq_len)),
                                   iteration)
        del input_seq

        if class_types > 1:
            # Multiple class types
            for k in range(class_types):
                [_, N, D] = output[k].size()
                output[k] = output[k].view(B * N, D)
                target[k] = target[k].repeat(1, N).view(-1)
                cur_loss = criterion(output[k], target[k])
                if k == 0:
                    loss = cur_loss
                else:
                    loss += cur_loss  # simply sum together

        else:
            # Single class type
            if isinstance(output, list):
                output = output[0]
            [_, N, D] = output.size()
            output = output.view(B * N, D)
            target = target.repeat(1, N).view(-1)
            loss = criterion(output, target)

        losses.update(loss.item(), B)
        acc = calc_accuracy(output, target)  # tensor or list of floats
        del target

        if class_types > 1:
            # Multiple class types
            acc_verb.update(acc[0], B)
            acc_noun.update(acc[1], B)
            accuracy.update(acc[2], B)
        else:
            # Single class type
            accuracy.update(acc.item(), B)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % args.print_freq == 0:
            writer_train.add_scalar('local/loss', losses.val, iteration)

            if class_types > 1:
                # Multiple class types
                print('Epoch: [{0}][{1}/{2}]\t Loss {loss.val:.4f} ({loss.local_avg:.4f})\t Time: {3:.2f}\t'.format(
                    epoch, idx, len(data_loader), time.time() - tic, loss=losses))
                print('Acc verb {acc_verb.val:.4f} ({acc_verb.local_avg:.4f})\t'
                      ' noun {acc_noun.val:.4f} ({acc_noun.local_avg:.4f})\t'
                      ' joint {accuracy.val:.4f} ({accuracy.local_avg:.4f})'.format(
                          acc_verb=acc_verb, acc_noun=acc_noun, accuracy=accuracy))
                writer_train.add_scalar(
                    'local/acc_verb', acc_verb.val, iteration)
                writer_train.add_scalar(
                    'local/acc_noun', acc_noun.val, iteration)
                writer_train.add_scalar(
                    'local/acc_joint', accuracy.val, iteration)

            else:
                # Single class type
                print('Epoch: [{0}][{1}/{2}]\t'
                      'loss {loss.val:.4f} ({loss.local_avg:.4f})\t'
                      'acc: {acc.val:.4f} ({acc.local_avg:.4f}) time: {3:.2f}\t'.format(
                          epoch, idx, len(data_loader), time.time() - tic,
                          loss=losses, acc=accuracy))
                writer_train.add_scalar(
                    'local/accuracy', accuracy.val, iteration)

            # weight decay
            total_weight = 0.0
            decay_weight = 0.0
            for m in model.parameters():
                if m.requires_grad:
                    decay_weight += m.norm(2).data
                total_weight += m.norm(2).data
            print('Decay weight / Total weight: %.3f/%.3f' %
                  (decay_weight, total_weight))

            iteration += 1

    if class_types > 1:
        return losses.avg, accuracy.avg, acc_verb.avg, acc_noun.avg
    else:
        return losses.avg, accuracy.avg


def validate_once(model, input_seq, target, do_encode):
    '''
    Forward input to produce one sequence of outputs.
    Returns loss of whole batch as well as per sequence.
    '''
    global args, class_types, criterion

    B = input_seq.size(0)
    # (action_output, context)
    if args.model == 'dpc':
        output = model(input_seq)[0]
    else:
        output = model(input_seq, do_encode=do_encode)[0]  # tensor or list of tensors
    del input_seq

    loss_items = torch.zeros(B)

    if class_types > 1:
        # Multiple class types
        for k in range(class_types):
            [_, N, D] = output[k].size()
            output[k] = output[k].view(B * N, D)
            target[k] = target[k].repeat(1, N).view(-1)

            # Calculate batch-level loss
            cur_loss = criterion(output[k], target[k])
            if k == 0:
                loss = cur_loss.clone()
            else:
                loss += cur_loss # simply sum together across class types

            # Calculate item-level losses
            for i in range(B):
                cur_loss_items = criterion(output[k][i].view(1, -1), target[k][i].view(1)) # target = batch index -> class index
                if k == 0:
                    loss_items[i] = cur_loss_items.clone()
                else:
                    loss_items[i] += cur_loss_items # simply sum together across class types

    else:
        # Single class type
        if isinstance(output, list):
            output = output[0]
        [_, N, D] = output.size()
        output = output.view(B * N, D)
        target = target.repeat(1, N).view(-1)

        # Calculate batch-level loss
        loss = criterion(output, target)
        
        # Calculate item-level losses
        for i in range(B):
            loss_items[i] = criterion(output[i].view(1, -1), target[i].view(1, -1))

    return output, loss, loss_items


def validate(data_loader, model, do_encode=False, store_results=False, oracle_samples=1):
    '''
    Validate for one epoch.
    do_encode: If True, use VAE encoder (passes ground truth through latent space) for evaluation.
    store_results: TODO Ruoshi.
    oracle_samples: If >1, only evaluate the best performing set of samples that was generated within every video sequence.
    NOTE: The video prediction protocol is treated holistically here, so under the hood,
    a triplet of randomly guided latent vectors all need to be favorable for this to succeed.
    '''
    # TODO: also return confusion matrix if specified, see test()
    # if isinstance(oracle_samples, int):
    #     oracle_samples = [oracle_samples]

    if do_encode:
        print('[test.py] validate() WITH encoder')
        if oracle_samples > 1:
            raise ValueError('It is almost certainly a mistake to generate multiple paths with VAE encoding enabled')
    else:
        print('[test.py] validate() WITHOUT encoder, select best among:', oracle_samples)

    global args, class_types, criterion, cuda
    losses = AverageMeter()
    accuracy = AverageMeter()  # also functions as joint
    acc_verb = AverageMeter()
    acc_noun = AverageMeter()
    model.eval()

    if store_results:
        # Store results in these variables
        # Initialize lists of empty tensors (length = 1 for single class type)
        output_all = list()
        pred_all = list()
        target_all = list()
        for k in range(class_types):
            output_all.append(torch.cuda.FloatTensor())
            pred_all.append(torch.cuda.FloatTensor())
            target_all.append(torch.cuda.FloatTensor())

    with torch.no_grad():
        for idx, mbatch in tqdm(enumerate(data_loader), total=len(data_loader)):

            input_seq = mbatch['t_seq']
            input_seq = input_seq.to(cuda)
            B = input_seq.size(0)
            if class_types > 1:
                # Multiple class types
                target_verb = mbatch['verb_class'].to(cuda)
                target_noun = mbatch['noun_class'].to(cuda)
                target = [target_verb, target_noun]
            else:
                # Single class type
                target = mbatch['label'].to(cuda)

            if oracle_samples == 1:
                # Should be identical to the code below, but try this for experimentation
                output, loss, _ = validate_once(model, input_seq, target, do_encode)

            else:
                # Try out multiple paths and select the best performing one, for every item within batch
                best_output = None
                min_loss_items = torch.ones(B) * 1e12
                for j in range(oracle_samples):
                    output, loss, loss_items = validate_once(model, input_seq, target, do_encode)
                    print(j, loss)
                    for i in range(B):
                        if loss_items[i].item() < min_loss_items[i].item():
                            min_loss_items[i] = loss_items[i]
                            if best_output is None:
                                # Copy all
                                best_output = []
                                if class_types > 1:
                                    for k in range(class_types):
                                        best_output.append(output[k].clone())
                                else:
                                    best_output = output.clone()
                            else:
                                # Copy item only
                                if class_types > 1:
                                    for k in range(class_types):
                                        best_output[k][i] = output[k][i]
                                else:
                                    best_output[i] = output[i]

                output = best_output
                loss = min_loss_items.mean()

            losses.update(loss.item(), B)
            acc = calc_accuracy(output, target)  # tensor or list of floats

            if class_types > 1:
                # Multiple class types
                acc_verb.update(acc[0], B)
                acc_noun.update(acc[1], B)
                accuracy.update(acc[2], B)
            else:
                # Single class type
                accuracy.update(acc.item(), B)

            if store_results:
                # Update tensors separately for every class type
                for k in range(class_types):
                    output_all[k] = torch.cat(
                        (output_all[k], output[k]), dim=0)
                    _, pred = torch.max(output[k], 1)
                    pred_all[k] = torch.cat((pred_all[k], pred.float()), 0)
                    target_all[k] = torch.cat(
                        (target_all[k], target[k].float()), dim=0)

    print('Val Loss {loss.avg:.4f}\t'
          'Acc: {acc.avg:.4f} \t'.format(loss=losses, acc=accuracy))

    if store_results:
        # Get path from model, create new folder named val_result
        # TODO: do this in set_path() instead
        val_path = args.resume.split('/')[:-2]
        val_path.append('val_result')
        val_path = os.path.join(*val_path)
        val_path = '/' + val_path
        if not os.path.exists(val_path):
            os.makedirs(val_path)

        # Detach & convert all data to numpy
        output_all = [x.detach().cpu().numpy() for x in output_all]
        pred_all = [x.detach().cpu().numpy() for x in pred_all]
        target_all = [x.detach().cpu().numpy() for x in target_all]

        torch.save(output_all, os.path.join(val_path, 'softmax_output.pt'))
        torch.save(pred_all, os.path.join(val_path, 'prediction.pt'))
        torch.save(target_all, os.path.join(val_path, 'ground_truth.pt'))
        print('Results saved to: ' + val_path)

    if class_types > 1:
        return losses.avg, accuracy.avg, acc_verb.avg, acc_noun.avg
    else:
        return losses.avg, accuracy.avg


def test(data_loader, model, do_encode=False, oracle_samples=1):
    '''
    Test the model, calculating top1 + top5 accuracy and confusion matrix.
    '''
    # TODO: do not use this method, it does not support multiple class types (yet)
    # Migrate functionality to validate() instead

    losses = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()
    confusion_mat = ConfusionMeter(args.num_class)

    model.eval()
    with torch.no_grad():
        for idx, mbatch in tqdm(enumerate(data_loader), total=len(data_loader)):

            input_seq = mbatch['t_seq']
            input_seq = input_seq.to(cuda)
            target = mbatch['label']
            target = target.to(cuda)

            B = input_seq.size(0)
            # TODO: I really want to disable this squeeze
            input_seq = input_seq.squeeze(0)  # squeeze the '1' batch dim
            # (action_output, context)
            if args.model == 'dpc':
                output = model(input_seq)[0]
            else:
                output = model(input_seq, do_encode=do_encode)[0]

            # TODO support multiple class types

            del input_seq
            top1, top5 = calc_topk_accuracy(torch.mean(
                                            torch.mean(
                                                nn.functional.softmax(
                                                    output, 2),
                                                0), 0, keepdim=True),
                                            target, (1, 5))
            acc_top1.update(top1.item(), B)
            acc_top5.update(top5.item(), B)
            del top1, top5

            output = torch.mean(torch.mean(output, 0), 0, keepdim=True)
            loss = criterion(output, target.squeeze(-1))

            losses.update(loss.item(), B)
            del loss

            _, pred = torch.max(output, 1)
            confusion_mat.update(pred, target.view(-1).byte())

    print('Test Loss {loss.avg:.4f}\t'
          'Acc top1: {top1.avg:.4f} Acc top5: {top5.avg:.4f} \t'.format(loss=losses, top1=acc_top1, top5=acc_top5))
    confusion_mat.plot_mat(args.test + '.svg')
    write_log(content='Loss {loss.avg:.4f}\t Acc top1: {top1.avg:.4f} Acc top5: {top5.avg:.4f} \t'.format(loss=losses, top1=acc_top1, top5=acc_top5, args=args),
              epoch=num_epoch,
              filename=os.path.join(os.path.dirname(args.test), 'test_log.md'))

    import ipdb
    ipdb.set_trace()
    return losses.avg, [acc_top1.avg, acc_top5.avg]


def get_data(transform, mode='train'):
    '''
    Creates a customized dataset and associated data loader instance.
    '''
    print('Loading data for "%s" ...' % mode)
    global dataset
    if args.dataset == 'ucf101':
        dataset = UCF101_3d(mode=mode,
                            transform=transform,
                            seq_len=args.seq_len,
                            num_seq=args.num_seq,
                            downsample=args.ds,
                            which_split=args.split)
    elif args.dataset == 'ucf11':
        dataset = UCF11_3d(mode=mode,
                           transform=transform,
                           seq_len=args.seq_len,
                           num_seq=args.num_seq,
                           downsample=args.ds,
                           which_split=args.split)
    elif args.dataset == 'hmdb51':
        dataset = HMDB51_3d(mode=mode,
                            transform=transform,
                            seq_len=args.seq_len,
                            num_seq=args.num_seq,
                            downsample=args.ds,
                            which_split=args.split)
    elif args.dataset == 'epic_antiact':
        dataset = epic_antiact(mode=mode,
                               transform=transform,
                               seq_len=args.seq_len,
                               num_seq=args.num_seq,
                               class_type=args.class_type,
                               downsample=args.ds)
    elif args.dataset == 'epic_end':
        # For action classification
        dataset = epic_action_based(mode=mode, transform=transform, seq_len=args.seq_len,
                                    num_seq=args.num_seq, downsample=args.ds,
                                    class_type=args.class_type, sample_method='match_end',
                                    label_fraction=args.label_fraction)
    elif args.dataset == 'epic_start':
        # For action classification
        dataset = epic_action_based(mode=mode, transform=transform, seq_len=args.seq_len,
                                    num_seq=args.num_seq, downsample=args.ds,
                                    class_type=args.class_type, sample_method='match_start',
                                    label_fraction=args.label_fraction)
    elif args.dataset == 'epic_within':
        # For action classification
        dataset = epic_action_based(mode=mode, transform=transform, seq_len=args.seq_len,
                                    num_seq=args.num_seq, downsample=args.ds,
                                    class_type=args.class_type, sample_method='within',
                                    label_fraction=args.label_fraction)
    elif args.dataset == 'epic_before':
        # For action anticipation
        dataset = epic_action_based(mode=mode, transform=transform, seq_len=args.seq_len,
                                    num_seq=args.num_seq, downsample=args.ds,
                                    class_type=args.class_type, sample_method='before',
                                    sample_offset=1, label_fraction=args.label_fraction)
    else:
        raise ValueError('dataset not supported')

    # Shuffle data
    # print (dataset.shape)
    my_sampler = data.RandomSampler(dataset)

    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=my_sampler,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'val':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True)
                                      # drop_last=True)
    elif mode == 'test':
        data_loader = data.DataLoader(dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


def set_path(args):
    '''
    Creates descriptive directories for model files, tensorboard logs, diversity outputs, and other results.
    '''

    if args.resume:
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    elif args.diverse_actions:
        # This will populate the 'divers' subfolder within the finetuned model
        exp_path = os.path.dirname(os.path.dirname(args.pretrain))
    else:
        if 'vrnn' in args.model:
            # VRNN
            exp_path = 'log_{args.prefix}/{args.dataset}-{args.img_dim}-\
sp{args.split}_{0}_{args.model}_{args.temporality}_bs{args.batch_size}_\
ls{args.vrnn_latent_size}_ks{args.vrnn_kernel_size}_do{args.vrnn_dropout}_\
lr{1}_wd{args.wd}_ds{args.ds}_seq{args.num_seq}_pred{args.pred_step}_len{args.seq_len}_\
dp{args.dropout}_class_type-{args.class_type}'.format(
                'r%s' % args.net[6::], args.old_lr if args.old_lr is not None else args.lr,
                args=args) + ('_fenc' if args.vae_encode_train else '') + \
                ('_divact_paths{args.paths}'.format(args=args) if args.diverse_actions else '')
        else:
            # DPC / TPC / CVAE
            exp_path = 'log_{args.prefix}/{args.dataset}-{args.img_dim}-\
sp{args.split}_{0}_{args.model}_{args.cvae_arch}_{args.temporality}_bs{args.batch_size}_\
lr{1}_wd{args.wd}_ds{args.ds}_seq{args.num_seq}_pred{args.pred_step}_len{args.seq_len}_\
dp{args.dropout}_class_type-{args.class_type}'.format(
                'r%s' % args.net[6::], args.old_lr if args.old_lr is not None else args.lr,
                args=args) + ('_fenc' if args.vae_encode_train else '') + \
                ('_divact_paths{args.paths}'.format(args=args) if args.diverse_actions else '') 
        if len(args.pretrain) < 10:
            exp_path += '_pt=' + args.pretrain # e.g. random
        else:
            exp_path += '_pt=' + hashlib.md5(args.pretrain.encode()).hexdigest()[:8]

    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    divers_path = os.path.join(exp_path, 'divers')
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(divers_path):
        os.makedirs(divers_path)
    return img_path, model_path, divers_path


def MultiStepLR_Restart_Multiplier(epoch, gamma=0.1, step=[10, 15, 20], repeat=3):
    '''return the multipier for LambdaLR, 
    0  <= ep < 10: gamma^0
    10 <= ep < 15: gamma^1 
    15 <= ep < 20: gamma^2
    20 <= ep < 30: gamma^0 ... repeat 3 cycles and then keep gamma^2'''

    max_step = max(step)
    effective_epoch = epoch % max_step
    if epoch // max_step >= repeat:
        exp = len(step) - 1
    else:
        exp = len([i for i in step if effective_epoch >= i])
    return gamma ** exp


if __name__ == '__main__':
    main()
