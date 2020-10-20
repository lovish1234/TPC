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
from dataset_3d import *
from model_3d import *

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

# saving log, calculating accuracy etc.
import torchvision.utils as vutils

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
parser.add_argument('--epochs', default=1000, type=int,
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

parser.add_argument('--new_augment', action='store_true',
                    help='Use data augmentation with consistent train-test distortion')

# TPC custom arguments
parser.add_argument('--distance', default='cosine', type=str,
                    help='distance metric used in embedding space')
parser.add_argument('--distance_type', default='uncertain', type=str,
                    help='whether or not applying radius')
parser.add_argument('--weighting', default='True', type=str,
                    help='whether or not applying weights to positive samples')
parser.add_argument('--margin', default='0.1', type=float,
                    help='margin in the loss function')
parser.add_argument('--drive', default='ssd', type=str,
                    help='type of hard-drive trainig data is located at')
parser.add_argument('--num_workers', default=32, type=int,
                    help='number of workers used in dataloader')
parser.add_argument('--num_sample', default=1, type=int,
                    help='number of samples sampled from each videos')
parser.add_argument('--action_from_frame', default='False', type=str,
                    help='retrieving action from frame')
parser.add_argument('--pool', default='None', type=str,
                    help='spatial pooling in f(.)')
parser.add_argument('--loss_type', default='CE', type=str,
                    help='type of loss function used')

def main():

    # set to constant for consistant results
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    global args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    global cuda
    cuda = torch.device('cuda')

    ### dpc model ###
    if args.model == 'dpc-rnn':
        model = DPC_RNN(sample_size=args.img_dim,
                        num_seq=args.num_seq,
                        seq_len=args.seq_len,
                        network=args.net,
                        pred_step=args.pred_step,
                        distance=args.distance,
                        distance_type=args.distance_type,
                        weighting=args.weighting,
                        margin=args.margin,
                        pool=args.pool,
                        loss_type=args.loss_type)
    else:
        raise ValueError('wrong model!')

    # parallelize the model
    model = nn.DataParallel(model)
    model = model.to(cuda)

    # load dict [frame_id -> action]
    if args.action_from_frame == 'True':
        with open('/proj/vondrick/ruoshi/github/TPC/tpc/action_from_frame.p', 'rb') as fp:
            map_action_frame = pickle.load(fp)

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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5)

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

    ### load transform & data ###
    if 'epic' in args.dataset:
        transform = transforms.Compose([
            RandomHorizontalFlip(consistent=True),
            # RandomSizedCrop(size=224, consistent=True, p=1.0),
            RandomCrop(size=224, consistent=True),
            Scale(size=(args.img_dim, args.img_dim)),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5,
                        saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])
    elif 'ucf' in args.dataset:  # designed for ucf101, short size=256, rand crop to 224x224 then scale to 128x128
        transform = transforms.Compose([
            RandomHorizontalFlip(consistent=True),
            RandomCrop(size=224, consistent=True),
            Scale(size=(args.img_dim, args.img_dim)),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5,
                        saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])
    elif args.dataset == 'k400':  # designed for kinetics400, short size=150, rand crop to 128x128
        transform = transforms.Compose([
            RandomHorizontalFlip(consistent=True),
            RandomSizedCrop(size=args.img_dim, consistent=True, p=1.0),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5,
                        saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])
    elif args.dataset in ['block_toy_1',
                          'block_toy_2',
                          'block_toy_3']:
        # get validation data with label for plotting the embedding
        transform = transforms.Compose([
            # centercrop
            CenterCrop(size=224),
            # RandomSizedCrop(consistent=True, size=224, p=0.0), # no effect
            Scale(size=(args.img_dim, args.img_dim)),
            ToTensor(),
            Normalize()
        ])
    elif args.dataset in ['block_toy_imagenet_1',
                          'block_toy_imagenet_2',
                          'block_toy_imagenet_3']:
        # may have to introduce more transformations with imagenet background
        transform = transforms.Compose([
            # centercrop
            CenterCrop(size=224),
            # RandomSizedCrop(consistent=True, size=224, p=0.0), # no effect
            Scale(size=(args.img_dim, args.img_dim)),
            ToTensor(),
            Normalize()
        ])

    train_loader = get_data(transform, mode='train')
    val_loader = get_data(transform, mode='val')

    # setup tools

    # denormalize to display input images via tensorboard
    global de_normalize
    de_normalize = denorm()

    global img_path
    img_path, model_path = set_path(args)
    global writer_train

    # book-keeping
    try:  # old version
        writer_val = SummaryWriter(log_dir=os.path.join(img_path, 'val'))
        writer_train = SummaryWriter(log_dir=os.path.join(img_path, 'train'))
    except:  # v1.7
        writer_val = SummaryWriter(logdir=os.path.join(img_path, 'val'))
        writer_train = SummaryWriter(logdir=os.path.join(img_path, 'train'))

    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_acc, train_accuracy_list, radius, radius_var = train(
            train_loader, model, optimizer, epoch)
        val_loss, val_acc, val_accuracy_list, radius, radius_var = validate(
            val_loader, model, epoch)

#         scheduler.step(val_loss)

        # save curve
        writer_train.add_scalar('global/loss', train_loss, epoch)
        writer_train.add_scalar('global/accuracy', train_acc, epoch)
        writer_train.add_scalar('global/radius', radius, epoch)
        writer_train.add_scalar('global/radius_var', radius_var, epoch)

        writer_val.add_scalar('global/loss', val_loss, epoch)
        writer_val.add_scalar('global/accuracy', val_acc, epoch)
        writer_val.add_scalar('global/radius', radius, epoch)
        writer_val.add_scalar('global/radius_var', radius_var, epoch)

        writer_train.add_scalar('accuracy/top1', train_accuracy_list[0], epoch)
        writer_train.add_scalar('accuracy/top3', train_accuracy_list[1], epoch)
        writer_train.add_scalar('accuracy/top5', train_accuracy_list[2], epoch)
        writer_val.add_scalar('accuracy/top1', val_accuracy_list[0], epoch)
        writer_val.add_scalar('accuracy/top3', val_accuracy_list[1], epoch)
        writer_val.add_scalar('accuracy/top5', val_accuracy_list[2], epoch)

        # save check_point
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


def train(data_loader, model, optimizer, epoch):

    # average losses
    losses = AverageMeter()

    # average accuracy
    accuracy = AverageMeter()

    # top-1, top-3 and top-5 accuracy
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]

    model.train()
    global iteration

    #print (len(data_loader))
    for idx, input_dict in enumerate(data_loader):

        tic = time.time()
        input_seq = input_dict['t_seq']
        input_seq = input_seq.to(cuda)
        B = input_seq.size(0)
        [final_score_, mask_, pred_radius_, score_] = model(input_seq)
        if args.loss_type == 'MSE':
            m = nn.MSELoss()
            loss = m(final_score_, torch.zeros_like(final_score_))
        elif args.loss_type == 'L1':
            loss = torch.sum(final_score_)

        del final_score_

        # visualize the input sequence to the network
        if (iteration == 0) or (iteration == args.print_freq):
            if B > 2:
                input_seq = input_seq[0:2, :]
            writer_train.add_image('input_seq',
                                   de_normalize(vutils.make_grid(
                                       input_seq.transpose(2, 3).contiguous(
                                       ).view(-1, 3, args.img_dim, args.img_dim),
                                       nrow=args.num_seq * args.seq_len)),
                                   iteration)
        del input_seq

        if args.distance == 'L2':
            # why only for idx 0 ?
            if idx == 0:
                target_, (_, B2, NS, NP, SQ) = process_output(mask_)

            target_flattened = target_.view(B * NP * SQ, B2 * NS * SQ)
            _, target_flattened = target_flattened.max(dim=1)

            # loss function, here cross-entropy for DPC
            top1, top3, top5 = calc_topk_accuracy(
                -score_, target_flattened, (1, 3, 5))

        elif args.distance == 'cosine':
            # why only for idx 0 ?
            if idx == 0:
                target_, (_, B2, NS, NP, SQ) = process_output(mask_)

            target_flattened = target_.view(B * NP * SQ, B2 * NS * SQ)
            target_flattened = target_flattened.double()
            _, target_flattened = target_flattened.max(dim=1)

            # loss function, here cross-entropy for DPC
            top1, top3, top5 = calc_topk_accuracy(
                score_, target_flattened, (1, 3, 5))
        del mask_
        # break
        accuracy_list[0].update(top1.item(), B)
        accuracy_list[1].update(top3.item(), B)
        accuracy_list[2].update(top5.item(), B)
        if args.loss_type == 'L1':
            losses.update(loss.item() / len(score_)**2, B)
        else:
            losses.update(loss.item(), B)
        accuracy.update(top1.item(), B)
        del score_

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del loss

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.6f} ({loss.local_avg:.4f})\t'
                  'Acc: top1 {3:.4f}; top3 {4:.4f}; top5 {5:.4f} T:{6:.2f}\t'
                  'radius: {radius:.4f}; radius_var: {radius_var:.4f}'.format(
                      epoch, idx, len(data_loader), top1, top3, top5, time.time() - tic, loss=losses,
                      radius=torch.mean(pred_radius_).detach().cpu().numpy(),
                      radius_var=torch.std(pred_radius_).detach().cpu().numpy())
                  )
            writer_train.add_scalar('local/loss', losses.val, iteration)
            writer_train.add_scalar('local/accuracy', accuracy.val, iteration)

            # TODO plot radius etc also in tensorboard

            iteration += 1
    return [losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list],
            torch.mean(pred_radius_).detach().cpu().numpy(), torch.std(pred_radius_).detach().cpu().numpy()]


def validate(data_loader, model, epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    model.eval()

    with torch.no_grad():

        for idx, input_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
            #print(idx, input_seq.shape)

            input_seq = input_dict['t_seq']
            input_seq = input_seq.to(cuda)
            B = input_seq.size(0)

            [final_score_, mask_, pred_radius_, score_] = model(input_seq)
            del input_seq

            m = nn.MSELoss()
            loss = m(final_score_, torch.zeros_like(final_score_))

            del final_score_

            if args.distance == 'L2':
                if idx == 0:
                    target_, (_, B2, NS, NP, SQ) = process_output(mask_)
                target_flattened = target_.view(B * NP * SQ, B2 * NS * SQ)

                target_flattened = target_flattened.double()
                _, target_flattened = target_flattened.max(dim=1)

                top1, top3, top5 = calc_topk_accuracy(
                    -score_, target_flattened, (1, 3, 5))
            elif args.distance == 'cosine':
                if idx == 0:
                    target_, (_, B2, NS, NP, SQ) = process_output(mask_)
                target_flattened = target_.view(B * NP * SQ, B2 * NS * SQ)

                target_flattened = target_flattened.double()
                _, target_flattened = target_flattened.max(dim=1)

                top1, top3, top5 = calc_topk_accuracy(
                    score_, target_flattened, (1, 3, 5))
            del score_, mask_

            losses.update(loss.item(), B)

            del loss

            accuracy.update(top1.item(), B)

            accuracy_list[0].update(top1.item(), B)
            accuracy_list[1].update(top3.item(), B)
            accuracy_list[2].update(top5.item(), B)

    print('[{0}/{1}] Loss {loss.local_avg:.4f}\t'
          'Acc: top1 {2:.4f}; top3 {3:.4f}; top5 {4:.4f}; \t'
          'radius {radius:.4f}; radius_var {radius_var:.4f}'
          .format(epoch, args.epochs, *[i.avg for i in accuracy_list], loss=losses,
                  radius=torch.mean(pred_radius_).detach().cpu().numpy(),
                  radius_var=torch.std(pred_radius_).detach().cpu().numpy()))
    return [losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list],
            torch.mean(pred_radius_).detach().cpu().numpy(), torch.std(pred_radius_).detach().cpu().numpy()]


def get_data(transform, mode='train'):
    print('Loading data for "%s" ...' % mode)
    if args.dataset == 'k400':
        use_big_K400 = args.img_dim > 140
        dataset = Kinetics400_full_3d(mode=mode,
                                      transform=transform,
                                      seq_len=args.seq_len,
                                      num_seq=args.num_seq,
                                      downsample=args.ds,
                                      big=use_big_K400,
                                      drive=args.drive)
    elif args.dataset == 'ucf101':
        dataset = UCF101_3d(mode=mode,
                            transform=transform,
                            seq_len=args.seq_len,
                            num_seq=args.num_seq,
                            downsample=args.ds,
                            drive=args.drive)
    elif args.dataset == 'ucf11':
        dataset = UCF11_3d(mode=mode,
                           transform=transform,
                           seq_len=args.seq_len,
                           num_seq=args.num_seq,
                           downsample=args.ds,
                           drive=args.drive)
    elif args.dataset == 'epic_gulp':
        dataset = epic_gulp(mode=mode,
                            transform=transform,
                            seq_len=args.seq_len,
                            num_seq=args.num_seq,
                            downsample=args.ds,
                            class_type='verb+noun')
    elif args.dataset == 'epic':
        dataset = epic(mode=mode,
                       transform=transform,
                       seq_len=args.seq_len,
                       num_seq=args.num_seq,
                       downsample=args.ds,
                       drive=args.drive,
                       num_sample=args.num_sample)
    elif args.dataset in ['block_toy_1',
                          'block_toy_2',
                          'block_toy_3']:
        num = args.dataset.split('_')[-1]
        dataset = block_toy(mode=mode,
                            num=num,
                            transform=transform,
                            seq_len=args.seq_len,
                            num_seq=args.num_seq,
                            downsample=args.ds,
                            drive=args.drive,
                            num_sample=args.num_sample)
    elif args.dataset in ['block_toy_imagenet_1',
                          'block_toy_imagenet_2',
                          'block_toy_imagenet_3']:

        num = args.dataset.split('_')[-1]
        dataset = block_toy_imagenet(mode=mode,
                                     num=num,
                                     transform=transform,
                                     seq_len=args.seq_len,
                                     num_seq=args.num_seq,
                                     downsample=args.ds,
                                     drive=args.drive,
                                     num_sample=args.num_sample)
    else:
        raise ValueError('dataset not supported')

    # randomize the instances
    print(len(dataset))
    sampler = data.RandomSampler(dataset)

    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'val':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      drop_last=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


def set_path(args):
    if args.resume:
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        exp_path = 'log_{args.prefix}/{args.dataset}-{args.img_dim}_{0}_{args.model}_\
bs{args.batch_size}_lr{1}_seq{args.num_seq}_pred{args.pred_step}_len{args.seq_len}_ds{args.ds}_\
train-{args.train_what}{2}_distance_{args.distance}_distance-type_{args.distance_type}_margin_{args.margin}_\
pool_{args.pool}_loss-type_{args.loss_type}'.format(
            'r%s' % args.net[6::],
            args.old_lr if args.old_lr is not None else args.lr,
            '_pt=%s' % args.pretrain.replace(
                '/', '-') if args.pretrain else '',
            args=args)
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return img_path, model_path


if __name__ == '__main__':
    main()
