import torch
import os

from dataset_epic import *
from dataset_other import *
from dataset_toy import *

from resnet_2d3d import neq_load_customized
from torchvision import datasets, models, transforms

# data augmentation methods
from augmentation import *


def get_transform(args):
        ### load transform & dataset ###
    # TODO: work on new_augment
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
    elif 'ucf' in args.dataset: # designed for ucf101, short size=256, rand crop to 224x224 then scale to 128x128
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
    return transform


def get_data(args, transform, mode='train', pm_path=None, epoch=None):   
    
    print('Loading data for "%s", from %s ...' % (mode, args.dataset))
    if args.dataset == 'k400':
        use_big_K400 = args.img_dim > 140
        dataset = Kinetics400_full_3d(mode=mode,
                                      transform=transform,
                                      seq_len=args.seq_len,
                                      num_seq=args.num_seq,
                                      downsample=args.ds,
                                      drive=args.drive,
                                      big=use_big_K400)
    elif args.dataset == 'ucf101':
        dataset = UCF101_3d(mode=mode,
                            transform=transform,
                            seq_len=args.seq_len,
                            num_seq=args.num_seq,
                            drive=args.drive,
                            downsample=args.ds)
    elif args.dataset == 'ucf11':
        dataset = UCF11_3d(mode=mode,
                           transform=transform,
                           seq_len=args.seq_len,
                           num_seq=args.num_seq,
                           drive=args.drive,
                           downsample=args.ds)
    elif args.dataset == 'epic_unlabeled':
        dataset = epic_unlabeled(mode=mode,
                                 transform=transform,
                                 seq_len=args.seq_len,
                                 num_seq=args.num_seq,
                                 pred_step=args.pred_step,
                                 downsample=args.ds,
                                 drive=args.drive,
                                 num_clips=args.num_clips)
    elif args.dataset == 'epic_within':
        dataset = epic_action_based(mode=mode,
                                    transform=transform,
                                    seq_len=args.seq_len,
                                    num_seq=args.num_seq,
                                    pred_step=args.pred_step,
                                    downsample=args.ds,
                                    drive=args.drive,
                                    sample_method='within')
    elif args.dataset == 'epic_before':
        dataset = epic_action_based(mode=mode,
                                    transform=transform,
                                    seq_len=args.seq_len,
                                    num_seq=args.num_seq,
                                    pred_step=args.pred_step,
                                    downsample=args.ds,
                                    drive=args.drive,
                                    sample_method='before',
                                    sample_offset=1)
    elif args.dataset == 'epic_present_matching':
        dataset = epic_present_matching(epoch, freq=args.pm_freq,
                                        mode=mode,
                                        transform=transform,
                                        seq_len=args.seq_len,
                                        num_seq=args.num_seq,
                                        pred_step=args.pred_step,
                                        downsample=args.ds,
                                        num_clips=args.num_clips,
                                        drive='ssd',
                                        NN_path=pm_path,
                                        num_ngbrs=args.pm_num_ngbrs)
    elif 'synthepic' in args.dataset:
        dataset = synthepic_action_pair(mode=mode,
                                        transform=transform,
                                        seq_len=args.seq_len,
                                        num_seq=args.num_seq,
                                        pred_step=args.pred_step,
                                        downsample=args.ds,
                                        drive=args.drive,
                                        sample_method='within' if 'within' in args.dataset else 'match_cut',
                                        exact_cuts='exact' in args.dataset)
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
                            num_sample=args.num_clips)
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
                                     num_sample=args.num_clips)
    else:
        raise ValueError('dataset not supported: ' + args.dataset)

    # randomize the instances
    #print (dataset)
    if args.dataset == 'epic_present_matching':
        sampler = data.SequentialSampler(dataset)
    else:
        sampler = data.RandomSampler(dataset)

    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=32,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'val':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=32,
                                      pin_memory=True,
                                      drop_last=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader