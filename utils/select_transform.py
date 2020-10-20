# May 2020
# Unify transform selection for DPC & TPC & VAE

# ==== ABANDONED, DO NOT USE ====

import torch
import torch.optim as optim
from torch.utils import data
import torch.nn as nn
from torchvision import datasets, models, transforms
import torchvision.utils as vutils
from augmentation import *


def get_transform_for(dataset, mode, img_dim):
    # NOTE: test and diversity are different, is this desired?

    if mode == 'test':
        # From DPC: eval/test
        return transforms.Compose([
            RandomSizedCrop(consistent=True, size=224, p=0.0),
            Scale(size=(img_dim, img_dim)),
            ToTensor(),
            Normalize()
        ])

    elif mode == 'diversity':
        return transforms.Compose([
            RandomHorizontalFlip(consistent=True),
            CenterCrop(size=224),
            Scale(size=(img_dim, img_dim)),
            ToTensor(),
            Normalize()
        ])

    elif 'epic' in dataset:
        if mode == 'train':
            return transforms.Compose([
                RandomHorizontalFlip(consistent=True),
                RandomCrop(size=224, consistent=True),
                Scale(size=(img_dim, img_dim)),
                RandomGray(consistent=False, p=0.5),
                ColorJitter(brightness=0.5, contrast=0.5,
                            saturation=0.5, hue=0.25, p=1.0),
                ToTensor(),
                Normalize()
            ])

        elif mode == 'val':
            return transforms.Compose([
                RandomHorizontalFlip(consistent=True),
                RandomCrop(size=224, consistent=True),
                Scale(size=(img_dim, img_dim)),
                RandomGray(consistent=False, p=0.5),
                ColorJitter(brightness=0.5, contrast=0.5,
                            saturation=0.5, hue=0.25, p=1.0),
                ToTensor(),
                Normalize()
            ])

    elif 'block_toy' in dataset:
        # may have to introduce more transformations with imagenet background
        transform = transforms.Compose([
            # centercrop
            CenterCrop(size=224),
            RandomSizedCrop(consistent=True, size=224, p=0.0),
            Scale(size=(img_dim, img_dim)),
            ToTensor(),
            Normalize()
        ])

    elif 'ucf' in dataset:
        if mode == 'train':
            return transforms.Compose([
                RandomHorizontalFlip(consistent=True),
                RandomCrop(size=224, consistent=True),
                Scale(size=(img_dim, img_dim)),
                RandomGray(consistent=False, p=0.5),
                ColorJitter(brightness=0.5, contrast=0.5,
                            saturation=0.5, hue=0.25, p=1.0),
                ToTensor(),
                Normalize()
            ])

        elif mode == 'val':
            return transforms.Compose([
                RandomHorizontalFlip(consistent=True),
                RandomCrop(size=224, consistent=True),
                Scale(size=(img_dim, img_dim)),
                RandomGray(consistent=False, p=0.5),
                ColorJitter(brightness=0.2, contrast=0.5,
                            saturation=0.5, hue=0.25, p=1.0),
                ToTensor(),
                Normalize()
            ])

    elif 'k400' in dataset:
        if mode == 'train':
            return transforms.Compose([
                RandomSizedCrop(size=img_dim, consistent=True, p=1.0),
                RandomHorizontalFlip(consistent=True),
                RandomGray(consistent=False, p=0.5),
                ColorJitter(brightness=0.5, contrast=0.5,
                            saturation=0.5, hue=0.25, p=1.0),
                ToTensor(),
                Normalize()
            ])
        elif mode == 'val':
            return transforms.Compose([
                RandomSizedCrop(size=img_dim, consistent=True, p=1.0),
                RandomHorizontalFlip(consistent=True),
                RandomGray(consistent=False, p=0.2),
                ColorJitter(brightness=0.2, contrast=0.2,
                            saturation=0.2, hue=0.1, p=0.3),
                ToTensor(),
                Normalize()
            ])

    raise Exception('No transform available, check arguments')
