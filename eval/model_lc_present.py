import math
import numpy as np
import sys
sys.path.append('../backbone')
sys.path.append('../vae-dpc')

from select_backbone import select_resnet
from convrnn import ConvGRU
from vrnn_model import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class LC_present(nn.Module):
    def __init__(self, sample_size, num_seq, seq_len,
                 network='resnet18', dropout=0.5, num_class=101):
        '''
        Original DPC, according to diagram in appendix
        No future prediction network involved
        num_class: If integer => single output layer; if list => multiple output layers (for example verb + noun)
        '''
        super(LC_present, self).__init__()
        torch.cuda.manual_seed(666)  # very innocent number

        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.num_class = num_class
        print('=> Using RNN + FC model with num_class:', num_class)

        print('=> Use 2D-3D %s!' % network)
        self.last_duration = int(math.ceil(seq_len / 4))
        self.last_size = int(math.ceil(sample_size / 32))
        track_running_stats = True

        self.backbone, self.param = select_resnet(
            network, track_running_stats=track_running_stats)
        self.param['num_layers'] = 1
        self.param['hidden_size'] = self.param['feature_size']

        print('=> using ConvRNN, kernel_size = 1')
        self.agg = ConvGRU(input_size=self.param['feature_size'],
                           hidden_size=self.param['hidden_size'],
                           kernel_size=1,
                           num_layers=self.param['num_layers'])
        self._initialize_weights(self.agg)

        self.final_bn = nn.BatchNorm1d(self.param['feature_size'])
        self.final_bn.weight.data.fill_(1)
        self.final_bn.bias.data.zero_()

        # Initializer final FC layer(s), one or multiple, depending on class configuration
        if isinstance(num_class, int):
            # Single
            self.multi_output = False
            self.final_fc = nn.Sequential(nn.Dropout(dropout),
                                          nn.Linear(self.param['feature_size'], self.num_class))
            self._initialize_weights(self.final_fc)

        elif isinstance(num_class, list):
            # Multi
            self.multi_output = True
            self.final_fc = []
            for cur_num_cls in num_class:
                cur_fc = nn.Sequential(nn.Dropout(dropout),
                                       nn.Linear(self.param['feature_size'], cur_num_cls))
                self._initialize_weights(cur_fc)
                self.final_fc.append(cur_fc)
            # IMPORTANT, otherwise pytorch won't register
            self.final_fc = nn.ModuleList(self.final_fc)

        else:
            raise ValueError(
                'num_class is of unknown type (expected int or list of ints)')

    def forward(self, block):
        # seq1: [B, N, C, SL, W, H]
        (B, N, C, SL, H, W) = block.shape
        block = block.view(B * N, C, SL, H, W)
        feature = self.backbone(block)
        del block
        feature = F.relu(feature)

        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=1)
        # [B*N,D,last_size,last_size]
        feature = feature.view(
            B, N, self.param['feature_size'], self.last_size, self.last_size)

        # NOTE: ALL features are used here, so take care of num_seq
        context, _ = self.agg(feature)

        # for the TPC future variant, inbetween here, a whole bunch of prediction stuff happens

        context = context[:, -1, :].unsqueeze(1)
        context = F.avg_pool3d(
            context, (1, self.last_size, self.last_size), stride=1).squeeze(-1).squeeze(-1)
        del feature

        # [B,N,C] -> [B,C,N] -> BN() -> [B,N,C], because BN operates on id=1 channel.
        context = self.final_bn(context.transpose(-1, -2)).transpose(-1, -2)
        if self.multi_output:
            # E.g. verb + noun within same network
            output = []
            for i in range(len(self.num_class)):
                cur_out = self.final_fc[i](context).view(
                    B, -1, self.num_class[i])
                output.append(cur_out)
        else:
            # E.g. verb or noun with separate networks
            output = self.final_fc(context).view(B, -1, self.num_class)

        return output, context

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet_3d.py


class LC_present_VRNN(DPC_VRNN):
    '''
    Uses existing DPC_VRNN class with action_cls_head='present', see vae-dpc/vrnn_model.py.
    NOTE: different from LC_present because of rnn.
    '''

    def __init__(self, sample_size, num_seq=8, seq_len=5, pred_step=3,
                 network='resnet18', latent_size=8, kernel_size=1, rnn_dropout=0.1,
                 cls_dropout=0.5, num_class=101, time_indep=False):
        super(LC_present_VRNN, self).__init__(
            sample_size, num_seq=num_seq, seq_len=seq_len, pred_step=pred_step,
            network=network, latent_size=latent_size, kernel_size=kernel_size, rnn_dropout=rnn_dropout,
            action_cls_head='present', cls_dropout=cls_dropout, num_class=num_class, time_indep=time_indep)
