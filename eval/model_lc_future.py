import math
import numpy as np
import sys
sys.path.append('../backbone')
sys.path.append('../vae-dpc')
sys.path.append('../tpc')
sys.path.append('../tpc-backbone')

from select_backbone import select_resnet
from convrnn import ConvGRU
from cvae_model import *
from vrnn_model import *
from model_3d import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class LC_future_DPC(nn.Module):
    def __init__(self, sample_size, num_seq, seq_len, pred_step,
                 network='resnet18', dropout=0.5, num_class=101):
        '''
        modified from dpc-eval model. Different from the original model,
        pred function is run pred_step number of times just like
        pretext training. The last context variable is used to project
        to the action space
        num_class: If integer => single output layer; if list => multiple output layers (for example verb + noun)
        '''
        super(LC_future_DPC, self).__init__()
        torch.cuda.manual_seed(666)

        # size of the image 128x128
        self.sample_size = sample_size

        # num_seq = 5 and seq_len = 8
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.pred_step = pred_step
        self.num_class = num_class
        print('=> Using RNN + FC model ')

        print('=> Use 2D-3D %s!' % network)

        # dimensions of the output
        self.last_duration = int(math.ceil(seq_len / 4))
        self.last_size = int(math.ceil(sample_size / 32))

        track_running_stats = True

        # f network (= extract representation given video)
        self.backbone, self.param = select_resnet(
            network, track_running_stats=track_running_stats)
        print('feature_size:', self.param['feature_size'])
        self.param['num_layers'] = 1
        self.param['hidden_size'] = self.param['feature_size']

        # g-network (= aggregate into present context)
        print('=> using ConvRNN, kernel_size = 1')
        self.agg = ConvGRU(input_size=self.param['feature_size'],
                           hidden_size=self.param['hidden_size'],
                           kernel_size=1,
                           num_layers=self.param['num_layers'])

        # two layered network \phi (= predict future given context)
        self.network_pred = nn.Sequential(
            nn.Conv2d(self.param['feature_size'],
                      self.param['feature_size'], kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.param['feature_size'],
                      self.param['feature_size'], kernel_size=1, padding=0)
        )

        self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.agg)
        self._initialize_weights(self.network_pred)

        # not in the training network
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
            # Multi, for predicting noun and verb simultaneously
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

        # [BatchSize, Num Sequences, Channels, Sequence Length, Height, Width]
        (B, N, C, SL, H, W) = block.shape
        block = block.view(B * N, C, SL, H, W)

        feature = self.backbone(block)
        del block
        feature = F.relu(feature)

        # set appropriate kernel size here
        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=1)
        # [B*N,D,last_size,last_size]
        feature = feature.view(
            B, N, self.param['feature_size'], self.last_size, self.last_size)

        ####
        context, hidden = self.agg(
            feature[:, 0:N - self.pred_step, :].contiguous())
        # [4, 1, 256, 4, 4]

        # after tanh, (-1,1). get the hidden state of last layer, last time step
        hidden = hidden[:, -1, :]
        # [4, 256, 4, 4]

        # aggregate the results for pre_step number of steps
        # pred = [] # NOTE: unused here

        # run network_pred for pred_step number of times, get the last
        # context variable and
        for i in range(self.pred_step):
            # sequentially pred future for pred_step number of times
            p_tmp = self.network_pred(hidden)
#             print(p_tmp.shape)

            # pred.append(p_tmp)
            context, hidden = self.agg(
                self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
            hidden = hidden[:, -1, :]

        #[4, 256, 4, 4]
        # pred = torch.stack(pred, 1)  # B, pred_step, xxx
        #[4, 3, 256, 4, 4]

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


if __name__ == '__main__':

    mymodel = LC_future_DPC(128, num_seq=8, seq_len=5,
                            pred_step=3, network='resnet18')
    mymodel = mymodel.cuda()

    # (B, N, C, SL, H, W)
    mydata = torch.cuda.FloatTensor(4, 8, 3, 5, 128, 128).cuda()

    nn.init.normal_(mydata)
    #import ipdb; ipdb.set_trace()
    output, context = mymodel(mydata)
    print(output.shape)


class LC_future_TPC(DPC_RNN):
    '''TPC model with LC'''

    def __init__(self, sample_size, num_seq, seq_len, pred_step,
                 network='resnet18', dropout=0.5, num_class=101, pool='None',
                 weighting=True, margin=25, radius_location='Phi',
                 loss_type='MSE'):
        super(LC_future_TPC, self).__init__(
            sample_size, num_seq=num_seq, seq_len=seq_len, pred_step=pred_step,
            network=network, action_cls_head=True,
            dropout=dropout, num_class=num_class)


class LC_future_CVAE(DPC_CVAE):
    '''
    Uses existing DPC_CVAE class with action_cls_head=True, see vae-dpc/cvae_model.py.
    '''

    def __init__(self, sample_size, num_seq=8, seq_len=5, pred_step=3,
                 network='resnet18', cvae_arch='fc', dropout=0.5, num_class=101):
        super(LC_future_CVAE, self).__init__(
            sample_size, num_seq=num_seq, seq_len=seq_len, pred_step=pred_step,
            network=network, cvae_arch=cvae_arch, action_cls_head=True,
            dropout=dropout, num_class=num_class)


class LC_future_VRNN(DPC_VRNN):
    '''
    Uses existing DPC_VRNN class with action_cls_head='future', see vae-dpc/vrnn_model.py.
    '''

    def __init__(self, sample_size, num_seq=8, seq_len=5, pred_step=3,
                 network='resnet18', latent_size=8, kernel_size=1, rnn_dropout=0.1,
                 cls_dropout=0.5, num_class=101, time_indep=False):
        super(LC_future_VRNN, self).__init__(
            sample_size, num_seq=num_seq, seq_len=seq_len, pred_step=pred_step,
            network=network, latent_size=latent_size, kernel_size=kernel_size, rnn_dropout=rnn_dropout,
            action_cls_head='future', cls_dropout=cls_dropout, num_class=num_class, time_indep=time_indep)
