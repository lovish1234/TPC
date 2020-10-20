import math
import sys

sys.path.append('../../uc-backbone')
from select_backbone import select_resnet
from convrnn import ConvGRU

import torch
import torch.nn as nn
import torch.nn.functional as FN


class model_visualize(nn.Module):
    def __init__(self,
                 sample_size,
                 num_seq=5,
                 seq_len=5,
                 network='resnet18',
                 distance_type='certain',
                 radius_type='linear',
                 feature_type='F',
                 pred_steps=1,
                 pool='avg'):
        super(model_visualize, self).__init__()
        torch.cuda.manual_seed(666)

        # size of the image 128x128
        self.sample_size = sample_size
        self.distance_type = distance_type
        self.radius_type = radius_type
        self.feature_type = feature_type
        self.network = network
        self.pool = pool
        self.pred_steps = pred_steps

        # num_seq = 5 and seq_len = 8
        self.num_seq = num_seq
        self.seq_len = seq_len

        if self.feature_type == 'F':
            print('[model_visualize.py] Using <<F>> mapping ')
        elif self.feature_type == 'G':
            print('[model_visualize.py] Using <<F+G>> mapping ')
        elif self.feature_type == 'Phi':
            print('[model_visualize.py] Using <<F+G+Phi>> mapping ')

        print('[model_visualize.py] Use 2D-3D %s!' % network)

        # dimensions of the output
        if self.network == 'resnet8' or self.network == 'resnet10':
            self.last_duration = int(math.ceil(seq_len / 2))
        else:
            self.last_duration = int(math.ceil(seq_len / 4))
        self.last_size = int(math.ceil(sample_size / 32))

        track_running_stats = True

        # f network
        print('[model_visualize.py] Using distance type : ',
              self.distance_type)
        print('[model_visualize.py] Using radius type : ', self.radius_type)

        # f - choose an appropriate feature extractor. In this case, a resent
        self.backbone, self.param = select_resnet(
            network, track_running_stats=False,
            distance_type=self.distance_type,
            radius_type=self.radius_type)

        #print (self.param)

        # number of layers in GRU
        self.param['num_layers'] = 1  # param for GRU
        self.param['hidden_size'] = self.param['feature_size']  # param for GRU

        self.agg = ConvGRU(input_size=self.param['feature_size'],
                           hidden_size=self.param['hidden_size'],
                           kernel_size=1,
                           num_layers=self.param['num_layers'],
                           radius_type=self.radius_type)

        self.avg_pool = nn.AvgPool3d(
            (1, self.last_size, self.last_size), stride=1)
        self.max_pool = nn.MaxPool3d(
            (1, self.last_size, self.last_size), stride=1)

        # two layered network \phi
        self.network_pred = nn.Sequential(
            nn.Conv2d(self.param['feature_size'],
                      self.param['feature_size'], kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.param['feature_size'],
                      self.param['feature_size'], kernel_size=1, padding=0)
        )

        if self.radius_type == 'log' and self.distance_type == 'uncertain':
            print('[model_3d.py] Using log as radius_type')
            self.activation = exp_activation()

        # what does mask do ?
        self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.agg)
        self._initialize_weights(self.network_pred)

    def forward(self, block):
        # seq1: [B, N, C, SL, W, H]

        # [BatchSize, Num Sequences, Channels, Sequence Length, Height, Width]
        (B, N, C, SL, H, W) = block.shape
        block = block.view(B * N, C, SL, H, W)

        ############## F ############
        feature = self.backbone(block)
        del block
        feature = FN.avg_pool3d(
            feature, (self.last_duration, 1, 1), stride=1)
        # [B*N,D,last_size,last_size]
        feature = feature.view(
            B, N, self.param['feature_size'], self.last_size, self.last_size)

        feature = FN.relu(feature)
        # print(feature.shape)
        if self.feature_type == 'F':
            if self.pool == 'avg':
                feature = self.avg_pool(feature).squeeze(-1).squeeze(-1)
            elif self.pool == 'max':
                feature = self.max_pool(feature).squeeze(-1).squeeze(-1)
            return feature

        # should also account for number of steps being predicted
        ############## G ############
        context, hidden = self.agg(feature)
        context = context[:, -1, :]
        #print (context.shape, hidden.shape)

        del feature
        if self.feature_type == 'G':
            context = context.unsqueeze(1)
            if self.pool == 'avg':
                context = self.avg_pool(context).squeeze(-1).squeeze(-1)
            elif self.pool == 'max':
                context = self.max_pool(context).squeeze(-1).squeeze(-1)
            return context

        # predict pred_step ahead
        # print ('Predicting representation <<', self.pred_steps, '>> time setps ahead')
        for i in range(self.pred_steps):
            # sequentially pred future for pred_step number of times
            future = self.network_pred(context)
            #print (future.shape, hidden.shape, context.shape)

            context, hidden = self.agg(
                self.relu(future).unsqueeze(1), context.unsqueeze(0))
            context = context[:, -1, :]

        #future = self.network_pred(context)
        if self.feature_type == 'Phi':

            if self.pool == 'avg':
                future = self.avg_pool(future).squeeze(-1).squeeze(-1)
            elif self.pool == 'max':
                future = self.max_pool(future).squeeze(-1).squeeze(-1)
            future = future.unsqueeze(1)
            return future

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet_3d.py


if __name__ == '__main__':

    mymodel_visualize = model_visualize(128,
                                        num_seq=5,
                                        seq_len=5,
                                        network='resnet18',
                                        distance_type='certain',
                                        radius_type='linear',
                                        feature_type='Phi',
                                        pred_steps=3,
                                        pool='avg')

    mymodel_visualize = mymodel_visualize.cuda()

    # (B, N, C, SL, H, W)
    mydata = torch.cuda.FloatTensor(16, 5, 3, 5, 128, 128)

    #import ipdb; ipdb.set_trace()
    y = mymodel_visualize(mydata)
    print(y.shape)
