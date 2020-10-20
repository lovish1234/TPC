# get the model as DPC-RNN

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../backbone')

# to extract the features
from select_backbone import select_resnet

# to aggregate the features in one
from convrnn import ConvGRU

# poincare distance
from poincare_distance import poincare_distance
import hyptorch.nn as hypnn


class DPC_RNN(nn.Module):
    '''DPC with RNN'''

    def __init__(self, sample_size, num_seq=8, seq_len=5,
                 pred_step=3, network='resnet50', distance='dot',
                 poincare_c=1.0, poincare_ball_dim=256):
        super(DPC_RNN, self).__init__()

        # to reproduce the experiments
        torch.cuda.manual_seed(233)
        print('Using DPC-RNN model')

        # number of dimensions in the image
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.distance = distance

        # how many futures to predict
        self.pred_step = pred_step

        # 2 if seq_len is 5
        if network == 'resnet8' or network == 'resnet10':
            self.last_duration = int(math.ceil(seq_len / 2))
        else:
            self.last_duration = int(math.ceil(seq_len / 4))

        # 4 if size of the image is 128

        # change for toy experiment
        #self.last_size = 1
        self.last_size = int(math.ceil(sample_size / 32))

        print('final feature map has size %dx%d' %
              (self.last_size, self.last_size))

        # f - choose an appropriate feature extractor. In this case, a resent
        self.backbone, self.param = select_resnet(
            network, track_running_stats=False, distance=self.distance)

        #print (self.param)

        self.param['num_layers'] = 1  # param for GRU
        self.param['hidden_size'] = self.param['feature_size']  # param for GRU

        self.agg = ConvGRU(input_size=self.param['feature_size'],
                           hidden_size=self.param['hidden_size'],
                           kernel_size=1,
                           num_layers=self.param['num_layers'])

        # two layered network \phi
        self.network_pred = nn.Sequential(
            nn.Conv2d(self.param['feature_size'],
                      self.param['feature_size'], kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.param['feature_size'],
                      self.param['feature_size'], kernel_size=1, padding=0)
        )

        # what does mask do ?
        self.mask = None
        self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.agg)
        self._initialize_weights(self.network_pred)
        
        # exponential map
        self.tp = hypnn.ToPoincare(c=1.0, train_x=True, train_c=True, ball_dim=self.param['feature_size'])

    def forward(self, block, return_c_t=False, regression=False):
        # block: [B, N, C, SL, W, H]
        ### extract feature ###
        # [ Batch , Number of sequences, Channels, Sequence Length, Height, Weight ]
        #         print(block.shape)
        (B, N, C, SL, H, W) = block.shape
        # [ 4, 8, 3, 256, 128, 128 ]

        # batch and number of sequences can be combined
        block = block.view(B * N, C, SL, H, W)
        # [ 32, 3, 256, 128, 128 ]

        # pass through backbone (f)
        feature = self.backbone(block)
        #[32, 256, 2, 4, 4]

        del block

        # pool{2} as denoted in the paper
        feature = F.avg_pool3d(
            feature, (self.last_duration, 1, 1), stride=(1, 1, 1))
        # [32, 256, 1, 4, 4]

        feature_inf_all = feature.view(
            B, N, self.param['feature_size'], self.last_size, self.last_size)  # before ReLU, (-inf, +inf)
        # [4, 8, 256, 4, 4]

        feature = self.relu(feature)  # [0, +inf)
        # [32, 256, 1, 4, 4]

        # [B,N,D,6,6], [0, +inf)
        feature = feature.view(
            B, N, self.param['feature_size'], self.last_size, self.last_size)
        # [4, 8, 256, 4, 4]

        # makes a copy of the tensor (why do we need this ?)
        feature_inf = feature_inf_all[:, N - self.pred_step::, :].contiguous()
        # [4, 3, 256, 4, 4]

        del feature_inf_all

        ### aggregate, predict future ###
        # [4, 5, 256, 4, 4]
        context, hidden = self.agg(
            feature[:, 0:N - self.pred_step, :].contiguous())
        # [4, 1, 256, 4, 4]

        #print (context[:,-1,:]==hidden)
        # after tanh, (-1,1). get the hidden state of last layer, last time step
        hidden = hidden[:, -1, :]
        # [4, 256, 4, 4]

        # Return context & stop if specified
        if return_c_t:
            hidden = F.avg_pool3d(hidden, (1, self.last_size, self.last_size), stride=1).squeeze(-1).squeeze(-1)
            return hidden
        
        # aggregate the results for pre_step number of steps
        # check this out ??
        pred = []
        for i in range(self.pred_step):
            # sequentially pred future for pred_step number of times
            p_tmp = self.network_pred(hidden)
            # print(p_tmp.shape)

            pred.append(p_tmp)
            context, hidden = self.agg(
                self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
            hidden = hidden[:, -1, :]

        #[4, 256, 4, 4]

        pred = torch.stack(pred, 1)  # B, pred_step, xxx
        #[4, 3, 256, 4, 4]

        del hidden

        ### Get similarity score ###

        # pred: [B, pred_step, D, last_size, last_size]
        # GT: [B, N, D, last_size, last_size]

        N = self.pred_step
        # dot product D dimension in pred-GT pair, get a 6d tensor. First 3 dims are from pred, last 3 dims are from GT.

        # predicted
        pred = pred.permute(0, 1, 3, 4, 2).contiguous().view(
            B * self.pred_step * self.last_size**2, self.param['feature_size'])

        # GT
        feature_inf = feature_inf.permute(0, 1, 3, 4, 2).contiguous().view(
            B * N * self.last_size**2, self.param['feature_size']).transpose(0, 1)
        
        if self.distance == 'poincare':
            pred = self.tp(pred)
            feature_inf = self.tp(feature_inf.transpose(0, 1)).transpose(0, 1)

        # dot product to get the score
        # .view(B, self.pred_step, self.last_size**2, B, N, self.last_size**2)
        if self.distance == 'dot':
            score = torch.matmul(pred, feature_inf)
        elif self.distance == 'poincare':
            score = -poincare_distance(pred, feature_inf.T)
        #print (score.shape, score)
        if not regression:
            del feature_inf, pred

        if self.mask is None:  # only compute mask once
            # mask meaning:
            # -2: omit,
            # -1: temporal neg (hard),
            # 0: easy neg,
            # 1: pos,
            # -3: spatial neg

            # easy negatives (do not take gradient here)
            mask = torch.zeros((B, self.pred_step, self.last_size**2, B, N, self.last_size**2),
                               dtype=torch.int8, requires_grad=False).detach().cuda()

            # spatial negative (mark everything in the same batch as spatial negative)
            mask[torch.arange(B), :, :, torch.arange(B),
                 :, :] = -3  # spatial neg

            # temporal negetive
            for k in range(B):
                mask[k, :, torch.arange(
                    self.last_size**2), k, :, torch.arange(self.last_size**2)] = -1  # temporal neg

            # positive
            tmp = mask.permute(0, 2, 1, 3, 5, 4).contiguous().view(
                B * self.last_size**2, self.pred_step, B * self.last_size**2, N)
            for j in range(B * self.last_size**2):
                tmp[j, torch.arange(self.pred_step), j, torch.arange(
                    N - self.pred_step, N)] = 1  # pos
            mask = tmp.view(B, self.last_size**2, self.pred_step,
                            B, self.last_size**2, N).permute(0, 2, 1, 3, 5, 4)
            self.mask = mask

        if regression:
            return [pred, feature_inf.transpose(0, 1), score, self.mask]
        return [score, self.mask]

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself

    def reset_mask(self):
        self.mask = None


if __name__ == '__main__':

    mymodel = DPC_RNN(128, num_seq=8, seq_len=5,
                      pred_step=3, network='resnet18')
    mymodel = mymodel.cuda()

    # (B, N, C, SL, H, W)
    mydata = torch.cuda.FloatTensor(1, 8, 3, 5, 128, 128).cuda()

    nn.init.normal_(mydata)
    #import ipdb; ipdb.set_trace()
    [score, mask] = mymodel(mydata)
    print(score)