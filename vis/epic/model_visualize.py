# get the model as DPC-RNN

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as FN
sys.path.append('../tpc-backbone')

# to extract the features
from select_backbone import select_resnet

# to aggregate the features in one
from convrnn import ConvGRU


class model_visualize(nn.Module):
    def __init__(self,
                 sample_size,
                 num_seq=5,
                 seq_len=5,
                 network='resnet18',
                 distance_type='uncertain',
                 feature_type='F',
                 pred_steps=1,
                 pool='avg',
                 radius_location='Phi'):
        super(model_visualize, self).__init__()
        torch.cuda.manual_seed(666)

        # size of the image 128x128
        self.sample_size = sample_size
        self.distance_type = distance_type
        self.feature_type = feature_type
        self.network = network
        self.pool = pool
        self.pred_steps = pred_steps
        self.radius_location = radius_location

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
        print('[model_visualize.py] Using distance type : <<',
              self.distance_type, ' >>')

        # f - choose an appropriate feature extractor. In this case, a resent
        if self.radius_location == 'Phi':
            self.backbone, self.param = select_resnet(
                network, track_running_stats=False,
                distance_type='certain')
        elif self.radius_location == 'F':
            self.backbone, self.param = select_resnet(
                network, track_running_stats=False,
                distance_type=self.distance_type)

        self.param['num_layers'] = 1  # param for GRU
        self.param['hidden_size'] = self.param['feature_size']  # param for GRU

        self.agg = ConvGRU(input_size=self.param['feature_size'],
                           hidden_size=self.param['hidden_size'],
                           kernel_size=1,
                           num_layers=self.param['num_layers'])

        # two layered network \phi
        if self.radius_location == 'Phi':
            if self.distance_type == 'certain':
                output_size = self.param['feature_size']
            elif self.distance_type == 'uncertain':
                output_size = self.param['feature_size'] + 1
            self.network_pred = nn.Sequential(
                nn.Conv2d(self.param['feature_size'],
                          self.param['feature_size'], kernel_size=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.param['feature_size'],
                          output_size, kernel_size=1, padding=0)
            )
        elif self.radius_location == 'F':
            self.network_pred = nn.Sequential(
                nn.Conv2d(self.param['feature_size'],
                          self.param['feature_size'], kernel_size=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.param['feature_size'],
                          self.param['feature_size'], kernel_size=1, padding=0)
            )

        self.avg_pool = nn.AvgPool3d(
            (1, self.last_size, self.last_size), stride=1)
        self.max_pool = nn.MaxPool3d(
            (1, self.last_size, self.last_size), stride=1)

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
        feature = FN.relu(feature, inplace=False)
        feature = feature.view(
            B, N, self.param['feature_size'], self.last_size, self.last_size)

        # print(feature.shape)
#         return feature
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
#         print ('Predicting representation <<', self.pred_steps, '>> time setps ahead')
        for i in range(self.pred_steps):
            # sequentially pred future for pred_step number of times
            p_tmp = self.network_pred(context)
            #print (future.shape, hidden.shape, context.shape)
            future = p_tmp
            if self.radius_location == 'Phi' and self.distance_type == 'uncertain':
                p_tmp = p_tmp[:, :-1, :, :]
            context, hidden = self.agg(
                self.relu(p_tmp).unsqueeze(1), context.unsqueeze(0))
            context = context[:, -1, :]
        # future = self.network_pred(context)
        if self.feature_type == 'Phi':
            if self.pool == 'avg':
                future = self.avg_pool(future).squeeze(-1).squeeze(-1)
            elif self.pool == 'max':
                future = self.max_pool(future).squeeze(-1).squeeze(-1)
#             future = future.unsqueeze(1)
            return future

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet_3d.py

    def apply_weight(self, score, mask, criterion='MSE'):
        score_view = score.view(mask.shape).cuda()
        weight = mask
        weight = weight.type(torch.DoubleTensor).cuda()
        if criterion == 'MSE':
            weight_value = 1.0 / math.sqrt(len(score))
        elif criterion == 'CE':
            weight_value = 1.0 / len(score)
#         weight[mask == 0] = weight_value # easy neg
#         weight[mask == -3] = weight_value # spatial neg
#         weight[mask == -1] = weight_value # temporal neg
        weight[mask == 1] = len(score)

        score_view = score_view * weight

        [A, B, C, A, B, C] = mask.shape
        final_score = score_view.view(A * B * C, A * B * C)
        return final_score


if __name__ == '__main__':

    mymodel = DPC_RNN(128, num_seq=3, seq_len=8, pred_step=3,
                      network='resnet34',
                      distance='L2',
                      distance_type='uncertain',
                      margin=0.1)
    mymodel = mymodel.cuda()
    # (B, N, C, SL, H, W)
    mydata = torch.cuda.FloatTensor(1, 8, 3, 5, 128, 128).cuda()

    nn.init.normal_(mydata)
    #import ipdb; ipdb.set_trace()
    [final_score, mask, pred_radius, score, loss] = mymodel(mydata)
    print(score)
