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

# to calculate loss
from uncertainty import process_uncertainty


class DPC_RNN(nn.Module):
    '''DPC with RNN'''

    def __init__(self, sample_size, num_seq=8, seq_len=5,
                 pred_step=3, network='resnet18', distance='cosine',
                 distance_type='uncertain', weighting=True,
                 margin=10, pool='None', radius_location='Phi',
                 loss_type='MSE', action_cls_head=False,
                 dropout=0.5, num_class=101):
        super(DPC_RNN, self).__init__()

        # to reproduce the experiments
        torch.cuda.manual_seed(233)
        print('Using DPC-RNN model')
        print('[model_3d.py] using loss function: %s' % loss_type)
        print('[model_3d.py] using distance type: %s' % distance_type)
        print('[model_3d.py] using distance metric: %s' % distance)

        # number of dimensions in the image
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len

        self.distance = distance
        self.distance_type = distance_type
        self.weighting = weighting

        # how many futures to predict
        self.pred_step = pred_step
        self.margin = margin
        self.pool = pool  # TODO verify pooling settings
        self.radius_location = radius_location
        self.loss_type = loss_type
        self.action_cls_head = action_cls_head  # If true, added a LC layer at the end

        # 2 if seq_len is 5
        if network == 'resnet8' or network == 'resnet10':
            self.last_duration = int(math.ceil(seq_len / 2))
        else:
            self.last_duration = int(math.ceil(seq_len / 4))

        # 4 if size of the image is 128

        # change for toy experiment
        #self.last_size = 1
        if self.pool in ['avg', 'max']:
            self.last_size = 1
        else:
            self.last_size = int(math.ceil(sample_size / 32))

        print('final feature map has size %dx%d' %
              (self.last_size, self.last_size))

        # f - choose an appropriate feature extractor. In this case, a resent
        if self.radius_location == 'Phi':
            self.backbone, self.param = select_resnet(
                network, track_running_stats=False)
        elif self.radius_location == 'F':
            self.backbone, self.param = select_resnet(
                network, track_running_stats=False)

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

        self.avg_pool = nn.AdaptiveAvgPool3d(
            (1, self.last_size, self.last_size))
        self.max_pool = nn.AdaptiveMaxPool3d(
            (1, self.last_size, self.last_size))

        # mask can be used to retrieve positive and negative distance
        self.mask = None
        self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.agg)
        self._initialize_weights(self.network_pred)

        self.action_cls_head = action_cls_head
        self.num_class = num_class
        self.dropout = dropout
        if action_cls_head:
            print('Using FC head for action classification')
            # See eval/model_3d_lc.py
            self.num_class = num_class
            self.final_bn = nn.BatchNorm1d(self.param['feature_size'])
            self.final_bn.weight.data.fill_(1)
            self.final_bn.bias.data.zero_()
            self.final_fc = nn.Sequential(nn.Dropout(dropout),
                                          nn.Linear(self.param['feature_size'], self.num_class))
            self._initialize_weights(self.final_fc)

    def forward(self, block):
        # block: [B, N, C, SL, W, H]
        ### extract feature ###
        # [ Batch , Number of sequences, Channels, Sequence Length, Height, Weight ]
        #         print(block.shape)
        (B, N, C, SL, H, W) = block.shape
        # [ 4, 8, 3, 256/257, 128, 128 ]

        # batch and number of sequences can be combined
        block = block.view(B * N, C, SL, H, W)
        # [ 32, 3, 256/257, 128, 128 ]

        # pass through backbone (f)
        feature = self.backbone(block)
        #[32, 256/257, 2, 4, 4]

        del block

        # pool{2} as denoted in the paper
        feature = F.avg_pool3d(
            feature, (self.last_duration, 1, 1), stride=(1, 1, 1))
        # [32, 256/257, 1, 4, 4]

        if self.pool == 'avg':
            feature = self.avg_pool(feature)
        elif self.pool == 'max':
            feature = self.max_pool(feature)

        feature_inf_all = feature.view(
            B, N, self.param['feature_size'], self.last_size, self.last_size)  # before ReLU, (-inf, +inf)
        # [4, 8, 256/257, 4, 4]

        feature = self.relu(feature)  # [0, +inf)
        # [32, 256/257, 1, 4, 4]

        # [B,N,D,6,6], [0, +inf)
        feature = feature.view(
            B, N, self.param['feature_size'], self.last_size, self.last_size)
        # [4, 8, 256/257, 4, 4]

        # makes a copy of the tensor (why do we need this ?)
        feature_inf = feature_inf_all[:, N - self.pred_step::, :].contiguous()
        # [4, 3, 256/257, 4, 4]

        del feature_inf_all

        ### aggregate, predict future ###
        # [4, 5, 256/257, 4, 4]
        context, hidden = self.agg(
            feature[:, 0:N - self.pred_step, :].contiguous())
        # [4, 1, 256/257, 4, 4]

        #print (context[:,-1,:]==hidden)
        # after tanh, (-1,1). get the hidden state of last layer, last time step
        hidden = hidden[:, -1, :]
        # [4, 256/257, 4, 4]

        # aggregate the results for pre_step number of steps
        # check this out ??
        pred = []
        for i in range(self.pred_step):
            # sequentially pred future for pred_step number of times
            p_tmp = self.network_pred(hidden)
            # print(p_tmp.shape)

            pred.append(p_tmp)
            if self.distance_type == 'uncertain' and self.radius_location == 'Phi':
                # remove radius channel before passing to agg
                p_tmp = p_tmp[:, :-1, :, :]
#             print(p_tmp.shape)
            context, hidden = self.agg(
                self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
            hidden = hidden[:, -1, :]

        if self.action_cls_head:
            # Supervised operation
            # Classify last context into action, see model_lc_future.py
            context = context[:, -1, :].unsqueeze(1)
            context = F.avg_pool3d(
                context, (1, self.last_size, self.last_size), stride=1).squeeze(-1).squeeze(-1)
            context = self.final_bn(
                context.transpose(-1, -2)).transpose(-1, -2)
            action_output = self.final_fc(context).view(B, -1, self.num_class)

            # DEBUG:
#             print('action_output:', action_output.shape) # TODO: we expect & want second dimension to be 1
            result = (action_output, context)
            return result
        #[4, 256/257, 4, 4]

        pred = torch.stack(pred, 1)  # B, pred_step, xxx
        #[4, 3, 256/257, 4, 4]

        del hidden

        ### Get similarity score ###

        # pred: [B, pred_step, D, last_size, last_size]
        # GT: [B, N, D, last_size, last_size]

        N = self.pred_step
        # dot product D dimension in pred-GT pair, get a 6d tensor. First 3 dims are from pred, last 3 dims are from GT.

        # predicted
        if self.distance_type == 'certain':
            pred = pred.permute(0, 1, 3, 4, 2).contiguous().view(
                B * self.pred_step * self.last_size**2, self.param['feature_size'])
        elif self.distance_type == 'uncertain':
            pred = pred.permute(0, 1, 3, 4, 2).contiguous().view(
                B * self.pred_step * self.last_size**2, self.param['feature_size'] + 1)
        # GT

        feature_inf = feature_inf.permute(0, 1, 3, 4, 2).contiguous().view(
            B * N * self.last_size**2, self.param['feature_size'])  # .transpose(0, 1)

        if self.distance_type == 'uncertain':
            pred_embedding = pred[:, :-1]
            pred_radius = pred[:, -1].expand(1, -1)
        elif self.distance_type == 'certain':
            pred_embedding = pred

        gt_embedding = feature_inf

#########################################Similarity Score#########################################

        if self.distance == 'dot':
            gt_embedding = gt_embedding.transpose(0, 1)
            score = torch.matmul(pred_embedding, gt_embedding)
            # print(score)
        elif self.distance == 'cosine':
            pred_norm = torch.norm(pred_embedding, dim=1)
            gt_norm = torch.norm(gt_embedding, dim=1)

            gt_embedding = gt_embedding.transpose(0, 1)
            score = torch.matmul(pred_embedding, gt_embedding)

            # row-wise division
            score = torch.div(score, pred_norm.expand(1, -1).T)
            # column-wise division
            score = torch.div(score, gt_norm)
#             score = 1 - (score + 1) / 2
#             print(score[:10, :10])

            del pred_embedding, gt_embedding

            # division by the magnitude of respective vectors
        elif self.distance == 'L2':
            pred_embedding_mult = pred_embedding.reshape(
                pred_embedding.shape[0], 1, pred_embedding.shape[1])
            difference = pred_embedding_mult - gt_embedding
            score = torch.sqrt(torch.einsum(
                'ijk,ijk->ij', difference, difference))
            # print(score)
            del pred_embedding_mult, gt_embedding, difference

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

# Uncertainty#############################################\

            if self.distance_type == 'uncertain':
                pred_radius_matrix = pred_radius.expand(score.shape).T
#                 print('here')
                [final_score, pred_radius, score, final_radius] = process_uncertainty(score, pred_radius,
                                                                                      weighting=self.weighting,
                                                                                      distance=self.distance,
                                                                                      margin=self.margin,
                                                                                      distance_type=self.distance_type,
                                                                                      loss_type=self.loss_type)
            elif self.distance_type == 'certain':
                # .view(B, self.pred_step, self.last_size**2, B, N, self.last_size**2)
                [final_score, pred_radius, score, final_radius] = process_uncertainty(score, None,
                                                                                      weighting=self.weighting,
                                                                                      distance=self.distance,
                                                                                      margin=self.margin,
                                                                                      distance_type=self.distance_type,
                                                                                      loss_type=self.loss_type)

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

        if self.distance_type == 'uncertain':
            return [final_score, self.mask, pred_radius, score]
        return [final_score, self.mask, torch.zeros(len(final_score)).cuda(), score]

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself

    def reset_mask(self):
        self.mask = None

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
    [final_score, mask, pred_radius, score] = mymodel(mydata)
    print(score)
