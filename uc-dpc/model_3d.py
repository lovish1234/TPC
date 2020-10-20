# get the model as DPC-RNN

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../uc-backbone')

# to extract the features
from select_backbone import select_resnet

# to aggregate the features in one
from convrnn import ConvGRU


class DPC_RNN(nn.Module):
    '''DPC with RNN'''

    def __init__(self,
                 sample_size,
                 num_seq=8,
                 seq_len=5,
                 pred_step=3,
                 network='resnet10',
                 distance='L2',
                 distance_type='uncertain',
                 positive_vs_negative='same',
                 radius_type='linear',
                 radius_which='pred'):
        super(DPC_RNN, self).__init__()

        # to reproduce the experiments
        torch.cuda.manual_seed(233)
        print('[model_3d.py] Using DPC-RNN model')

        # number of dimensions in the image
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len

        self.distance = distance
        self.distance_type = distance_type
        self.positive_vs_negative = positive_vs_negative
        self.radius_which = radius_which
        self.radius_type = radius_type

        print('[model_3d.py] Using distance metric : ', self.distance)
        print('[model_3d.py] Using distance type : ', self.distance_type)
        print('[model_3d.py] Treating positive and negative instances as : ',
              self.positive_vs_negative)
        print('[model_3d.py] Using radius type : ', self.radius_type)
        # how many futures to predict
        self.pred_step = pred_step

        # what is sample size ?
        # 2 if seq_len is 5
        if network == 'resnet8' or network == 'resnet10':
            self.last_duration = int(math.ceil(seq_len / 2))
        else:
            self.last_duration = int(math.ceil(seq_len / 4))

        self.last_size = int(math.ceil(sample_size / 32))

        # print('final feature map has size %dx%d' %
        #       (self.last_size, self.last_size))

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
        self.mask = None
        self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.agg)
        self._initialize_weights(self.network_pred)

    def forward(self, block):
        # block: [B, N, C, SL, W, H]
        ### extract feature ###
        # [ Batch , Number of sequences, Channels, Sequence Length, Height, Weight ]

        (B, N, C, SL, H, W) = block.shape
        # [ 4, 8, 3, 256, 128, 128 ]

        # batch and number of sequences can be combined
        block = block.view(B * N, C, SL, H, W)
        # [ 32, 3, 256, 128, 128 ]

        # pass through backbone (f)
        feature = self.backbone(block)
        #[32, 256, 2, 4, 4]

        # if self.distance == 'circle' and self.radius_type=='log':
        #     # predict abs(r) instead of (r)
        #     feature[:,-1,:,:,:] = torch.exp(feature[:,-1,:,:,:])

        del block

        # pool{2} as denoted in the paper
        feature = F.avg_pool3d(
            feature, (self.last_duration, 1, 1), stride=(1, 1, 1))
        # [32, 256, 1, 4, 4]
        # In case we use circle loss, this would be [32, 257, 1, 4, 4]

        # logging the radii of tubes here
        # We have
        #print (self.param['feature_size'], feature.shape)
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
        gt = feature_inf_all[:, N - self.pred_step::, :].contiguous()
        # [4, 3, 256, 4, 4]
        del feature_inf_all

        ### aggregate, predict future ###
        # [4, 5, 256, 4, 4]
        _, hidden = self.agg(feature[:, 0:N - self.pred_step, :].contiguous())
        # [4, 1, 256, 4, 4]
        # after tanh, (-1,1). get the hidden state of last layer, last time step
        hidden = hidden[:, -1, :]
        # [4, 256, 4, 4]
        # get the results for pre_step number of steps
        pred = []
        for i in range(self.pred_step):
            # sequentially pred future for pred_step number of times

            #print (hidden.shape)
            p_tmp = self.network_pred(hidden)

            #print (p_tmp[:,-1,:,:])
            if self.distance_type == 'uncertain' and self.radius_type == 'log':
                p_tmp = self.activation(p_tmp)

            #print (p_tmp[:,-1,:,:])
            pred.append(p_tmp)

            # take hidden state along with encoding
            _, hidden = self.agg(
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
        # leave the radius out

        if self.distance_type == 'uncertain':
            pred_embedding = pred[:, :-1]
            pred_radius = pred[:, -1].expand(1, -1)
        elif self.distance_type == 'certain':
            pred_embedding = pred

        # GT
        gt = gt.permute(0, 1, 3, 4, 2).contiguous().view(
            B * N * self.last_size**2, self.param['feature_size'])
        # leave the radius out

        if self.distance_type == 'uncertain':
            gt_embedding = gt[:, :-1]
            gt_radius = gt[:, -1].expand(1, -1)
        elif self.distance_type == 'certain':
            gt_embedding = gt

        # dot product to get the score
        # change this using einstein notation

        if self.distance == 'dot':
            gt_embedding = gt_embedding.transpose(0, 1)
            score = torch.matmul(pred_embedding, gt_embedding)
            # print(score)
        elif self.distance == 'cosine':
            pred_norm = torch.norm(pred_embedding, dim=1)
            gt_norm = torch.norm(gt_embedding, dim=1)

            #print(pred_embedding.shape, pred_norm.shape, pred_norm.expand(1,-1).shape)
            #print(gt_embedding.shape, gt_norm.shape, gt_norm.expand(1,-1).shape)

            gt_embedding = gt_embedding.transpose(0, 1)
            score = torch.matmul(pred_embedding, gt_embedding)

            #print("score shape: (%d, %d)" % (score.shape[0], score.shape[1]))
            # print("max value of dot product: %f" %
            #       torch.max(score).detach().cpu().numpy())
            # print("min value of dot product: %f" %
            #       torch.min(score).detach().cpu().numpy())
            # normalizing with magnitudes

            # row-wise division
            score = torch.div(score, pred_norm.expand(1, -1).T)
            # column-wise division
            score = torch.div(score, gt_norm)
            # print("max value of cosine: %f" %
            #       np.max(score.detach().cpu().numpy()))
            # print("min value of cosine: %f" %
            #       np.min(score.detach().cpu().numpy()))
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

        # on the certainity of distances
        if self.distance_type == 'uncertain':
            if self.radius_which == 'pred':
                #print ('[model_3d.py] Using the pred radii of tube')
                # .view(B, self.pred_step, self.last_size**2, B, N, self.last_size**2)

                #print (score.shape, pred_radius.shape)
                final_score = (score - pred_radius.T).contiguous()
            elif self.radius_which == 'gt':
                #print ('[model_3d.py] Using the ground truth radii of tube')
                # .view(B, self.pred_step, self.last_size**2, B, N, self.last_size**2)

                #print (score.shape, gt_radius.shape)
                final_score = (score - gt_radius).contiguous()

            zero_tensor = torch.zeros_like(final_score)

            # treat both positive and negative instances as same wrt score function
            if self.positive_vs_negative == 'same':
                #print ('[model_3d.py] Setting distance_type to same')
                final_score = torch.max(torch.stack(
                    [final_score, zero_tensor]), axis=0).values
                del zero_tensor
            elif self.positive_vs_negative == 'different':
                #print ('[model_3d.py] Setting distance_type to different')
                # check if it's a positive or negative instance
                # if positive leave as is
                # if negative multiply with -1
                ones_tensor = -torch.ones_like(final_score)
                torch.diagonal(ones_tensor).fill_(1.0)

                # invert the score if negatives, take maximum
                final_score = torch.max(torch.stack(
                    [final_score * ones_tensor, zero_tensor]), axis=0).values
                # corresponding to first block - take first column
                del zero_tensor, ones_tensor
        elif self.distance_type == 'certain':
            # .view(B, self.pred_step, self.last_size**2, B, N, self.last_size**2)
            final_score = score.contiguous()
            zero_tensor = torch.zeros_like(final_score)
            if self.positive_vs_negative == 'same':
                #print ('[model_3d.py] Setting distance_type to same')
                final_score = torch.max(torch.stack(
                    [final_score, zero_tensor]), axis=0).values
                del zero_tensor
            elif self.positive_vs_negative == 'different':
                ones_tensor = -torch.ones_like(final_score)
                torch.diagonal(ones_tensor).fill_(1.0)
                final_score = torch.max(torch.stack(
                    [final_score * ones_tensor, zero_tensor]), axis=0).values
                #print (final_score)
                del zero_tensor, ones_tensor
        del score

        # Mask Calculation
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

        # final_score returned as predxGT matrix
        return [final_score, self.mask]

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself

    def reset_mask(self):
        self.mask = None


class exp_activation(nn.Module):
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()  # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return exp_radius(input)


def exp_radius(input):

    input[:, -1, :, :] = torch.exp(input[:, -1, :, :])
    return input


if __name__ == '__main__':

    mymodel = DPC_RNN(128, num_seq=8, seq_len=5, pred_step=3,
                      distance='L2',
                      distance_type='uncertain',
                      positive_vs_negative='different',
                      radius_type='linear',
                      radius_which='pred',
                      network='resnet18')

    mymodel = mymodel.cuda()

    # (B, N, C, SL, H, W)
    mydata = torch.cuda.FloatTensor(1, 8, 3, 5, 128, 128)

    nn.init.uniform_(mydata,100,100000)
    #import ipdb; ipdb.set_trace()
    mymodel(mydata)

    # x = torch.tensor([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
    # y = torch.tensor([[8,8,8],[16,16,16],[32,32,32],[64,64,64]])
    # x_em = x[:,:-1]
    # y_em = y[:,:-1]
    # x_ri = x[:,-1].expand(1,-1)
    # z = y_em.reshape(y_em.shape[0], 1, y_em.shape[1])
    # difference = z - x_em
    # score = torch.sqrt(torch.einsum('ijk,ijk->ij', difference, difference).float())
    # score = score - x_ri
    # score = score.permute(1,0)