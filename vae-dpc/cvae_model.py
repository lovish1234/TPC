# BVH, April 2020
# DPC with CVAE-RNN to model video future uncertainty

import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../backbone')

from select_backbone import select_resnet  # to extract the features
from convrnn import ConvGRU  # to aggregate the features in one
from cvae_backbone import *  # to perform future projection
from vae_common import *  # to measure variance


class DPC_CVAE(nn.Module):
    '''
    DPC with RNN / CVAE.
    action_cls_head: If True, adds a fully connected layer for action space classification.
    force_encode_train / val: If True, always use the encoder to produce Gaussian distribution parameters.
    return_c_t: If True, return the present context c_t and nothing else (similar to LC_present).
    '''

    def __init__(self, img_dim, num_seq=8, seq_len=5,
                 pred_step=3, network='resnet50', cvae_arch='fc',
                 action_cls_head=False, dropout=0.5, num_class=101):
        super(DPC_CVAE, self).__init__()

        # to reproduce the experiments
        torch.cuda.manual_seed(233)
        print('Using DPC-CVAE model ' + network + ' ' + cvae_arch)

        # number of dimensions in the image
        self.img_dim = img_dim
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.action_cls_head = action_cls_head

        if action_cls_head:
            print('Action classification head(s) enabled with final FC')
        # if force_encode_train:
        #     print('::: WARNING ::: Gaussian parameter encoding will take place during TRAIN, which might inflate accuracy!')
        # if force_encode_eval:
        #     print('::: WARNING ::: Gaussian parameter encoding will take place during EVAL, which will inflate accuracy!')

        # how many futures to predict
        self.pred_step = pred_step

        # 2 if seq_len is 5
        if network == 'resnet8' or network == 'resnet10':
            self.last_duration = int(math.ceil(seq_len / 2))
        else:
            self.last_duration = int(math.ceil(seq_len / 4))

        # 4 if size of the image is 128
        self.last_size = int(math.ceil(img_dim / 32))
        self.spatial_size = self.last_size

        print('final feature map has size %dx%d' %
              (self.last_size, self.last_size))

        # f - choose an appropriate feature extractor. In this case, a resent
        self.backbone, self.param = select_resnet(
            network, track_running_stats=False)

        #print (self.param)

        self.param['num_layers'] = 1  # param for GRU
        self.param['hidden_size'] = self.param['feature_size']  # param for GRU

        # Converts input (video block representation) + old hidden state to new hidden state
        self.agg = ConvGRU(input_size=self.param['feature_size'],
                           hidden_size=self.param['hidden_size'],
                           kernel_size=1,
                           num_layers=self.param['num_layers'])

        # two layered network \phi
        # Replaced with CVAE
        # self.network_pred = nn.Sequential(
        #     nn.Conv2d(self.param['feature_size'],
        #               self.param['feature_size'], kernel_size=1, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.param['feature_size'],
        #               self.param['feature_size'], kernel_size=1, padding=0)
        # )
        if cvae_arch == 'fc':
            print('Using CVAE class: My_CVAE_FC')
            self.network_pred_cvae = My_CVAE_FC(self.param['feature_size'] * self.last_size * self.last_size,
                                                self.param['feature_size'] *
                                                self.last_size * self.last_size,
                                                latent_size=256, hidden_size=1024)
        elif cvae_arch == 'conv' or cvae_arch == 'conv_a':
            # Conv 1x1 version A
            print('Using CVAE class: My_CVAE_Conv1x1_A (latent=64x4x4, hidden=128x4x4)')
            self.network_pred_cvae = My_CVAE_Conv1x1(self.param['feature_size'], self.param['feature_size'],
                                                     latent_size=64, hidden_size=128)
        elif cvae_arch == 'conv_b':
            # Conv 1x1 version B (smaller latent dimension)
            print('Using CVAE class: My_CVAE_Conv1x1_B (latent=16x4x4, hidden=128x4x4)')
            self.network_pred_cvae = My_CVAE_Conv1x1(self.param['feature_size'], self.param['feature_size'],
                                                     latent_size=16, hidden_size=128)
        elif cvae_arch == 'conv_c':
            # Conv 1x1 version C (even smaller latent dimension)
            print('Using CVAE class: My_CVAE_Conv1x1_C (latent=4x4x4, hidden=128x4x4)')
            self.network_pred_cvae = My_CVAE_Conv1x1(self.param['feature_size'], self.param['feature_size'],
                                                     latent_size=4, hidden_size=128)
        elif cvae_arch == 'conv_d':
            # Conv + FC version D (global latent space)
            print('Using CVAE class: My_CVAE_ConvFC (latent=8x1x1, hidden=256x4x4)')
            self.network_pred_cvae = My_CVAE_ConvFC(self.param['feature_size'], self.param['feature_size'],
                                                    latent_size=8, hidden_size=256, spatial_size=self.last_size)
        elif cvae_arch == 'conv_e':
            # Conv + FC version E (global latent space, size 16)
            print('Using CVAE class: My_CVAE_ConvFC (latent=16x1x1, hidden=256x4x4)')
            self.network_pred_cvae = My_CVAE_ConvFC(self.param['feature_size'], self.param['feature_size'],
                                                    latent_size=16, hidden_size=256, spatial_size=self.last_size)
        elif cvae_arch == 'vrnn_a':
            print('Using VRNN class: My_VRNN_A')
        else:
            raise Exception('CVAE architecture not recognized: ' + cvae_arch)

        self.mask = None
        self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.agg)
        self._initialize_weights(self.network_pred_cvae)

        self.action_cls_head = action_cls_head
        if action_cls_head:
            # See eval/model_3d_lc.py
            if isinstance(num_class, int):
                num_class = [num_class] # singleton to simplify code
            assert(isinstance(num_class, list))
            self.num_class = num_class
            self.final_bn = nn.BatchNorm1d(self.param['feature_size'])
            self.final_bn.weight.data.fill_(1)
            self.final_bn.bias.data.zero_()
            self.final_fc = []
            for cur_num_cls in num_class:
                cur_fc = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(self.param['feature_size'], cur_num_cls))
                self._initialize_weights(cur_fc)
                self.final_fc.append(cur_fc)
            self.final_fc = nn.ModuleList(self.final_fc) # IMPORTANT, otherwise pytorch won't register


    def get_output_vectors(self, context):
        '''
        Classify the given context into one or multiple output layers (depending on model configuration).
        '''
        B = context.shape[0]
        action_outputs = []
        for i in range(len(self.num_class)):
            cur_out = self.final_fc[i](context).view(B, -1, self.num_class[i])
            action_outputs.append(cur_out)
        return action_outputs


    def forward(self, block, do_encode=False, get_pred_sim=False, diversity=False,
                paths=20, return_c_t=False, spatial_separation=False):
        '''
        get_pred_sim: Measure and include cosine distance between two random predictions for diversity control purposes.
        return_c_t: If True, return hidden state c_t instead of normal output (used for present matching)
        spatial_separation: Whether to avoid averaging over H, W when calculating mean distance metrics (relevant for diversity only).
        '''

        if diversity:
            return self.forward_diversity(block, paths=paths, spatial_separation=spatial_separation)

        # block: [B, N, C, SL, H, W]
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
        # print('context:', context.shape) # batch size 32 => (8, 5, 256, 4, 4)
        # print('hidden:', hidden.shape) # batch size 32 => (8, 1, 256, 4, 4)
        # after tanh, (-1,1). get the hidden state of last layer, last time step
        # [4, 256, 4, 4]
        context = context[:, -1, :]
        hidden = hidden[:, -1, :]
        assert(torch.abs(context - hidden).sum().item() < 1e-3)
        
        if return_c_t:
            context = F.avg_pool3d(context, (1, self.last_size, self.last_size), stride=1).squeeze(-1).squeeze(-1)
            return context

        # Predictions should be sampled using CVAE => many examples allow estimation of diversity

        pred = []
        mus = []  # stays empty during eval
        logvars = []  # stays empty during eval
        pred_sims = []  # stays empty during eval or if get_pred_sim is False

        for i in range(self.pred_step):
            # sequentially pred future for pred_step number of times
            # p_tmp = self.network_pred(hidden)

            # Use CVAE
            # cvae_x = context # follow paper
            cvae_x = hidden
            cvae_y = feature[:, N - self.pred_step + i]
            if do_encode:
                # Uses biased Gaussian parameters inferred from ground truth future
                p_tmp, mu_tmp, logvar_tmp = self.network_pred_cvae.forward_train(
                    cvae_y, cvae_x)

            if get_pred_sim:
                # Measure diversity of two samples from N(0, I) to encourage diversity with extra loss term
                # NOTE: this is just the direct output variance and does not capture context whatsoever
                # Implemented as cosine similarity between two spatially pooled predictions
                p_div1 = self.network_pred_cvae.forward_test(cvae_x).view(
                    B, self.param['feature_size'], self.last_size, self.last_size).mean(dim=[2, 3])
                p_div2 = self.network_pred_cvae.forward_test(cvae_x).view(
                    B, self.param['feature_size'], self.last_size, self.last_size).mean(dim=[2, 3])
                pred_sim = F.cosine_similarity(p_div1, p_div2, dim=1)
                pred_sims.append(pred_sim)
                del p_div1, p_div2

            else:
                # During evaluation / testing OR finetuning, DO NOT look into the future!
                p_tmp = self.network_pred_cvae.forward_test(cvae_x)

            p_tmp = p_tmp.view(
                B, self.param['feature_size'], self.last_size, self.last_size)
            pred.append(p_tmp)
            if do_encode:
                # mu_tmp = (8, 4, 4, 4) = (B, latent_size, y, x)
                mus.append(mu_tmp)
                logvars.append(logvar_tmp)

            context, hidden = self.agg(
                self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
            # context = context[:, -1, :]
            hidden = hidden[:, -1, :]

        if do_encode:
            # if conv: mus = (8, 3, 4, 4, 4) = (B, pred_step, latent_size, H, W)
            mus = torch.stack(mus, dim=1)  # B, pred_step, latent_size(, H, W)
            # B, pred_step, latent_size(, H, W)
            logvars = torch.stack(logvars, dim=1)
        if get_pred_sim:
            pred_sims = torch.stack(pred_sims, dim=1)  # B, pred_step

        if self.action_cls_head:
            # Supervised operation
            # Classify last context into action, see model_lc_future.py
            context = context[:, -1, :].unsqueeze(1)
            context = F.avg_pool3d(
                context, (1, self.last_size, self.last_size), stride=1).squeeze(-1).squeeze(-1)
            context = self.final_bn(context.transpose(-1, -2)).transpose(-1, -2)
            action_outputs = self.get_output_vectors(context)  # tensor or list of tensors
            # DEBUG:
            # print('action_output:', action_output.shape) # (16, 1, 125)
            if do_encode:
                if get_pred_sim:
                    result = (action_outputs, context, mus, logvars, pred_sims)
                else:
                    result = (action_outputs, context, mus, logvars)
            else:
                result = (action_outputs, context)
            return result

        else:
            # Self-supervised operation

            #[4, 256, 4, 4]
            pred = torch.stack(pred, 1)  # B, pred_step, xxx
            #[4, 3, 256, 4, 4]

            del context, hidden

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

            # dot product to get the score
            # .view(B, self.pred_step, self.last_size**2, B, N, self.last_size**2)
            score = torch.matmul(pred, feature_inf)
            #print (score.shape, score)
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

            # return [score, self.mask]
            if do_encode:
                if get_pred_sim:
                    result = (score, self.mask, mus, logvars, pred_sims)
                else:
                    result = (score, self.mask, mus, logvars)
            else:
                result = (score, self.mask)
            return result

    def forward_diversity(self, block, paths=20, spatial_separation=False):
        '''
        Generate multiple samples for every future time step, and measure distance.
        NOTE: this method never uses the VAE encoder.
        spatial_separation: Whether to avoid averaging over H, W when calculating mean distance metrics.
        '''
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
        # [32, 256, 2, 4, 4]
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
        # [4, 256, 4, 4]
        context = context[:, -1, :]
        hidden = hidden[:, -1, :]
        assert(torch.abs(context - hidden).sum().item() < 1e-3)

        '''
        Predictions sampled using CVAE => many examples allow estimation of diversity as follows:
             -- z_t+1 -- c_t+1 -- z_t+2 -- ..
            /
        c_t --- z_t+1 -- c_t+1 -- z_t+2 -- ..
            \
             -- z_t+1 -- c_t+1 -- z_t+2 -- ..
        The mean distance calculation for both predictions (z) and contexts (c) happens vertically.
        If action classification is enabled, return the most likely output class per sample as well.
        (These are calculated based on the aggregated contexts at every time step.)
        '''
        mean_diversity = torch.zeros(B, 12, self.pred_step).cuda()
        # (index within batch, metric kind, future step index)

        class_types = len(self.num_class)
        action_outputs = torch.zeros(B, paths, self.pred_step, class_types).cuda()
        # (index within batch, sample index, future step index, class type)

        # Start by duplicating hidden states & contexts to create a number of independent, random future paths
        # all_contexts = []
        all_hiddens = []
        for j in range(paths):
            # all_contexts.append(context.clone())
            all_hiddens.append(hidden.clone())

        # Loop over future time steps to predict (without ever seeing ground truth)
        for i in range(self.pred_step):
            all_preds = []

            for j in range(paths):
                # Use context belonging to corresponding path
                # cur_context = all_contexts[j]
                cur_hidden = all_hiddens[j]
                # During evaluation / testing, cannot look into the future ground truth
                # p_tmp = self.network_pred_cvae.forward_test(cur_context) # follow paper
                p_tmp = self.network_pred_cvae.forward_test(cur_hidden)
                p_tmp = p_tmp.view(
                    B, self.param['feature_size'], self.last_size, self.last_size)
                all_preds.append(p_tmp)
                # Aggregate to new hidden state (NOTE: this operation is deterministic here)
                new_context, new_hidden = self.agg(
                    self.relu(p_tmp).unsqueeze(1), cur_hidden.unsqueeze(0))
                new_context = new_context[:, -1, :]
                new_hidden = new_hidden[:, -1, :]
                assert(torch.abs(new_context - new_hidden).sum().item() < 1e-3)
                # all_contexts[j] = new_context # simply replace within list
                all_hiddens[j] = new_hidden  # simply replace within list

                if self.action_cls_head:
                    # Classify hidden state samples over time into action space
                    # See forward() & eval/model_lc_future.py
                    cls_input = new_context.unsqueeze(1)
                    cls_input = F.avg_pool3d(
                        cls_input, (1, self.last_size, self.last_size), stride=1).squeeze(-1).squeeze(-1)
                    cls_input = self.final_bn(
                        cls_input.transpose(-1, -2)).transpose(-1, -2)
                    # shape (16, 1, 125 or 352) or list thereof
                    cls_output = self.get_output_vectors(cls_input)
                    # E.g. verb + noun within same network
                    for k in range(class_types):
                        cur_cls_out = torch.argmax(
                            cls_output[k], dim=2).squeeze()  # shape (16)
                        action_outputs[:, j, i, k] = cur_cls_out

            # Store mean 'diversity' of both z_t's and c_t's of this time step (by several metrics)
            # NOTE: switched to 'variance' on 04 May 2020
            if spatial_separation:
                raise Exception('Not yet supported for CVAE, please ask Basile to update this code')
            mean_var_preds = calculate_variance(all_preds)  # (B, 6)
            mean_diversity[:, :6, i] = mean_var_preds
            mean_var_hiddens = calculate_variance(all_hiddens)  # (B, 6)
            mean_diversity[:, 6:, i] = mean_var_hiddens
            del all_preds

        # [4, 3, 256, 4, 4]
        del hidden, all_hiddens, feature_inf

        # Return diversity information only (no score available) and action classes if desired
        results = {'mean_var': mean_diversity}
        if self.action_cls_head:
            results['actions'] = action_outputs
        return results

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

    mymodel = DPC_CVAE(128, num_seq=8, seq_len=5,
                       pred_step=3, network='resnet18')
    mymodel = mymodel.cuda()

    # (B, N, C, SL, H, W)
    mydata = torch.cuda.FloatTensor(1, 8, 3, 5, 128, 128).cuda()

    nn.init.normal_(mydata)
    #import ipdb; ipdb.set_trace()
    [score, mask] = mymodel(mydata)
    print(score)
