# BVH, May 2020
# DPC with VRNN to model video future uncertainty

import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../backbone')

from select_backbone import select_resnet # to extract the features
from vrnn_backbone import * # to perform context aggregation & future projection
from vae_common import * # to measure variance


class DPC_VRNN(nn.Module):
    '''
    DPC with VRNN.
    action_cls_head: If True, adds a fully connected layer for action space classification.
    '''

    def __init__(self, img_dim, num_seq=8, seq_len=5, pred_step=3,
                 network='resnet50', latent_size=8, kernel_size=1, rnn_dropout=0.1,
                 action_cls_head=None, cls_dropout=0.5, num_class=101, time_indep=False):
        super(DPC_VRNN, self).__init__()
        torch.cuda.manual_seed(233)
        print('Using DPC-VRNN model, feature extractor: ' + network + ', latent size: ' + str(latent_size) + ', kernel size: ' + str(kernel_size))

        self.img_dim = img_dim
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.pred_step = pred_step
        self.latent_size = latent_size
        self.kernel_size = kernel_size
        self.rnn_dropout = rnn_dropout
        self.action_cls_head = action_cls_head
        self.cls_dropout = cls_dropout
        self.num_class = num_class
        self.time_indep = time_indep
        if action_cls_head is not None:
            print('Action classification head(s) enabled for: ' + action_cls_head)
        if time_indep:
            print('Temporal independence => ablation study enabled => VRNN is now like CVAE')

        if network == 'resnet8' or network == 'resnet10':
            # 3 if seq_len is 5
            self.last_duration = int(math.ceil(seq_len / 2))
        else:
            # 2 if seq_len is 5
            self.last_duration = int(math.ceil(seq_len / 4))

        # 4 if size of the image is 128
        self.spatial_size = int(math.ceil(img_dim / 32))

        # Feature extractor f
        self.feat_backbone, self.param = select_resnet(network, track_running_stats=False)

        # Context aggregator g and projection function phi governed by random z
        self.input_size = self.param['feature_size']
        self.state_size = self.param['feature_size']
        self.param['input_size'] = self.input_size # for input w_t and output ^w_t
        self.param['state_size'] = self.state_size # for context c_t
        self.param['latent_size'] = self.latent_size # for probabilistic latent vector z_t
        self.param['kernel_size'] = self.kernel_size # for all conv layers
        self.param['spatial_size'] = self.spatial_size # H, W
        self.param['rnn_dropout'] = self.rnn_dropout
        self.vrnn_backbone = My_VRNN_Conv_GRU(
            input_size=self.param['input_size'],
            state_size=self.param['state_size'],
            latent_size=self.param['latent_size'],
            spatial_size=self.param['spatial_size'],
            kernel_size=self.param['kernel_size'],
            dropout=self.param['rnn_dropout'],
            time_indep=self.time_indep
        )
        self.sim_div_dropout = nn.Dropout(p=self.rnn_dropout, inplace=False) # for diversity measurements

        self.mask = None
        self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.vrnn_backbone)

        if action_cls_head is not None:
            # Initializer final FC layer(s), one or multiple, depending on class configuration
            if isinstance(num_class, int):
                num_class = [num_class] # singleton to simplify code
            assert(isinstance(num_class, list))
            self.num_class = num_class
            self.final_bn = nn.BatchNorm1d(self.param['state_size'])
            self.final_bn.weight.data.fill_(1)
            self.final_bn.bias.data.zero_()
            self.final_fc = []
            for cur_num_cls in num_class:
                cur_fc = nn.Sequential(nn.Dropout(cls_dropout),
                                    nn.Linear(self.param['state_size'], cur_num_cls))
                self._initialize_weights(cur_fc)
                self.final_fc.append(cur_fc)
            self.final_fc = nn.ModuleList(self.final_fc) # IMPORTANT, otherwise pytorch won't register


    def perform_action_cls(self, context):
        '''
        Classify the given context into one or multiple output layers (depending on model configuration).
        '''
        B = context.shape[0]
        # print('context:', context.shape)
        tmp = F.avg_pool3d(context, kernel_size=(1, self.spatial_size, self.spatial_size), stride=(1, 1, 1)).squeeze(-1).squeeze(-1)
        # (B, state_size)
        # print('tmp:', tmp.shape)
        # tmp = self.final_bn(tmp.transpose(-1, -2)).transpose(-1, -2) # for old unsqueezed variant only
        tmp = self.final_bn(tmp) # forgot before 05/20
        action_outputs = []
        for i in range(len(self.num_class)):
            cur_out = self.final_fc[i](tmp).view(B, -1, self.num_class[i])
            action_outputs.append(cur_out)
        return action_outputs # (class type within list, batch index, 1, class)


    def forward(self, x, do_encode=False, get_pred_sim=False, return_c_t=False,
                return_feature=False, diversity=False, paths=20,
                force_rnn_dropout=False, spatial_separation=False):
        '''
        diversity: Branch off into many plausible embeddings at time = present.
        do_encode: Use posterior instead of prior Gaussian distribution for z.
        get_pred_sim: Measure and include cosine distance between two random predictions for diversity control purposes.
        return_c_t: Simply return the present context c_t without future predictions.
        force_rnn_dropout: Simulate dropout on c_t (p = 0.1) independently for each path (even at test time).
        spatial_separation: Whether to avoid averaging over H, W when calculating mean distance metrics (relevant for diversity only).
        '''

        if force_rnn_dropout:
            self.sim_div_dropout.train() # always perform dropout if desired

        # x: (B, N, C, SL, H, W) = (batch, num_seq, channel, seq_len, height, width)
        (B, N, C, SL, H, W) = x.shape
        # Combine batch with number of blocks
        x = x.view(B * N, C, SL, H, W)
        # Extract feature representations
        w = self.feat_backbone(x)
        # Delete video frames
        del x
        # Pool over temporal dimension
        w = F.avg_pool3d(w, kernel_size=(self.last_duration, 1, 1), stride=(1, 1, 1))
        # View features as: (B, N, F, S, S)
        w_all = w.view(B, N, self.param['input_size'], self.spatial_size, self.spatial_size)
        # Get features before and after ReLU
        w_inf_all = w_all.contiguous()
        w_relu_all = self.relu(w_all)
        if return_feature: # return output of feature extractor for nearest neighbor retrieval
            w_relu_all = F.avg_pool3d(w_relu_all, (1, self.spatial_size, self.spatial_size), stride=1).squeeze(-1).squeeze(-1)
            return w_relu_all
        # Separate input & ground truth future embeddings
        w_inf_future = w_inf_all[:, N-self.pred_step:] # needed for contrastive loss
        w_relu_seen = w_relu_all[:, :N-self.pred_step] # needed for input
        w_relu_future = w_relu_all[:, N-self.pred_step:] # needed for latent posterior
        del w_inf_all, w_relu_all

        # Aggregate all past & present features
        # NOTE: the prior distribution is not regularized, as intended
        h_last = None
        for t in range(N - self.pred_step):
            w_cur = w_relu_seen[:, t]
            if t == 0:
                mu_prio, logvar_prio = self.vrnn_backbone.prior_first(B)
            else:
                mu_prio, logvar_prio = self.vrnn_backbone.prior(h_last)
            z_cur = sample_latent(mu_prio, logvar_prio)
            no_dropout = (diversity and force_rnn_dropout and t == N-self.pred_step-1)
            h_cur = self.vrnn_backbone.recurrence(w_cur, h_last, z_cur, no_dropout=no_dropout)
            # Update embeddings for next time step
            h_last = h_cur

        # Calculate diversity if specified
        if diversity:
            return self.forward_diversity(h_last, paths, force_rnn_dropout, spatial_separation)

        # Perform context dropout if desired (NOTE: this must be after check for diversity)
        if force_rnn_dropout:
            h_last = self.sim_div_dropout(h_last)

        # Return context & stop if specified
        if return_c_t:
            h_last = F.avg_pool3d(h_last, (1, self.spatial_size, self.spatial_size), stride=1).squeeze(-1).squeeze(-1)
            return h_last

        # Perform present action classification if specified
        if self.action_cls_head == 'present':
            # Classify c_t into an action via one or multiple output vectors
            action_outputs = self.perform_action_cls(h_last)
            result = (action_outputs, h_last)
            return result

        # Initialize stochasticity & diversity related variables
        preds = []
        mus_prio = []
        mus_post = []
        logvars_prio = []
        logvars_post = []
        pred_sims = []

        # Sequentially predict the future for multiple time steps
        for t in range(self.pred_step):
            # Complete one cycle of the VRNN
            # See paper 'A recurrent latent variable model for sequential data' for explanation
            w_cur = w_relu_future[:, t] # ground truth for encoding
            mu_prio, logvar_prio = self.vrnn_backbone.prior(h_last)

            # Gather parameters for latent space Gaussian probability distribution
            if do_encode:
                # Look into the future to get tailored parameters
                mu_post, logvar_post = self.vrnn_backbone.inference(h_last, w_cur)
                mu, logvar = mu_post, logvar_post # sample z from posterior
            else:
                # Do not look into the future
                mu, logvar = mu_prio, logvar_prio # sample z from prior

            # Update input & hidden state
            z_cur = sample_latent(mu, logvar)
            w_hat = self.vrnn_backbone.generation(h_last, z_cur)
            h_cur = self.vrnn_backbone.recurrence(w_hat, h_last, z_cur) # TODO: allowed to use same z_cur sample here?

            # Measure pairwise variance (always with prior) along with latent distance if specified
            if get_pred_sim:
                z_prio1 = sample_latent(mu_prio, logvar_prio)
                z_prio2 = sample_latent(mu_prio, logvar_prio)
                w_hat1 = self.vrnn_backbone.generation(h_last, z_prio1).mean(dim=[2, 3])
                w_hat2 = self.vrnn_backbone.generation(h_last, z_prio2).mean(dim=[2, 3])
                pred_sim = F.cosine_similarity(w_hat1, w_hat2, dim=1)
                # pred_sim = F.cosine_similarity(w_hat1.mean(dim=2).mean(dim=2), w_hat2.mean(dim=2).mean(dim=2), dim=1) # (B)
                z_dist = torch.norm((z_prio1 - z_prio2).view(B, -1), p=2, dim=1) # (B)
                # print('pred_sim:', pred_sim)
                cat = torch.stack([pred_sim, z_dist], dim=-1) # (B, 2)
                # print('cat:', cat)
                pred_sims.append(cat)

            # Update embeddings for next time step & statistics
            h_last = h_cur
            preds.append(w_hat)
            if do_encode:
                mus_prio.append(mu_prio)
                mus_post.append(mu_post)
                logvars_prio.append(logvar_prio)
                logvars_post.append(logvar_post)

        if do_encode:
            mus_prio = torch.stack(mus_prio, dim=1) # (B, pred_step, latent_size)
            mus_post = torch.stack(mus_post, dim=1) # (B, pred_step, latent_size)
            logvars_prio = torch.stack(logvars_prio, dim=1) # (B, pred_step, latent_size)
            logvars_post = torch.stack(logvars_post, dim=1) # (B, pred_step, latent_size)
        if get_pred_sim:
            pred_sims = torch.stack(pred_sims, dim=1) # (B, pred_step, 2)

        if self.action_cls_head == 'future':
            # Supervised operation (action anticipation)
            # Classify c_t+3 into an action via one or multiple output vectors
            action_outputs = self.perform_action_cls(h_last)
            if do_encode:
                if get_pred_sim:
                    result = (action_outputs, h_last, mus_prio, mus_post, logvars_prio, logvars_post, pred_sims)
                else:
                    result = (action_outputs, h_last, mus_prio, mus_post, logvars_prio, logvars_post)
            else:
                if get_pred_sim:
                    result = (action_outputs, h_last, pred_sims)
                else:
                    result = (action_outputs, h_last)
            return result

        else:
            # Self-supervised operation
            assert(self.action_cls_head is None)
            preds = torch.stack(preds, dim=1) # (B, pred_step, F, S, S)
            N = self.pred_step

            # Reshape predicted and ground truth video block embeddings into desired format
            preds = preds.permute(0, 1, 3, 4, 2).contiguous().view(
                B * N * self.spatial_size ** 2, self.param['input_size'])
            truths = w_inf_future.permute(0, 1, 3, 4, 2).contiguous().view(
                B * N * self.spatial_size ** 2, self.param['input_size']).transpose(0, 1)

            # Calculate score matrix (dot products) & mask (only once)
            score = torch.matmul(preds, truths)
            self.compute_mask(B, N)

            # Return results
            if do_encode:
                if get_pred_sim:
                    result = (score, self.mask, mus_prio, mus_post, logvars_prio, logvars_post, pred_sims)
                else:
                    result = (score, self.mask, mus_prio, mus_post, logvars_prio, logvars_post)
            else:
                if get_pred_sim:
                    result = (score, self.mask, pred_sims)
                else:
                    result = (score, self.mask)
            return result


    def forward_diversity(self, root_h_last, paths, force_rnn_dropout, spatial_separation):
        '''
        Performs future prediction but with a specified number of independent paths, branching off from the present.
        Never uses the encoder because our goal is to cover the output probability space with sufficient samples.
        These samples are used to calculate variance metrics in both the prediction (= block respresentation)
        and hidden state (= context) space.
        force_rnn_dropout: Simulate dropout on c_t (p = 0.1) independently for each path.
        spatial_separation: Whether to avoid averaging over H, W when calculating mean distance metrics.
        '''
        B = root_h_last.shape[0]
        if force_rnn_dropout:
            self.sim_div_dropout.train() # always perform dropout if desired

        all_preds = torch.zeros(B, paths, self.pred_step, self.input_size, self.spatial_size, self.spatial_size).cuda()
        # (batch index, path, future step, F, S, S)
        all_contexts = torch.zeros(B, paths, self.pred_step, self.state_size, self.spatial_size, self.spatial_size).cuda()
        # (batch index, path, future step, F, S, S)
        
        populate_actions = (self.action_cls_head == 'present' or self.action_cls_head == 'future')
        if populate_actions:
            class_types = len(self.num_class)
            action_outputs = []
            # (class type within list, batch index, path, future step, class) => logit
            for k in range(class_types):
                action_outputs.append(torch.zeros(B, paths, self.pred_step, self.num_class[k]).cuda())

        # Start loop over future time steps
        for t in range(self.pred_step):
            if t == 0 and not(force_rnn_dropout):
                mu, logvar = self.vrnn_backbone.prior(root_h_last)

            # Process every branch independently
            for j in range(paths):
                # Select c_t
                if force_rnn_dropout:
                    first_h_last = self.sim_div_dropout(root_h_last)
                    if t == 0:
                        mu, logvar = self.vrnn_backbone.prior(first_h_last)
                else:
                    first_h_last = root_h_last

                # Run prior + generation + recurrence
                h_last = all_contexts[:, j, t-1] if t > 0 else first_h_last
                if t > 0:
                    mu, logvar = self.vrnn_backbone.prior(h_last)
                z_cur = sample_latent(mu, logvar)
                w_hat = self.vrnn_backbone.generation(h_last, z_cur)
                h_cur = self.vrnn_backbone.recurrence(w_hat, h_last, z_cur)

                # Store embeddings
                all_preds[:, j, t] = w_hat
                all_contexts[:, j, t] = h_cur

                # Store action output (all vector values) if possible
                if populate_actions:
                    cur_act_out = self.perform_action_cls(h_last) # (class type within list, batch index, 1, class)
                    for k in range(class_types):
                        action_outputs[k][:, j, t] = cur_act_out[k].view(B, -1)

        # Include distances to quantify both instantaneous and aggregated embedding uncertainty
        if spatial_separation:
            # (batch index, metric kind, future step, S, S)
            mean_variance = torch.zeros(B, 6, self.pred_step, self.spatial_size, self.spatial_size).cuda()
            mean_variance[:, :3] = calculate_variance_tensor(all_preds, spatial_separation=True)
            mean_variance[:, 3:] = calculate_variance_tensor(all_contexts, spatial_separation=True)
        else:
            # (batch index, metric kind, future step)
            mean_variance = torch.zeros(B, 12, self.pred_step).cuda()
            mean_variance[:, :6] = calculate_variance_tensor(all_preds, spatial_separation=False)
            mean_variance[:, 6:] = calculate_variance_tensor(all_contexts, spatial_separation=False)
        
        results = {'mean_var': mean_variance, 'preds': all_preds, 'contexts': all_contexts}
        if populate_actions:
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


    def compute_mask(self, B, N):
        if self.mask is None: # only compute mask once
            # mask meaning:
            # -2: omit,
            # -1: temporal neg (hard),
            # 0: easy neg,
            # 1: pos,
            # -3: spatial neg
            # easy negatives (do not take gradient here)
            mask = torch.zeros((B, self.pred_step, self.spatial_size**2, B, N, self.spatial_size**2),
                            dtype=torch.int8, requires_grad=False).detach().cuda()
            # spatial negative (mark everything in the same batch as spatial negative)
            mask[torch.arange(B), :, :, torch.arange(B),
                :, :] = -3  # spatial neg
            # temporal negetive
            for k in range(B):
                mask[k, :, torch.arange(
                    self.spatial_size**2), k, :, torch.arange(self.spatial_size**2)] = -1  # temporal neg
            # positive
            tmp = mask.permute(0, 2, 1, 3, 5, 4).contiguous().view(
                B * self.spatial_size**2, self.pred_step, B * self.spatial_size**2, N)
            for j in range(B * self.spatial_size**2):
                tmp[j, torch.arange(self.pred_step), j, torch.arange(
                    N - self.pred_step, N)] = 1  # pos
            mask = tmp.view(B, self.spatial_size**2, self.pred_step,
                            B, self.spatial_size**2, N).permute(0, 2, 1, 3, 5, 4)
            self.mask = mask
        return self.mask
