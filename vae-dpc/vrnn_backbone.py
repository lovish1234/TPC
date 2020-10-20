# BVH, May 2020
# Components of VRNN

import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../backbone')
from convrnn import *


class My_VRNN_Conv_GRU(nn.Module):
    '''
    Variational Recurrent Neural Network (VRNN) with Convolutional Gated Recurrent Unit (ConvGRU).
    1) Aggregates a present embedding with a previous context to a new context.
    2) Maps a present context to a predicted plausible future embedding.
    Code inspired by:
    Paper: Chung, J. et al. (2015). A recurrent latent variable model for sequential data.
    https://github.com/emited/VariationalRecurrentNeuralNetwork/blob/master/model.py
    https://github.com/yiweilu3/CONV-VRNN-for-Anomaly-Detection/blob/master/VRNN.py
    NOTE: assumes num_layers == 1 since just one GRU cell is used everywhere in DPC.
    '''

    def __init__(self, input_size, state_size, latent_size, spatial_size, kernel_size,
                 dropout=0.1, time_indep=False):
        super(My_VRNN_Conv_GRU, self).__init__()
        self.input_size = input_size  # also acts as output_size
        self.state_size = state_size
        self.latent_size = latent_size
        self.spatial_size = spatial_size
        self.kernel_size = kernel_size  # for all conv layers including GRU
        self.hidden_size = min(input_size, state_size)
        self.time_indep = time_indep
        padding = kernel_size // 2  # as in GRU

        # Prior network
        self.prior_conv = nn.Sequential(
            nn.Conv2d(self.state_size, self.hidden_size,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        self.prior_fc_mu = nn.Linear(
            self.hidden_size * self.spatial_size ** 2, self.latent_size)
        self.prior_fc_logvar = nn.Linear(
            self.hidden_size * self.spatial_size ** 2, self.latent_size)

        # Encoder network (inference of approximate posterior)
        self.enc_conv = nn.Sequential(
            nn.Conv2d(self.input_size + self.state_size, self.hidden_size,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        self.enc_fc_mu = nn.Linear(
            self.hidden_size * self.spatial_size ** 2, self.latent_size)
        self.enc_fc_logvar = nn.Linear(
            self.hidden_size * self.spatial_size ** 2, self.latent_size)

        # Decoder network (generation)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(self.state_size + self.latent_size, self.hidden_size,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        self.dec_conv2 = nn.Conv2d(
            self.hidden_size, self.input_size, kernel_size=kernel_size, padding=padding)

        # Recurrence network
        if dropout > 0.0:
            self.rnn = ConvGRUCellDropout(
                self.input_size + self.latent_size, self.state_size, self.kernel_size, dropout=dropout)
        else:
            self.rnn = ConvGRUCell(
                self.input_size + self.latent_size, self.state_size, self.kernel_size)

    def prior_first(self, B):
        # No hidden state yet; sometimes we are just starting out
        mu_prio = torch.zeros(B, self.latent_size).cuda()
        logvar_prio = torch.zeros(B, self.latent_size).cuda()
        return mu_prio, logvar_prio

    def prior(self, h_last):
        B = h_last.shape[0]
        if self.time_indep:
            return self.prior_first(B)
        assert(h_last.shape == (B, self.state_size,
                                self.spatial_size, self.spatial_size))

        # DEBUG:
        # return self.prior_first(B)

        tmp = self.prior_conv(h_last)
        tmp = tmp.view(B, -1)
        mu_prio = self.prior_fc_mu(tmp)
        logvar_prio = self.prior_fc_logvar(tmp)
        return mu_prio, logvar_prio

    def generation(self, h_last, z_cur):
        B = h_last.shape[0]
        assert(h_last.shape == (B, self.state_size,
                                self.spatial_size, self.spatial_size))
        assert(z_cur.shape == (B, self.latent_size))

        z_cur = z_cur.repeat(self.spatial_size, self.spatial_size, 1, 1).permute(
            2, 3, 0, 1)  # broadcast across space
        # (B, state_size + latent_size, spatial_size, spatial_size)
        tmp = torch.cat([h_last, z_cur], dim=1)
        tmp = self.dec_conv1(tmp)
        x_cur = self.dec_conv2(tmp)
        return x_cur

    def recurrence(self, x_cur, h_last, z_cur, no_dropout=False):
        B = x_cur.shape[0]
        assert(x_cur.shape == (B, self.input_size,
                               self.spatial_size, self.spatial_size))
        if h_last is not None:  # sometimes we are just starting out
            assert(h_last.shape == (B, self.state_size,
                                    self.spatial_size, self.spatial_size))
        # if not(self.time_indep): # sometimes we do not let h depend on z (NOTE: this part is untrue, see paper)
        assert(z_cur.shape == (B, self.latent_size))

        z_cur = z_cur.repeat(self.spatial_size, self.spatial_size, 1, 1).permute(
            2, 3, 0, 1)  # broadcast across space
        # (B, input_size + latent_size, spatial_size, spatial_size)
        tmp = torch.cat([x_cur, z_cur], dim=1)
        # else:
        #     tmp = x_cur

        h_cur = self.rnn(tmp, h_last, no_dropout=no_dropout)
        return h_cur

    def inference(self, x_cur, h_last):
        B = x_cur.shape[0]
        assert(x_cur.shape == (B, self.input_size,
                               self.spatial_size, self.spatial_size))
        assert(h_last.shape == (B, self.state_size,
                                self.spatial_size, self.spatial_size))
#         assert(mu_prio.shape == (B, self.latent_size))
#         assert(logvar_prio.shape == (B, self.latent_size))

        # TODO: how can this method know the prior parameters to stay close?

        # (B, input_size + state_size, spatial_size, spatial_size)
        tmp = torch.cat([x_cur, h_last], dim=1)
        tmp = self.enc_conv(tmp)
        tmp = tmp.view(B, -1)
        mu_post = self.enc_fc_mu(tmp)
        logvar_post = self.enc_fc_logvar(tmp)
        return mu_post, logvar_post


class ConvGRUCellDropout(nn.Module):
    ''' Adapted from DPC convrnn: added dropout to cell itself. '''

    def __init__(self, input_size, hidden_size, kernel_size, dropout=0.1):
        super(ConvGRUCellDropout, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        self.reset_gate = nn.Conv2d(
            input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(
            input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(
            input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.dropout_layer = nn.Dropout(p=dropout)

        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)
        nn.init.constant_(self.reset_gate.bias, 0.)
        nn.init.constant_(self.update_gate.bias, 0.)
        nn.init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_tensor, hidden_state, no_dropout=False):
        if hidden_state is None:
            B, C, *spatial_dim = input_tensor.size()
            hidden_state = torch.zeros(
                [B, self.hidden_size, *spatial_dim]).cuda()
        # [B, C, H, W]
        combined = torch.cat([input_tensor, hidden_state],
                             dim=1)  # concat in C
        update = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.reset_gate(combined))
        out = torch.tanh(self.out_gate(
            torch.cat([input_tensor, hidden_state * reset], dim=1)))
        new_state = hidden_state * (1 - update) + out * update
        # perform dropout at every time step
        if not(no_dropout):
            new_state = self.dropout_layer(new_state)
        return new_state
