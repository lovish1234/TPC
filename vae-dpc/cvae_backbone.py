# BVH, May 2020
# Components of CVAE

import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../backbone')


class My_CVAE_FC(nn.Module):
    '''
    Fully Connected Conditional Variational Autoencoder that maps an aggregated present
    representation (c_t) to a plausible future representation (z_t+1 a.k.a. w_t+1).
    Code inspired by https://github.com/jojonki/AutoEncoders/blob/master/cvae.ipynb
    '''

    def __init__(self, input_size, output_size, latent_size, hidden_size):
        super(My_CVAE_FC, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size

        # Encoder
        self.enc_fc1 = nn.Sequential(
            nn.Linear(output_size + input_size, hidden_size),
            nn.ReLU(inplace=False)
        )
        self.enc_fc2_mu = nn.Linear(hidden_size, latent_size)
        self.enc_fc2_logvar = nn.Linear(hidden_size, latent_size)

        # Decoder
        self.dec_fc1 = nn.Sequential(
            nn.Linear(latent_size + input_size, hidden_size),
            nn.ReLU(inplace=False)
        )
        self.dec_fc2 = nn.Linear(hidden_size, output_size)

    def encode(self, y, x):
        '''
        Implements Q(z|Y, X), concretely gaussian_params(z) = (mu(Y, X), logvar(Y, X)).
        y: (batch_size, output_size), x: (batch_size, input_size).
        '''
        y = y.view(-1, self.output_size)
        x = x.view(-1, self.input_size)
        # (batch_size, output_size + input_size)
        inputs = torch.cat([y, x], dim=1)
        h = self.enc_fc1(inputs)
        z_mu = self.enc_fc2_mu(h)
        z_logvar = self.enc_fc2_logvar(h)
        return (z_mu, z_logvar)

    def decode(self, z, x):
        '''
        Implements P(Y|z, X), concretely Y = f(z, X).
        z: (batch_size, latent_size), x: (batch_size, input_size).
        '''
        z = z.view(-1, self.latent_size)
        x = x.view(-1, self.input_size)
        # (batch_size, latent_size + input_size)
        inputs = torch.cat([z, x], dim=1)
        h = self.dec_fc1(inputs)
        y = self.dec_fc2(h)
        return y

    def forward_train(self, y, x):
        y = y.view(-1, self.output_size)
        x = x.view(-1, self.input_size)
        mu, logvar = self.encode(y, x)
        # standardnormal Gaussian distribution
        eps = torch.empty_like(logvar).normal_()
        eps = eps.cuda()
        sigma = logvar.mul(0.5).exp_()
        z = eps.mul(sigma) + mu
        return self.decode(z, x), mu, logvar

    def forward_test(self, x):
        x = x.view(-1, self.input_size)
        # standardnormal Gaussian distribution
        z = torch.empty(x.shape[0], self.latent_size).normal_()
        z = z.cuda()
        return self.decode(z, x)


class My_CVAE_Conv1x1(nn.Module):
    '''
    1x1 Convolutional Conditional Variational Autoencoder that maps an aggregated present
    representation (c_t) to a plausible future representation (z_t+1 a.k.a. w_t+1).
    NOTE: all spatial locations act independently of each other, even the latent space and its distribution parameters.
    '''

    def __init__(self, input_size, output_size, latent_size, hidden_size):
        super(My_CVAE_Conv1x1, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size

        # Encoder
        self.enc_fc1 = nn.Sequential(
            nn.Conv2d(output_size + input_size, hidden_size,
                      kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.enc_fc2_mu = nn.Conv2d(
            hidden_size, latent_size, kernel_size=1, padding=0)
        self.enc_fc2_logvar = nn.Conv2d(
            hidden_size, latent_size, kernel_size=1, padding=0)

        # Decoder
        self.dec_fc1 = nn.Sequential(
            nn.Conv2d(latent_size + input_size, hidden_size,
                      kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.dec_fc2 = nn.Conv2d(
            hidden_size, output_size, kernel_size=1, padding=0)

    def encode(self, y, x):
        '''
        Implements Q(z|Y, X), concretely gaussian_params(z) = (mu(Y, X), logvar(Y, X)).
        y: (batch_size, output_size, height, width), x: (batch_size, input_size, height, width).
        '''
        inputs = torch.cat(
            [y, x], dim=1)  # (batch_size, output_size + input_size, height, width)
        h = self.enc_fc1(inputs)
        z_mu = self.enc_fc2_mu(h)
        z_logvar = self.enc_fc2_logvar(h)
        return (z_mu, z_logvar)

    def decode(self, z, x):
        '''
        Implements P(Y|z, X), concretely Y = f(z, X).
        z: (batch_size, latent_size, height, width), x: (batch_size, input_size, height, width).
        '''
        inputs = torch.cat(
            [z, x], dim=1)  # (batch_size, latent_size + input_size, height, width)
        h = self.dec_fc1(inputs)
        y = self.dec_fc2(h)
        return y

    def forward_train(self, y, x):
        mu, logvar = self.encode(y, x)
        # standardnormal Gaussian distribution
        eps = torch.empty_like(logvar).normal_()
        eps = eps.cuda()
        sigma = logvar.mul(0.5).exp_()
        z = eps.mul(sigma) + mu
        return self.decode(z, x), mu, logvar

    def forward_test(self, x):
        # standardnormal Gaussian distribution
        z = torch.empty(x.shape[0], self.latent_size,
                        x.shape[2], x.shape[3]).normal_()
        z = z.cuda()
        return self.decode(z, x)


class My_CVAE_ConvFC(nn.Module):
    '''
    Convolutional Conditional Variational Autoencoder that maps an aggregated present
    representation (c_t) to a plausible future representation (z_t+1 a.k.a. w_t+1).
    The latent space exists in-between two pseudo fully connected layers such that
    encoding and decoding can share information across all spatial blocks.
    '''

    def __init__(self, input_size, output_size, latent_size, hidden_size, spatial_size):
        super(My_CVAE_ConvFC, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.spatial_size = spatial_size

        # Encoder
        self.enc_fc1 = nn.Sequential(
            nn.Conv2d(output_size + input_size, hidden_size,
                      kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.enc_fc2_mu = nn.Linear(
            hidden_size * spatial_size ** 2, latent_size)
        self.enc_fc2_logvar = nn.Linear(
            hidden_size * spatial_size ** 2, latent_size)

        # Decoder
        self.dec_fc1 = nn.Sequential(
            nn.Conv2d(latent_size + input_size, hidden_size,
                      kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.dec_fc2 = nn.Conv2d(
            hidden_size, output_size, kernel_size=1, padding=0)

    def encode(self, y, x):
        '''
        Implements Q(z|Y, X), concretely gaussian_params(z) = (mu(Y, X), logvar(Y, X)).
        y: (batch_size, output_size, height, width), x: (batch_size, input_size, height, width).
        '''
        inputs = torch.cat(
            [y, x], dim=1)  # (batch_size, output_size + input_size, height, width)
        h = self.enc_fc1(inputs)
        # flatten before linear
        h = h.view(-1, self.hidden_size * self.spatial_size ** 2)
        z_mu = self.enc_fc2_mu(h)
        z_logvar = self.enc_fc2_logvar(h)
        return (z_mu, z_logvar)

    def decode(self, z, x):
        '''
        Implements P(Y|z, X), concretely Y = f(z, X).
        z: (batch_size, latent_size), x: (batch_size, input_size, height, width).
        '''
        z = z.repeat(self.spatial_size, self.spatial_size, 1, 1).permute(
            2, 3, 0, 1)  # broadcast across space
        # (batch_size, latent_size + input_size, height, width)
        inputs = torch.cat([z, x], dim=1)
        h = self.dec_fc1(inputs)
        y = self.dec_fc2(h)
        return y

    def forward_train(self, y, x):
        mu, logvar = self.encode(y, x)
        # standardnormal Gaussian distribution
        eps = torch.empty_like(logvar).normal_()
        eps = eps.cuda()
        sigma = logvar.mul(0.5).exp_()
        z = eps.mul(sigma) + mu
        return self.decode(z, x), mu, logvar

    def forward_test(self, x):
        # standardnormal Gaussian distribution
        z = torch.empty(x.shape[0], self.latent_size).normal_()
        z = z.cuda()
        return self.decode(z, x)
