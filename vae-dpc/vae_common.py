# BVH, May 2020
# Common code for VRNN / CVAE-RNN

import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_latent(mu, logvar):
    assert(mu.shape == logvar.shape)
    eps = torch.empty_like(logvar).normal_() # standardnormal Gaussian distribution
    eps = eps.cuda()
    sigma = logvar.mul(0.5).exp_()
    z = eps.mul(sigma) + mu
    return z


def calculate_diversity(embs_list):
    ''' NOTE: use the newer calculate_variance_tensor() instead.
    Calculates the mean distance between all pairs of given embeddings
    (independently for all elements within a mini-batch). Different metrics exist,
    so the returned tuple is:
    (Euclidean, pooled Euclidean, L2-norm Euclidean,
     pooled L2-norm Euclidean, cosine, pooled cosine)
    where 'pooled' averages over spatial dimensions first,
    and L2-norm normalizes vectors first. '''

    B = embs_list[0].shape[0]
    sum_dist = torch.zeros(B, 6)
    num_terms = 0

    for i in range(len(embs_list)):
        x = embs_list[i] # (4, 256, 4, 4)
        x_pool = x.mean(dim=2).mean(dim=2) # (4, 256)
        for j in range(i + 1, len(embs_list)):
            y = embs_list[j] # (4, 256, 4, 4)
            y_pool = y.mean(dim=2).mean(dim=2) # (4, 256)

            for k in range(B):
                # Calculate L2-normalized and pooled vectors
                x_norm = x[k] / torch.norm(x[k].flatten(), 2)
                y_norm = y[k] / torch.norm(y[k].flatten(), 2)
                x_norm_pool = x_pool[k] / torch.norm(x_pool[k].flatten(), 2)
                y_norm_pool = y_pool[k] / torch.norm(y_pool[k].flatten(), 2)

                # Euclidean distance between raw embeddings
                sum_dist[k, 0] += torch.norm((x[k] - y[k]).flatten(), 2)
                # Euclidean distance between spatially averaged embeddings
                sum_dist[k, 1] += torch.norm((x_pool[k] - y_pool[k]).flatten(), 2)
                # Euclidean distance between L2-normalized embeddings
                sum_dist[k, 2] += torch.norm((x_norm[k] - y_norm[k]).flatten(), 2)
                # Euclidean distance between L2-normalized spatially averaged embeddings
                sum_dist[k, 3] += torch.norm((x_norm_pool[k] - y_norm_pool[k]).flatten(), 2)
                # cosine distance between raw embeddings, mean afterwards
                sum_dist[k, 4] += F.cosine_similarity(x[k], y[k], dim=0).mean()
                # cosine distance between spatially averaged embeddings
                sum_dist[k, 5] += F.cosine_similarity(x_pool[k], y_pool[k], dim=0)

            num_terms += 1

    mean_dist = sum_dist / num_terms
    return mean_dist


def calculate_variance(embs_list):
    ''' NOTE: use the newer calculate_variance_tensor() instead.
    Calculates the mean distance of all embeddings relative to their average
    (independently for all elements within a mini-batch). Different metrics exist,
    so the returned tuple is:
    (L2, pooled L2, normalized L2, pooled normalized L2, cosine, pooled cosine)
    where 'pooled' averages over spatial dimensions first,
    and L2-norm normalizes vectors first. '''
    
    B = embs_list[0].shape[0]
    sum_dist = torch.zeros(B, 6)
    preds_all = torch.stack(embs_list) # (SPS, 4, 256, 4, 4)

    # Compute average embeddings
    avg = preds_all.mean(dim=0) # (4, 256, 4, 4)
    avg_pool = avg.mean(dim=2).mean(dim=2) # (4, 256)
    avg_norm = torch.zeros_like(avg) # (4, 256, 4, 4)
    avg_norm_pool = torch.zeros_like(avg_pool) # (4, 256)
    for k in range(B):
        avg_norm[k] = avg[k] / torch.norm(avg[k].flatten(), 2)
        avg_norm_pool[k] = avg_pool[k] / torch.norm(avg_pool[k].flatten(), 2)

    # Sum contributions of every prediction
    for i in range(len(embs_list)):
        x = embs_list[i] # (4, 256, 4, 4)
        x_pool = x.mean(dim=2).mean(dim=2) # (4, 256)

        for k in range(B):
            # Calculate L2-normalized and pooled vectors
            x_norm = x[k] / torch.norm(x[k].flatten(), 2)
            x_norm_pool = x_pool[k] / torch.norm(x_pool[k].flatten(), 2)

            # Euclidean distance between raw embeddings
            sum_dist[k, 0] += torch.norm((x[k] - avg[k]).flatten(), 2)
            # Euclidean distance between spatially averaged embeddings
            sum_dist[k, 1] += torch.norm((x_pool[k] - avg_pool[k]).flatten(), 2)
            # Euclidean distance between L2-normalized embeddings
            sum_dist[k, 2] += torch.norm((x_norm[k] - avg_norm[k]).flatten(), 2)
            # Euclidean distance between L2-normalized spatially averaged embeddings
            sum_dist[k, 3] += torch.norm((x_norm_pool[k] - avg_norm_pool[k]).flatten(), 2)
            # cosine distance between raw embeddings, mean afterwards
            sum_dist[k, 4] += F.cosine_similarity(x[k], avg[k], dim=0).mean()
            # cosine distance between spatially averaged embeddings
            sum_dist[k, 5] += F.cosine_similarity(x_pool[k], avg_pool[k], dim=0)

    mean_dist = sum_dist / len(embs_list)
    return mean_dist


def calculate_variance_tensor(embeddings, spatial_separation=False):
    '''
    Calculates the mean distance of all embeddings relative to their average
    over all paths, independently for every index within a mini-batch.
    embeddings: (B, paths, pred_step, F, S, S).
    spatial_separation: Whether to avoid averaging over H, W.
    
    If False, returns: (B, 6, pred_step) where the 6 metrics =
    (L2, pooled L2, normalized L2, pooled normalized L2, cosine, pooled cosine).
    NOTE: pooling is done before everything else, followed by L2 normalization.

    If True, returns: (B, 3, pred_step, S, S) where the 3 metrics =
    (L2, normalized L2, cosine).
    '''

    embeddings = embeddings.cpu()
    (B, paths, pred_step, _, H, W) = embeddings.shape
    
    # Construct auxiliary variables
    embs_pool = embeddings.mean(dim=[4, 5]) # (B, paths, pred_step, F)
    embs_norm = torch.zeros_like(embeddings) # (B, paths, pred_step, F, S, S)
    embs_norm_pool = torch.zeros_like(embs_pool) # (B, paths, pred_step, F)
    avg = embeddings.mean(dim=1) # (B, pred_step, F, S, S)
    avg_pool = avg.mean(dim=[3, 4]) # (B, pred_step, F)
    avg_norm = torch.zeros_like(avg) # (B, pred_step, F, S, S)
    avg_norm_pool = torch.zeros_like(avg_pool) # (B, pred_step, F)

    if spatial_separation:
        sum_metrics = torch.zeros(B, 3, pred_step, H, W) # (B, metric, pred_step, S, S)
    else:
        sum_metrics = torch.zeros(B, 6, pred_step) # (B, metric, pred_step)

        # Get normalized & pooled embeddings first
        for i in range(B):
            for t in range(pred_step):
                for j in range(paths):
                    embs_norm[i, j, t] = embeddings[i, j, t] / torch.norm(embeddings[i, j, t], 2)
                    embs_norm_pool[i, j, t] = embs_pool[i, j, t] / torch.norm(embs_pool[i, j, t], 2)
                avg_norm[i, t] = avg[i, t] / torch.norm(avg[i, t], 2)
                avg_norm_pool[i, t] = avg_pool[i, t] / torch.norm(avg_pool[i, t], 2)

    # Calculate variance metrics (distances from average over all paths)
    for i in range(B):
        for t in range(pred_step):
            for j in range(paths):

                if spatial_separation:
                    # Process every position independently; pooling is not relevant here
                    for y in range(H):
                        for x in range(W):
                            # L2-distance between raw spatio-temporal blocks
                            sum_metrics[i, 0, t, y, x] += torch.norm(embeddings[i, j, t, :, y, x] - avg[i, t, :, y, x], 2)
                            # L2-distance between L2-normalized spatio-temporal blocks
                            sum_metrics[i, 1, t, y, x] += torch.norm(embs_norm[i, j, t, :, y, x] - avg_norm[i, t, :, y, x], 2)
                            # Cosine similarity between raw spatio-temporal blocks
                            sum_metrics[i, 2, t, y, x] += F.cosine_similarity(embeddings[i, j, t, :, y, x], avg[i, t, :, y, x], dim=0)

                else:
                    # L2-distance between raw embeddings
                    sum_metrics[i, 0, t] += torch.norm(embeddings[i, j, t] - avg[i, t], 2)
                    # L2-distance between spatially averaged embeddings
                    sum_metrics[i, 1, t] += torch.norm(embs_pool[i, j, t] - avg_pool[i, t], 2)
                    # L2-distance between L2-normalized embeddings
                    sum_metrics[i, 2, t] += torch.norm(embs_norm[i, j, t] - avg_norm[i, t], 2)
                    # L2-distance between L2-normalized spatially averaged embeddings
                    sum_metrics[i, 3, t] += torch.norm(embs_norm_pool[i, j, t] - avg_norm_pool[i, t], 2)
                    # Cosine similarity between raw embeddings, mean afterwards
                    sum_metrics[i, 4, t] += F.cosine_similarity(embeddings[i, j, t], avg[i, t], dim=0).mean()
                    # Cosine similarity between spatially averaged embeddings
                    sum_metrics[i, 5, t] += F.cosine_similarity(embs_pool[i, j, t], avg_pool[i, t], dim=0)
                
    result = sum_metrics / paths # average over number of paths
    return result
