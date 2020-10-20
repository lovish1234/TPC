import torch
import numpy as np

def poincare_distance(pred, gt):
    '''
    Calculate pair-wise poincare distance between each row in two input tensors
    
    See equation (1) in this paper for mathematical expression:
    https://arxiv.org/abs/1705.08039
    '''
    (N, D) = pred.shape
    pred = pred / torch.norm(pred).expand(D, N).T
    gt = gt / torch.norm(gt).expand(D, N).T
    a = (1 - square_norm(pred)).view(N, 1)
    b = (1 - square_norm(gt)).view(1, N)
    return arcosh(1 + 2 * pairwise_distances(pred, gt) / torch.matmul(a, b))

def arcosh(x):
    """
    arcosh(x) = log( x + sqrt[x^2 - 1] )

    """
    return torch.log(x+torch.sqrt(x*x - 1))
    
def square_norm(x):
    """
    Helper function returning square of the euclidean norm.

    Also here we clamp it since it really likes to die to zero.

    """
    norm = torch.norm(x,dim=-1,p=2)**2
    return torch.clamp(norm,min=1e-5)
    
def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)