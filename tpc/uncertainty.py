import torch
import math
import torch.nn as nn


def L2_MSE(score, pred_radius, margin, distance_type='uncertain', weighting=True, loss_type='MSE'):
    '''
    Given L2 similarity matrix 'score', and predicted radius vector
    'pred_radius', calculate final score and apply MSE loss function
    '''
    MSE = nn.MSELoss()
    # mask for appying distance function
    mask_score = torch.ones_like(score)
    torch.diagonal(mask_score).fill_(-1)
    mask_score = -mask_score

    # apply uncertainty
    final_score = score
    final_score = mask_score * final_score
    final_radius = pred_radius

    # certain case
    if pred_radius is not None:
        final_radius = final_radius.expand(score.shape).T.cuda()
        final_radius = (-mask_score) * final_radius

    # add margin and radius
    margin_matrix = torch.ones_like(final_score) * margin
    torch.diagonal(margin_matrix).fill_(0)
    if distance_type == 'uncertain':
        final_score = final_score + final_radius + margin_matrix
    elif distance_type == 'certain':
        final_score = final_score + margin_matrix

    # replace negative value with zeros
    zero_tensor = torch.zeros_like(final_score)
    final_score = torch.max(torch.stack(
        [final_score, zero_tensor]), axis=0).values

    # apply weight
    if weighting:
        mask_weight = torch.ones_like(score)
        if loss_type == 'MSE':
            torch.diagonal(mask_weight).fill_(math.sqrt(len(score)))
        elif loss_type == 'L1':
            torch.diagonal(mask_weight).fill_(len(score))
        final_score = final_score * mask_weight

    if distance_type == 'uncertain':
        return [final_score, pred_radius, score, final_radius]

    return [final_score, pred_radius, score, None]


def cosine_MSE(score, pred_radius, margin, distance_type='uncertain', weighting=True):
    '''
    Given cosine similarity matrix 'score', and predicted radius vector
    'pred_radius', calculate final score and apply MSE loss function
    '''
    MSE = nn.MSELoss()
    # mask for applying distance function
    mask_score = torch.ones_like(score)
    torch.diagonal(mask_score).fill_(-1)
    mask_score = -mask_score

    # margin for negative samples
    margin = torch.ones_like(score) * margin
    torch.diagonal(margin).fill_(0)

    # two tensor
    two_tensor = torch.ones_like(score) * 2
    two_tensor = two_tensor * mask_score

    # add all together
    final_score = score + 1
    final_score = (-mask_score) * final_score
    final_radius = pred_radius

    # certain case
    if pred_radius is not None:
        final_radius = torch.div(
            2 * torch.exp(final_radius), (1 + torch.exp(final_radius)))
        final_radius = final_radius.expand(score.shape).T.cuda()
        final_radius = (-mask_score) * final_radius

    if distance_type == 'uncertain':
        final_score = final_score + final_radius + margin + two_tensor
    elif distance_type == 'certain':
        final_score = final_score + margin + two_tensor

    # replace negative value with zeros
    zero_tensor = torch.zeros_like(final_score)
    final_score = torch.max(torch.stack(
        [final_score, zero_tensor]), axis=0).values

    # apply weight
    if weighting:
        mask_weight = torch.ones_like(score)
        torch.diagonal(mask_weight).fill_(math.sqrt(len(score)))
        final_score = final_score * mask_weight

    if distance_type == 'uncertain':
        return [final_score, pred_radius, score, final_radius]

    return [final_score, pred_radius, score, None]


def CE(score, pred_radius, distance_type='uncertain'):
    '''
    Given similarity score and predicted radius, calculate
    loss with NLL(log(softmax))
    '''
    log = torch.log
    softmax = torch.nn.Softmax(dim=1)
    NLL = torch.nn.NLLLoss()
    CE = torch.nn.CrossEntropyLoss()

    # apply softmax to get probability
    probs = softmax(score)

    # process radius:
    # subtract from negative sample,
    # plus for positive sample
    final_radius = pred_radius.expand(score.shape).T.cuda()
    mask = -torch.ones_like(pred_radius)
    torch.diagonal(mask).fill_(1)
    final_radius = final_radius * mask

    # add radius to probability and thresholding
    if distance_type == 'uncertain':
        probs = probs + pred_radius
        probs = torch.min(probs, torch.ones_like(probs))
        probs = torch.max(probs, torch.zeros_like(probs))

    # target for NLLLoss (1, 2, ...., len(score))
    target = torch.arange(0, score.shape[0]).cuda()

    # calculate loss
    loss = NLL(log(probs), target)
    return [score, pred_radius, score, final_radius, loss]

def process_uncertainty(score, pred_radius, weighting=True, distance='L2',
                        margin=10, distance_type='uncertain', loss_type='CE'):

    # float64 required
    score = score.double()
    if loss_type in ['MSE', 'L1']:
        if distance == 'L2':
            [final_score, pred_radius, score, final_radius] = L2_MSE(score, pred_radius,
                                                                     margin, distance_type=distance_type,
                                                                     weighting=weighting,
                                                                     loss_type=loss_type)
        if distance == 'cosine':
            [final_score, pred_radius, score, final_radius] = cosine_MSE(score, pred_radius,
                                                                         margin, distance_type=distance_type,
                                                                         weighting=weighting)
    elif loss_type == 'CE':

        '''
        L2: larger distance -> more different
        cosine: closer to 1 -> more similar
        dot: larger value   -> more similar
        Therefore invert L2 similarity matrix
        '''
        if distance == 'L2':
            score = -score
        [final_score, pred_radius, score, final_radius] = CE(
            score, pred_radius, distance_type=distance_type)

    return [final_score, pred_radius, score, final_radius]
