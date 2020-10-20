import torch
from tqdm import tqdm
import os
from torchvision import datasets, models, transforms

# data augmentation methods
from augmentation import *
from dataset_epic import *
from dataset_synthepic import *


def present_matching(args, model, pm_cache, epoch, mode):
    '''
    perform present matching (hard negative mining) once across training
    or validation data.
    
    args     : variable passed from main program
    model    : currently training model used for retrieving nearest neighbors
    pm_cache : directory to store present matching results
    epoch    : number of epoch
    mode     : train or validation
    dataset  : dataset class (epic_unlabeled, epic_sythepic etc.)
    '''

    # create cache directory
    pm_path = os.path.join(pm_cache, 'epoch_%s_%d' % (mode, epoch))
    if not os.path.exists(pm_path):
        os.makedirs(pm_path)

    print('GPU cache memory cleared')
    torch.cuda.empty_cache()  # clear GPU cache memory

    transform = transforms.Compose([
            RandomHorizontalFlip(consistent=True),
            # RandomSizedCrop(size=224, consistent=True, p=1.0),
            RandomCrop(size=224, consistent=True),
            Scale(size=(128, 128)),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5,
                        saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])
    if args.dataset == 'epic_unlabeled':
        dataset = epic_unlabeled(mode=mode, transform=transform, seq_len=args.seq_len, num_seq=args.num_seq - args.pred_step,
                             downsample=args.ds, num_clips=args.pm_num_samples, train_val_split=0.2, drive=args.drive,
                             retrieve_actions=False, present_matching=True, pred_step=args.pred_step)
    elif 'synthepic' in args.dataset:
        dataset = synthepic_action_pair(mode=mode,
                                        transform=transform,
                                        seq_len=args.seq_len,
                                        num_seq=args.num_seq,
                                        pred_step=3,
                                        downsample=args.ds,
                                        drive=args.drive,
                                        sample_method='match_cut',
                                        exact_cuts=True)
    else:
        dataset = None
        sys.exit('\n dataset type not supported by present matching training\n')

    
    
    if 'synthepic' in args.dataset: # last epoch size is not necessarily compatible with dataparallel
        data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  sampler=data.RandomSampler(dataset),
                                  shuffle=False,
                                  num_workers=32,
                                  pin_memory=False,
                                  drop_last=True)
    else:
        data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  sampler=data.RandomSampler(dataset),
                                  shuffle=False,
                                  num_workers=32,
                                  pin_memory=False,
                                  drop_last=False)

    print('Generating embeddings for present matching')
    context_all = torch.FloatTensor()  # hold all embeddings
    vpath_list = []  # hold video path of dataset
    idx_block_list = []  # hold frame index of dataset
    tic = time.time()

    model.eval()
    with torch.no_grad():
        for idx, input_dict in tqdm(enumerate(data_loader), total=len(data_loader)):

            # save video path of each sequences in dataset
            vpath_list.extend(input_dict['vpath'])
            # save frame number of each sequences in dataset
            idx_block_list.extend(input_dict['idx_block'])

            input_seq = input_dict['t_seq']
            input_seq = input_seq.to(args.cuda)
            output = model(input_seq, return_c_t=True)
            context_all = torch.cat(
                [context_all, output.detach().cpu()], dim=0)

    print(time.time() - tic)
    print('GPU cache memory cleared')
    torch.cuda.empty_cache()  # clear GPU cache memory

    # get nearest neighbors and save adjacency matrix
    if context_all.shape[0] < 10000 or 'synthepic' in args.dataset: # calculate adjacency matrix by smaller trunk
        score_all = calc_embed_dist_large(context_all.to(
            args.cuda), piece=1000, distance='cosine')
    else:
        score_all = calc_embed_dist_large(context_all.to(
            args.cuda), piece=10000, distance='cosine')
    nearest_K = torch.topk(score_all, 100, dim=1)

    del score_all, context_all

    dist_adj = nearest_K[1]

    # filtering out sequences that are too close to each other
    dist_adj = filter_close_sequence(
        dist_adj, idx_block_list, vpath_list, args.pm_num_ngbrs, args.pm_gap_thres)

    torch.save(dist_adj, os.path.join(pm_path, 'argmax_100.pt'))
    torch.save(nearest_K[1], os.path.join(pm_path, 'argmax_100_unfiltered.pt'))
    torch.save(nearest_K[0], os.path.join(pm_path, 'max_100_unfiltered.pt'))
    del nearest_K, dist_adj

    # save video information (video path and idx_block)
    with open(os.path.join(pm_path, "vpath.p"), "wb") as fp:  # Pickling
        pickle.dump(vpath_list, fp)
    with open(os.path.join(pm_path, "idx_block.p"), "wb") as fp:  # Pickling
        pickle.dump(idx_block_list, fp)
    del vpath_list, idx_block_list

    print('GPU cache memory cleared')
    torch.cuda.empty_cache()  # clear GPU cache memory


def filter_close_sequence(dist_adj, idx_block_list, vpath_list, num_ngbrs, gap_thres):
    '''Filter out rows in adjacency matrix where minimum pair-wise distance is
    smaller than a threshold (default 30 frames)
    '''
    print('\n[present_matching.py] Filtering sequences too close to each other (threshold: %d frames)\n' % gap_thres)

    keep_idx = []
    for i in tqdm(range(len(dist_adj))):
        if check_inner_distance(dist_adj, idx_block_list, vpath_list, i, num_ngbrs, gap_thres):
            keep_idx.append(i)

    dist_adj = dist_adj[keep_idx, :]

    print('\n[present_matching.py] Number of sequences left: %d\n' %
          len(keep_idx))
    return dist_adj


def check_inner_distance(dist_adj, idx_block_list, vpath_list, index, num_ngbrs, gap_thres):
    '''Check if minimum pair-wise distance is smaller than a threshold (30 frames)
    dist_adj       : adjacency matrix
    idx_block_list : list of index block
    vpath_list     : list of video path
    index          : index of row in adjacency matrix
    num_ngbrs      : number of neighbors needed for training
    gap_thres      : threshold of nearest distance
    
    
    TODO: debug vpath_list
    '''
    for i in range(num_ngbrs):
        for j in range(num_ngbrs):
            if i != j:
                seq_i = dist_adj[index][i]
                seq_j = dist_adj[index][j]
                if vpath_list[seq_i] == vpath_list[seq_j]: # only filter out sequences from the same video
                    # two sequences too close to each other
                    if idx_block_list[seq_i][0].item() in\
                            range(idx_block_list[seq_j][0].item() - gap_thres, idx_block_list[seq_j][0].item() + gap_thres):
                        return False
    return True


def calc_embed_dist_large(context_all, piece=10000, distance='cosine'):
    '''Calculate distance adjacency matrix between each row in 'context_all'

    context_all  :  [N, dim]  dim -> dimension of embeddings
    piece        :  10000 how large a piece to process
    distance     :  ['cosine', 'L2' (not recommend), 'dot']
    '''
    print('[present_matching.py] calculating %s adjacency matrix of input embeddings of size [%d, %d]' % (
        distance, context_all.shape[0], context_all.shape[1]))
    # N: number of embedding vector; dim: dimention of embeddings
    [N, dim] = context_all.shape
    if N % piece != 0:
        print('[present_matching.py] WARNING: piece not divisible by number of embeddings')
#         return None
    score_all = torch.FloatTensor(piece * (N // piece), piece * (N // piece))
    for i in tqdm(range(int(N / piece))):
        left = context_all[i * piece:(i + 1) * piece]
        for j in range(int(N / piece)):
            right = context_all[j * piece:(j + 1) * piece]
            score_all[i * piece:(i + 1) * piece, j * piece:(j + 1)
                      * piece] = calc_embed_dist(left, right, distance)
    return score_all


def calc_embed_dist(pred_embedding, gt_embedding, distance):
    '''Calculate a distance adjcency matrix between two sets of embeddings

    pred_embedding : size [N , dim]  dim -> dimension of embeddings
    gt_embedding   : size [M , dim]
    distance       : ['dot', 'cosine', 'L2'(not recommend for large input)] TODO: implement more efficient L2 distance calculation
    return a matrix of size [N, M]

    Both pred_embedding, gt_embedding should be cuda tensor, or else it's
    extremely slow.

    NOTE: only support 1D embeddings, for 3D embeddings, needs to .view(-1)
    before passing in'''
    if not pred_embedding.is_cuda and gt_embedding.is_cuda:
        print('WARNING: input tensors are not on cuda device')
    if distance == 'dot':
        gt_embedding = gt_embedding.transpose(0, 1)
        score = torch.matmul(pred_embedding, gt_embedding)

    elif distance == 'cosine':
        pred_norm = torch.norm(pred_embedding, dim=1)
        gt_norm = torch.norm(gt_embedding, dim=1)

        gt_embedding = gt_embedding.transpose(0, 1)
        score = torch.matmul(pred_embedding, gt_embedding)

        # row-wise division
        score = torch.div(score, pred_norm.expand(1, -1).T)
        # column-wise division
        score = torch.div(score, gt_norm)

        del pred_embedding, gt_embedding

        # division by the magnitude of respective vectors
    elif distance == 'L2':
        print(pred_embedding.shape)
        pred_embedding_mult = pred_embedding.reshape(
            pred_embedding.shape[0], 1, pred_embedding.shape[1])
        difference = pred_embedding_mult - gt_embedding
        score = torch.sqrt(torch.einsum(
            'ijk,ijk->ij', difference, difference))
    return score
