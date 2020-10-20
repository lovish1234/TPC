import os
import argparse
from tqdm import tqdm


import torch
import torch.nn as nn


parser = argparse.ArgumentParser()

# directory of the embeddings
parser.add_argument('--embed-dir', default='', type=str)

# distance (score function)
parser.add_argument('--distance', default='cosine',
                    type=str, help='can be cosine or L2')

# distance (score function)
parser.add_argument('--epoch', default=100,
                    type=int, help='which iteration of model to visualize')

parser.add_argument('--gpu', default='6,7', type=str)


def main():
    global args
    args = parser.parse_args()
    path = os.path.join(args.embed_dir, "%0.5d" % args.epoch)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    global cuda
    cuda = torch.device('cuda')

    output_embed = torch.load(os.path.join(path, 'output_embed.pt')).to(cuda)
    # target_embed = torch.load(os.path.join(path, 'target_embed.pt')).to(cuda)

    if args.distance == 'cosine':
        distance_f = nn.CosineSimilarity(dim=0, eps=1e-6)

    adj = torch.zeros(output_embed.shape[0], output_embed.shape[0])
    for i in tqdm(range(output_embed.shape[0])):
        for j in range(output_embed.shape[0]):
            a = output_embed[i].reshape(-1)
            b = output_embed[j].reshape(-1)
            if args.distance == 'cosine':
                adj[i, j] = distance_f(a, b)
            elif args.distance == 'L2':
                adj[i, j] = torch.dist(a, b, p=2)
    if args.distance == 'cosine':
        torch.save(adj, os.path.join(path, 'cosine_distance.pt'))
    elif args.distance == 'L2':
        torch.save(adj, os.path.join(path, 'L2_distance.pt'))


if __name__ == '__main__':
    main()
