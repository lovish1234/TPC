import sys


sys.path.append('../utils' )

from augmentation import *
# from model_F import F

from model_visualize import *
from dataset_visualize import *

import os
import argparse
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
from torch.utils import data
import torch.nn as nn
import matplotlib.pyplot as plt





parser = argparse.ArgumentParser()

# trained folder details
# parser.add_argument('--prefix', default='tmp', type=str)
# parser.add_argument('--dataset', default='ucf101', type=str)

# model for which we need the embedding
parser.add_argument('--test-dir', default='', type=str)
parser.add_argument('--dataset', default='ucf11', type=str)
parser.add_argument('--gpu', default='0,1', type=str)


parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument('--net', default='resnet10', type=str)

# batchsize - depends on the number og GPUs
parser.add_argument('--batch_size', default=64, type=int)

# according to DPC paper
parser.add_argument('--num_seq', default=5, type=int)
parser.add_argument('--seq_len', default=3, type=int)
parser.add_argument('--ds', default=3, type=int,
                    help='frame downsampling rate')

# embedding for certain vs uncertain
parser.add_argument('--distance-type', default='certain',
                    type=str, help='can be certain or uncertain')
parser.add_argument('--score-type', default='L2',
                    type=str, help='can be L2 or cosine')


# note that the test split is fixed by default - as per DPC
parser.add_argument('--split', default=1, type=int)

parser.add_argument('--epoch', default=100, type=int)

# custom embeddings writer or tensorboardX summarywriter
# weather to get embedding or kNN
parser.add_argument('--output', default='embedding', type=str,
                    help='embeddings or neighbours')

parser.add_argument('--K', default=10, type=int)
parser.add_argument('--background', default='white', type=str)


def get_data(transform, mode='train'):
    print('[embeddings.py] Loading data for "%s" ...' % mode)
    global dataset
    if args.dataset == 'ucf101':
        dataset = UCF101_3d(mode=mode,
                            transform=transform,
                            seq_len=args.seq_len,
                            num_seq=args.num_seq,
                            downsample=args.ds,
                            which_split=args.split)
    elif args.dataset == 'hmdb51':
        dataset = HMDB51_3d(mode=mode,
                            transform=transform,
                            seq_len=args.seq_len,
                            num_seq=args.num_seq,
                            downsample=args.ds,
                            which_split=args.split)
    elif args.dataset == 'ucf11':
        # no split here
        dataset = UCF11_3d(mode=mode,
                           transform=transform,
                           num_seq=args.num_seq,
                           seq_len=args.seq_len,
                           downsample=args.ds)
    elif args.dataset == 'block_toy':
        # no split here
        dataset = block_toy(mode=mode,
                           transform=transform,
                           num_seq=args.num_seq,
                           seq_len=args.seq_len,
                           downsample=args.ds,
                           background=args.background)
    else:
        raise ValueError('dataset not supported')

    # shuffle data
    # print (dataset.shape)
    my_sampler = data.SequentialSampler(dataset)
    # print(len(dataset)

    if mode == 'val':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=my_sampler,
                                      shuffle=False,
                                      num_workers=16,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'test':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=my_sampler,
                                      shuffle=False,
                                      num_workers=16,
                                      pin_memory=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader

def generate_distance_matrix(embed_dir, epoch = 50, gpu = '6,7', distance = 'L2'):
    path = os.path.join(embed_dir, "%0.5d" % epoch)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    global cuda
    cuda = torch.device('cuda')

    output_embed = torch.load(os.path.join(path, 'output_embed.pt')).to(cuda)
    # target_embed = torch.load(os.path.join(path, 'target_embed.pt')).to(cuda)

    if distance == 'cosine':
        distance_f = nn.CosineSimilarity(dim=0, eps=1e-6)

    adj = torch.zeros(output_embed.shape[0], output_embed.shape[0])
    for i in tqdm(range(output_embed.shape[0])):
        for j in range(output_embed.shape[0]):
            a = output_embed[i]#.reshape(-1)
            b = output_embed[j]#.reshape(-1)
            if distance == 'cosine':
                adj[i, j] = distance_f(a, b)
            elif distance == 'L2':
                adj[i, j] = torch.dist(a, b, p=2)
    if distance == 'cosine':
        torch.save(adj, os.path.join(path, 'cosine_distance.pt'))
    elif distance == 'L2':
        torch.save(adj, os.path.join(path, 'L2_distance.pt'))

def KNN_retrieval(embed_dir, distance='L2', img_dim=256, dataset='block_toy', mode='test', seq_len=5, num_seq=6, ds=1, epoch=200, K=10):
    if distance == 'cosine':
        distance_f = nn.CosineSimilarity(dim=0, eps=1e-6)

    transform = transforms.Compose([
        CenterCrop(size=224),
        Scale(size=(img_dim, img_dim)),
        ToTensor(),
    ])

    # construct dictionary of class name
    if dataset == 'ucf11':
        classInd_dir = '/proj/vondrick/lovish/datasets/ucf11/classInd.txt'
    elif dataset == 'ucf101':
        classInd_dir = '/proj/vondrick/lovish/datasets/ucf101/splits_classification/classInd.txt'
    elif dataset == 'block_toy':
        classInd_dir = '/proj/vondrick/lovish/data/block_toy/classInd.txt'
    class_name = {}
    with open(classInd_dir) as f:
        for line in f:
            (key, val) = line.split()
            class_name[int(key)] = val
    print(class_name)


    if dataset == 'ucf101':
        dataset = UCF101_3d(mode=mode,
                            transform=transform,
                            seq_len=seq_len,
                            num_seq=num_seq,
                            downsample=ds)
    elif dataset == 'hmdb51':
        dataset = HMDB51_3d(mode=mode,
                            transform=transform,
                            seq_len=seq_len,
                            num_seq=num_seq,
                            downsample=ds)
    elif dataset == 'ucf11':
        # no split here
        dataset = UCF11_3d(mode=mode,
            transform=transform,
            num_seq=num_seq,
            seq_len=seq_len,
            downsample=ds)
    elif dataset == 'block_toy':
        # no split here
        dataset = block_toy(mode=mode,
            transform=transform,
            num_seq=num_seq,
            seq_len=seq_len,
            downsample=ds,
            background=args.background)
    else:
        raise ValueError('dataset not supported')


    # print(dataset[300][0].shape)
    # print(dataset[300][1])
    # print(dataset[300][2])

    # classes = list(map(int, args.classes.split(",")))
    # print(classes)

    distance_path = os.path.join(embed_dir, "%0.5d" % epoch)
    if distance == 'cosine':
        adj = torch.load(os.path.join(distance_path, 'cosine_distance.pt')).numpy()
    elif distance == 'L2':
        adj = torch.load(os.path.join(distance_path, 'L2_distance.pt')).numpy()

    labels = torch.load(os.path.join(distance_path, 'target_embed.pt')).numpy()
    # images = torch.load(os.path.join(args.embed_dir, 'input_embed.pt')).numpy() # input embeddings

    save_dir = os.path.join(distance_path, 'nearest_%d' % K)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # print(images.shape, labels.shape, adj.shape)
    # images = images * 0.15 + 0.45
    for i in class_name.keys():
        np.random.seed(0)
        try:
            index_origin = np.random.choice(np.where(labels == i-1)[0], 1)[0] # sample 1 image from action class 'i'
        except Exception:
            break
        print(index_origin)
        if distance == 'L2':
            indices_quiried = adj[index_origin].argsort()[0:K-1] # query K nearest neighbors from adjacency matrix
        elif distance == 'cosine':
            indices_quiried = adj[index_origin].argsort()[-K+1:][::-1]
        # print(adj[0])
        # print(adj[1])
        # print(indices_quiried)
        # print("********")
        # print(indices_quiried)
        image_origin = dataset[index_origin][0].permute(1,0,2,3,4)[:,2,2,:,:].numpy()
        class_dir = os.path.join(save_dir, class_name[i]) # directory for class being quiried
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        plt.imsave(os.path.join(class_dir, 'origin_class_%s.jpg' % class_name[i]), np.transpose(image_origin, (1, 2, 0)))
        images_quiried_labels = labels[indices_quiried]
        # print(len(indices_quiried))
        # print(images_quiried_labels)
        # print(indices_quiried)
        # print(images_quiried_labels)
        for j in range(len(indices_quiried)):
            num = indices_quiried[j]
            image_quiried = dataset[num][0].permute(1,0,2,3,4)[:,2,2,:,:].numpy()
            label = dataset[num][1].numpy()
            plt.imsave(os.path.join(class_dir, 'quiried_num_%d_class_%s.jpg' % (j, class_name[images_quiried_labels[j]+1])), np.transpose(image_quiried, (1, 2, 0)))

def main():

    global args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    global cuda
    cuda = torch.device('cuda')

    # get validation data with label for plotting the embedding
    transform = transforms.Compose([
        # centercrop
        CenterCrop(size=224),
        RandomSizedCrop(consistent=True, size=224, p=0.0),
        Scale(size=(args.img_dim, args.img_dim)),
        ToTensor(),
        Normalize()
    ])

    data_loader = get_data(transform, 'test')

    if args.output == 'embedding':
        # write the embeddings to writer
        try:  # old version
            writer = SummaryWriter(
                log_dir=args.test_dir + 'embeddings')
        except:  # v1.7
            writer = SummaryWriter(
                logdir=args.test_dir + 'embeddings')

    elif args.output == 'neighbours':

        base_path = args.test_dir + 'neighbours'
        try:
            if not os.path.exists(base_path):
                os.mkdir(base_path)
        except OSError as error:
            print(error)

    # assuming that model is saved after every 10 epochs

    # extract the files corresponding to each epochs
    model_path = os.path.join(args.test_dir, "model")
    model_filename = "epoch" + str(args.epoch) + ".pth.tar"
    model_file = os.path.join(model_path, model_filename)
    # print(model_file)
    try:
        checkpoint = torch.load(model_file)
    except:
        return

    # print values corresponding to the keys

    if args.output == 'embedding':
        model = model_visualize(sample_size=args.img_dim,
                  seq_len=args.seq_len,
                  pred_steps=1,
                  network=args.net,
                  feature_type='Phi',
                  distance_type=args.distance_type)
    elif args.output == 'neighbours':
        model = model_visualize(sample_size=args.img_dim,
                  seq_len=args.seq_len,
                  pred_steps=1,
                  network=args.net,
                  feature_type='Phi',
                  distance_type=args.distance_type)
    model = model.to(cuda)
    model.eval()

    model = neq_load_customized(model, checkpoint['state_dict'])
    model = nn.DataParallel(model)

    input_embed_all = torch.Tensor()
    output_embed_all = torch.Tensor().to(cuda)
    target_embed_all = torch.LongTensor()
    if args.distance_type == 'uncertain':
        radius_embed_all = torch.Tensor().to(cuda)

    with torch.no_grad():
        for idx, (input_seq, target) in enumerate(data_loader):

            # get the middle frame of the middle sequence to represent with visualization
            input_embed = input_seq.permute(0, 2, 1, 3, 4, 5)[:, :, 2, 2, :, :]
            input_seq = input_seq.to(cuda)
            output = model(input_seq)

            # get a 256-dim embedding from F+G
            output_embed = output.squeeze()

            # consider only first 256 dimensions in case we use radii
            # Note that radii will not be displayed in the embeddings

            if args.distance_type == 'uncertain':
                output_embed = output_embed[:, :-1]
                # radius_embed = output_embed[:, -1].expand(1, -1)
                # print(radius_embed.shape)
            elif args.distance_type == 'certain':
                output_embed = output_embed

            # actionlabels for each video
            target_embed = target.squeeze()
            target_embed = target_embed.reshape(-1)

            # concatenate all pertaining to current batch
            input_embed_all = torch.cat(
                (input_embed_all, input_embed), 0)
            output_embed_all = torch.cat(
                (output_embed_all, output_embed), 0)
            target_embed_all = torch.cat(
                (target_embed_all, target_embed), 0)
            # if args.distance_type == 'uncertain':
                # radius_embed_all = torch.cat(
                    # (radius_embed_all, radius_embed), 1)
            # print("Here")

    if args.output == 'embedding':
        # store the embedding as tensorboard projection to visualize
        writer.add_embedding(
            output_embed_all,
            metadata=target_embed_all,
            # label_img=input_embed_all,
            global_step=i)
        print ("Right here")
    elif args.output == 'neighbours':

        # store all the inputs, outputs and targets as a single file
        # output = output.squeeze()
        # output_embed = output[:, :-1, :, :]
        # radius = output[:, -1, :, :]
        # target = target.squeeze()

        # input_embed = input_seq.squeeze().detach().cpu()

        # print(input_embed.shape)

        # choose middle step and middle frame as representation
        # input_embed = input_embed[:, :, 2, :, :].squeeze()
        path = os.path.join(base_path, "%0.5d" % args.epoch)
        try:
            os.mkdir(path)
        except OSError as error:
            print(error)
        torch.save(input_embed_all, os.path.join(
            base_path, 'input_embed.pt'))
        torch.save(output_embed_all.detach().cpu(),
                   os.path.join(path, 'output_embed.pt'))
        torch.save(target_embed_all.detach().cpu(),
                   os.path.join(path, 'target_embed.pt'))
        # store radius in case of uncertain weights
        # if args.distance_type == 'uncertain':
            # torch.save(radius_embed_all.detach().cpu(),
            #      os.path.join(path, 'radius.pt'))

        print(input_embed_all.shape, output_embed_all.shape,
              target_embed_all.shape)

    generate_distance_matrix(embed_dir = base_path, epoch = args.epoch, gpu = args.gpu, distance = args.score_type)
    KNN_retrieval(embed_dir = base_path, distance=args.score_type, img_dim=256, dataset=args.dataset, mode='test', seq_len=args.seq_len, num_seq=args.num_seq, ds=args.ds, epoch=args.epoch, K=args.K)


def neq_load_customized(model, pretrained_dict):
    ''' load pre-trained model in a not-equal way,
    when new model has been partially modified '''
    model_dict = model.state_dict()
    tmp = {}

    for k, v in pretrained_dict.items():
        k_alt = ".".join(k.split(".")[1:])
        if k_alt in model_dict:
            # print(k_alt)
            tmp[k_alt] = v
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model


if __name__ == '__main__':
    # model = model_F_G.F(128, seq_len=3, network='resnet10', distance_type='certain')
    main()
