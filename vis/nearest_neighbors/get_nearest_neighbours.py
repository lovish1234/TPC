import sys


sys.path.append('../../utils')
sys.path.append('../util')

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
parser.add_argument('--num_seq', default=8,
                    type=int, help='when testing 6 pred 2 model on first prediction, use 5 pred 1')  # number of sequences
# number of frames in each sequence
parser.add_argument('--seq_len', default=5, type=int)
parser.add_argument('--pred_steps', default=1, type=int)
parser.add_argument('--ds', default=3, type=int,
                    help='frame downsampling rate')

# embedding for certain vs uncertain
parser.add_argument('--distance-type', default='certain',
                    type=str, help='can be certain or uncertain')
parser.add_argument('--score-type', default='L2',
                    type=str, help='can be L2 or cosine')

parser.add_argument('--visualize-tube', default=0, type=int)

# note that the test split is fixed by default - as per DPC
parser.add_argument('--split', default=1, type=int)

parser.add_argument('--epoch', default=100, type=int)

# custom embeddings writer or tensorboardX summarywriter
# weather to get embedding or kNN
parser.add_argument('--output', default='embedding', type=str,
                    help='embeddings or neighbours')

parser.add_argument('--K', default=20, type=int)

parser.add_argument('--difficulty', default=2, type=int)
parser.add_argument('--subset-portion', default=0.2, type=float)


def get_data(transform, mode='train', random_frames=0, num_seq_NN=5):
    print('[embeddings.py] Loading data for "%s" ...' % mode)
    global dataset
    if args.dataset == 'ucf101':
        dataset = UCF101_3d(mode=mode,
                            transform=transform,
                            seq_len=args.seq_len,
                            num_seq=num_seq_NN,
                            downsample=args.ds,
                            which_split=args.split)
    elif args.dataset == 'hmdb51':
        dataset = HMDB51_3d(mode=mode,
                            transform=transform,
                            seq_len=args.seq_len,
                            num_seq=num_seq_NN,
                            downsample=args.ds,
                            which_split=args.split)
    elif args.dataset == 'ucf11':
        # no split here
        dataset = UCF11_3d(mode=mode,
                           transform=transform,
                           num_seq=num_seq_NN,
                           seq_len=args.seq_len,
                           downsample=args.ds)
    elif args.dataset == 'block_toy':
        # no split here
        dataset = block_toy(mode=mode,
                            transform=transform,
                            num_seq=num_seq_NN,
                            seq_len=args.seq_len,
                            downsample=args.ds,
                            num=args.difficulty,
                            random_frames=random_frames)
    elif args.dataset == 'block_toy_imagenet':
        # no split here
        dataset = block_toy_imagenet(mode=mode,
                                     transform=transform,
                                     num_seq=num_seq_NN,
                                     seq_len=args.seq_len,
                                     downsample=args.ds,
                                     num=args.difficulty,
                                     random_frames=random_frames)
    elif args.dataset == 'block_toy_combined_test':
        # no split here
        dataset = block_toy_combined_test(transform=transform,
                                          num_seq=num_seq_NN,
                                          seq_len=args.seq_len,
                                          downsample=args.ds,
                                          num=args.difficulty,
                                          random_frames=random_frames)
    elif args.dataset == 'block_toy_combined_train':
        # no split here
        dataset = block_toy_combined_train(transform=transform,
                                           num_seq=num_seq_NN,
                                           seq_len=args.seq_len,
                                           downsample=args.ds,
                                           num=args.difficulty,
                                           random_frames=random_frames)
    else:
        raise ValueError('dataset not supported')

    # shuffle data
    # print (dataset.shape)

    if args.subset_portion == 1:
        my_dataset = dataset
        # print(len(dataset)
    else:
        data_indexes = list(
            range(0, len(dataset) - 1, int(1 / args.subset_portion)))
        print(max(data_indexes))
        print(min(data_indexes))

        my_dataset = data.Subset(dataset, data_indexes)
        print(len(my_dataset))
    my_sampler = data.SequentialSampler(my_dataset)
    if mode == 'val':
        data_loader = data.DataLoader(my_dataset,
                                      batch_size=args.batch_size,
                                      sampler=my_sampler,
                                      shuffle=False,
                                      num_workers=16,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'test':
        data_loader = data.DataLoader(my_dataset,
                                      batch_size=args.batch_size,
                                      sampler=my_sampler,
                                      shuffle=False,
                                      num_workers=16,
                                      pin_memory=True)
    print('"%s" dataset size: %d' % (mode, len(my_dataset)))
    print('length of dataloader:', len(data_loader))
    return data_loader


def KNN_retrieval(embed_dir, score_type='L2', img_dim=256, dataset='block_toy', mode='test', seq_len=5, num_seq=6, ds=1, epoch=200, K=10):
    if score_type == 'cosine':
        distance_f = nn.CosineSimilarity(dim=0, eps=1e-6)

    # construct dictionary of class name
    if dataset == 'ucf11':
        classInd_dir = '/proj/vondrick/lovish/datasets/ucf11/classInd.txt'
    elif dataset == 'ucf101':
        classInd_dir = '/proj/vondrick/lovish/datasets/ucf101/splits_classification/classInd.txt'
    elif dataset in ['block_toy', 'block_toy_imagenet', 'block_toy_combined_test', 'block_toy_combined_train']:
        classInd_dir = '/proj/vondrick/lovish/data/block_toy/classInd.txt'
    class_name = {}
    with open(classInd_dir) as f:
        for line in f:
            (key, val) = line.split()
            class_name[int(key)] = val
    print(class_name)

    distance_path = os.path.join(embed_dir, "%0.5d" % args.epoch)

    output_embed = torch.load(os.path.join(distance_path, 'output_embed.pt'))
    target_embed = torch.load(os.path.join(distance_path, 'target_embed.pt'))
    input_embed = torch.load(os.path.join(
        embed_dir, 'input_embed.pt'))  # input embeddings
    radius_embed = torch.load(os.path.join(
        distance_path, 'radius.pt')).numpy()[0]
    GT_output_embed = torch.load(os.path.join(
        distance_path, 'GT_output_embed.pt'))
    GT_target_embed = torch.load(os.path.join(
        distance_path, 'GT_target_embed.pt'))
    GT_input_embed = torch.load(os.path.join(
        embed_dir, 'GT_input_embed.pt'))  # ground truth embeddings

    save_dir = os.path.join(distance_path, 'nearest_%d' % K)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # print('here')

    input_embed = input_embed * 0.15 + 0.55
    GT_input_embed = GT_input_embed * 0.15 + 0.55

    for i in class_name.keys():
        np.random.seed(0)
        print(i)
        try:
            # sample 1 image from action class 'i'
            index_origin_list = np.random.choice(
                np.where(target_embed == i - 1)[0], 20)
            print(index_origin_list)
        except Exception:
            continue
        print(index_origin_list)

        # directory for class being quiried
        class_dir = os.path.join(save_dir, class_name[i])
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)

        for index_origin in index_origin_list:
            origin_embed = output_embed[index_origin]
            distances = []
            for j in tqdm(range(GT_output_embed.shape[0])):
                GT_embed = GT_output_embed[j, :]
                if score_type == 'cosine':
                    distances.append(distance_f(origin_embed, GT_embed))
                elif score_type == 'L2':
                    distances.append(torch.dist(origin_embed, GT_embed, p=2))
            distances = np.asarray(distances, dtype=np.float32)
            print('average distance: ', np.average(distances))
            # print(distances[0:10][0:10])

            if score_type == 'L2':
                if args.distance_type == 'uncertain' and args.visualize_tube == 1:
                    neg_idx = np.where(distances > radius_embed[index_origin])[
                        0]   # pick out indexes of images outside radius
                    pos_idx = np.where(
                        distances <= radius_embed[index_origin])[0]

                    print('length of neg: ', len(neg_idx))
                    print('length of pos: ', len(pos_idx))
                    print('radius: ', radius_embed[index_origin])
                    neg_K = neg_idx[distances[neg_idx].argsort()][0:np.minimum(
                        K - 1, len(neg_idx))]   # find smallest distance among these images
                    pos_K = pos_idx[distances[pos_idx].argsort(
                    )][0:np.minimum(K - 1, len(pos_idx))]
                    print('neg_distance:', distances[neg_idx])
                    print(neg_K)
                    print('pos_distance:', distances[pos_idx])
                    print(pos_K)
                else:
                    # query K nearest neighbors from adjacency matrix
                    indices_quiried = distances.argsort()[0:K]
                    # far_quiried = distances.argsort()[-K+1:][::-1]
            elif score_type == 'cosine':
                indices_quiried = distances.argsort()[-K + 1:][::-1]

            images_origin = input_embed[index_origin].numpy()

            im_dir = os.path.join(class_dir, str(index_origin))
            if not os.path.exists(im_dir):
                os.mkdir(im_dir)

            for j in range(images_origin.shape[0]):
                plt.imsave(os.path.join(im_dir, 'origin_class_%s_%d.jpg' % (
                    class_name[i], j)), np.transpose(images_origin[j], (1, 2, 0)))
            # images_quiried_labels = target_embed[indices_quiried]
            # print(len(indices_quiried))
            # print(images_quiried_labels)
            # print(indices_quiried)
            # print(images_quiried_labels)
            # print(indices_quiried)
            if score_type == 'L2' and args.distance_type == 'uncertain' and args.visualize_tube == 1:
                for j in range(len(pos_K)):
                    num = pos_K[j]
                    image_quiried = GT_input_embed[num].numpy()
                    # label = dataset[num][1].numpy()
                    plt.imsave(os.path.join(im_dir, 'quiried_pos_%d.jpg' %
                                            j), np.transpose(image_quiried, (1, 2, 0)))
                for j in range(len(neg_K)):
                    num = neg_K[j]
                    image_quiried = GT_input_embed[num].numpy()
                    # label = dataset[num][1].numpy()
                    plt.imsave(os.path.join(im_dir, 'quiried_neg_%d.jpg' %
                                            j), np.transpose(image_quiried, (1, 2, 0)))
            else:
                for j in range(len(indices_quiried)):
                    num = indices_quiried[j]
                    image_quiried = GT_input_embed[num].numpy()
                    # label = dataset[num][1].numpy()
                    plt.imsave(os.path.join(im_dir, 'quiried_num_%d.jpg' %
                                            j), np.transpose(image_quiried, (1, 2, 0)))

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

    data_loader_pred = get_data(transform, 'test', 0, args.num_seq)
    # dataloader to generate GT embeddings
    data_loader_GT = get_data(transform, 'test', 0, args.num_seq)

    if args.output == 'embedding':
        # write the embeddings to writer
        try:  # old version
            writer = SummaryWriter(
                log_dir=args.test_dir + 'embeddings')
        except:  # v1.7
            writer = SummaryWriter(
                logdir=args.test_dir + 'embeddings')

    elif args.output == 'neighbours':

        base_path = os.path.join(
            args.test_dir, 'neighbours_%s_%s' % (args.score_type, args.dataset))
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
    try:
        checkpoint = torch.load(model_file)
    except:
        print('model failed to load')
        return
    # print values corresponding to the keys
    if args.output == 'embedding':
        model = model_visualize(sample_size=args.img_dim,
                                seq_len=args.seq_len,
                                pred_steps=args.pred_steps,
                                network=args.net,
                                feature_type='Phi',
                                distance_type=args.distance_type)
    elif args.output == 'neighbours':
        model = model_visualize(sample_size=args.img_dim,
                                seq_len=args.seq_len,
                                pred_steps=args.pred_steps,
                                network=args.net,
                                feature_type='F',
                                distance_type=args.distance_type)
    model = model.to(cuda)
    model.eval()

    model_F = model_visualize(sample_size=args.img_dim,
                              seq_len=args.seq_len,
                              pred_steps=1,
                              network=args.net,
                              feature_type='F',
                              distance_type=args.distance_type)

    model_F = model_F.to(cuda)
    model_F.eval()

    model_F = neq_load_customized(model_F, checkpoint['state_dict'])
    model_F = nn.DataParallel(model_F)
    model = neq_load_customized(model, checkpoint['state_dict'])
    model = nn.DataParallel(model)

    # print('here')

    # process ground truth embeddings
    output_embed_all = torch.Tensor().to(cuda)
    target_embed_all = torch.LongTensor()

    GT_input_embed_all = torch.Tensor()
    GT_output_embed_all = torch.Tensor().to(cuda)
    GT_target_embed_all = torch.LongTensor()
    jj = 0
    with torch.no_grad():
        for idx, (input_seq, target) in enumerate(data_loader_GT):
            print('Processing batch #', jj)
            jj += 1

            # input_seq: (B, N, C, SL, H, W) || (B, C, N, SL, H, W)
            input_embed = torch.Tensor()
            # print(input_seq.shape)
            print(input_seq.permute(0, 2, 1, 3, 4, 5)[:, :, -1, 0, :, :].shape)
            for i in range(args.num_seq):
                input_embed = torch.cat((input_embed, input_seq.permute(0, 2, 1, 3, 4, 5)[
                                        :, :, i, -1, :, :]), 0)  # concatenate batch
            # print(input_embed.shape)
            input_seq = input_seq.to(cuda)
            output = model_F(input_seq)
            # print('output shape: ', output.shape)

            output_embed = torch.Tensor().to(cuda)
            for i in range(args.num_seq):
                output_embed = torch.cat(
                    (output_embed, output[:, i, :]), 0)  # concatenate batch

            output_embed = output_embed.squeeze()

            if args.distance_type == 'uncertain':
                output_embed = output_embed[:, :-1]
                # radius_embed = output_embed[:, -1].expand(1, -1)
                # print(radius_embed.shape)
            elif args.distance_type == 'certain':
                output_embed = output_embed

            target_embed = target.squeeze()
            target_embed = target_embed.reshape(-1)

            # concatenate all pertaining to current batch
            GT_input_embed_all = torch.cat(
                (GT_input_embed_all, input_embed), 0)
            GT_output_embed_all = torch.cat(
                (GT_output_embed_all, output_embed), 0)
            GT_target_embed_all = torch.cat(
                (GT_target_embed_all, target_embed), 0)

    input_embed_all = torch.Tensor()
    output_embed_all = torch.Tensor().to(cuda)
    target_embed_all = torch.LongTensor()
    if args.distance_type == 'uncertain':
        radius_embed_all = torch.Tensor().to(cuda)

    # print('here')

    with torch.no_grad():
        for idx, (input_seq, target) in tqdm(enumerate(data_loader_pred)):

            # get the middle frame of the middle sequence to represent with visualization
            # input_seq: (B, N, C, SL, H, W)
            # input_embed: (B, N, SL, C, H, W)
            input_embed = torch.Tensor()
            print(input_seq.permute(0, 1, 3, 2, 4, 5)[:, :, -1, :, :, :].shape)
            input_embed = torch.cat((input_embed, input_seq.permute(
                0, 1, 3, 2, 4, 5)[:, :, -1, :, :, :]), 0)
            input_seq = input_seq.to(cuda)
            output = model(input_seq)
            # print(output.shape)

            # get a 256-dim embedding from F+G
            output_embed = output.squeeze()

            # consider only first 256 dimensions in case we use radii
            # Note that radii will not be displayed in the embeddings

            if args.distance_type == 'uncertain':
                radius_embed = output_embed[:, -1].expand(1, -1)
                output_embed = output_embed[:, :-1]
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
            if args.distance_type == 'uncertain':
                radius_embed_all = torch.cat(
                    (radius_embed_all, radius_embed), 1)
            # print("Here")

    if args.output == 'embedding':
        # store the embedding as tensorboard projection to visualize
        writer.add_embedding(
            output_embed_all,
            metadata=target_embed_all,
            # label_img=input_embed_all,
            global_step=i)
        print("Right here")
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
        print(path)
        try:
            os.mkdir(path)
        except OSError as error:
            print(error)
        torch.save(GT_input_embed_all, os.path.join(
            base_path, 'GT_input_embed.pt'))
        torch.save(GT_output_embed_all.detach().cpu(),
                   os.path.join(path, 'GT_output_embed.pt'))
        torch.save(GT_target_embed_all.detach().cpu(),
                   os.path.join(path, 'GT_target_embed.pt'))

        torch.save(input_embed_all, os.path.join(
            base_path, 'input_embed.pt'))
        torch.save(output_embed_all.detach().cpu(),
                   os.path.join(path, 'output_embed.pt'))
        torch.save(target_embed_all.detach().cpu(),
                   os.path.join(path, 'target_embed.pt'))
        # store radius in case of uncertain weights
        if args.distance_type == 'uncertain':
            torch.save(radius_embed_all.detach().cpu(),
                       os.path.join(path, 'radius.pt'))

        print('input_embed:', input_embed_all.shape, '    output_embed: ',
              output_embed_all.shape, '    target_embed: ', target_embed_all.shape)
        print('GT_input_embed:', GT_input_embed_all.shape, '    GT_output_embed: ',
              GT_output_embed_all.shape, '    GT_target_embed: ', GT_target_embed_all.shape)

    # generate_distance_matrix(embed_dir = base_path, epoch = args.epoch, gpu = args.gpu, distance = args.score_type)


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
    # main()
    global args
    args = parser.parse_args()
    base_path = os.path.join(args.test_dir, 'neighbours_%s_%s' %
                             (args.score_type, args.dataset))
    main()
    KNN_retrieval(embed_dir=base_path, score_type=args.score_type, img_dim=256, dataset=args.dataset,
                  mode='test', seq_len=args.seq_len, num_seq=args.num_seq, ds=args.ds, epoch=args.epoch, K=args.K)
