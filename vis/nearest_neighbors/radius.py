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
parser.add_argument('--num_seq', default=5, type=int)
parser.add_argument('--pred_steps', default=1, type=int)
parser.add_argument('--seq_len', default=3, type=int)
parser.add_argument('--ds', default=3, type=int,
                    help='frame downsampling rate')


parser.add_argument('--score-type', default='L2',
                    type=str, help='can be L2 or cosine')

parser.add_argument('--epoch', default=100, type=int)

# custom embeddings writer or tensorboardX summarywriter
# weather to get embedding or kNN
parser.add_argument('--output', default='embedding', type=str,
                    help='embeddings or neighbours')

parser.add_argument('--distance_type', default='uncertain', type=str)

parser.add_argument('--K', default=10, type=int)
parser.add_argument('--background', default='white', type=str)


def get_data(transform, mode='train', random_frames=0):
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
  elif args.dataset == 'block_toy_combined_test':
    dataset = block_toy_combined_test(transform=transform,
                                      num_seq=args.num_seq,
                                      seq_len=args.seq_len,
                                      downsample=args.ds,
                                      random_frames=random_frames)
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

def main():

  global args
  args = parser.parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

  global cuda
  cuda = torch.device('cuda')

  # get validation data with label for plotting the embedding
  transform = transforms.Compose([
      # centercrop
      # CenterCrop(size=224),
      # RandomSizedCrop(consistent=True, size=224, p=0.0),
      Scale(size=(args.img_dim, args.img_dim)),
      ToTensor(),
      Normalize()
  ])

  data_loader = get_data(transform, 'test', 0)

  if args.output == 'embedding':
    # write the embeddings to writer
    try:  # old version
      writer = SummaryWriter(
          log_dir=args.test_dir + 'embeddings')
    except:  # v1.7
      writer = SummaryWriter(
          logdir=args.test_dir + 'embeddings')

  elif args.output == 'neighbours':

    base_path = os.path.join(args.test_dir, 'radius')
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
                            pred_steps=args.pred_steps,
                            network=args.net,
                            feature_type='Phi',
                            distance_type=args.distance_type)
  elif args.output == 'neighbours':
    model = model_visualize(sample_size=args.img_dim,
                            seq_len=args.seq_len,
                            pred_steps=args.pred_steps,
                            network=args.net,
                            feature_type='Phi',
                            distance_type=args.distance_type)
  model = model.to(cuda)
  model.eval()

  model = neq_load_customized(model, checkpoint['state_dict'])
  model = nn.DataParallel(model)

  radius_embed_all = torch.Tensor().to(cuda)
  target_embed_all = torch.LongTensor()

  with torch.no_grad():
    for idx, (input_seq, target) in tqdm(enumerate(data_loader)):

      # get the middle frame of the middle sequence to represent with visualization
      input_embed = input_seq.permute(0, 2, 1, 3, 4, 5)[:, :, 2, 2, :, :]
      input_seq = input_seq.to(cuda)
      output = model(input_seq)

      # get a 256-dim embedding from F+G
      output_embed = output.squeeze()

      # consider only first 256 dimensions in case we use radii
      # Note that radii will not be displayed in the embeddings
      radius_embed = output_embed[:, -1].expand(1, -1)
      radius_embed_all = torch.cat(
          (radius_embed_all, radius_embed), 1)
      # print("Here")

      target_embed = target.squeeze()
      target_embed = target_embed.reshape(-1)
      target_embed_all = torch.cat(
          (target_embed_all, target_embed), 0)

  if args.output == 'embedding':
    # store the embedding as tensorboard projection to visualize
    writer.add_embedding(
        output_embed_all,
        metadata=target_embed_all,
        # label_img=input_embed_all,
        global_step=i)
    print("Right here")
  elif args.output == 'neighbours':
    path = os.path.join(base_path, "%0.5d" % args.epoch)
    try:
      os.mkdir(path)
    except OSError as error:
      print(error)
    torch.save(radius_embed_all.detach().cpu(),
               os.path.join(path, 'radius.pt'))
    torch.save(target_embed_all.detach().cpu(),
               os.path.join(path, 'target_embed.pt'))

    print(radius_embed_all.shape, target_embed_all.shape)


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
