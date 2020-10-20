import sys
sys.path.append('util/')
from model_visualize import *
from dataset_visualize import *


sys.path.append('../utils')
from augmentation import *
from utils import denorm


import argparse
import os

from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt

# saving log, calculating accuracy etc.
import torchvision.utils as vutils


parser = argparse.ArgumentParser()

parser.add_argument(
    '--test-dir', default='/proj/vondrick/lovish/DPC/dpc/log_tmp/epic-128_r18_dpc-rnn_bs64_lr0.001_seq8_pred3_len5_ds6_train-all', type=str)
parser.add_argument('--num_seq', default=1, type=int)
parser.add_argument('--seq_len', default=5, type=int)
parser.add_argument('--ds', default=6, type=int,
                    help='frame downsampling rate')

parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument('--gpu', default='6,7', type=str)
parser.add_argument('--dataset', default='epic', type=str)

parser.add_argument('--distance-type', default='certain',
                    type=str, help='can be certain or uncertain')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--split', default=1, type=int)

parser.add_argument('--distance', default='cosine', type=str)
parser.add_argument('--num-neighbours', default=5, type=int)

parser.add_argument('--metric', default='embedding',
                    type=str, help='embedding or prediction')


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
    elif args.dataset == 'epic':
        dataset = epic(mode=mode,
                       transform=transform,
                       num_seq=args.num_seq,
                       seq_len=args.seq_len,
                       downsample=args.ds)
    else:
        raise ValueError('dataset not supported')

    sampler = data.RandomSampler(dataset)

    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=32,
                                      pin_memory=True,
                                      drop_last=False)
    elif mode == 'val':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=32,
                                      pin_memory=True,
                                      drop_last=False)

    print('"%s" dataset size: %d' % (mode, len(dataset)))
    print('length of dataloader:', len(data_loader))
    return data_loader

def get_model(feature_type='F'):

    # extract the files corresponding to each epochs
    model_path = os.path.join(args.test_dir, "model")

    # use original DPC weights
    model_filename = "model_best_epoch974.pth.tar"
    model_file = os.path.join(model_path, model_filename)

    try:
        checkpoint = torch.load(model_file)
    except:
        print('model failed to load')
        return

    model = model_visualize(sample_size=args.img_dim,
                            seq_len=args.seq_len,
                            pred_steps=1,
                            network=args.net,
                            feature_type='F',
                            distance_type=args.distance_type)
    model = model.to(cuda)
    model.eval()

    model = neq_load_customized(model, checkpoint['state_dict'])
    model = nn.DataParallel(model)
    return model

def get_distance(embedding):

    output_embed = embedding
    if args.distance == 'cosine':

        distance_f = nn.CosineSimilarity(dim=0, eps=1e-6)

        adj = torch.zeros(output_embed.shape[0], output_embed.shape[0])
        for i in tqdm(range(output_embed.shape[0])):
            for j in range(output_embed.shape[0]):
                a = output_embed[i].reshape(-1)
                b = output_embed[j].reshape(-1)
                adj[i, j] = distance_f(a, b)

    return adj

def get_class(class_id):

    # get action list [here just the values of y]
    action_dict_decode = {}

    action_file = os.path.join(
        '/proj/vondrick/datasets/epic-kitchens/data/annotations', 'EPIC_verb_classes.csv')
    action_df = pd.read_csv(action_file, sep=' ', header=None)
    for _, row in action_df.iterrows():
        act_id, act_name = row
        act_id = int(act_id) - 1  # let id start from 0
        action_dict_decode[act_id] = act_name
    return action_dict_decode[class_id]

def main():

    global args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    global cuda
    cuda = torch.device('cuda')

    transform = transforms.Compose([
        # centercrop
        CenterCrop(size=224),
        RandomSizedCrop(consistent=True, size=224, p=0.0),
        Scale(size=(args.img_dim, args.img_dim)),
        ToTensor(),
        Normalize()
    ])

    # the NN of a particular sequence should be the same sequence itself
    data_loader_pred = get_data(transform, 'val')
    model = get_model(feature_type='F')

    input_embed_all = torch.Tensor()
    output_embed_all = torch.Tensor().to(cuda)
    target_embed_all = torch.LongTensor()
    frame_min_embed_all = torch.LongTensor()
    frame_max_embed_all = torch.LongTensor()
    path_list = []

    f = open('fig1_epic.html', 'w')
    header = """ <!DOCTYPE HTML>
                 <html lang = "en">
                 <head>
                 <title>basicTable.html</title>
                 <meta charset = "UTF-8" />
                 <style type = "text/css">
                 table, td, th {
                 border: 1px solid black;
                 } 
                 </style>
                 </head>
                 <body>
                 <h2>DPC Nearest Neighbours</h2>
                 <table>
                 <tr>
                     <th>S.No.</th>
                     <th>Query</th>
                     <th>Neighbour 1</th>
                     <th>Neighbour 2</th>
                     <th>Neighbour 3</th>
                     <th>Neighbour 4</th>
                     <th>Neighbour 5</th>
                 </tr>"""
    f.write(header)
    de_normalize = denorm()

    with torch.no_grad():
        for idx, (input_seq, vpath, frame_first_last) in enumerate(data_loader_pred):
            input_embed = torch.Tensor()
            input_embed = torch.cat((input_embed, input_seq.permute(
                0, 1, 3, 2, 4, 5)[:, :, -1, :, :, :]), 0)

            path_list.extend(vpath)

            # save the files in a grid
            # make a table later
            index = idx * args.batch_size
            for i in range(input_seq.size()[0]):
                input_seq_display = de_normalize(vutils.make_grid(input_seq.transpose(2, 3).contiguous()[
                    i, :, ::2, :, :, :].squeeze()))
                input_seq_display = input_seq_display.cpu()
                index_image = i + index
                index_name = 'image_' + str(index_image) + '.jpg'
                vutils.save_image(input_seq_display, os.path.join(
                    'image_grids_epic', index_name))

            output = model(input_seq)
            output_embed = output.squeeze()
            # target_embed = target.squeeze()
            # target_embed = target_embed.reshape(-1)

            frame_min_embed, frame_max_embed = frame_first_last
            frame_min_embed, frame_max_embed = frame_min_embed.cpu(), frame_max_embed.cpu()

            # concatenate all pertaining to current batch
            input_embed_all = torch.cat(
                (input_embed_all, input_embed), 0)
            output_embed_all = torch.cat(
                (output_embed_all, output_embed), 0)
            # target_embed_all = torch.cat(
            #     (target_embed_all, target_embed), 0)
            frame_min_embed_all = torch.cat(
                (frame_min_embed_all, frame_min_embed), 0)
            frame_max_embed_all = torch.cat(
                (frame_max_embed_all, frame_max_embed), 0)

        # save the output embeddings and input sequence in a html table

    torch.save(input_embed_all, os.path.join(
        '.', 'GT_input_embed.pt'))
    torch.save(output_embed_all.detach().cpu(),
               os.path.join('.', 'GT_output_embed.pt'))
    # torch.save(target_embed_all.detach().cpu(),
    #            os.path.join('.', 'GT_target_embed.pt'))
    torch.save(frame_max_embed_all.detach().cpu(),
               os.path.join('.', 'GT_frame_max_embed.pt'))
    torch.save(frame_min_embed_all.detach().cpu(),
               os.path.join('.', 'GT_frame_min_embed.pt'))

    output_embed = torch.load(os.path.join('.', 'GT_output_embed.pt')).to(cuda)
    # target_embed = torch.load(os.path.join('.', 'GT_target_embed.pt')).to(cuda)
    frame_min_embed = torch.load(os.path.join(
        '.', 'GT_frame_min_embed.pt')).to(cuda)
    frame_max_embed = torch.load(os.path.join(
        '.', 'GT_frame_max_embed.pt')).to(cuda)

    adj = get_distance(output_embed)
    torch.save(adj, os.path.join('.', 'cosine_distance.pt'))

    # sort the cosine distances
    ngbrs = np.argsort(np.array(1 - adj), axis=1)
    ngbrs_value = np.sort(np.array(1 - adj), axis=1)

    # take argument of k-NN
    ngbrs = ngbrs[:, 0:args.num_neighbours + 1]
    ngbrs_value = ngbrs_value[:, 0:args.num_neighbours + 1]

    for i in range(adj.shape[0]):

        # cosine distance
        f.write("""<tr>""")
        f.write("""<td>""")
        f.write("""<center>""")
        s_no = str(i)
        f.write(s_no)
        f.write("""</center>""")
        f.write(""" </td> """)

        f.write("""<td>""")
        f.write("""<center>""")
        query_value = str(1 - ngbrs_value[i][0])
        f.write(query_value)
        f.write("""</center>""")
        f.write(""" </td> """)

        for j in range(args.num_neighbours):
            f.write("""<td>""")
            f.write("""<center>""")
            probe_value = str(1 - ngbrs_value[i][j + 1])
            f.write(probe_value)
            f.write("""</center>""")
            f.write(""" </td> """)
        f.write("""</tr>""")

        # images
        f.write("""<tr>""")
        f.write("""<td>""")
        f.write("""<center>""")
        s_no = 'Clip'
        f.write(s_no)
        f.write("""</center>""")
        f.write(""" </td> """)

        f.write("""<td><img src= """)
        query_name = "image_" + str(i) + ".jpg"
        f.write(os.path.join('image_grids_epic', query_name))
        f.write(""" alt="" border=3 height=100 width=300></img></td> """)

        for j in range(args.num_neighbours):
            f.write("""<td><img src= """)
            probe_name = "image_" + str(ngbrs[i][j + 1]) + ".jpg"
            f.write(os.path.join('image_grids_epic', probe_name))
            f.write(""" alt="" border=3 height=100 width=300></img></td> """)
        f.write("""</tr>""")

        # image class
        f.write("""<tr>""")
        f.write("""<td>""")
        f.write("""<center>""")
        s_no = 'Class'
        f.write(s_no)
        f.write("""</center>""")
        f.write(""" </td> """)

        # f.write("""<td>""")
        # f.write("""<center>""")
        # query_target = str(get_class(target_embed.cpu().numpy()[i]))
        # f.write(query_target)
        # f.write("""</center>""")
        # f.write(""" </td> """)

        # for j in range(args.num_neighbours):
        #   f.write("""<td>""")
        #   f.write("""<center>""")
        #   probe_target = str(get_class(target_embed.cpu().numpy()[ngbrs[i][j+1]]))
        #   f.write(probe_target)
        #   f.write("""</center>""")
        #   f.write("""</td>""")
        # f.write("""</tr>""")

        # video file name
        f.write("""<tr>""")
        f.write("""<td>""")
        f.write("""<center>""")
        s_no = 'Filename'
        f.write(s_no)
        f.write("""</center>""")
        f.write(""" </td> """)

        f.write("""<td>""")
        f.write("""<center>""")
        query_filename = path_list[i]
        f.write(query_filename)
        f.write("""</center>""")
        f.write(""" </td> """)

        for j in range(args.num_neighbours):
            f.write("""<td>""")
            f.write("""<center>""")
            probe_filename = path_list[ngbrs[i][j + 1]]
            #probe_target = str(get_class(target_embed.cpu().numpy()[ngbrs[i][j+1]]))
            f.write(probe_filename)
            f.write("""</center>""")
            f.write("""</td>""")
        f.write("""</tr>""")

        # first/last frame
        f.write("""<tr>""")
        f.write("""<td>""")
        f.write("""<center>""")
        s_no = 'First/Last Frame'
        f.write(s_no)
        f.write("""</center>""")
        f.write(""" </td> """)

        f.write("""<td>""")
        f.write("""<center>""")
        query_frames = str(frame_min_embed[i].item(
        )) + '/' + str(frame_max_embed[i].item())
        f.write(query_frames)
        f.write("""</center>""")
        f.write(""" </td> """)

        for j in range(args.num_neighbours):
            f.write("""<td>""")
            f.write("""<center>""")
            probe_frames = str(frame_min_embed[ngbrs[i][j + 1]].item()) + '/' + str(
                frame_max_embed[ngbrs[i][j + 1]].item())
            #probe_target = str(get_class(target_embed.cpu().numpy()[ngbrs[i][j+1]]))
            f.write(probe_frames)
            f.write("""</center>""")
            f.write("""</td>""")
        f.write("""</tr>""")

    footer = """</tr>
                </table>
                </body>
                </html>"""
    f.write(footer)


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
    main()

    #x = torch.load(os.path.join('.', 'cosine_distance.pt'))
    #print(np.amax(x.numpy()), np.amin(x.numpy()), np.mean(x.numpy()), np.std(x.numpy()))

    # global args
    # args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # global cuda
    # cuda = torch.device('cuda')

    # transform = transforms.Compose([
    #     # centercrop
    #     CenterCrop(size=224),
    #     RandomSizedCrop(consistent=True, size=224, p=0.0),
    #     Scale(size=(args.img_dim, args.img_dim)),
    #     ToTensor(),
    #     Normalize()
    # ])

    # data_loader_pred = get_data(transform, 'val')
    # with torch.no_grad():
    #     for idx, (input_seq, target, vpath, frames_first_last) in enumerate(data_loader_pred):
    #       print (idx, frames_first_last)
    #       break
