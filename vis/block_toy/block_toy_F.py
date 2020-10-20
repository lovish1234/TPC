import sys

sys.path.append('../utils')
sys.path.append('../../tpc-backbone')
from model_visualize import *

sys.path.append('../../tpc')
sys.path.append('../../utils')

from dataset_3d import *
from utils import denorm


from augmentation import *
from utils import denorm


import argparse
import os

from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt

# saving log, calculating accuracy etc.
import torchvision.utils as vutils
from epic_kitchens.meta import class_to_noun, class_to_verb


parser = argparse.ArgumentParser()

parser.add_argument(
    '--test-dir', default='/proj/vondrick/ruoshi/github/TPC/tpc/log_tmp/4-15/block_toy_1-128_r34_dpc-rnn_bs64_lr0.001_seq4_pred1_len5_ds3_train-all_distance_L2_distance-type_uncertain_margin_10.0', type=str)
parser.add_argument(
    '--model-name', default='model_best_epoch947.pth.tar', type=str)
parser.add_argument('--num_seq', default=1, type=int)
parser.add_argument('--seq_len', default=5, type=int)
parser.add_argument('--ds', default=3, type=int,
                    help='frame downsampling rate')

parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument('--gpu', default='0,1,2,3', type=str)
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
parser.add_argument('--drive', default='ssd', type=str)
parser.add_argument('--num_sample', default=5, type=int)


def get_data(transform, mode='train'):
    print('[block_toy_F.py] Loading data for "%s" ...' % mode)
#     global dataset
    print(args.dataset)
    if args.dataset == 'block_toy':
        dataset = block_toy(mode=mode,
                            num=args.split,
                            transform=transform,
                            seq_len=args.seq_len,
                            num_seq=args.num_seq,
                            downsample=args.ds,
                            drive=args.drive,
                            num_sample=args.num_sample)
    elif args.dataset == 'ucf101':
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
        dataset = epic_gulp(mode=mode,
                            transform=transform,
                            num_seq=args.num_seq,
                            seq_len=args.seq_len,
                            downsample=args.ds,
                            class_type='verb+noun',
                            drive=args.drive,
                            num_sample=args.num_sample)
    else:
        raise ValueError('dataset not supported')

    sampler = data.RandomSampler(dataset)

    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  sampler=sampler,
                                  shuffle=False,
                                  num_workers=8,
                                  pin_memory=True,
                                  drop_last=False)

    print('"%s" dataset size: %d' % (mode, len(dataset)))
    print('length of dataloader:', len(data_loader))
    return data_loader


def get_model(feature_type='F'):

    # extract the files corresponding to each epochs
    model_path = os.path.join(args.test_dir, "model")

    # use original DPC weights
    model_filename = args.model_name
    model_file = os.path.join(model_path, model_filename)

    print('loading model from: ' + model_file)
    try:
        checkpoint = torch.load(model_file)
    except:
        print('model failed to load')
        return

    # initialize model architecture
    model = model_visualize(sample_size=args.img_dim,
                            seq_len=args.seq_len,
                            pred_steps=1,
                            network=args.net,
                            feature_type='F',
                            distance_type=args.distance_type)
    model = model.to(cuda)
    model.eval()

    # initialize model weights
    model = neq_load_customized(model, checkpoint['state_dict'])
    model = nn.DataParallel(model)
    return model


def get_distance(pred_embedding, gt_embedding, distance):
    if distance == 'dot':
        gt_embedding = gt_embedding.transpose(0, 1)
        score = torch.matmul(pred_embedding, gt_embedding)
        # print(score)
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
        pred_embedding_mult = pred_embedding.reshape(
            pred_embedding.shape[0], 1, pred_embedding.shape[1])
        difference = pred_embedding_mult - gt_embedding
        score = torch.sqrt(torch.einsum(
            'ijk,ijk->ij', difference, difference))
    return score


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

    # get dataset object
    data_loader_pred = get_data(transform, mode='test')

    # load and initialize model
    model = get_model(feature_type='F')

    # initialize temporary tensor object to be plotted
    input_embed_all = torch.Tensor()
    output_embed_all = torch.Tensor().to(cuda)
    radius_embed_all = torch.Tensor().to(cuda)
    frame_min_embed_all = torch.LongTensor()
    frame_max_embed_all = torch.LongTensor()

    vpath_list = []
    idx_block_list = []
#     noun_list = []
#     verb_list = []
#     participant_list = []
#     video_list = []

    f = open('block_toy_F.html', 'w')
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
        for idx, input_dict in tqdm(enumerate(data_loader_pred), total=len(data_loader_pred)):

            #print ("Here")

            input_embed = torch.Tensor()
            input_seq = input_dict['t_seq']
            vpath = input_dict['vpath']
            idx_block = input_dict['idx_block']
            input_embed = torch.cat((input_embed, input_seq.permute(
                0, 1, 3, 2, 4, 5)[:, :, -1, :, :, :]), 0)
            index = idx * args.batch_size
            for i in range(len(input_seq)):
                new_im = Image.new('RGB', (3 * args.img_dim, args.img_dim))
                x_offset = 0
                for j in range(3):
                    image_dir = os.path.join(
                        vpath[i], 'image_%0.5d.jpg' % idx_block[i][2 * j].item())
                    im = Image.open(image_dir)
                    im = im.resize([args.img_dim, args.img_dim])

                    new_im.paste(im, (x_offset, 0))
                    x_offset += im.size[0]
                index_image = i + index
                index_name = 'image_' + str(index_image) + '.jpg'
                new_im.save(os.path.join('image_grids', index_name))
#                 index += len(input_seq)

            output = model(input_seq)
            output_embed = output.squeeze()

            vpath_list.extend(vpath)
            idx_block_list.extend(idx_block)

            # concatenate all pertaining to current batch
            input_embed_all = torch.cat(
                (input_embed_all, input_embed), 0)
            output_embed_all = torch.cat(
                (output_embed_all, output_embed), 0)
            # break
        # save the output embeddings and input sequence in a html table

    if not os.path.exists('cache'):
        os.makedirs('cache')

    torch.save(input_embed_all, os.path.join(
        '.', 'cache/GT_input_embed.pt'))
    torch.save(output_embed_all.detach().cpu(),
               os.path.join('.', 'cache/GT_output_embed.pt'))

    output_embed = torch.load(os.path.join(
        'cache', 'GT_output_embed.pt')).to(cuda)

    adj = get_distance(output_embed, output_embed, args.distance).cpu()
    torch.save(adj, os.path.join('.', 'cache/%s_distance.pt' % args.distance))

    # sort the distances
    if args.distance == 'cosine':
        adj = 1 - adj
    ngbrs = np.argsort(np.array(adj), axis=1)
    ngbrs_value = np.sort(np.array(adj), axis=1)

    # take argument of k-NN
    ngbrs = ngbrs[:, 0:args.num_neighbours + 1]
    ngbrs_value = ngbrs_value[:, 0:args.num_neighbours + 1]

    for i in range(adj.shape[0]):

        # f.write("""<hr>""")
        # f.write("""<hr>""")

        # distance
        f.write("""<tr>""")
        f.write("""<td>""")
        f.write("""<center>""")
        s_no = str(i)
        f.write(s_no)
        f.write("""</center>""")
        f.write(""" </td> """)
        f.write("""</tr>""")

        f.write("""<tr>""")
        f.write("""<td>""")
        f.write("""<center>""")
        s_no = '%s distance' % args.distance
        f.write(s_no)
        f.write("""</center>""")
        f.write(""" </td> """)

        f.write("""<td>""")
        f.write("""<center>""")
        query_value = str(ngbrs_value[i][0])
        f.write(query_value)
        f.write("""</center>""")
        f.write(""" </td> """)

        for j in range(args.num_neighbours):
            f.write("""<td>""")
            f.write("""<center>""")
            probe_value = str(ngbrs_value[i][j + 1])
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
        f.write(os.path.join('image_grids', query_name))
        f.write(""" alt="" border=3 height=100 width=300></img></td> """)

        for j in range(args.num_neighbours):
            f.write("""<td><img src= """)
            probe_name = "image_" + str(ngbrs[i][j + 1]) + ".jpg"
            print(probe_name)
            f.write(os.path.join('image_grids', probe_name))
            f.write(""" alt="" border=3 height=100 width=300></img></td> """)
        f.write("""</tr>""")

        # vpath
        f.write("""<tr>""")
        f.write("""<td>""")
        f.write("""<center>""")
        s_no = 'Block Position'
        f.write(s_no)
        f.write("""</center>""")
        f.write(""" </td> """)

        f.write("""<td>""")
        f.write("""<center>""")
        query_frames = vpath_list[i].split('/')[-3]
        f.write(query_frames)
        f.write("""</center>""")
        f.write(""" </td> """)

        for j in range(args.num_neighbours):
            f.write("""<td>""")
            f.write("""<center>""")
            probe_frames = vpath_list[ngbrs[i][j + 1]].split('/')[-3]
            #probe_target = str(get_class(target_embed.cpu().numpy()[ngbrs[i][j+1]]))
            f.write(probe_frames)
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
        query_frames = str(idx_block_list[i][0].item(
        )) + '/' + str(idx_block_list[i][-1].item())
        f.write(query_frames)
        f.write("""</center>""")
        f.write(""" </td> """)

        for j in range(args.num_neighbours):
            f.write("""<td>""")
            f.write("""<center>""")
            probe_frames = str(idx_block_list[ngbrs[i][j + 1]][0].item()) + '/' + str(
                idx_block_list[ngbrs[i][j + 1]][-1].item())
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
