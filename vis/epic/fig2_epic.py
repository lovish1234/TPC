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


# 5 pred 1
parser = argparse.ArgumentParser()

parser.add_argument(
    '--test-dir', default='/proj/vondrick/lovish/DPC/dpc/log_tmp/epic-128_r18_dpc-rnn_bs64_lr0.001_seq8_pred3_len5_ds6_train-all', type=str)
parser.add_argument('--num_seq', default=5, type=int)
parser.add_argument('--seq_len', default=5, type=int)
parser.add_argument('--ds', default=6, type=int,
                    help='frame downsampling rate')

parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument('--gpu', default='4,5,6,7', type=str)
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


def get_data(transform, mode='train', num_seq=1):
    print('[embeddings.py] Loading data for "%s" ...' % mode)
    global dataset
    if args.dataset == 'ucf101':
        dataset = UCF101_3d(mode=mode,
                            transform=transform,
                            seq_len=args.seq_len,
                            num_seq=num_seq,
                            downsample=args.ds,
                            which_split=args.split)
    elif args.dataset == 'hmdb51':
        dataset = HMDB51_3d(mode=mode,
                            transform=transform,
                            seq_len=args.seq_len,
                            num_seq=num_seq,
                            downsample=args.ds,
                            which_split=args.split)
    elif args.dataset == 'ucf11':
        # no split here
        dataset = UCF11_3d(mode=mode,
                           transform=transform,
                           num_seq=num_seq,
                           seq_len=args.seq_len,
                           downsample=args.ds)
    elif args.dataset == 'epic':
        dataset = epic_gulp(mode=mode,
                            transform=transform,
                            num_seq=num_seq,
                            seq_len=args.seq_len,
                            downsample=args.ds,
                            class_type='verb+noun')
    else:
        raise ValueError('dataset not supported')

    sampler = data.RandomSampler(dataset)

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
                            feature_type=feature_type,
                            distance_type=args.distance_type)
    model = model.to(cuda)
    model.eval()

    model = neq_load_customized(model, checkpoint['state_dict'])
    model = nn.DataParallel(model)
    return model


def get_distance(embedding, future):

    output_embed_1 = embedding
    output_embed_2 = future

    if args.distance == 'cosine':

        distance_f = nn.CosineSimilarity(dim=0, eps=1e-6)

        adj = torch.zeros(output_embed_1.shape[0], output_embed_2.shape[0])
        for i in tqdm(range(output_embed_1.shape[0])):
            for j in range(output_embed_2.shape[0]):
                a = output_embed_1[i].reshape(-1)
                b = output_embed_2[j].reshape(-1)
                adj[i, j] = distance_f(a, b)

    return adj


def get_class(class_id):

    # get action list [here just the values of y]
    action_dict_decode = {}

    action_file = os.path.join(
        '/proj/vondrick/lovish/data/ucf101', 'classInd.txt')
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
    data_loader_pred = get_data(transform, 'val', num_seq=args.num_seq)
    model_pred = get_model(feature_type='Phi')

    input_embed_all = torch.Tensor()
    output_embed_all = torch.Tensor().to(cuda)
    target_embed_all = torch.LongTensor()
    frame_min_embed_all = torch.LongTensor()
    frame_max_embed_all = torch.LongTensor()
    path_list_pred = []

    noun_list_pred = []
    verb_list_pred = []
    participant_list_pred = []
    video_list_pred = []

    f = open('fig2_epic.html', 'w')
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
        for idx, (input_seq, participant, video, action, noun, frame_first_last) in enumerate(data_loader_pred):

            input_embed = torch.Tensor()
            input_embed = torch.cat((input_embed, input_seq.permute(
                0, 1, 3, 2, 4, 5)[:, :, -1, :, :, :]), 0)

            path_list_pred.extend(vpath)

            # save the files in a grid
            # make a table later
            index = idx * args.batch_size
            for i in range(input_seq.size()[0]):
                # print (input_seq.shape)
                # print (input_seq.transpose(2, 3).contiguous()[
                #     i, ::2, 2, :, :, :].shape)
                input_seq_display = de_normalize(vutils.make_grid(input_seq.transpose(2, 3).contiguous()[
                    i, ::2, 2, :, :, :].squeeze()))
                input_seq_display = input_seq_display.cpu()
                index_image = i + index
                index_name = 'image_' + str(index_image) + '.jpg'
                vutils.save_image(input_seq_display, os.path.join(
                    'image_grids2_epic_pred', index_name))

            output = model_pred(input_seq)
            output_embed = output.squeeze()
            # target_embed = target.squeeze()
            # target_embed = target_embed.reshape(-1)

            noun_embed = noun.squeeze()
            noun_embed = noun_embed.reshape(-1).cpu().tolist()

            verb_embed = action.squeeze()
            verb_embed = verb_embed.reshape(-1).cpu().tolist()

            # participant_embed = participant.squeeze()
            # participant_embed = participant_embed.reshape(-1).cpu()

            # video_embed = video.squeeze()
            # video_embed = video_embed.reshape(-1).cpu()

            noun_embed = [class_to_noun(x) for x in noun_embed]
            verb_embed = [class_to_verb(x) for x in verb_embed]

            #print (noun_embed, verb_embed, type(noun_embed), type(verb_embed))

            noun_list_pred.extend(noun_embed)
            verb_list_pred.extend(verb_embed)
            participant_list_pred.extend(participant)
            video_list_pred.extend(video)

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

            break
        # save the output embeddings and input sequence in a html table

    torch.save(input_embed_all, os.path.join(
        '.', 'pred_input_embed.pt'))
    torch.save(output_embed_all.detach().cpu(),
               os.path.join('.', 'pred_output_embed.pt'))
    # torch.save(target_embed_all.detach().cpu(),
    #            os.path.join('.', 'pred_target_embed.pt'))
    torch.save(frame_max_embed_all.detach().cpu(),
               os.path.join('.', 'pred_frame_max_embed.pt'))
    torch.save(frame_min_embed_all.detach().cpu(),
               os.path.join('.', 'pred_frame_min_embed.pt'))

    output_embed_pred = torch.load(os.path.join(
        '.', 'pred_output_embed.pt')).to(cuda)
    target_embed_pred = torch.load(os.path.join(
        '.', 'pred_target_embed.pt')).to(cuda)
    frame_min_embed_pred = torch.load(os.path.join(
        '.', 'pred_frame_min_embed.pt')).to(cuda)
    frame_max_embed_pred = torch.load(os.path.join(
        '.', 'pred_frame_max_embed.pt')).to(cuda)

    # the NN of a particular sequence should be the same sequence itself
    data_loader_GT = get_data(transform, 'val', num_seq=1)
    model_GT = get_model(feature_type='F')

    input_embed_all = torch.Tensor()
    output_embed_all = torch.Tensor().to(cuda)
    target_embed_all = torch.LongTensor()
    frame_min_embed_all = torch.LongTensor()
    frame_max_embed_all = torch.LongTensor()
    path_list_GT = []

    noun_list_GT = []
    verb_list_GT = []
    participant_list_GT = []
    video_list_GT = []

    with torch.no_grad():
        for idx, (input_seq, vpath, frame_first_last) in enumerate(data_loader_GT):
            input_embed = torch.Tensor()
            input_embed = torch.cat((input_embed, input_seq.permute(
                0, 1, 3, 2, 4, 5)[:, :, -1, :, :, :]), 0)

            path_list_GT.extend(vpath)

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
                    'image_grids2_epic_GT', index_name))

            output = model_GT(input_seq)
            output_embed = output.squeeze()
            # target_embed = target.squeeze()
            # target_embed = target_embed.reshape(-1)

            noun_embed = noun.squeeze()
            noun_embed = noun_embed.reshape(-1).cpu().tolist()

            verb_embed = action.squeeze()
            verb_embed = verb_embed.reshape(-1).cpu().tolist()

            # participant_embed = participant.squeeze()
            # participant_embed = participant_embed.reshape(-1).cpu()

            # video_embed = video.squeeze()
            # video_embed = video_embed.reshape(-1).cpu()

            noun_embed = [class_to_noun(x) for x in noun_embed]
            verb_embed = [class_to_verb(x) for x in verb_embed]

            #print (noun_embed, verb_embed, type(noun_embed), type(verb_embed))

            noun_list_GT.extend(noun_embed)
            verb_list_GT.extend(verb_embed)
            participant_list_GT.extend(participant)
            video_list_GT.extend(video)

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

            break
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

    output_embed_GT = torch.load(os.path.join(
        '.', 'GT_output_embed.pt')).to(cuda)
    target_embed_GT = torch.load(os.path.join(
        '.', 'GT_target_embed.pt')).to(cuda)
    frame_min_embed_GT = torch.load(os.path.join(
        '.', 'GT_frame_min_embed.pt')).to(cuda)
    frame_max_embed_GT = torch.load(os.path.join(
        '.', 'GT_frame_max_embed.pt')).to(cuda)

    adj = get_distance(output_embed_pred, output_embed_GT)
    torch.save(adj, os.path.join('.', 'cosine_distance_2.pt'))

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
        f.write(os.path.join('image_grids2_epic_pred', query_name))
        f.write(""" alt="" border=3 height=100 width=300></img></td> """)

        for j in range(args.num_neighbours):
            f.write("""<td><img src= """)
            probe_name = "image_" + str(ngbrs[i][j + 1]) + ".jpg"
            f.write(os.path.join('image_grids2_epic_GT', probe_name))
            f.write(""" alt="" border=3 height=100 width=300></img></td> """)
        f.write("""</tr>""")

        # noun class
        f.write("""<tr>""")
        f.write("""<td>""")
        f.write("""<center>""")
        s_no = ' Noun Class '
        f.write(s_no)
        f.write("""</center>""")
        f.write(""" </td> """)

        f.write("""<td>""")
        f.write("""<center>""")
        #print (type(noun_list[i]), noun_list[i])
        query_noun = noun_list_pred[i]
        f.write(query_noun)
        f.write("""</center>""")
        f.write(""" </td> """)

        for j in range(args.num_neighbours):
            f.write("""<td>""")
            f.write("""<center>""")
            probe_noun = noun_list_GT[ngbrs[i][j + 1]]
            f.write(probe_noun)
            f.write("""</center>""")
            f.write("""</td>""")
        f.write("""</tr>""")

        # verb class
        f.write("""<tr>""")
        f.write("""<td>""")
        f.write("""<center>""")
        s_no = ' Verb Class '
        f.write(s_no)
        f.write("""</center>""")
        f.write(""" </td> """)

        f.write("""<td>""")
        f.write("""<center>""")
        query_verb = verb_list_pred[i]
        f.write(query_verb)
        f.write("""</center>""")
        f.write(""" </td> """)

        for j in range(args.num_neighbours):
            f.write("""<td>""")
            f.write("""<center>""")
            probe_verb = verb_list_GT[ngbrs[i][j + 1]]
            f.write(probe_verb)
            f.write("""</center>""")
            f.write("""</td>""")
        f.write("""</tr>""")

        # participant ID
        f.write("""<tr>""")
        f.write("""<td>""")
        f.write("""<center>""")
        s_no = 'Participant ID'
        f.write(s_no)
        f.write("""</center>""")
        f.write(""" </td> """)

        f.write("""<td>""")
        f.write("""<center>""")
        query_participant = participant_list_pred[i]
        #print (participant_list[i], type(participant_list[i]))
        f.write(query_participant)
        f.write("""</center>""")
        f.write(""" </td> """)

        for j in range(args.num_neighbours):
            f.write("""<td>""")
            f.write("""<center>""")
            probe_participant = participant_list_GT[ngbrs[i][j + 1]]
            #probe_target = str(get_class(target_embed.cpu().numpy()[ngbrs[i][j+1]]))
            f.write(probe_participant)
            f.write("""</center>""")
            f.write("""</td>""")
        f.write("""</tr>""")

        # video ID
        f.write("""<tr>""")
        f.write("""<td>""")
        f.write("""<center>""")
        s_no = 'Video ID'
        f.write(s_no)
        f.write("""</center>""")
        f.write(""" </td> """)

        f.write("""<td>""")
        f.write("""<center>""")
        query_video = video_list_pred[i]
        f.write(query_video)
        f.write("""</center>""")
        f.write(""" </td> """)

        for j in range(args.num_neighbours):
            f.write("""<td>""")
            f.write("""<center>""")
            probe_video = video_list_GT[ngbrs[i][j + 1]]
            #probe_target = str(get_class(target_embed.cpu().numpy()[ngbrs[i][j+1]]))
            f.write(probe_video)
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
        query_frames = str(frame_min_embed_pred[i].item(
        )) + '/' + str(frame_max_embed_pred[i].item())
        f.write(query_frames)
        f.write("""</center>""")
        f.write(""" </td> """)

        for j in range(args.num_neighbours):
            f.write("""<td>""")
            f.write("""<center>""")
            probe_frames = str(frame_min_embed_GT[ngbrs[i][j + 1]].item()) + '/' + str(
                frame_max_embed_GT[ngbrs[i][j + 1]].item())
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
