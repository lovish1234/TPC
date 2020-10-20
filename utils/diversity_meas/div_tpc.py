# April 2020
# TPC uncertainty

from div_common import *  # contains imports


def measure_tpc_uncertainty_loader(data_loader, model, print_freq):
    '''
    Calculates the diversity among multiple predictions for every sequence in the given dataloader.
    Returns a dictionary mapping every item of the dataloader to a set of uncertainty values,
    as measured over multiple time steps and by different metrics (i.e. L2, cosine, normalized L2).
    Example output = {('P12_08', 1234): [[4.6277223  2.4677615  2.546618  ]
                                         [0.3095014  0.14996651 0.15751362]
                                         [0.9995848  0.9991826  0.9988366 ]
                                         [0.99996156 0.99993753 0.99990815]], ...}.
    Here, the horizontal index is time step, and the vertical index is distance metric
    (L2 / L2 after pooling / cosine / cosine after pooling / normalized L2 / normalized L2 after pooling).
    '''
    cuda = torch.device('cuda')

    # Store useful values for every sample in a list
    results = dict()
    model.eval()

    with torch.no_grad():
        for idx, input_dict in tqdm(enumerate(data_loader)):

            input_seq = input_dict['t_seq'].to(cuda)
            idx_block = input_dict['idx_block']
            vpath = input_dict['vpath']
            B = input_seq.size(0)

#             print(idx_block)
#             print(vpath)
#             print(B)

            output = model(input_seq)
            output_embed = output.squeeze()
            radius_embed = output_embed[:, -1].expand(1, -1)

            del input_seq

            radius_embed = radius_embed.squeeze().detach().cpu().numpy()

            if idx % print_freq == 0:
                print('Idx: [{} / {}]   radius {:.2f}'.format(
                    idx, len(data_loader), radius_embed[idx]))

            # Keep track of diversity metrics in dictionary; key identifies video and start frame
            for j in range(B):
                cur_path = vpath[j]
                cur_start = idx_block[j, 0].item()
                cur_key = (ntpath.basename(cur_path), cur_start)
                cur_divers = radius_embed[j]
                results[cur_key] = cur_divers
                if idx == 9 and j == 0:
                    print('batch 10, first item:', cur_key, cur_divers)
                elif idx == 9 and j == B - 1:
                    print('batch 10, last item:', cur_key, cur_divers)

    print('Done!')
    return results


def measure_tpc_uncertainty_video(pretrained_path, video_path, results_dir, net_name='resnet34', cvae_arch='conv',
                                  gpus_str='0,1', batch_size=64, start_stride=2, num_seq=8, seq_len=5, downsample=6,
                                  img_dim=128, pred_step=3, margin=10):
    '''
    Calculates the diversity among multiple predictions, sweeping over the video file and
    densely extracting all sequences in a moving window fashion. The returned dictionary follows the
    same format as measure_uncertainty_loader(), although the key contains the video file name (not the full path).
    NOTE: The results start 2.5 seconds into the video and stop 1.5 seconds before the end due to how DPC works.
    TODO: margin argument is unused?
    '''

    # If available, read cached results using pickle
    results_path = os.path.join(results_dir, 'tpc_results.p')
    if results_path is not None and os.path.exists(results_path):
        print('Loading calculated results from ' + results_path + '...')
        with open(results_path, 'rb') as f:
            return pickle.load(f)

    # Set to constant for consistent results
    torch.manual_seed(0)
    np.random.seed(0)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_str
    global cuda
    cuda = torch.device('cuda')

    ### dpc model ###
#     model = DPC_CVAE(sample_size=img_dim,
#                      num_seq=num_seq,
#                      seq_len=seq_len,
#                      network=net_name,
#                      pred_step=pred_step,
#                      cvae_arch=cvae_arch)

    model = model_visualize(sample_size=img_dim,
                            seq_len=seq_len,
                            pred_steps=pred_step,
                            network=net_name,
                            feature_type='Phi',
                            distance_type='uncertain',
                            radius_location='Phi')

    # parallelize the model
    model = nn.DataParallel(model)
    model = model.to(cuda)

    # Load pretrained model
    if os.path.isfile(pretrained_path):
        print("=> loading pretrained checkpoint '{}'".format(pretrained_path))
        checkpoint = torch.load(
            pretrained_path, map_location=torch.device('cpu'))

        # neq_load_customized
        model = neq_load_customized(model, checkpoint['state_dict'])
        print("=> loaded pretrained checkpoint '{}' (epoch {})"
              .format(pretrained_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(pretrained_path))

    ### load data ###
    # Uncertainty evaluation: center crop & no color adjustments
    tf_diversity = transforms.Compose([
        RandomHorizontalFlip(consistent=True),
        CenterCrop(size=224),
        Scale(size=(img_dim, img_dim)),
        ToTensor(),
        Normalize()
    ])

    # Note: avoid all shuffling of the data
    # TODO: drop_last must be False to avoid skipping the last bunch of frames; will this not introduce bugs?
    dataset = SingleVideoDataset(video_path, transform=tf_diversity, start_stride=start_stride,
                                 num_seq=num_seq, seq_len=seq_len, downsample=downsample)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=16, pin_memory=True, drop_last=False)
    print('Dataset size:', len(dataset), ' loader size', len(loader))

    # Apply model
    print('Measuring diversity of ' + video_path + '...')
    results = measure_tpc_uncertainty_loader(loader, model, 10)

    # If desired, store results using pickle
    if results_path is not None:
        print('Storing calculated results to ' + results_path + '...')
        if not(os.path.exists(Path(results_path).parent)):
            os.makedirs(Path(results_path).parent)
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)

    return results


def measure_tpc_uncertainty_epic(pretrained_path, kitchen_id, start_frame, results_dir, net_name='resnet34',
                                 cvae_arch='conv', gpus_str='0,1', batch_size=64, start_stride=2, num_seq=8, seq_len=5,
                                 downsample=6, img_dim=128, pred_step=3, margin=10):
    '''
    Calculates the diversity among multiple predictions within the given Epic Kitchens sequence.
    The returned dictionary follows the same format as measure_uncertainty_loader(), although
    the key contains the participant ID (not the video file name).
    '''
    # First convert EPIC sequence from set of images to video
    video_path = 'tmp.mp4'
    convert_epic_video(participant_id, start_frame, video_path)

    # Then forward to existing method
    return measure_tpc_uncertainty_video(pretrained_path, video_path, results_dir, net_name=net_name, cvae_arch=cvae_arch,
                                         gpus_str=gpus_str, batch_size=batch_size, start_stride=start_stride, num_seq=num_seq, seq_len=seq_len,
                                         downsample=downsample, img_dim=img_dim, pred_step=pred_step, margin=margin)
