# April 2020
# CVAE / VRNN uncertainty

from div_common import *  # contains imports


def measure_vae_uncertainty_loader(data_loader, model, paths, print_freq=10,
                                   collect_actions=False, collect_embeddings=False,
                                   force_context_dropout=False, spatial_separation=False):
    '''
    Calculates the diversity among multiple predictions for every sequence in the given dataloader.
    Returns a dictionary mapping every item of the dataloader to a set of uncertainty values,
    as measured over multiple time steps and by different metrics (i.e. L2, cosine, normalized L2).
    Example output = {('P12_08', 1234): {'mean_var': [[4.6277223  2.4677615  2.546618  ]
                                                      [0.3095014  0.14996651 0.15751362]
                                                      [0.9995848  0.9991826  0.9988366 ]
                                                      [0.99996156 0.99993753 0.99990815]], ...}, ...}.
    Here, the horizontal index is time step, and the vertical index is distance metric:
    (L2 / L2 after pooling / normalized L2 / normalized L2 after pooling / cosine / cosine after pooling),
    first 6 = predictions, last 6 = hidden states.
    paths: Number of future embeddings w_t+1 to generate from c_t.
    collect_actions: If True, also gather predicted actions for every sample (head must be present).
    collect_embeddings: If True, also gether embeddings for predictions and contexts for every sample (WARNING: huge file size).
    force_context_dropout: If True, every path will proceed with different dropout mask (p = 0.1) for c_t.
    spatial_separation: Whether to avoid averaging over H, W when calculating mean distance metrics.
    '''
    cuda = torch.device('cuda')

    # Store useful values for every sample in a list
    results = dict()
    model.eval()

    with torch.no_grad():
        for idx, mbatch in tqdm(enumerate(data_loader)):
            input_seq = mbatch['t_seq'].to(cuda)
            idx_block = mbatch['idx_block']
            vpath = mbatch['vpath'] if 'vpath' in mbatch else mbatch['video_id']
            B = input_seq.size(0)

            # Forward and obtain relevant variables
            model_res = model(input_seq, diversity=True, paths=paths,
                              force_rnn_dropout=force_context_dropout, spatial_separation=spatial_separation)
            mean_var = model_res['mean_var'].cpu().numpy()
            # (batch index, metric kind, future step)
            if collect_actions:
                action_outputs = [x.cpu().numpy() for x in model_res['actions']]
                # (class type within list, batch index, path, future step, class) => logit
            if collect_embeddings:
                all_preds = model_res['preds'].cpu().numpy()
                # (batch index, path, future step, F, S, S)
                all_contexts = model_res['contexts'].cpu().numpy()
                # (batch index, path, future step, F, S, S)
            del input_seq
            
            # Print stats
            if idx % print_freq == 0:
                if spatial_separation:
                    mean_mean = mean_var.mean(axis=[3, 4])
                    print('Idx: [{} / {}]  {}  L2 pred ({:.2f}, {:.2f}, {:.2f}) L2 context ({:.2f}, {:.2f}, {:.2f})'.format(
                        idx, len(data_loader), vpath[0], mean_mean[0, 0, 0], mean_mean[0, 0, 1], mean_mean[0, 0, 2],
                                                         mean_mean[0, 3, 0], mean_mean[0, 3, 1], mean_mean[0, 3, 2]))
                    print('norm-L2 pred ({:.2f}, {:.2f}, {:.2f}) norm-L2 context ({:.2f}, {:.2f}, {:.2f})'.format(
                        mean_mean[0, 1, 0], mean_mean[0, 1, 1], mean_mean[0, 1, 2],
                        mean_mean[0, 4, 0], mean_mean[0, 4, 1], mean_mean[0, 4, 2]))
                else:
                    print('Idx: [{} / {}]  {}  L2 pred ({:.2f}, {:.2f}, {:.2f}) L2 context ({:.2f}, {:.2f}, {:.2f})'.format(
                        idx, len(data_loader), vpath[0], mean_var[0, 0, 0], mean_var[0, 0, 1], mean_var[0, 0, 2],
                                                         mean_var[0, 6, 0], mean_var[0, 6, 1], mean_var[0, 6, 2]))
                    print('norm-L2 pred ({:.2f}, {:.2f}, {:.2f}) norm-L2 context ({:.2f}, {:.2f}, {:.2f})'.format(
                        mean_var[0, 1, 0], mean_var[0, 1, 1], mean_var[0, 1, 2],
                        mean_var[0, 7, 0], mean_var[0, 7, 1], mean_var[0, 7, 2]))

            # Keep track of diversity metrics in dictionary; key identifies one video and start frame.
            # Value is itself a dictionary with keys 'mean_var', 'actions', 'preds', and 'contexts'.
            for j in range(B):
                cur_path = vpath[j]
                cur_start = idx_block[j, 0].item()
                cur_key = (ntpath.basename(cur_path), cur_start)
                cur_info = dict()
                cur_info['mean_var'] = mean_var[j] # (metric kind, future step)
                if collect_actions:
                    cur_info['actions'] = [x[j] for x in action_outputs] # (class type within list, path, future step, class) => logit
                if collect_embeddings:
                    cur_info['preds'] = all_preds[j] # (path, future step, F, S, S)
                    cur_info['contexts'] = all_contexts[j] # (path, future step, F, S, S)
                results[cur_key] = cur_info

                # Print example video info
                if idx == 5 and (j == 0 or j == B-1):
                    print('batch 10, one item:', cur_key, '-> mean_var ->', cur_info['mean_var'])
                    if collect_actions:
                        print('action outputs:', cur_key, '-> actions ->', [x.argmax(axis=2) for x in cur_info['actions']])

    print('Done!')
    return results


def measure_vae_uncertainty_video(pretrained_path, video_path, results_dir, model_type,
                                  net_name='resnet34', cvae_arch='conv', gpus_str='0,1',
                                  vrnn_latent_size=8, vrnn_kernel_size=1, vrnn_dropout=0.1,
                                  batch_size=64, start_stride=2, num_seq=8, seq_len=5, downsample=6,
                                  img_dim=128, pred_step=3, paths=20,
                                  collect_actions=False, collect_embeddings=False,
                                  force_context_dropout=False, spatial_separation=False):
    '''
    Calculates the diversity among multiple predictions, sweeping over the video file and
    densely extracting all sequences in a moving window fashion. The returned dictionary follows the
    same format as measure_vae_uncertainty_loader(), although the key contains the video file name (not the full path).
    NOTE: The results start 2.5 seconds into the video and stop 1.5 seconds before the end due to how DPC works.
    model_type: cvae / vrnn.
    start_stride: Frame start number increment within the full video for every sequence.
        Example: if start_stride=2 and downsample=3, we first process frames (0, 3, 6, ..., 117) for item index = 0
        and then (2, 5, 8, ..., 119) for item index = 1 and so on.
    (Other arguments: see above method.)
    '''

    # If available, read cached results using pickle
    results_path = os.path.join(results_dir, model_type + '_results.p')
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

    if model_type == 'cvae':
        model = DPC_CVAE(img_dim=img_dim,
                         num_seq=num_seq,
                         seq_len=seq_len,
                         pred_step=pred_step,
                         network=net_name,
                         cvae_arch=cvae_arch,
                         action_cls_head=collect_actions,
                         num_class=[125, 352])
    elif 'vrnn' in model_type:
        model = DPC_VRNN(img_dim=img_dim,
                         num_seq=num_seq,
                         seq_len=seq_len,
                         pred_step=pred_step,
                         network=net_name,
                         latent_size=vrnn_latent_size,
                         kernel_size=vrnn_kernel_size,
                         rnn_dropout=vrnn_dropout,
                         action_cls_head='future' if collect_actions else None,
                         num_class=[125, 352],
                         time_indep='-i' in model_type)
    else:
        raise ValueError('Unknown model type: ' + model_type)

    # parallelize the model
    model = nn.DataParallel(model)
    model = model.to(cuda)

    # Load pretrained model
    # NOTE: carefully check the loaded weights when executing this, especially the action classification head!
    print("=> loading pretrained checkpoint '{}'".format(pretrained_path))
    checkpoint = torch.load(
        pretrained_path, map_location=torch.device('cpu'))
    model = neq_load_customized(model, checkpoint['state_dict'])
    print("=> loaded pretrained checkpoint '{}' (epoch {})"
            .format(pretrained_path, checkpoint['epoch']))

    ### load data ###
    # Uncertainty evaluation: center crop & no color adjustments
    tf_diversity = transforms.Compose([
        RandomHorizontalFlip(consistent=True),
        CenterCrop(size=224),
        Scale(size=(img_dim, img_dim)),
        ToTensor(),
        Normalize()
    ])

    # NOTE: avoid shuffling data here
    dataset = SingleVideoDataset(video_path, transform=tf_diversity, start_stride=start_stride,
                                 num_seq=num_seq, seq_len=seq_len, downsample=downsample)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=16, pin_memory=True, drop_last=False)
    print('Dataset size:', len(dataset), ' loader size', len(loader))

    # Apply model
    # Forward data into model
    print('Measuring diversity of ' + video_path + '...')
    print('collect_actions:', collect_actions, ' collect_embeddings:', collect_embeddings, ' force_context_dropout:', force_context_dropout)
    results = measure_vae_uncertainty_loader(loader, model, paths,
        collect_actions=collect_actions, collect_embeddings=collect_embeddings,
        force_context_dropout=force_context_dropout, spatial_separation=spatial_separation)

    # If desired, store results using pickle
    if results_path is not None:
        print('Storing calculated results to ' + results_path + '...')
        if not(os.path.exists(Path(results_path).parent)):
            os.makedirs(Path(results_path).parent)
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)

    return results


def measure_vae_uncertainty_epic(pretrained_path, kitchen_id, start_frame, results_dir, model_type, **kwargs):
    '''
    Calculates the diversity among multiple predictions within the given Epic Kitchens sequence.
    The returned dictionary follows the same format as measure_uncertainty_loader(), although
    the key contains the participant ID (not the video file name).
    '''
    # First convert EPIC sequence from set of images to video
    video_path = 'tmp.mp4'
    convert_epic_video(participant_id, start_frame, video_path)

    # Then forward to existing method
    return measure_vae_uncertainty_video(pretrained_path, video_path, results_dir, model_type, **kwargs)
