import torch
import os

from dataset_epic import *
from dataset_synthepic import *
from dataset_other import *
from dataset_toy import *

from resnet_2d3d import neq_load_customized
from torchvision import datasets, models, transforms

# data augmentation methods
from augmentation import *
from vae_main import *


def get_transform(args, mode='train'):
    if mode == 'train':
        ### load transform & dataset ###
        if 'epic' in args.dataset:
            transform = transforms.Compose([
                RandomHorizontalFlip(consistent=True),
                # RandomSizedCrop(size=224, consistent=True, p=1.0),
                RandomCrop(size=224, consistent=True),
                Scale(size=(args.img_dim, args.img_dim)),
                RandomGray(consistent=False, p=0.5),
                ColorJitter(brightness=0.5, contrast=0.5,
                            saturation=0.5, hue=0.25, p=1.0),
                ToTensor(),
                Normalize()
            ])
        elif 'ucf' in args.dataset:  # designed for ucf101, short size=256, rand crop to 224x224 then scale to 128x128
            transform = transforms.Compose([
                RandomHorizontalFlip(consistent=True),
                RandomCrop(size=224, consistent=True),
                Scale(size=(args.img_dim, args.img_dim)),
                RandomGray(consistent=False, p=0.5),
                ColorJitter(brightness=0.5, contrast=0.5,
                            saturation=0.5, hue=0.25, p=1.0),
                ToTensor(),
                Normalize()
            ])
        elif args.dataset == 'k400':  # designed for kinetics400, short size=150, rand crop to 128x128
            transform = transforms.Compose([
                RandomHorizontalFlip(consistent=True),
                RandomSizedCrop(size=args.img_dim, consistent=True, p=1.0),
                RandomGray(consistent=False, p=0.5),
                ColorJitter(brightness=0.5, contrast=0.5,
                            saturation=0.5, hue=0.25, p=1.0),
                ToTensor(),
                Normalize()
            ])
        elif args.dataset in ['block_toy_1',
                              'block_toy_2',
                              'block_toy_3']:
            # get validation data with label for plotting the embedding
            transform = transforms.Compose([
                # centercrop
                CenterCrop(size=224),
                # RandomSizedCrop(consistent=True, size=224, p=0.0), # no effect
                Scale(size=(args.img_dim, args.img_dim)),
                ToTensor(),
                Normalize()
            ])
        elif args.dataset in ['block_toy_imagenet_1',
                              'block_toy_imagenet_2',
                              'block_toy_imagenet_3']:
            # may have to introduce more transformations with imagenet background
            transform = transforms.Compose([
                # centercrop
                CenterCrop(size=224),
                # RandomSizedCrop(consistent=True, size=224, p=0.0), # no effect
                Scale(size=(args.img_dim, args.img_dim)),
                ToTensor(),
                Normalize()
            ])
    elif mode == 'val':
        if 'epic' in args.dataset:
            transform = transforms.Compose([
                RandomHorizontalFlip(consistent=True),
                CenterCrop(size=224, consistent=True),
                Scale(size=(args.img_dim, args.img_dim)),
                ColorJitter(brightness=0.2, contrast=0.2,
                            saturation=0.2, hue=0.1, p=1.0),
                ToTensor(),
                Normalize()
            ])
        elif 'ucf' in args.dataset:  # designed for ucf101, short size=256, rand crop to 224x224 then scale to 128x128
            transform = transforms.Compose([
                RandomHorizontalFlip(consistent=True),
                CenterCrop(size=224, consistent=True),
                Scale(size=(args.img_dim, args.img_dim)),
                ColorJitter(brightness=0.2, contrast=0.2,
                            saturation=0.2, hue=0.1, p=1.0),
                ToTensor(),
                Normalize()
            ])
        elif args.dataset == 'k400':  # designed for kinetics400, short size=150, rand crop to 128x128
            transform = transforms.Compose([
                RandomHorizontalFlip(consistent=True),
                CenterCrop(size=224),
                ColorJitter(brightness=0.2, contrast=0.2,
                            saturation=0.2, hue=0.1, p=1.0),
                ToTensor(),
                Normalize()
            ])
        elif args.dataset in ['block_toy_1',
                              'block_toy_2',
                              'block_toy_3']:
            # get validation data with label for plotting the embedding
            transform = transforms.Compose([
                # centercrop
                CenterCrop(size=224),
                Scale(size=(args.img_dim, args.img_dim)),
                ToTensor(),
                Normalize()
            ])
        elif args.dataset in ['block_toy_imagenet_1',
                              'block_toy_imagenet_2',
                              'block_toy_imagenet_3']:
            # may have to introduce more transformations with imagenet background
            transform = transforms.Compose([
                # centercrop
                CenterCrop(size=224),
                Scale(size=(args.img_dim, args.img_dim)),
                ToTensor(),
                Normalize()
            ])
    elif mode == 'test':
        if 'epic' in args.dataset:
            transform = transforms.Compose([
                CenterCrop(size=224),
                Scale(size=(args.img_dim, args.img_dim)),
                ToTensor(),
                Normalize()
            ])
        elif 'ucf' in args.dataset:  # designed for ucf101, short size=256, rand crop to 224x224 then scale to 128x128
            transform = transforms.Compose([
                CenterCrop(size=224),
                Scale(size=(args.img_dim, args.img_dim)),
                ToTensor(),
                Normalize()
            ])
        elif args.dataset == 'k400':  # designed for kinetics400, short size=150, rand crop to 128x128
            transform = transforms.Compose([
                CenterCrop(size=224),
                Scale(size=(args.img_dim, args.img_dim)),
                ToTensor(),
                Normalize()
            ])
        elif args.dataset in ['block_toy_1',
                              'block_toy_2',
                              'block_toy_3']:
            transform = transforms.Compose([
                CenterCrop(size=224),
                Scale(size=(args.img_dim, args.img_dim)),
                ToTensor(),
                Normalize()
            ])
        elif args.dataset in ['block_toy_imagenet_1',
                              'block_toy_imagenet_2',
                              'block_toy_imagenet_3']:
            transform = transforms.Compose([
                CenterCrop(size=224),
                Scale(size=(args.img_dim, args.img_dim)),
                ToTensor(),
                Normalize()
            ])
    return transform
    return transform


def get_data(args, transform, mode='train', pm_path=None, epoch=None):   
    
    print('Loading data for "%s", from %s ...' % (mode, args.dataset))
    if args.dataset == 'k400':
        use_big_K400 = args.img_dim > 140
        dataset = Kinetics400_full_3d(mode=mode,
                                      transform=transform,
                                      seq_len=args.seq_len,
                                      num_seq=args.num_seq,
                                      downsample=args.ds,
                                      big=use_big_K400)
    elif args.dataset == 'ucf101':
        dataset = UCF101_3d(mode=mode,
                            transform=transform,
                            seq_len=args.seq_len,
                            num_seq=args.num_seq,
                            downsample=args.ds)
    elif args.dataset == 'ucf11':
        dataset = UCF11_3d(mode=mode,
                           transform=transform,
                           seq_len=args.seq_len,
                           num_seq=args.num_seq,
                           downsample=args.ds)
    elif args.dataset == 'epic_unlabeled':
        dataset = epic_unlabeled(mode=mode,
                                 transform=transform,
                                 seq_len=args.seq_len,
                                 num_seq=args.num_seq,
                                 pred_step=args.pred_step,
                                 downsample=args.ds,
                                 drive=args.drive,
                                 num_clips=args.num_clips)
    elif args.dataset == 'epic_within':
        dataset = epic_action_based(mode=mode,
                                    transform=transform,
                                    seq_len=args.seq_len,
                                    num_seq=args.num_seq,
                                    pred_step=args.pred_step,
                                    downsample=args.ds,
                                    drive=args.drive,
                                    sample_method='within')
    elif args.dataset == 'epic_before':
        dataset = epic_action_based(mode=mode,
                                    transform=transform,
                                    seq_len=args.seq_len,
                                    num_seq=args.num_seq,
                                    pred_step=args.pred_step,
                                    downsample=args.ds,
                                    drive=args.drive,
                                    sample_method='before',
                                    sample_offset=1)
    elif args.dataset == 'epic_present_matching':
        dataset = epic_present_matching(epoch, freq=args.pm_freq,
                                        mode=mode,
                                        transform=transform,
                                        seq_len=args.seq_len,
                                        num_seq=args.num_seq,
                                        downsample=args.ds,
                                        num_clips=args.num_clips,
                                        drive='ssd',
                                        NN_path=pm_path,
                                        num_ngbrs=args.pm_num_ngbrs)
    elif 'synthepic' in args.dataset:
        dataset = synthepic_action_pair(mode=mode,
                                        transform=transform,
                                        seq_len=args.seq_len,
                                        num_seq=args.num_seq,
                                        pred_step=args.pred_step,
                                        downsample=args.ds,
                                        drive=args.drive,
                                        sample_method='within' if 'within' in args.dataset else 'match_cut',
                                        exact_cuts='exact' in args.dataset)
    elif args.dataset in ['block_toy_1',
                          'block_toy_2',
                          'block_toy_3']:
        num = args.dataset.split('_')[-1]
        dataset = block_toy(mode=mode,
                            num=num,
                            transform=transform,
                            seq_len=args.seq_len,
                            num_seq=args.num_seq,
                            downsample=args.ds,
                            num_sample=args.num_clips)
    elif args.dataset in ['block_toy_imagenet_1',
                          'block_toy_imagenet_2',
                          'block_toy_imagenet_3']:
        num = args.dataset.split('_')[-1]
        dataset = block_toy_imagenet(mode=mode,
                                     num=num,
                                     transform=transform,
                                     seq_len=args.seq_len,
                                     num_seq=args.num_seq,
                                     downsample=args.ds,
                                     num_sample=args.num_clips)
    else:
        raise ValueError('dataset not supported: ' + args.dataset)

    # randomize the instances
    #print (dataset)
    if args.dataset == 'epic_present_matching':
        sampler = data.SequentialSampler(dataset)
    else:
        sampler = data.RandomSampler(dataset)

    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=32,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'val':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=32,
                                      pin_memory=True,
                                      drop_last=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


def train(data_loader, model, optimizer, epoch, criterion, writer_train, args,
          de_normalize, cur_vae_kl_weight, cur_pred_divers_wt, do_encode=True):
    cuda = args.cuda

    # average losses
    losses_dpc = AverageMeter()
    losses_vae_kl = AverageMeter()
    losses_vae_divers = AverageMeter()
    losses = AverageMeter()

    # average accuracy
    accuracy = AverageMeter()

    # top-1, top-3 and top-5 accuracy
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]

    model.train()
    iteration = 0

    # print (len(data_loader))
    # with autograd.detect_anomaly():

    for idx, mbatch in enumerate(data_loader):

        tic = time.time()
        input_seq = mbatch['t_seq']
        input_seq = input_seq.to(cuda)
        B = input_seq.size(0)

        cur_res = model(input_seq, do_encode=do_encode,
                        get_pred_sim=(cur_pred_divers_wt > 0.0))

        if args.model == 'cvae':
            if do_encode:
                if cur_pred_divers_wt > 0.0:
                    # CVAE with encoding, with sample variance
                    [score_, mask_, mus, logvars, pred_sims] = cur_res
                else:
                    # CVAE with encoding, without sample variance
                    [score_, mask_, mus, logvars] = cur_res
            else:
                if cur_pred_divers_wt > 0.0:
                    # CVAE without encoding, with sample variance
                    [score_, mask_, pred_sims] = cur_res
                else:
                    # CVAE without encoding, without sample variance
                    [score_, mask_] = cur_res
        elif 'vrnn' in args.model:
            if do_encode:
                if cur_pred_divers_wt > 0.0:
                    # VRNN with encoding, with sample variance
                    [score_, mask_, mus_prio, mus_post, logvars_prio,
                        logvars_post, pred_sims] = cur_res
                else:
                    # VRNN with encoding, without sample variance
                    [score_, mask_, mus_prio, mus_post,
                        logvars_prio, logvars_post] = cur_res
            else:
                if cur_pred_divers_wt > 0.0:
                    # VRNN without encoding, with sample variance
                    [score_, mask_, pred_sims] = cur_res
                else:
                    # VRNN without encoding, without sample variance
                    [score_, mask_] = cur_res

            # DEBUG:
            if idx % 20 == 0 and do_encode:
                print('mus_prio:', mus_prio.shape, mus_prio.min().item(),
                      mus_prio.mean().item(), mus_prio.max().item())
                print('mus_post:', mus_post.shape, mus_post.min().item(),
                      mus_post.mean().item(), mus_post.max().item())
                print('logvars_prio:', logvars_prio.shape, logvars_prio.min().item(
                ), logvars_prio.mean().item(), logvars_prio.max().item())
                print('logvars_post:', logvars_post.shape, logvars_post.min().item(
                ), logvars_post.mean().item(), logvars_post.max().item())

        # visualize the input sequence to the network
        if (idx == 0) or (idx == args.print_freq):
            if B > 8:
                input_seq = input_seq[0:8, :]
            writer_train.add_image('input_seq',
                de_normalize(vutils.make_grid(input_seq.transpose(2, 3).contiguous(
                ).view(-1, 3, args.img_dim, args.img_dim), nrow=args.num_seq * args.seq_len)), idx)
        del input_seq

        # ==== DPC Loss ====

        if idx == 0:
            target_, (_, B2, NS, NP, SQ) = process_output(mask_)

        # score is a 6d tensor: [B, P, SQ, B, N, SQ]
        # Number Predicted NP, Number Sequence NS
        score_flattened = score_.view(B * NP * SQ, B2 * NS * SQ)
        target_flattened = target_.view(B * NP * SQ, B2 * NS * SQ)
        #print(score_flattened.shape, target_flattened.shape)

        # added
        target_flattened = target_flattened.double()
        target_flattened = target_flattened.argmax(dim=1)

        # ==== KL & Divergence Loss ====

        theo_loss_vae_kl = torch.tensor(0.0).cuda()  # no multiplicative factors
        # weighted by both base factor & warmup
        used_loss_vae_kl = torch.tensor(0.0).cuda()
        loss_vae_divers = torch.tensor(0.0).cuda()
        for i in range(B):

            # KL divergence loss
            if do_encode:
                if args.model == 'cvae':
                    # all (pred_step, latent_size, ...)
                    mu, logvar = mus[i], logvars[i]
                    cur_loss_kl = loss_kl_divergence(mu, logvar, normalization=not(args.vae_no_kl_norm))
                elif 'vrnn' in args.model:
                    mu_prio, mu_post, logvar_prio, logvar_post = \
                        mus_prio[i], mus_post[i], logvars_prio[i], logvars_post[i]  # all (pred_step, latent_size, ...)
                    cur_loss_kl = loss_kl_divergence_prior(
                        mu_prio, mu_post, logvar_prio, logvar_post, normalization=not(args.vae_no_kl_norm))
                theo_loss_vae_kl += cur_loss_kl
                used_loss_vae_kl += cur_loss_kl * cur_vae_kl_weight

            # Prediction diversity loss (per value for safety)
            if cur_pred_divers_wt > 0.0:
                for j in range(args.pred_step):
                    pred_sim = pred_sims[i, j, 0]
                    z_dist = pred_sims[i, j, 1]
                    loss_vae_divers += loss_pred_variance(pred_sim, z_dist,
                        args.pred_divers_formula) * cur_pred_divers_wt / args.pred_step

        # ==== Combine ====

        # loss function, here cross-entropy for DPC
        loss_dpc = criterion(score_flattened, target_flattened)
        loss = loss_dpc + used_loss_vae_kl + loss_vae_divers
        top1, top3, top5 = calc_topk_accuracy(
            score_flattened, target_flattened, (1, 3, 5))

        accuracy_list[0].update(top1.item(), B)
        accuracy_list[1].update(top3.item(), B)
        accuracy_list[2].update(top5.item(), B)

        losses_dpc.update(loss_dpc.item(), B)
        # mentally multiply by warmup factor etc.
        losses_vae_kl.update(theo_loss_vae_kl.item(), B)
        losses_vae_divers.update(loss_vae_divers.item(), B)
        losses.update(loss.item(), B)
        accuracy.update(top1.item(), B)

        del score_

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del loss

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]  '
                  'Loss DPC {loss_dpc.val:.3f} + VAE {loss_vae_kl.val:.3f} + DIV {loss_vae_divers.val:.3f} = {loss.val:.3f} ({loss.local_avg:.3f})  '
                  'Acc top1 {3:.3f} top3 {4:.3f} top5 {5:.3f}  T {6:.2f} '.format(
                      epoch, idx, len(data_loader), top1, top3, top5, time.time() - tic, loss=losses,
                      loss_dpc=losses_dpc, loss_vae_kl=losses_vae_kl, loss_vae_divers=losses_vae_divers))

            writer_train.add_scalar(
                'local/loss_dpc', losses_dpc.val, iteration)
            writer_train.add_scalar(
                'local/loss_vae_kl', losses_vae_kl.val, iteration)
            writer_train.add_scalar(
                'local/loss_vae_divers', losses_vae_divers.val, iteration)
            writer_train.add_scalar('local/loss', losses.val, iteration)
            writer_train.add_scalar('local/accuracy', accuracy.val, iteration)

            iteration += 1

    return losses_dpc.avg, losses_vae_kl.avg, losses_vae_divers.avg, losses.avg, \
           accuracy.avg, [i.avg for i in accuracy_list]


def validate_once(model, input_seq, criterion, processed_mask, args, do_encode, cur_vae_kl_weight, cur_pred_divers_wt):
    '''
    Forward input to produce one sequence of outputs.
    '''
    B = input_seq.size(0)
    cur_res = model(input_seq, do_encode=do_encode, get_pred_sim=(cur_pred_divers_wt > 0.0))
    if args.model == 'cvae':
        if do_encode:
            if cur_pred_divers_wt > 0.0:
                # CVAE with encoding, with sample variance
                [score_, mask_, mus, logvars, pred_sims] = cur_res
            else:
                # CVAE with encoding, without sample variance
                [score_, mask_, mus, logvars] = cur_res
        else:
            if cur_pred_divers_wt > 0.0:
                # CVAE without encoding, with sample variance
                [score_, mask_, pred_sims] = cur_res
            else:
                # CVAE without encoding, without sample variance
                [score_, mask_] = cur_res
    elif 'vrnn' in args.model:
        if do_encode:
            if cur_pred_divers_wt > 0.0:
                # VRNN with encoding, with sample variance
                [score_, mask_, mus_prio, mus_post, logvars_prio,
                    logvars_post, pred_sims] = cur_res
            else:
                # VRNN with encoding, without sample variance
                [score_, mask_, mus_prio, mus_post,
                    logvars_prio, logvars_post] = cur_res
        else:
            if cur_pred_divers_wt > 0.0:
                # VRNN without encoding, with sample variance
                [score_, mask_, pred_sims] = cur_res
            else:
                # VRNN without encoding, without sample variance
                [score_, mask_] = cur_res
    del input_seq

    # ==== DPC Loss ====

    if processed_mask is None:
        processed_mask = process_output(mask_)
    target_, (_, B2, NS, NP, SQ) = processed_mask

    # [B, P, SQ, B, N, SQ]
    score_flattened = score_.view(B * NP * SQ, B2 * NS * SQ)
    target_flattened = target_.view(B * NP * SQ, B2 * NS * SQ)

    target_flattened = target_flattened.double()
    target_flattened = target_flattened.argmax(dim=1)

    # ==== KL & Divergence Loss ====

    theo_loss_vae_kl = torch.tensor(0.0).cuda() # no multiplicative factors
    # weighted by both base factor & warmup
    used_loss_vae_kl = torch.tensor(0.0).cuda()
    loss_vae_divers = torch.tensor(0.0).cuda()
    for i in range(B):

        # KL divergence loss
        if do_encode:
            if args.model == 'cvae':
                # all (pred_step, latent_size, ...)
                mu, logvar = mus[i], logvars[i]
                cur_loss_kl = loss_kl_divergence(mu, logvar, normalization=not(args.vae_no_kl_norm))
            elif 'vrnn' in args.model:
                mu_prio, mu_post, logvar_prio, logvar_post = \
                    mus_prio[i], mus_post[i], logvars_prio[i], logvars_post[i]  # all (pred_step, latent_size, ...)
                cur_loss_kl = loss_kl_divergence_prior(
                    mu_prio, mu_post, logvar_prio, logvar_post, normalization=not(args.vae_no_kl_norm))
            theo_loss_vae_kl += cur_loss_kl
            used_loss_vae_kl += cur_loss_kl * cur_vae_kl_weight

        # Prediction diversity loss (per value for safety)
        if cur_pred_divers_wt > 0.0:
            for j in range(args.pred_step):
                pred_sim = pred_sims[i, j, 0]
                z_dist = pred_sims[i, j, 1]
                loss_vae_divers += loss_pred_variance(pred_sim, z_dist,
                    args.pred_divers_formula) * cur_pred_divers_wt / args.pred_step
             
    # ==== Combine ====       

    loss_dpc = criterion(score_flattened, target_flattened)
    loss = loss_dpc + used_loss_vae_kl + loss_vae_divers

    return loss, loss_dpc, theo_loss_vae_kl, used_loss_vae_kl, loss_vae_divers, processed_mask, score_flattened, target_flattened


def validate(data_loader, model, epoch, args, criterion,
             cur_vae_kl_weight, cur_pred_divers_wt, do_encode=False, select_best_among=1):
    '''
    Validate for one epoch.
    select_best_among: If >1, only evaluate the best performing set of samples that was generated within every video sequence.
    '''
    cuda = args.cuda
    processed_mask = None

    if do_encode:
        print('[train_util.py] validate() WITH encoder')
        if select_best_among > 1:
            raise ValueError('It is almost certainly a mistake to generate multiple paths with VAE encoding enabled')
    else:
        print('[train_util.py] validate() WITHOUT encoder, select best among: ' + str(select_best_among))
        
    losses_dpc = AverageMeter()
    losses_vae_kl = AverageMeter()
    losses_vae_divers = AverageMeter()
    losses = AverageMeter()

    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    model.eval()

    with torch.no_grad():

        for idx, mbatch in tqdm(enumerate(data_loader), total=len(data_loader)):

            input_seq = mbatch['t_seq']
            input_seq = input_seq.to(cuda)
            B = input_seq.size(0)

            # Try out multiple paths and select the best performing one
            best_output = None
            min_loss = torch.tensor(1e12)
            for i in range(select_best_among):
                output = validate_once(model, input_seq, criterion, processed_mask, args, do_encode, cur_vae_kl_weight, cur_pred_divers_wt)
                loss = output[0]
                if loss.item() < min_loss.item():
                    best_output = output
                    min_loss = loss

            (loss, loss_dpc, theo_loss_vae_kl, used_loss_vae_kl, loss_vae_divers, processed_mask, score_flattened, target_flattened) = best_output
            top1, top3, top5 = calc_topk_accuracy(
                score_flattened, target_flattened, (1, 3, 5))

            accuracy_list[0].update(top1.item(), B)
            accuracy_list[1].update(top3.item(), B)
            accuracy_list[2].update(top5.item(), B)

            losses_dpc.update(loss_dpc.item(), B)
            # mentally multiply by warmup factor etc.
            losses_vae_kl.update(theo_loss_vae_kl.item(), B)
            losses_vae_divers.update(loss_vae_divers.item(), B)
            losses.update(loss.item(), B)
            accuracy.update(top1.item(), B)

    print('Val: [{0}/{1}]  '
          'Loss DPC {loss_dpc.avg:.3f} + VAE {loss_vae_kl.avg:.3f} + DIV {loss_vae_divers.avg:.3f} = {loss.avg:.3f}  '
          'Acc top1 {2:.3f} top3 {3:.3f} top5 {4:.3f} \t'.format(
              epoch, args.epochs, *[i.avg for i in accuracy_list], loss=losses,
              loss_dpc=losses_dpc, loss_vae_kl=losses_vae_kl, loss_vae_divers=losses_vae_divers))
    return losses_dpc.avg, losses_vae_kl.avg, losses_vae_divers.avg, losses.avg, \
           accuracy.avg, [i.avg for i in accuracy_list]


def set_path(args):
    '''
    Creates descriptive directories for model files, tensorboard logs, diversity outputs, and other results.
    '''

    if args.resume:
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    elif args.test_diversity:
        # This will populate the 'divers' subfolder within the pretrained model
        exp_path = os.path.dirname(os.path.dirname(args.pretrain))
    else:
        # NOTE: be careful in changing this path, for backward compatibility
        if args.model == 'cvae':
            # if not(args.test_diversity):
            exp_path = 'log_{args.prefix}/{args.dataset}-{args.img_dim}_{0}_{args.model}_{args.cvae_arch}_\
bs{args.batch_size}_lr{1}_seq{args.num_seq}_pred{args.pred_step}_len{args.seq_len}_ds{args.ds}_\
kl{args.vae_kl_weight}{3}_warm{args.vae_kl_warmup}_nc{args.num_clips}_train-{args.train_what}{2}'.format(
                'r%s' % args.net[6::], args.old_lr if args.old_lr is not None else args.lr,
                '_pt=%s' % args.pretrain.replace('/', '-') if args.pretrain else '',
                'nrm' if not(args.vae_no_kl_norm) else '',
                args=args)
#             else:
#                 exp_path = 'log_{args.prefix}/{args.dataset}-{args.img_dim}_{0}_{args.model}_{args.cvae_arch}_\
# bs{args.batch_size}_seq{args.num_seq}_pred{args.pred_step}_len{args.seq_len}_ds{args.ds}_\
# ns{args.num_clips}_test-div_paths{args.paths}{1}'.format(
#                     'r%s' % args.net[6::],
#                     '_pt=%s' % args.pretrain.replace('/', '-') if args.pretrain else '',
#                     args=args)
        elif 'vrnn' in args.model:
            exp_path = 'log_{args.prefix}/{args.dataset}-{args.img_dim}_{0}_{args.model}_\
ls{args.vrnn_latent_size}_ks{args.vrnn_kernel_size}_kl{args.vae_kl_weight}{3}_\
els{args.vae_encoderless_epochs}_int{args.vae_inter_kl_epochs}_warm{args.vae_kl_beta_warmup}_\
div{args.pred_divers_wt}{args.pred_divers_formula}_do{args.vrnn_dropout}_\
bs{args.batch_size}_lr{1}_seq{args.num_seq}_pred{args.pred_step}_len{args.seq_len}_ds{args.ds}_\
nc{args.num_clips}_train-{args.train_what}{2}'.format(
                'r%s' % args.net[6::], args.old_lr if args.old_lr is not None else args.lr,
                '_pt=%s' % args.pretrain.replace('/', '-') if args.pretrain else '',
                'nrm' if not(args.vae_no_kl_norm) else '',
                args=args)
        # All VAE models
        if 0 <= args.pm_start and args.pm_start <= args.epochs:
            exp_path += '_pm-start{args.pm_start}_pm-freq{args.pm_freq}_pm-ngbrs{args.pm_num_ngbrs}'.format(args=args)

    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    divers_path = os.path.join(exp_path, 'divers')
    pm_cache = os.path.join(exp_path, 'pm_cache')
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(divers_path):
        os.makedirs(divers_path)
    if not os.path.exists(pm_cache):
        os.makedirs(pm_cache)
    return img_path, model_path, divers_path, pm_cache
