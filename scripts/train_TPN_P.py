import argparse
import gc
import logging
import os
import sys
import time
import numpy as np

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from TPN.data.loader import data_loader_TPN
from TPN.losses import gan_g_loss, gan_d_loss, l2_loss
from TPN.losses import displacement_error, final_displacement_error

from TPN.models import TrajectoryDiscriminator, TrajectoryGeneratorTPNPooling
from TPN.utils import int_tuple, bool_flag, get_total_norm
from TPN.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default='eth', type=str)  # FIXME (eth/hotel/univ/zara1/zara2)
parser.add_argument('--delim', default='tab')
parser.add_argument('--loader_num_workers', default=0, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=12, type=int)
parser.add_argument('--skip', default=1, type=int)

# Optimization
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=10000, type=int)
parser.add_argument('--num_epochs', default=200, type=int)  # FIXME default=400, eth 200, other 400

# Model Options
parser.add_argument('--embedding_dim', default=16, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0.0, type=float)
parser.add_argument('--batch_norm', default=1, type=bool_flag)
parser.add_argument('--mlp_dim', default=64, type=int)

# Generator Options
parser.add_argument('--encoder_h_dim_g', default=32, type=int)
parser.add_argument('--decoder_h_dim_g', default=32, type=int)
parser.add_argument('--noise_dim', default=(8,), type=int_tuple)
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='global')
parser.add_argument('--clipping_threshold_g', default=2.0, type=float)
parser.add_argument('--g_learning_rate', default=1e-4, type=float)  # default 1e-4
parser.add_argument('--g_steps', default=1, type=int)

# Pooling Options
parser.add_argument('--pooling_type', default='pool_net')  # FIXME none or 'pool_net' or 'spool'
parser.add_argument('--pool_every_timestep', default=0, type=bool_flag)  # FIXME dafault 0, we should pool every step, so we set to be 1

# Pool Net Option
parser.add_argument('--bottleneck_dim', default=8, type=int)

# Social Pooling Options
parser.add_argument('--neighborhood_size', default=2.0, type=float)
parser.add_argument('--grid_size', default=8, type=int)

# Discriminator Options
parser.add_argument('--d_type', default='local', type=str)
parser.add_argument('--encoder_h_dim_d', default=48, type=int)
parser.add_argument('--d_learning_rate', default=2e-4, type=float)  # FIXME default 2e-4
parser.add_argument('--d_steps', default=1, type=int)
parser.add_argument('--clipping_threshold_d', default=0, type=float)

# Loss Options
parser.add_argument('--l2_loss_weight', default=1.0, type=float)
parser.add_argument('--best_k', default=20, type=int)  # default 20

# Output
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--print_every', default=100, type=int)
parser.add_argument('--checkpoint_every', default=300, type=int)
parser.add_argument('--checkpoint_name', default='eth_TPN_P_with_Multi_with_model.pt')  # default='checkpoint'
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=1, type=int)
parser.add_argument('--num_samples_check', default=5000, type=int)
# Misc
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    train_path = get_dset_path(args.dataset_name, 'train')
    val_path = get_dset_path(args.dataset_name, 'val')

    long_dtype, float_dtype = get_dtypes(args)

    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader_TPN(args, train_path)

    logger.info("Initializing val dataset")
    _, val_loader = data_loader_TPN(args, val_path)

    iterations_per_epoch = len(train_dset)/args.batch_size/(args.d_steps + args.g_steps)
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)

    logger.info(
        'There are {} iterations per epoch'.format(iterations_per_epoch)
    )

    generator = TrajectoryGeneratorTPNPooling(
        obs_len=args.obs_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)

    generator.apply(init_weights)
    generator.type(float_dtype).train()
    logger.info('Here is the generator:')
    logger.info(generator)

    discriminator = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_d,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        d_type=args.d_type)

    discriminator.apply(init_weights)
    discriminator.type(float_dtype).train()
    logger.info('Here is the discriminator:')
    logger.info(discriminator)

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss

    #  FIXME change the way to optimize 1
    # optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate)  # FIXME for eth lr_g:1e-4, lr_d:2e-4
    # optimizer_d = optim.Adam(discriminator.parameters(), lr=args.d_learning_rate)
    # FIXME change the way to optimize 2
    optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate, betas=(0.9, 0.999), weight_decay=4e-4)
    scheduler_g = torch.optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=[57, 104, 161, 218], gamma=0.5)  # FIXME for univ, zara1, zara2 lr_g:1e-3, lr_d:2e-3
    # scheduler_g = torch.optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=[120], gamma=0.5)  # Fixme for hotel, lr_g:1e-4, lr_d:2e-4
    # # # # #
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.d_learning_rate, betas=(0.9, 0.999), weight_decay=4e-4)
    scheduler_d = torch.optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=[57, 104, 161, 218], gamma=0.5)  # FIXME for univ, zara1, zara2 lr_g:1e-3, lr_d:2e-3
    # scheduler_d = torch.optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=[120], gamma=0.5)  # Fixme for hotel, lr_g:1e-4, lr_d:2e-4

    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir,
                                    '%s_with_model.pt' % args.checkpoint_name)

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        checkpoint['args']['angle_loss_weight'] = args.angle_loss_weight   # FIXME update the weight
        checkpoint['args']['num_epochs'] = args.num_epochs
        generator.load_state_dict(checkpoint['g_state'])
        discriminator.load_state_dict(checkpoint['d_state'])
        optimizer_g.load_state_dict(checkpoint['g_optim_state'])
        optimizer_d.load_state_dict(checkpoint['d_optim_state'])
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'G_losses': defaultdict(list),
            'D_losses': defaultdict(list),
            'losses_ts': [],
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'sample_ts': [],
            'restore_ts': [],
            'norm_g': [],
            'norm_d': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'g_state': None,
            'g_optim_state': None,
            'merge_state': None,
            'merge_optim_state': None,
            'd_state': None,
            'd_optim_state': None,
            'g_best_state': None,
            'd_best_state': None,
            'merge_best_state': None,
            'best_t': None,
            'g_best_nl_state': None,
            'd_best_state_nl': None,
            'best_t_nl': None,
        }
    t0 = None
    while t < args.num_iterations:
        gc.collect()
        d_steps_left = args.d_steps
        g_steps_left = args.g_steps
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))
        for batches in train_loader:
            if args.timing == 1:
                torch.cuda.synchronize()
                t1 = time.time()

            # FIXME Multi Scale input
            (batch, batch_4, batch_8, batch_16, batch_32, mean_sta) = batches
            (traj_mean_x_8, traj_std_x_8, traj_mean_y_8, traj_std_y_8,traj_rel_mean_x_8, traj_rel_std_x_8,
             traj_rel_mean_y_8, traj_rel_std_y_8) = mean_sta
            if d_steps_left > 0:
                scheduler_d.step(epoch)  # change the lr to optimize
                step_type = 'd'
                losses_d = discriminator_step(args, batch_4, batch_8, batch_16, batch_32, generator,
                                              discriminator, d_loss_fn,
                                              optimizer_d)
                checkpoint['norm_d'].append(
                    get_total_norm(discriminator.parameters()))
                d_steps_left -= 1
            elif g_steps_left > 0:
                scheduler_g.step(epoch)
                step_type = 'g'
                losses_g = generator_step(args, batch_4, batch_8, batch_16, batch_32, generator,
                                          discriminator, g_loss_fn, optimizer_g,
                                          traj_mean_x_8, traj_std_x_8, traj_mean_y_8,
                                          traj_std_y_8)
                checkpoint['norm_g'].append(
                    get_total_norm(generator.parameters())
                )
                g_steps_left -= 1

            if args.timing == 1:
                torch.cuda.synchronize()
                t2 = time.time()
                logger.info('{} step took {}'.format(step_type, t2 - t1))

            # Skip the rest if we are not at the end of an iteration
            if d_steps_left > 0 or g_steps_left > 0:
                continue

            if args.timing == 1:
                if t0 is not None:
                    logger.info('Interation {} took {}'.format(
                        t - 1, time.time() - t0
                    ))
                t0 = time.time()

            # Maybe save loss
            if t % args.print_every == 0:
                logger.info('t = {} / {}'.format(t + 1, args.num_iterations))
                for k, v in sorted(losses_d.items()):
                    logger.info('  [D] {}: {:.3f}'.format(k, v))
                    checkpoint['D_losses'][k].append(v)
                for k, v in sorted(losses_g.items()):
                    logger.info('  [G] {}: {:.3f}'.format(k, v))
                    checkpoint['G_losses'][k].append(v)
                checkpoint['losses_ts'].append(t)

            # Maybe save a checkpoint
            if t > 0 and t % int(iterations_per_epoch * 40) == 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                metrics_val = check_accuracy(
                    args, val_loader, generator, discriminator, d_loss_fn
                )
                logger.info('Checking stats on train ...')
                metrics_train = check_accuracy(
                    args, train_loader, generator, discriminator,
                    d_loss_fn, limit=True
                )

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)

                min_ade = min(checkpoint['metrics_val']['ade'])
                min_ade_nl = min(checkpoint['metrics_val']['ade_nl'])

                if metrics_val['ade'] == min_ade:
                    logger.info('New low for avg_disp_error')
                    checkpoint['best_t'] = t
                    checkpoint['g_best_state'] = generator.state_dict()
                    checkpoint['d_best_state'] = discriminator.state_dict()

                if metrics_val['ade_nl'] == min_ade_nl:
                    logger.info('New low for avg_disp_error_nl')
                    checkpoint['best_t_nl'] = t
                    checkpoint['g_best_nl_state'] = generator.state_dict()
                    checkpoint['d_best_nl_state'] = discriminator.state_dict()

                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint['g_state'] = generator.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint['d_state'] = discriminator.state_dict()
                checkpoint['d_optim_state'] = optimizer_d.state_dict()
                checkpoint_path = os.path.join(
                    args.output_dir, 'epoch%s' % epoch + '%s_with_model.pt' % args.checkpoint_name
                )
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')

                # Save a checkpoint with no model weights by making a shallow
                # copy of the checkpoint excluding some items
                checkpoint_path = os.path.join(
                    args.output_dir, 'epoch%s' % epoch + '%s_no_model.pt' % args.checkpoint_name)
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                key_blacklist = [
                    'g_state', 'd_state', 'g_best_state', 'g_best_nl_state',
                    'g_optim_state', 'd_optim_state', 'd_best_state',
                    'd_best_nl_state'
                ]
                small_checkpoint = {}
                for k, v in checkpoint.items():
                    if k not in key_blacklist:
                        small_checkpoint[k] = v
                torch.save(small_checkpoint, checkpoint_path)
                logger.info('Done.')
            if t > 0 and t % args.checkpoint_every == 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                metrics_val = check_accuracy(
                    args, val_loader, generator, discriminator, d_loss_fn
                )
                logger.info('Checking stats on train ...')
                metrics_train = check_accuracy(
                    args, train_loader, generator, discriminator,
                    d_loss_fn, limit=True
                )

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)

                min_ade = min(checkpoint['metrics_val']['ade'])
                min_ade_nl = min(checkpoint['metrics_val']['ade_nl'])

                if metrics_val['ade'] == min_ade:
                    logger.info('New low for avg_disp_error')
                    checkpoint['best_t'] = t
                    checkpoint['g_best_state'] = generator.state_dict()
                    checkpoint['d_best_state'] = discriminator.state_dict()

                if metrics_val['ade_nl'] == min_ade_nl:
                    logger.info('New low for avg_disp_error_nl')
                    checkpoint['best_t_nl'] = t
                    checkpoint['g_best_nl_state'] = generator.state_dict()
                    checkpoint['d_best_nl_state'] = discriminator.state_dict()

                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint['g_state'] = generator.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint['d_state'] = discriminator.state_dict()
                checkpoint['d_optim_state'] = optimizer_d.state_dict()
                checkpoint_path = os.path.join(
                    args.output_dir, '%s_with_model.pt' % args.checkpoint_name
                )
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')

                # Save a checkpoint with no model weights by making a shallow
                # copy of the checkpoint excluding some items
                checkpoint_path = os.path.join(
                    args.output_dir, '%s_no_model.pt' % args.checkpoint_name)
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                key_blacklist = [
                    'g_state', 'd_state', 'g_best_state', 'g_best_nl_state',
                    'g_optim_state', 'd_optim_state', 'd_best_state',
                    'd_best_nl_state'
                ]
                small_checkpoint = {}
                for k, v in checkpoint.items():
                    if k not in key_blacklist:
                        small_checkpoint[k] = v
                torch.save(small_checkpoint, checkpoint_path)
                logger.info('Done.')

            t += 1
            d_steps_left = args.d_steps
            g_steps_left = args.g_steps
            if t >= args.num_iterations:
                break


def discriminator_step(
    args, batch_4, batch_8, batch_16, batch_32, generator, discriminator, d_loss_fn, optimizer_d
):
    batch_4 = [tensor.cuda() for tensor in batch_4]
    batch_8 = [tensor.cuda() for tensor in batch_8]
    batch_16 = [tensor.cuda() for tensor in batch_16]
    batch_32 = [tensor.cuda() for tensor in batch_32]
    (obs_traj_4, pred_traj_gt_4, obs_traj_rel_4, pred_traj_gt_rel_4, non_linear_ped,
     seq_start_end) = batch_4
    (obs_traj_8, pred_traj_gt_8, obs_traj_rel_8, pred_traj_gt_rel_8, non_linear_ped,
     seq_start_end) = batch_8
    (obs_traj_16, pred_traj_gt_16, obs_traj_rel_16, pred_traj_gt_rel_16, non_linear_ped,
     seq_start_end) = batch_16
    (obs_traj_32, pred_traj_gt_32, obs_traj_rel_32, pred_traj_gt_rel_32, non_linear_ped,
     seq_start_end) = batch_32
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt_8)  # changes tensor to the device as the same as pred_traj_gt_8

    generator_out_4, generator_out_8, generator_out_16, generator_out_32, generator_out_final = generator(obs_traj_4,
                                                                                                          obs_traj_rel_4,
                                                                                                          obs_traj_8,
                                                                                                          obs_traj_rel_8,
                                                                                                          obs_traj_16,
                                                                                                          obs_traj_rel_16,
                                                                                                          obs_traj_32,
                                                                                                          obs_traj_rel_32,
                                                                                                          seq_start_end,
                                                                                                          pred_traj_gt_4.shape[
                                                                                                              0])

    pred_traj_fake_rel = generator_out_final
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj_8[-1])

    traj_real = torch.cat([obs_traj_8, pred_traj_gt_8], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel_8, pred_traj_gt_rel_8], dim=0)
    traj_fake = torch.cat([obs_traj_8, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel_8, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

    # Compute loss with optional gradient penalty
    data_loss = d_loss_fn(scores_real, scores_fake)
    losses['D_data_loss'] = data_loss.item()
    loss += data_loss
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    if args.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(),
                                 args.clipping_threshold_d)
    optimizer_d.step()

    return losses


def generator_step(
    args, batch_4, batch_8, batch_16, batch_32, generator, discriminator, g_loss_fn, optimizer_g,obs_traj_mean_x_8, obs_traj_std_x_8, obs_traj_mean_y_8, obs_traj_std_y_8):
    batch_4 = [tensor.cuda() for tensor in batch_4]
    batch_8 = [tensor.cuda() for tensor in batch_8]
    batch_16 = [tensor.cuda() for tensor in batch_16]
    batch_32 = [tensor.cuda() for tensor in batch_32]
    (obs_traj_4, pred_traj_gt_4, obs_traj_rel_4, pred_traj_gt_rel_4, non_linear_ped,
     seq_start_end) = batch_4
    (obs_traj_8, pred_traj_gt_8, obs_traj_rel_8, pred_traj_gt_rel_8, non_linear_ped,
     seq_start_end) = batch_8
    (obs_traj_16, pred_traj_gt_16, obs_traj_rel_16, pred_traj_gt_rel_16, non_linear_ped,
     seq_start_end) = batch_16
    (obs_traj_32, pred_traj_gt_32, obs_traj_rel_32, pred_traj_gt_rel_32, non_linear_ped,
     seq_start_end) = batch_32
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt_8)
    g_l2_loss_rel_4 = []
    g_l2_loss_rel_8 = []
    g_l2_loss_rel_16 = []
    g_l2_loss_rel_32 = []
    g_l2_loss_rel_final = []
    loss_weight = [2.0, 1.0, 0.5, 0.25, 0.25, 1.0]  # FIXME weights of multiscale output

    for _ in range(args.best_k):
        generator_out_4, generator_out_8, generator_out_16, generator_out_32, generator_out_final = generator(obs_traj_4, obs_traj_rel_4, obs_traj_8, obs_traj_rel_8, obs_traj_16, obs_traj_rel_16, obs_traj_32, obs_traj_rel_32, seq_start_end, pred_traj_gt_4.shape[0])
        pred_traj_fake_rel_4 = generator_out_4
        pred_traj_fake_rel_8 = generator_out_8
        pred_traj_fake_rel_16 = generator_out_16
        pred_traj_fake_rel_32 = generator_out_32
        pred_traj_fake_rel_final = generator_out_final

        pred_traj_fake_final = relative_to_abs(pred_traj_fake_rel_final, obs_traj_8[-1])
        if args.l2_loss_weight > 0:
            g_l2_loss_rel_4.append(args.l2_loss_weight * l2_loss(
                pred_traj_fake_rel_4,
                pred_traj_gt_rel_4,
                mode='raw'))
            g_l2_loss_rel_8.append(args.l2_loss_weight * l2_loss(
                pred_traj_fake_rel_8,
                pred_traj_gt_rel_8,
                mode='raw'))
            g_l2_loss_rel_16.append(args.l2_loss_weight * l2_loss(
                pred_traj_fake_rel_16,
                pred_traj_gt_rel_16,
                mode='raw'))
            g_l2_loss_rel_32.append(args.l2_loss_weight * l2_loss(
                pred_traj_fake_rel_32,
                pred_traj_gt_rel_32,
                mode='raw'))
            g_l2_loss_rel_final.append(args.l2_loss_weight * l2_loss(
                pred_traj_fake_rel_final,
                pred_traj_gt_rel_8,
                mode='raw'))

    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt_8)
    g_l2_loss_sum_rel_fusion = torch.zeros(1).to(pred_traj_gt_8)
    if args.l2_loss_weight > 0:
        g_l2_loss_rel_4 = torch.stack(g_l2_loss_rel_4, dim=1)
        g_l2_loss_rel_8 = torch.stack(g_l2_loss_rel_8, dim=1)
        g_l2_loss_rel_16 = torch.stack(g_l2_loss_rel_16, dim=1)
        g_l2_loss_rel_32 = torch.stack(g_l2_loss_rel_32, dim=1)
        g_l2_loss_rel_final = torch.stack(g_l2_loss_rel_final, dim=1)
        # FIXME variety  loss with min 1,
        for start, end in seq_start_end.data:
            _g_l2_loss_rel_4 = g_l2_loss_rel_4[start:end]
            _g_l2_loss_rel_8 = g_l2_loss_rel_8[start:end]
            _g_l2_loss_rel_16 = g_l2_loss_rel_16[start:end]
            _g_l2_loss_rel_32 = g_l2_loss_rel_32[start:end]
            _g_l2_loss_rel_final = g_l2_loss_rel_final[start:end]

            _g_l2_loss_rel_4 = torch.sum(_g_l2_loss_rel_4, dim=0)
            _g_l2_loss_rel_4 = torch.min(_g_l2_loss_rel_4) / (int(end-start) * float(pred_traj_fake_rel_4.shape[0]))
            _g_l2_loss_rel_8 = torch.sum(_g_l2_loss_rel_8, dim=0)
            _g_l2_loss_rel_8 = torch.min(_g_l2_loss_rel_8) / (int(end-start) * float(pred_traj_fake_rel_8.shape[0]))
            _g_l2_loss_rel_16 = torch.sum(_g_l2_loss_rel_16, dim=0)
            _g_l2_loss_rel_16 = torch.min(_g_l2_loss_rel_16) / (int(end-start) * float(pred_traj_fake_rel_16.shape[0]))
            _g_l2_loss_rel_32 = torch.sum(_g_l2_loss_rel_32, dim=0)
            _g_l2_loss_rel_32 = torch.min(_g_l2_loss_rel_32) / (int(end-start) * float(pred_traj_fake_rel_32.shape[0]))
            _g_l2_loss_rel_final = torch.sum(_g_l2_loss_rel_final, dim=0)
            _g_l2_loss_rel_final = torch.min(_g_l2_loss_rel_final) / (int(end-start) * float(pred_traj_fake_rel_8.shape[0]))

            g_l2_loss_sum_rel += (_g_l2_loss_rel_4*loss_weight[0] + _g_l2_loss_rel_8*loss_weight[1] + _g_l2_loss_rel_16*loss_weight[2] + _g_l2_loss_rel_32*loss_weight[3]) * loss_weight[4]
            g_l2_loss_sum_rel_fusion += _g_l2_loss_rel_final * loss_weight[5]
        losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
        losses['G_l2_loss_rel_fusion'] = g_l2_loss_sum_rel_fusion.item()
        loss += (g_l2_loss_sum_rel + g_l2_loss_sum_rel_fusion)

    traj_fake_final = torch.cat([obs_traj_8, pred_traj_fake_final], dim=0)
    traj_fake_rel_final = torch.cat([obs_traj_rel_8, pred_traj_fake_rel_final], dim=0)
    scores_fake_final = discriminator(traj_fake_final, traj_fake_rel_final, seq_start_end)
    discriminator_loss_final = g_loss_fn(scores_fake_final)
    loss += discriminator_loss_final
    losses['G_discriminator_loss'] = discriminator_loss_final.item()
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generator.parameters(), args.clipping_threshold_g
        )
    optimizer_g.step()
    return losses


def unstandardize(data, data_mean_x, data_std_x, data_mean_y, data_std_y):
    data_copy = np.copy(data.cpu().detach().numpy())
    data_copy[:, :, 0] = data_copy[:, :, 0] * data_std_x + data_mean_x
    data_copy[:, :, 1] = data_copy[:, :, 1] * data_std_y + data_mean_y
    data_copy = torch.FloatTensor(data_copy).cuda()
    return data_copy


def check_accuracy(
    args, loader, generator, discriminator, d_loss_fn, limit=False
):
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    generator.eval()
    with torch.no_grad():
        for batches in loader:
            # FIXME Multi Scale input
            (batch, batch_4, batch_8, batch_16, batch_32, mean_sta) = batches
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask,
             seq_start_end) = batch
            (traj_mean_x_8, traj_std_x_8, traj_mean_y_8, traj_std_y_8,
             traj_rel_mean_x_8, traj_rel_std_x_8, traj_rel_mean_y_8, traj_rel_std_y_8) = mean_sta
            obs_traj = obs_traj.cuda()
            pred_traj_gt = pred_traj_gt.cuda()
            obs_traj_rel = obs_traj_rel.cuda()
            pred_traj_gt_rel = pred_traj_gt_rel.cuda()
            loss_mask = loss_mask.cuda()
            batch_4 = [tensor.cuda() for tensor in batch_4]
            batch_8 = [tensor.cuda() for tensor in batch_8]
            batch_16 = [tensor.cuda() for tensor in batch_16]
            batch_32 = [tensor.cuda() for tensor in batch_32]
            (obs_traj_4, pred_traj_gt_4, obs_traj_rel_4, pred_traj_gt_rel_4, non_linear_ped,
             seq_start_end) = batch_4
            (obs_traj_8, pred_traj_gt_8, obs_traj_rel_8, pred_traj_gt_rel_8, non_linear_ped,
             seq_start_end) = batch_8
            (obs_traj_16, pred_traj_gt_16, obs_traj_rel_16, pred_traj_gt_rel_16, non_linear_ped,
             seq_start_end) = batch_16
            (obs_traj_32, pred_traj_gt_32, obs_traj_rel_32, pred_traj_gt_rel_32, non_linear_ped,
             seq_start_end) = batch_32
            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len:]

            generator_out_4, generator_out_8, generator_out_16, generator_out_32, generator_out_final = generator(
                obs_traj_4, obs_traj_rel_4, obs_traj_8, obs_traj_rel_8, obs_traj_16, obs_traj_rel_16, obs_traj_32,
                obs_traj_rel_32, seq_start_end, pred_traj_gt_4.shape[0])

            # FIXME unstandardize
            pred_traj_fake = relative_to_abs(
                generator_out_final, obs_traj_8[-1]
            )
            unstandar_pred_traj_fake = unstandardize(pred_traj_fake, traj_mean_x_8,
                                                     traj_std_x_8, traj_mean_y_8,
                                                     traj_std_y_8)
            unstandar_pred_traj_fake_rel = unstandardize(generator_out_final, traj_rel_mean_x_8, traj_rel_std_x_8, traj_rel_mean_y_8, traj_rel_std_y_8)

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, unstandar_pred_traj_fake,
                unstandar_pred_traj_fake_rel
            )
            ade, ade_l, ade_nl = cal_ade(
                pred_traj_gt, unstandar_pred_traj_fake, linear_ped, non_linear_ped
            )

            fde, fde_l, fde_nl = cal_fde(
                pred_traj_gt, unstandar_pred_traj_fake, linear_ped, non_linear_ped
            )

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, unstandar_pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, unstandar_pred_traj_fake_rel], dim=0)

            scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
            scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

            d_loss = d_loss_fn(scores_real, scores_fake)
            d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt_8.size(1)
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()
            if limit and total_traj >= args.num_samples_check:
                break

    metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj
    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0
    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (
            total_traj_nl * args.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    generator.train()
    return metrics


def cal_l2_losses(
    pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,
    loss_mask =None
):
    g_l2_loss_abs = l2_loss(
        pred_traj_fake, pred_traj_gt, loss_mask, mode='sum'
    )
    g_l2_loss_rel = l2_loss(
        pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum'
    )
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)  # 只计算线性轨迹的行人轨迹误差
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)  # 只计算非线性轨迹的行人轨迹误差
    return ade, ade_l, ade_nl


def cal_fde(
    pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fde_l = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], linear_ped
    )
    fde_nl = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped
    )
    return fde, fde_l, fde_nl


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
