import argparse
import os
import torch
import numpy as np
import random

from attrdict import AttrDict

from TPN.data.loader import data_loader_TPN_test
from TPN.models import TrajectoryGeneratorTPNPooling
from TPN.losses import displacement_error, final_displacement_error
from TPN.utils import relative_to_abs, get_dset_path


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='../models/TPN_P/TPN_P_with_Multi/zara2_TPN_P_with_Multi_with_model.pt', type=str)
parser.add_argument('--num_samples', default=20, type=int)  # FIXME default=20
parser.add_argument('--seed', help='manual seed to use, default is 321',
                    type=int, default=321)
parser.add_argument('--dset_type', default='test', type=str)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def get_generator_merge(checkpoint):
    args = AttrDict(checkpoint['args'])
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
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate(args, loader, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batches in loader:
            (batch, batch_4, batch_8, batch_16, batch_32, mean_sta) = batches
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
             loss_mask, seq_start_end) = batch
            (traj_mean_x_8, traj_std_x_8, traj_mean_y_8, traj_std_y_8,
             traj_rel_mean_x_8, traj_rel_std_x_8, traj_rel_mean_y_8, traj_rel_std_y_8) = mean_sta
            pred_traj_gt = pred_traj_gt.cuda()
            obs_traj_rel = obs_traj_rel.cuda()
            batch_4 = [tensor.cuda() for tensor in batch_4]
            batch_8 = [tensor.cuda() for tensor in batch_8]
            batch_16 = [tensor.cuda() for tensor in batch_16]
            batch_32 = [tensor.cuda() for tensor in batch_32]
            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)
            (obs_traj_4, pred_traj_gt_4, obs_traj_rel_4, pred_traj_gt_rel_4, non_linear_ped,
             seq_start_end) = batch_4
            (obs_traj_8, pred_traj_gt_8, obs_traj_rel_8, pred_traj_gt_rel_8, non_linear_ped,
             seq_start_end) = batch_8
            (obs_traj_16, pred_traj_gt_16, obs_traj_rel_16, pred_traj_gt_rel_16, non_linear_ped,
             seq_start_end) = batch_16
            (obs_traj_32, pred_traj_gt_32, obs_traj_rel_32, pred_traj_gt_rel_32, non_linear_ped,
             seq_start_end) = batch_32

            for _ in range(num_samples):
                generator_out_4, generator_out_8, generator_out_16, generator_out_32, generator_out_final = generator(
                    obs_traj_4, obs_traj_rel_4, obs_traj_8, obs_traj_rel_8, obs_traj_16, obs_traj_rel_16, obs_traj_32,
                    obs_traj_rel_32, seq_start_end, pred_traj_gt_4.shape[0])
                pred_traj_fake_rel_final = generator_out_final
                obsv_v = torch.sqrt(torch.sum(torch.pow(obs_traj_rel, 2), dim=2))
            #     # FIXME change the output when the history is stopping.
                history_error = torch.sum(obsv_v, dim=0) / 8
                pred_traj_fake_rel_final[:, history_error < 0.05] = 0.0   # FIXME setting the threshold

                # FIXME unstandardize
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel_final, obs_traj_8[-1]
                )
                unstandar_pred_traj_fake = unstandardize(pred_traj_fake, traj_mean_x_8,
                                                             traj_std_x_8, traj_mean_y_8,
                                                             traj_std_y_8)
                ade.append(displacement_error(
                    unstandar_pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    unstandar_pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde


def unstandardize(data, data_mean_x, data_std_x, data_mean_y, data_std_y):
    data_copy = np.copy(data.cpu())
    data_copy[:, :, 0] = data_copy[:, :, 0] * data_std_x + data_mean_x
    data_copy[:, :, 1] = data_copy[:, :, 1] * data_std_y + data_mean_y
    data_copy = torch.FloatTensor(data_copy).cuda()
    return data_copy


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        checkpoint = torch.load(path)
        generator = get_generator_merge(checkpoint)
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader_TPN_test(_args, path)  # FIXME data_loader_TPN_test for Spline Offline
        ade, fde = evaluate(_args, loader, generator, args.num_samples)
        print('Dataset: {}, Pred Len: {}, ADE: {:.3f}, FDE: {:.3f}'.format(
            _args.dataset_name, _args.pred_len, ade, fde))


if __name__ == '__main__':
    main(args)
