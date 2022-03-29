from torch.utils.data import DataLoader
from TPN.data.trajectories import TrajectoryDatasetTPN, seq_collate_TPN


def data_loader_TPN(args, path):
    dset = TrajectoryDatasetTPN(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate_TPN)
    return dset, loader


def data_loader_TPN_test(args, path):
    dset = TrajectoryDatasetTPN(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate_TPN)
    return dset, loader