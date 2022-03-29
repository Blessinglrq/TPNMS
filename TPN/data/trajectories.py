import logging
import os
import math

import numpy as np
from scipy import interpolate

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def seq_collate_TPN(data):
    (batch, batch_4, batch_8, batch_16, batch_32, mean_sta) = zip(*data)
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list) = zip(*batch)
    (obs_seq_list_4, pred_seq_list_4, obs_seq_rel_list_4, pred_seq_rel_list_4,
     non_linear_ped_list) = zip(*batch_4)
    (obs_seq_list_8, pred_seq_list_8, obs_seq_rel_list_8, pred_seq_rel_list_8,
     non_linear_ped_list) = zip(*batch_8)
    (obs_seq_list_16, pred_seq_list_16, obs_seq_rel_list_16, pred_seq_rel_list_16,
     non_linear_ped_list) = zip(*batch_16)
    (obs_seq_list_32, pred_seq_list_32, obs_seq_rel_list_32, pred_seq_rel_list_32,
     non_linear_ped_list) = zip(*batch_32)
    (traj_mean_x_8, traj_std_x_8, traj_mean_y_8, traj_std_y_8,
             traj_rel_mean_x_8, traj_rel_std_x_8, traj_rel_mean_y_8, traj_rel_std_y_8) = zip(*mean_sta)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)  # pytorch 改变维度,使之符合LSTM输入
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    obs_traj_4 = torch.cat(obs_seq_list_4, dim=0).permute(2, 0, 1)  # pytorch 改变维度,使之符合LSTM输入
    pred_traj_4 = torch.cat(pred_seq_list_4, dim=0).permute(2, 0, 1)
    obs_traj_rel_4 = torch.cat(obs_seq_rel_list_4, dim=0).permute(2, 0, 1)
    pred_traj_rel_4 = torch.cat(pred_seq_rel_list_4, dim=0).permute(2, 0, 1)
    obs_traj_8 = torch.cat(obs_seq_list_8, dim=0).permute(2, 0, 1)  # pytorch 改变维度,使之符合LSTM输入
    pred_traj_8 = torch.cat(pred_seq_list_8, dim=0).permute(2, 0, 1)
    obs_traj_rel_8 = torch.cat(obs_seq_rel_list_8, dim=0).permute(2, 0, 1)
    pred_traj_rel_8 = torch.cat(pred_seq_rel_list_8, dim=0).permute(2, 0, 1)
    obs_traj_16 = torch.cat(obs_seq_list_16, dim=0).permute(2, 0, 1)  # pytorch 改变维度,使之符合LSTM输入
    pred_traj_16 = torch.cat(pred_seq_list_16, dim=0).permute(2, 0, 1)
    obs_traj_rel_16 = torch.cat(obs_seq_rel_list_16, dim=0).permute(2, 0, 1)
    pred_traj_rel_16 = torch.cat(pred_seq_rel_list_16, dim=0).permute(2, 0, 1)
    obs_traj_32 = torch.cat(obs_seq_list_32, dim=0).permute(2, 0, 1)  # pytorch 改变维度,使之符合LSTM输入
    pred_traj_32 = torch.cat(pred_seq_list_32, dim=0).permute(2, 0, 1)
    obs_traj_rel_32 = torch.cat(obs_seq_rel_list_32, dim=0).permute(2, 0, 1)
    pred_traj_rel_32 = torch.cat(pred_seq_rel_list_32, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    out_o = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped, loss_mask, seq_start_end]
    out_4 = [
        obs_traj_4, pred_traj_4, obs_traj_rel_4, pred_traj_rel_4, non_linear_ped, seq_start_end]
    out_8 = [
        obs_traj_8, pred_traj_8, obs_traj_rel_8, pred_traj_rel_8, non_linear_ped, seq_start_end]
    out_16 = [
        obs_traj_16, pred_traj_16, obs_traj_rel_16, pred_traj_rel_16, non_linear_ped, seq_start_end]
    out_32 = [
        obs_traj_32, pred_traj_32, obs_traj_rel_32, pred_traj_rel_32, non_linear_ped, seq_start_end]
    mean_sta = [traj_mean_x_8[0], traj_std_x_8[0], traj_mean_y_8[0], traj_std_y_8[0],
                traj_rel_mean_x_8[0], traj_rel_std_x_8[0], traj_rel_mean_y_8[0], traj_rel_std_y_8[0]]

    return tuple(out_o), tuple(out_4), tuple(out_8), tuple(out_16), tuple(out_32), tuple(mean_sta)


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)  # 返回在指定范围内的均匀间隔的数字（组成的数组），也即返回一个等差数列(起始点,结束点,元素个数)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]  # 二次多项式拟合 如果轨迹不是线性分布的 则定义为非线性轨迹
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def standardize(data):
    data_copy = np.copy(data)
    data_mean_x = np.mean(data_copy[:, 0])
    data_std_x = np.std(data_copy[:, 0])
    data_copy[:, 0] = (data_copy[:, 0]-data_mean_x)/data_std_x
    data_mean_y = np.mean(data_copy[:, 1])
    data_std_y = np.std(data_copy[:, 1])
    data_copy[:, 1] = (data_copy[:, 1]-data_mean_y)/data_std_y
    return data_copy, data_mean_x, data_std_x, data_mean_y, data_std_y


def spline_interpolation(obs_len, traj_pos, traj_pos_rel, inter_scale=0.5):
    if inter_scale <= 1:
        traj_pos_new = torch.FloatTensor(traj_pos.shape[0], traj_pos.shape[1], int(traj_pos.shape[2] * inter_scale)).numpy()
        traj_pos_new = traj_pos[:, :, [int(1/inter_scale) * i for i in range(int(traj_pos.shape[2]*inter_scale))]]
        obs_traj_new = traj_pos_new[:, :, :int(obs_len * inter_scale)]
        pred_traj_gt_new = traj_pos_new[:, :, int(obs_len * inter_scale):]
    else:
        traj_pos_new = torch.FloatTensor(traj_pos.shape[0], traj_pos.shape[1], int((traj_pos.shape[2]-1) * inter_scale + 1)).numpy()
        timestep = torch.linspace(1, traj_pos.shape[2], steps=traj_pos.shape[2], out=None).numpy()
        new_timestep = torch.linspace(1, traj_pos.shape[2], steps=(traj_pos_new.shape[2]), out=None).numpy()
        for i in range(traj_pos_rel.shape[0]):
            f_x = interpolate.interp1d(timestep, traj_pos[:, 0][i], kind='cubic')  # Fixme ['linear','zero', 'slinear', 'quadratic', 'cubic', 4, 5]
            traj_pos_new[i, 0] = f_x(new_timestep)
            f_y = interpolate.interp1d(timestep, traj_pos[:, 1][i], kind='cubic')
            traj_pos_new[i, 1] = f_y(new_timestep)
        obs_traj_new = traj_pos_new[:, :, :int((obs_len-1) * inter_scale + 1)]
        pred_traj_gt_new = traj_pos_new[:, :, int((obs_len-1) * inter_scale + 1):]
    traj_pos_rel_new = torch.zeros(traj_pos_new.shape).numpy()
    traj_pos_rel_new[:, :, 1:] = \
        traj_pos_new[:, :, 1:] - traj_pos_new[:, :, :-1]  # 坐标位置相减得到速度，默认第一个位置的速度为原始的速度
    obs_traj_rel_new = traj_pos_rel_new[:, :, :int((obs_len-1) * inter_scale + 1)]
    pred_traj_gt_rel = traj_pos_rel_new[:, :, int((obs_len-1) * inter_scale + 1):]
    out = [obs_traj_new, pred_traj_gt_new, obs_traj_rel_new, pred_traj_gt_rel]
    return out


class TrajectoryDatasetTPN(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDatasetTPN, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()  # 提取帧数, np.unique() 保留数组中不同的值
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])  # 提取相同帧的行人位置信息
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))  # 有效的输入序列数量

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)  # 提取当前帧相关的20帧数据
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])  # 提取当前帧相关的20帧数据行人id
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]  # 提取当前帧相关的20帧数据中个行人的位置信息,不一定都出现20次
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)  # 返回四舍五入后的值，指定精度4小数位数
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx  # 索引到帧数对应的索引
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue        # 20帧中，如果行人出现的帧数不到20，则不满足目标行人的条件，pass
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])  # 每一列表示不同帧的位置信息
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]  # 坐标位置相减得到速度，默认第一个位置的速度为0
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq  # 对满足条件的行人的位置信息放入20帧的行人序列中，不满足的为0
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1  # 1 表示当前索引在各位置有人
                    num_peds_considered += 1
                # 将不同序列得信息整合起来
                if num_peds_considered > min_ped:  # 如果当前20帧序列满足条件得行人数大于1
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)  # 满足条件得序列得数量
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)
        seq_list_sta, self.seq_list_sta_mean_x, self.seq_list_sta_std_x, self.seq_list_sta_mean_y, self.seq_list_sta_std_y = standardize(
            seq_list)
        seq_list_sta_rel, self.seq_list_sta_rel_mean_x, self.seq_list_sta_rel_std_x, self.seq_list_sta_rel_mean_y, self.seq_list_sta_rel_std_y = standardize(
            seq_list_rel)
        interpolation_4 = spline_interpolation(self.obs_len, seq_list_sta, seq_list_sta_rel, inter_scale=0.5)
        interpolation_8 = spline_interpolation(self.obs_len, seq_list_sta, seq_list_sta_rel, inter_scale=1)
        interpolation_16 = spline_interpolation(self.obs_len, seq_list_sta, seq_list_sta_rel, inter_scale=2)
        interpolation_32 = spline_interpolation(self.obs_len, seq_list_sta, seq_list_sta_rel, inter_scale=4)
        (obs_traj_list_4, pred_traj_list_gt_4, obs_traj_rel_list_4, pred_traj_gt_rel_list_4) = interpolation_4
        (obs_traj_list_8, pred_traj_list_gt_8, obs_traj_rel_list_8, pred_traj_gt_rel_list_8) = interpolation_8
        (obs_traj_list_16, pred_traj_list_gt_16, obs_traj_rel_list_16, pred_traj_gt_rel_list_16) = interpolation_16
        (obs_traj_list_32, pred_traj_list_gt_32, obs_traj_rel_list_32, pred_traj_gt_rel_list_32) = interpolation_32
        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_4 = torch.from_numpy(obs_traj_list_4).type(torch.float)
        self.pred_traj_4 = torch.from_numpy(pred_traj_list_gt_4).type(torch.float)
        self.obs_traj_rel_4 = torch.from_numpy(obs_traj_rel_list_4).type(torch.float)
        self.pred_traj_rel_4 = torch.from_numpy(pred_traj_gt_rel_list_4).type(torch.float)
        self.obs_traj_8 = torch.from_numpy(obs_traj_list_8).type(torch.float)
        self.pred_traj_8 = torch.from_numpy(pred_traj_list_gt_8).type(torch.float)
        self.obs_traj_rel_8 = torch.from_numpy(obs_traj_rel_list_8).type(torch.float)
        self.pred_traj_rel_8 = torch.from_numpy(pred_traj_gt_rel_list_8).type(torch.float)
        self.obs_traj_16 = torch.from_numpy(obs_traj_list_16).type(torch.float)
        self.pred_traj_16 = torch.from_numpy(pred_traj_list_gt_16).type(torch.float)
        self.obs_traj_rel_16 = torch.from_numpy(obs_traj_rel_list_16).type(torch.float)
        self.pred_traj_rel_16 = torch.from_numpy(pred_traj_gt_rel_list_16).type(torch.float)
        self.obs_traj_32 = torch.from_numpy(obs_traj_list_32).type(torch.float)
        self.pred_traj_32 = torch.from_numpy(pred_traj_list_gt_32).type(torch.float)
        self.obs_traj_rel_32 = torch.from_numpy(obs_traj_rel_list_32).type(torch.float)
        self.pred_traj_rel_32 = torch.from_numpy(pred_traj_gt_rel_list_32).type(torch.float)

        #
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()  # np.cumsum() 累加，得到每个序列开始得行人得idx
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]


    def __len__(self):
        return self.num_seq
    # 返回得是 满足条件得行人得历史轨迹和未来轨迹 以及对应得速度和是否为线性轨迹,不满足条件得不返回

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out_o = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :]
        ]
        out_4 = [
            self.obs_traj_4[start:end, :], self.pred_traj_4[start:end, :],
            self.obs_traj_rel_4[start:end, :], self.pred_traj_rel_4[start:end, :],
            self.non_linear_ped[start:end]
        ]
        out_8 = [
            self.obs_traj_8[start:end, :], self.pred_traj_8[start:end, :],
            self.obs_traj_rel_8[start:end, :], self.pred_traj_rel_8[start:end, :],
            self.non_linear_ped[start:end]
        ]
        out_16 = [
            self.obs_traj_16[start:end, :], self.pred_traj_16[start:end, :],
            self.obs_traj_rel_16[start:end, :], self.pred_traj_rel_16[start:end, :],
            self.non_linear_ped[start:end]
        ]
        out_32 = [
            self.obs_traj_32[start:end, :], self.pred_traj_32[start:end, :],
            self.obs_traj_rel_32[start:end, :], self.pred_traj_rel_32[start:end, :],
            self.non_linear_ped[start:end]
        ]
        mean_sta = [self.seq_list_sta_mean_x, self.seq_list_sta_std_x, self.seq_list_sta_mean_y, self.seq_list_sta_std_y, self.seq_list_sta_rel_mean_x, self.seq_list_sta_rel_std_x, self.seq_list_sta_rel_mean_y, self.seq_list_sta_rel_std_y]
        return [out_o, out_4, out_8, out_16, out_32, mean_sta]

