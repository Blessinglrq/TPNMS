import torch
import torch.nn as nn
from TPN.utils import relative_to_abs


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


def interpolation_for_pred(pred_traj_rel, last_pos, inter_scale=0.5):
    if pred_traj_rel.shape[0] == 6 and inter_scale == 2:
        pred_traj_pos_new = torch.FloatTensor(int(pred_traj_rel.shape[0] * inter_scale), pred_traj_rel.shape[1], pred_traj_rel.shape[2]).cuda()
        pred_traj_pos_new[[int(inter_scale) * i for i in range(int(pred_traj_rel.shape[0]))]] = pred_traj_rel
        last_step = (pred_traj_rel[-1] + (pred_traj_rel[-1] - pred_traj_rel[-2])).unsqueeze(0)
        pred_traj_rel = torch.cat((pred_traj_rel, last_step), dim=0)
        pred_traj_pos_new[[int(inter_scale) * i + 1 for i in range(int(pred_traj_rel.shape[0] - 1))]] = (pred_traj_rel[:-1] + pred_traj_rel[1:]) / inter_scale
    elif inter_scale == 2:
        pred_traj_pos_new = torch.FloatTensor(int((pred_traj_rel.shape[0]) * inter_scale), pred_traj_rel.shape[1], pred_traj_rel.shape[2]).cuda()
        pred_traj_rel = torch.cat((last_pos.unsqueeze(0), pred_traj_rel), dim=0)
        # pred_traj_pos_new[[int(inter_scale) * i for i in range(int(pred_traj_rel.shape[0]))]] = pred_traj_rel
        pred_traj_pos_new[[int(inter_scale) * i for i in range(int(pred_traj_rel.shape[0] - 1))]] = (pred_traj_rel[:-1] + pred_traj_rel[1:]) / inter_scale
        pred_traj_pos_new[[int(inter_scale) * i + 1 for i in range(int(pred_traj_rel.shape[0]-1))]] = pred_traj_rel[1:]
    elif inter_scale <= 1:
        pred_traj_pos_new = pred_traj_rel[[int(1 / inter_scale) * (i+1)-1 for i in range(int(pred_traj_rel.shape[0] * inter_scale))]]
    return pred_traj_pos_new


class Encoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
        dropout=0.0
    ):
        super(Encoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        self.spatial_embedding = nn.Linear(2, embedding_dim)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(
            -1, batch, self.embedding_dim
        )
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        # final_h = state[0]  # state,tuple(h,c)  ---> h.shape = (self.num_layers, batch, self.h_dim)
        final_h = output  # output[-1] == state[0]
        return final_h


class DecoderMulti(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, pooling_type='pool_net',
        neighborhood_size=2.0, grid_size=8
    ):
        super(DecoderMulti, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep

        self.decoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        if pool_every_timestep:
            if pooling_type == 'pool_net':
                self.pool_net = PoolHiddenNet(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout
                )
            elif pooling_type == 'spool':
                self.pool_net = SocialPooling(
                    h_dim=self.h_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    neighborhood_size=neighborhood_size,
                    grid_size=grid_size
                )

            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, hidden_state, last_pos, last_pos_rel, state_tuple, seq_start_end, seq_len):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        """
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for _ in range(seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos

            if self.pool_every_timestep:
                # decoder_h = state_tuple[0]
                hidden_state = torch.cat([hidden_state[1:], output], dim=0)
                pool_h = self.pool_net(hidden_state, seq_start_end, curr_pos)
                decoder_h = torch.cat(
                    [hidden_state[-1].view(-1, self.h_dim), pool_h], dim=1)
                decoder_h = self.mlp(decoder_h)
                decoder_h = torch.unsqueeze(decoder_h, 0)
                state_tuple = (decoder_h, state_tuple[1])

            embedding_input = rel_pos

            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]


class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 128, bottleneck_dim]

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            # Repeat -> [H1, H2, H3, ...][H1, H2, H3, ...]...
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            # Repeat position -> [P1, P2, P3, ...][P1, P2, P3, ...]...
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            # Repeat position -> [P1, P1, P1, ...][P2, P2, P2, ...]...
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2  # 得到行人的end_pos间的相对关系，并交给感知机去具体处理。每个行人与其他行人的相对位置关系由num_ped项，合计有num_ped**2项。
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            # 这里得maxpooling 分别取对于不同行人得每一个历史帧得最大(影响力最大)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]  # 返回每一行中最大值的那个元素，且返回索引,只取元素值
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)  # (num_ped,8)--->(all_num_ped,8)
        return pool_h


class SocialPooling(nn.Module):
    """Current state of the art pooling mechanism:
    http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf"""
    # FIXME 暂时看不懂
    def __init__(
        self, h_dim=64, activation='relu', batch_norm=True, dropout=0.0,
        neighborhood_size=2.0, grid_size=8, pool_dim=None
    ):
        super(SocialPooling, self).__init__()
        self.h_dim = h_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        if pool_dim:
            mlp_pool_dims = [grid_size * grid_size * h_dim, pool_dim]
        else:
            mlp_pool_dims = [grid_size * grid_size * h_dim, h_dim]

        self.mlp_pool = make_mlp(
            mlp_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def get_bounds(self, ped_pos):
        top_left_x = ped_pos[:, 0] - self.neighborhood_size / 2
        top_left_y = ped_pos[:, 1] + self.neighborhood_size / 2
        bottom_right_x = ped_pos[:, 0] + self.neighborhood_size / 2
        bottom_right_y = ped_pos[:, 1] - self.neighborhood_size / 2
        top_left = torch.stack([top_left_x, top_left_y], dim=1)
        bottom_right = torch.stack([bottom_right_x, bottom_right_y], dim=1)
        return top_left, bottom_right

    def get_grid_locations(self, top_left, other_pos):
        cell_x = torch.floor(
            ((other_pos[:, 0] - top_left[:, 0]) / self.neighborhood_size) *
            self.grid_size)  # 床函数向下取整，得到x方向的网格数
        cell_y = torch.floor(
            ((top_left[:, 1] - other_pos[:, 1]) / self.neighborhood_size) *
            self.grid_size)
        grid_pos = cell_x + cell_y * self.grid_size  # ???
        return grid_pos

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tesnsor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - end_pos: Absolute end position of obs_traj (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, h_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            grid_size = self.grid_size * self.grid_size
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_hidden_repeat = curr_hidden.repeat(num_ped, 1)
            curr_end_pos = end_pos[start:end]
            curr_pool_h_size = (num_ped * grid_size) + 1
            curr_pool_h = curr_hidden.new_zeros((curr_pool_h_size, self.h_dim))
            # curr_end_pos = curr_end_pos.data
            top_left, bottom_right = self.get_bounds(curr_end_pos)

            # Repeat position -> P1, P2, P1, P2
            curr_end_pos = curr_end_pos.repeat(num_ped, 1)
            # Repeat bounds -> B1, B1, B2, B2
            top_left = self.repeat(top_left, num_ped)
            bottom_right = self.repeat(bottom_right, num_ped)

            grid_pos = self.get_grid_locations(
                    top_left, curr_end_pos).type_as(seq_start_end)
            # Make all positions to exclude as non-zero
            # Find which peds to exclude
            x_bound = ((curr_end_pos[:, 0] >= bottom_right[:, 0]) +
                       (curr_end_pos[:, 0] <= top_left[:, 0]))
            y_bound = ((curr_end_pos[:, 1] >= top_left[:, 1]) +
                       (curr_end_pos[:, 1] <= bottom_right[:, 1]))

            within_bound = x_bound + y_bound
            within_bound[0::num_ped + 1] = 1  # Don't include the ped itself
            within_bound = within_bound.view(-1)

            # This is a tricky way to get scatter add to work. Helps me avoid a
            # for loop. Offset everything by 1. Use the initial 0 position to
            # dump all uncessary adds.
            grid_pos += 1
            total_grid_size = self.grid_size * self.grid_size
            offset = torch.arange(
                0, total_grid_size * num_ped, total_grid_size
            ).type_as(seq_start_end)

            offset = self.repeat(offset.view(-1, 1), num_ped).view(-1)
            grid_pos += offset
            grid_pos[within_bound != 0] = 0
            grid_pos = grid_pos.view(-1, 1).expand_as(curr_hidden_repeat)

            curr_pool_h = curr_pool_h.scatter_add(0, grid_pos,
                                                  curr_hidden_repeat)
            curr_pool_h = curr_pool_h[1:]
            pool_h.append(curr_pool_h.view(num_ped, -1))

        pool_h = torch.cat(pool_h, dim=0)
        pool_h = self.mlp_pool(pool_h)
        return pool_h


class MergeModelConv(torch.nn.Module):
    def __init__(self):
        super(MergeModelConv, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, kernel_size=1, stride=1, bias=True)
        # self.relu1 = torch.nn.LeakyReLU()
        self.conv2 = torch.nn.Conv2d(8, 4, kernel_size=1, stride=1, bias=True)
        # self.relu2 = torch.nn.LeakyReLU()
        self.conv3 = torch.nn.Conv2d(4, 1, kernel_size=1, stride=1, bias=True)
        # self.relu3 = torch.nn.LeakyReLU()

    def forward(self, generator_merge):
        x = self.conv1(generator_merge)
        # x = self.relu1(x)  # RELU is unfair for negative number,
        x = self.conv2(x)
        # x = self.relu2(x)
        x = self.conv3(x)
        # x = self.relu3(x)
        return x


class TrajectoryGeneratorTPNPooling(nn.Module):
    def __init__(
        self, obs_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='leakyrelu', batch_norm=True, neighborhood_size=2.0, grid_size=8
    ):
            super(TrajectoryGeneratorTPNPooling, self).__init__()

            if pooling_type and pooling_type.lower() == 'none':
                pooling_type = None

            self.obs_len = obs_len
            self.mlp_dim = mlp_dim
            self.encoder_h_dim = encoder_h_dim
            self.decoder_h_dim = decoder_h_dim
            self.embedding_dim = embedding_dim
            self.noise_dim = noise_dim
            self.num_layers = num_layers
            self.noise_type = noise_type
            self.noise_mix_type = noise_mix_type
            self.pooling_type = pooling_type
            self.noise_first_dim = 0
            self.pool_every_timestep = pool_every_timestep
            self.bottleneck_dim = bottleneck_dim
            # self.da = 64

            self.encoder = nn.LSTM(
                embedding_dim, decoder_h_dim, num_layers, dropout=dropout
            )
            self.decoder = nn.LSTM(
                embedding_dim, decoder_h_dim, num_layers, dropout=dropout
            )
            self.spatial_embedding = nn.Linear(2, embedding_dim)
            self.hidden2pos = nn.Linear(decoder_h_dim, 2)

            if pooling_type == 'pool_net':
                self.pool_net = PoolHiddenNet(
                    embedding_dim=self.embedding_dim,
                    h_dim=encoder_h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm
                )
            elif pooling_type == 'spool':
                self.pool_net = SocialPooling(
                    h_dim=encoder_h_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    neighborhood_size=neighborhood_size,
                    grid_size=grid_size
                )

            if self.noise_dim[0] == 0:
                self.noise_dim = None
            else:
                self.noise_first_dim = noise_dim[0]
            # Decoder Hidden
            if pooling_type:
                input_dim = encoder_h_dim + bottleneck_dim
            else:
                input_dim = encoder_h_dim

            if self.mlp_decoder_needed():
                mlp_decoder_context_dims = [
                        input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
                    ]
                self.mlp_decoder_context = make_mlp(
                    mlp_decoder_context_dims,
                    activation=activation,
                    batch_norm=False,
                    dropout=dropout)

            self.mlp_4 = nn.Sequential(
                nn.Linear(6, 24),
                nn.LeakyReLU(inplace=True),
                nn.Linear(24, 12))

            self.mlp_8 = nn.Sequential(
                nn.Linear(12, 48),
                nn.LeakyReLU(inplace=True),
                nn.Linear(48, 24))

            self.mlp_16 = nn.Sequential(
                nn.Linear(24, 96),
                nn.LeakyReLU(inplace=True),
                nn.Linear(96, 48))

            self.merge_model = MergeModelConv().cuda()
            if pool_every_timestep:
                self.mlp = nn.Sequential(
                    nn.Linear(decoder_h_dim + bottleneck_dim, mlp_dim),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(mlp_dim, decoder_h_dim),
                    nn.LeakyReLU(inplace=True)
                )

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.decoder_h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.decoder_h_dim).cuda()
        )

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def Encoder(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(
            -1, batch, self.embedding_dim
        )
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]  # state,tuple(h,c)  ---> h.shape = (self.num_layers, batch, self.h_dim)
        return final_h

    def add_noise(self, _input, seq_start_end, user_noise=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == 'global':
            noise_shape = (seq_start_end.size(0), ) + self.noise_dim   # 每个序列对应一个8维的噪声
        else:
            noise_shape = (_input.size(0), ) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)  # 同一场景下，不同行人的噪声向量一致
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)  # 所有行人的噪声向量都不一样

        return decoder_h

    def mlp_decoder_needed(self):
        if (
            self.noise_dim or self.pooling_type or
            self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False

    def DecoderMulti(self, last_pos, last_pos_rel, state_tuple, seq_start_end, seq_len):
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for _ in range(seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.decoder_h_dim))
            curr_pos = rel_pos + last_pos

            if self.pool_every_timestep:
                decoder_h = state_tuple[0]
                pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos)
                decoder_h = torch.cat(
                    [output.view(-1, self.decoder_h_dim), pool_h], dim=1)
                decoder_h = self.mlp(decoder_h)
                decoder_h = torch.unsqueeze(decoder_h, 0)
                state_tuple = (decoder_h, state_tuple[1])

            embedding_input = rel_pos

            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]


    def forward(self, obs_traj_4, obs_traj_rel_4, obs_traj_8, obs_traj_rel_8, obs_traj_16, obs_traj_rel_16, obs_traj_32, obs_traj_rel_32, seq_start_end, seq_len, user_noise=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel_8.size(1)
        # Encode seq
        final_encoder_h_4 = self.Encoder(obs_traj_rel_4)  # (self.num_layers, batch, self.h_dim)
        final_encoder_h_8 = self.Encoder(obs_traj_rel_8)
        final_encoder_h_16 = self.Encoder(obs_traj_rel_16)
        final_encoder_h_32 = self.Encoder(obs_traj_rel_32)
        # Pool States
        if self.pooling_type:
            end_pos_4 = obs_traj_4[-1, :, :]
            end_pos_8 = obs_traj_8[-1, :, :]
            end_pos_16 = obs_traj_16[-1, :, :]
            end_pos_32 = obs_traj_32[-1, :, :]

            final_attention_encoder_h_4 = self.pool_net(final_encoder_h_4, seq_start_end, end_pos_4)
            final_attention_encoder_h_8 = self.pool_net(final_encoder_h_8, seq_start_end, end_pos_8)
            final_attention_encoder_h_16 = self.pool_net(final_encoder_h_16, seq_start_end, end_pos_16)
            final_attention_encoder_h_32 = self.pool_net(final_encoder_h_32, seq_start_end, end_pos_32)
            # Construct input hidden states for decoder
            mlp_decoder_context_input_4 = torch.cat(
                [final_encoder_h_4.view(-1, self.encoder_h_dim), final_attention_encoder_h_4], dim=1)
            mlp_decoder_context_input_8 = torch.cat(
                [final_encoder_h_8.view(-1, self.encoder_h_dim), final_attention_encoder_h_8], dim=1)
            mlp_decoder_context_input_16 = torch.cat(
                [final_encoder_h_16.view(-1, self.encoder_h_dim), final_attention_encoder_h_16], dim=1)
            mlp_decoder_context_input_32 = torch.cat(
                [final_encoder_h_32.view(-1, self.encoder_h_dim), final_attention_encoder_h_32], dim=1)
        else:
            mlp_decoder_context_input_4 = final_encoder_h_4.view(
                -1, self.encoder_h_dim)
            mlp_decoder_context_input_8 = final_encoder_h_8.view(
                -1, self.encoder_h_dim)
            mlp_decoder_context_input_16 = final_encoder_h_16.view(
                -1, self.encoder_h_dim)
            mlp_decoder_context_input_32 = final_encoder_h_32.view(
                -1, self.encoder_h_dim)

        # Add Noise
        if self.mlp_decoder_needed():
            noise_input_4 = self.mlp_decoder_context(mlp_decoder_context_input_4)
            noise_input_8 = self.mlp_decoder_context(mlp_decoder_context_input_8)
            noise_input_16 = self.mlp_decoder_context(mlp_decoder_context_input_16)
            noise_input_32 = self.mlp_decoder_context(mlp_decoder_context_input_32)
        else:
            noise_input_4 = mlp_decoder_context_input_4
            noise_input_8 = mlp_decoder_context_input_8
            noise_input_16 = mlp_decoder_context_input_16
            noise_input_32 = mlp_decoder_context_input_32
        decoder_h_4 = self.add_noise(
            noise_input_4, seq_start_end, user_noise=user_noise)  # 将隐藏状态加上噪声向量
        decoder_h_4 = decoder_h_4.repeat(self.num_layers, 1, 1)  # 5.20 add num_layer

        decoder_h_8 = self.add_noise(
            noise_input_8, seq_start_end, user_noise=user_noise)  # 将隐藏状态加上噪声向量
        decoder_h_8 = decoder_h_8.repeat(self.num_layers, 1, 1)  # 5.20 add num_layer
        decoder_h_16 = self.add_noise(
            noise_input_16, seq_start_end, user_noise=user_noise)  # 将隐藏状态加上噪声向量
        decoder_h_16 = decoder_h_16.repeat(self.num_layers, 1, 1)  # 5.20 add num_layer
        decoder_h_32 = self.add_noise(
            noise_input_32, seq_start_end, user_noise=user_noise)  # 将隐藏状态加上噪声向量
        decoder_h_32 = decoder_h_32.repeat(self.num_layers, 1, 1)  # 5.20 add num_layer

        decoder_c = torch.zeros(
            self.num_layers, batch, self.decoder_h_dim
        ).cuda()

        state_tuple_4 = (decoder_h_4, decoder_c)
        state_tuple_8 = (decoder_h_8, decoder_c)
        state_tuple_16 = (decoder_h_16, decoder_c)
        state_tuple_32 = (decoder_h_32, decoder_c)
        last_pos_4 = obs_traj_4[-1]
        last_pos_rel_4 = obs_traj_rel_4[-1]
        last_pos_8 = obs_traj_8[-1]
        last_pos_rel_8 = obs_traj_rel_8[-1]
        last_pos_16 = obs_traj_16[-1]
        last_pos_rel_16 = obs_traj_rel_16[-1]
        last_pos_32 = obs_traj_32[-1]
        last_pos_rel_32 = obs_traj_rel_32[-1]
        # Predict Trajectory

        decoder_out_4 = self.DecoderMulti(
            last_pos_4,
            last_pos_rel_4,
            state_tuple_4,
            seq_start_end,
            seq_len,
        )
        decoder_out_8 = self.DecoderMulti(
            last_pos_8,
            last_pos_rel_8,
            state_tuple_8,
            seq_start_end,
            seq_len*2,
        )
        decoder_out_16 = self.DecoderMulti(
            last_pos_16,
            last_pos_rel_16,
            state_tuple_16,
            seq_start_end,
            seq_len * 4,
        )
        decoder_out_32 = self.DecoderMulti(
            last_pos_32,
            last_pos_rel_32,
            state_tuple_32,
            seq_start_end,
            seq_len * 8,
        )
        pred_traj_fake_rel_4, final_decoder_h_4 = decoder_out_4
        pred_traj_fake_rel_8, final_decoder_h_8 = decoder_out_8
        pred_traj_fake_rel_16, final_decoder_h_16 = decoder_out_16
        pred_traj_fake_rel_32, final_decoder_h_32 = decoder_out_32

        rel_curr_ped_seq_4_o = self.mlp_4(pred_traj_fake_rel_4.permute(1, 2, 0)).permute(2, 0, 1)

        rel_curr_ped_seq_4_m = rel_curr_ped_seq_4_o.unsqueeze(0)  # 坐标位置相减得到速度，默认第一个位置的速度为0

        rel_curr_ped_seq_8 = (pred_traj_fake_rel_8 + rel_curr_ped_seq_4_o)
        rel_curr_ped_seq_8_o = self.mlp_8(rel_curr_ped_seq_8.permute(1, 2, 0)).permute(2, 0, 1)
        pred_traj_fake_8 = relative_to_abs(rel_curr_ped_seq_8, last_pos_8)
        rel_curr_ped_seq_16 = (pred_traj_fake_rel_16 + rel_curr_ped_seq_8_o)
        rel_curr_ped_seq_16_o = self.mlp_16(rel_curr_ped_seq_16.permute(1, 2, 0)).permute(2, 0, 1)
        pred_traj_fake_16 = relative_to_abs(rel_curr_ped_seq_16, last_pos_16)

        rel_curr_ped_seq_32 = (pred_traj_fake_rel_32 + rel_curr_ped_seq_16_o)
        pred_traj_fake_32 = relative_to_abs(rel_curr_ped_seq_32, last_pos_32)
        generator_inter_32 = interpolation_for_pred(pred_traj_fake_32, last_pos_32, inter_scale=0.25)
        generator_inter_32 = torch.cat((last_pos_32.unsqueeze(0), generator_inter_32), 0)
        rel_curr_ped_seq_32_m = (generator_inter_32[1:] - generator_inter_32[:-1]).unsqueeze(0)

        generator_inter_8_o = interpolation_for_pred(pred_traj_fake_8, last_pos_8, inter_scale=1)
        generator_inter_8_o = torch.cat((last_pos_8.unsqueeze(0), generator_inter_8_o), 0)
        rel_curr_ped_seq_8_m = (generator_inter_8_o[1:] - generator_inter_8_o[:-1]).unsqueeze(0)

        generator_inter_16_o = interpolation_for_pred(pred_traj_fake_16, last_pos_16, inter_scale=0.5)
        generator_inter_16_o = torch.cat((last_pos_16.unsqueeze(0), generator_inter_16_o), 0)
        rel_curr_ped_seq_16_m = (generator_inter_16_o[1:] - generator_inter_16_o[:-1]).unsqueeze(0)

        generator_out_merge = torch.cat((rel_curr_ped_seq_4_m, rel_curr_ped_seq_8_m, rel_curr_ped_seq_16_m, rel_curr_ped_seq_32_m), dim=0)
        # FIXME merge result
        generator_out_merge = generator_out_merge.permute(2, 0, 1, 3)
        generator_out_final = self.merge_model(generator_out_merge).squeeze(1)
        generator_out_final = generator_out_final.permute(1, 0, 2)

        return pred_traj_fake_rel_4, rel_curr_ped_seq_8, rel_curr_ped_seq_16, rel_curr_ped_seq_32, generator_out_final


class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='leakyrelu', batch_norm=True, dropout=0.0,
        d_type='local'
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        self.real_classifier = nn.Sequential(
            nn.Linear(h_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(mlp_dim, 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm
            )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        final_h = self.encoder(traj_rel)
        if self.d_type == 'local':
            classifier_input = final_h[-1].squeeze()
        else:
            classifier_input = self.pool_net(
                final_h[-1].squeeze(), seq_start_end, traj[0]
            )
        scores = self.real_classifier(classifier_input)
        return scores
