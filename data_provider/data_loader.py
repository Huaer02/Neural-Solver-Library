import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
import scipy.io as scio
from data_provider.shapenet_utils import get_datalist
from data_provider.shapenet_utils import GraphDataset
from torch.utils.data import Dataset
from utils.normalizer import UnitTransformer


class plas(object):
    def __init__(self, args):
        self.DATA_PATH = args.data_path + '/plas_N987_T20.mat'
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.out_dim = args.out_dim
        self.T_out = args.T_out
        self.normalize = args.normalize

    def random_collate_fn(self, batch):
        shuffled_batch = []
        shuffled_u = None
        shuffled_t = None
        shuffled_a = None
        shuffled_pos = None
        for item in batch:
            pos = item[0]
            t = item[1]
            a = item[2]
            u = item[3]

            num_timesteps = t.size(0)
            permuted_indices = torch.randperm(num_timesteps)
            t = t[permuted_indices]
            u = u.reshape(u.shape[0], num_timesteps, -1)[..., permuted_indices, :].reshape(u.shape[0], -1)

            if shuffled_t is None:
                shuffled_pos = pos.unsqueeze(0)
                shuffled_t = t.unsqueeze(0)
                shuffled_u = u.unsqueeze(0)
                shuffled_a = a.unsqueeze(0)
            else:
                shuffled_pos = torch.cat((shuffled_pos, pos.unsqueeze(0)), 0)
                shuffled_t = torch.cat((shuffled_t, t.unsqueeze(0)), 0)
                shuffled_u = torch.cat((shuffled_u, u.unsqueeze(0)), 0)
                shuffled_a = torch.cat((shuffled_a, a.unsqueeze(0)), 0)

        shuffled_batch.append(shuffled_pos)
        shuffled_batch.append(shuffled_t)
        shuffled_batch.append(shuffled_a)
        shuffled_batch.append(shuffled_u)

        return shuffled_batch  # B N T 4

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley
        s1 = int(((101 - 1) / r1) + 1)
        s2 = int(((31 - 1) / r2) + 1)

        data = scio.loadmat(self.DATA_PATH)
        input = torch.tensor(data['input'], dtype=torch.float)
        output = torch.tensor(data['output'], dtype=torch.float)
        print(input.shape, output.shape)
        x_train = input[:self.ntrain, ::r1][:, :s1].reshape(self.ntrain, s1, 1).repeat(1, 1, s2)
        x_train = x_train.reshape(self.ntrain, -1, 1)
        y_train = output[:self.ntrain, ::r1, ::r2][:, :s1, :s2]
        y_train = y_train.reshape(self.ntrain, -1, self.T_out * self.out_dim)
        x_test = input[-self.ntest:, ::r1][:, :s1].reshape(self.ntest, s1, 1).repeat(1, 1, s2)
        x_test = x_test.reshape(self.ntest, -1, 1)
        y_test = output[-self.ntest:, ::r1, ::r2][:, :s1, :s2]
        y_test = y_test.reshape(self.ntest, -1, self.T_out * self.out_dim)
        print(x_train.shape, y_train.shape)

        x_normalizer = UnitTransformer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
        x_normalizer.cuda()

        if self.normalize:
            self.y_normalizer = UnitTransformer(y_train)
            y_train = self.y_normalizer.encode(y_train)
            self.y_normalizer.cuda()

        x = np.linspace(0, 1, s2)
        y = np.linspace(0, 1, s1)
        x, y = np.meshgrid(x, y)
        pos = np.c_[x.ravel(), y.ravel()]
        pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)

        pos_train = pos.repeat(self.ntrain, 1, 1)
        pos_test = pos.repeat(self.ntest, 1, 1)

        t = np.linspace(0, 1, self.T_out)
        t = torch.tensor(t, dtype=torch.float).unsqueeze(0)
        t_train = t.repeat(self.ntrain, 1)
        t_test = t.repeat(self.ntest, 1)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, t_train, x_train, y_train),
                                                   batch_size=self.batch_size, shuffle=True,
                                                   collate_fn=self.random_collate_fn)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, t_test, x_test, y_test),
                                                  batch_size=self.batch_size, shuffle=False)
        print("Dataloading is over.")
        return train_loader, test_loader, [s1, s2]


class elas(object):
    def __init__(self, args):
        self.PATH_Sigma = args.data_path + '/elasticity/Meshes/Random_UnitCell_sigma_10.npy'
        self.PATH_XY = args.data_path + '/elasticity/Meshes/Random_UnitCell_XY_10.npy'
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.normalize = args.normalize

    def get_loader(self):
        input_s = np.load(self.PATH_Sigma)
        input_s = torch.tensor(input_s, dtype=torch.float).permute(1, 0)
        input_xy = np.load(self.PATH_XY)
        input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2, 0, 1)

        train_s = input_s[:self.ntrain, :, None]
        test_s = input_s[-self.ntest:, :, None]
        train_xy = input_xy[:self.ntrain]
        test_xy = input_xy[-self.ntest:]

        print(input_s.shape, input_xy.shape)

        if self.normalize:
            self.y_normalizer = UnitTransformer(train_s)
            train_s = self.y_normalizer.encode(train_s)
            self.y_normalizer.cuda()

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_xy, train_xy, train_s),
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_xy, test_xy, test_s),
                                                  batch_size=self.batch_size,
                                                  shuffle=False)
        print("Dataloading is over.")
        return train_loader, test_loader, [train_s.shape[1]]


class pipe(object):
    def __init__(self, args):
        self.INPUT_X = args.data_path + '/Pipe_X.npy'
        self.INPUT_Y = args.data_path + '/Pipe_Y.npy'
        self.OUTPUT_Sigma = args.data_path + '/Pipe_Q.npy'
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.normalize = args.normalize

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley
        s1 = int(((129 - 1) / r1) + 1)
        s2 = int(((129 - 1) / r2) + 1)

        inputX = np.load(self.INPUT_X)
        inputX = torch.tensor(inputX, dtype=torch.float)
        inputY = np.load(self.INPUT_Y)
        inputY = torch.tensor(inputY, dtype=torch.float)
        input = torch.stack([inputX, inputY], dim=-1)

        output = np.load(self.OUTPUT_Sigma)[:, 0]
        output = torch.tensor(output, dtype=torch.float)
        print(input.shape, output.shape)

        x_train = input[:self.ntrain, ::r1, ::r2][:, :s1, :s2]
        y_train = output[:self.ntrain, ::r1, ::r2][:, :s1, :s2]
        x_test = input[self.ntrain:self.ntrain + self.ntest, ::r1, ::r2][:, :s1, :s2]
        y_test = output[self.ntrain:self.ntrain + self.ntest, ::r1, ::r2][:, :s1, :s2]
        x_train = x_train.reshape(self.ntrain, -1, 2)
        x_test = x_test.reshape(self.ntest, -1, 2)
        y_train = y_train.reshape(self.ntrain, -1, 1)
        y_test = y_test.reshape(self.ntest, -1, 1)

        if self.normalize:
            self.x_normalizer = UnitTransformer(x_train)
            self.y_normalizer = UnitTransformer(y_train)

            x_train = self.x_normalizer.encode(x_train)
            x_test = self.x_normalizer.encode(x_test)
            y_train = self.y_normalizer.encode(y_train)

            self.x_normalizer.cuda()
            self.y_normalizer.cuda()

        x = np.linspace(0, 1, s2)
        y = np.linspace(0, 1, s1)
        x, y = np.meshgrid(x, y)
        pos = np.c_[x.ravel(), y.ravel()]
        pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)

        pos_train = pos.repeat(self.ntrain, 1, 1)
        pos_test = pos.repeat(self.ntest, 1, 1)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, pos_train, y_train),
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, pos_test, y_test),
                                                  batch_size=self.batch_size,
                                                  shuffle=False)
        print("Dataloading is over.")
        return train_loader, test_loader, [s1, s2]


class airfoil(object):
    def __init__(self, args):
        self.INPUT_X = args.data_path + '/NACA_Cylinder_X.npy'
        self.INPUT_Y = args.data_path + '/NACA_Cylinder_Y.npy'
        self.OUTPUT_Sigma = args.data_path + '/NACA_Cylinder_Q.npy'
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.normalize = args.normalize

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley
        s1 = int(((221 - 1) / r1) + 1)
        s2 = int(((51 - 1) / r2) + 1)

        inputX = np.load(self.INPUT_X)
        inputX = torch.tensor(inputX, dtype=torch.float)
        inputY = np.load(self.INPUT_Y)
        inputY = torch.tensor(inputY, dtype=torch.float)
        input = torch.stack([inputX, inputY], dim=-1)

        output = np.load(self.OUTPUT_Sigma)[:, 4]
        output = torch.tensor(output, dtype=torch.float)
        print(input.shape, output.shape)

        x_train = input[:self.ntrain, ::r1, ::r2][:, :s1, :s2]
        y_train = output[:self.ntrain, ::r1, ::r2][:, :s1, :s2]
        x_test = input[self.ntrain:self.ntrain + self.ntest, ::r1, ::r2][:, :s1, :s2]
        y_test = output[self.ntrain:self.ntrain + self.ntest, ::r1, ::r2][:, :s1, :s2]
        x_train = x_train.reshape(self.ntrain, -1, 2)
        x_test = x_test.reshape(self.ntest, -1, 2)
        y_train = y_train.reshape(self.ntrain, -1, 1)
        y_test = y_test.reshape(self.ntest, -1, 1)

        if self.normalize:
            self.x_normalizer = UnitTransformer(x_train)
            self.y_normalizer = UnitTransformer(y_train)

            x_train = self.x_normalizer.encode(x_train)
            x_test = self.x_normalizer.encode(x_test)
            y_train = self.y_normalizer.encode(y_train)

            self.x_normalizer.cuda()
            self.y_normalizer.cuda()

        x = np.linspace(0, 1, s2)
        y = np.linspace(0, 1, s1)
        x, y = np.meshgrid(x, y)
        pos = np.c_[x.ravel(), y.ravel()]
        pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)

        pos_train = pos.repeat(self.ntrain, 1, 1)
        pos_test = pos.repeat(self.ntest, 1, 1)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, pos_train, y_train),
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, pos_test, y_test),
                                                  batch_size=self.batch_size,
                                                  shuffle=False)
        print("Dataloading is over.")
        return train_loader, test_loader, [s1, s2]


class darcy(object):
    def __init__(self, args):
        self.train_path = args.data_path + '/piececonst_r421_N1024_smooth1.mat'
        self.test_path = args.data_path + '/piececonst_r421_N1024_smooth2.mat'
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.normalize = args.normalize

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley
        s1 = int(((421 - 1) / r1) + 1)
        s2 = int(((421 - 1) / r2) + 1)

        train_data = scio.loadmat(self.train_path)
        x_train = train_data['coeff'][:self.ntrain, ::r1, ::r2][:, :s1, :s2]
        x_train = x_train.reshape(self.ntrain, -1, 1)
        x_train = torch.from_numpy(x_train).float()
        y_train = train_data['sol'][:self.ntrain, ::r1, ::r2][:, :s1, :s2]
        y_train = y_train.reshape(self.ntrain, -1, 1)
        y_train = torch.from_numpy(y_train)

        test_data = scio.loadmat(self.test_path)
        x_test = test_data['coeff'][:self.ntest, ::r1, ::r2][:, :s1, :s2]
        x_test = x_test.reshape(self.ntest, -1, 1)
        x_test = torch.from_numpy(x_test).float()
        y_test = test_data['sol'][:self.ntest, ::r1, ::r2][:, :s1, :s2]
        y_test = y_test.reshape(self.ntest, -1, 1)
        y_test = torch.from_numpy(y_test)

        print(train_data['coeff'].shape, train_data['sol'].shape)
        print(test_data['coeff'].shape, test_data['sol'].shape)

        if self.normalize:
            self.x_normalizer = UnitTransformer(x_train)
            self.y_normalizer = UnitTransformer(y_train)

            x_train = self.x_normalizer.encode(x_train)
            x_test = self.x_normalizer.encode(x_test)
            y_train = self.y_normalizer.encode(y_train)

            self.x_normalizer.cuda()
            self.y_normalizer.cuda()

        x = np.linspace(0, 1, s2)
        y = np.linspace(0, 1, s1)
        x, y = np.meshgrid(x, y)
        pos = np.c_[x.ravel(), y.ravel()]
        pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)

        pos_train = pos.repeat(self.ntrain, 1, 1)
        pos_test = pos.repeat(self.ntest, 1, 1)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, x_train, y_train),
                                                   batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, x_test, y_test),
                                                  batch_size=self.batch_size, shuffle=False)
        print("Dataloading is over.")
        return train_loader, test_loader, [s1, s2]


class ns(object):
    def __init__(self, args):
        self.data_path = args.data_path + '/NavierStokes_V1e-5_N1200_T20.mat'
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.out_dim = args.out_dim
        self.T_in = args.T_in
        self.T_out = args.T_out
        self.normalize = args.normalize

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley
        s1 = int(((64 - 1) / r1) + 1)
        s2 = int(((64 - 1) / r2) + 1)

        data = scio.loadmat(self.data_path)
        print(data['u'].shape)
        train_a = data['u'][:self.ntrain, ::r1, ::r2, None, :self.T_in][:, :s1, :s2, :, :]
        train_a = train_a.reshape(train_a.shape[0], -1, self.out_dim * train_a.shape[-1])
        train_a = torch.from_numpy(train_a)
        train_u = data['u'][:self.ntrain, ::r1, ::r2, None, self.T_in:self.T_out + self.T_in][:, :s1, :s2, :, :]
        train_u = train_u.reshape(train_u.shape[0], -1, self.out_dim * train_u.shape[-1])
        train_u = torch.from_numpy(train_u)

        test_a = data['u'][-self.ntest:, ::r1, ::r2, None, :self.T_in][:, :s1, :s2, :, :]
        test_a = test_a.reshape(test_a.shape[0], -1, self.out_dim * test_a.shape[-1])
        test_a = torch.from_numpy(test_a)
        test_u = data['u'][-self.ntest:, ::r1, ::r2, None, self.T_in:self.T_out + self.T_in][:, :s1, :s2, :, :]
        test_u = test_u.reshape(test_u.shape[0], -1, self.out_dim * test_u.shape[-1])
        test_u = torch.from_numpy(test_u)

        if self.normalize:
            self.x_normalizer = UnitTransformer(train_a)
            self.y_normalizer = UnitTransformer(train_u)

            train_a = self.x_normalizer.encode(train_a)
            test_a = self.x_normalizer.encode(test_a)
            train_u = self.y_normalizer.encode(train_u)

            self.x_normalizer.cuda()
            self.y_normalizer.cuda()

        x = np.linspace(0, 1, s2)
        y = np.linspace(0, 1, s1)
        x, y = np.meshgrid(x, y)
        pos = np.c_[x.ravel(), y.ravel()]
        pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
        pos_train = pos.repeat(self.ntrain, 1, 1)
        pos_test = pos.repeat(self.ntest, 1, 1)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, train_a, train_u),
                                                   batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, test_a, test_u),
                                                  batch_size=self.batch_size, shuffle=False)

        print("Dataloading is over.")
        return train_loader, test_loader, [s1, s2]


class pdebench_autoregressive(object):
    def __init__(self, args):
        self.file_path = args.data_path
        self.train_ratio = args.train_ratio
        self.T_in = args.T_in
        self.T_out = args.T_out
        self.batch_size = args.batch_size
        self.out_dim = args.out_dim

    def get_loader(self):
        train_dataset = pdebench_dataset_autoregressive(file_path=self.file_path, train_ratio=self.train_ratio,
                                                        test=False,
                                                        T_in=self.T_in, T_out=self.T_out, out_dim=self.out_dim)
        test_dataset = pdebench_dataset_autoregressive(file_path=self.file_path, train_ratio=self.train_ratio,
                                                       test=True,
                                                       T_in=self.T_in, T_out=self.T_out, out_dim=self.out_dim)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        return train_loader, test_loader, train_dataset.shapelist


class pdebench_dataset_autoregressive(Dataset):
    def __init__(self,
                 file_path: str,
                 train_ratio: int,
                 test: bool,
                 T_in: int,
                 T_out: int,
                 out_dim: int):
        self.file_path = file_path
        with h5py.File(self.file_path, "r") as h5_file:
            data_list = sorted(h5_file.keys())
            self.shapelist = h5_file[data_list[0]]["data"].shape[1:-1]  # obtain shapelist
        self.ntrain = int(len(data_list) * train_ratio)
        self.test = test
        if not self.test:
            self.data_list = data_list[:self.ntrain]
        else:
            self.data_list = data_list[self.ntrain:]
        self.T_in = T_in
        self.T_out = T_out
        self.out_dim = out_dim

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, "r") as h5_file:
            data_group = h5_file[self.data_list[idx]]

            # data dim = [t, x1, ..., xd, v]
            data = np.array(data_group["data"], dtype="f")
            dim = len(data.shape) - 2
            T, *_, V = data.shape
            # change data shape
            data = torch.tensor(data, dtype=torch.float).movedim(0, -2).contiguous().reshape(*self.shapelist, -1)
            # x, y and z are 1-D arrays
            # Convert the spatial coordinates to meshgrid
            if dim == 1:
                grid = np.array(data_group["grid"]["x"], dtype="f")
                grid = torch.tensor(grid, dtype=torch.float).unsqueeze(-1)
            elif dim == 2:
                x = np.array(data_group["grid"]["x"], dtype="f")
                y = np.array(data_group["grid"]["y"], dtype="f")
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                X, Y = torch.meshgrid(x, y, indexing="ij")
                grid = torch.stack((X, Y), axis=-1)
            elif dim == 3:
                x = np.array(data_group["grid"]["x"], dtype="f")
                y = np.array(data_group["grid"]["y"], dtype="f")
                z = np.array(data_group["grid"]["z"], dtype="f")
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                z = torch.tensor(z, dtype=torch.float)
                X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
                grid = torch.stack((X, Y, Z), axis=-1)

        return grid, data[:, :self.T_in * self.out_dim], \
            data[:, (self.T_in) * self.out_dim:(self.T_in + self.T_out) * self.out_dim]


class pdebench_steady_darcy(object):
    def __init__(self, args):
        self.file_path = args.data_path
        self.ntrain = args.ntrain
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley
        s1 = int(((128 - 1) / r1) + 1)
        s2 = int(((128 - 1) / r2) + 1)
        with h5py.File(self.file_path, "r") as h5_file:
            data_nu = np.array(h5_file['nu'], dtype='f')[:, ::r1, ::r2][:, :s1, :s2]
            data_solution = np.array(h5_file['tensor'], dtype='f')[:, ::r1, ::r2][:, :s1, :s2]
            data_nu = torch.from_numpy(data_nu)
            data_solution = torch.from_numpy(data_solution)
            x = np.array(h5_file['x-coordinate'])
            y = np.array(h5_file['y-coordinate'])
            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            X, Y = torch.meshgrid(x, y, indexing="ij")
            grid = torch.stack((X, Y), axis=-1)[None, ::r1, ::r2, :][:, :s1, :s2, :]

        grid = grid.repeat(data_nu.shape[0], 1, 1, 1)

        pos_train = grid[:self.ntrain, :, :, :].reshape(self.ntrain, -1, 2)
        x_train = data_nu[:self.ntrain, :, :].reshape(self.ntrain, -1, 1)
        y_train = data_solution[:self.ntrain, 0, :, :].reshape(self.ntrain, -1, 1)  # solutions only have 1 channel

        pos_test = grid[self.ntrain:, :, :, :].reshape(data_nu.shape[0] - self.ntrain, -1, 2)
        x_test = data_nu[self.ntrain:, :, :].reshape(data_nu.shape[0] - self.ntrain, -1, 1)
        y_test = data_solution[self.ntrain:, 0, :, :].reshape(data_nu.shape[0] - self.ntrain, -1,
                                                              1)  # solutions only have 1 channel

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, x_train, y_train),
                                                   batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, x_test, y_test),
                                                  batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader, [s1, s2]


class car_design(object):
    def __init__(self, args):
        self.file_path = args.data_path
        self.radius = args.radius
        self.test_fold_id = 0

    def get_samples(self, obj_path):
        folds = [f'param{i}' for i in range(9)] 
        samples = []
        for fold in folds:
            fold_samples = []
            files = os.listdir(os.path.join(obj_path, fold))
            for file in files:
                path = os.path.join(obj_path, os.path.join(fold, file))
                if os.path.isdir(path):
                    fold_samples.append(os.path.join(fold, file))
            samples.append(fold_samples)
        return samples  # 100 + 99 + 97 + 100 + 100 + 96 + 100 + 98 + 99 = 889 samples

    def load_train_val_fold(self):
        samples = self.get_samples(os.path.join(self.file_path, 'training_data'))
        trainlst = []
        for i in range(len(samples)):
            if i == self.test_fold_id:
                continue
            trainlst += samples[i]
        vallst = samples[self.test_fold_id] if 0 <= self.test_fold_id < len(samples) else None

        if os.path.exists(os.path.join(self.file_path, 'preprocessed_data')):
            print("use preprocessed data")
            preprocessed = True
        else:
            preprocessed = False
        print("loading data")
        train_dataset, coef_norm = get_datalist(self.file_path, trainlst, norm=True,
                                                savedir=os.path.join(self.file_path, 'preprocessed_data'),
                                                preprocessed=preprocessed)
        val_dataset = get_datalist(self.file_path, vallst, coef_norm=coef_norm,
                                   savedir=os.path.join(self.file_path, 'preprocessed_data'),
                                   preprocessed=preprocessed)
        print("load data finish")
        return train_dataset, val_dataset, coef_norm, vallst

    def get_loader(self):
        train_data, val_data, coef_norm, vallst = self.load_train_val_fold()
        train_loader = GraphDataset(train_data, use_cfd_mesh=False, r=self.radius, coef_norm=coef_norm)
        test_loader = GraphDataset(val_data, use_cfd_mesh=False, r=self.radius, coef_norm=coef_norm, valid_list=vallst)
        return train_loader, test_loader, [train_data[0].x.shape[0]]
