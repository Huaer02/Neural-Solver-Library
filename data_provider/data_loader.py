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
from utils.normalizer import UnitTransformer, UnitGaussianNormalizer


class plas(object):
    def __init__(self, args):
        self.DATA_PATH = args.data_path + "/plas_N987_T20.mat"
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.out_dim = args.out_dim
        self.T_out = args.T_out
        self.normalize = args.normalize
        self.norm_type = args.norm_type

        # Validate norm_type
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(
                f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'."
            )

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
        input = torch.tensor(data["input"], dtype=torch.float)
        output = torch.tensor(data["output"], dtype=torch.float)
        print(input.shape, output.shape)
        x_train = input[: self.ntrain, ::r1][:, :s1].reshape(self.ntrain, s1, 1).repeat(1, 1, s2)
        x_train = x_train.reshape(self.ntrain, -1, 1)
        y_train = output[: self.ntrain, ::r1, ::r2][:, :s1, :s2]
        y_train = y_train.reshape(self.ntrain, -1, self.T_out * self.out_dim)
        x_test = input[-self.ntest :, ::r1][:, :s1].reshape(self.ntest, s1, 1).repeat(1, 1, s2)
        x_test = x_test.reshape(self.ntest, -1, 1)
        y_test = output[-self.ntest :, ::r1, ::r2][:, :s1, :s2]
        y_test = y_test.reshape(self.ntest, -1, self.T_out * self.out_dim)
        print(x_train.shape, y_train.shape)

        # Use appropriate normalizer based on norm_type
        if self.norm_type == "UnitTransformer":
            x_normalizer = UnitTransformer(x_train)
        elif self.norm_type == "UnitGaussianNormalizer":
            x_normalizer = UnitGaussianNormalizer(x_train)

        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
        x_normalizer.cuda()

        if self.normalize:
            # Use appropriate normalizer based on norm_type
            if self.norm_type == "UnitTransformer":
                self.y_normalizer = UnitTransformer(y_train)
            elif self.norm_type == "UnitGaussianNormalizer":
                self.y_normalizer = UnitGaussianNormalizer(y_train)

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

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_train, t_train, x_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.random_collate_fn,
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_test, t_test, x_test, y_test), batch_size=self.batch_size, shuffle=False
        )
        print("Dataloading is over.")
        return train_loader, test_loader, [s1, s2]


class elas(object):
    def __init__(self, args):
        self.PATH_Sigma = args.data_path + "/elasticity/Meshes/Random_UnitCell_sigma_10.npy"
        self.PATH_XY = args.data_path + "/elasticity/Meshes/Random_UnitCell_XY_10.npy"
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.normalize = args.normalize
        self.norm_type = args.norm_type

        # Validate norm_type
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(
                f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'."
            )

    def get_loader(self):
        input_s = np.load(self.PATH_Sigma)
        input_s = torch.tensor(input_s, dtype=torch.float).permute(1, 0)
        input_xy = np.load(self.PATH_XY)
        input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2, 0, 1)

        train_s = input_s[: self.ntrain, :, None]
        test_s = input_s[-self.ntest :, :, None]
        train_xy = input_xy[: self.ntrain]
        test_xy = input_xy[-self.ntest :]

        print(input_s.shape, input_xy.shape)

        if self.normalize:
            # Use appropriate normalizer based on norm_type
            if self.norm_type == "UnitTransformer":
                self.y_normalizer = UnitTransformer(train_s)
            elif self.norm_type == "UnitGaussianNormalizer":
                self.y_normalizer = UnitGaussianNormalizer(train_s)

            train_s = self.y_normalizer.encode(train_s)
            self.y_normalizer.cuda()

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_xy, train_xy, train_s), batch_size=self.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_xy, test_xy, test_s), batch_size=self.batch_size, shuffle=False
        )
        print("Dataloading is over.")
        return train_loader, test_loader, [train_s.shape[1]]


class pipe(object):
    def __init__(self, args):
        self.INPUT_X = args.data_path + "/Pipe_X.npy"
        self.INPUT_Y = args.data_path + "/Pipe_Y.npy"
        self.OUTPUT_Sigma = args.data_path + "/Pipe_Q.npy"
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.normalize = args.normalize
        self.norm_type = args.norm_type

        # Validate norm_type
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(
                f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'."
            )

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

        x_train = input[: self.ntrain, ::r1, ::r2][:, :s1, :s2]
        y_train = output[: self.ntrain, ::r1, ::r2][:, :s1, :s2]
        x_test = input[self.ntrain : self.ntrain + self.ntest, ::r1, ::r2][:, :s1, :s2]
        y_test = output[self.ntrain : self.ntrain + self.ntest, ::r1, ::r2][:, :s1, :s2]
        x_train = x_train.reshape(self.ntrain, -1, 2)
        x_test = x_test.reshape(self.ntest, -1, 2)
        y_train = y_train.reshape(self.ntrain, -1, 1)
        y_test = y_test.reshape(self.ntest, -1, 1)

        if self.normalize:
            # Use appropriate normalizer based on norm_type
            if self.norm_type == "UnitTransformer":
                self.x_normalizer = UnitTransformer(x_train)
                self.y_normalizer = UnitTransformer(y_train)
            elif self.norm_type == "UnitGaussianNormalizer":
                self.x_normalizer = UnitGaussianNormalizer(x_train)
                self.y_normalizer = UnitGaussianNormalizer(y_train)

            x_train = self.x_normalizer.encode(x_train)
            x_test = self.x_normalizer.encode(x_test)
            y_train = self.y_normalizer.encode(y_train)

            self.x_normalizer.cuda()
            self.y_normalizer.cuda()

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, x_train, y_train), batch_size=self.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_test, x_test, y_test), batch_size=self.batch_size, shuffle=False
        )
        print("Dataloading is over.")
        return train_loader, test_loader, [s1, s2]


class airfoil(object):
    def __init__(self, args):
        self.INPUT_X = args.data_path + "/NACA_Cylinder_X.npy"
        self.INPUT_Y = args.data_path + "/NACA_Cylinder_Y.npy"
        self.OUTPUT_Sigma = args.data_path + "/NACA_Cylinder_Q.npy"
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.normalize = args.normalize
        self.norm_type = args.norm_type

        # Validate norm_type
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(
                f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'."
            )

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

        x_train = input[: self.ntrain, ::r1, ::r2][:, :s1, :s2]
        y_train = output[: self.ntrain, ::r1, ::r2][:, :s1, :s2]
        x_test = input[self.ntrain : self.ntrain + self.ntest, ::r1, ::r2][:, :s1, :s2]
        y_test = output[self.ntrain : self.ntrain + self.ntest, ::r1, ::r2][:, :s1, :s2]
        x_train = x_train.reshape(self.ntrain, -1, 2)
        x_test = x_test.reshape(self.ntest, -1, 2)
        y_train = y_train.reshape(self.ntrain, -1, 1)
        y_test = y_test.reshape(self.ntest, -1, 1)

        if self.normalize:
            # Use appropriate normalizer based on norm_type
            if self.norm_type == "UnitTransformer":
                self.x_normalizer = UnitTransformer(x_train)
                self.y_normalizer = UnitTransformer(y_train)
            elif self.norm_type == "UnitGaussianNormalizer":
                self.x_normalizer = UnitGaussianNormalizer(x_train)
                self.y_normalizer = UnitGaussianNormalizer(y_train)

            x_train = self.x_normalizer.encode(x_train)
            x_test = self.x_normalizer.encode(x_test)
            y_train = self.y_normalizer.encode(y_train)

            self.x_normalizer.cuda()
            self.y_normalizer.cuda()

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, x_train, y_train), batch_size=self.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_test, x_test, y_test), batch_size=self.batch_size, shuffle=False
        )
        print("Dataloading is over.")
        return train_loader, test_loader, [s1, s2]


class darcy(object):
    def __init__(self, args):
        self.train_path = args.data_path + "/piececonst_r421_N1024_smooth1.mat"
        self.test_path = args.data_path + "/piececonst_r421_N1024_smooth2.mat"
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.normalize = args.normalize
        self.norm_type = args.norm_type

        # Validate norm_type
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(
                f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'."
            )

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley
        s1 = int(((421 - 1) / r1) + 1)
        s2 = int(((421 - 1) / r2) + 1)

        train_data = scio.loadmat(self.train_path)
        x_train = train_data["coeff"][: self.ntrain, ::r1, ::r2][:, :s1, :s2]
        x_train = x_train.reshape(self.ntrain, -1, 1)
        x_train = torch.from_numpy(x_train).float()
        y_train = train_data["sol"][: self.ntrain, ::r1, ::r2][:, :s1, :s2]
        y_train = y_train.reshape(self.ntrain, -1, 1)
        y_train = torch.from_numpy(y_train)

        test_data = scio.loadmat(self.test_path)
        x_test = test_data["coeff"][: self.ntest, ::r1, ::r2][:, :s1, :s2]
        x_test = x_test.reshape(self.ntest, -1, 1)
        x_test = torch.from_numpy(x_test).float()
        y_test = test_data["sol"][: self.ntest, ::r1, ::r2][:, :s1, :s2]
        y_test = y_test.reshape(self.ntest, -1, 1)
        y_test = torch.from_numpy(y_test)

        print(train_data["coeff"].shape, train_data["sol"].shape)
        print(test_data["coeff"].shape, test_data["sol"].shape)

        if self.normalize:
            # Use appropriate normalizer based on norm_type
            if self.norm_type == "UnitTransformer":
                self.x_normalizer = UnitTransformer(x_train)
                self.y_normalizer = UnitTransformer(y_train)
            elif self.norm_type == "UnitGaussianNormalizer":
                self.x_normalizer = UnitGaussianNormalizer(x_train)
                self.y_normalizer = UnitGaussianNormalizer(y_train)

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

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_train, x_train, y_train), batch_size=self.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_test, x_test, y_test), batch_size=self.batch_size, shuffle=False
        )
        print("Dataloading is over.")
        return train_loader, test_loader, [s1, s2]


class ns(object):
    def __init__(self, args):
        self.data_path = args.data_path + "/NavierStokes_V1e-5_N1200_T20.mat"
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.out_dim = args.out_dim
        self.T_in = args.T_in
        self.T_out = args.T_out
        self.normalize = args.normalize
        self.norm_type = args.norm_type

        # Validate norm_type
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(
                f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'."
            )

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley
        s1 = int(((64 - 1) / r1) + 1)
        s2 = int(((64 - 1) / r2) + 1)

        data = scio.loadmat(self.data_path)
        print(data["u"].shape)
        train_a = data["u"][: self.ntrain, ::r1, ::r2, None, : self.T_in][:, :s1, :s2, :, :]
        train_a = train_a.reshape(train_a.shape[0], -1, self.out_dim * train_a.shape[-1])
        train_a = torch.from_numpy(train_a)
        train_u = data["u"][: self.ntrain, ::r1, ::r2, None, self.T_in : self.T_out + self.T_in][:, :s1, :s2, :, :]
        train_u = train_u.reshape(train_u.shape[0], -1, self.out_dim * train_u.shape[-1])
        train_u = torch.from_numpy(train_u)

        test_a = data["u"][-self.ntest :, ::r1, ::r2, None, : self.T_in][:, :s1, :s2, :, :]
        test_a = test_a.reshape(test_a.shape[0], -1, self.out_dim * test_a.shape[-1])
        test_a = torch.from_numpy(test_a)
        test_u = data["u"][-self.ntest :, ::r1, ::r2, None, self.T_in : self.T_out + self.T_in][:, :s1, :s2, :, :]
        test_u = test_u.reshape(test_u.shape[0], -1, self.out_dim * test_u.shape[-1])
        test_u = torch.from_numpy(test_u)

        if self.normalize:
            # Use appropriate normalizer based on norm_type
            if self.norm_type == "UnitTransformer":
                self.x_normalizer = UnitTransformer(train_a)
                self.y_normalizer = UnitTransformer(train_u)
            elif self.norm_type == "UnitGaussianNormalizer":
                self.x_normalizer = UnitGaussianNormalizer(train_a)
                self.y_normalizer = UnitGaussianNormalizer(train_u)

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

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_train, train_a, train_u), batch_size=self.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_test, test_a, test_u), batch_size=self.batch_size, shuffle=False
        )

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
        train_dataset = pdebench_dataset_autoregressive(
            file_path=self.file_path,
            train_ratio=self.train_ratio,
            test=False,
            T_in=self.T_in,
            T_out=self.T_out,
            out_dim=self.out_dim,
        )
        test_dataset = pdebench_dataset_autoregressive(
            file_path=self.file_path,
            train_ratio=self.train_ratio,
            test=True,
            T_in=self.T_in,
            T_out=self.T_out,
            out_dim=self.out_dim,
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        return train_loader, test_loader, train_dataset.shapelist


class pdebench_dataset_autoregressive(Dataset):
    def __init__(self, file_path: str, train_ratio: int, test: bool, T_in: int, T_out: int, out_dim: int):
        self.file_path = file_path
        with h5py.File(self.file_path, "r") as h5_file:
            data_list = sorted(h5_file.keys())
            self.shapelist = h5_file[data_list[0]]["data"].shape[1:-1]  # obtain shapelist
        self.ntrain = int(len(data_list) * train_ratio)
        self.test = test
        if not self.test:
            self.data_list = data_list[: self.ntrain]
        else:
            self.data_list = data_list[self.ntrain :]
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

        return (
            grid,
            data[:, : self.T_in * self.out_dim],
            data[:, (self.T_in) * self.out_dim : (self.T_in + self.T_out) * self.out_dim],
        )


class pdebench_npy_dataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        T_in: int,
        T_out: int,
        out_dim: int = None,
        input_normalizer=None,
        output_normalizer=None,
    ):
        """
        Args:
            data_path: 包含train.npy, val.npy, test.npy的目录路径
            split: 数据划分 'train', 'val', 'test'
            T_in: 输入时间步数
            T_out: 输出时间步数
            out_dim: 输出维度（如果为None则自动推断）
        """
        self.data_path = data_path
        self.split = split
        self.T_in = T_in
        self.T_out = T_out
        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer

        # 构建数据文件路径
        self.data_file = os.path.join(data_path, f"{split}.npy")

        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        # 加载数据
        self.data = np.load(self.data_file)
        print(f"Loaded {split} data with shape: {self.data.shape}")

        # 推断数据维度和形状
        self.n_samples = self.data.shape[0]
        self.total_timesteps = self.data.shape[1]

        # 判断是1D还是2D数据并推断out_dim
        if len(self.data.shape) == 4:  # 1D: (N, T, X, C)
            self.dim = 1
            self.spatial_shape = (self.data.shape[2],)
            self.out_dim = out_dim if out_dim is not None else self.data.shape[3]
        elif len(self.data.shape) == 5:  # 2D: (N, T, H, W, C)
            self.dim = 2
            self.spatial_shape = (self.data.shape[2], self.data.shape[3])
            self.out_dim = out_dim if out_dim is not None else self.data.shape[4]
        elif len(self.data.shape) == 3:  # 没有channel维度: (N, T, X)
            self.dim = 1
            self.spatial_shape = (self.data.shape[2],)
            self.out_dim = out_dim if out_dim is not None else 1
            # 添加channel维度
            self.data = self.data[..., None]
        else:
            raise ValueError(f"Unsupported data shape: {self.data.shape}")

        # 验证时间步数
        if T_in + T_out > self.total_timesteps:
            raise ValueError(
                f"T_in({T_in}) + T_out({T_out}) = {T_in + T_out} > total timesteps({self.total_timesteps})"
            )

        # 创建坐标网格
        self._create_coordinates()

    def _create_coordinates(self):
        """创建空间坐标网格"""
        if self.dim == 1:
            # 尝试加载坐标文件
            x_coord_file = os.path.join(self.data_path, "x_coordinate.npy")

            if os.path.exists(x_coord_file):
                x_coords = np.load(x_coord_file)
                # 归一化到[0,1]
                x_coords = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min())
            else:
                # 创建默认坐标
                x_coords = np.linspace(0, 1, self.spatial_shape[0])

            self.grid = torch.tensor(x_coords, dtype=torch.float).unsqueeze(-1)  # (X, 1)

        elif self.dim == 2:
            # 创建2D网格坐标
            x_coords = np.linspace(0, 1, self.spatial_shape[1])  # W
            y_coords = np.linspace(0, 1, self.spatial_shape[0])  # H
            X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")

            # 展平并堆叠
            grid = np.stack([X.ravel(), Y.ravel()], axis=-1)  # (H*W, 2)
            self.grid = torch.tensor(grid, dtype=torch.float)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 获取单个样本数据
        sample_data = self.data[idx]  # (T, ..., C)

        # 提取输入和输出时间步
        input_data = sample_data[: self.T_in]  # (T_in, ..., C)
        output_data = sample_data[self.T_in : self.T_in + self.T_out]  # (T_out, ..., C)

        # 转换为torch tensor
        input_data = torch.tensor(input_data, dtype=torch.float)
        output_data = torch.tensor(output_data, dtype=torch.float)

        # 重新整理数据形状以匹配现有格式
        if self.dim == 1:
            # 1D: (T, X, C) -> (X, T*C)
            input_data = input_data.permute(1, 0, 2).reshape(self.spatial_shape[0], -1)
            output_data = output_data.permute(1, 0, 2).reshape(self.spatial_shape[0], -1)
        elif self.dim == 2:
            # 2D: (T, H, W, C) -> (H*W, T*C)
            input_data = input_data.permute(1, 2, 0, 3).reshape(-1, self.T_in * self.out_dim)
            output_data = output_data.permute(1, 2, 0, 3).reshape(-1, self.T_out * self.out_dim)

        # 应用归一化
        if self.input_normalizer is not None:
            input_data = self.input_normalizer.encode(input_data.unsqueeze(0)).squeeze(0)

        if self.output_normalizer is not None:
            output_data = self.output_normalizer.encode(output_data.unsqueeze(0)).squeeze(0)

        return self.grid, input_data, output_data


class pdebench_npy(object):
    def __init__(self, args):
        self.data_path = args.data_path  # 直接指向包含train.npy等文件的目录
        self.T_in = args.T_in
        self.T_out = args.T_out
        self.batch_size = args.batch_size
        self.out_dim = getattr(args, "out_dim", None)  # 允许自动推断
        self.normalize = getattr(args, "normalize", False)
        self.norm_type = getattr(args, "norm_type", "UnitGaussianNormalizer")

        # Validate norm_type
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(
                f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'."
            )

    def get_loader(self):
        # 首先创建一个临时数据集来获取数据信息
        temp_dataset = pdebench_npy_dataset(
            data_path=self.data_path, split="train", T_in=self.T_in, T_out=self.T_out, out_dim=self.out_dim
        )

        # 更新out_dim（如果之前是None）
        self.out_dim = temp_dataset.out_dim

        input_normalizer = None
        output_normalizer = None

        # 如果需要归一化，计算归一化参数
        if self.normalize:
            print("Computing normalization parameters...")
            all_input_data = []
            all_output_data = []

            # 收集所有训练数据
            for i in range(len(temp_dataset)):
                _, input_data, output_data = temp_dataset[i]
                all_input_data.append(input_data)
                all_output_data.append(output_data)

            all_input_data = torch.stack(all_input_data, dim=0)
            all_output_data = torch.stack(all_output_data, dim=0)

            # 创建归一化器
            if self.norm_type == "UnitTransformer":
                input_normalizer = UnitTransformer(all_input_data)
                output_normalizer = UnitTransformer(all_output_data)
            elif self.norm_type == "UnitGaussianNormalizer":
                input_normalizer = UnitGaussianNormalizer(all_input_data)
                output_normalizer = UnitGaussianNormalizer(all_output_data)

            input_normalizer.cuda()
            output_normalizer.cuda()

            # 保存归一化器
            self.input_normalizer = input_normalizer
            self.output_normalizer = output_normalizer

            print("Normalization parameters computed.")

        # 创建带归一化的数据集
        train_dataset = pdebench_npy_dataset(
            data_path=self.data_path,
            split="train",
            T_in=self.T_in,
            T_out=self.T_out,
            out_dim=self.out_dim,
            input_normalizer=input_normalizer,
            output_normalizer=output_normalizer,
        )

        val_dataset = pdebench_npy_dataset(
            data_path=self.data_path,
            split="val",
            T_in=self.T_in,
            T_out=self.T_out,
            out_dim=self.out_dim,
            input_normalizer=input_normalizer,
            output_normalizer=output_normalizer,
        )

        test_dataset = pdebench_npy_dataset(
            data_path=self.data_path,
            split="test",
            T_in=self.T_in,
            T_out=self.T_out,
            out_dim=self.out_dim,
            input_normalizer=input_normalizer,
            output_normalizer=output_normalizer,
        )

        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # 返回空间形状信息
        if train_dataset.dim == 1:
            shape_info = [train_dataset.spatial_shape[0]]
        else:  # 2D
            shape_info = list(train_dataset.spatial_shape)

        print("Dataloading is over.")
        return train_loader, test_loader, shape_info


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
            data_nu = np.array(h5_file["nu"], dtype="f")[:, ::r1, ::r2][:, :s1, :s2]
            data_solution = np.array(h5_file["tensor"], dtype="f")[:, :, ::r1, ::r2][:, :, :s1, :s2]
            data_nu = torch.from_numpy(data_nu)
            data_solution = torch.from_numpy(data_solution)
            x = np.array(h5_file["x-coordinate"])
            y = np.array(h5_file["y-coordinate"])
            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            X, Y = torch.meshgrid(x, y, indexing="ij")
            grid = torch.stack((X, Y), axis=-1)[None, ::r1, ::r2, :][:, :s1, :s2, :]

        grid = grid.repeat(data_nu.shape[0], 1, 1, 1)

        pos_train = grid[: self.ntrain, :, :, :].reshape(self.ntrain, -1, 2)
        x_train = data_nu[: self.ntrain, :, :].reshape(self.ntrain, -1, 1)
        y_train = data_solution[: self.ntrain, 0, :, :].reshape(self.ntrain, -1, 1)  # solutions only have 1 channel

        pos_test = grid[self.ntrain :, :, :, :].reshape(data_nu.shape[0] - self.ntrain, -1, 2)
        x_test = data_nu[self.ntrain :, :, :].reshape(data_nu.shape[0] - self.ntrain, -1, 1)
        y_test = data_solution[self.ntrain :, 0, :, :].reshape(
            data_nu.shape[0] - self.ntrain, -1, 1
        )  # solutions only have 1 channel

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_train, x_train, y_train), batch_size=self.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_test, x_test, y_test), batch_size=self.batch_size, shuffle=False
        )
        return train_loader, test_loader, [s1, s2]


class car_design(object):
    def __init__(self, args):
        self.file_path = args.data_path
        self.radius = args.radius
        self.test_fold_id = 0

    def get_samples(self, obj_path):
        folds = [f"param{i}" for i in range(9)]
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
        samples = self.get_samples(os.path.join(self.file_path, "training_data"))
        trainlst = []
        for i in range(len(samples)):
            if i == self.test_fold_id:
                continue
            trainlst += samples[i]
        vallst = samples[self.test_fold_id] if 0 <= self.test_fold_id < len(samples) else None

        if os.path.exists(os.path.join(self.file_path, "preprocessed_data")):
            print("use preprocessed data")
            preprocessed = True
        else:
            preprocessed = False
        print("loading data")
        train_dataset, coef_norm = get_datalist(
            self.file_path,
            trainlst,
            norm=True,
            savedir=os.path.join(self.file_path, "preprocessed_data"),
            preprocessed=preprocessed,
        )
        val_dataset = get_datalist(
            self.file_path,
            vallst,
            coef_norm=coef_norm,
            savedir=os.path.join(self.file_path, "preprocessed_data"),
            preprocessed=preprocessed,
        )
        print("load data finish")
        return train_dataset, val_dataset, coef_norm, vallst

    def get_loader(self):
        train_data, val_data, coef_norm, vallst = self.load_train_val_fold()
        train_loader = GraphDataset(train_data, use_cfd_mesh=False, r=self.radius, coef_norm=coef_norm)
        test_loader = GraphDataset(val_data, use_cfd_mesh=False, r=self.radius, coef_norm=coef_norm, valid_list=vallst)
        return train_loader, test_loader, [train_data[0].x.shape[0]]


class cfd_3d_dataset(Dataset):
    def __init__(
        self, data_path, downsamplex, downsampley, downsamplez, T_in, T_out, out_dim, is_train=True, train_ratio=0.8
    ):
        self.data_path = data_path
        self.T_in = T_in
        self.T_out = T_out
        self.out_dim = out_dim
        self.is_train = is_train

        # Calculate grid sizes
        self.r1 = downsamplex
        self.r2 = downsampley
        self.r3 = downsamplez
        self.s1 = int(((128 - 1) / self.r1) + 1)
        self.s2 = int(((128 - 1) / self.r2) + 1)
        self.s3 = int(((128 - 1) / self.r3) + 1)

        # Create position grid once (reused for all samples)
        with h5py.File(data_path, "r") as h5_file:
            x_coords = np.array(h5_file["x-coordinate"][:: self.r1])[: self.s1]
            y_coords = np.array(h5_file["y-coordinate"][:: self.r2])[: self.s2]
            z_coords = np.array(h5_file["z-coordinate"][:: self.r3])[: self.s3]

            # Create grid
            x = torch.tensor(x_coords, dtype=torch.float)
            y = torch.tensor(y_coords, dtype=torch.float)
            z = torch.tensor(z_coords, dtype=torch.float)
            X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
            self.grid = torch.stack((X, Y, Z), axis=-1)
            self.grid_flat = self.grid.reshape(-1, 3)

            first_field = sorted(h5_file.keys())[0]
            num_samples = h5_file[first_field].shape[0]
            self.ntrain = int(num_samples * train_ratio)

            # Set indices based on train or test
            if self.is_train:
                self.indices = np.arange(self.ntrain)
            else:
                self.indices = np.arange(self.ntrain, num_samples)

        self.fields = ["Vx", "Vy", "Vz", "pressure", "density"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx = self.indices[idx]

        # Initialize data arrays for this sample only (much smaller memory footprint)
        a_data = np.zeros((self.grid_flat.shape[0], self.T_in * self.out_dim))
        u_data = np.zeros((self.grid_flat.shape[0], self.T_out * self.out_dim))
        # import pdb; pdb.set_trace()

        with h5py.File(self.data_path, "r") as h5_file:
            # Load input timesteps
            for t_in in range(self.T_in):
                for f_idx, field in enumerate(self.fields):
                    var_data = h5_file[field][sample_idx, t_in, :: self.r1, :: self.r2, :: self.r3][
                        : self.s1, : self.s2, : self.s3
                    ]
                    var_data_flat = var_data.reshape(-1)
                    a_data[:, t_in * self.out_dim + f_idx] = var_data_flat

            # Load output timesteps
            for t_out in range(self.T_out):
                for f_idx, field in enumerate(self.fields):
                    var_data = h5_file[field][sample_idx, self.T_in + t_out, :: self.r1, :: self.r2, :: self.r3][
                        : self.s1, : self.s2, : self.s3
                    ]
                    var_data_flat = var_data.reshape(-1)
                    u_data[:, t_out * self.out_dim + f_idx] = var_data_flat

        # Convert to tensors
        a_data = torch.tensor(a_data, dtype=torch.float)
        u_data = torch.tensor(u_data, dtype=torch.float)

        return self.grid_flat, a_data, u_data


class cfd3d(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.downsamplez = args.downsamplez
        self.batch_size = args.batch_size
        self.train_ratio = args.train_ratio
        self.out_dim = args.out_dim
        self.T_in = args.T_in
        self.T_out = args.T_out

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley
        r3 = self.downsamplez
        s1 = int(((128 - 1) / r1) + 1)
        s2 = int(((128 - 1) / r2) + 1)
        s3 = int(((128 - 1) / r3) + 1)

        train_dataset = cfd_3d_dataset(
            self.data_path,
            self.downsamplex,
            self.downsampley,
            self.downsamplez,
            self.T_in,
            self.T_out,
            self.out_dim,
            is_train=True,
            train_ratio=self.train_ratio,
        )

        test_dataset = cfd_3d_dataset(
            self.data_path,
            self.downsamplex,
            self.downsampley,
            self.downsamplez,
            self.T_in,
            self.T_out,
            self.out_dim,
            is_train=False,
            train_ratio=self.train_ratio,
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        return train_loader, test_loader, [s1, s2, s3]


class pdebench_unified_dataset(Dataset):
    def __init__(
        self,
        file_path: str,
        split: str,
        T_in: int,
        T_out: int,
        train_ratio: float,
        val_ratio: float = 0.0,
        test_ratio: float = None,
        out_dim: int = None,
        downsamplex: int = 1,
        downsampley: int = 1,
        downsamplez: int = 1,
        preload: bool = False,
        normalize: bool = False,
        norm_type: str = "UnitGaussianNormalizer",
    ):
        """
        统一的PDEBench数据集加载器

        Args:
            file_path: HDF5文件路径
            split: 数据集划分 'train', 'val', 'test'
            train_ratio: 训练集比例
            val_ratio: 验证集比例，默认为0
            test_ratio: 测试集比例，如果为None则为1-train_ratio-val_ratio
            T_in: 输入时间步数
            T_out: 输出时间步数
            out_dim: 输出维度（如果为None则自动推断）
            downsamplex: X方向下采样比例
            downsampley: Y方向下采样比例
            downsamplez: Z方向下采样比例
            preload: 是否预加载所有数据到内存
            normalize: 是否进行数据归一化
            norm_type: 归一化类型
        """
        self.file_path = file_path
        self.split = split
        self.T_in = T_in
        self.T_out = T_out
        self.preload = preload
        self.normalize = normalize
        self.norm_type = norm_type

        # 下采样参数
        self.downsamplex = downsamplex
        self.downsampley = downsampley
        self.downsamplez = downsamplez

        # 验证数据集划分
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

        # 验证比例参数
        if test_ratio is None:
            test_ratio = 1.0 - train_ratio - val_ratio

        if train_ratio + val_ratio + test_ratio != 1.0:
            raise ValueError(
                f"train_ratio({train_ratio}) + val_ratio({val_ratio}) + test_ratio({test_ratio}) must equal 1.0"
            )

        if any(ratio < 0 for ratio in [train_ratio, val_ratio, test_ratio]):
            raise ValueError("All ratios must be non-negative")

        # 验证归一化类型
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}")

        # 读取数据基本信息
        with h5py.File(self.file_path, "r") as h5_file:
            # 获取数据张量信息
            tensor_data = h5_file["tensor"]
            self.original_data_shape = tensor_data.shape  # (N, T, Nx, [Ny], [Nz], C)
            self.n_samples = self.original_data_shape[0]
            self.total_timesteps = self.original_data_shape[1]

            # 判断数据维度并计算下采样后的形状
            if len(self.original_data_shape) == 4:  # 1D: (N, T, Nx, C)
                self.dim = 1
                self.original_spatial_shape = (self.original_data_shape[2],)
                self.spatial_shape = (int(((self.original_data_shape[2] - 1) / self.downsamplex) + 1),)
                self.out_dim = out_dim if out_dim is not None else self.original_data_shape[3]
            elif len(self.original_data_shape) == 5:  # 2D: (N, T, Nx, Ny, C)
                self.dim = 2
                self.original_spatial_shape = (self.original_data_shape[2], self.original_data_shape[3])
                self.spatial_shape = (
                    int(((self.original_data_shape[2] - 1) / self.downsamplex) + 1),
                    int(((self.original_data_shape[3] - 1) / self.downsampley) + 1),
                )
                self.out_dim = out_dim if out_dim is not None else self.original_data_shape[4]
            elif len(self.original_data_shape) == 6:  # 3D: (N, T, Nx, Ny, Nz, C)
                self.dim = 3
                self.original_spatial_shape = (
                    self.original_data_shape[2],
                    self.original_data_shape[3],
                    self.original_data_shape[4],
                )
                self.spatial_shape = (
                    int(((self.original_data_shape[2] - 1) / self.downsamplex) + 1),
                    int(((self.original_data_shape[3] - 1) / self.downsampley) + 1),
                    int(((self.original_data_shape[4] - 1) / self.downsamplez) + 1),
                )
                self.out_dim = out_dim if out_dim is not None else self.original_data_shape[5]
            else:
                raise ValueError(f"Unsupported data shape: {self.original_data_shape}")

            # 验证时间步数
            if T_in + T_out > self.total_timesteps:
                raise ValueError(f"T_in({T_in}) + T_out({T_out}) > total timesteps({self.total_timesteps})")

            # 创建坐标网格
            self._create_coordinates(h5_file)

        # 划分训练/验证/测试集
        self.ntrain = int(self.n_samples * train_ratio)
        self.nval = int(self.n_samples * val_ratio)
        self.ntest = self.n_samples - self.ntrain - self.nval

        if split == "train":
            self.sample_indices = list(range(self.ntrain))
        elif split == "val":
            if self.nval == 0:
                self.sample_indices = []
            else:
                self.sample_indices = list(range(self.ntrain, self.ntrain + self.nval))
        else:  # test
            self.sample_indices = list(range(self.ntrain + self.nval, self.n_samples))

        # 如果启用预加载
        if self.preload:
            self._preload_data()

        # 初始化归一化器
        self.input_normalizer = None
        self.output_normalizer = None
        if self.normalize:
            self._setup_normalizers()

    def _create_coordinates(self, h5_file):
        """创建空间坐标网格"""
        if self.dim == 1:
            x_coords = np.array(h5_file["x-coordinate"], dtype="f")
            # 应用下采样
            x_coords = x_coords[:: self.downsamplex][: self.spatial_shape[0]]
            # 归一化到[0,1]
            x_coords = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min())
            self.grid = torch.tensor(x_coords, dtype=torch.float).unsqueeze(-1)  # (Nx, 1)

        elif self.dim == 2:
            x_coords = np.array(h5_file["x-coordinate"], dtype="f")
            y_coords = np.array(h5_file["y-coordinate"], dtype="f")

            # 应用下采样
            x_coords = x_coords[:: self.downsamplex][: self.spatial_shape[0]]
            y_coords = y_coords[:: self.downsampley][: self.spatial_shape[1]]

            # 归一化到[0,1]
            x_coords = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min())
            y_coords = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min())

            x = torch.tensor(x_coords, dtype=torch.float)
            y = torch.tensor(y_coords, dtype=torch.float)
            X, Y = torch.meshgrid(x, y, indexing="ij")
            self.grid = torch.stack((X, Y), axis=-1)  # (Nx, Ny, 2)

        elif self.dim == 3:
            x_coords = np.array(h5_file["x-coordinate"], dtype="f")
            y_coords = np.array(h5_file["y-coordinate"], dtype="f")
            z_coords = np.array(h5_file["z-coordinate"], dtype="f")

            # 应用下采样
            x_coords = x_coords[:: self.downsamplex][: self.spatial_shape[0]]
            y_coords = y_coords[:: self.downsampley][: self.spatial_shape[1]]
            z_coords = z_coords[:: self.downsamplez][: self.spatial_shape[2]]

            # 归一化到[0,1]
            x_coords = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min())
            y_coords = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min())
            z_coords = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min())

            x = torch.tensor(x_coords, dtype=torch.float)
            y = torch.tensor(y_coords, dtype=torch.float)
            z = torch.tensor(z_coords, dtype=torch.float)
            X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
            self.grid = torch.stack((X, Y, Z), axis=-1)  # (Nx, Ny, Nz, 3)

    def _preload_data(self):
        """预加载所有数据到内存"""
        print(f"Preloading {len(self.sample_indices)} samples from {self.file_path}...")
        self.preloaded_data = {}

        with h5py.File(self.file_path, "r") as h5_file:
            tensor_data = h5_file["tensor"]

            for sample_idx in self.sample_indices:
                # 加载单个样本的所有时间步数据并应用下采样
                if self.dim == 1:
                    sample_data = np.array(tensor_data[sample_idx, :, :: self.downsamplex, :], dtype="f")
                    sample_data = sample_data[:, : self.spatial_shape[0], :]  # (T, Nx, C)
                elif self.dim == 2:
                    sample_data = np.array(
                        tensor_data[sample_idx, :, :: self.downsamplex, :: self.downsampley, :], dtype="f"
                    )
                    sample_data = sample_data[:, : self.spatial_shape[0], : self.spatial_shape[1], :]  # (T, Nx, Ny, C)
                elif self.dim == 3:
                    sample_data = np.array(
                        tensor_data[sample_idx, :, :: self.downsamplex, :: self.downsampley, :: self.downsamplez, :],
                        dtype="f",
                    )
                    sample_data = sample_data[
                        :, : self.spatial_shape[0], : self.spatial_shape[1], : self.spatial_shape[2], :
                    ]  # (T, Nx, Ny, Nz, C)

                self.preloaded_data[sample_idx] = torch.tensor(sample_data, dtype=torch.float)

        print("Preloading completed!")

    def _setup_normalizers(self):
        """设置归一化器"""
        if self.split == "train":  # 只在训练集上计算归一化参数
            print("Computing normalization parameters...")
            all_input_data = []
            all_output_data = []

            # 收集所有训练数据
            for i in range(min(100, len(self.sample_indices))):  # 限制样本数量以节省内存
                _, input_data, output_data = self._get_raw_item(i)
                all_input_data.append(input_data)
                all_output_data.append(output_data)

            all_input_data = torch.stack(all_input_data, dim=0)
            all_output_data = torch.stack(all_output_data, dim=0)

            # 创建归一化器
            if self.norm_type == "UnitTransformer":
                self.input_normalizer = UnitTransformer(all_input_data)
                self.output_normalizer = UnitTransformer(all_output_data)
            elif self.norm_type == "UnitGaussianNormalizer":
                self.input_normalizer = UnitGaussianNormalizer(all_input_data)
                self.output_normalizer = UnitGaussianNormalizer(all_output_data)

            print("Normalization parameters computed.")

    def _get_raw_item(self, idx):
        """获取原始数据项（不应用归一化）"""
        sample_idx = self.sample_indices[idx]

        if self.preload:
            # 使用预加载的数据
            sample_data = self.preloaded_data[sample_idx]
        else:
            # 实时读取数据并应用下采样
            with h5py.File(self.file_path, "r") as h5_file:
                tensor_data = h5_file["tensor"]
                if self.dim == 1:
                    sample_data = np.array(tensor_data[sample_idx, :, :: self.downsamplex, :], dtype="f")
                    sample_data = sample_data[:, : self.spatial_shape[0], :]  # (T, Nx, C)
                elif self.dim == 2:
                    sample_data = np.array(
                        tensor_data[sample_idx, :, :: self.downsamplex, :: self.downsampley, :], dtype="f"
                    )
                    sample_data = sample_data[:, : self.spatial_shape[0], : self.spatial_shape[1], :]  # (T, Nx, Ny, C)
                elif self.dim == 3:
                    sample_data = np.array(
                        tensor_data[sample_idx, :, :: self.downsamplex, :: self.downsampley, :: self.downsamplez, :],
                        dtype="f",
                    )
                    sample_data = sample_data[
                        :, : self.spatial_shape[0], : self.spatial_shape[1], : self.spatial_shape[2], :
                    ]  # (T, Nx, Ny, Nz, C)
                sample_data = torch.tensor(sample_data, dtype=torch.float)

        # 提取输入和输出时间步
        input_data = sample_data[: self.T_in]  # (T_in, Nx, [Ny], [Nz], C)
        output_data = sample_data[self.T_in : self.T_in + self.T_out]  # (T_out, Nx, [Ny], [Nz], C)

        # 重新整理数据形状以匹配现有格式
        if self.dim == 1:
            # 1D: (T, Nx, C) -> (Nx, T*C)
            input_data = input_data.permute(1, 0, 2).reshape(self.spatial_shape[0], -1)
            output_data = output_data.permute(1, 0, 2).reshape(self.spatial_shape[0], -1)
            grid = self.grid  # (Nx, 1)
        elif self.dim == 2:
            # 2D: (T, Nx, Ny, C) -> (Nx*Ny, T*C)
            input_data = input_data.permute(1, 2, 0, 3).reshape(-1, self.T_in * self.out_dim)
            output_data = output_data.permute(1, 2, 0, 3).reshape(-1, self.T_out * self.out_dim)
            grid = self.grid.reshape(-1, 2)  # (Nx*Ny, 2)
        elif self.dim == 3:
            # 3D: (T, Nx, Ny, Nz, C) -> (Nx*Ny*Nz, T*C)
            input_data = input_data.permute(1, 2, 3, 0, 4).reshape(-1, self.T_in * self.out_dim)
            output_data = output_data.permute(1, 2, 3, 0, 4).reshape(-1, self.T_out * self.out_dim)
            grid = self.grid.reshape(-1, 3)  # (Nx*Ny*Nz, 3)

        return grid, input_data, output_data

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        grid, input_data, output_data = self._get_raw_item(idx)

        # 应用归一化
        if self.normalize and self.input_normalizer is not None:
            input_data = self.input_normalizer.encode(input_data.unsqueeze(0)).squeeze(0)

        if self.normalize and self.output_normalizer is not None:
            output_data = self.output_normalizer.encode(output_data.unsqueeze(0)).squeeze(0)

        return grid, input_data, output_data


class pdebench_unified(object):
    def __init__(self, args):
        self.file_path = args.data_path
        self.train_ratio = getattr(args, "train_ratio", 0.8)
        self.val_ratio = getattr(args, "val_ratio", 0.0)
        self.test_ratio = getattr(args, "test_ratio", None)
        self.T_in = args.T_in
        self.T_out = args.T_out
        self.batch_size = args.batch_size
        self.out_dim = getattr(args, "out_dim", None)  # 允许自动推断
        self.downsamplex = getattr(args, "downsamplex", 1)
        self.downsampley = getattr(args, "downsampley", 1)
        self.downsamplez = getattr(args, "downsamplez", 1)
        self.preload = getattr(args, "preload", True)
        self.num_workers = getattr(args, "num_workers", 4)  # 默认值改为4
        self.normalize = getattr(args, "normalize", False)
        self.norm_type = getattr(args, "norm_type", "UnitGaussianNormalizer")

        # 验证归一化类型
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}")

    def get_loader(self):
        # 创建训练数据集
        train_dataset = pdebench_unified_dataset(
            file_path=self.file_path,
            split="train",
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            T_in=self.T_in,
            T_out=self.T_out,
            out_dim=self.out_dim,
            downsamplex=self.downsamplex,
            downsampley=self.downsampley,
            downsamplez=self.downsamplez,
            preload=self.preload,
            normalize=self.normalize,
            norm_type=self.norm_type,
        )

        # 创建验证数据集
        val_dataset = None
        val_loader = None
        if self.val_ratio > 0:
            val_dataset = pdebench_unified_dataset(
                file_path=self.file_path,
                split="val",
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                T_in=self.T_in,
                T_out=self.T_out,
                out_dim=self.out_dim,
                downsamplex=self.downsamplex,
                downsampley=self.downsampley,
                downsamplez=self.downsamplez,
                preload=self.preload,
                normalize=False,  # 验证集不重新计算归一化参数
                norm_type=self.norm_type,
            )

        # 创建测试数据集
        test_dataset = pdebench_unified_dataset(
            file_path=self.file_path,
            split="test",
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            T_in=self.T_in,
            T_out=self.T_out,
            out_dim=self.out_dim,
            downsamplex=self.downsamplex,
            downsampley=self.downsampley,
            downsamplez=self.downsamplez,
            preload=self.preload,
            normalize=False,  # 测试集不重新计算归一化参数
            norm_type=self.norm_type,
        )

        # 如果训练集启用了归一化，将归一化器传递给验证集和测试集
        if self.normalize and train_dataset.input_normalizer is not None:
            if val_dataset is not None:
                val_dataset.input_normalizer = train_dataset.input_normalizer
                val_dataset.output_normalizer = train_dataset.output_normalizer
                val_dataset.normalize = True

            test_dataset.input_normalizer = train_dataset.input_normalizer
            test_dataset.output_normalizer = train_dataset.output_normalizer
            test_dataset.normalize = True

            # 保存归一化器到主对象
            self.input_normalizer = train_dataset.input_normalizer
            self.output_normalizer = train_dataset.output_normalizer

            # 移动归一化器到GPU
            if hasattr(self.input_normalizer, "cuda"):
                self.input_normalizer.cuda()
            if hasattr(self.output_normalizer, "cuda"):
                self.output_normalizer.cuda()

        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

        if val_dataset is not None:
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
            )

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        # 返回空间形状信息
        shape_info = list(train_dataset.spatial_shape)

        print(f"Dataloading is over. Dataset info:")
        print(f"  - Dimension: {train_dataset.dim}D")
        print(f"  - Original spatial shape: {train_dataset.original_spatial_shape}")
        print(f"  - Downsampled spatial shape: {train_dataset.spatial_shape}")
        print(f"  - Downsample ratios: x={self.downsamplex}, y={self.downsampley}, z={self.downsamplez}")
        print(f"  - Output channels: {train_dataset.out_dim}")
        print(f"  - Train samples: {len(train_dataset)}")
        if val_dataset is not None:
            print(f"  - Val samples: {len(val_dataset)}")
        print(f"  - Test samples: {len(test_dataset)}")
        print(f"  - Preload: {self.preload}")
        print(f"  - Normalize: {self.normalize}")

        return train_loader, val_loader, test_loader, shape_info
