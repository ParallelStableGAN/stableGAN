import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset


def generate_data_SingleBatch(num_mode=100, radius=24, center=(0, 0),
                              sigma=0.01, batchSize=64):
    num_data_per_class = int(np.ceil(batchSize/num_mode))
    total_data = {}

    t = np.linspace(0, 2*np.pi, num_mode + 1)
    t = t[:-1]
    x = np.cos(t)*radius + center[0]
    y = np.sin(t)*radius + center[1]

    modes = np.vstack([x, y]).T

    for idx, mode in enumerate(modes):
        x = np.random.normal(mode[0], sigma, num_data_per_class)
        y = np.random.normal(mode[1], sigma, num_data_per_class)
        total_data[idx] = np.vstack([x, y]).T

    all_points = np.vstack([values for values in total_data.values()])
    all_points = np.random.permutation(all_points)[0:batchSize]
    return torch.from_numpy(all_points).float()


class MoGDataset(Dataset):
    """MoG Data Generator"""

    def __init__(self, num_mode=100, radius=24, center=(0, 0), sigma=0.01,
                 batchSize=1):
        self.num_mode = num_mode
        self.radius = radius
        self.center = center
        self.sigma = sigma
        self.batchSize = batchSize

    def __len__(self):
        return self.batchSize

    def __getitem__(self, idx):
        return generate_data_SingleBatch(
            num_mode=self.num_mode, radius=self.radius, center=self.center,
            sigma=self.sigma, batchSize=self.batchSize)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.orthogonal(m.weight)
        init.constant(m.bias, 0.1)


class _netG(nn.Module):
    def __init__(self, ngpu, nz, ngf):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(nz, ngf),
            nn.Tanh(),
            nn.Linear(ngf, ngf),
            nn.Tanh(),
            nn.Linear(ngf, 2),
        )

    def forward(self, input):
        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)
        return output


class _netD(nn.Module):
    def __init__(self, ngpu, ndf):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(2, ndf),
            nn.Tanh(),
            nn.Linear(ndf, ndf),
            nn.Tanh(),
            nn.Linear(ndf, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)
