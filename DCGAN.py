import torch
import torch.nn as nn
import torch.optim as optim
import distcpu as myd
# import torch.distributed as dist
# from torchvision.utils import make_grid

import time

# from adamPre import AdamPre
from prediction import PredOpt


class _netG(nn.Module):
    def __init__(self, nz, ngf, nc=3):
        super(_netG, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


class _netD(nn.Module):
    def __init__(self, ndf, nc=3):
        super(_netD, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGAN():
    def __init__(self, opt, verbose=False):
        self.opt = opt
        self.distributed = opt.distributed
        self.verbose = verbose
        self.cuda = opt.cuda
        self.local_rank = opt.local_rank
        device = 'cpu'
        if opt.cuda:
            device = 'cuda' + ':' + str(opt.local_rank)
        self.device = torch.device(device)

        ################################################################
        # Initializing Generator and Discriminator Networks
        ################################################################
        self.nz = int(opt.nz)
        ngf = int(opt.ngf)
        ndf = int(opt.ndf)
        nc = int(opt.nc)

        self.G = _netG(self.nz, ngf, nc).to(self.device)
        self.G.apply(weights_init)

        if opt.netG != '':
            self.G.load_state_dict(torch.load(opt.netG))
            self.G_losses = torch.load('{}/G_losses.pth'.format(self.opt.outf))

        self.D = _netD(ndf, nc).to(self.device)
        self.D.apply(weights_init)

        if opt.netD != '':
            self.D.load_state_dict(torch.load(opt.netD))
            self.D_losses = torch.load('{}/D_losses.pth'.format(self.opt.outf))

        if self.verbose:
            print(self.G)
            print(self.D)

        ################################################################
        # Initialize Loss Function
        ################################################################
        self.criterion = nn.BCELoss()
        if opt.cuda:
            self.criterion.cuda(opt.local_rank)

        ################################################################
        # Set Prediction Enabled Adam Optimizer settings
        ################################################################
        self.optimizerD = optim.Adam(self.D.parameters(), lr=opt.lr,
                                     betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.G.parameters(), lr=opt.lr,
                                     betas=(opt.beta1, 0.999))
        self.optimizer_predD = PredOpt(self.D.parameters())
        self.optimizer_predG = PredOpt(self.G.parameters())

        ################################################################
        # Handle special Distributed training modes
        ################################################################
        if opt.distributed:
            if opt.cuda:
                ids = [
                    i for i in range(opt.ngpu*opt.local_rank, opt.ngpu +
                                     opt.local_rank*opt.ngpu)
                ]

                self.D = nn.parallel.DistributedDataParallel(
                    self.D, device_ids=ids).to(self.device)

                self.G = nn.parallel.DistributedDataParallel(
                    self.G, device_ids=ids).to(self.device)
            else:
                if opt.sync_every == 1:
                    self.D = nn.parallel.DistributedDataParallelCPU(self.D)
                    self.G = nn.parallel.DistributedDataParallelCPU(self.G)
                else:
                    self.D = myd.DistributedDataParallelCPU(self.D)
                    self.G = myd.DistributedDataParallelCPU(self.G)
        else:
            if opt.cuda:
                # torch.cuda.set_device(opt.local_rank)
                if torch.cuda.device_count() > 1:
                    self.D = nn.parallel.DataParallel(self.D).to(self.device)
                    self.G = nn.parallel.DataParallel(self.G).to(self.device)

    def checkpoint(self, epoch):
        torch.save(self.G.state_dict(), '{0}/netG_epoch_{1}.pth'.format(
            self.opt.outf, epoch))
        torch.save(self.D.state_dict(), '{0}/netD_epoch_{1}.pth'.format(
            self.opt.outf, epoch))
        torch.save(self.G_losses, '{}/G_losses.pth'.format(self.opt.outf))
        torch.save(self.D_losses, '{}/D_losses.pth'.format(self.opt.outf))
        torch.save(self.Dxs, '{}/D_xs.pth'.format(self.opt.outf))
        torch.save(self.DGz1s, '{}/D_G_z1s.pth'.format(self.opt.outf))
        torch.save(self.DGz2s, '{}/D_G_z2s.pth'.format(self.opt.outf))

    def train(self, niter, dataset, gpred_step=1.0, dpred_step=0.0,
              sync_every=1, viz_every=10):
        # n_batches_viz=10):
        """
        Custom DCGAN training function using prediction steps
        """

        real_label = 1
        fake_label = 0
        self.D_losses = []
        self.G_losses = []
        self.Dxs = []
        self.DGz1s = []
        self.DGz2s = []
        img_list = []

        # fixed_noise = torch.randn(n_batches_viz, self.nz, 1, 1)
        # if self.cuda:
        #     fixed_noise.cuda(self.local_rank, non_blocking=True)

        itr = 0

        for epoch in range(niter):
            c0 = time.time()
            for i, data in enumerate(dataset):

                c1 = time.time()

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################

                self.D.zero_grad()

                # train on real first
                real_cpu, _ = data
                b_size = real_cpu.size(0)

                input = real_cpu
                if self.cuda:
                    input.cuda(self.local_rank, non_blocking=True)

                label = torch.full((b_size, ), real_label, device=self.device)
                output = self.D(input)
                errD_real = self.criterion(output, label)
                errD_real.backward()
                D_x = output.data.mean()

                # train with fake
                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)

                # Compute gradient of D w/ predicted G
                with self.optimizer_predG.lookahead(step=gpred_step):
                    fake = self.G(noise)
                    label.fill_(fake_label)
                    output = self.D(fake.detach())
                    errD_fake = self.criterion(output, label)
                    errD_fake.backward()
                    D_G_z1 = output.data.mean()
                    errD = errD_real + errD_fake
                    self.optimizerD.step()
                    self.optimizer_predD.step()

                ############################
                # (2) Update G network: maximize -log(1 - D(G(z)))
                ###########################
                self.G.zero_grad()
                label.fill_(real_label)

                # Compute gradient of G w/ predicted D
                with self.optimizer_predD.lookahead(step=dpred_step):
                    fake = self.G(noise)
                    output = self.D(fake)
                    errG = self.criterion(output, label)
                    errG.backward()
                    D_G_z2 = output.data.mean()
                    self.optimizerG.step()
                    self.optimizer_predG.step()

                self.G_losses.append(errG.data)
                self.D_losses.append(errD.data)
                self.Dxs.append(D_x)
                self.DGz1s.append(D_G_z1)
                self.DGz2s.append(D_G_z2)

                sync_print = ''
                if sync_every != 1 and itr % sync_every == 0:
                    sync_print = '\t --  synced at iteration ' + str(itr)
                    self.D.sync_parameters()
                    self.G.sync_parameters()

                if self.verbose:
                    print('[{}/{}][{}/{}] {:0.2f} secs, Loss_D:{:0.4f} Loss_G:'
                          '{:0.4f} D(x): {:0.4f} D(G(z)): {:0.4f} / {:0.4f}{}'.
                          format(epoch, niter, i, len(dataset),
                                 time.time() - c1, errD.data, errG.data, D_x,
                                 D_G_z1, D_G_z2, sync_print))

                    if itr % viz_every == 0:
                        self.checkpoint(epoch)

                itr += 1

            if sync_every != 1:
                if self.verbose:
                    print('Synchronizing Parameters at epoch:', i)
                self.D.sync_parameters()
                self.G.sync_parameters()

            if self.verbose:
                self.checkpoint(epoch)

            if self.verbose:
                print("Finished epoch in {:0.2f} seconds".format(time.time() - c0))

        return (self.G_losses, self.D_losses, self.Dxs, self.DGz1s, self.DGz2s,
                img_list)
