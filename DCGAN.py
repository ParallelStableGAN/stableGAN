import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim

import time

from adamPre import AdamPre
from prediction import PredOpt


class _netG(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc=3):
        super(_netG, self).__init__()
        self.ngpu = ngpu
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
        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)
        return output


class _netD(nn.Module):
    def __init__(self, ngpu, ndf, nc=3):
        super(_netD, self).__init__()
        self.ngpu = ngpu
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
        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGAN():
    def __init__(self, opt, nc):
        self.opt = opt

        ################################################################
        # Initializing Generator and Discriminator Networks
        ################################################################
        self.nz = int(opt.nz)
        ngf = int(opt.ngf)
        ndf = int(opt.ndf)
        self.device = 'cuda' if opt.cuda else 'cpu'

        self.G = _netG(opt.ngpu, self.nz, ngf, nc).to(self.device)
        self.G.apply(weights_init)

        if opt.netG != '':
            self.G.load_state_dict(torch.load(opt.netG))

        self.D = _netD(opt.ngpu, ndf, nc).to(self.device)
        self.D.apply(weights_init)

        if opt.netD != '':
            self.D.load_state_dict(torch.load(opt.netD))

        if opt.verbose and (not self.opt.distributed or dist.get_rank() == 0):
            print(self.G)
            print(self.D)

        ################################################################
        # Initialize Loss Function
        ################################################################
        self.criterion = nn.BCELoss().to(self.device)

        ################################################################
        # Set Prediction Enabled Adam Optimizer settings
        ################################################################
        # self.optimizerD = AdamPre(self.D.parameters(), lr=opt.lr/opt.DLRatio,
        #                           betas=(opt.beta1, 0.999), name='optD')
        # self.optimizerG = AdamPre(self.G.parameters(), lr=opt.lr/opt.GLRatio,
        #                           betas=(opt.beta1, 0.999), name='optG')
        self.optimizerD = optim.Adam(self.D.parameters(), lr=opt.lr,
                                     betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.G.parameters(), lr=opt.lr,
                                     betas=(opt.beta1, 0.999))
        self.optimizer_predD = PredOpt(self.D.parameters())
        self.optimizer_predG = PredOpt(self.G.parameters())

        ################################################################
        # Handle special Distributed training modes
        ################################################################
        self.verbose = opt.verbose
        if opt.distributed:
            if opt.cuda:
                self.D = torch.nn.parallel.DistributedDataParallel(self.D)
                self.G = torch.nn.parallel.DistributedDataParallel(self.G)
                self.verbose = opt.verbose and dist.get_rank() == 0
            else:
                self.D = torch.nn.parallel.DistributedDataParallelCPU(self.D)
                self.G = torch.nn.parallel.DistributedDataParallelCPU(self.G)
                self.verbose = opt.verbose and dist.get_rank() == 0

    def train(self, niter, dataset, lookahead_step=1.0, plotLoss=False,
              n_batches_viz=1):
        """
        Custom DCGAN training function using prediction steps
        """

        real_label = 1
        fake_label = 0
        fs = []

        for epoch in range(niter):
            for i, data in enumerate(dataset):
                if self.verbose:
                    c1 = time.clock()
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################

                self.D.zero_grad()

                # train on real first
                real_cpu, _ = data
                b_size = real_cpu.size(0)
                input = real_cpu.to(self.device)
                label = torch.full((b_size, ), real_label, device=self.device)

                output = self.D(input)
                errD_real = self.criterion(output, label)
                errD_real.backward()
                D_x = output.data.mean()

                # train with fake
                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)

                # Compute gradient of D w/ predicted G
                with self.optimizer_predG.lookahead(step=lookahead_step):
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
                with self.optimizer_predD.lookahead(step=lookahead_step):
                    fake = self.G(noise)
                    output = self.D(fake)
                    errG = self.criterion(output, label)
                    errG.backward()
                    D_G_z2 = output.data.mean()
                    self.optimizerG.step()
                    self.optimizer_predG.step()

                if plotLoss:
                    f = [errD.data[0], errG.data[0]]
                    fs.append(f)

                if self.verbose:
                    print('[%d/%d][%d/%d] Loss_D:%.4f Loss_G:%.4f D(x)'
                          ': %.4f D(G(z)): %.4f / %.4f' %
                          (epoch, niter, i, len(dataset), errD.data[0],
                           errG.data[0], D_x, D_G_z1, D_G_z2))

                    print("itr=", epoch, "clock time elapsed=",
                          time.clock() - c1)
                # if i % self.opt.viz_every == 0 or epoch == niter - 1:
                #         iterViz(self.opt, i, self.G, self.fixed_noise)

            if self.verbose:
                # save checkpoints
                torch.save(
                    self.G.state_dict(), '{0}/netG_epoch_{1}.pth'.format(
                        self.opt.outf, epoch))
                torch.save(
                    self.D.state_dict(), '{0}/netD_epoch_{1}.pth'.format(
                        self.opt.outf, epoch))
