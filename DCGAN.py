import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid

import time

# from adamPre import AdamPre
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
    def __init__(self, opt, verbose=False):
        self.opt = opt
        self.device = 'cuda' if opt.cuda else 'cpu'
        self.verbose = verbose
        # if opt.verbose:
        #     self.verbose = not opt.distributed or dist.get_rank() == 0

        ################################################################
        # Initializing Generator and Discriminator Networks
        ################################################################
        self.nz = int(opt.nz)
        ngf = int(opt.ngf)
        ndf = int(opt.ndf)
        nc = int(opt.nc)

        self.G = _netG(opt.ngpu, self.nz, ngf, nc).to(self.device)
        self.G.apply(weights_init)
        self.G_losses = []

        if opt.netG != '':
            self.G.load_state_dict(torch.load(opt.netG))
            self.G_losses = torch.load('{}/G_losses.pth'.format(self.opt.outf))

        self.D = _netD(opt.ngpu, ndf, nc).to(self.device)
        self.D.apply(weights_init)
        self.D_losses = []

        if opt.netD != '':
            self.D.load_state_dict(torch.load(opt.netD))
            self.D_losses = torch.load('{}/D_losses.pth'.format(self.opt.outf))

        if self.verbose:
            print(self.G)
            print(self.D)

        ################################################################
        # Initialize Loss Function
        ################################################################
        self.criterion = nn.BCELoss().to(self.device)

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
                self.D = torch.nn.parallel.DistributedDataParallel(self.D)
                self.G = torch.nn.parallel.DistributedDataParallel(self.G)
            else:
                self.D = torch.nn.parallel.DistributedDataParallelCPU(self.D)
                self.G = torch.nn.parallel.DistributedDataParallelCPU(self.G)

    def checkpoint(self, epoch):
        torch.save(self.G.state_dict(), '{0}/netG_epoch_{1}.pth'.format(
            self.opt.outf, epoch))
        torch.save(self.D.state_dict(), '{0}/netD_epoch_{1}.pth'.format(
            self.opt.outf, epoch))
        torch.save(self.G_losses, '{}/G_losses.pth'.format(self.opt.outf))
        torch.save(self.D_losses, '{}/D_losses.pth'.format(self.opt.outf))

    def train(self, niter, dataset, lookahead_step=1.0, n_batches_viz=1,
              viz_every=1000):
        """
        Custom DCGAN training function using prediction steps
        """

        real_label = 1
        fake_label = 0
        img_list = []
        fixed_noise = torch.randn(1, self.nz, 1, 1)
        itr = 0

        for epoch in range(niter):
            for i, data in enumerate(dataset):
                if self.verbose:
                    c1 = time.time()
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

                self.G_losses.append(errG.data)
                self.D_losses.append(errD.data)

                if self.verbose:
                    print('[%d/%d][%d/%d] %.2f secs, Loss_D:%.4f '
                          'Loss_G:%.4f D(x): %.4f D(G(z)): %.4f / %.4f' %
                          (epoch, niter, i, len(dataset), time.time() - c1,
                           errD.data, errG.data, D_x, D_G_z1, D_G_z2))

                    if itr % viz_every == 0:
                        self.checkpoint(epoch + i/len(dataset))

                        with torch.no_grad():
                            fake = self.G(fixed_noise).detach().cpu()
                            img_list.append(make_grid(fake, padding=2))

                itr += 1

            if self.verbose:
                self.checkpoint(epoch)

        return self.G_losses, self.D_losses, img_list