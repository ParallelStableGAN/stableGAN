import torch
import torch.nn as nn
import torch.nn.init as init
import torch.distributed as dist
from torch.autograd import Variable

import time

from adamPre import AdamPre
from mogdata import generate_data_SingleBatch
from stabVisualize import iterViz


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


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.orthogonal(m.weight)
        init.constant(m.bias, 0.1)


class GAN():
    def __init__(self, opt):
        self.opt = opt

        ################################################################
        # Initializing Generator and Discriminator Network
        ################################################################
        nz = int(opt.nz)
        ngf = int(opt.ngf)
        ndf = int(opt.ndf)

        self.G = _netG(opt.ngpu, nz, ngf)
        self.G.apply(weights_init)

        if opt.netG != '':
            self.G.load_state_dict(torch.load(opt.netG))

        self.D = _netD(opt.ngpu, ndf)
        self.D.apply(weights_init)

        if opt.netD != '':
            self.D.load_state_dict(torch.load(opt.netD))

        if opt.verbose and not self.opt.distributed or dist.get_rank() == 0:
            print(self.G)
            print(self.D)

        ################################################################
        # Initialize Loss Function
        ################################################################
        self.criterion = nn.BCELoss()

        # Initialize label inputs for this training
        self.input = torch.FloatTensor(opt.batchSize, 2)
        self.noise = torch.FloatTensor(opt.batchSize, nz)
        self.label = torch.FloatTensor(opt.batchSize)
        self.real_label = 1
        self.fake_label = 0
        self.fixed_noise = torch.FloatTensor(opt.batchSize*opt.n_batches_viz,
                                             nz).normal_(0, 1)

        ################################################################
        # Handle special training modes CUDA and Distributed
        ################################################################
        self.verbose = opt.verbose
        if opt.cuda:
            self.D.cuda()
            self.G.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()
            self.noise = self.noise.cuda()
            self.fixed_noise = self.fixed_noise.cuda()
            if opt.distributed:
                self.D = torch.nn.parallel.DistributedDataParallel(self.D)
                self.G = torch.nn.parallel.DistributedDataParallel(self.G)
                self.verbose = opt.verbose and dist.get_rank() == 0
            else:
                self.verbose = opt.verbose
        elif opt.distributed:
            self.D = torch.nn.parallel.DistributedDataParallelCPU(self.D)
            self.G = torch.nn.parallel.DistributedDataParallelCPU(self.G)
            self.verbose = opt.verbose and dist.get_rank() == 0

        self.input = Variable(self.input)
        self.label = Variable(self.label)
        self.noise = Variable(self.noise)
        self.fixed_noise = Variable(self.fixed_noise)

        ################################################################
        # Set Prediction Enabled Adam Optimizer settings
        ################################################################
        self.optimizerD = AdamPre(self.D.parameters(), lr=opt.lr/opt.DLRatio,
                                  betas=(opt.beta1, 0.999), name='optD')
        self.optimizerG = AdamPre(self.G.parameters(), lr=opt.lr/opt.GLRatio,
                                  betas=(opt.beta1, 0.999), name='optG')

    def train(self, niter):

        fs = []
        # np_samples = []

        for i in range(niter):
            if self.verbose:
                c1 = time.clock()

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            # sampling input batch
            real_cpu = generate_data_SingleBatch(
                num_mode=self.opt.modes, radius=self.opt.radius, center=(0, 0),
                sigma=self.opt.sigma, batchSize=self.opt.batchSize)

            batch_size = real_cpu.size(0)
            self.input.data.resize_(real_cpu.size()).copy_(real_cpu)
            self.label.data.resize_(batch_size).fill_(self.real_label)

            self.D.zero_grad()

            output = self.D(self.input)
            errD_real = self.criterion(output, self.label)
            errD_real.backward()
            D_x = output.data.mean()

            # Update the generator weights with prediction
            # We avoid update during the first iteration
            if not i == 0 and self.opt.pdhgGLookAhead:
                self.optimizerG.stepLookAhead()

            # train with fake
            self.noise.data.resize_(batch_size, self.opt.nz)
            self.noise.data.normal_(0, 1)
            self.label.data.resize_(batch_size)
            self.label.data.fill_(self.fake_label)

            fake = self.G(self.noise)
            output = self.D(fake.detach())
            errD_fake = self.criterion(output, self.label)
            errD_fake.backward()
            D_G_z1 = output.data.mean()

            errD = errD_real + errD_fake
            self.optimizerD.step()

            # restore the previous (non-predicted) weights of Generator
            if not i == 0 and self.opt.pdhgGLookAhead:
                self.optimizerG.restoreStepLookAhead()

            ############################
            # (2) Update G network: maximize -log(1 - D(G(z)))
            ###########################
            # Update discriminator weights with prediction;
            # restore after the generator update.
            if self.opt.pdhgDLookAhead:
                self.optimizerD.stepLookAhead()

            # Unlike DCGAN code, we use original loss for generator.
            # Hence we fill fake labels.
            self.label.data.fill_(self.fake_label)

            self.G.zero_grad()

            fake = self.G(self.noise)
            output = self.D(fake)
            errG = -self.criterion(output, self.label)
            errG.backward()
            D_G_z2 = output.data.mean()
            self.optimizerG.step()

            # restore back discriminator weights
            if self.opt.pdhgDLookAhead:
                self.optimizerD.restoreStepLookAhead()

            if self.opt.plotLoss:
                f = [errD.data[0], errG.data[0]]
                fs.append(f)

            if self.verbose:
                print('[%d/%d] Loss_D:%.4f Loss_G:%.4f D(x)'
                      ': %.4f D(G(z)): %.4f / %.4f' %
                      (i, niter, errD.data[0], errG.data[0], D_x, D_G_z1,
                       D_G_z2))

                print("itr=", i, "clock time elapsed=", time.clock() - c1)

            if i % self.opt.viz_every == 0 or i == niter - 1:

                if self.verbose:
                    # save checkpoints
                    torch.save(
                        self.G.state_dict(), '{0}/netG_epoch_{1}.pth'.format(
                            self.opt.outf, i))
                    torch.save(
                        self.D.state_dict(), '{0}/netD_epoch_{1}.pth'.format(
                            self.opt.outf, i))

                    iterViz(self.opt, i, self.G, self.fixed_noise)
