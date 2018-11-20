from __future__ import print_function

from DCGAN import DCGAN
# from stabVisualize import finalViz
# from mogdata import MoGDataset, _netG, _netD, weights_init
# from mnist_gan import mnist_data

import argparse
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

# Information regarding data input
parser.add_argument('--batchSize', type=int, default=64,
                    help='input batch size')

# Information regarding network
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3, help='Number of channels')
parser.add_argument('--nz', type=int, default=100,
                    help='size of the latent z vector')
parser.add_argument('--ngpu', type=int, default=0,
                    help='number of GPUs to use')
parser.add_argument('--netG', default='',
                    help="path to netG (to continue training)")
parser.add_argument('--netD', default='',
                    help="path to netD (to continue training)")

# Training/Optimizer information
parser.add_argument('--niter', type=int, default=50000,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate, default=0.001')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--pred', action='store_true',
                    help='enables generator and discriminator lookahead')
parser.add_argument('--GLRatio', type=float, default=1.0,
                    help='scaling factor for lr of generator')
parser.add_argument('--DLRatio', type=float, default=1.0,
                    help='scaling factor for lr of discriminator')

# Miscellaneous information
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='.',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--deviceID', type=int, help='deviceID', default=0)
parser.add_argument('--num_workers', type=int, default=1,
                    help='DataLoader worker number')
parser.add_argument('--verbose', action='store_true',
                    help='displays additional information')

# Options for visualization
parser.add_argument('--viz_every', type=int, default=100,
                    help='plotting visualization every few iteration')
parser.add_argument('--n_batches_viz', type=int, default=10,
                    help='number of samples used for visualization')
parser.add_argument('--markerSize', type=float, help='input batch size')
parser.add_argument('--plotRealData', action='store_true',
                    help='saves real samples')
parser.add_argument('--plotLoss', action='store_true',
                    help='Enables plotting of loss function')

# Added options for distributed training
parser.add_argument('--distributed', action='store_true',
                    help='enables distributed processes')
parser.add_argument('--local_rank', default=0, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist_backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--dist_group', default='', type=str,
                    help='distributed group name')
parser.add_argument('--dist_init', default='env://', type=str,
                    help='url used to set up distributed training')

# parser.add_argument('--world_size', default=1, type=int,
#                     help='Number of concurrent processes')


def main():
    opt = parser.parse_args()
    # print(opt)

    if opt.distributed:
        if opt.cuda:
            torch.cuda.set_device(opt.local_rank)

        dist.init_process_group(backend=opt.dist_backend, init_method='env://')

        print("INITIALIZED! Rank:", dist.get_rank())
        opt.batchSize = int(opt.batchSize/dist.get_world_size())

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
        torch.backends.cudnn.enabled = False
        print("torch.backends.cudnn.enabled is:", torch.backends.cudnn.enabled)

    cudnn.benchmark = True

    if torch.cuda.is_available():
        opt.ngpu = int(opt.ngpu)
        if not opt.cuda:
            print("WARNING: You have a CUDA device,"
                  " so you should probably run with --cuda")
    else:
        if int(opt.ngpu) > 0:
            print("WARNING: CUDA not available, cannot use --ngpu =", opt.ngpu)
        opt.ngpu = 0

    verbose = (not opt.distributed
               or dist.get_rank() == 0) if opt.verbose else False

    lookahead_step = 1.0 if opt.pred else 0.0

    data = datasets.MNIST(
        './data', download=True, transform=transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    sampler = DistributedSampler(data) if opt.distributed else None
    ganLoader = DataLoader(data, batch_size=opt.batchSize, sampler=sampler,
                           shuffle=(sampler is None),
                           num_workers=opt.num_workers, pin_memory=True)
    gan = DCGAN(opt, verbose)
    G_losses, D_losses, img_list = gan.train(opt.niter, ganLoader,
                                             lookahead_step=lookahead_step,
                                             viz_every=opt.viz_every)

    if verbose:
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('{}/Loss_plot.png'.format(opt.outf))

        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)]
               for i in img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000,
                                        repeat_delay=1000, blit=True)
        ani.save('{}/Fized_noise.mp4'.format(opt.outf))

        # Grab a batch of real images from the dataloader
        real_batch = next(iter(ganLoader))

        # Plot the real images
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imsave(
            '{}/Real_images.png'.format(opt.outf),
            np.transpose(
                make_grid(real_batch[0][:64], padding=5, normalize=True).cpu(),
                (1, 2, 0)))

        # Plot the fake images from the last epoch
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imsave('{}/Fake_images.png'.format(opt.outf),
                   np.transpose(img_list[-1], (1, 2, 0)))


if __name__ == '__main__':
    main()
