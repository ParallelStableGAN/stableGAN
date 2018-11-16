from __future__ import print_function

from basemodel import GAN
from stabVisualize import finalViz

import argparse
import os
import random

import numpy as np

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

# Information regarding data input
parser.add_argument('--batchSize', type=int, default=512,
                    help='input batch size')
parser.add_argument('--modes', type=int, default=8,
                    help='total number of gaussian modes to consider')
parser.add_argument('--radius', type=int, default=1,
                    help='radius of circle with MoG')
parser.add_argument('--sigma', type=float, default=0.01,
                    help='variance of gaussians, default=0.01')

# Information regarding network
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--ndf', type=int, default=128)
parser.add_argument('--nz', type=int, default=2,
                    help='size of the latent z vector')
parser.add_argument('--ngpu', type=int, default=1,
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
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--pdhgGLookAhead', action='store_true',
                    help='enables generator lookahead')
parser.add_argument('--pdhgDLookAhead', action='store_true',
                    help='enables discriminator lookahead')
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
parser.add_argument('--verbose', action='store_true',
                    help='displays additional information')

# Options for visualization
parser.add_argument('--viz_every', type=int, default=10000,
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

    gan = GAN(opt)

    gan.train(opt.niter)

    finalViz(opt)


if __name__ == '__main__':
    main()
