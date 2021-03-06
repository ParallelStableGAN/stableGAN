import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse

from DCGAN import _netG, weights_init

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./outf', help='Path to saved weights')
parser.add_argument('--spath', default='./outf', help='Path to saved weights')
parser.add_argument('--ngf', type=int, default=64,
                    help='Number of initial features in generator')
parser.add_argument('--nc', type=int, default=3,
                    help='Number of channels (RGB)')
parser.add_argument('--nz', type=int, default=100,
                    help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=0,
                    help='number of epochs to train for')
parser.add_argument('--nrow', type=int, default=1,
                    help='number of images in row of grid')
# parser.add_argument('--dist', action=store_true,
#                     help='')

args = parser.parse_args()

netG = _netG(args.nz, args.ngf, args.nc).apply(weights_init)
snetG = _netG(args.nz, args.ngf, args.nc).apply(weights_init)
fixed_noise = torch.randn(args.nrow, args.nz, 1, 1)
img_list = []

with torch.no_grad():
    fake = netG(fixed_noise).detach().cpu()
    sfake = snetG(fixed_noise).detach().cpu()
    img_list.append(
        make_grid(
            torch.cat((fake, sfake)), padding=2, normalize=True))

for i in range(args.niter):
    print('.', end="", flush=True)
    weights = ("{}/netG_epoch_{}.pth".format(args.path, i))
    sweights = ("{}/netG_epoch_{}.pth".format(args.spath, i))
    try:
        netG.load_state_dict(torch.load(weights, map_location='cpu'))
        snetG.load_state_dict(torch.load(sweights, map_location='cpu'))
    except:
        netG = torch.nn.parallel.DataParallel(netG)
        snetG = torch.nn.parallel.DataParallel(snetG)
        netG.load_state_dict(torch.load(weights, map_location='cpu'))
        snetG.load_state_dict(torch.load(sweights, map_location='cpu'))
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
        sfake = snetG(fixed_noise).detach().cpu()
        img_list.append(
            make_grid(
                torch.cat((fake, sfake)), padding=2, normalize=True))

# Animate fixed noise
fig = plt.figure(figsize=(args.nrow, args.nrow))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)]
       for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=20000,
                                blit=True)
ani.save('{}/comp_Fixed_noise.gif'.format(args.path), dpi=80, writer='imagemagick')
ani.save('{}/comp_Fixed_noise.mp4'.format(args.path))
