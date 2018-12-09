import torch
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse
import os

from DCGAN import _netG, weights_init

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./outf', help='Path to saved weights')
parser.add_argument('--spath', default='./outf',
                    help='Path to stabilized weights')
parser.add_argument('--ngf', type=int, default=64,
                    help='Number of initial features in generator')
parser.add_argument('--nc', type=int, default=3,
                    help='Number of channels (RGB)')
parser.add_argument('--nz', type=int, default=100,
                    help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=0,
                    help='number of epochs to train for')
parser.add_argument('--nrow', type=int, default=8,
                    help='number of images in row of grid')
parser.add_argument('--one', action='store_true',
                    help='Visualize all of one GAN training')
parser.add_argument('--two', action='store_true',
                    help='Visualize all of two GAN trainings')
parser.add_argument('--final', action='store_true',
                    help='Visualize final GAN faces')
parser.add_argument('--frechet', action='store_true',
                    help='Output nrow generated image files')

args = parser.parse_args()

img_list = []


def load_new_weights(net, i, path):
    weights = ("{}/netG_epoch_{}.pth".format(path, i))
    try:
        net.load_state_dict(torch.load(weights, map_location='cpu'))
    except:
        net = torch.nn.parallel.DataParallel(netG)
        net.load_state_dict(torch.load(weights, map_location='cpu'))
    return net


def next_frame(fixed_noise, net, i=0, path=None):
    if path is not None:
        load_new_weights(net, i, path)

    with torch.no_grad():
        return net(fixed_noise).detach().cpu()


def animate(prefix):
    # Animate fixed noise
    fig = plt.figure(figsize=(args.nrow, args.nrow))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)]
           for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=20000,
                                    blit=True)
    ani.save('{}/{}Fixed_noise.gif'.format(args.path, prefix), dpi=80,
             writer='imagemagick')
    ani.save('{}/{}Fixed_noise.mp4'.format(args.path, prefix))


# # # # #  main
if args.one:
    netG = _netG(args.nz, args.ngf, args.nc).apply(weights_init)
    fixed_noise = torch.randn(args.nrow*args.nrow, args.nz, 1, 1)
    fake = next_frame(fixed_noise, netG)
    img_list.append(make_grid(fake, nrow=args.nrow, padding=2, normalize=True))

    for i in range(args.niter):
        print('.', end="", flush=True)
        fake = next_frame(fixed_noise, netG, i, args.path)
        img_list.append(
            make_grid(fake, nrow=args.nrow, padding=2, normalize=True))

    animate("")

if args.two:
    netG = _netG(args.nz, args.ngf, args.nc).apply(weights_init)
    snetG = _netG(args.nz, args.ngf, args.nc).apply(weights_init)
    fixed_noise = torch.randn(args.nrow, args.nz, 1, 1)
    fake = next_frame(fixed_noise, netG)
    sfake = next_frame(fixed_noise, snetG)
    frame = torch.cat((fake, sfake))
    img_list.append(make_grid(frame, padding=2, normalize=True))

    for i in range(args.niter):
        print('.', end="", flush=True)
        fake = next_frame(fixed_noise, netG, i, args.path)
        sfake = next_frame(fixed_noise, snetG, i, args.spath)
        frame = torch.cat((fake, sfake))
        img_list.append(make_grid(frame, padding=2, normalize=True))

    animate("comp_")

if args.final:
    netG = _netG(args.nz, args.ngf, args.nc).apply(weights_init)
    fixed_noise = torch.randn(args.nrow*args.nrow, args.nz, 1, 1)
    fake = next_frame(fixed_noise, netG, args.niter - 1, args.path)

    # Plot the fake images from the last epoch
    imgs = make_grid(fake, nrow=args.nrow, padding=2, normalize=True,
                     scale_each=True)
    plt.axis("off")
    plt.title("Generated Images")
    plt.imsave('{}/Generated_images.png'.format(args.path),
               np.transpose(imgs, (1, 2, 0)))

if args.frechet:
    netG = _netG(args.nz, args.ngf, args.nc).apply(weights_init)
    netG = load_new_weights(netG, args.niter - 1, args.path)
    fixed_noise = torch.randn(args.nrow, args.nz, 1, 1)
    fakes = next_frame(fixed_noise, netG)
    outpath = os.path.join(args.path, "generated")
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    for i, img in enumerate(fakes):
        save_image(img, "{}/fake_{}.png".format(outpath, i), nrow=1, padding=0,
                   normalize=True, scale_each=True, pad_value=0)
