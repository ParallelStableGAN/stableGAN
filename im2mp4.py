import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./outf')
args = parser.parse_args()

# Animate fixed noise
img_list = torch.load('{}/img_list.pth'.format(args.path))
fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True, normalize=True)]
       for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=100,
                                repeat_delay=1000, blit=True)
ani.save('{}/Fixed_noise.gif'.format(args.path), dpi=80, writer='imagemagick')
ani.save('{}/Fixed_noise.mp4'.format(args.path))
