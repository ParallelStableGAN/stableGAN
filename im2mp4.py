import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Animate fixed noise
# torch.save(img_list, '{}/img_list.pth'.format(opt.outf))
img_list = torch.load('out_predicition/img_list.pth')
fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)]
       for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000,
                                repeat_delay=1000, blit=True)
ani.save('out_predicition/Fized_noise.mp4')
