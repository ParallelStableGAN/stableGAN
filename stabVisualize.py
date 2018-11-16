import matplotlib.pyplot as plt
from mogdata import generate_data_SingleBatch

# TODO: Needed while running on server. Change the GUI accordingly.
plt.switch_backend('agg')


def finalViz(opt, np_samples=[], np_samples_real=[]):

    if opt.plotRealData:
        real_cpu_temp = generate_data_SingleBatch(
            num_mode=opt.modes, radius=opt.radius, center=(0, 0),
            sigma=opt.sigma, batchSize=opt.batchSize*opt.n_batches_viz)
        tmp_cpu = real_cpu_temp.numpy()
        np_samples_real.append(tmp_cpu)

        # fig = plt.figure(figsize=(5, 5))
        if opt.markerSize:
            plt.scatter(tmp_cpu[:, 0], tmp_cpu[:, 1], c='g', edgecolor='none',
                        s=opt.markerSize)    # green is ground truth
        else:
            plt.scatter(tmp_cpu[:, 0], tmp_cpu[:, 1], c='g',
                        edgecolor='none')    # green is ground truth

        plt.axis('off')
        plt.savefig('%s/MoG_Real.pdf' % (opt.outf))
        plt.close()


# #     Final KDE plot for paper. It also plots log likelihood
#     xmax = 1.3
#     nLevels = 20
#     np_samples_ = np_samples[::1]
#     cols = len(np_samples_)
#     bg_color  = sns.color_palette('Greens', n_colors=256)[0]
#     plt.figure(figsize=(2*cols, 2))
#     for i, samps in enumerate(np_samples_):
#         if i == 0:
#             ax = plt.subplot(1,cols,1)
#         else:
#             plt.subplot(1,cols,i+1, sharex=ax, sharey=ax)
#         ax2 = sns.kdeplot(samps[:, 0], samps[:, 1], shade=True,
#                           cmap='Greens', n_levels=nLevels,
#                           clip=[[-xmax,xmax]]*2)
#         ax2.set_facecolor(bg_color)
#         plt.xticks([]); plt.yticks([])
#         plt.title('step %d'%(i*opt.viz_every))

#     plt.gcf().tight_layout()
#     plt.savefig('{0}/all.png'.format(opt.outf))

#     if opt.plotLoss:
#         plt.figure()
#         fs = np.array(fs)
#         plt.plot(fs)
#         plt.legend(('Discriminator loss', 'Generator loss'))
#         plt.savefig('{0}/losses.pdf'.format(opt.outf))

#     plt.close('all')


def iterViz(opt, i, netG, fixed_noise):
    tmp_cpu = ((netG(fixed_noise)).data).cpu().numpy()
    # np_samples.append(tmp_cpu)

    fig = plt.figure(figsize=(5, 5))
    if opt.markerSize:
        plt.scatter(tmp_cpu[:, 0], tmp_cpu[:, 1], c='g', edgecolor='none',
                    s=opt.markerSize)
    else:
        plt.scatter(tmp_cpu[:, 0], tmp_cpu[:, 1], c='g', edgecolor='none')

    plt.axis('off')
    plt.savefig('%s/MoG_Fake_withP_%03d.pdf' % (opt.outf, i))
    plt.close()
