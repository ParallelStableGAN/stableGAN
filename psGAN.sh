#!/bin/bash

#SBATCH -t 00:03:00
#SBATCH --ntasks 4
#SBATCH -p debug
#####SBATCH --exclusive

. ~/.profile
module load pytorch gloo/gnu/4.9.3/openmpi/1.8.6/cuda/9.1.85/20180727

COMMAND="${HOME}/code/stableGAN/main.py --verbose --distributed --dist_backend=tcp \
  --outf ${HOME}/code/stableGAN/out_${SLURM_JOB_ID}_celeba_cpu_${SLURM_NPROCS}_test \
  --dataset celebA --dataroot /lustre/cmsc714-1o01/data/celeba_align_resized \
  --batchSize 128 --niter 1 --lr 0.001 --beta 0.5 --manualSeed 5206 --gpred --nc 3 \
  --viz_every 8 --sync_every 2"

MASTER=`/bin/hostname -s`
MPORT=1234

mpirun -n $SLURM_NPROCS pytorch-python3 $COMMAND --dist_init="tcp://${MASTER}:${MPORT}"
