#!/bin/bash

#SBATCH -t 08:00:00
#SBATCH --ntasks 80
#SBATCH --exclusive
#####SBATCH -p debug

echo "Train with Both prediction for 8 hours with ${SLURM_NPROCS} tasks, Distributed Data Parallel syncing, small step size, MPI, no workers."
#echo "test MPI"

. ~/.profile
module load nccl/2.3.7-1 \
  openmpi/gnu/4.9.3/1.8.6 \
  cuda/9.1.85 \
  cuDNN/7.2.1 \
  gloo/gnu/4.9.3/openmpi/1.8.6/cuda/9.1.85/20180727

export LD_LIBRARY_PATH="${HOME}/.local/lib/python3.7/site-packages/torch/lib:${LD_LIBRARY_PATH}"

COMMAND="${HOME}/code/stableGAN/main.py --verbose --distributed --dist_backend=mpi \
  --outf ${HOME}/code/stableGAN/out_${SLURM_JOB_ID}_celeba_cpu_${SLURM_NPROCS} \
  --dataset celebA --dataroot /lustre/cmsc714-1o01/data/celeba_align_resized \
  --batchSize 128 --niter 42 --lr 0.0002 --beta 0.5 --manualSeed 5206 --nc 3 \
  --num_workers=0 --pred\
  --viz_every 5 --sync_every 1"

echo $COMMAND

MASTER=`/bin/hostname -s`
MPORT=1234

echo $(which python)

mpirun -n $SLURM_NPROCS python $COMMAND --dist_init="tcp://${MASTER}:${MPORT}"
