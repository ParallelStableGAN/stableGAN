#!/bin/bash

#SBATCH -t 00:10:00
#SBATCH -N 2
#SBATCH --gres=gpu:2
#SBATCH --exclusive
####SBATCH -p debug

#Number of processes per node to launch (20 for CPU nodes, 2 for GPU nodes)
NPROC_PER_NODE=2

printenv

. ~/.profile
module load pytorch gloo/gnu/4.9.3/openmpi/1.8.6/cuda/9.1.85/20180727
COMMAND="main.py --distributed --dist_backend=gloo --verbose --outf out_celeba_gpu_${SLURM_JOB_ID}_test --dataset celebA \
--dataroot /lustre/cmsc714-1o01/data/celeba_align_resized --batchSize 16 --niter 1 --lr 0.0002 --beta1 0.5 \
--manualSeed 5206 --gpred --nc 3 --n_batches_viz 64 --viz_every 128 --cuda --ngpu 1"

#MASTER=`/bin/hostname -s`
MASTER=$HOSTNAME
HOSTLIST=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER`
MPORT=29500 # `ss -tan | awk '{print $4}' | cut -d':' -f2 | grep "[2-9][0-9]\{3,3\}" | grep -v "[0-9]\{5,5\}" | sort | uniq | shuf | head -1`

RANK=0

pytorch-python3 -m torch.distributed.launch \
  --nproc_per_node=$NPROC_PER_NODE \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --node_rank=$RANK \
  --master_addr="$HOSTNAME" \
  --master_port=${MPORT} \
  $COMMAND &

RANK=1

for node in $HOSTLIST; do
  echo "$node $RANK"
  ssh -q $node \
    module load pytorch gloo/gnu/4.9.3/openmpi/1.8.6/cuda/9.1.85/20180727; \
    pytorch-python3 -m torch.distributed.launch \
      --nproc_per_node=$NPROC_PER_NODE \
      --nnodes=$SLURM_JOB_NUM_NODES \
      --node_rank=$RANK \
      --master_addr="$HOSTNAME" \
      --master_port=${MPORT} \
      $COMMAND &

  RANK=$((RANK+1))
done

wait
