#!/bin/bash

#SBATCH -t 00:02:00
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --exclusive
####SBATCH -p debug

#Number of processes per node to launch (20 for CPU nodes, 2 for GPU nodes)
NPROC_PER_NODE=2

#printenv

. ~/.profile
module load pytorch
COMMAND="main.py --distributed --dist_backend=gloo --verbose --outf out_celeba_gpu_${SLURM_JOB_ID}_test --dataset celebA \
--dataroot /lustre/cmsc714-1o01/data/celeba_align_resized --batchSize 64 --niter 1 --lr 0.0002 --beta1 0.9 \
--manualSeed 5206 --gpred --nc 3 --n_batches_viz 64 --viz_every 128 --cuda --ngpu 1"

MASTER=`/bin/hostname -s`
NODES=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER`
HOSTLIST="$MASTER $NODES"
MPORT=1234 # `ss -tan | awk '{print $4}' | cut -d':' -f2 | grep "[2-9][0-9]\{3,3\}" | grep -v "[0-9]\{5,5\}" | sort | uniq | shuf | head -1`


RANK=0

for node in $HOSTLIST; do
  echo "$node $RANK"
  ssh -q $node \
    module load pytorch; \
    pytorch-python3 -m torch.distributed.launch \
      --nproc_per_node=$NPROC_PER_NODE \
      --nnodes=$SLURM_JOB_NUM_NODES \
      --node_rank=$RANK \
      --master_addr="${MASTER}" \
      --master_port=$MPORT \
      $COMMAND &

  RANK=$((RANK+1))
done
wait

