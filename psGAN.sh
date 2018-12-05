#!/bin/bash

#SBATCH -t 00:03:00
#SBATCH --ntasks 64
#SBATCH -p debug
#####SBATCH --exclusive

. ~/.profile
module load pytorch gloo/gnu/4.9.3/openmpi/1.8.6/cuda/9.1.85/20180727

COMMAND="${HOME}/code/stableGAN/main.py --verbose --distributed --dist_backend=tcp \
  --outf out_${SLURM_JOB_ID}_celeba_cpu_80_test --dataset celebA \
  --dataroot /lustre/cmsc714-1o01/data/celeba_align_resized --batchSize 128 \
  --niter 1 --lr 0.001 --beta 0.5 --manualSeed 5206 --gpred --nc 3 \
  --n_batches_viz 64 --viz_every 128"
#

MASTER=`/bin/hostname -s`
NODES=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER`
HOSTLIST="$MASTER $NODES"
MPORT=1234 #`ss -tan | awk '{print $4}' | cut -d':' -f2 | grep "[2-9][0-9]\{3,3\}" | grep -v "[0-9]\{5,5\}" | sort | uniq | shuf | head -1`

mpirun -n $SLURM_NPROCS pytorch-python3 $COMMAND --dist_init="tcp://${MASTER}:${MPORT}"

###Launch the pytorch processes
#RANK=0
#
#for node in $HOSTLIST; do
#  for i in $(seq 1 $NPROC_PER_NODE); do
#    # echo "$node $RANK"
#    ssh -q $node \
#      module load pytorch;
#      pytorch-python3 -m torch.distributed.launch \
#        --nproc_per_node=1 \
#        --nnodes=1 \
#        --node_rank=$RANK \
#        --master_addr="${MASTER}" \
#        --master_port=$MPORT \
#        $COMMAND &
#
#    RANK=$((RANK+1))
#  done
#done
#wait

