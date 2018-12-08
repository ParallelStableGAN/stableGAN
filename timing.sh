#!/bin/bash

#
#  Iterate over the number of computes nodes allocated 
#
for ((i = 1 ; i < 17 ; i=i*2)); do
  for ((j = 2; j < 21 ; j=j*2)); do
    sbatch -N ${i} psGANTiming.sh ${j}
  done
done
