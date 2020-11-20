#!/bin/bash

#PBS -N gz2_train_1250_lr50_b1250_{{S}}
#PBS -o gz2_test_1250_lr50_b1250_{{S}}_out
#PBS -l select=1:ncpus=1:ngpus=1:mem=8G
#PBS -l walltime=12:00:00

module load lang/cuda
module load lang/python/anaconda/pytorch

cd "${PBS_O_WORKDIR}"
python3 gz2_train.py data/ results/gz2_1250_lr50_b1250_{{S}} --S {{S}} \
  --cycle 1200 --noise_epochs 960 --sample_epochs 1040 --M 8 \
  --lr-factor 50 --batch-size 1250
