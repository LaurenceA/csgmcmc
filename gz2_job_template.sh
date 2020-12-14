#!/bin/bash

#PBS -N gz2_train_2500_lr50_b1250_cat_{{S}}
#PBS -o gz2_test_2500_lr50_b1250_cat_{{S}}_out
#PBS -l select=1:ncpus=1:ngpus=1:mem=8G
#PBS -l walltime=12:00:00

module load lang/cuda
module load lang/python/anaconda/pytorch

cd "${PBS_O_WORKDIR}"
python3 gz2_train.py data/ results/gz2_2500_lr50_b1250_cat_{{S}} --S {{S}} \
  --cycle 600 --noise_epochs 480 --sample_epochs 520 --M 8 \
  --lr-factor 50 --batch-size 1250
