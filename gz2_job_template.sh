#!/bin/bash

#PBS -N gz2_train_5000_{{S}}
#PBS -o gz2_test_5000_{{S}}_out
#PBS -l select=1:ncpus=1:ngpus=1:mem=8G
#PBS -l walltime=12:00:00

module load lang/cuda
module load lang/python/anaconda/pytorch

cd "${PBS_O_WORKDIR}"
python3 gz2_train.py data/ results/gz2_5000_{{S}} --S {{S}} --cycle 300 --noise_epochs 240 --sample_epochs 260 --M 8
