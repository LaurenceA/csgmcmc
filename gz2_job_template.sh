#!/bin/bash

#PBS -N gz2_train_{{S}}
#PBS -o gz2_test_{{S}}_out
#PBS -l select=1:ncpus=1:ngpus=1:mem=8G
#PBS -l walltime=24:00:00

module load lang/cuda
module load lang/python/anaconda/pytorch

cd "${PBS_O_WORKDIR}"
python3 gz2_train.py data/ results/gz2_{{S}} --S {{S}} --cycle 150 --noise_epochs 120 --sample_epochs 130 --M 8
