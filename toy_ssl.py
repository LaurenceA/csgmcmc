'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributions as td
import torch.utils.data as tud

from consensus import ConsensusLabelled, ConsensusUnlabelled

from torch.distributions import Categorical

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import os
import argparse

from experiments.models import *
import numpy as np
import pandas as pd
import random

parser = argparse.ArgumentParser(description='cSG-MCMC CIFAR10 Training')
parser.add_argument('output_filename', type=str, nargs='?', default='test')
parser.add_argument('--alpha', type=float, default=0.9,
                    help='1: SGLD; <1: SGHMC')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--periods', type=int, default=50,  help='cycle length')
parser.add_argument('--S', type=int, default=3, nargs='?')
parser.add_argument('--train_obj', default='raw', nargs='?', type=str, choices=['true', 'raw'])
parser.add_argument('--lr', default=0.01, type=float, nargs='?')


args = parser.parse_args()
torch.manual_seed(args.seed)

def logPw(net):
    total = 0.
    for mod in net.modules():
        if isinstance(mod, Linear):
            total += -(mod.weight**2).sum((-1, -2)) * mod.in_features / 4
    return total

num_unlab = torch.tensor([0, 100, 200, 400, 800, 1600])[:, None, None].cuda()
max_unlab = torch.max(num_unlab)
num_datasets = len(num_unlab)
#[unlab_datasets, 1, max_unlab]
unlab_mask = (torch.arange(max_unlab).cuda() < num_unlab).to(dtype=torch.float)


num_lab = 100
num_train = num_lab + max_unlab
num_test = 1000
num_data = num_lab + max_unlab + num_test

num_nets = 500
in_features = 5
hiddens = 30
num_classes = 2

X = torch.randn(num_nets, num_data, in_features).cuda()
X_train = X[..., :(-num_test), :]
X_test  = X[..., -num_test:, :]

X_unlab = X_train[..., :max_unlab , :]
X_lab   = X_train[...,  max_unlab:, :]

class Scale(nn.Module):
    def forward(self, x):
        return 3*x

class Linear(nn.Module):
    def __init__(self, in_features, out_features, shape):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(*shape, in_features, out_features) * math.sqrt(2/in_features))

    def forward(self, x):
        return x @ self.weight

gen_model = nn.Sequential(
    Linear(in_features, hiddens, (1, num_nets)),
    nn.ReLU(),
    Linear(hiddens, num_classes, (1, num_nets)),
    Scale()
).cuda()

with torch.no_grad():
    gen_logits = gen_model(X)
    gen_Py = ConsensusLabelled(gen_logits, S=args.S).dist()
    Y = gen_Py.sample()
    Y[..., :max_unlab] = (0 < Y[..., :max_unlab])

    Y_train = Y[..., :(-num_test) ]
    Y_test  = Y[...,  (-num_test):]

    Y_unlab = Y_train[..., :max_unlab ]
    Y_lab   = Y_train[...,  max_unlab:]

net = nn.Sequential(
    Linear(in_features, hiddens, (num_datasets, num_nets)),
    nn.ReLU(),
    Linear(hiddens, num_classes, (num_datasets, num_nets)),
    Scale()
).cuda()

def update_params(lr):
    for p in net.parameters():
        eps = torch.randn_like(p) * (2.0*lr/num_train)**0.5
        p.data.add_(p.grad.data, alpha=lr)
        p.data.add_(eps)

def lab_lp(output, Y):
    dist = ConsensusLabelled(output, S=args.S).dist()
    return dist.log_prob(Y).sum(-1)

def unlab_lp(output, Y):
    dist = ConsensusUnlabelled(output, S=args.S).dist()
    return (dist.log_prob(Y) * unlab_mask).sum(-1)

def raw_obj(output, Y, avg=False):
    _Y = Y - 1 + (Y==0).to(dtype=torch.int)
    lp = Categorical(logits=output).log_prob(_Y)
    lp = (lp*(Y!=0)).sum(-1)
    return lp

def train_lab():
    for _ in range(400):
        net.zero_grad()
        output = net(X_lab)

        lp = lab_lp(output, Y_lab)
     
        ll = (lp + logPw(net)).sum()/num_train
        ll.backward()
        update_params(args.lr)

def train():
    for _ in range(400):
        net.zero_grad()
        output = net(X_train)
        #temp applied in logPw
        unlab_output = output[..., :max_unlab , :]
        lab_output   = output[...,  max_unlab:, :]

        train_lab_lp = lab_lp(lab_output, Y_lab)
        train_unlab_lp = unlab_lp(unlab_output, Y_unlab)
     
        ll = (train_lab_lp + train_unlab_lp  + logPw(net)).sum()/num_train
        ll.backward()
        update_params(args.lr)

def test():
    with torch.no_grad():
        output = net(X_test)
    return output


outputs = []
test_lls = []
test_ll_raws = []

for period in range(10):
    print(period)
    train_lab()

for period in range(10): 
    print(period)
    train()

for period in range(args.periods):
    print(period)
    train()
    output = test()
    outputs.append(output.cpu())

output = torch.stack(outputs, 0)
lp = output.log_softmax(-1)
lp = lp.logsumexp(0) - math.log(lp.shape[0])

pred = lp.argmax(-1) + 1
acc = ((pred == Y_test.cpu()).sum(-1) / (pred != 0).sum(-1)).mean(-1).numpy()

result = raw_obj(lp, Y_test.cpu())
result = result / (Y_test.cpu() != 0).sum(-1)
result = result.mean(-1)

pd.DataFrame({
    'num_unlab' : num_unlab.cpu()[:, 0, 0],
    'test_ll' : result,
    'acc' : acc,
}).to_csv(args.output_filename)
