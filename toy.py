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

from consensus import ConsensusLabelled

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
parser.add_argument('--S', type=int, default=1, nargs='?')
parser.add_argument('--train_obj', default='raw', nargs='?', type=str, choices=['true', 'raw'])
parser.add_argument('--lr', default=0.01, type=float, nargs='?')


args = parser.parse_args()
torch.manual_seed(args.seed)

#temp = 1/args.inv_temp
#print(temp)

def logPw(net):
    total = 0.
    for mod in net.modules():
        if isinstance(mod, Linear):
            total += -(temp*mod.weight**2).sum((-1, -2)) * mod.in_features / 4
    return total

inv_temps = torch.tensor([128, 64, 32, 16, 8, 7, 6, 5, 4, 3, 2, 1, 1/2, 1/4, 1/8, 1/16])[:, None, None, None]
#inv_temps = torch.tensor([1.])[:, None, None, None]
temp = (1/inv_temps).cuda()
num_temps = len(temp)

num_nets = 500
num_train = 100 #100  #num_nets=40, num_train=25
num_test = 1000
num_data = num_train + num_test
in_features = 5
hiddens = 30
num_classes = 2
X = torch.randn(num_nets, num_data, in_features).cuda()
X_train = X[..., :num_train, :]
X_test  = X[..., num_train:, :]

class Scale(nn.Module):
    def forward(self, x):
        return x

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
Y_train = Y[..., :num_train]
Y_test  = Y[..., num_train:]

con_train = (Y_train != 0).sum(-1, keepdim=True)
con_test = (Y_test != 0).sum(-1, keepdim=True)

net = nn.Sequential(
    Linear(in_features, hiddens, (num_temps, num_nets)),
    nn.ReLU(),
    Linear(hiddens, num_classes, (num_temps, num_nets)),
    Scale()
).cuda()




def update_params(lr):
    for p in net.parameters():
        eps = torch.randn_like(p) * (2.0*lr*temp/num_train)**0.5
        p.data.add_(p.grad.data, alpha=lr)
        p.data.add_(eps)

    #for p in net.parameters():
    #    if not hasattr(p,'buf'):
    #        p.buf = torch.zeros(p.size()).cuda()
    #    d_p = p.grad.data
    #    buf_new = (1-args.alpha)*p.buf - lr*d_p
    #    eps = torch.randn(p.size()).cuda()
    #    buf_new += (2.0*lr*args.alpha*temp/num_train)**.5*eps
    #    p.data.add_(buf_new)
    #    p.buf = buf_new

def true_obj(net, X, Y):
    dist = ConsensusLabelled(net(X), S=args.S).dist()
    return dist.log_prob(Y)

#def raw_obj(net, X, Y):
#    return ConsensusLabelled(net(X), S=args.S).log_prob_raw(Y)

def raw_obj(output, Y, avg=True):
    _Y = Y - 1 + (Y==0).to(dtype=torch.int)
    lp = Categorical(logits=output).log_prob(_Y)
    lp = (lp*(Y!=0)).sum(-1) #/ (Y!=0).sum(-1)
    if avg:
        lp = lp / (Y!=0).sum(-1)
    return lp

def train():
    for _ in range(400):
        net.zero_grad()
        output = net(X_train)
        #temp applied in logPw
     
        ll = (raw_obj(output, Y_train, False)  + logPw(net)).sum()/num_train
        ll.backward()
        update_params(args.lr)
        #print(loss.item())

def test():
    with torch.no_grad():
        output = net(X_test)
    return output

         


outputs = []
test_lls = []
test_ll_raws = []
for period in range(10):
    test()
    train()

for period in range(args.periods):
    train()
    output = test()
    outputs.append(output.cpu())

output = torch.stack(outputs, 0)
lp = output.log_softmax(-1)
lp = lp.logsumexp(0) - math.log(lp.shape[0])

result = raw_obj(lp, Y_test.cpu())

temps = 1/inv_temps[:, 0, 0, 0]
result = result.mean(-1)
print(torch.stack([inv_temps[:, 0, 0, 0], result], 1))

pd.DataFrame({
    'lambda' : temps,
    'test_ll' : result
}).to_csv(args.output_filename)
