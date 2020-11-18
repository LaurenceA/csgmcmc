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
parser.add_argument('--alpha', type=float, default=0.,
                    help='1: SGLD; <1: SGHMC')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--periods', type=int, default=30,  help='cycle length')
parser.add_argument('--S', type=int, default=4, nargs='?')
parser.add_argument('--inv_temp', type=float, default=1, nargs='?')
parser.add_argument('--train_obj', default='raw', nargs='?', type=str, choices=['true', 'raw'])
parser.add_argument('--lr', default=0.03, type=float, nargs='?')


args = parser.parse_args()
torch.manual_seed(args.seed)

temp = 1/args.inv_temp
print(temp)

def logPw(net):
    total = 0.
    for mod in net.modules():
        if isinstance(mod, nn.Linear):
            total += -(mod.weight**2).sum() * mod.in_features / 4
    return total

num_nets = 40
num_train = 10000
num_test = 10000
num_data = num_train + num_test
in_features = 5
hiddens = 30
num_classes = 2
X = torch.randn(num_nets, num_data, in_features).cuda()
X_train = X[:, :num_train]
X_test  = X[:, num_train:]

class Scale(nn.Module):
    def forward(self, x):
        return 5*x

class Linear(nn.Module):
    def __init__(self, in_features, out_features, num_nets):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.randn(num_nets, in_features, out_features) * math.sqrt(2/in_features))

    def forward(self, x):
        return x @ self.weights

gen_model = nn.Sequential(
    Linear(in_features, hiddens, num_nets),
    nn.ReLU(),
    #Linear(hiddens, hiddens, num_nets),
    #nn.ReLU(),
    nn.Linear(hiddens, num_classes, num_nets),
    Scale()
).cuda()

with torch.no_grad():
    gen_logits = gen_model(X)
    gen_Py = ConsensusLabelled(gen_logits, S=args.S).dist()
    Y = gen_Py.sample()
Y_train = Y[:, :num_train]
Y_test  = Y[:, :num_test]

net = nn.Sequential(
    nn.Linear(in_features, hiddens, num_nets),
    nn.ReLU(),
    #nn.Linear(hiddens, hiddens, num_nets),
    #nn.ReLU(),
    nn.Linear(hiddens, num_classes, num_nets),
    Scale()
).cuda()




def update_params(lr):
    for p in net.parameters():
        if not hasattr(p,'buf'):
            p.buf = torch.zeros(p.size()).cuda()
        d_p = p.grad.data
        buf_new = (1-args.alpha)*p.buf - lr*d_p
        eps = torch.randn(p.size()).cuda()
        buf_new += (2.0*lr*args.alpha*temp/num_train)**.5*eps
        p.data.add_(buf_new)
        p.buf = buf_new

def true_obj(net, X, Y):
    dist = ConsensusLabelled(net(X), S=args.S).dist()
    return dist.log_prob(Y)

def raw_obj(net, X, Y):
    return ConsensusLabelled(net(X), S=args.S).log_prob_raw(Y)

train_obj = {
    'true' : true_obj,
    'raw' : raw_obj,
}[args.train_obj]
    

def train():
    for _ in range(100):
        net.zero_grad()
        loss = -train_obj(net, X_train, Y_train).sum() / num_train - temp*logPw(net)/num_train
        loss.backward()
        update_params(args.lr)
        #print(loss.item())

def test():
    with torch.no_grad():
        net.eval()
        test_loss = 0
        correct = 0
        total = 0

        output = net(X_test)
        c = torch.arange(num_classes+1).cuda()[:, None, None]
        test_ll = ConsensusLabelled(output, S=args.S).dist().log_prob(c).permute(1, 2, 0)
        test_ll_raw = ConsensusLabelled(output, S=args.S).log_prob_raw(c).permute(1, 2, 0)
    return test_ll, test_ll_raw

#        test_ll = true_obj(net, X_test, Y_test).sum() / num_test
#        test_ll_raw = raw_obj(net, X_test, Y_test).sum() / num_test
#    return test_ll.item(), test_ll_raw.item()
         


#### Save 10000 x 10 output matrix for all sampled networks
test_lls = []
test_ll_raws = []
for period in range(args.periods):
    test_ll, test_ll_raw = test()
    train()
    test_lls.append(test_ll)
    test_ll_raws.append(test_ll_raw)


test_ll     = torch.stack(test_lls, 0).logsumexp(0) - math.log(len(test_lls))
test_ll_raw = torch.stack(test_ll_raws, 0).logsumexp(0) - math.log(len(test_ll_raws))

mean_test_ll_raw = test_ll_raw[torch.arange(num_nets)[:, None], torch.arange(10000), Y_test].sum(-1)/(Y_test != 0).sum(-1)
print(mean_test_ll_raw.mean())



##### Compute test-log-likelihood.
#Py = Categorical(logits=test_ll)
##[Samples, 10000]: compute log-prob for targets
#log_Py = Py.log_prob(Y_test)
#obj = log_Py.mean().item()
#print(obj)
#
#pd.DataFrame({
#  'loss': [loss],
#  'error' : [1-correct/total]
#}).to_csv(args.output_filename)
