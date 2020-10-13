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

from gz1.gz1 import GZ1

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
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--alpha', type=float, default=0.9,
                    help='1: SGLD; <1: SGHMC')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--S', type=int, default=1, help='inverse temperature')
parser.add_argument('--cycle', type=int, default=50,  help='cycle length')
parser.add_argument('--M', type=int, default=4,  help='number of cycles')
parser.add_argument('--noise_epochs', type=int, default=45,  help='epoch in cycle after which we add noise')
parser.add_argument('--sample_epochs', type=int, default=47, help='epoch in cycle after which we save samples')
parser.add_argument('--trainset', default='train', nargs='?', choices=["train", "test", "cifar10h"]) #, "gz1"])
parser.add_argument('--PCIFAR100', type=float, nargs='?')
parser.add_argument('--Pnoise', type=float, nargs='?')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
temp = 1/args.S



# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

num_classes = 10


trainset = torchvision.datasets.CIFAR10(root='data', train=(args.trainset=="train"), download=True, transform=transform_train)
lr_factor = 1.
if args.trainset=="cifar10h":
    cifar10h = np.load("cifar10h.npy").astype(np.float)
    trainset.targets = cifar10h
    lr_factor = 50.

testset = torchvision.datasets.CIFAR10(root='data', train=(args.trainset!="train"), download=True, transform=transform_test)

if args.Pnoise is not None:
    assert args.trainset == "train"

    train_noise = int(50000*args.Pnoise)
    test_noise  = int(10000*args.Pnoise)

    trainset.targets[:train_noise] = list(np.random.randint(10, size=train_noise))
    testset.targets[:test_noise]   = list(np.random.randint(10, size=test_noise))
    

if args.PCIFAR100 is not None:
    assert args.trainset == "train"
    
    cifar100_trainset = torchvision.datasets.CIFAR100(root='data', train=True, download=True, transform=transform_train)
    cifar100_testset = torchvision.datasets.CIFAR100(root='data', train=False,transform=transform_test)

    #Assign random labels
    cifar100_trainset.targets = list(np.random.randint(10, size=len(cifar100_trainset.targets)))
    cifar100_testset.targets = list(np.random.randint(10, size=len(cifar100_testset.targets)))

    Ncifar100_train = int(50000*args.PCIFAR100)
    Ncifar100_test  = int(10000*args.PCIFAR100)

    Ncifar10_train = 50000 - Ncifar100_train
    Ncifar10_test  = 10000 - Ncifar100_test

    print(f"CIFAR100 train :{Ncifar100_train}")
    print(f"CIFAR10  train :{Ncifar10_train}")
    print(f"CIFAR100 test  :{Ncifar100_test}")
    print(f"CIFAR10  test  :{Ncifar10_test}")

    assert 50000 == Ncifar100_train + Ncifar10_train
    assert 10000 == Ncifar100_test  + Ncifar10_test

    #Subsets
    cifar100_trainset = tud.Subset(cifar100_trainset, range(Ncifar100_train))
    cifar100_testset  = tud.Subset(cifar100_testset,  range(Ncifar100_test))

    cifar10_trainset = tud.Subset(trainset, range(Ncifar10_train))
    cifar10_testset  = tud.Subset(testset,  range(Ncifar10_test))

    #Combined
    trainset = tud.ConcatDataset([cifar100_trainset, cifar10_trainset])
    testset  = tud.ConcatDataset([cifar100_testset,  cifar10_testset])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)

#elif args.trainset == 'gz1':
#    #class RandomRotate:
#    #    def __call__(self, x):
#    #        angle = random.choice([0, 90, -90, 180])
#    #        return TF.rotate(x, angle)
#    trans = transforms.Compose([
#        transforms.RandomCrop(32, padding=4),
#        #RandomRotate(),
#        transforms.RandomRotation(180),
#        transforms.RandomHorizontalFlip(),
#        transforms.ToTensor()
#    ])
#
#    num_classes = 6
#    lr_factor = 50.
#    trainset = GZ1(True, transform=trans)
#    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
#
#    testset = GZ1(False, transform=transforms.Compose([transforms.CenterCrop(50), transforms.ToTensor()]))
#    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    



# Model
print('==> Building model..')
net = ResNet18(num_classes=num_classes)
if use_cuda:
    net.cuda()
    cudnn.benchmark = True
    #cudnn.deterministic = True

def update_params(lr,epoch):
    for p in net.parameters():
        if not hasattr(p,'buf'):
            p.buf = torch.zeros(p.size()).cuda()
        d_p = p.grad.data
        d_p.add_(p.data, alpha=weight_decay*lr_factor) #temp*100/datasize)
        buf_new = (1-args.alpha)*p.buf - lr*d_p
        if (epoch%args.cycle)+1>args.noise_epochs:
            eps = torch.randn(p.size()).cuda()
            buf_new += (2.0*lr*args.alpha*temp/datasize)**.5*eps
        p.data.add_(buf_new)
        p.buf = buf_new

def adjust_learning_rate(epoch, batch_idx):
    rcounter = epoch*num_batch+batch_idx
    assert isinstance(rcounter, int)
    cos_inner = np.pi * (rcounter % (T // args.M))
    cos_inner /= T // args.M
    cos_out = np.cos(cos_inner) + 1
    lr = 0.5*cos_out*lr_0
    lr = lr / lr_factor
    return lr

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_likelihood = td.Multinomial if args.trainset in ["cifar10h","gz1"] else td.Categorical

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        net.zero_grad()
        lr = adjust_learning_rate(epoch,batch_idx)
        outputs = net(inputs)
        #loss = criterion(outputs, targets)
        loss = -train_likelihood(logits=outputs).log_prob(targets).mean()
        loss.backward()
        update_params(lr,epoch)

        train_loss += loss.data.item()
        predicted = torch.argmax(outputs, -1)
        if args.trainset in ["cifar10h", "gz1"]:
            targets = torch.argmax(targets, -1)
        total += targets.size(0)
        correct += predicted.eq(targets).cpu().sum()
        if batch_idx%100==0:
            #os.system('nvidia-smi')
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    with torch.no_grad():
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        outputs = []

        for inputs, targets in testloader:
            inputs, targets = inputs.cuda(), targets.cuda()
            output = net(inputs)
            #loss = F.cross_entropy(output, targets)
            if args.trainset == "gz1":
                loss = -td.Multinomial(logits=output).log_prob(targets).mean()
                targets = torch.argmax(targets, -1)
            else:
                loss = -td.Categorical(logits=output).log_prob(targets).mean()

            test_loss += loss.data.item()
            predicted = torch.argmax(output.data, -1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            outputs.append(output.detach().cpu())

        print(outputs[0].shape)


        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss/len(testloader), correct, total,
        100. * correct / total))

    return torch.cat(outputs, 0)

weight_decay = 5e-4
datasize = len(trainset)
num_batch = datasize//args.batch_size+1
lr_0 = 0.5 # initial lr
T = args.M*args.cycle*num_batch # total number of iterations
criterion = nn.CrossEntropyLoss()
mt = 0

#### Save 10000 x 10 output matrix for all sampled networks
outputs = []
for epoch in range(args.M*args.cycle):
    train(epoch)
    test(epoch)
    if (epoch%args.cycle)+1>args.sample_epochs: # save 3 models per cycle
        output = test(epoch)
        outputs.append(output)


######## Evaluate accuracy using saved matrices
targetss = []
for _, targets in testloader:
    targetss.append(targets)
targets = torch.cat(targetss, 0)
if args.trainset == "gz1":
    targets = torch.argmax(targets, -1)

#[Samples, 10000, 10]
outputs = torch.stack(outputs, 0)

#### Compute preds
#[10000, 10]: compute probs and average over networks
Py = F.softmax(outputs, -1).mean(0)
pred = torch.argmax(Py, -1)
correct = (pred == targets).sum().item()
total = pred.shape[0]

#### Compute test-log-likelihood.
Py = Categorical(logits=outputs)
#[Samples, 10000]: compute log-prob for targets
log_Py = Py.log_prob(targets)
#[10000]: average over networks
log_Py = torch.logsumexp(log_Py, 0) - math.log(log_Py.shape[0])
loss = log_Py.mean().item()

print(correct)
print(total)
print(loss)

pd.DataFrame({
  'loss': [loss],
  'error' : [1-correct/total]
}).to_csv(args.output_filename)
#for epoch in range(args.epochs):
#    train(epoch)
#    test(epoch)
#    if (epoch%50)+1>47: # save 3 models per cycle
#        print('save!')
#        net.cpu()
#        torch.save(net.state_dict(),args.dir + '/cifar_csghmc_%i.pt'%(mt))
#        mt += 1
#        net.cuda(device_id)
