"""Train CIFAR10 with PyTorch."""
from __future__ import print_function
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd

from experiments.models import *

def logPw(net):
    total = 0.
    for mod in net.modules():
        if isinstance(mod, nn.Linear):
            total += -(mod.weight**2).sum() * mod.in_features / 4
        if isinstance(mod, nn.Conv2d):
            total += -(mod.weight**2).sum() * (mod.in_channels*mod.kernel_size[0]*mod.kernel_size[1]) / 4
    return total

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('output_filename', type=str)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--S', type=int)
parser.add_argument('--epochs', type=int, default=250)
args = parser.parse_args()

device = 'cuda'
torch.manual_seed(args.seed)

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

dataset = torchvision.datasets.CIFAR10
trainset = dataset(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = dataset(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

#net = AlexNet(num_classes=10).to(device=device)
net = ResNet18_nobatchnorm(num_classes=10).to(device=device)

opt = optim.SGD(net.parameters(), args.lr, momentum=args.momentum)
#opt = optim.Adam(net.parameters(), lr=1E-3)

def train(net, epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)

        opt.zero_grad()
        loss = criterion(outputs, targets) - logPw(net) / (len(trainset) * args.S)
        loss.backward()

        opt.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.shape[0]
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print('train acc %.3f' % accuracy)

    return accuracy, train_loss/len(train_loader)


def test(net):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.shape[0]
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(' test acc %.3f' % accuracy)

    return accuracy, test_loss/len(test_loader)


torch.manual_seed(args.seed)
fname = args.output_filename


criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[150, 200], gamma=0.1)

train_acc = []
test_acc = []
train_loss = []
test_loss = []
epochs = []


def _epoch(epoch):
    start = timer()
    epochs.append(epoch)

    acc, loss = train(net, epoch)
    train_acc.append(acc)
    train_loss.append(loss)

    acc, loss  = test(net)
    test_acc.append(acc)
    test_loss.append(loss)

    scheduler.step()

    print(timer()-start)

    if not os.path.isdir(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))
    pd.DataFrame({
        'epoch':      epochs,
        'train_acc':  train_acc, 
        'train_loss': train_loss,
        'test_acc':   test_acc,
        'test_loss':  test_loss,
    }).to_csv(fname, index=False)

for epoch in range(args.epochs):
    _epoch(epoch)
