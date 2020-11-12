import argparse
import random
import numpy as np
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
import torch.distributions as td
from torch.distributions import Categorical

import torchvision.transforms as transforms

import gz2
from experiments.models import *


def parse_args():
    parser = argparse.ArgumentParser(description='cSG-MCMC GZ2 Training')
    parser.add_argument('data_dir', type=str, nargs='?', default='data/')
    parser.add_argument('output_filename', type=str, nargs='?', default='test')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--alpha', type=float, default=0.9,
                        help='1: SGLD; <1: SGHMC')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--S', type=int, default=1, help='inverse temperature')
    parser.add_argument('--cycle', type=int, default=50, help='cycle length')
    parser.add_argument('--M', type=int, default=4, help='number of cycles')
    parser.add_argument('--noise_epochs', type=int, default=45, help='epoch in cycle after which we add noise')
    parser.add_argument('--sample_epochs', type=int, default=47, help='epoch in cycle after which we save samples')
    parser.add_argument('--PCIFAR100', type=float, nargs='?')
    parser.add_argument('--Pnoise', type=float, nargs='?')
    parser.add_argument('--curated', action='store_true')

    return parser.parse_args()


def zeros(size, use_cuda):
    x = torch.zeros(size)
    if use_cuda:
        x = x.cuda()
    return x


def randn(size, use_cuda):
    x = torch.randn(size)
    if use_cuda:
        x = x.cuda()
    return x


def update_params(args, use_cuda, net, epoch, lr, weight_decay, lr_factor, temp, data_size):
    for p in net.parameters():
        if not hasattr(p, 'buf'):
            p.buf = zeros(p.size(), use_cuda)
        d_p = p.grad.data
        d_p.add_(p.data, alpha=weight_decay * lr_factor)  # temp*100/datasize)
        buf_new = (1 - args.alpha) * p.buf - lr * d_p
        if (epoch % args.cycle) + 1 > args.noise_epochs:
            eps = randn(p.size(), use_cuda)
            buf_new += (2.0 * lr * args.alpha * temp / data_size) ** .5 * eps
        p.data.add_(buf_new)
        p.buf = buf_new


def adjust_learning_rate(args, epoch, T, lr_0, lr_factor, batch_idx, num_batch):
    rcounter = epoch * num_batch + batch_idx
    assert isinstance(rcounter, int)
    cos_inner = np.pi * (rcounter % (T // args.M))
    cos_inner /= T // args.M
    cos_out = np.cos(cos_inner) + 1
    lr = 0.5 * cos_out * lr_0
    lr = lr / lr_factor
    return lr


def train(args, use_cuda, net, trainloader, epoch, num_batch, T, lr_0, lr_factor, weight_decay, temp, data_size,
          train_likelihood=td.Multinomial):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        net.zero_grad()
        lr = adjust_learning_rate(args, epoch, T, lr_0, lr_factor, batch_idx, num_batch)
        outputs = net(inputs)
        # loss = criterion(outputs, targets)
        loss = -train_likelihood(logits=outputs).log_prob(targets).mean()
        loss.backward()
        update_params(args, use_cuda, net, epoch, lr, weight_decay, lr_factor, temp, data_size)

        train_loss += loss.data.item()
        predicted = torch.argmax(outputs, -1)
        targets = torch.argmax(targets, -1)
        total += targets.size(0)
        correct += predicted.eq(targets).cpu().sum()
        if batch_idx % 1 == 0:
            # os.system('nvidia-smi')
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(args, use_cuda, net, testloader, epoch):
    with torch.no_grad():
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        outputs = []

        for inputs, targets in testloader:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            output = net(inputs)
            # loss = F.cross_entropy(output, targets)
            loss = -td.Multinomial(logits=output).log_prob(targets).mean()
            targets = torch.argmax(targets, -1)

            test_loss += loss.data.item()
            predicted = torch.argmax(output.data, -1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            outputs.append(output.detach().cpu())

        print(outputs[0].shape)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss / len(testloader), correct, total,
            100. * correct / total))

    return torch.cat(outputs, 0)


def prepare_data(args, transform_train, transform_test):
    # sample_size = 20000
    # train_prop = 0.5
    # sample_size = 15000
    # train_prop = 1.0 / 3.0
    sample_size = 12500
    train_prop = 1.0 / 5.0
    if args.curated:
        # corresponds to the top 20014 data points in terms of consensus
        # consensus_quantile = 0.918
        # corresponds to the top 15002 data points in terms of consensus
        # consensus_quantile = 0.9384
        # corresponds to the top 12574 data points in terms of consensus
        consensus_quantile = 0.949153
    else:
        consensus_quantile = 0.0

    return gz2.load_consensus_data(args.data_dir, consensus_quantile, sample_size, train_prop,
                                   train_transform=transform_train, test_transform=transform_test)


def main():
    args = parse_args()

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    temp = 1 / args.S

    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    num_classes, trainset, testset = prepare_data(args, transform_train, transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    print(f"Num Classes: {num_classes}")
    print(f"Train size: {len(trainset)}")
    print(f"Test size: {len(testset)}")
    print()

    # Model
    print('==> Building model..')
    net = ResNet18(num_classes=num_classes)
    if use_cuda:
        net.cuda()
        cudnn.benchmark = True
        # cudnn.deterministic = True

    weight_decay = 5e-4
    data_size = len(trainset)
    num_batch = data_size // args.batch_size + 1
    lr_0 = 0.5  # initial lr
    lr_factor = 50.
    T = args.M * args.cycle * num_batch  # total number of iterations
    criterion = nn.CrossEntropyLoss()
    mt = 0

    #### Save 10000 x 10 output matrix for all sampled networks
    outputs = []
    for epoch in range(args.M * args.cycle):
        train(args, use_cuda, net, trainloader, epoch, num_batch, T, lr_0, lr_factor, weight_decay, temp, data_size)
        test(args, use_cuda, net, testloader, epoch)
        if (epoch % args.cycle) + 1 > args.sample_epochs:  # save 3 models per cycle
            output = test(args, use_cuda, net, testloader, epoch)
            outputs.append(output)

    ######## Evaluate accuracy using saved matrices
    targetss = []
    for _, targets in testloader:
        targetss.append(targets)
    targets = torch.cat(targetss, 0)
    # if args.trainset == "gz1":
    targets = torch.argmax(targets, -1)

    # [Samples, 10000, 10]
    outputs = torch.stack(outputs, 0)

    #### Compute preds
    # [10000, 10]: compute probs and average over networks
    Py = F.softmax(outputs, -1).mean(0)
    pred = torch.argmax(Py, -1)
    correct = (pred == targets).sum().item()
    total = pred.shape[0]

    #### Compute test-log-likelihood.
    Py = Categorical(logits=outputs)
    # [Samples, 10000]: compute log-prob for targets
    log_Py = Py.log_prob(targets)
    # [10000]: average over networks
    log_Py = torch.logsumexp(log_Py, 0) - math.log(log_Py.shape[0])
    loss = log_Py.mean().item()

    print(correct)
    print(total)
    print(loss)

    pd.DataFrame({
        'loss': [loss],
        'error': [1 - correct / total]
    }).to_csv(args.output_filename, index=False)


if __name__ == '__main__':
    main()
