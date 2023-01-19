from __future__ import print_function

import argparse
import csv
import os, logging

import numpy as np
import torch
from torch.autograd import Variable, grad
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from models import loadmodel
from utils import progress_bar, set_logging_defaults
from mydataset import OCTData
import pandas as pd


parser = argparse.ArgumentParser(description='Noise Regularized Self Distillation')
parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--epoch', default=30, type=int, help='total epochs to run')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--ngpu', default=2, type=int, help='number of gpu')
parser.add_argument('--sgpu', default=0, type=int, help='gpu index (start)')
parser.add_argument('--saveroot', default='./results', type=str, help='save directory')
parser.add_argument('--temp', default=4.0, type=float, help='temperature scaling')
parser.add_argument('--lamda', default=1.0, type=float, help='srsd loss weight ratio')
parser.add_argument('--srsd', default=False, type=bool, help='do pair wise training (noise regularized self distillation)')
parser.add_argument('--addnoise', default=False, type=bool, help='do noisy training')
parser.add_argument('--model', default='ShuffleNetV2',  type=str, help='which model : ShuffleNetV2, ResNet18, MobileNetV2')
parser.add_argument('--dataset', default='UCSD', type=str, help='do noisy training')
parser.add_argument('--datapath',default='./UCSD Data/AuthorFold/Data/', type = str, help = 'directory of data')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

best_val = 0  # best validation accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

cudnn.benchmark = True

# Data
print('==> Preparing The Dataloaders ... !')

# make the data iterator for training data
train_data = OCTData('./UCSD Data/AuthorFold/F1train.csv', args.datapath, srsd = args.srsd, addnoise = args.addnoise)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle = True, num_workers=2)

# make the data iterator for validation data
val_data = OCTData('./UCSD Data/AuthorFold/F1val.csv', args.datapath, srsd = False, addnoise = False)
valloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, num_workers=2)

print('Number of Mini Batches In Training  : ' ,len(trainloader))
print('Number of Mini Batches In Validation: ' ,len(valloader))

# Model
print('==> Building model .. !')

net = loadmodel(args.model)

# print(net)

if use_cuda:
    torch.cuda.set_device(args.sgpu)
    net.cuda()
    print(torch.cuda.device_count())
    print('Using CUDA..')

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.sgpu, args.sgpu + args.ngpu)))

optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.decay)

logdir = os.path.join(args.saveroot, args.dataset, args.model, args.name)
set_logging_defaults(logdir, args)
logger = logging.getLogger('main')
logname = os.path.join(logdir, 'log.csv')


# Resume
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(os.path.join(logdir, 'ckpt.t7'))
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)

criterion = nn.CrossEntropyLoss()


class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss

kdloss = KDLoss(args.temp)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_srsd_loss = 0
    for batch_idx, data in enumerate(trainloader):
        if use_cuda:
            if not args.srsd:
                inputs_x, targets = data[0].cuda(), data[1].cuda()
                batch_size = inputs_x.size(0)
            else:
                inputs_x, inputs_y, targets = data[0].cuda(), data[1].cuda(), data[2].cuda()
                batch_size = inputs_x.size(0)
        else:
            if not args.srsd:
                inputs_x, targets = data[0], data[1]
            else:
                inputs_x, inputs_y, targets = data[0], data[1], data[2]    

        
        if not args.srsd:
            outputs = net(inputs_x)
            #print(outputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).sum().float().cpu()
            
        else:
            outputs = net(inputs_x) 
            loss = criterion(outputs, targets)
            outputs_srsd = net(inputs_y) 
            loss += criterion(outputs_srsd, targets)
            train_loss += loss.item()   
            srsd_loss = kdloss(outputs_srsd, outputs.detach()) + kdloss(outputs, outputs_srsd.detach()) # KL loss
            loss +=  args.lamda * srsd_loss
            train_srsd_loss += srsd_loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).sum().float().cpu()

        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()
        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Acc: %.3f  | (%d/%d) | srsd: %.3f'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, train_srsd_loss/(batch_idx+1)))

    logger = logging.getLogger('train')
    logger.info('[Epoch {}] [Loss {:.3f}] [srsd {:.3f}] [Acc {:.3f}]'.format(
        epoch,
        train_loss/(batch_idx+1),
        train_srsd_loss/(batch_idx+1),
        100.*correct/total))

    return train_loss/batch_idx, 100.*correct/total, train_srsd_loss/batch_idx

def val(epoch):
    global best_val
    net.eval()
    val_loss = 0.0
    correct = 0.0
    total = 0.0

    # Define a data loader for evaluating
    loader = valloader

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if use_cuda:
                inputs_x, targets = data[0].cuda(), data[1].cuda()
            else:
                inputs_x, targets = data[0], data[1]
            
            outputs = net(inputs_x)
            loss = torch.mean(criterion(outputs, targets))

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().float()

            progress_bar(batch_idx, len(loader),
                         'Loss: %.3f | Acc: %.3f | (%d/%d) '
                         % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    logger = logging.getLogger('val')
    logger.info('[Epoch {}] [Loss {:.3f}] [Acc {:.3f}]'.format(
        epoch,
        val_loss/(batch_idx+1),
        acc))

    if acc > best_val:
        best_val = acc
        checkpoint(acc, epoch)

    return (val_loss/(batch_idx+1), acc)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    torch.save(state, os.path.join(logdir, 'ckpt.t7'))


def adjust_learning_rate(optimizer, epoch):
    #decrease the learning rate at 100 and 150 epoch
    lr = args.lr
    if epoch >= 0.5 * args.epoch:
        lr /= 10
    if epoch >= 0.75 * args.epoch:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Logs
for epoch in range(start_epoch, args.epoch):
    train_loss, train_acc, train_srsd_loss = train(epoch)
    val_loss, val_acc = val(epoch)
    adjust_learning_rate(optimizer, epoch)

print("Best Accuracy : {}".format(best_val))
logger = logging.getLogger('best')
logger.info('[Acc {:.3f}]'.format(best_val))

