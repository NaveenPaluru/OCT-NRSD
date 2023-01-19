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
from utils import progress_bar
from mydataset import OCTData
from sklearn.metrics import classification_report
import sys

parser = argparse.ArgumentParser(description='Speckle Regularized Self Distillation')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--ngpu', default=2, type=int, help='number of gpu')
parser.add_argument('--sgpu', default=0, type=int, help='gpu index (start)')
parser.add_argument('--saveroot', default='./results', type=str, help='save directory')
parser.add_argument('--srsd', default=False, type=bool, help='do pair wise training or not')
parser.add_argument('--addnoise', default=False, type=bool, help='do noisy training')
parser.add_argument('--model', default='ShuffleNetV2',  type=str, help='which model : ShuffleNetV2, ResNet18, MobileNetV2, SqueezeNet')
parser.add_argument('--dataset', default='UCSD', type=str, help='do noisy training')
parser.add_argument('--datapath',default='./UCSD Data/AuthorFold/Data/', type = str, help = 'directory of data')


args = parser.parse_args()
use_cuda = torch.cuda.is_available()

# Data
print('==> Preparing The Dataloaders ... !')

# make the data iterator for training data
test_data = OCTData('./UCSD Data/AuthorFold/F1test.csv', args.datapath, srsd = args.srsd, addnoise = args.addnoise)
testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2)

print('Number of Mini Batches In Testing  : ' ,len(testloader))
print('Number of OCT B -Scans In Testing  : ' , test_data.__len__())

# Model
print('==> Building model .. !')

logdir = os.path.join(args.saveroot, args.dataset, args.model, args.name)
net = loadmodel(args.model)
# print(net)

if use_cuda:
    torch.cuda.set_device(args.sgpu)
    net.cuda()
    print(torch.cuda.device_count())
    print('Using CUDA..')


if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.sgpu, args.sgpu + args.ngpu)))

checkpoint = torch.load(os.path.join(logdir, 'ckpt.t7'))
net.load_state_dict(checkpoint['net'])
net.eval()

with torch.no_grad():
	for i,data in (enumerate(testloader)):  
		print('Batch Index : ', i+1)
		sys.stdout.write("\033[F")           
		# start iterations
		images,imtruth = Variable(data[0]),Variable(data[1])
		# ckeck if gpu is available
		if use_cuda:
		    images  = images.cuda()
		    imtruth = imtruth.cuda()

		_, pred= torch.max(net(images), dim=1)

		#print(pred.shape)  
		
		if i==0:
		    tmp = pred.cpu().detach()
		    tmpl=imtruth.cpu().detach()        
		else:
		    tmp = torch.cat((tmp ,pred.cpu().detach()),dim=0)
		    tmpl= torch.cat((tmpl,imtruth.cpu().detach()),dim=0) 

tmp  = tmp.numpy()
tmpl = tmpl.numpy()  
hh=np.reshape(tmp, (-1,1))      
gg=np.reshape(tmpl,(-1,1)) 
print(classification_report(gg,hh))        