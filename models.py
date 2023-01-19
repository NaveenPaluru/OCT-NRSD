#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 18:31:48 2022

@author: cds
"""


import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ptflops import get_model_complexity_info

def loadmodel(model):
    if model =='ShuffleNetV2':
        net = ShuffleNetV2()        
    elif model == 'MobileNetV2':
        net = MobileNetV2()
    elif model == 'ResNet18':
        net = ResNet18()
    else:
        print('Model Not Identified..!')
        net = None
    return net


def ShuffleNetV2():
    net = models.shufflenet_v2_x1_0()
    net.conv1[0]=torch.nn.Conv2d(1, 24,kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    net.fc = nn.Linear(1024,4,bias=True)   
    return net

def MobileNetV2():    
    net = models.mobilenet_v2()
    net.features[0][0]=torch.nn.Conv2d(1, 32, 
                                       kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    net.classifier[1]=nn.Linear(1280,4,bias=True)
    return net

def ResNet18():
    net = models.resnet18()
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net.fc=nn.Linear(512,4,bias=True)  
    return net

def getnumberofparams(model):
     pp=0
     for p in (model.parameters()):
         nn=1
         for s in (p.size()):
             nn = nn*s
         pp += nn
     return pp


if __name__ =="__main__":
    def tic():
        # Homemade version of matlab tic and toc functions
        import time
        global startTime_for_tictoc
        startTime_for_tictoc = time.time()

    def toc():
        import time
        if 'startTime_for_tictoc' in globals():
            print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
        else:
            print("Toc: start time not set")
    net = ShuffleNetV2()
    print(net)
    #net.cuda()
    x = torch.rand(1,1,224,224)#.cuda()
    #tic()
    y = net(x)
    print(y)
    #toc()
    print(y.shape)
    print(getnumberofparams(net))
    print((getnumberofparams(net)*32)/(8*1024*1024))
    macs, params = get_model_complexity_info(net, (1, 224, 224), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
