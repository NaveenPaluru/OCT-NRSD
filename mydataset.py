#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 17:54:31 2020

@author: cds
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from skimage import io
from skimage.transform import resize
import numpy as np
import torchvision.transforms as transforms
from skimage.color import rgb2gray
import numpy as np
import random

class OCTData(Dataset):
    
    def __init__(self, csv_file, root_dir, srsd, addnoise, transform=transforms.ToTensor()):
        self.names = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.srsd = srsd
        self.addnoise = addnoise
        self.levels = np.array([0.7, 1.0, 1.3])
       
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.names['Name'][idx])
        #print(img_name)
        image = io.imread(img_name)       
                
        if len(image.shape)==3:
            image = rgb2gray(image) * 255 # (rgb2gray scales in between [0,1]) # there are only 2 such images
        
        x = image.shape
        
        if self.srsd:
            im = np.array(image, dtype=np.float32)
            vr = self.levels[random.randint(0,2)]           
            a  = -1 * np.sqrt(3*vr)
            b  = -1 * a
            ns = a + (b - a) * np.random.rand(x[0], x[1])
            y  = im + im * ns
            mn = y.min()
            mx = y.max()                
            I = ((y - mn)/(mx-mn)) * 255
            I = I.astype(np.uint8)
            image_noise = resize(I,(224,224))           
            image_origin= resize(image,(224,224))
            label = self.names['Label'][idx]
            image_noise = self.transform(image_noise)
            image_noise = image_noise.float()
            image_origin = self.transform(image_origin)
            image_origin = image_origin.float()
            label = torch.tensor(label)
            return image_origin, image_noise, label
        
        elif self.addnoise:
            flag = np.random.rand(1)
            if flag>0.5:
                im = np.array(image, dtype=np.float32)
                vr = self.levels[random.randint(0,2)]           
                a  = -1 * np.sqrt(3*vr)
                b  = -1 * a
                ns = a + (b - a) * np.random.rand(x[0], x[1])
                y  = im + im * ns
                mn = y.min()
                mx = y.max()                
                I = ((y - mn)/(mx-mn)) * 255
                I = I.astype(np.uint8)
                image_noise = resize(I,(224,224))           
                image_noise = self.transform(image_noise)
                image_noise = image_noise.float()
                label = self.names['Label'][idx]
                label = torch.tensor(label)
                return image_noise, label
            else:
                image = resize(image,(224,224))
                image = self.transform(image)
                image = image.float()
                label = self.names['Label'][idx]
                label = torch.tensor(label)
                return image, label            
        else:
            image = resize(image,(224,224))
            image = self.transform(image)
            image = image.float()
            label = self.names['Label'][idx]
            label = torch.tensor(label)
            return image, label     
    

            
