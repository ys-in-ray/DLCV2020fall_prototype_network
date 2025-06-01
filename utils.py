#UTILS.PY

import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import os
import sys
import argparse
import csv
import random
import numpy as np
import pandas as pd
from PIL import Image

filenameToPILImage = lambda x: Image.open(x)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class Convnet(nn.Module):
    def __init__(self,in_channels=3, hid_channels=64, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, out_channels),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(False),
        nn.MaxPool2d(2)
    )

class Hallucinate(nn.Module):
    def __init__(self,in_channels=1600,noise_dim=1600):
        super().__init__()
        self.in_channels = in_channels
        self.noise_dim = noise_dim
        self.expand_layer = nn.Sequential(
            nn.Linear(self.in_channels+self.noise_dim,self.in_channels),
            nn.LeakyReLU(0.1),
            nn.Linear(self.in_channels,self.in_channels),
            nn.LeakyReLU(0.1),
            nn.Linear(self.in_channels,self.in_channels),
            nn.LeakyReLU(0.1),
            nn.Linear(self.in_channels,self.in_channels),
        )

    def forward(self,x):
        # x:B * 1600
        # z:B * 320
        z = torch.rand(x.shape[0], self.noise_dim)*0.001
        z = Variable(z,requires_grad=False).cuda()
        # print(z.shape)
        # print(x.shape)
        y = torch.cat((x,z),axis=1)
        # print(y)
        big_noise = self.expand_layer(y)
        return big_noise

class Hallucinate2(nn.Module):
    def __init__(self,in_channels=1600,noise_dim=1600):
        super().__init__()
        self.in_channels = in_channels
        self.noise_dim = noise_dim
        self.expand_layer = nn.Sequential(
            nn.Linear(self.in_channels+self.noise_dim,self.in_channels),
            nn.LeakyReLU(0.1),
            nn.Linear(self.in_channels,self.in_channels),
            nn.LeakyReLU(0.1),
            nn.Linear(self.in_channels,self.in_channels),
            nn.LeakyReLU(0.1),
            nn.Linear(self.in_channels,self.in_channels),
        )

    def forward(self,x):
        # x:B * 1600
        # z:B * 320
        z = torch.rand(x.shape[0], self.noise_dim)*0.001
        z = Variable(z,requires_grad=False).cuda()
        # print(z.shape)
        # print(x.shape)
        y = torch.cat((x,z),axis=1)
        # print(y)
        big_noise = self.expand_layer(y)
        return x+big_noise



def E_d(x1,x2):
    shape_1 = x1.shape
    shape_2 = x2.shape
    #Euclidean distance
    x1=x1.view(-1,shape_1[-1])
    x2=x2.view(-1,shape_2[-1])
    pdist = nn.PairwiseDistance(p=2)
    dist = pdist(x1, x2)
    dist = dist.view(shape_1[:-1])
    return dist

def E_d2(x1,x2):
    
    shape_1 = x1.shape
    shape_2 = x2.shape
    # print(shape_1)
    #Euclidean distance
    x1=x1.view(-1,shape_1[-1])
    x2=x2.view(-1,shape_2[-1])
    criteria = nn.MSELoss(reduction='none')
    dist = criteria(x1, x2)
    dist = torch.sum(dist,axis=1)
    dist = dist.view(shape_1[:-1])
    # # print(dist.shape)
    # print(x1)
    # print(x2)
    # print(dist)
    return dist

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return -1*torch.pow(x - y, 2).sum(2)

def cosine_similarity(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    cos = nn.CosineSimilarity(dim=2, eps=1e-8)

    return cos(x,y)

class Parametric(nn.Module):
    def __init__(self):
        super().__init__()
        self.dist = nn.Sequential(
            nn.Linear(1600,128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,1),
            nn.LeakyReLU(0.1),
            # nn.Linear(128,1)
            )

    def forward(self, x):
        x = self.dist(x)
        return x

def para_dist(x,y,par_model):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    conc = (x-y).view(-1,d)
    # print(conc.shape)
    dist = par_model(conc).view(n,m)
    # print(dist.shape)
    return -1*dist