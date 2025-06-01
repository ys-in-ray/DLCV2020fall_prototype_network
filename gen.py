import math

import torch
import torch.nn as nn
import torch.nn.functional as F
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
import sys
from utils import *


N_way = 5
N_query = 15
N_shot = 10
N_total_class = 16
N_each_class = 600

N_samples = 600
sampled_sequence = []

class_order = np.array([])
for i in range(N_samples):
    class_order = np.append(class_order,np.random.randint(N_total_class, size=N_way))


# print(class_order)
for j in range(N_samples):
    each_batch_s=[]
    each_batch_q=[]
    if j==N_samples-1:
        top_index = len(class_order)
    else:
        top_index = (j+1)*N_way

    for i in class_order[j*N_way:top_index]:
        add = np.random.randint(N_each_class, size= N_shot+N_query)
        add = add+i*N_each_class
        add = add.astype(int)
        batch_index_s = add[:N_shot]
        batch_index_q = add[N_shot:]
        each_batch_s.extend(batch_index_s.tolist())
        each_batch_q.extend(batch_index_q.tolist())

    sampled_sequence.extend(each_batch_s)
    sampled_sequence.extend(each_batch_q)

# print(sampled_sequence)
sampled_sequence = np.reshape(np.array(sampled_sequence),(-1,N_way*(N_shot+N_query)))
print(sampled_sequence.shape)
col = []
for i in range(N_way):
    new_col = ['class'+str(i)+'_support'+str(j) for j in range(N_shot)]
    col.extend(new_col)
col2 = ['query'+str(i) for i in range(N_way*N_query)]
col.extend(col2)
print(col)
print(len(col))
df = pd.DataFrame(sampled_sequence,index = range(len(sampled_sequence)), columns = col)
df.index.name = 'episode_id'
df.to_csv('10shot.csv',index = True)