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
# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(10, fill=(0,)),
            # transforms.RandomResizedCrop((84,84), scale=(0.8, 1.0)),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        # print(index)
        return image, label

    def __len__(self):
        return len(self.data_df)

class RandomGeneratorSampler(Sampler):
    def __init__(self, N_way,N_query,N_shot,N_total_class = 64,N_each_class = 600,replacement = False,N_samples=None):
        self.N_way = N_way
        self.N_query = N_query
        self.N_shot = N_shot
        self.N_total_class = N_total_class
        self.N_each_class = N_each_class
        self.replacement = replacement
        self.N_samples = N_samples
        if self.N_samples is not None and replacement is False:
            raise ValueError("With replacement=False, N_samples should not be specified, \n"+
                "since a random permute will be performed.")

    def __iter__(self):
        self.sampled_sequence = []
        if self.replacement:
            class_order = np.array([])
            for i in range(self.N_samples):
                class_order = np.append(class_order,np.random.randint(self.N_total_class, size=self.N_way))
        else:
            class_order = torch.randperm(self.N_total_class).tolist()
            # print(class_order)
            self.N_samples = math.ceil(self.N_total_class / self.N_way)

        # print(class_order)
        for j in range(self.N_samples):
            each_batch_s=[]
            each_batch_q=[]
            if j==self.N_samples-1:
                top_index = len(class_order)
            else:
                top_index = (j+1)*self.N_way

            for i in class_order[j*self.N_way:top_index]:
                add = np.random.randint(self.N_each_class, size= self.N_shot+self.N_query)
                add = add+i*self.N_each_class
                add = add.astype(int)
                batch_index_s = add[:self.N_shot]
                batch_index_q = add[self.N_shot:]
                each_batch_s.extend(batch_index_s.tolist())
                each_batch_q.extend(batch_index_q.tolist())

            self.sampled_sequence.extend(each_batch_s)
            self.sampled_sequence.extend(each_batch_q)

        # print(len(self.sampled_sequence))
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)


def loss_func(p,q,gt,N_way,N_query,d=E_d2):

    q = q.unsqueeze(1)
    p = p.unsqueeze(0)

    p2 = p[:,gt,:].unsqueeze(0).repeat(q.shape[0],1,1)
    # print(p2.shape)
    ret = torch.sum(d(q,p2)/40)

    p = p.repeat(q.shape[0],1,1)
    q = q.repeat(1,p.shape[1],1)
    # print(p2.shape)
    # print(q2.shape)
    ret += torch.sum(torch.log(torch.sum(torch.exp(-1*d(q,p)/40),axis=1)))
    return ret/(N_way*N_query)

def loss_func2(p,q,N_way,N_query,d):

    target_inds = torch.arange(0, N_way).view(N_way, 1, 1).expand(N_way, N_query, 1).long()
    target_inds = Variable(target_inds, requires_grad=False).cuda()
    dists = d(q, p)

    log_p_y = F.log_softmax(dists, dim=1).view(N_way, N_query, -1)

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

    _, y_hat = log_p_y.max(2)
    acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

    return loss_val, {
        'loss': loss_val.item(),
        'acc': acc_val.item()
    }

def train(N_way,N_query,N_shot,N_total_class,N_each_class,train_loader,optimizer,device,epoch,N_samples,distance_function):
    # N_samples = math.floor(N_each_class/N_query)
    total_loss = 0
    total_acc = 0
    for i, (data, target) in enumerate(train_loader):
        data = data.to(device)
        # split data into support and query data
        this_way = int(data.shape[0]/((N_query + N_shot)))
        support_input = data[:this_way * N_shot,:,:,:] 
        query_input   = data[this_way * N_shot:,:,:,:]

        # create the relative label (0 ~ N_way-1) for query data
        label_encoder = {target[j * N_shot] : j for j in range(this_way)}
        query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[this_way * N_shot:]])
        query_label = query_label.to(device)
        # print(target)
        # print(query_label)

        # TODO: extract the feature of support and query data
        support_feature = model(support_input)
        query_feature = model(query_input)
        support_feature = support_feature.view(N_shot,this_way,-1)
        query_feature = query_feature.view(N_query*this_way,-1)

        # TODO: calculate the prototype for each class according to its support data
        prototype = torch.mean(support_feature,axis=0)
        # print("hey")
        # print(prototype.shape)
        # print(query_feature.shape)
        # print("hey")

        # TODO: classify the query data depending on the its distense with each prototype
        optimizer.zero_grad()

        loss, dic = loss_func2(prototype,query_feature,this_way,N_query,d=distance_function)
        loss.backward()
        optimizer.step()
        total_loss+=loss
        total_acc +=dic['acc']

        percentage = i/(N_samples)*100
        sys.stdout.write('\r')
        sys.stdout.write("ep[%3d] "%(epoch))
        sys.stdout.write("[%-20s] %d%% (%d/%d)" % ('='*int(percentage/5), percentage+1,i,N_samples))

    print("acc= %.3f%%, loss= %.4f"%(total_acc/N_samples*100,total_loss.item()/N_samples))
    # percentage = 100
    # sys.stdout.write('\r')
    # sys.stdout.write("ep[%3d] "%(epoch))
    # sys.stdout.write("[%-20s] %d%% (%d/%d)" % ('='*20, percentage,N_samples,N_samples))
    # print()

def save_model(model,epoch,v):
    version = 'p1_try10v2'
    torch.save(model.state_dict(),'p1_model/'+version+'_'+str(epoch)+v+'.pth')

def init_weight(m):
    # print(type(m))
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight)

if __name__ == '__main__':
    trian_csv = '../hw4_data/train.csv'
    train_data_dir = '../hw4_data/train/'
    traincase_csv = '../hw4_data/val_testcase.csv'

    N_way = 5
    N_query = 15
    N_shot = 1
    N_episode = 50
    N_total_class = 64
    N_each_class = 600
    N_samples = 100
    # N_samples =50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 4e-3

    model = Convnet().to(device)
    model.apply(init_weight)
    

    train_dataset = MiniDataset(trian_csv, train_data_dir)

    train_loader = DataLoader(
        train_dataset, batch_size=N_way * (N_query + N_shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=RandomGeneratorSampler(N_way,N_query,N_shot,replacement=True,N_samples=N_samples))

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    lrdecay=1
    if sys.argv[1]=='ed':
        print("use:euclidean dist")
        distance_function = euclidean_dist
    elif sys.argv[1]=='cos':
        distance_function = cosine_similarity
        print("use:cosine similarity")
    else:
        print("use:(default)euclidean dist")
        distance_function = euclidean_dist

    for i in range(1,N_episode):
        
        if i%20 ==0:
            lrdecay = lrdecay/2
            print('lr = '+str(lr*lrdecay))
        for g in optimizer.param_groups:
            g['lr'] = lr*lrdecay


        train(N_way,N_query,N_shot,N_total_class,N_each_class,train_loader,optimizer,device,i,N_samples,distance_function)
        # save_model(model,i*N_samples)

        # save model
        if i *N_samples%1000==0:
            save_model(model,i*N_samples,sys.argv[1])

    save_model(model,N_episode*N_samples,sys.argv[1])

