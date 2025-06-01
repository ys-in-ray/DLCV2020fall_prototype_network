import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd

from PIL import Image
from utils import *


# fix random seeds for reproducibility
def set_seed():
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(SEED)
    np.random.seed(SEED)

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
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

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)

class RandomGeneratorSampler(Sampler):
    def __init__(self, N_way,N_query,N_shot,N_total_class = 16,N_each_class = 600,replacement = False,N_samples=None):
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


def find_close(p,q,d=euclidean_dist):

    dist = d(q,p)
    # print(dist.shape)
    # dist = torch.sum(dist,axis=2)
    # dist = torch.sum(dist,axis=2)

    close = torch.argmax(dist,axis=1)
    # print(close)
    # print(close.shape)
    return close


def predict(args, model, data_loader):
    prediction_results = []
    with torch.no_grad():
        # each batch represent one episode (support data + query data)
        for i, (data, target) in enumerate(data_loader):
            data = data.to(device)
            # split data into support and query data
            support_input = data[:args.N_way * args.N_shot,:,:,:] 
            query_input   = data[args.N_way * args.N_shot:,:,:,:]
            # print(query_input.shape)
            # print(target)
            # create the relative label (0 ~ N_way-1) for query data
            label_encoder = {target[j * args.N_shot] : j for j in range(args.N_way)}
            # print(label_encoder)
            query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[args.N_way * args.N_shot:]])
            query_label = query_label.to(device)
            # print(label_encoder)
            # print(query_label)

            # TODO: extract the feature of support and query data
            support_feature = model(support_input)
            query_feature = model(query_input)
            support_feature = support_feature.view(args.N_shot,args.N_way,-1)
            query_feature = query_feature.view(args.N_query*args.N_way,-1)
            # print(support_feature.shape)
            # print(query_feature.shape)

            # TODO: calculate the prototype for each class according to its support data
            prototype = torch.mean(support_feature,axis=0)
            # TODO: classify the query data depending on the its distense with each prototype
            close = find_close(prototype,query_feature)
            # print(close)
            # pd = query_label
            pd = close
            prediction_results.extend(pd.tolist())

    prediction_results = np.reshape(np.array(prediction_results),(-1,args.N_way*args.N_query))
    return prediction_results

def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N-way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N-shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N-query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--test_csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str, help="Testing images directory")
    parser.add_argument('--testcase_csv', type=str, help="Test case csv")
    parser.add_argument('--output_csv', type=str, help="Output filename")

    return parser.parse_args()

if __name__=='__main__':
    set_seed()
    args = parse_args()

    test_dataset = MiniDataset(args.test_csv, args.test_data_dir)

    test_loader = DataLoader(
        test_dataset, batch_size=args.N_way * (args.N_query + args.N_shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.testcase_csv))

    # TODO: load your model
    # load_version = 'p1_model/p1_try1_180'
    load_version = args.load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Convnet().to(device)
    model.load_state_dict(torch.load(load_version+'.pth'))

    prediction_results = predict(args, model, test_loader)

    # TODO: output your prediction to csv
    col = ['query'+str(i) for i in range(args.N_way*args.N_query)]
    df = pd.DataFrame(prediction_results,index = range(len(prediction_results)), columns = col)
    df.index.name = 'episode_id'
    df.to_csv(args.output_csv,index = True)
