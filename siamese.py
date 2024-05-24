import os
import cv2 as cv
import torch
import random
import pandas as pd
import numpy as np
import pickle

from torch import optim

from torch.nn import Module, Sequential, ReLU, Conv2d, Linear, MaxPool2d
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from numpy import linalg, transpose
from torchvision.models import VGG16_Weights

from preprocessing import Resize, ToTensor, MeanStdRecorder


class SiameseDataset(Dataset):

    def __init__(self, dataset, test=False, transform=None, load_stats=False):
        self.dataset = dataset
        self.test = test
        self.N_CHANNELS = 3
        self.transform = transform

        if self.test:
            self.query_path = '/Test/query_images/'
            self.reference_path = '/Test/reference_images/'
        else:
            self.query_path = '/Train/query_images/'
            self.reference_path = '/Train/reference_images/'

        self.all_query_fns = os.listdir(self.dataset + self.query_path)
        self.all_ref_fns = os.listdir(self.dataset + self.reference_path)

        if not load_stats:
            all_query_paths = [self.dataset + self.query_path + q_fn for q_fn in self.all_query_fns]
            all_ref_paths = [self.dataset + self.query_path + ref_fn for ref_fn in self.all_ref_fns]

            batch = 16
            stats = MeanStdRecorder(np.array([cv.imread(img) for img in all_query_paths[:batch]]))
            for i in range(batch, len(all_query_paths), batch):
                stats.update(np.array([cv.imread(img) for img in all_query_paths[i:i + batch]]))

            self.q_mean = stats.means
            self.q_std = stats.stds

            stats = MeanStdRecorder(np.array([cv.imread(img) for img in all_ref_paths[:batch]]))
            for i in range(batch, len(all_ref_paths), batch):
                stats.update(np.array([cv.imread(img) for img in all_ref_paths[i:i + batch]]))

            self.ref_mean = stats.means
            self.ref_std = stats.stds

            with open(dataset + '/mean_std_values.pkl', 'wb') as file:
                pickle.dump(self.q_mean, file)
                pickle.dump(self.q_std, file)
                pickle.dump(self.ref_mean, file)
                pickle.dump(self.ref_std, file)
        else:
            with open(dataset + '/mean_std_values.pkl', 'rb') as file:
                self.q_mean = pickle.load(file)
                self.q_std = pickle.load(file)
                self.ref_mean = pickle.load(file)
                self.ref_std = pickle.load(file)

        print(self.q_mean, self.q_std)



    def __len__(self):
        return len(os.listdir(self.dataset + self.reference_path))

    def __getitem__(self, idx):
        q_image = self.all_query_fns[idx]

        if random.randint(0, 10) < 4:
            if self.dataset == 'ALTO':
                gt_df = pd.read_csv(self.dataset + self.query_path[:-13] + 'gt_matches.csv')
                ref_image = gt_df.loc[gt_df['query_name'] == q_image, 'ref_name'].values[0][14:]
            else:
                ref_image = self.all_ref_fns[idx]
            label = 0
        else:
            ref_image = self.all_ref_fns[random.randint(0, len(self.all_ref_fns) - 1)]
            label = 1


        sample = {'query': cv.imread(self.dataset + self.query_path + q_image),
                  'reference': cv.imread(self.dataset + self.reference_path + ref_image)}


        if self.transform:
            sample = self.transform(sample)

        q_t = transforms.Normalize(mean=self.q_mean, std=self.q_std)
        r_t = transforms.Normalize(mean=self.ref_mean, std=self.ref_std)

        sample['query'] = q_t(sample['query'])
        sample['reference'] = r_t(sample['reference'])

        sample.update({'label': label})

        return sample


class SiameseNetwork(Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)
        self.vgg16.eval()

    def forward_once(self, x):
        output = self.vgg16.forward(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2


class ContrastiveLoss(Module):

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.cdist(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


if __name__ == '__main__':
    dataset = SiameseDataset('ALTO', load_stats=True, transform=transforms.Compose([Resize((512, 512)),
                                                                                     ToTensor()]))

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    net = SiameseNetwork().cuda()
    criterion = ContrastiveLoss(2)
    optimizer = optim.RMSprop(net.parameters(), lr=1e-4, alpha=0.99, eps=1e-8, weight_decay=0.0005, momentum=0.9)

    num_epochs = 1
    counter = []
    history_loss = []

    for epoch in range(num_epochs):
        for idx, sample_batched in enumerate(dataloader):
            input1, input2, label = sample_batched['query'], sample_batched['reference'], sample_batched['label']
            input1, input2, label = input1.cuda(), input2.cuda(), label.cuda()

            optimizer.zero_grad()

            output1, output2 = net.forward(input1, input2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            if idx % 50 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss.item()))
                counter.append(idx)
                history_loss.append(loss.item())
