#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import EyeDataset
from torch.utils.data import DataLoader
from model import *
from easydict import EasyDict as edict
from tqdm import tqdm
from misc.util import point_to_heatmap

class EyeNetTrainer():

    def __init__(self, model, loader, epochs):

        self.m = model
        self.loader = loader
        self.epochs = epochs

    def train(self):
        total_loss = []

        for e in range(self.epochs):
            print(f'Epoch: {e+1}/{self.epochs}')
            loss_sum = 0
            for d in tqdm(self.loader):
                B = self.loader.batch_size
                imgs = d['image'].cuda()
                labels = d['label'].cuda()
#                label_clss = d['label_cls'].cuda()
#                label_poss = d['label_pos'].cuda()

#                self.m.net.zero_grad()
#                p_clss, p_poss = self.m.net(imgs)
#                a = 0.4
#                cls_loss = torch.nn.CrossEntropyLoss()(p_clss.view(B,-1), label_clss.view(B,-1).max(dim=1)[1]) # Classification
#                b = 0.6
#                pos_loss = F.mse_loss(p_poss*(label_poss != 0.), label_poss) # Regression
#                loss = a*cls_loss + b*pos_loss

                pred = self.m.net(imgs)
                loss = F.mse_loss(pred, labels) # Regression

                loss.backward()
                self.m.opt.step()
                loss_sum += loss

            total_loss.append(loss_sum / (len(self.loader)*self.loader.batch_size))
            yield self.m, e, total_loss[-1]

if __name__ == '__main__':

    epochs = 100
    batch_size = 64
    grid_size = (8,8)

    dataset = EyeDataset('data', grid_size=grid_size)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4,
                        shuffle=True, drop_last=True)

    m = edict()
    #m.net = EyeNet_16_9(grid_size).cuda()
    m.net = EyeNet().cuda()
    m.opt = optim.Adam(m.net.parameters(), lr=0.001)

    trainer = EyeNetTrainer(m, loader, epochs)
    best_loss = 1000
    for _,epoch, loss in trainer.train():
        print(f'Loss: {float(loss)}')
        if epoch > 50 and loss < best_loss:
            best_loss = loss
            save_path = f'models/eyenet/epoch_{epoch}_direct_regression_again'
            torch.save(m.net.state_dict(), save_path)
