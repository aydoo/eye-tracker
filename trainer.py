#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import EyeDataset
from torch.utils.data import DataLoader
from model import EyeNet
from easydict import EasyDict as edict
from tqdm import tqdm

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
                imgs = d['image'].cuda()
                labels = d['label'].cuda()

                self.m.net.zero_grad()
                pred = self.m.net(imgs)

                loss = F.mse_loss(pred, labels)
                loss.backward()
                self.m.opt.step()
                loss_sum += loss

            total_loss.append(loss_sum / (len(self.loader)*self.loader.batch_size))
            yield self.m, e, total_loss[-1]

if __name__ == '__main__':

    epochs = 100
    batch_size = 32

    dataset = EyeDataset('data')
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4,
                        shuffle=True, drop_last=True)

    m = edict()
    m.net = EyeNet().cuda()
    m.opt = optim.Adam(m.net.parameters(), lr=0.001)

    trainer = EyeNetTrainer(m, loader, epochs)
    for _,epoch, loss in trainer.train():
        print(f'Loss: {float(loss)}')
        if (epoch+1) % 10 == 0:
            save_path = f'models/eyenet/epoch_{epoch}_faces_mark_based'
            torch.save(m.net.state_dict(), save_path)
