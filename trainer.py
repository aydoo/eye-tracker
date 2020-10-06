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

        for e in range(epochs):
            loss_sum = 0
            for d in tqdm(loader):
                imgs = d['image'].cuda()
                labels = d['label'].cuda()

                self.m.net.zero_grad()
                pred = self.m.net(imgs)
                loss = F.mse_loss(pred, labels)
                loss.backward()
                self.m.opt.step()
                loss_sum += loss

            total_loss.append(loss_sum / len(loader))
            print(total_loss[-1])

if __name__ == '__main__':

    epochs = 50
    batch_size = 20

    dataset = EyeDataset('data')
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0,
                        shuffle=True, drop_last=True)

    m = edict()
    m.net = EyeNet().cuda()
    m.opt = optim.Adam(m.net.parameters(), lr=0.001)

    trainer = EyeNetTrainer(m, loader, epochs)
    trainer.train()

    save_path = f'models/epoch_{epochs}'
    torch.save(m.net.state_dict(), save_path)
