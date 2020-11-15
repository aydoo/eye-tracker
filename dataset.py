#!/usr/bin/env python3
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
from easydict import EasyDict as edict
from tqdm import tqdm


class EyeDataset(Dataset):

    def __init__(self, root, instances=None, resolution=(1920,1080), grid_size=None):
        self.root = root
        self.instances = instances if instances else os.listdir(f'{self.root}/faces/')
        self.instances = sorted(self.instances)
        self.resolution = resolution
        self.grid_size = grid_size
        self.labels = self.read_labels(self.instances)
        self.index = self.create_index(self.labels)

    def __len__(self):
        return len(self.index)

    def read_labels(self, instances):
        labels = {}
        for i in instances:
            l = np.genfromtxt(f'{self.root}/labels/{i}.txt', delimiter=',')
            l = torch.tensor(l[:,[1,2]]).float()
            l[:,0] /= self.resolution[0] # output is in [0,1] and can be used with any screen res
            l[:,1] /= self.resolution[1]
            labels[i] = l
        return labels

    def create_index(self, labels):
        index = []
        for i in labels:
            for f,_ in enumerate(labels[i]):
                index.append(edict({'instance': i, 'frame': str(f).zfill(5)}))
        return index

    def __getitem__(self, i):
        e = self.index[i]

        d = edict()
        d.index = i
        d.instance = e.instance
        d.frame = e.frame
        d.image = cv2.imread(f'{self.root}/faces/{d.instance}/{d.frame}.jpg', cv2.IMREAD_UNCHANGED)
        d.image = torch.tensor(d.image).permute(2,0,1).float()
        d.label = self.labels[d.instance][int(d.frame)]

        if self.grid_size is not None:
            d.label_cls = torch.zeros(*self.grid_size).long()
            d.label_pos = torch.zeros(*self.grid_size, 2).float()
            cx = int((d.label[0]-1e-5)*self.grid_size[0])
            cy = int((d.label[1]-1e-5)*self.grid_size[1])
            d.label_cls[cx,cy] = 1.

            d.label_pos[cx,cy,0] = d.label[0] - (cx + 0.5) / self.grid_size[0] # Use cell center as anchor
            d.label_pos[cx,cy,1] = d.label[1] - (cy + 0.5) / self.grid_size[1]

        return d


if __name__ == '__main__':
    print('Debug.')

    # Plot label distribution
    import matplotlib.pyplot as plt
    from dataset import EyeDataset
    import numpy as np

    root = 'data'
    w, h = 1920, 1080
    gw, gh = 16, 9

    data = EyeDataset(root=root, grid_size=(gw,gh))
    print('Data:', len(data))

    x = [d['label'][0]*w for d in data]
    y = [d['label'][1]*h for d in data]

    plt.scatter(x, y, alpha=0.1)
    plt.axis('equal')
    plt.show()

#    hist = np.zeros((gh,gw))
#    for d in data:
#
#        idx = d['label_cls'].squeeze().argmax()
#        cx = idx // gh
#        cy = idx % gh
#        hist[cy,cx] += 1
#    print(hist)
#    plt.matshow(hist,cmap='gray')
#    plt.show()
