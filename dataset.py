#!/usr/bin/env python3
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
from easydict import EasyDict as edict
from tqdm import tqdm


class EyeDataset(Dataset):

    def __init__(self, root_path, instances=None):
        self.root = root_path
        self.instances = instances if instances else os.listdir(f'{self.root}/images/')
        self.instances = sorted(self.instances)
        self.labels = self.read_labels(self.instances)
        self.index = self.create_index(self.labels)

    def __len__(self):
        return len(self.index)

    def read_labels(self, instances):
        labels = {}
        for i in instances:
            l = np.genfromtxt(f'{self.root}/labels/{i}.txt', delimiter=',')
            l = torch.tensor(l[:,[1,2]]).float()
            l[:,0] /= 1920 # so that output is between 0 and 1 and can be used with any screen res
            l[:,1] /= 1080
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
        d.image = cv2.imread(f'{self.root}/images/{d.instance}/{d.frame}.jpg', cv2.IMREAD_UNCHANGED)
        d.image = torch.tensor(d.image).permute(2,0,1).float()
        # TODO this should already be NCHW and float
        d.label = self.labels[d.instance][int(d.frame)]

        return d


if __name__ == '__main__':
    print('Debug.')
    import cv2

    dataset = EyeDataset('data')
    loader = DataLoader(dataset, batch_size=1, num_workers=0,
                        shuffle=True, drop_last=True)

    for d in tqdm(loader):
        # First image of batch in CHW format
        img = d['image'][0].permute(1,2,0).numpy()
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

