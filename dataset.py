#!/usr/bin/env python3
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
from easydict import EasyDict as edict
from tqdm import tqdm


class EyeDataset(Dataset):

    def __init__(self, root_path):
        self.root = root_path
        self.labels = np.genfromtxt(f'{self.root}/labels.txt', delimiter=',')
        self.labels = torch.tensor(self.labels[:,[1,2]]).float()
        self.labels[:,0] /= 1920 # TODO this should be down before
        self.labels[:,1] /= 1080 # TODO this should be down before

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        d = edict()
        d.index = i

        img_name = str(i).zfill(5)
        d.image = cv2.imread(f'{self.root}/images/{img_name}.jpg', cv2.IMREAD_UNCHANGED)
        d.image = torch.tensor(d.image).permute(2,0,1).float() / 255
        # TODO this should already be NCHW and float
        d.label = self.labels[i]

        return d


if __name__ == '__main__':
    import cv2

    dataset = EyeDataset('data')
    loader = DataLoader(dataset, batch_size=5, num_workers=0,
                        shuffle=True, drop_last=True)

    for d in tqdm(loader):

#        img_batch = d['image'].cuda()
#        label_batch = d['label'].cuda()

        cv2.imshow('frame', d['image'][0].numpy())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

