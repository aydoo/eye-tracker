import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class EyeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.bn11 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn12 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn13 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bn14 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256*8*8, 512)
        self.bn21 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn22 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn23 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 2)

    def forward(self, img):
        x = img
        x = self.bn11(F.relu(self.conv1(x)))
        x = self.pool1(x)
        x = self.bn12(F.relu(self.conv2(x)))
        x = self.pool2(x)
        x = self.bn13(F.relu(self.conv3(x)))
        x = self.pool3(x)
        x = self.bn14(F.relu(self.conv4(x)))

        x = x.view(-1,256*8*8)

        x = self.bn21(F.relu(self.fc1(x)))
        x = self.bn22(F.relu(self.fc2(x)))
        x = self.bn23(F.relu(self.fc3(x)))
        x = self.fc4(x)

        return x

class EyeNet_16_9(nn.Module):
    def __init__(self, grid_size=(16,9)):
        super().__init__()

        self.gx, self.gy = grid_size

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.bn11 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn12 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn13 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bn14 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256*8*8, 512)
        self.bn21 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn22 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, self.gx*self.gy*3)

    def forward(self, img):
        B = img.shape[0]
        x = img
        x = self.bn11(F.relu(self.conv1(x)))
        x = self.pool1(x)
        x = self.bn12(F.relu(self.conv2(x)))
        x = self.pool2(x)
        x = self.bn13(F.relu(self.conv3(x)))
        x = self.pool3(x)
        x = self.bn14(F.relu(self.conv4(x)))

        x = x.view(-1,256*8*8)

        x = self.bn21(F.relu(self.fc1(x)))
        x = self.bn22(F.relu(self.fc2(x)))
        x = self.fc3(x)

        x = x.view(-1,self.gx,self.gy,3)

        cls = x[:,:,:,0].contiguous()
        #cls = nn.functional.log_softmax(x[:,:,:,0].view(B,-1)).view(x[:,:,:,0].shape)
        pos = x[:,:,:,1:].contiguous()

        return cls, pos


if __name__ == '__main__':
    m = EyeNet_16_9().cuda()
    x = torch.zeros((32,3,100,100)).cuda()
    m(x)
