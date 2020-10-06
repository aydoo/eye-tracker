
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class EyeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(128, 256, 3)

        #self.fccat = nn.Linear(128, 64)

        self.fc1 = nn.Linear(256*6*26, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, img):
        batchsize = len(img)
        x = img
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))

        x = x.view(-1,256*6*26)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
#        x = F.batch_norm(F.relu(self.fc1(x)))
#        x = F.batch_norm(F.relu(self.fc2(x)))
#        x = F.batch_norm(F.relu(self.fc3(x)))

        return x


if __name__ == '__main__':
    m = EyeNet().cuda()
    x = torch.zeros((100,3,20,30)).cuda()
    m(x)

