#!/usr/bin/env python
import torch
import cv2
from model import *
from misc.eye_extractor import extract_face
import numpy as np

class Tracker():
    def __init__(self, model_path, resolution=(1920,1080), grid_size=(160,90)):
        torch.set_grad_enabled(False)
        #self.m = EyeNet_16_9(grid_size=grid_size)
        self.m = EyeNet()
        self.m.load_state_dict(torch.load(model_path))
        self.m.eval()
        self.resolution = resolution
        self.grid_size = grid_size
        self.x = self.y = 0

    def predict(self, face):
        xx, yy = self.m(face).detach().numpy()[0]
        xx = xx * self.resolution[0]
        yy = yy * self.resolution[1]
#        s = 0.4 # Filter
#        self.x = self.x * s + xx * (1-s) # filter
#        self.y = self.y * s + yy * (1-s)
        self.x = int(xx)
        self.y = int(yy)
        return int(self.x), int(self.y)

#    def predict(self, face):
#        cls, pos = self.m(face)
#        cls = cls.detach().numpy()[0]
#        pos = pos.detach().numpy()[0]
#
#        cls = torch.nn.functional.log_softmax(torch.tensor(cls).float().view(-1)).view(cls.shape).numpy()
#        idx = cls.argmax()
#        cx = idx // self.grid_size[1]
#        cy = idx % self.grid_size[1]
#        x = (pos[cx,cy][0] + (cx + 0.5) / self.grid_size[0]) * self.resolution[0]
#        y = (pos[cx,cy][1] + (cy + 0.5) / self.grid_size[1]) * self.resolution[1]
#
#        return int(x), int(y)



cap = cv2.VideoCapture(0)
cv2.namedWindow("tracker", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("tracker",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
model_path = 'models/eyenet/epoch_99_faces_fixed'
tracker = Tracker(model_path, resolution=(1920, 1080))#, grid_size=(8,8))

while(True):

    # Read current 'face state' i.e. camera input
    _, img = cap.read()

    face = extract_face(img, resize_to=(100,100))
    if face is False: continue
    face = torch.tensor(face).permute(2,0,1)[None].float()

    x, y = tracker.predict(face)

    # Generate white dot on black background
    img = np.zeros((1080,1920,3))
    s = 2
    img[y-s:y+s,x-s:x+s,:] = 1
    cv2.imshow("tracker", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

