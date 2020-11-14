#!/usr/bin/env python
import torch
import cv2
from model import EyeNet as EyeNet
from misc.eye_extractor import extract_eyes
import numpy as np

class Tracker():
    def __init__(self, model_path, w, h):
        torch.set_grad_enabled(False)
        self.m = EyeNet()
        self.m.load_state_dict(torch.load(model_path))
        self.m.eval()
        self.w, self.h = w, h
        self.x = self.y = 0

    def predict(self, eyes):
        pred = self.m(eyes).detach().numpy()
        xx, yy = pred[0][0] * self.w, pred[0][1] * self.h # Scale to resolution
        s = 0.4
#        self.x = self.x * s + xx * (1-s) # filter
#        self.y = self.y * s + yy * (1-s)
        self.x = int(xx)
        self.y = int(yy)
        return int(self.x), int(self.y)



cap = cv2.VideoCapture(0)
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
model_path = 'models/eyenet/epoch_99_255_2_pool'
tracker = Tracker(model_path, 1920, 1080)

while(True):

    # Read current 'face state' i.e. camera input
    _, img = cap.read()
    eyes = extract_eyes(img, resize_to=(30,20))
    if not eyes: continue
    eyes = np.hstack(eyes)
    eyes = torch.tensor(eyes).permute(2,0,1)[None].float()

    x, y = tracker.predict(eyes)
    print('pred',x,y)

    # Generate white dot on black background
    img = np.zeros((1080,1920,3))
    s = 2
    img[y-s:y+s,x-s:x+s,:] = 1
    cv2.imshow("window", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

