import torch
import cv2
from model import EyeNet
from misc.eye_extractor import extract_eyes
import numpy as np

cap = cv2.VideoCapture(0)
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

model_path = 'models/epoch_50'
m = EyeNet()
m.load_state_dict(torch.load(model_path))

while(True):

    # Save current 'face state' i.e. camera input
    _, img = cap.read()
    eyes = extract_eyes(img, resize_to=(30,20))
    if not eyes: continue
    eyes = np.hstack(eyes)
    eyes = torch.tensor(eyes).permute(2,0,1)[None].float() / 255
    pred = m(eyes).detach().numpy()
    x, y = int(pred[0][0] * 1920), int(pred[0][1] * 1080)
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

