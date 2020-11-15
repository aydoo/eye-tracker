#!/usr/bin/env python
import os
import numpy as np
import cv2
from tqdm import tqdm
from misc.eye_extractor import extract_face
from datetime import datetime

class CursorChaser():
    def __init__(self, w=1920, h=1080, s=3):
        self.bg = np.zeros((h,w,3))
        self.w = w
        self.h = h
        self.s = s

        cv2.namedWindow('CC',cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('CC',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback('CC',self.update)
        self.cam = cv2.VideoCapture(0)

        self.cursor_positions = []
        self.images = []
        self.start()

    def start(self):
        while(1):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cam.release()
                cv2.destroyAllWindows()
                return

    def update(self, event,x,y,flag,param):
        self.bg *= 0.95
        s = self.s
        self.bg[y-s:y+s,x-s:x+s,1] = 1.
        cv2.imshow('CC', self.bg)

        # Capture current 'face state' i.e. camera input
        _, img = self.cam.read()

        self.cursor_positions.append((x,y))
        self.images.append(img)

def extract_faces(images, cursor_positions, img_shape): # TODO can be batched

    faces, labels = [], []
    for i in tqdm(range(len(images))):
        f = extract_face(images[i], resize_to=img_shape)
        if f is not False: # if a face was detected
            labels.append(cursor_positions[i])
            faces.append(f)
    return faces, labels

def save_data(root, instances, faces, labels):

    img_path = f'{root}/faces/{instance}'
    os.makedirs(img_path, exist_ok=True)

    labels_path = f'{root}/labels/{instance}.txt'
    labels_file = open(labels_path, "a")

    for i,(x,y) in enumerate(tqdm(labels)):

        # Writing image
        img_name = str(i).zfill(5)
        cv2.imwrite(f'{img_path}/{img_name}.jpg', faces[i])

        # Write label
        labels_file.write(f'{img_name},{x},{y}\n')

    labels_file.close()


if __name__ == '__main__':

    # Save paths
    root = 'data'
    instance = datetime.now().strftime("%Y-%m-%d_%H-%M")
    w, h = 1920, 1080

    print(f'Instance: {instance}')
    print(f'Assumed resolution: {w}x{h}')
    print('-'*40)

    print(f'Running cursor chaser...')
    cc = CursorChaser(w=w,h=h)

    print(f'Extracting faces...')
    img_shape = (100,100)
    faces, labels = extract_faces(cc.images, cc.cursor_positions, img_shape=img_shape)

    print(f'Saving data to {root}/...')
    save_data(root, instance, faces, labels)
