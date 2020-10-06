#!/usr/bin/env python
import os
import numpy as np
import cv2
from tqdm import tqdm
from misc.eye_extractor import extract_eyes
from datetime import datetime

def points_to_path(points, num_points=50):
    path = []
    for i in range(1, len(points)):
        d = points[i-1]
        path += list(np.linspace(points[i-1], points[i], num_points).astype(int))
    return path

def run_point_chasing(w=1920, h=1080, s=10, s_2=5, look_ahead_len=20):

    w, h = 1920, 1080
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cam = cv2.VideoCapture(0)

    points = [[  s,  s],
              [h-s,  s],
              [h-s,w-s],
              [  s,w-s],
              [  s,  s]]
    points += list(np.vstack([np.random.randint(s, h-s, 16),
                              np.random.randint(s, w-s, 16)]).T)
    path = points_to_path(points)
    images = []
    for i,(y,x) in enumerate(tqdm(path)):

        img = np.zeros((int(h),int(w),3))

        # Look ahead
        for j,(yy,xx) in enumerate(path[i+1:i+look_ahead_len]):
            img[yy-s_2:yy+s_2,xx-s_2:xx+s_2,:] = 0.99*(1./(j+1))

        # Generate white dot on black background
        img[y-s:y+s,x-s:x+s,1] = 1

        cv2.imshow("window", img)

        # Save current 'face state' i.e. camera input
        _, img = cam.read()
        images.append(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()

    cam.release()
    cv2.destroyAllWindows()

    return images, path

def extract_both_eyes(images, resize_to=None):
    combined_eyes = []
    for img in tqdm(images):
        eyes = extract_eyes(img, resize_to=resize_to)
        combined_eyes.append(np.hstack(eyes))
    return combined_eyes

def save_data(eyes, labels, root, instance):

    img_path = f'{root}/images/{instance}'
    os.makedirs(img_path, exist_ok=True)

    labels_path = f'{root}/labels/{instance}.txt'
    labels_file = open(labels_path, "a")

    for i,(y,x) in enumerate(tqdm(path)):

        # Writing image
        img_name = str(i).zfill(5)
        cv2.imwrite(f'{img_path}/{img_name}.jpg', eyes[i])

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
    print(f'Running point chaser:')
    images, path = run_point_chasing(w=w, h=h)
    print(f'Extracting eyes:')
    eyes = extract_both_eyes(images, resize_to=(30,20))
    print(f'Saving data to {root}/:')
    save_data(eyes, path, root, instance)
