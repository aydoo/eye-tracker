#!/usr/bin/env python
import numpy as np
import cv2
from tqdm import tqdm
from misc.eye_extractor import extract_eyes

def points_to_path(points, num_points=60):
    path = []
    for i in range(1, len(points)):
        d = points[i-1]
        path += list(np.linspace(points[i-1], points[i], num_points).astype(int))
    return path


w, h = 1920, 1080
scale = 1
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cap = cv2.VideoCapture(0)

s = 10
path = points_to_path([[  s,  s],
                       [h-s,  s],
                       [h-s,w-s],
                       [  s,w-s],
                       [  s,  s]])
path += points_to_path(list(np.vstack([np.random.randint(s, h-s, 20),
                                       np.random.randint(s, w-s, 20)]).T))

images = []
for i,(y,x) in enumerate(path):

    # Generate white dot on black background
    img = np.zeros((int(h*scale),int(w*scale),3))
    img[y-s:y+s,x-s:x+s,:] = 1
    cv2.imshow("window", img)

    # Save current 'face state' i.e. camera input
    _, img = cap.read()
    images.append(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

jjjjjjjjj

# Save paths
img_path = 'data/images'
labels_path = 'data/labels.txt'
labels_file = open(labels_path, "a")

for i,(y,x) in tqdm(enumerate(path)):

    eyes = extract_eyes(images[i], resize_to=(30,20))
    combined_eyes = np.hstack(eyes)

    # Writing image
    img_name = str(i).zfill(5)
    cv2.imwrite(f'{img_path}/{img_name}.jpg', combined_eyes)

    # Write label
    labels_file.write(f'{img_name},{x},{y},\n')


# When everything done, release the capture
labels_file.close()

