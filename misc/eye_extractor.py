import numpy as np
import face_recognition as FR
import cv2

def extract_eyes(image, offset=5, resize_to=None):
    dets = FR.face_landmarks(image, model='large')
    if len(dets) > 0:
        l_eye_marks = dets[0]['left_eye']
        r_eye_marks = dets[0]['right_eye']

        l = np.array(l_eye_marks)
        r = np.array(r_eye_marks)

        # Eye boxes
        lb = min(l[:,1]),max(l[:,1]),min(l[:,0]),max(l[:,0])
        rb = min(r[:,1]),max(r[:,1]),min(r[:,0]),max(r[:,0])

        # Extract eyes
        o = offset
        l_eye = image[lb[0]-o:lb[1]+o,lb[2]-o:lb[3]+o]
        r_eye = image[rb[0]-o:rb[1]+o,rb[2]-o:rb[3]+o]

        if resize_to:
            w,h = resize_to
            l_eye = cv2.resize(l_eye, (w,h))
            r_eye = cv2.resize(r_eye, (w,h))

        return l_eye, r_eye
    else:
        return False

