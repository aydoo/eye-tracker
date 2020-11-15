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


# Faster, but boxes are jumpy
def extract_face_2(image, offset=5, resize_to=None):
    dets = FR.face_locations(image, model='small', number_of_times_to_upsample=0)
    FR.face_locations
    if len(dets) > 0:
        c = dets[0]
        o = offset
        face = image[c[0]:c[2],
                     c[3]:c[1]]

        if resize_to:
            w,h = resize_to
            face = cv2.resize(face, resize_to)

        return face
    else:
        return False

# Slow, but less jumpy. Concludes box from landmarks.
def extract_face(image, offset=5, resize_to=None):
    dets = FR.face_landmarks(image, model='large')
    if len(dets) > 0:
        c = np.array(dets[0]['chin'])
        leb = np.array(dets[0]['right_eyebrow'])
        reb = np.array(dets[0]['right_eyebrow'])
        merged = np.vstack([c,leb,reb])
        xmax,ymax = merged.max(axis=0)
        xmin,ymin = merged.min(axis=0)

        # Extract eyes
        o = offset
        face = image[ymin-o:ymax+o,xmin-o:xmax+o]

        if resize_to:
            w,h = resize_to
            face = cv2.resize(face, (w,h))

        return face
    else:
        return False

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)

    while(True):
        ret, frame = cam.read()
#        eyes = extract_eyes(frame, resize_to=(30,20))
#
#        if eyes is not False:
#            combined_eyes = np.hstack(eyes)
#            cv2.imshow('frame',combined_eyes)
#            if cv2.waitKey(1) & 0xFF == ord('q'):
#                break

        face = extract_face2(frame, resize_to=(100,100))

        if face is not False:
            cv2.imshow('frame',face)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

#        face = extract_face(frame, resize_to=(128,128))
#
#        if face is not None:
#            cv2.imshow('frame', face)
#            if cv2.waitKey(1) & 0xFF == ord('q'):
#                break
#
    # When everything done, release the camture
    cam.release()
    cv2.destroyAllWindows()
