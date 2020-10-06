
import numpy as np
import cv2

w, h = 1920, 1080
scale = 1
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cap = cv2.VideoCapture(0)

for i in range(h):
    img = np.zeros((int(h*scale),int(w*scale),3))
    img[i,i,:] = 1
    cv2.imshow("window", img)
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

