#!/usr/bin/env python3

import cv2
import os
from numpy import *

classifiers_dir = './classifiers/'
face_classifier_file = 'haarcascade_frontalface_default.xml'
eye_classifier_file = 'haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(os.path.join(classifiers_dir, face_classifier_file))
eye_cascade = cv2.CascadeClassifier(os.path.join(classifiers_dir, eye_classifier_file))

class Face(object):
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

class Eye(object):
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.coords = (x, y, w, h)

    def isValid(self, face):
        ratio_ymin, ratio_ymax = (1/4, 1/2)
        ratio_xmin_left, ratio_xmax_left = (1/8, 3/8)
        ratio_xmin_right, ratio_xmax_right = (5/8, 7/8)
        ratio_hmin, ratio_hmax = (1/8, 1/2)

        (ex, ey, ew, eh) = self.coords
        (ecx, ecy) = (ex + eh/2, ey+ew/2)
        if not (ratio_ymin*face.h<ecy<ratio_ymax*face.h):
            return False
        if not (ratio_xmin_left*face.w<ecx<ratio_xmax_left*face.w or ratio_xmin_right*face.w<ecx<ratio_xmax_right*face.w):
            return False
        if not (ratio_hmin*face.h<eh<ratio_hmax*face.h):
            return False
        return True

def getValidEyes(eyes, face):
    if type(eyes)==tuple: # TODO: != np.ndarray
        return []
    valid_eyes = eyes[apply_along_axis(isValid, 1, eyes, face)]
    xv, yv = meshgrid(eyes, eyes)
    return valid_eyes

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (400,300))

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(len(faces))
    for (x,y,w,h) in faces:
        face = Face(x, y, w, h)
        frame = cv2.rectangle(frame,(x,y), (x+w,y+h), (255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eye = Eye(ex, ey, ew, eh)
            color = (0, 255, 0)
            if eye.isValid(face):
                cv2.rectangle(roi_color,(ex,ey), (ex+ew,ey+eh), color,2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
