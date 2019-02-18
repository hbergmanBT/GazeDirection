#!/usr/bin/env python3

import cv2
import os
from numpy import *

classifiers_dir = './classifiers/'
face_classifier_file = 'haarcascade_frontalface_default.xml'
eye_classifier_file = 'haarcascade_eye.xml'
right_eye_classifier_file = 'haarcascade_righteye_2splits.xml'
left_eye_classifier_file = 'haarcascade_lefteye_2splits.xml'

face_cascade = cv2.CascadeClassifier(os.path.join(classifiers_dir, face_classifier_file))
eye_cascade = cv2.CascadeClassifier(os.path.join(classifiers_dir, eye_classifier_file))
right_eye_cascade = cv2.CascadeClassifier(os.path.join(classifiers_dir, right_eye_classifier_file))
left_eye_cascade = cv2.CascadeClassifier(os.path.join(classifiers_dir, left_eye_classifier_file))

class Face(object):
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.centroid = (x+w/2, y+h/2)

class Eye(object):
    def __init__(self, x, y, w, h, face):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.face = face
        self.coords = (x, y, w, h)
        self.centroid = (x+w/2, y+h/2)
        self.score_right = self.scoreRight(self.face)
        self.score_left = self.scoreLeft(self.face)

    def score(self, face, side):
        if side == 'left':
            center = array([2/3*face.w, 1/3*face.h])
        if side == 'right':
            center = array([1/3*face.w, 1/3*face.h])
        distance = sum(power(center-array(list(self.centroid)),2))
        return distance

    def scoreLeft(self, face):
        return self.score(face, side='left')

    def scoreRight(self, face):
        return self.score(face, side='right')


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
    frame = cv2.flip(frame, 1)

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        face = Face(x, y, w, h)
        frame = cv2.rectangle(frame,(x,y), (x+w,y+h), (255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            eye = Eye(ex, ey, ew, eh, face)
            color = (0, 255, 0)
            cv2.rectangle(roi_color,(ex,ey), (ex+ew,ey+eh), color,2)
            # cv2.imshow('roi',roi_color[ey:ey+eh,ex:ex+ew])
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
