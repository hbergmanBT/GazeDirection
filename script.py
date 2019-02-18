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

    def selectEyes(self, eyes_list):
        """
        Selects left and right eyes from a list of potential eyes
        according to their relative position to the face
        """
        left_center = array([2/3*self.w, 1/3*self.h])
        right_center = array([1/3*self.w, 1/3*self.h])
        best_eye_left, best_eye_right = None, None
        best_score_left, best_score_right = float('inf'), float('inf')
        for current_eye in eyes_list:
            current_score_left = current_eye.distanceToPoint(left_center)
            current_score_right = current_eye.distanceToPoint(right_center)
            if(current_score_left < best_score_left):
                best_eye_left = current_eye
                best_score_left = current_score_left
            if(current_score_right < best_score_right):
                best_eye_right = current_eye
                best_score_right = current_score_right
        return best_eye_left, best_eye_right

class Eye(object):
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.coords = (x, y, w, h)
        self.centroid = (x+w/2, y+h/2)

    def distanceToPoint(self, point):
        return sum(power(point - array(list(self.centroid)), 2))

    def getTopLeft(self):
        return (self.x, self.y)

    def getBotRight(self):
        return (self.x + self.w, self.y + self.h)

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

        #for (ex, ey, ew, eh) in eyes:
        #    eye = Eye(ex, ey, ew, eh)
        #    color = (0, 255, 0)
        #    cv2.rectangle(roi_color,(ex,ey), (ex+ew,ey+eh), color,2)
        #    # cv2.imshow('roi',roi_color[ey:ey+eh,ex:ex+ew])

        if(0 < len(eyes)):
            (left_eye, right_eye) = face.selectEyes([Eye(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes])

            cv2.rectangle(roi_color, left_eye.getTopLeft(), left_eye.getBotRight(), (0, 255, 255), 2)
            cv2.rectangle(roi_color, right_eye.getTopLeft(), right_eye.getBotRight(), (0, 0, 255), 2)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
