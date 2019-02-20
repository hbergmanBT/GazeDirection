import os
import cv2
import numpy as np
from eye import Eye

CLASSIFIERS_DIR = './classifiers/'
EYE_CLASSIFIER_FILE = 'haarcascade_eye.xml'
RIGHT_EYE_CLASSIFIER_FILE = 'haarcascade_righteye_2splits.xml'
LEFT_EYE_CLASSIFIER_FILE = 'haarcascade_lefteye_2splits.xml'

eye_cascade = cv2.CascadeClassifier(os.path.join(CLASSIFIERS_DIR, EYE_CLASSIFIER_FILE))
right_eye_cascade = cv2.CascadeClassifier(os.path.join(CLASSIFIERS_DIR, RIGHT_EYE_CLASSIFIER_FILE))
left_eye_cascade = cv2.CascadeClassifier(os.path.join(CLASSIFIERS_DIR, LEFT_EYE_CLASSIFIER_FILE))

class Face(object):
    def __init__(self, x, y, w, h, frame):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.frame = np.copy(frame)
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.centroid = (x+w/2, y+h/2)
        self.area = self.w*self.h
        self.eyes = []
        self.best_eye_left = None
        self.best_eye_right = None


    def detectEyes(self):
        eyes = eye_cascade.detectMultiScale(self.gray, 1.3, 5)
        self.eyes = [Eye(x, y, w, h) for (x, y, w, h) in eyes]

    def getLeft(self):
        return self.x;

    def getRight(self):
        return self.x + self.w;

    def getTop(self):
        return self.y;

    def getBot(self):
        return self.y + self.h;

    def getTopLeft(self):
        return (self.x, self.y)

    def getBotRight(self):
        return (self.x + self.w, self.y + self.h)

    def selectEyes(self):
        """
        Selects left and right eyes from a list of potential eyes
        according to their relative position to the face
        """
        left_center = np.array([2/3*self.w, 1/3*self.h]) # a priori on the relative position of the left eye on the face
        right_center = np.array([1/3*self.w, 1/3*self.h]) # a priori on the relative position of the right eye on the face
        self.best_eye_left, self.best_eye_right = None, None
        best_score_left, best_score_right = float('inf'), float('inf')
        for current_eye in self.eyes:
            current_score_left = current_eye.distanceToPoint(left_center)
            current_score_right = current_eye.distanceToPoint(right_center)
            if(current_score_left < best_score_left):
                self.best_eye_left = current_eye
                best_score_left = current_score_left
            if(current_score_right < best_score_right):
                self.best_eye_right = current_eye
                best_score_right = current_score_right
        return self.best_eye_left, self.best_eye_right
