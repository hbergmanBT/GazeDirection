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

class Calibrator(object):
    directions = ['TOP', 'BOTTOM', 'LEFT', 'RIGHT', 'CENTER']
    directions_nb = len(directions)

    def __init__(self):
        self.reset();

    def reset(self):
        self.state = 0;
        self.vectorized_double_moments = [None] * Calibrator.directions_nb;

    def isWaitingForEntry(self):
        return self.state < Calibrator.directions_nb;

    def getInstruction(self):
        if(self.isWaitingForEntry()):
            return 'Look at the ' + Calibrator.directions[self.state];
        else:
            return 'Have fun';

    def vectorOfMoment(self, moment):
        res = [float(v) for v in moment.values()];
        return array(res);

    def vectorOfDoubleMoment(self, moment_left, moment_right):
        return concatenate((self.vectorOfMoment(moment_left), self.vectorOfMoment(moment_right)))

    def addDoubleEntry(self, moment_left, moment_right):
        if(self.isWaitingForEntry()):
            self.vectorized_double_moments[self.state] = self.vectorOfDoubleMoment(moment_left, moment_right)
            self.state += 1;

    def getDoubleScores(self, moment_left, moment_right):
        if(self.isWaitingForEntry()):
            return []
        else:
            vectorized_double_moment = self.vectorOfDoubleMoment(moment_left, moment_right)
            return [(d, float(sum(power(vectorized_double_moment - v, 2)))) for (d, v) in zip(Calibrator.directions, self.vectorized_double_moments)]

    def getDoubleResult(self, moment_left, moment_right):
        if(self.isWaitingForEntry()):
            return 'Calibration is not done yet'
        else:
            best_direction = 'Nowhere'
            best_score = float('inf')
            vectorized_double_moment = self.vectorOfDoubleMoment(moment_left, moment_right)
            for(current_direction, current_score) in zip(Calibrator.directions, self.getDoubleScores(moment_left, moment_right)):
                if(current_score < best_score):
                    best_score = current_score
                    best_direction = current_direction
            return best_direction

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

def printCalibratorInfos(frame, calibrator, m_left, m_right):
    frame_h, frame_w = frame.shape[0], frame.shape[1]
    cv2.putText(frame, calibrator.getInstruction(), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    #cv2.putText(frame, calibrator.getDoubleResult(m_left, m_right), (50, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    pos = frame_h - 15
    scores = calibrator.getDoubleScores(m_left, m_right)
    score_total = float(sum([s for (_, s) in scores]))
    for (d, s) in scores:
        cv2.putText(frame, d, (10, pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        x_end = int(100 + 600. * s / score_total)
        cv2.rectangle(frame, (100, pos), (x_end, pos), (0, 255, 255), 2)
        pos -= 40



cap = cv2.VideoCapture(0)
calibrator = Calibrator()

keepLoop = True;
while(keepLoop):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (800,600))
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

            m_left = cv2.moments(roi_gray[left_eye.getTop():left_eye.getBot(), left_eye.getLeft():left_eye.getRight()])
            m_right = cv2.moments(roi_gray[right_eye.getTop():right_eye.getBot(), right_eye.getLeft():right_eye.getRight()])
            
            printCalibratorInfos(frame, calibrator, m_left, m_right)

            if cv2.waitKey(1) & 0xFF == ord('v'):
                calibrator.addDoubleEntry(m_left, m_right);
            if cv2.waitKey(1) & 0xFF == ord('r'):
                calibrator.reset();

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        keepLoop = False

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
