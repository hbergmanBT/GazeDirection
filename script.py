#!/usr/bin/env python3

import cv2
import os
from numpy import *
import matplotlib.pyplot as plt

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
    nb_directions = len(directions)
    nb_samples = 5
    nb_neighbours = 3

    def __init__(self):
        self.reset();

    def reset(self):
        self.state = 0;
        self.vectorized_moments_left = [None] * (Calibrator.nb_directions * Calibrator.nb_samples)
        self.vectorized_moments_right = [None] * (Calibrator.nb_directions * Calibrator.nb_samples)
        self.vectorized_double_moments = [None] * Calibrator.nb_directions; ##############
        self.x_values = []
        self.y_values = []

    def isWaitingForEntry(self):
        return self.state < Calibrator.nb_directions * Calibrator.nb_samples;

    def getInstruction(self):
        if(self.isWaitingForEntry()):
            return 'Look at the %s (%d/%d)' % (Calibrator.directions[self.state % Calibrator.nb_directions], self.state, Calibrator.nb_directions * Calibrator.nb_samples)
        else:
            return 'Have fun';

    def vectorOfMoment(self, moment):
        """
        Evaluates the 7 moment invariants defined by Hu
        Returns them ordered in an array
        """
        mu = {}
        for key, value in moment.items():
            if 'nu' in key:
                mu[key.replace('nu','')] = value
        HMI1 = mu['02']*mu['20']
        HMI2 = (mu['02']-mu['20'])**2 + 4*mu['11']
        HMI3 = (mu['30']-3*mu['12'])**2 + (+3*mu['21']-mu['03'])**2
        HMI4 = (mu['30']+mu['12'])**2 + (mu['21']+mu['03'])**2
        HMI5 = (mu['30']-3*mu['12'])*(mu['30']+mu['12'])\
                *((mu['30']+mu['12'])**2-3*(mu['21']+mu['03'])**2)\
                +(3*mu['21']-mu['03'])*(mu['03']+mu['21'])\
                *(3*(mu['12']+mu['30'])**2-(mu['03']+mu['21'])**2)
        HMI6 = (mu['02']-mu['20'])*((mu['30']+mu['12'])**2-(mu['21']+mu['03'])**2)\
                +4*(mu['30']+mu['12'])*(mu['21']+mu['03'])
        HMI7 = (3*mu['21']-mu['03'])*(mu['30']+mu['12'])\
                *((mu['30']+mu['12'])**2-3*(mu['21']+mu['03'])**2)\
                -(mu['21']+mu['03'])*(mu['30']-3*mu['12'])\
                *(3*(mu['30']+mu['12'])**2-(mu['21']+mu['03'])**2)
        res = [HMI1, HMI2, HMI3, HMI4, HMI5, HMI6, HMI7]
        #print(res)
        return array(res);

    def addEntry(self, moment_left, moment_right):
        if(self.isWaitingForEntry()):
            self.vectorized_moments_left[self.state] = self.vectorOfMoment(moment_left)
            self.vectorized_moments_right[self.state] = self.vectorOfMoment(moment_right)
            self.state += 1;

    def addXYData(self, moment_left, direction_num):
        self.x_values.append(self.vectorOfMoment(moment_left))
        self.y_values.append(direction_num)
        print(str(len(self.y_values)))

    def showXYData(self):
        x_data = stack(self.x_values)
        x_data = (x_data - mean(x_data, 0)) / std(x_data, 0)
        y_data = stack(self.y_values)
        for i in range(0, 7):
            for j in range(i + 1, 7):
                plt.subplot(6, 6, 1 + i + 6 * (j-1))
                plt.scatter(x_data[argwhere(y_data==0),i], x_data[argwhere(y_data==0),j], c='r')
                plt.scatter(x_data[argwhere(y_data==1),i], x_data[argwhere(y_data==1),j], c='b')
                plt.scatter(x_data[argwhere(y_data==2),i], x_data[argwhere(y_data==2),j], c='g')
                plt.scatter(x_data[argwhere(y_data==3),i], x_data[argwhere(y_data==3),j], c='y')
                plt.scatter(x_data[argwhere(y_data==4),i], x_data[argwhere(y_data==4),j], c='k')
        plt.show()

    def getScores(self, moment_left, moment_right):
        if(self.isWaitingForEntry()):
            return []
        else:
            vectorized_moment_left = self.vectorOfMoment(moment_left)
            vectorized_moment_right = self.vectorOfMoment(moment_right)

            all_distances_left = sum(power(vectorized_moment_left - self.vectorized_moments_left, 2), 1)
            best_args_left = argsort(all_distances_left)[:Calibrator.nb_neighbours] % Calibrator.nb_directions
            scores_left = histogram(best_args_left, bins=arange(Calibrator.nb_directions + 1))[0]

            all_distances_right = sum(power(vectorized_moment_right - self.vectorized_moments_right, 2), 1)
            best_args_right = argsort(all_distances_right)[:Calibrator.nb_neighbours] % Calibrator.nb_directions
            scores_right = histogram(best_args_right, bins=arange(Calibrator.nb_directions + 1))[0]

            return [(d, s) for (d, s) in zip(Calibrator.directions, amin([scores_left, scores_right], 0))]

    def getResult(self, moment_left, moment_right):
        if(self.isWaitingForEntry()):
            return 'Calibration is not done yet'
        else:
            best_direction = 'Nowhere'
            best_score = float('inf')
            for(current_direction, current_score) in self.getScores(moment_left, moment_right):
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
    #cv2.putText(frame, calibrator.getResult(m_left, m_right), (50, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    pos = frame_h - 15
    scores = calibrator.getScores(m_left, m_right)
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

        if(0 < len(eyes)):
            (left_eye, right_eye) = face.selectEyes([Eye(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes])
            cv2.rectangle(roi_color, left_eye.getTopLeft(), left_eye.getBotRight(), (0, 255, 255), 2)
            cv2.rectangle(roi_color, right_eye.getTopLeft(), right_eye.getBotRight(), (0, 0, 255), 2)

            m_left = cv2.moments(roi_gray[left_eye.getTop():left_eye.getBot(), left_eye.getLeft():left_eye.getRight()])
            m_right = cv2.moments(roi_gray[right_eye.getTop():right_eye.getBot(), right_eye.getLeft():right_eye.getRight()])
            
            printCalibratorInfos(frame, calibrator, m_left, m_right)

            pressed_key = cv2.waitKey(1)

            if pressed_key & 0xFF == ord('v'):
                calibrator.addEntry(m_left, m_right);
            if pressed_key & 0xFF == ord('r'):
                calibrator.reset();

            if pressed_key & 0xFF == ord('e'):
                calibrator.addXYData(m_left, 0)
            if pressed_key & 0xFF == ord('s'):
                calibrator.addXYData(m_left, 2)
            if pressed_key & 0xFF == ord('f'):
                calibrator.addXYData(m_left, 3)
            if pressed_key & 0xFF == ord('d'):
                calibrator.addXYData(m_left, 4)
            if pressed_key & 0xFF == ord('c'):
                calibrator.addXYData(m_left, 1)
            if pressed_key & 0xFF == ord('z'):
                calibrator.showXYData()

            if pressed_key & 0xFF == ord('q'):
                keepLoop = False

    cv2.imshow('frame',frame)
    

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
