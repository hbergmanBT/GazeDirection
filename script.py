#!/usr/bin/env python3

import cv2
import os
import matplotlib.pyplot as plt
from numpy import *
from calibrator import Calibrator
from image import Image
from face import Face
from eye import Eye

CLASSIFIERS_DIR = './classifiers/'
FACE_CLASSIFIER_FILE = 'haarcascade_frontalface_default.xml'
EYE_CLASSIFIER_FILE = 'haarcascade_eye.xml'
RIGHT_EYE_CLASSIFIER_FILE = 'haarcascade_righteye_2splits.xml'
LEFT_EYE_CLASSIFIER_FILE = 'haarcascade_lefteye_2splits.xml'

face_cascade = cv2.CascadeClassifier(os.path.join(CLASSIFIERS_DIR, FACE_CLASSIFIER_FILE))
eye_cascade = cv2.CascadeClassifier(os.path.join(CLASSIFIERS_DIR, EYE_CLASSIFIER_FILE))
right_eye_cascade = cv2.CascadeClassifier(os.path.join(CLASSIFIERS_DIR, RIGHT_EYE_CLASSIFIER_FILE))
left_eye_cascade = cv2.CascadeClassifier(os.path.join(CLASSIFIERS_DIR, LEFT_EYE_CLASSIFIER_FILE))


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
    frame = cv2.resize(frame, (640,480))
    frame = cv2.flip(frame, 1)

    img = Image(frame)
    gray = copy(img.gray)
    img.detectFaces()
    face = img.best_face
    if face:
        frame = cv2.rectangle(frame, face.getTopLeft(), face.getBotRight(), (255,0,0),2)
        roi_gray = gray[face.y:face.y+face.h,face.x:face.x+face.w]
        roi_color = frame[face.y:face.y+face.h,face.x:face.x+face.w]
        face.detectEyes()
        if face.eyes:
            (left_eye, right_eye) = face.selectEyes()
            cv2.rectangle(roi_color, left_eye.getTopLeft(), left_eye.getBotRight(), (0, 255, 255), 2)
            cv2.rectangle(roi_color, right_eye.getTopLeft(), right_eye.getBotRight(), (0, 0, 255), 2)

            m_left = cv2.moments(roi_gray[left_eye.getTop():left_eye.getBot(), left_eye.getLeft():left_eye.getRight()])
            m_right = cv2.moments(roi_gray[right_eye.getTop():right_eye.getBot(), right_eye.getLeft():right_eye.getRight()])

            printCalibratorInfos(frame, calibrator, m_left, m_right)

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


    cv2.imshow('frame',frame)
    pressed_key = cv2.waitKey(1)
    if pressed_key & 0xFF == ord('q'):
        keepLoop = False



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
