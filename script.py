#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from data_collector import DataCollector
from image import Image
from face import Face
from eye import Eye
from data import Dataset
from classifier import Classifier
from buffer import Buffer

matplotlib.use('TkAgg')


class EyeDirection(object):
    # FACE_CLASSIFIER_FILE = 'haarcascade_frontalface_default.xml'
    # EYE_CLASSIFIER_FILE = 'haarcascade_eye.xml'
    # RIGHT_EYE_CLASSIFIER_FILE = 'haarcascade_righteye_2splits.xml'
    # LEFT_EYE_CLASSIFIER_FILE = 'haarcascade_lefteye_2splits.xml'
    def __init__(self):
        self.dataset = Dataset()
        self.cap = None

        self.showMoments = False
        self.showEvaluation = False

        rollingWindowLength = 3
        self.bufferFace = Buffer(rollingWindowLength)
        self.bufferLeftEye = Buffer(rollingWindowLength)
        self.bufferRightEye = Buffer(rollingWindowLength)

    def startCapture(self):
        self.cap = cv2.VideoCapture(0)

    def stopCapture(self):
        # When everything done, release the capture
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        self.startCapture()
        data_collector = DataCollector(self.dataset)

        keepLoop = True
        while keepLoop:
            pressed_key = cv2.waitKey(1)

            img = self.getCameraImage()
            face, left_eye, right_eye = img.detectEyes(self.bufferFace, self.bufferLeftEye, self.bufferRightEye)
            if face:
                face.draw(img)
            if left_eye:
                left_eye.draw(face)
            if right_eye:
                right_eye.draw(face)

            # Controls
            if pressed_key & 0xFF == ord('q'):
                keepLoop = False
            if pressed_key & 0xFF == ord('s'):
                self.dataset.save()
            if pressed_key & 0xFF == ord('l'):
                self.dataset.load()
            if pressed_key & 0xFF == ord('m'):
                self.showMoments = not self.showMoments
            if pressed_key & 0xFF == ord('e'):
                self.showEvaluation = not self.showEvaluation

            data_collector.step(img.canvas, pressed_key, left_eye, right_eye)

            txt = 'Dataset: {} (s)ave - (l)oad'.format(len(self.dataset))
            cv2.putText(img.canvas, txt, (20, img.canvas.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (32, 32, 32), 2)

            if left_eye and right_eye:
                direction = self.dataset.estimateDirection(left_eye.computeMomentVectors(), right_eye.computeMomentVectors())
                txt = 'Estimated direction: {}'.format(direction.name)
                cv2.putText(img.canvas, txt, (20, img.canvas.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (32, 32, 32), 2)

            img.show()

            if self.showEvaluation:
                fig = self.dataset.showValidationScoreEvolution()
                plt.show()
                self.showEvaluation = False

            if self.showMoments:
                fig = self.dataset.drawVectorizedMoments()
                plt.show()
                # cv2.imshow('moments', self.fig2cv(fig))
                # plt.close(fig)
                self.showMoments = False

        self.stopCapture()

    def fig2cv(self, fig):
        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        return img

    def getCameraImage(self):
        # Capture frame-by-frame
        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (640,480))
        frame = cv2.flip(frame, 1)

        return Image(frame)


if __name__ == '__main__':
    ed = EyeDirection()
    ed.run()
