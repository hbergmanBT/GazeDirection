import os
import cv2
import numpy as np
from face import Face


CLASSIFIERS_DIR = './classifiers/'
FACE_CLASSIFIER_FILE = 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(os.path.join(CLASSIFIERS_DIR, FACE_CLASSIFIER_FILE))

class Image(object):
    """docstring for Image."""
    def __init__(self, frame):
        self.frame = np.copy(frame)
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.faces = []

    @property
    def shape(self):
        return self.frame.shape

    def detectFaces(self):
        faces = face_cascade.detectMultiScale(self.gray, 1.3, 5)
        self.faces = [Face(x, y, w, h, self.frame[y:y+h, x:x+w]) for (x, y, w, h) in faces]

    @property
    def best_face(self):
        n_faces = len(self.faces)
        biggest_area, biggest_face = 0, None
        if n_faces > 0:
            for value in self.faces:
                if biggest_area < value.area:
                    biggest_face = value
                    biggest_area = value.area
        return biggest_face
