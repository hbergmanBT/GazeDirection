import os
import cv2
import enum


class ClassifierType(enum.Enum):
    FACE = 'haarcascade_frontalface_default.xml'
    EYE = 'haarcascade_eye.xml'
    RIGHT_EYE = 'haarcascade_righteye_2splits.xml'
    LEFT_EYE = 'haarcascade_lefteye_2splits.xml'


class Classifier(object):
    CLASSIFIERS_DIR = './classifiers/'
    classifiers = {}

    @classmethod
    def init(cls):
        for type_ in ClassifierType:
            cls.classifiers[type_.name] = cv2.CascadeClassifier(os.path.join(cls.CLASSIFIERS_DIR, type_.value))

    @classmethod
    def get(cls, type_):
        return cls.classifiers[type_.name]


Classifier.init()
