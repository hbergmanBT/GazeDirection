import cv2
import enum
import numpy as np


class EyeType(enum.Enum):
    UNDEFINED = 'undefined'
    LEFT = 'left'
    RIGHT = 'right'


class Eye(object):
    COLORS = {
                EyeType.UNDEFINED: (255, 255, 255),
                EyeType.LEFT: (0, 255, 255),
                EyeType.RIGHT: (0, 0, 255),
             }

    def __init__(self, x, y, w, h, frame, canvas, type_=EyeType.UNDEFINED):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.coords = (x, y, w, h)
        self.centroid = (x+w/2, y+h/2)

        self.type = type_

        self.frame = np.copy(frame[y:y+h,x:x+w])
        self.gray = cv2.cvtColor(frame[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)
        self.canvas = canvas[y:y+h,x:x+w]

        self.moments = None
        self.momentVectors = None

    def distanceToPoint(self, point):
        return np.sum(np.power(point - np.array(list(self.centroid)), 2))

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

    def draw(self, face):
        cv2.rectangle(face.canvas, self.getTopLeft(), self.getBotRight(), self.COLORS[self.type], 2)
        cv2.putText(face.canvas, self.type.value, (self.getLeft(), self.getBot() + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, self.COLORS[self.type], 1)

    def computeMoments(self):
        self.moments = cv2.moments(self.gray)
        return self.moments

    def computeMomentVectors(self):
        """
        Evaluates the 7 moment invariants defined by Hu
        Returns them ordered in an array
        """
        self.computeMoments()
        mu = {}
        for key, value in self.moments.items():
            if 'nu' in key:
                mu[key.replace('nu','')] = value

        mvs = [None] * 7
        mvs[0] = mu['02']*mu['20']
        mvs[1] = (mu['02']-mu['20'])**2 + 4*mu['11']
        mvs[2] = (mu['30']-3*mu['12'])**2 + (+3*mu['21']-mu['03'])**2
        mvs[3] = (mu['30']+mu['12'])**2 + (mu['21']+mu['03'])**2
        mvs[4] = (mu['30']-3*mu['12'])*(mu['30']+mu['12'])\
                *((mu['30']+mu['12'])**2-3*(mu['21']+mu['03'])**2)\
                +(3*mu['21']-mu['03'])*(mu['03']+mu['21'])\
                *(3*(mu['12']+mu['30'])**2-(mu['03']+mu['21'])**2)
        mvs[5] = (mu['02']-mu['20'])*((mu['30']+mu['12'])**2-(mu['21']+mu['03'])**2)\
                +4*(mu['30']+mu['12'])*(mu['21']+mu['03'])
        mvs[6] = (3*mu['21']-mu['03'])*(mu['30']+mu['12'])\
                *((mu['30']+mu['12'])**2-3*(mu['21']+mu['03'])**2)\
                -(mu['21']+mu['03'])*(mu['30']-3*mu['12'])\
                *(3*(mu['30']+mu['12'])**2-(mu['21']+mu['03'])**2)

        self.momentVectors = np.array(mvs)
        return self.momentVectors
