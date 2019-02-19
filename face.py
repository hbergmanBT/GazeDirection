import numpy as np

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
        left_center = np.array([2/3*self.w, 1/3*self.h])
        right_center = np.array([1/3*self.w, 1/3*self.h])
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
