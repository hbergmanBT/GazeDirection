import cv2
import os
import matplotlib.pyplot as plt
from numpy import *

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
        #self.vectorized_double_moments = [None] * Calibrator.nb_directions; ##############
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

    def showVectorizedMoments(self):
        if(self.isWaitingForEntry()):
            return
        x_data_left = copy(stack(self.vectorized_moments_left))
        x_data_right = copy(stack(self.vectorized_moments_right))
        y_data = arange(Calibrator.nb_directions * Calibrator.nb_samples) % Calibrator.nb_directions
        for x_data in [x_data_left, x_data_right]:
            x_data = (x_data - mean(x_data, 0)) / std(x_data, 0)
            for i in range(0, 7):
                for j in range(i + 1, 7):
                    plt.subplot(6, 6, 1 + i + 6 * (j-1))
                    plt.scatter(x_data[argwhere(y_data==0),i], x_data[argwhere(y_data==0),j], c='r')
                    plt.scatter(x_data[argwhere(y_data==1),i], x_data[argwhere(y_data==1),j], c='b')
                    plt.scatter(x_data[argwhere(y_data==2),i], x_data[argwhere(y_data==2),j], c='g')
                    plt.scatter(x_data[argwhere(y_data==3),i], x_data[argwhere(y_data==3),j], c='y')
                    plt.scatter(x_data[argwhere(y_data==4),i], x_data[argwhere(y_data==4),j], c='k')
            plt.show()

    def addEntry(self, moment_left, moment_right):
        if(self.isWaitingForEntry()):
            self.vectorized_moments_left[self.state] = self.vectorOfMoment(moment_left)
            self.vectorized_moments_right[self.state] = self.vectorOfMoment(moment_right)
            self.state += 1;

    def addXYData(self, moment_left, direction_num): ############################
        self.x_values.append(self.vectorOfMoment(moment_left))
        self.y_values.append(direction_num)
        print(str(len(self.y_values)))

    def showXYData(self): #############################
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
