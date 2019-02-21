import enum
import pickle
import numpy as np
import matplotlib.pyplot as plt


class Direction(enum.Enum):
    UNDEFINED = enum.auto()
    CENTER = enum.auto()
    UP = enum.auto()
    LEFT = enum.auto()
    RIGHT = enum.auto()
    DOWN = enum.auto()

    @classmethod
    def getString(cls, direction):
        return {
            cls.UNDEFINED: '?',
            cls.CENTER: 'center',
            cls.UP: 'up',
            cls.LEFT: 'left',
            cls.RIGHT: 'right',
            cls.DOWN: 'down',
        }[direction]

    @classmethod
    def successor(cls, direction):
        if direction == cls.DOWN:
            return cls.CENTER
        return Direction(direction.value + 1)

    @classmethod
    def precessor(cls, direction):
        if direction == cls.CENTER:
            return cls.DOWN
        return Direction(direction.value - 1)


class Data(object):
    def __init__(self, frame, left_eye, right_eye, direction=Direction.UNDEFINED):
        self.frame = np.copy(frame)
        self.left_eye = left_eye.__dict__
        self.right_eye = right_eye.__dict__
        self.direction = direction.value

        self.left_moments = left_eye.computeMomentVectors()
        self.right_moments = right_eye.computeMomentVectors()


class Dataset(object):
    NB_NEIGHBOURS = 3
    PERCENT_TRAINING_SET = 0.3

    def __init__(self):
        self.dataset_file = "dataset"
        self.clear()

    def append(self, data):
        if data not in self.data:
            self.data.append(data)

    def deleteLastEntry(self):
        if self.data:
            self.data = self.data[:-1]
            return True
        return False

    def clear(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def save(self):
        pickle.dump(self.data, open(self.dataset_file, 'wb'))

    def load(self):
        try:
            self.data = pickle.load(open(self.dataset_file, 'rb'))
        except Exception as e:
            print(e)

    def leftMoments(self, ids=None):
        data = self.data if ids is None else np.array(self.data)[ids]
        if len(data) == 0:
            return []
        return np.stack([d.left_moments for d in data])

    def rightMoments(self, ids=None):
        data = self.data if ids is None else np.array(self.data)[ids]
        if len(data) == 0:
            return []
        return np.stack([d.right_moments for d in data])

    def labels(self, ids=None):
        data = self.data if ids is None else np.array(self.data)[ids]
        if len(data) == 0:
            return []
        return np.array([d.direction for d in data])

    def directionProbabilities(self, moment_left, moment_right, idsTraining=None):
        if not self.data:
            return []
        labels = self.labels(idsTraining)

        all_distances_left = np.sum(np.power(moment_left - self.leftMoments(idsTraining), 2), 1)
        best_args_left = labels[np.argsort(all_distances_left)[:self.NB_NEIGHBOURS]] - 1
        scores_left = np.histogram(best_args_left, bins=np.arange(len(Direction) + 1))[0]

        all_distances_right = np.sum(np.power(moment_right - self.rightMoments(idsTraining), 2), 1)
        best_args_right = labels[np.argsort(all_distances_right)[:self.NB_NEIGHBOURS]] - 1
        scores_right = np.histogram(best_args_right, bins=np.arange(len(Direction) + 1))[0]

        return np.array(list(zip(Direction, np.sum([scores_left, scores_right], 0))))

    def estimateDirection(self, moment_left, moment_right, idsTraining=None):
        best_direction = Direction.UNDEFINED
        scores = self.directionProbabilities(moment_left, moment_right, idsTraining=idsTraining)
        if len(scores) > 0:
            best_direction = scores[np.argmax(scores[:, 1]), 0]
        return best_direction

    def getValidationScore(self, maxLimit=None):
        idsTraining, idsValidation = self.getCrossValidationIds(maxLimit)
        scores = []
        for left_moments, right_moments, label in zip(self.leftMoments(idsValidation),
                                                      self.rightMoments(idsValidation),
                                                      self.labels(idsValidation)):
            estimation = self.estimateDirection(left_moments, right_moments, idsTraining=idsTraining)
            scores.append(float(label == estimation.value))
        score = np.mean(scores)
        return score

    def getValidationScoreEvolution(self, step=1):
        scores = []
        for i in range(2, len(self.data), step):
            scores.append((i, self.getValidationScore(i)))
        return scores

    def getCrossValidationIds(self, maxLimit=None):
        maxLimit = maxLimit if maxLimit else len(self.data)
        numberValidation = max(1, int(maxLimit * self.PERCENT_TRAINING_SET))
        ids = np.arange(maxLimit)
        np.random.shuffle(ids)
        idsValidation = ids[:numberValidation]
        idsTraining = ids[numberValidation:]
        return idsTraining, idsValidation

    def drawVectorizedMoments(self):
        x_data_left = np.copy(self.leftMoments())
        x_data_right = np.copy(self.rightMoments())
        y_data = np.copy(self.labels())
        colors = {
            Direction.CENTER: 'k',
            Direction.UP: 'r',
            Direction.LEFT: 'g',
            Direction.RIGHT: 'y',
            Direction.DOWN: 'b',
        }
        # for x_data in [x_data_left, x_data_right]:
        for x_data in [x_data_left]:
            x_data = (x_data - np.mean(x_data, 0)) / np.std(x_data, 0)
            fig = plt.figure()
            for i in range(0, 7):
                for j in range(i + 1, 7):
                    plt.subplot(6, 6, 1 + i + 6 * (j-1))
                    for direction in Direction:
                        if direction != Direction.UNDEFINED:
                            ids = np.argwhere(y_data==direction.value)
                            plt.scatter(x_data[ids,i], x_data[ids,j], c=colors[direction], label=direction.name)
            plt.figlegend(labels=colors.keys(),loc="upper right")
        return fig

    def showValidationScoreEvolution(self):
        scores = np.array(self.getValidationScoreEvolution())
        fig = plt.figure()
        plt.plot(scores[:, 0], scores[:, 1])
        return fig
