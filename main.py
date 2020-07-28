import numpy as np

def distance(pointA, pointB, _norm=np.linalg.norm):
    return _norm(pointA - pointB, axis=1)

class KNNeighbors:
    def __init__(self, formula=distance, k=1):
        self.points = np.array([])
        self.formula = distance
        self.k = k

    def fit(self, points, classes):
        self.points = points.reshape(-1, points.shape)
        self.classes = classes.reshape(-1, 1)

    def predict(self, point):
        distances = np.hstack((self.points, self.formula(self.points, point).reshape(-1, 1), self.classes))
        distances = np.array(distance, dtype=[('x', float), ('y', float), ('d', float), ('class', int)])
        distances = np.sort(distances, order='d')
        classes, frequency = np.unique(distances['class'][self.k], return_counts=True)
        return classes[frequency.argmax()]
