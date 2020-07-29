import numpy as np
import pandas as pd

def distance(pointA, pointB, _norm=np.linalg.norm):
    return _norm(pointA - pointB, axis=1)

class KNNeighbors:
    def __init__(self, formula=distance, k=1):
        self.points = np.array([])
        self.formula = distance
        self.k = k

    def fit(self, points, classes):
        self.points = points.values.reshape(-1, points.shape[1])
        self.classes = classes.values.reshape(-1, 1)

    def predict(self, point):
        distances = np.hstack((self.points, self.formula(self.points, point).reshape(-1, 1), self.classes))
        distances = pd.DataFrame(data=distances, columns=['x', 'y', 'dist', 'class'])
        distances = distances.sort_values(by='dist')
        classification = distances.iloc[:self.k]['class'].value_counts().index[0]

        return classification
