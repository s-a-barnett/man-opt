import numpy as np
from numpy.linalg import norm

class MLE(Cost):

    def __init__(self, manifold, Y, B):
        if type(manifold).__name__ in ['LinearProduct', 'StiefelProduct']:
            raise ValueError('manifold must be FixedRank')

        self.manifold = manifold
        self.Y = Y
        self.B = B

    def _eval(self, xx):
        return 0.5 * (norm((xx - self.Y) * self.B) ** 2)

    def _euclideanGradient(self, xx):
        return ((xx - self.Y) * self.B)

    def _euclideanHessian(self, xx, hh):
        return hh * self.B
