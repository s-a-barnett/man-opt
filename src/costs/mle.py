import numpy as np
import scipy.sparse as sparse

from .cost import Cost

class MLE(Cost):

    def __init__(self, manifold, Y, B):
        if type(manifold).__name__ in ['LinearProduct', 'StiefelProduct']:
            raise ValueError('manifold must be FixedRank')

        self.manifold = manifold
        self.Y = Y
        self.B = B

    def _eval(self, xx):
        return 0.5 * (sparse.linalg.norm(self.B.multiply(self.manifold._reprToPoint(xx) - self.Y)) ** 2)

    def _euclideanGradient(self, xx):
        return self.B.multiply(self.manifold._reprToPoint(xx) - self.Y)

    def _euclideanHessian(self, xx, hh):
        return self.B.multiply(self.manifold._reprToPointTangent(xx, hh))
