from .manifold import Manifold
import numpy as np

class LinearProduct(Manifold):

    def __init__(self, m, n, r):
        self.m = m
        self.n = n
        self.r = r

    def _reprToPoint(self, xx):
        return xx[0] @ xx[1].T

    def _inner(self, xx, yy):
        return np.sum(xx[0]*yy[0]) + np.sum(xx[1]*yy[1])

    def _norm(self, xx):
        return np.sqrt(self._inner(xx, xx))

    def _retract(self, xx, uu):
        return xx[0] + uu[0], xx[1] + uu[1]

    def _project(self, xx, zz):
        return zz

    def _riemannianGradient(self, euclideanGradient):
        return euclideanGradient

    def _riemannianHessian(self, euclideanGradient, euclideanHessian):
        return euclideanHessian

    def _randomPoint(self):
        return np.random.randn(self.m, self.r), np.random.randn(self.n, self.r)
