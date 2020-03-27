from .manifold import Manifold
import numpy as np
from numpy.linalg import svd, qr

# helper function to compute symmetric part of matrix X
def sym(X):
    return (X + X.T) / 2

class StiefelProduct(Manifold):

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
        _, s, vh = svd(xx[0] + uu[0])
        s_inv = np.diag(1 / s)
        r_0 = (xx[0] + uu[0]) @ (vh.T @ s_inv @ vh)
        r_1 = xx[1] + uu[1]
        return r_0, r_1

    def _project(self, xx, zz):
        p_0 = zz[0] - (xx[0] @ sym(xx[0].T @ zz[0]))
        return p_0, zz[1]

    def _riemannianGradient(self, euclideanGradient):
        return lambda xx: self._project(xx, euclideanGradient(xx))

    def _riemannianHessian(self, euclideanGradient, euclideanHessian):

        def rH(xx, vv):
            r_0 = self._project(xx, euclideanHessian(xx, vv))[0] - (vv[0] @ sym(xx[0].T @ euclideanGradient(xx)[0]))
            r_1 = euclideanHessian(xx, vv)[1]
            return r_0, r_1

        return rH

    def _randomPoint(self):
        a = np.random.randn(self.m, self.r)
        q, _ = qr(a)
        return q, np.random.randn(self.n, self.r)
