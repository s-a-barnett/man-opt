import numpy as np
from numpy.linalg import norm
import scipy.sparse.linalg as splinalg

from .cost import Cost

class RegularizedMLE(Cost):

    def __init__(self, manifold, Y, B, mu=0.):
        if type(manifold).__name__ == 'FixedRank':
            raise ValueError('manifold must be LinearProduct or StiefelProduct')

        self.manifold = manifold
        self.Y = Y
        self.B = B
        self.mu = mu

        self.f = lambda X: 0.5 * (splinalg.norm((B.multiply(X - Y))) ** 2)
        self.gradf = lambda X: (B.multiply(X - Y))

    def _eval(self, xx):
        L, R = xx
        reg = (self.mu / 4) * (norm((L.T@L) - (R.T@R)) ** 2)
        return self.f(self.manifold._reprToPoint(xx)) + reg

    def _euclideanGradient(self, xx):
        L, R = xx
        grad = self.gradf(self.manifold._reprToPoint(xx))
        left = (grad@R) + (L@((L.T@L) - (R.T@R)))*(self.mu)
        right = (grad.T@L) - (R@((L.T@L) - (R.T@R)))*(self.mu)
        return left, right

    def _euclideanHessian(self, xx, hh):
        L, R = xx; U_L, U_R = hh

        left = ((self.B.multiply(U_L@R.T)) @ R) + \
               ((self.B.multiply(L@U_R.T)) @ R) + \
               ((self.B.multiply((L@R.T) - self.Y)) @ U_R) + \
               ((U_L @ ((L.T @ L) - (R.T @ R)))*(self.mu)) + \
               ((L @ ((U_L.T@L) + (L.T@U_L) - (R.T@U_R) - (U_R.T@R)))*(self.mu))

        right = ((self.B.multiply(L@U_R.T)).T @ L) + \
                ((self.B.multiply(U_L@R.T)).T @ L) + \
                ((self.B.multiply((L@R.T) - self.Y)).T @ U_L) + \
                ((U_R @ ((R.T @ R) - (L.T @ L)))*(self.mu)) + \
                ((R @ ((U_R.T@R) + (R.T@U_R) - (U_L.T@L) - (L.T@U_L)))*(self.mu))

        return left, right
