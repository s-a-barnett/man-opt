import numpy as np
from numpy.linalg import norm

class RegularizedMLE(Cost):

    def __init__(self, manifold, Y, B, mu=0.):
        if type(manifold).__name__ == 'FixedRank':
            raise ValueError('manifold must be LinearProduct or StiefelProduct')

        self.manifold = manifold
        self.Y = Y
        self.B = B
        self.mu = mu

        self.f = lambda X: 0.5 * (norm((X - Y) * B) ** 2)
        self.gradf = lambda X: ((X - Y) * B)

    def _eval(self, xx):
        L, R = xx
        reg = (self.mu / 4) * (norm((L.T@L) - (R.T@R)) ** 2)
        return self.f(self.manifold._reprToPoint(xx)) + reg

    def _euclideanGradient(self, xx):
        L, R = xx
        grad = self.gradf(self.manifold._reprToPoint(xx))
        left = (grad@R) + (self.mu * (L@((L.T@L) - (R.T@R))))
        right = (grad.T@L) - (self.mu * (R@((L.T@L) - (R.T@R))))
        return left, right

    def _euclideanHessian(self, xx, hh):
        L, R = xx; U_L, U_R = hh
        left = ((L@U_R.T) * self.B).T @ L + \
               self.mu * U_R @ ((R.T @ R) - (L.T @ L)) + \
               self.mu * R @ ((U_R.T@R) + (R.T@U_R))
        right = ((U_L@R.T) * self.B) @ R + \
               self.mu * U_L @ ((L.T @ L) - (R.T @ R)) + \
               self.mu * L @ ((U_L.T@L) + (L.T@U_L))

        return left, right
