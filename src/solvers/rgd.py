import numpy as np
import progressbar
from .solver import Solver
import time
import pdb

class RiemannianGradientDescent(Solver):

    def __init__(self, manifold, cost, initGuess='random', maxiter=1000,
                 timeiter=None, verbose=False, tau=0.5, r=1e-4):
        self.manifold = manifold
        self.cost = cost
        self.rmnGrad = self.manifold._riemannianGradient(self.cost._euclideanGradient)
        self.rmnHess = self.manifold._riemannianHessian(self.cost._euclideanGradient, self.cost._euclideanHessian)

        self.maxiter = maxiter
        self.verbose = verbose
        self.tau = tau
        self.r = r

        if initGuess == 'random':
            self.initGuess = manifold._randomPoint()
        else:
            self.initGuess = initGuess

        if timeiter == None:
            self.timeiter = maxiter
        else:
            self.timeiter = timeiter

    def _step(self, xx):
        multiply = self.manifold._multiplyTangent
        grad = self.rmnGrad(xx)
        fxx = self.cost._eval(xx)
        alpha = self._getStepSize(xx, fxx, grad)
        xx_next = self.manifold._retract(xx, multiply(-alpha, grad))
        if self.verbose == False:
            return xx_next
        else:
            return xx_next, fxx, self.manifold._norm(grad)

    def _getStepSize(self, xx, fxx, gradfxx):
        # compute backtracking line search
        multiply = self.manifold._multiplyTangent
        norm_gradfxx = self.manifold._norm(gradfxx)
        alpha = 1 / norm_gradfxx # TODO: write dynamic alpha-bar
        lhs = fxx - self.cost._eval(self.manifold._retract(xx, multiply(-alpha, gradfxx)))
        rhs = self.r * alpha * (norm_gradfxx ** 2)
        while lhs < rhs:
            alpha *= self.tau
            lhs = fxx - self.cost._eval(self.manifold._retract(xx, multiply(-alpha, gradfxx)))
            rhs = self.r * alpha * (norm_gradfxx ** 2)
        return alpha


    def solve(self):
        xx = self.initGuess
        if self.verbose == False:
            for ii in progressbar.progressbar(range(self.maxiter)):
                xx = self._step(xx)

            return xx
        else:
            tic = time.time()
            costs = []
            grads = []
            for ii in progressbar.progressbar(range(self.maxiter)):
                xx, fxx, grad = self._step(xx)
                grads.append(grad)
                costs.append(fxx)
                if ii == self.timeiter-1:
                    toc = time.time()

            return xx, costs, grads, (toc - tic)
