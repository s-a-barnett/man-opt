import numpy as np
import progressbar
from .solver import Solver
import time

class RiemannianGradientDescent(Solver):

    def __init__(self, manifold, cost, initGuess='random', maxiter=1000,
                 timeiter=None, verbose=False, tau=0.8, r=1e-4):
        self.manifold = manifold
        self.cost = cost
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
        grad = self.cost._riemannianGradient()(xx)
        alpha = self._getStepSize(xx, grad)
        xx_next = self.manifold._retract(xx, -alpha*grad)
        if verbose == False:
            return xx_next
        else:
            return xx_next, grad

    def _getStepSize(self, xx, gradfxx):
        # compute backtracking line search
        alpha = 1.0 # TODO: write dynamic alpha-bar
        lhs = self.cost._eval(xx) - self.cost._eval(self.manifold._retract(xx, -alpha * gradfxx))
        rhs = self.r * self.alpha * (self.manifold._norm(gradfxx) ** 2)
        while lhs < rhs:
            alpha = self.tau * alpha
            lhs = self.cost._eval(xx) - self.cost._eval(self.manifold._retract(xx, -alpha * gradfxx))
            rhs = self.r * self.alpha * (self.manifold._norm(gradfxx) ** 2)
        return alpha


    def solve(self):
        xx = self.initGuess
        if verbose == False:
            for ii in progressbar.progressbar(range(self.maxiter)):
                xx = self._step(xx)

            return xx
        else:
            tic = time.time()
            grads = []
            for ii in progressbar.progressbar(range(self.maxiter)):
                xx, grad = self._step(xx)
                grads.append(grad)
                if ii == timeiter:
                    toc = time.time()

            return xx, grads, (toc - tic)
