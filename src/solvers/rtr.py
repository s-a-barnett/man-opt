import numpy as np
import progressbar
from .solver import Solver
import time

class RiemannianTrustRegions(Solver):

    def __init__(self, manifold, cost, initGuess='random', maxiter=1000,
                 timeiter=None, verbose=False, DeltaBar=None, Delta0=None,
                 rhoPrime=0.1, rhoReg=1e3, epsMac=1e-16, kappa=0.1, theta=1.,
                 maxiter_tCG=None, mingradnorm=1e-6):
        self.manifold = manifold
        self.cost = cost
        self.rmnGrad = self.manifold._riemannianGradient(self.cost._euclideanGradient)
        self.rmnHess = self.manifold._riemannianHessian(self.cost._euclideanGradient, self.cost._euclideanHessian)

        self.maxiter = maxiter
        self.verbose = verbose
        if DeltaBar == None:
            self.DeltaBar = np.sqrt(manifold.dim)
        else:
            self.DeltaBar = DeltaBar

        if Delta0 == None:
            self.Delta0 = self.DeltaBar / 8
        else:
            self.Delta0 = Delta0

        self.rhoPrime = rhoPrime
        self.rhoReg = rhoReg
        self.epsMac = epsMac
        self.kappa = kappa
        self.theta = theta
        self.mingradnorm = mingradnorm

        if maxiter_tCG == None:
            self.maxiter_tCG = maxiter
        else:
            self.maxiter_tCG = maxiter_tCG

        if initGuess == 'random':
            self.initGuess = manifold._randomPoint()
        else:
            self.initGuess = initGuess

        if timeiter == None:
            self.timeiter = maxiter
        else:
            self.timeiter = timeiter

    def _step(self, xx, Delta):
        grad = self.rmnGrad(xx)
        ss, Hss = self._tCG(xx, grad, Delta)
        xx_tent = self.manifold._retract(xx, ss)
        fxx = self.cost._eval(xx)
        rhoAdd = np.max([1., np.abs(fxx)]) * self.epsMac * self.rhoReg

        rho = fxx - self.cost._eval(xx_tent) + rhoAdd
        rho /= -self.manifold._inner(grad, ss) \
                 -(0.5 * self.manifold._inner(ss, Hss)) \
                 + rhoAdd

        if rho > self.rhoPrime:
            xx_next = xx_tent
        else:
            xx_next = xx

        if rho < 0.25:
            Delta_next = 0.25 * Delta
        elif rho > 0.75 and np.isclose(self.manifold._norm(ss), Delta):
            Delta_next = np.min([2 * Delta, DeltaBar])
        else:
            Delta_next = Delta


        if self.verbose == False:
            return xx_next, Delta_next
        else:
            return xx_next, Delta_next, fxx, self.manifold._norm(grad)

    def _tCG(self, xx, grad, Delta):
        # compute truncated Conjugate Gradients
        add = self.manifold._addTangent
        multiply = self.manifold._multiplyTangent
        b = grad
        v = self.manifold._zeroTangent(); r = b; p = r;
        r_0norm = self.manifold._norm(r)
        for _ in range(self.maxiter_tCG):
            Hp = self.rmnHess(xx, p)
            inner = self.manifold._inner(p, Hp)
            alpha = (self.manifold._norm(r) ** 2) / inner
            v_tent = add(v, multiply(alpha, p))

            if (inner <= 0) or (self.manifold._norm(v_tent) >= Delta):
                p_norm = self.manifold._norm(p)
                pv_inner = self.manifold._inner(p, v)
                v_norm = self.manifold._norm(v)
                t = -(pv_inner + np.sqrt((pv_inner ** 2) - p_norm*(v_norm**2 - Delta**2))) / p_norm
                v = add(v, multiply(t, p))
                return v, add(b, add(multiply(-1.0, r), multiply(t, Hp)))
            else:
                v = v_tent

            r_prevnorm = self.manifold._norm(r)
            r = add(r, multiply(-alpha, Hp))
            r_nextnorm = self.manifold._norm(r)
            if r_nextnorm <= r_0norm*np.min([r_0norm**self.theta, self.kappa]):
                return v, add(b, multiply(-1.0, r))

            beta = (r_nextnorm / r_prevnorm) ** 2
            p = add(r, multiply(beta, p))

        return v, add(b, multiply(-1.0, r))


    def solve(self):
        xx = self.initGuess
        Delta = self.Delta0
        if self.verbose == False:
            for ii in progressbar.progressbar(range(self.maxiter)):
                xx, Delta = self._step(xx, Delta)

            return xx
        else:
            tic = time.time()
            grads = []
            costs = []
            for ii in progressbar.progressbar(range(self.maxiter)):
                xx, Delta, fxx, grad = self._step(xx, Delta)
                grads.append(grad)
                costs.append(fxx)
                if grad < self.mingradnorm:
                    toc = time.time()
                    break
                if ii == self.timeiter-1:
                    toc = time.time()

            return xx, costs, grads, (toc - tic)
