from .manifold import Manifold
import numpy as np
from numpy.linalg import svd, qr
from sklearn.utils.extmath import randomized_svd
from scipy.stats import ortho_group

class FixedRank(Manifold):

    def __init__(self, m, n, r):
        self.m = m
        self.n = n
        self.r = r

    def _reprToPoint(self, xx):
        return xx[0] @ xx[1] @ xx[2].T

    def _reprToPointTangent(self, xx, hh):
        U, Sigma, V = xx; U_p, M, V_p = hh
        return (U @ M @ V.T) + (U_p @ V.T) + (U @ V_p.T)

    def _inner(self, xx, yy):
        return np.sum(xx[0]*yy[0]) + np.sum(xx[1]*yy[1]) + np.sum(xx[2]*yy[2])

    def _norm(self, xx):
        return np.sqrt(self._inner(xx, xx))

    def _retract(self, xx, hh):
        U, Sigma, V = xx; U_p, M, V_p = hh
        r = U.shape[1]
        Q_U, R_U = qr(np.concatenate([U, U_p], axis=1))
        Q_V, R_V = qr(np.concatenate([V, V_p], axis=1))
        block = np.concatenate([np.concatenate([Sigma+M, np.eye(r)], axis=1), \
                                np.concatenate([np.eye(r), np.zeros((r, r))], axis=1)], \
                                axis=0)
        U_tilde, Sigma_tilde, Vh_tilde = randomized_svd(R_U @ block @ R_V.T, n_components = r)
        return Q_U@U_tilde, np.diag(Sigma_tilde), Q_V@Vh_tilde.T

    def _project(self, xx, zz):
        U, Sigma, V = xx
        M = U.T @ zz @ V
        U_p = (Z@V) - (U@M)
        V_p = (Z.T@U) - (V@M.T)
        return U_p, M, V_p

    def _riemannianGradient(self, euclideanGradient):
        return lambda xx: self._project(xx, euclideanGradient(xx))

    def _riemannianHessian(self, euclideanGradient, euclideanHessian):

        def rH(xx, hh):
            U, Sigma, V = xx; U_p, M, V_p = hh
            Sigma_inv = np.diag(1 / np.diag(Sigma))
            Z = euclideanGradient(xx); Zdot = euclideanHessian(xx, hh)
            Pu_orth = np.eye(self.m) - (U @ U.T)
            Pv_orth = np.eye(self.n) - (V @ V.T)

            M_tilde = U.T @ Zdot @ V
            Up_tilde = Pu_orth.T @ ((Zdot @ V) + (Z @ V_p @ Sigma_inv))
            Vp_tilde = Pv_orth.T @ ((Zdot.T @ U) + (Z.T @ U_p @ Sigma_inv))

            return Up_tilde, M_tilde, Vp_tilde

        return rH

    def _randomPoint(self):
        U = ortho_group.rvs(self.m); V = ortho_group.rvs(self.n)
        Sigma = np.zeros((self.m, self.n))
        di = np.diag_indices(self.r)
        Sigma[di] = np.sort(np.random.rand(self.r))[::-1]
        return U, Sigma, V
