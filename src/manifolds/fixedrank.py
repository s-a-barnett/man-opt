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
        self.dim = r * (m + n - r)

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
        U_tilde, Sigma_tilde, Vh_tilde = svd(R_U @ block @ R_V.T)
        U_tilde = U_tilde[:, :r]
        Sigma_tilde = Sigma_tilde[:r]
        Vh_tilde = Vh_tilde[:r, :]
        return Q_U@U_tilde, np.diag(Sigma_tilde), Q_V@Vh_tilde.T

    def _project(self, xx, zz):
        U, Sigma, V = xx
        M = U.T @ zz @ V
        U_p = (zz@V) - (U@M)
        V_p = (zz.T@U) - (V@M.T)
        return U_p, M, V_p

    def _riemannianGradient(self, euclideanGradient):
        return lambda xx: self._project(xx, euclideanGradient(xx))

    def _riemannianHessian(self, euclideanGradient, euclideanHessian):

        def rH(xx, hh):
            U, Sigma, V = xx; U_p, M, V_p = hh
            Sigma_inv = np.diag(np.divide(1, np.diag(Sigma), \
                                out=np.zeros_like(np.diag(Sigma)), \
                                where=np.diag(Sigma)!=0))
            Z = euclideanGradient(xx); Zdot = euclideanHessian(xx, hh)
            Pu_orth = np.eye(self.m) - (U @ U.T)
            Pv_orth = np.eye(self.n) - (V @ V.T)

            M_tilde = U.T @ Zdot @ V
            Up_tilde = Pu_orth.T @ ((Zdot @ V) + (Z @ V_p @ Sigma_inv))
            Vp_tilde = Pv_orth.T @ ((Zdot.T @ U) + (Z.T @ U_p @ Sigma_inv))

            return Up_tilde, M_tilde, Vp_tilde

        return rH

    def _randomPoint(self, scale=1.):
        a = np.random.randn(self.m, self.r); b = np.random.randn(self.n, self.r)
        U, _ = qr(a); V, _ = qr(b)
        Sigma = scale * np.diag(np.random.randn(self.r))
        return U, Sigma, V

    def _zeroTangent(self):
        return np.zeros((self.m, self.r)), np.zeros((self.r, self.r)), np.zeros((self.n, self.r))

    def _addTangent(self, ss, tt):
        U_p0, M0, V_p0 = ss; U_p1, M1, V_p1 = tt;
        return U_p0 + U_p1, M0 + M1, V_p0 + V_p1

    def _multiplyTangent(self, alpha, ss):
        U_p, M, V_p = ss
        return alpha * U_p, alpha * M, alpha * V_p
