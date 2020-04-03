import pytest
import numpy as np
import scipy.sparse as sparse
import scipy.stats as stats
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sys.path.append('../')

from src.manifolds.fixedrank import FixedRank
from src.costs.mle import MLE

@pytest.fixture
def fixedrank():
    m = 10; n = 20; r = 5;
    return FixedRank(m, n, r)

@pytest.fixture
def mle(fixedrank):
    Y = np.random.randn(fixedrank.m, fixedrank.n)
    B = sparse.csr_matrix(np.random.choice(2, (fixedrank.m, fixedrank.n)))
    return MLE(fixedrank, Y, B)

class TestMLE:

    def test_eval(self, fixedrank, mle):
        X = fixedrank._randomPoint()
        fX = mle._eval(X)
        xx = fixedrank._reprToPoint(X)
        assert np.isclose(fX, 0.5 * mle.B.multiply(xx - mle.Y).power(2).sum())

    def test_gradientShape(self, fixedrank, mle):
        X = fixedrank._randomPoint()

        grad = mle._euclideanGradient(X)
        assert grad.shape == (fixedrank.m, fixedrank.n)

    def test_hessianShape(self, fixedrank, mle):
        X = fixedrank._randomPoint()
        H = fixedrank._zeroTangent()

        hess = mle._euclideanHessian(X, H)
        assert hess.shape == (fixedrank.m, fixedrank.n)

    def test_riemannianGradient(self, fixedrank, mle):
        rmnGrad = fixedrank._riemannianGradient(mle._euclideanGradient)
        X = fixedrank._randomPoint()
        v = fixedrank._project(X, np.random.randn(fixedrank.m, fixedrank.n))
        v = fixedrank._multiplyTangent(1/fixedrank._norm(v), v)
        assert np.isclose(fixedrank._norm(v), 1.)
        fX = mle._eval(X)
        gradfX = rmnGrad(X)
        U, Sigma, V = X; U_p, M, V_p = v;
        assert np.allclose(U.T @ U_p, np.zeros((fixedrank.r, fixedrank.r)))
        assert np.allclose(V.T @ V_p, np.zeros((fixedrank.r, fixedrank.r)))
        inner = fixedrank._inner(gradfX, v)
        ts = np.logspace(-8, 0)
        Ets = []
        for t in ts:
            Et = np.abs(mle._eval(fixedrank._retract(X, fixedrank._multiplyTangent(t, v))) \
                        - fX - (t * inner))
            Ets.append(Et)

        data = pd.DataFrame(data={'x': ts, 'y': Ets})
        f, ax = plt.subplots(figsize=(7, 7))
        ax.set(xscale="log", yscale="log")
        sns.regplot("x", "y", data, ax=ax, scatter_kws={"s": 100})
        plt.plot([1e-6, 1e-3], [1e-8, 1e-2], linewidth=2.0)
        plt.title("gradient check: FixedRank")
        plt.xlabel("t")
        plt.ylabel("E(t)")
        plt.show()

    def test_riemannianHessian(self, fixedrank, mle):
        rmnGrad = fixedrank._riemannianGradient(mle._euclideanGradient)
        rmnHess = fixedrank._riemannianHessian(mle._euclideanGradient, mle._euclideanHessian)
        add = fixedrank._addTangent; multiply = fixedrank._multiplyTangent
        X = fixedrank._randomPoint()
        u = fixedrank._project(X, np.random.randn(fixedrank.m, fixedrank.n))
        v = fixedrank._project(X, np.random.randn(fixedrank.m, fixedrank.n))
        a = np.random.rand(); b = np.random.rand()
        lincomb = add(multiply(a, u), multiply(b, v))
        Hess_lincomb = rmnHess(X, lincomb)
        lincomb_Hess = add(multiply(a, rmnHess(X, u)), multiply(b, rmnHess(X, v)))
        assert np.allclose(fixedrank._reprToPointTangent(X, Hess_lincomb), fixedrank._reprToPointTangent(X, lincomb_Hess))
        inner_0 = fixedrank._inner(u, rmnHess(X, v))
        inner_1 = fixedrank._inner(rmnHess(X, u), v)
        assert np.isclose(inner_0, inner_1)

        v = fixedrank._multiplyTangent(1/fixedrank._norm(v), v)
        fX = mle._eval(X)
        gradfX = rmnGrad(X)
        HessfXv = rmnHess(X, v)
        inner_grad = fixedrank._inner(gradfX, v)
        inner_Hess = fixedrank._inner(HessfXv, v)
        ts = np.logspace(-8, 0)
        Ets = []
        for t in ts:
            Et = np.abs(mle._eval(fixedrank._retract(X, fixedrank._multiplyTangent(t, v))) \
                        - fX - (t * inner_grad) - (0.5 * (t**2) * inner_Hess))
            Ets.append(Et)

        data = pd.DataFrame(data={'x': ts, 'y': Ets})
        f, ax = plt.subplots(figsize=(7, 7))
        ax.set(xscale="log", yscale="log")
        sns.regplot("x", "y", data, ax=ax, scatter_kws={"s": 100})
        plt.plot([1e-6, 1e-3], [1e-8, 1e1], linewidth=2.0)
        plt.title("Hessian check: FixedRank")
        plt.xlabel("t")
        plt.ylabel("E(t)")
        plt.show()
