import pytest
import numpy as np
import scipy.sparse as sparse
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sys.path.append('../')

from src.manifolds.linearproduct import LinearProduct
from src.manifolds.stiefelproduct import StiefelProduct
from src.costs.regularized_mle import RegularizedMLE

@pytest.fixture
def linearproduct():
    m = 10; n = 20; r = 5;
    return LinearProduct(m, n, r)

@pytest.fixture
def stiefelproduct(linearproduct):
    m = linearproduct.m; n = linearproduct.n; r = linearproduct.r
    return StiefelProduct(m, n, r)


@pytest.fixture
def regularizedmle(linearproduct):
    Y = np.random.randn(linearproduct.m, linearproduct.n)
    B = sparse.csr_matrix(np.random.choice(2, (linearproduct.m, linearproduct.n)))
    mu = 10.
    return RegularizedMLE(linearproduct, Y, B, mu=mu)

@pytest.fixture
def regularizedmle_stiefel(stiefelproduct):
    Y = np.random.randn(stiefelproduct.m, stiefelproduct.n)
    B = sparse.csr_matrix(np.random.choice(2, (stiefelproduct.m, stiefelproduct.n)))
    return RegularizedMLE(stiefelproduct, Y, B)

class TestRegularizedMLE:

    def test_eval(self, linearproduct, regularizedmle):
        X = (np.random.randn(linearproduct.m, linearproduct.r), np.random.randn(linearproduct.n, linearproduct.r))
        fX = regularizedmle._eval(X)
        xx = linearproduct._reprToPoint(X)
        L, R = X
        reg = (regularizedmle.mu / 4) * np.sum(np.square(((L.T@L) - (R.T@R))))
        assert np.isclose(fX, 0.5 * regularizedmle.B.multiply(xx - regularizedmle.Y).power(2).sum() + reg)

    def test_gradientShape(self, linearproduct, regularizedmle):
        X = (np.random.randn(linearproduct.m, linearproduct.r), np.random.randn(linearproduct.n, linearproduct.r))

        grad = regularizedmle._euclideanGradient(X)
        assert grad[0].shape == (linearproduct.m, linearproduct.r)
        assert grad[1].shape == (linearproduct.n, linearproduct.r)

    def test_hessianShape(self, linearproduct, regularizedmle):
        X = (np.random.randn(linearproduct.m, linearproduct.r), np.random.randn(linearproduct.n, linearproduct.r))
        H = (np.random.randn(linearproduct.m, linearproduct.r), np.random.randn(linearproduct.n, linearproduct.r))

        hess = regularizedmle._euclideanHessian(X, H)
        assert hess[0].shape == (linearproduct.m, linearproduct.r)
        assert hess[1].shape == (linearproduct.n, linearproduct.r)

    def test_riemannianGradient(self, linearproduct, regularizedmle):
        rmnGrad = linearproduct._riemannianGradient(regularizedmle._euclideanGradient)
        X = linearproduct._randomPoint()
        v = (np.random.randn(linearproduct.m, linearproduct.r), np.random.randn(linearproduct.n, linearproduct.r))
        v = linearproduct._multiplyTangent(1/linearproduct._norm(v), v)
        assert np.isclose(linearproduct._norm(v), 1.)
        fX = regularizedmle._eval(X)
        gradfX = rmnGrad(X)
        inner = linearproduct._inner(gradfX, v)
        ts = np.logspace(-8, 0)
        Ets = []
        for t in ts:
            Et = np.abs(regularizedmle._eval(linearproduct._retract(X, linearproduct._multiplyTangent(t, v))) \
                        - fX - (t * inner))
            Ets.append(Et)

        data = pd.DataFrame(data={'x': ts, 'y': Ets})
        f, ax = plt.subplots(figsize=(7, 7))
        ax.set(xscale="log", yscale="log")
        sns.regplot("x", "y", data, ax=ax, scatter_kws={"s": 100})
        plt.plot([1e-6, 1e-3], [1e-8, 1e-2], linewidth=2.0)
        plt.title("gradient check: LinearProduct")
        plt.xlabel("t")
        plt.ylabel("E(t)")
        plt.show()

    def test_riemannianHessian(self, linearproduct, regularizedmle):
        rmnGrad = linearproduct._riemannianGradient(regularizedmle._euclideanGradient)
        rmnHess = linearproduct._riemannianHessian(regularizedmle._euclideanGradient, regularizedmle._euclideanHessian)
        add = linearproduct._addTangent; multiply = linearproduct._multiplyTangent
        X = linearproduct._randomPoint()
        u = (np.random.randn(linearproduct.m, linearproduct.r), np.random.randn(linearproduct.n, linearproduct.r))
        v = (np.random.randn(linearproduct.m, linearproduct.r), np.random.randn(linearproduct.n, linearproduct.r))
        a = np.random.rand(); b = np.random.rand()
        lincomb = add(multiply(a, u), multiply(b, v))
        Hess_lincomb = rmnHess(X, lincomb)
        lincomb_Hess = add(multiply(a, rmnHess(X, u)), multiply(b, rmnHess(X, v)))
        assert np.allclose(Hess_lincomb[0], lincomb_Hess[0])
        assert np.allclose(Hess_lincomb[1], lincomb_Hess[1])
        inner_0 = linearproduct._inner(u, rmnHess(X, v))
        inner_1 = linearproduct._inner(rmnHess(X, u), v)
        assert np.isclose(inner_0, inner_1)

        v = linearproduct._multiplyTangent(1/linearproduct._norm(v), v)
        assert np.isclose(linearproduct._norm(v), 1.0)
        fX = regularizedmle._eval(X)
        gradfX = rmnGrad(X)
        HessfXv = rmnHess(X, v)
        inner_grad = linearproduct._inner(gradfX, v)
        inner_Hess = linearproduct._inner(HessfXv, v)
        ts = np.logspace(-8, 0)
        Ets = []
        for t in ts:
            Et = np.abs(regularizedmle._eval(linearproduct._retract(X, linearproduct._multiplyTangent(t, v))) \
                        - fX - (t * inner_grad) - (0.5 * (t**2) * inner_Hess))
            Ets.append(Et)

        data = pd.DataFrame(data={'x': ts, 'y': Ets})
        f, ax = plt.subplots(figsize=(7, 7))
        ax.set(xscale="log", yscale="log")
        sns.regplot("x", "y", data, ax=ax, scatter_kws={"s": 100})
        plt.plot([1e-6, 1e-3], [1e-8, 1e1], linewidth=2.0)
        plt.title("Hessian check: LinearProduct")
        plt.xlabel("t")
        plt.ylabel("E(t)")
        plt.show()

    def test_riemannianGradientStiefel(self, stiefelproduct, regularizedmle):
        rmnGrad = stiefelproduct._riemannianGradient(regularizedmle._euclideanGradient)
        X = stiefelproduct._randomPoint()
        v = (np.random.randn(stiefelproduct.m, stiefelproduct.r), np.random.randn(stiefelproduct.n, stiefelproduct.r))
        v = stiefelproduct._project(X, v)
        v = stiefelproduct._multiplyTangent(1/stiefelproduct._norm(v), v)
        assert np.isclose(stiefelproduct._norm(v), 1.)
        fX = regularizedmle._eval(X)
        gradfX = rmnGrad(X)
        X_0, _ = X
        v_0, _ = v
        assert np.allclose((X_0.T @ v_0), -(v_0.T @ X_0))
        inner = stiefelproduct._inner(gradfX, v)
        ts = np.logspace(-8, 0)
        Ets = []
        for t in ts:
            Et = np.abs(regularizedmle._eval(stiefelproduct._retract(X, stiefelproduct._multiplyTangent(t, v))) \
                        - fX - (t * inner))
            Ets.append(Et)

        data = pd.DataFrame(data={'x': ts, 'y': Ets})
        f, ax = plt.subplots(figsize=(7, 7))
        ax.set(xscale="log", yscale="log")
        sns.regplot("x", "y", data, ax=ax, scatter_kws={"s": 100})
        plt.plot([1e-6, 1e-3], [1e-8, 1e-2], linewidth=2.0)
        plt.title("gradient check: StiefelProduct")
        plt.xlabel("t")
        plt.ylabel("E(t)")
        plt.show()

    def test_riemannianHessianStiefel(self, stiefelproduct, regularizedmle):
        rmnGrad = stiefelproduct._riemannianGradient(regularizedmle._euclideanGradient)
        rmnHess = stiefelproduct._riemannianHessian(regularizedmle._euclideanGradient, regularizedmle._euclideanHessian)
        add = stiefelproduct._addTangent; multiply = stiefelproduct._multiplyTangent
        X = stiefelproduct._randomPoint()
        u = (np.random.randn(stiefelproduct.m, stiefelproduct.r), np.random.randn(stiefelproduct.n, stiefelproduct.r))
        v = (np.random.randn(stiefelproduct.m, stiefelproduct.r), np.random.randn(stiefelproduct.n, stiefelproduct.r))
        u = stiefelproduct._project(X, u)
        v = stiefelproduct._project(X, v)
        a = np.random.rand(); b = np.random.rand()
        lincomb = add(multiply(a, u), multiply(b, v))
        Hess_lincomb = rmnHess(X, lincomb)
        lincomb_Hess = add(multiply(a, rmnHess(X, u)), multiply(b, rmnHess(X, v)))
        assert np.allclose(Hess_lincomb[0], lincomb_Hess[0])
        assert np.allclose(Hess_lincomb[1], lincomb_Hess[1])
        inner_0 = stiefelproduct._inner(u, rmnHess(X, v))
        inner_1 = stiefelproduct._inner(rmnHess(X, u), v)
        assert np.isclose(inner_0, inner_1)

        v = stiefelproduct._multiplyTangent(1/stiefelproduct._norm(v), v)
        assert np.isclose(stiefelproduct._norm(v), 1.0)
        fX = regularizedmle._eval(X)
        gradfX = rmnGrad(X)
        HessfXv = rmnHess(X, v)
        inner_grad = stiefelproduct._inner(gradfX, v)
        inner_Hess = stiefelproduct._inner(HessfXv, v)
        ts = np.logspace(-8, 0)
        Ets = []
        for t in ts:
            Et = np.abs(regularizedmle._eval(stiefelproduct._retract(X, stiefelproduct._multiplyTangent(t, v))) \
                        - fX - (t * inner_grad) - (0.5 * (t**2) * inner_Hess))
            Ets.append(Et)

        data = pd.DataFrame(data={'x': ts, 'y': Ets})
        f, ax = plt.subplots(figsize=(7, 7))
        ax.set(xscale="log", yscale="log")
        sns.regplot("x", "y", data, ax=ax, scatter_kws={"s": 100})
        plt.plot([1e-6, 1e-3], [1e-8, 1e1], linewidth=2.0)
        plt.title("Hessian check: StiefelProduct")
        plt.xlabel("t")
        plt.ylabel("E(t)")
        plt.show()
