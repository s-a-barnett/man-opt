import pytest
import numpy as np
import sys
sys.path.append('../')

from src.manifolds.stiefelproduct import StiefelProduct

@pytest.fixture
def stiefelproduct():
    m = 10; n = 10; r = 10;
    return StiefelProduct(m, n, r)

class TestStiefelProduct:

    def test_reprToPoint(self, stiefelproduct):
        m = stiefelproduct.m; n = stiefelproduct.n; r = stiefelproduct.r;

        L = np.ones((m, r)); R = np.zeros((n, r))

        assert np.allclose(stiefelproduct._reprToPoint((L, R)), np.zeros((m, n)))

    def test_inner(self, stiefelproduct):
        m = stiefelproduct.m; n = stiefelproduct.n; r = stiefelproduct.r;

        L = np.ones((m, r)); R = np.ones((n, r))
        xx = (L, R); yy = xx;

        assert stiefelproduct._inner(xx, yy) == r*(m+n)

    def test_retract(self, stiefelproduct):
        m = stiefelproduct.m
        xx = (0.25*np.eye(m), np.zeros((m, m)))
        yy = (np.zeros((m, m)), np.zeros((m, m)))
        rr = stiefelproduct._retract(xx, yy)
        assert np.allclose(rr[0], np.eye(m))

    def test_retractZero(self, stiefelproduct):
        X = stiefelproduct._randomPoint()
        Z = stiefelproduct._zeroTangent()
        R = stiefelproduct._retract(X, Z)
        assert np.allclose(stiefelproduct._reprToPoint(X), stiefelproduct._reprToPoint(R))

    def test_project(self, stiefelproduct):
        m = stiefelproduct.m
        Z = np.random.randn(m, m); X = np.random.randn(m, m);
        rr = stiefelproduct._project((X, X), (Z, Z))
        r_0 = Z - X @ ((X.T @ Z + Z.T @ X) / 2)
        assert np.allclose(rr[0], r_0)

    def test_randomPoint(self, stiefelproduct):
        x, _ = stiefelproduct._randomPoint()
        assert np.allclose(x.T @ x, np.eye(stiefelproduct.r))
