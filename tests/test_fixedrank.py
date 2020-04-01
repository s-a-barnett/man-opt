import pytest
import numpy as np
import sys
sys.path.append('../')

from src.manifolds.fixedrank import FixedRank

@pytest.fixture
def fixedrank():
    m = 10; n = 10; r = 10;
    return FixedRank(m, n, r)

class TestFixedRank:

    def test_inner(self, fixedrank):
        X = fixedrank._randomPoint()
        U_p, V_p = np.zeros((fixedrank.m, fixedrank.r)), np.zeros((fixedrank.n, fixedrank.r))
        M_0, M_1 = np.random.randn(fixedrank.r, fixedrank.r), np.random.randn(fixedrank.r, fixedrank.r)
        tangent_0, tangent_1 = (U_p, M_0, V_p), (U_p, M_1, V_p)
        inner_0 = fixedrank._inner(tangent_0, tangent_1)
        inner_1 = np.sum(fixedrank._reprToPointTangent(X, tangent_0) * fixedrank._reprToPointTangent(X, tangent_1))
        assert np.isclose(inner_0, inner_1)

    def test_retract(self, fixedrank):
        # tests that retraction of random points is on manifold
        X = fixedrank._randomPoint()
        U_p, V_p = np.zeros((fixedrank.m, fixedrank.r)), np.zeros((fixedrank.n, fixedrank.r))
        M = np.random.randn(fixedrank.r, fixedrank.r)
        hh = (U_p, M, V_p)
        rr = fixedrank._reprToPoint(fixedrank._retract(X, hh))
        assert np.linalg.matrix_rank(rr) == fixedrank.r

    def test_retractZero(self, fixedrank):
        X = fixedrank._randomPoint()
        Z = fixedrank._zeroTangent()
        R = fixedrank._retract(X, Z)
        assert np.allclose(fixedrank._reprToPoint(X), fixedrank._reprToPoint(R))

    def test_reprToRandomPoint(self, fixedrank):
        X = fixedrank._reprToPoint(fixedrank._randomPoint())
        assert np.linalg.matrix_rank(X) == fixedrank.r

    def test_project(self, fixedrank):
        X = fixedrank._randomPoint()
        U, Sigma, V = X
        Z = np.random.randn(fixedrank.m, fixedrank.n)
        U_p, M, V_p = fixedrank._project(X, Z)
        assert U_p.shape == (fixedrank.m, fixedrank.r)
        assert M.shape == (fixedrank.r, fixedrank.r)
        assert V_p.shape == (fixedrank.n, fixedrank.r)
        assert np.allclose(U.T @ U_p, np.zeros((fixedrank.r, fixedrank.r)))
        assert np.allclose(V.T @ V_p, np.zeros((fixedrank.r, fixedrank.r)))
