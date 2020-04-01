import pytest
import numpy as np
import scipy.sparse as sparse
import sys
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

    def test_gradientShape(self, fixedrank, mle):
        X = fixedrank._randomPoint()

        grad = mle._euclideanGradient(X)
        assert grad.shape == (fixedrank.m, fixedrank.n)

    def test_hessianShape(self, fixedrank, mle):
        X = fixedrank._randomPoint()
        H = fixedrank._zeroTangent()

        hess = mle._euclideanHessian(X, H)
        assert hess.shape == (fixedrank.m, fixedrank.n)
