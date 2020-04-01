import pytest
import numpy as np
import scipy.sparse as sparse
import sys
sys.path.append('../')

from src.manifolds.linearproduct import LinearProduct
from src.costs.regularized_mle import RegularizedMLE

@pytest.fixture
def linearproduct():
    m = 10; n = 20; r = 5;
    return LinearProduct(m, n, r)

@pytest.fixture
def regularizedmle(linearproduct):
    Y = np.random.randn(linearproduct.m, linearproduct.n)
    B = sparse.csr_matrix(np.random.choice(2, (linearproduct.m, linearproduct.n)))
    mu = 10.
    return RegularizedMLE(linearproduct, Y, B, mu=mu)

class TestRegularizedMLE:

    def test_eval(self, linearproduct, regularizedmle):
        X = (np.random.randn(linearproduct.m, linearproduct.r), np.random.randn(linearproduct.n, linearproduct.r))
        fX = regularizedmle._eval(X)

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
