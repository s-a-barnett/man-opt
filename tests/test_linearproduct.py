import pytest
import numpy as np
import sys
sys.path.append('../')

from src.manifolds.linearproduct import LinearProduct

@pytest.fixture
def linearproduct():
    m = 200; n = 100; r = 5;
    return LinearProduct(m, n, r)

class TestLinearProduct:

    def test_reprToPoint(self, linearproduct):
        m = linearproduct.m; n = linearproduct.n; r = linearproduct.r;

        L = np.ones((m, r)); R = np.zeros((n, r))

        assert np.allclose(linearproduct._reprToPoint((L, R)), np.zeros((m, n)))

    def test_inner(self, linearproduct):
        m = linearproduct.m; n = linearproduct.n; r = linearproduct.r;

        L = np.ones((m, r)); R = np.ones((n, r))
        xx = (L, R); yy = xx;

        assert linearproduct._inner(xx, yy) == r*(m+n)

    def test_retract(self, linearproduct):
        m = linearproduct.m; n = linearproduct.n; r = linearproduct.r;

        L = np.ones((m, r)); R = np.ones((n, r))
        xx = (L, R); yy = xx;

        assert np.allclose(linearproduct._retract(xx, yy)[0], 2*np.ones((m, r)))
        assert np.allclose(linearproduct._retract(xx, yy)[1], 2*np.ones((n, r)))
