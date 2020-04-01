from .linearproduct import LinearProduct
from .stiefelproduct import StiefelProduct
from .fixedrank import FixedRank

def manifoldFactory(manifold_str, m, n, r):
    if manifold_str.lower() == 'linearproduct':
        return LinearProduct(m, n, r)
    elif manifold_str.lower() == 'stiefelproduct':
        return StiefelProduct(m, n, r)
    elif manifold_str.lower() == 'fixedrank':
        return FixedRank(m, n, r)
    else:
        raise ValueError('manifold must be "linearproduct", "stiefelproduct", or "fixedrank"')
