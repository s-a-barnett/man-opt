from .mle import MLE
from .regularized_mle import RegularizedMLE

def costFactory(cost_str, manifold, Y, B, *args, **kwargs):
    if cost_str.lower() == 'mle':
        return MLE(manifold, Y, B)
    elif cost_str.lower() == 'regularized_mle':
        return RegularizedMLE(manifold, Y, B, mu=kwargs['mu'])
    else:
        raise ValueError('cost must be "mle" or "regularized_mle"')
