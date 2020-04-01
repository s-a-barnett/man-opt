from .rtr import RiemannianTrustRegions
from .rgd import RiemannianGradientDescent

def solverFactory(solver_str, manifold, cost, *args, **kwargs):
    if (solver_str.lower() == 'rgd') or (solver_str.lower() == 'riemanniangradientdescent'):
        return RiemannianGradientDescent(manifold, cost, **kwargs)
    elif (solver_str.lower() == 'rtr') or (solver_str.lower() == 'riemanniantrustregions'):
        return RiemannianTrustRegions(manifold, cost, **kwargs)
    else:
        raise ValueError('solver must be "rgd" or "rtr"')
