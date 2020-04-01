from src.manifolds.manifold_factory import manifoldFactory
from src.costs.cost_factory import costFactory
from src.solvers.solver_factory import solverFactory

import pdb

import argparse

import numpy as np
import scipy.sparse as sparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# auxiliary function to find row-column positions
def ind2sub(shape, ind):
    rows = (ind.astype('int') / shape[1])
    cols = (ind.astype('int') % shape[1])
    return (rows, cols)

def main(args):
    # target matrix size (m x n), rank r
    m = args.m;
    n = args.n;
    r = args.r;

    # target matrix defined by Al*Ar'
    Al = np.random.randn(m, r);
    Ar = np.random.randn(n, r);

    Y = Al @ Ar.T

    osf = args.osf;  # Oversampling factor
    k = osf*r*(m+n-r);  # We observe about k entries of the target matrix.

    # sample positions
    ind = np.unique(np.random.randint(0,n*m-1,k))
    k = len(ind); # redefine k in case there were repetitions

    # I and J below are vectors of length k such that
    # (I(i), J(i)) is the row-column position of an observed
    # entry of the target matrix: this defines Omega.
    # B is the sparse binary matrix whose support is Omega
    [I, J] = ind2sub([m, n], ind)
    B = sparse.coo_matrix((np.ones(len(I)), (I, J)),shape=(m, n)).tocsr()

    mu = args.mu # regularization parameter

    manifold = manifoldFactory(args.manifold, m, n, r)

    if args.manifold == 'fixedrank':
        cost = costFactory('mle', manifold, Y, B)
    elif args.manifold == 'stiefelproduct':
        cost = costFactory('regularized_mle', manifold, Y, B, mu=0.)
    else:
        cost = costFactory('regularized_mle', manifold, Y, B, mu=args.mu)

    solver = solverFactory(args.solver, manifold, cost, maxiter=args.maxiter, verbose=True)

    xx_final, costs, grads, time = solver.solve()
    print('total time to compute: {} seconds'.format(time))
    print(grads[-1])
    print(np.min(grads))
    print(np.min(costs))

    df_costs = pd.DataFrame(dict(time=np.arange(len(costs)),
                            value=costs))
    df_grads = pd.DataFrame(dict(time=np.arange(len(grads)),
                            value=grads))

    plt.figure()
    g = sns.relplot(x="time", y="value", kind="line", data=df_costs)
    g.fig.autofmt_xdate()
    g.savefig("costs.png")
    plt.figure()
    g = sns.relplot(x="time", y="value", kind="line", data=df_grads)
    g.fig.autofmt_xdate()
    g.savefig("grads.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", default=200, type=int)
    parser.add_argument("-n", default=500, type=int)
    parser.add_argument("-r", default=5, type=int)
    parser.add_argument("--osf", default=6, type=int)
    parser.add_argument('--manifold', default='fixedrank', type=str)
    parser.add_argument("--solver", default='rtr', type=str)
    parser.add_argument("--maxiter", default=1000, type=int)
    parser.add_argument("--mu", default=1e2, type=float)

    args = parser.parse_args()

    main(args)
