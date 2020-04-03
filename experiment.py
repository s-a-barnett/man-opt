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

    fixedrank = manifoldFactory('fixedrank', m, n, r)
    linear = manifoldFactory('linearproduct', m, n, r)
    stiefel = manifoldFactory('stiefelproduct', m, n, r)

    # ensure same initial guess
    initGuess_fixedrank = fixedrank._randomPoint(scale=args.scale)
    initGuess_product = (initGuess_fixedrank[0], initGuess_fixedrank[2]@initGuess_fixedrank[1])

    cost_fixedrank = costFactory('mle', fixedrank, Y, B)
    cost_linear = costFactory('regularized_mle', linear, Y, B, mu=args.mu)
    cost_stiefel = costFactory('regularized_mle', stiefel, Y, B, mu=0)

    rtr_fixedrank = solverFactory('rtr', fixedrank, cost_fixedrank, maxiter=args.maxiter, verbose=True, initGuess=initGuess_fixedrank)
    rtr_linear = solverFactory('rtr', linear, cost_linear, maxiter=args.maxiter, verbose=True, initGuess=initGuess_product)
    rtr_stiefel = solverFactory('rtr', stiefel, cost_stiefel, maxiter=args.maxiter, verbose=True, initGuess=initGuess_product)
    rgd_fixedrank = solverFactory('rgd', fixedrank, cost_fixedrank, maxiter=args.maxiter, verbose=True, initGuess=initGuess_fixedrank)
    rgd_linear = solverFactory('rgd', linear, cost_linear, maxiter=args.maxiter, verbose=True, initGuess=initGuess_product)
    rgd_stiefel = solverFactory('rgd', stiefel, cost_stiefel, maxiter=args.maxiter, verbose=True, initGuess=initGuess_product)

    df_list = []
    for solver in [rtr_linear, rtr_stiefel, rtr_fixedrank, rgd_linear, rgd_stiefel, rgd_fixedrank]:
        xx_final, costs, grads, time = solver.solve()
        df = pd.DataFrame(dict(num_iterations=np.arange(len(costs)),
                                cost=costs, solver=type(solver).__name__,
                                manifold=type(solver.manifold).__name__,
                                grad=grads))
        df_list.append(df)

    df = pd.concat(df_list)
    plt.figure()
    g = sns.relplot(x="num_iterations", y="cost", hue="manifold", col="solver", kind="line", data=df)
    plt.title("(m, n, r) = ({}, {}, {})".format(m, n, r))
    g.fig.autofmt_xdate()
    g.savefig("figures/costs_m{}n{}r{}.png".format(m, n, r))

    plt.figure()
    g = sns.relplot(x="num_iterations", y="grad", hue="manifold", col="solver", kind="line", data=df)
    plt.ylabel("|grad f|")
    plt.title("(m, n, r) = ({}, {}, {})".format(m, n, r))
    g.fig.autofmt_xdate()
    g.savefig("figures/grads_m{}n{}r{}.png".format(m, n, r))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", default=200, type=int)
    parser.add_argument("-n", default=500, type=int)
    parser.add_argument("-r", default=5, type=int)
    parser.add_argument("--osf", default=6, type=int)
    parser.add_argument("--maxiter", default=500, type=int)
    parser.add_argument("--mu", default=1, type=float)
    parser.add_argument("--scale", default=1e-2, type=float)

    args = parser.parse_args()

    main(args)
