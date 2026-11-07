"""Check of (A_n) over all permutations. A_c(sigma) = max E[X.C], X uniform on the square (m x m grid), C uniform on the
n points ((i+1/2)/n, (sigma_i+1/2)/n). Smoothed semi-discrete dual in n weights psi, minimised by L-BFGS:
f(psi) = eps*mean_x logsumexp((S-psi)/eps) + mean(psi), grad = 1/n - mean softmax. Primal value from the softmax plan.
Same m and eps for every sigma, so bias cancels in A_c(sigma) - A_c(id). Usage: python a_enum.py [m] [eps] [nmax]"""
import sys, itertools
import numpy as np
from scipy.special import logsumexp, softmax
from scipy.optimize import minimize
m = int(sys.argv[1]) if len(sys.argv) > 1 else 100
eps = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
nmax = int(sys.argv[3]) if len(sys.argv) > 3 else 6
g = (np.arange(m) + .5) / m
XA, XB = np.meshgrid(g, g, indexing="ij"); X = np.stack([XA.ravel(), XB.ravel()], 1)

def Ac(sig):
    n = len(sig); c = (np.arange(n) + .5) / n
    S = X @ np.stack([c, c[list(sig)]], 1).T
    def f(psi):
        Z = (S - psi) / eps
        P = softmax(Z, axis=1)
        return eps * logsumexp(Z, axis=1).mean() + psi.mean(), 1.0 / n - P.mean(0)
    r = minimize(f, np.zeros(n), jac=True, method="L-BFGS-B", options={"gtol": 1e-10, "ftol": 1e-15, "maxiter": 2000})
    P = softmax((S - r.x) / eps, axis=1)
    return float((P * S).sum(1).mean()), float(np.abs(P.mean(0) - 1.0 / n).max())

lines = []
for n in range(2, nmax + 1):
    a0, e0 = Ac(list(range(n)))
    vals = {s: Ac(s) for s in itertools.permutations(range(n))}
    worst = min(vals, key=lambda s: vals[s][0])
    neg = sum(1 for v in vals.values() if v[0] - a0 < -1e-6)
    merr = max(v[1] for v in vals.values())
    t = "n=%d id %.6f  min over %d perms %.6f at %s (d %+.6f)  #below id %d  max marginal err %.1e" % (
        n, a0, len(vals), vals[worst][0], worst, vals[worst][0] - a0, neg, merr)
    lines.append(t); print(t, flush=True)
    open(__file__.replace(".py", ".out"), "w", newline="\n").write("\n".join(lines) + "\n")
