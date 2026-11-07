"""Grid-resolution check (jittered grid: one uniform point per cell, so no ties at the cut lines) for gq_cross_formula_check.py: grid minus formula at grids of 500, 1000 and 2000 for a few bands.
A gap that halves when the grid doubles is grid error, not a formula error. Usage: gq_cross_formula_res.py."""
import numpy as np
from mpmath import mpf
from gq_scan import cuts
import gq_cross_iv as G

def grid_val(n, a, b, u, w):
    rng = np.random.default_rng(n); i = np.arange(n); X1, X2 = np.meshgrid(i, i, indexing="ij"); X1 = (X1 + rng.random(X1.shape)) / n; X2 = (X2 + rng.random(X2.shape)) / n; S0 = (X1 + X2).ravel(); V0 = (X1 - X2).ravel()
    k = (S0 > a) & (S0 < b); s, v = S0[k], V0[k]; del X1, X2, S0, V0
    qv = np.quantile(v, [0.2, 0.4, 0.6, 0.8]); qs = np.quantile(s, [0.2, 0.4, 0.6, 0.8])
    c = [v <= qv[0]] + [(v > qv[j - 1]) & (v <= qv[j]) for j in range(1, 4)] + [v > qv[3]]
    sj = [s <= qs[0]] + [(s > qs[j - 1]) & (s <= qs[j]) for j in range(1, 4)] + [s > qs[3]]
    val = sum(u[i] * s[c[i]].mean() + w[i] * v[c[i]].mean() for i in range(5))
    return val - sum((2 * j - 4) * s[sj[j]].mean() for j in range(5))

out = []
for tau, (a, b) in (("12430", (0.6, 1.5)), ("13402", (0.8, 1.6)), ("41230", (0.95, 1.2))):
    t = tuple(int(x) for x in tau); (u, w), C = cuts(t)[0], G.coeffs(t)[0]
    f, M = G.lower(mpf(a), mpf(a), mpf(2 - b), mpf(2 - b), [C]); fv = float((5 * f / M.a).a)
    gaps = [grid_val(n, a, b, u, w) - fv for n in (500, 1000, 2000)]
    out.append("tau=%s band (%.2f,%.2f): formula %+.6f; jittered grid minus formula at n=500,1000,2000: %s" % (
        tau, a, b, fv, ", ".join("%+.2e" % x for x in gaps)))
    print(out[-1], flush=True)
open("gq_cross_formula_res.out", "w", newline="\n").write("\n".join(out) + "\n")
