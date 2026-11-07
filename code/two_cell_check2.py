"""Per-split form of the block-reversal bound. Exact identity (Lemma 2.2 of the paper):
   n^2 D = sum_{t=1}^{r-1} t(r-t)/r * ( Dv_t - DS_t ),
DS_t = E[S | x<S<b] - E[S | a<S<x] (x = strip boundary q_{k+t}), Dv_t = same gap for v = x1-x2 on U split at its own
tau-quantile, tau = P(a<S<x)/P(a<S<b). Check per split, continuously: H(a,x,b) = Dv - DS >= 0 for 0<=a<x<b<=2,
(a,b) != (0,2). Grid of N points per axis, v-quadrature RES."""
import sys
import numpy as np
N = int(sys.argv[1]) if len(sys.argv) > 1 else 121
RES = int(sys.argv[2]) if len(sys.argv) > 2 else 4001
def M(a, b, p):
    def piece(lo, hi, sgn):
        lo, hi = max(lo, 0 if sgn > 0 else 1), min(hi, 1 if sgn > 0 else 2)
        if hi <= lo: return 0.0
        if sgn > 0: return (hi**(p+2) - lo**(p+2)) / (p+2)
        return 2*(hi**(p+1) - lo**(p+1))/(p+1) - (hi**(p+2) - lo**(p+2))/(p+2)
    return piece(a, b, 1) + piece(a, b, -1)
s = np.linspace(0, 2, N)
v = np.linspace(-1, 1, RES); av = np.abs(v); dv = v[1] - v[0]
worst = (9, None); worst_rel = (9, None); cnt = 0
for ia in range(N):
    for ib in range(ia + 2, N):
        a, b = s[ia], s[ib]
        if ia == 0 and ib == N - 1: continue
        g = 0.5 * np.clip(np.minimum(b, 2 - av) - np.maximum(a, av), 0, None)
        cm = np.concatenate([[0], np.cumsum((g[1:] + g[:-1]) / 2 * dv)])
        cv = np.concatenate([[0], np.cumsum(((g * v)[1:] + (g * v)[:-1]) / 2 * dv)])
        tot, totv = cm[-1], cv[-1]
        m0 = M(a, b, 0)
        for ix in range(ia + 1, ib):
            x = s[ix]
            lo0, hi0 = M(a, x, 0), M(x, b, 0)
            DS = M(x, b, 1) / hi0 - M(a, x, 1) / lo0
            tau = lo0 / m0
            low_v = np.interp(tau * tot, cm, cv)
            Dv = (totv - low_v) / (tot * (1 - tau)) - low_v / (tot * tau)
            H = Dv - DS; cnt += 1
            if H < worst[0]: worst = (H, (a, x, b))
            if H / (b - a) < worst_rel[0]: worst_rel = (H / (b - a), (a, x, b))
print("splits", cnt, "min H %.3e at (a,x,b)=%s" % (worst[0], tuple(round(t, 4) for t in worst[1])))
print("min H/(b-a) %.3e at %s" % (worst_rel[0], tuple(round(t, 4) for t in worst_rel[1])))
