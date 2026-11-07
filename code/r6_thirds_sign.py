"""Fully explicit partition for r = 6: cut the band into s-thirds (quantiles 1/3, 2/3), split each third at v = 0. Six cells
of equal mass M/6, with int s = (K(x_{2k+2}) - K(x_{2k}))/2 and int v = +-(J(x_{2k+2}) - J(x_{2k}))/2, K(s) = int_0^s t L(t) dt,
J(s) = int_0^s L(t)^2/2 dt, L(t) = min(t, 2-t) (the square has mass 1). The best assignment of the six points to the six cells
(720 bijections) gives (M/6) F = max_pi sum_i p_i . c_pi(i) - [5K(a) + 5K(b) - 2 sum_{j=1..5} K(x_j)]... with the sign of the rank
term as in the window lemma. Scan F/m over the win_r6_mass grid for every block (or the given ones)."""
import sys, itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from win_r import indec
Hs = lambda s: s * s / 2 if s <= 1 else 1 - (2 - s) ** 2 / 2
Hinv = lambda m: (2 * m) ** 0.5 if m <= 0.5 else 2 - (2 * (1 - m)) ** 0.5
K = lambda s: s ** 3 / 3 if s <= 1 else s * s - s ** 3 / 3 - 1 / 3
J = lambda s: s ** 3 / 6 if s <= 1 else (2 - (2 - s) ** 3) / 6
PERMS = list(itertools.permutations(range(6)))
def F6(a, b, tau):
    r = 6; ma, mb = Hs(a), Hs(b); M = mb - ma
    x = [a] + [Hinv(ma + j * M / 6) for j in range(1, 6)] + [b]
    cells = []
    for k in range(3):
        cs = (K(x[2 * k + 2]) - K(x[2 * k])) / 2; cv = (J(x[2 * k + 2]) - J(x[2 * k])) / 2
        cells += [(cs, -cv), (cs, cv)]
    pts = [(i + tau[i] - (r - 1), i - tau[i]) for i in range(r)]
    W = np.array([[p[0] * c[0] + p[1] * c[1] for c in cells] for p in pts])
    ri, ci = linear_sum_assignment(-W); best = W[ri, ci].sum()
    idv = 5 * K(a) + 5 * K(b) - 2 * sum(K(x[j]) for j in range(1, 6))
    # identity: strips j with u'_id = 2j-5 -> sum (2j-5)(K(x_{j+1})-K(x_j)) = 5K(b)+5K(a)-2 sum K(x_j)
    return (best - idv) / (M / 6), M
if __name__ == "__main__":
    MMAX, G = 6 / 9, 11
    TAUS = [t for t in itertools.permutations(range(6)) if indec(t) and t != (5, 4, 3, 2, 1, 0)]
    sel = [tuple(int(c) for c in s) for s in sys.argv[1:]] or TAUS
    Q = Hinv
    out, res = [], []
    for tau in sel:
        best = (9, None, None)
        for m in list(np.linspace(MMAX / G, MMAX, G)) + [0.002, 0.005]:
            for q in np.linspace(0, 1 - m, G):
                f, _ = F6(Q(q), Q(min(q + m, 1.0)), tau)
                if f / m < best[0]: best = (f / m, q, m)
        res.append((best[0], tau)); out.append("tau=%s  min F6/m %+.4f at q=%.3f m=%.3f" % ("".join(map(str, tau)), *best))
    res.sort(); out.append("negative: %d of %d; worst: %s" % (sum(f <= 0 for f, _ in res), len(res), ", ".join("%s %+.4f" % ("".join(map(str, t)), f) for f, t in res[:8])))
    print(out[-1]); NL = chr(10)
    open("r6_thirds_sign.out" if not sys.argv[1:] else "r6_thirds_sign_sel.out", "w", newline=NL).write(NL.join(out) + NL)
