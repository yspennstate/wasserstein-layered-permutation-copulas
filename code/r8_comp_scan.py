"""Blocks of eight: float scan of composition partitions. Cut the band into S-pieces of masses k_i/8 (k a composition of 8)
and split each piece into k_i cells by equal fractions of u = (V+L)/(2L); every cell has mass M/8. For a cell with S-range
(x0,x1) and u-range (u0,u1): int s = (K(x1)-K(x0)) (u1-u0), int v = 2 (J(x1)-J(x0)) ((u1^2-u0^2)-(u1-u0)).
F/m is scanned over bands of mass <= 8/9; per band the best composition in the list and the best assignment are taken.
Usage: python r8_comp_scan.py COMPS [tau ...]   COMPS like 2222,22211,11222 (or 'all')."""
import sys, itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from win_r import indec
Hs = lambda s: s * s / 2 if s <= 1 else 1 - (2 - s) ** 2 / 2
Hinv = lambda m: (2 * m) ** 0.5 if m <= 0.5 else 2 - (2 * (1 - m)) ** 0.5
K = lambda s: s ** 3 / 3 if s <= 1 else s * s - s ** 3 / 3 - 1 / 3
J = lambda s: s ** 3 / 6 if s <= 1 else (2 - (2 - s) ** 3) / 6
R = 8

def comps_all(n):
    if n == 0: yield (); return
    for k in range(1, n + 1):
        for c in comps_all(n - k): yield (k,) + c

def cells_for(a, b, comp):
    ma, mb = Hs(a), Hs(b); M = mb - ma
    out, acc = [], 0
    for k in comp:
        x0 = Hinv(ma + acc * M / R); x1 = Hinv(ma + (acc + k) * M / R) if acc + k < R else b
        if acc == 0: x0 = a
        dK, dJ = K(x1) - K(x0), J(x1) - J(x0)
        for j in range(k):
            u0, u1 = j / k, (j + 1) / k
            out.append((dK * (u1 - u0), 2 * dJ * ((u1 * u1 - u0 * u0) - (u1 - u0))))
        acc += k
    return np.array(out), M

def idv(a, b):
    ma, mb = Hs(a), Hs(b); M = mb - ma
    x = [a] + [Hinv(ma + j * M / R) for j in range(1, R)] + [b]
    return sum((2 * j - (R - 1)) * (K(x[j + 1]) - K(x[j])) for j in range(R))

def score(pts, C):
    W = pts @ C.T
    ri, ci = linear_sum_assignment(-W)
    return W[ri, ci].sum()

if __name__ == "__main__":
    arg = sys.argv[1]
    COMPS = list(comps_all(R)) if arg == "all" else [tuple(int(c) for c in s) for s in arg.split(",")]
    TAUS = [t for t in itertools.permutations(range(R)) if indec(t) and t != tuple(range(R - 1, -1, -1))]
    sel = [tuple(int(c) for c in s) for s in sys.argv[2:]] or TAUS
    MMAX, G = 8 / 9, 9
    bands = []
    for m in list(np.linspace(MMAX / G, MMAX, G)) + [0.003]:
        for q in np.linspace(0, 1 - m, G):
            a, b = Hinv(q), Hinv(min(q + m, 1.0))
            Cs = [cells_for(a, b, c)[0] for c in COMPS]
            bands.append((q, m, Cs, idv(a, b)))
    # control: the identity must be negative under every composition
    idp = np.array([(2 * i - (R - 1), 0.0) for i in range(R)])
    ctrl = max(max(score(idp, C) for C in Cs) - iv for (_, _, Cs, iv) in bands)
    res, out = [], ["identity control: max over bands of best-composition F (must be <= 0): %+.3e" % ctrl]
    for tau in sel:
        pts = np.array([(i + tau[i] - (R - 1), i - tau[i]) for i in range(R)], float)
        worst = (9e9, None, None, None)
        for q, m, Cs, iv in bands:
            vals = [score(pts, C) for C in Cs]
            k = int(np.argmax(vals)); f = (vals[k] - iv) / (m / R) / m
            if f < worst[0]: worst = (f, q, m, COMPS[k])
        res.append((worst[0], tau))
        out.append("tau=%s min F/m %+.4f at q=%.3f m=%.3f comp=%s" % ("".join(map(str, tau)), worst[0], worst[1], worst[2], "".join(map(str, worst[3]))))
    res.sort()
    out.append("comps=%s negative: %d of %d; worst: %s" % (arg, sum(f <= 0 for f, _ in res), len(res),
               ", ".join("%s %+.4f" % ("".join(map(str, t)), f) for f, t in res[:10])))
    NL = chr(10)
    tag = arg.replace(",", "_")[:40]
    open("r8_comp_%s.out" % tag if not sys.argv[2:] else "r8_comp_%s_sel.out" % tag, "w", newline=NL).write(NL.join(out) + NL)
    print(out[0]); print(out[-1])
