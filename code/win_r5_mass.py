"""Window lemma, r = 5: the 'v then s' / 's then v' cut F(a,b,tau) of win_r.py for every indecomposable non-layered
block of five values, on bands of mass <= MMAX (5/9: n >= 9). Bands parametrised by lower quantile q and mass m.
Reports min F/m per tau (and the band), then the worst taus. Usage: win_r5_mass.py [MMAX] [G]."""
import sys, itertools
import numpy as np
from win_r import F, indec
Q = lambda p: (2 * p) ** 0.5 if p <= 0.5 else 2 - (2 * (1 - p)) ** 0.5
r = 5; MMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 5 / 9; G = int(sys.argv[2]) if len(sys.argv) > 2 else 11
TAUS = [t for t in itertools.permutations(range(r)) if indec(t) and t != tuple(range(r - 1, -1, -1))]
out, res = ["r=5: %d indecomposable non-reversal blocks, mass<=%.4f, grid %d" % (len(TAUS), MMAX, G)], []
for tau in TAUS:
    best = (9, None, None)
    for m in np.linspace(MMAX / G, MMAX, G):
        for q in np.linspace(0, 1 - m, G):
            f = F(Q(q), Q(min(q + m, 1.0)), tau) / m
            if f < best[0]: best = (f, q, m)
    res.append((best[0], tau))
    out.append("tau=%s  min F/m %+.4f at q=%.3f m=%.3f" % ("".join(map(str, tau)), best[0], best[1], best[2]))
    print(out[-1], flush=True)
    open("win_r5_mass.out", "w", newline="\n").write("\n".join(out) + "\n")
res.sort()
out.append("negative: %d of %d; worst five: %s" % (sum(1 for f, _ in res if f <= 0), len(res),
           ", ".join("%s %+.4f" % ("".join(map(str, t)), f) for f, t in res[:5])))
print(out[-1]); open("win_r5_mass.out", "w", newline="\n").write("\n".join(out) + "\n")
