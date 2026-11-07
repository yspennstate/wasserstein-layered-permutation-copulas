"""Control for the orbit reduction in r8_iv.py: for tau, its inverse and rc(tau), the best-assignment F agrees on every band
(inverse: the same band; rc: the band reflected by s -> 2 - s). Checked with r8_comp_scan.py (cell integrals, independent of
the octile-mean code) on random blocks and a grid of bands, crossing the diagonal included. Also counts the orbits."""
import itertools, random
import numpy as np
import r8_comp_scan as S
from r8_iv import indec, inv, rc, rep, R
TAUS = [t for t in itertools.permutations(range(R)) if indec(t) and t != tuple(range(R - 1, -1, -1))]
REPS = sorted({rep(t) for t in TAUS})
assert all(indec(inv(t)) and indec(rc(t)) for t in TAUS)
assert {u for t in REPS for u in (t, inv(t), rc(t), rc(inv(t)))} == set(TAUS)
def F(t, a, b):
    C, M = S.cells_for(a, b, (2, 2, 2, 2))
    pts = np.array([(i + t[i] - (R - 1), i - t[i]) for i in range(R)], float)
    return (S.score(pts, C) - S.idv(a, b)) / (M / R)
random.seed(1)
bands = [(a, b) for a in np.linspace(0, 1.8, 10) for b in np.linspace(0.1, 2, 10) if b > a + 0.05 and S.Hs(b) - S.Hs(a) <= 8 / 9]
d_inv = d_rc = 0.0; n = 0
for t in random.sample(TAUS, 60) + [(1, 2, 3, 4, 5, 6, 7, 0), (7, 6, 5, 3, 4, 2, 1, 0)]:
    for a, b in bands:
        f = F(t, a, b)
        d_inv = max(d_inv, abs(F(inv(t), a, b) - f))
        d_rc = max(d_rc, abs(F(rc(t), 2 - b, 2 - a) - f)); n += 1
print("blocks %d, orbit representatives %d; %d band checks; max |F(inv) - F| %.2e; max |F(rc, reflected band) - F| %.2e"
      % (len(TAUS), len(REPS), n, d_inv, d_rc))
