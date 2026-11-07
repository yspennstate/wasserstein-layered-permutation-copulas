"""Control for the orbit reduction in r9_iv.py: for tau, its inverse and rc(tau), the best-assignment F agrees on every band
(inverse: the same band with composition 12222; rc: the band reflected by s -> 2 - s with the composition reversed, 22221).
Checked with r9_comp_scan.py (cell integrals, independent of the nonile-mean code) on random blocks and a grid of bands of
mass up to 1, crossing the diagonal included. Also counts the orbits."""
import itertools, random
import numpy as np
import r9_comp_scan as S
from r9_iv import indec, inv, rc, rep, R
TAUS = [t for t in itertools.permutations(range(R)) if indec(t) and t != tuple(range(R - 1, -1, -1))]
REPS = sorted({rep(t) for t in TAUS})
assert all(indec(inv(t)) and indec(rc(t)) for t in TAUS)
assert {u for t in REPS for u in (t, inv(t), rc(t), rc(inv(t)))} == set(TAUS)
def F(t, a, b, comp):
    C, M = S.cells_for(a, b, comp)
    pts = np.array([(i + t[i] - (R - 1), i - t[i]) for i in range(R)], float)
    return (S.score(pts, C) - S.idv(a, b)) / (M / R)
random.seed(1)
bands = [(a, b) for a in np.linspace(0, 1.8, 10) for b in np.linspace(0.1, 2, 10) if b > a + 0.05] + [(0.0, 2.0)]
d_inv = d_rc = 0.0; n = 0
for t in random.sample(TAUS, 60) + [(1, 2, 3, 4, 5, 6, 7, 8, 0), (7, 8, 2, 6, 5, 4, 3, 1, 0)]:
    for a, b in bands:
        f = F(t, a, b, (1, 2, 2, 2, 2))
        d_inv = max(d_inv, abs(F(inv(t), a, b, (1, 2, 2, 2, 2)) - f))
        d_rc = max(d_rc, abs(F(rc(t), 2 - b, 2 - a, (2, 2, 2, 2, 1)) - f)); n += 1
print("blocks %d, orbit representatives %d; %d band checks; max |F(inv) - F| %.2e; max |F(rc, reflected band) - F| %.2e"
      % (len(TAUS), len(REPS), n, d_inv, d_rc))
