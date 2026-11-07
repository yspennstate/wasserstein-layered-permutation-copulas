"""Control: the composition code at R=6 with comp 222 must reproduce r6_thirds_sign.F6 band by band."""
import numpy as np, itertools
import r7_comp_scan as S
from r6_thirds_sign import F6, Hinv
S.R = 6
err = 0.0
for tau in [(1, 2, 3, 4, 5, 0), (5, 0, 1, 2, 3, 4), (5, 4, 3, 0, 2, 1)]:
    pts = np.array([(i + tau[i] - 5, i - tau[i]) for i in range(6)], float)
    for q, m in [(0.0, 0.3), (0.2, 0.6), (0.45, 0.1), (0.7, 0.25), (0.0, 0.003)]:
        a, b = Hinv(q), Hinv(min(q + m, 1.0))
        C, M = S.cells_for(a, b, (2, 2, 2))
        f = (S.score(pts, C) - S.idv(a, b)) / (M / 6)
        g, _ = F6(a, b, tau)
        err = max(err, abs(f - g))
print("max |diff| vs F6: %.3e" % err)
