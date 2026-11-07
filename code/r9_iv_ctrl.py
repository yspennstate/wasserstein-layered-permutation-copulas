"""Control for r9_iv.py: at point bands the interval F must contain the float F of r9_comp_scan.py (composition 12222, best
assignment), computed there from cell integrals rather than nonile means. Bands with a >= 1 are also checked through the
reflection used by family T (points (-u, v), composition 22221, band [2 - b, 2 - a]). The full band [0, 2] (mass 1) is
included. The identity is negative throughout."""
from mpmath import iv, mpf
import r9_comp_scan as S
import r9_iv as P
import numpy as np
CB, CT = P.cell_forms((1, 2, 2, 2, 2)), P.cell_forms((2, 2, 2, 2, 1))
worst, worst_t, n = 0.0, 0.0, 0
for tau in [(1, 2, 3, 4, 5, 6, 7, 8, 0), (8, 0, 1, 2, 3, 4, 5, 6, 7), (7, 8, 2, 6, 5, 4, 3, 1, 0), (1, 3, 2, 4, 5, 6, 7, 8, 0),
            (3, 6, 0, 8, 1, 5, 2, 7, 4)]:
    pts = [(i + tau[i] - 8, i - tau[i]) for i in range(9)]
    for a, b in [(0.0, 1.0), (0.3, 0.9), (0.95, 1.0), (0.0, 1.5), (0.4, 1.3), (0.9, 1.1), (0.99, 1.01), (1.0, 1.6), (1.2, 1.9),
                 (0.0, 2.0), (0.05, 1.97)]:
        C, M = S.cells_for(a, b, (1, 2, 2, 2, 2))
        fs = (S.score(np.array(pts, float), C) - S.idv(a, b)) / (M / 9)
        asg, ff = P.F_f(pts, a, b, CB)
        z = P.F_iv(pts, mpf(a), mpf(a), mpf(b), mpf(b), asg, CB)
        worst = max(worst, abs(ff - fs), max(0.0, float(z.a) - fs - 1e-12), max(0.0, fs - float(z.b) - 1e-12))
        if a >= 1:
            pr = [(-u, v) for u, v in pts]
            asg2, ff2 = P.F_f(pr, 2 - b, 2 - a, CT)
            worst_t = max(worst_t, abs(ff2 - fs)); n += 1
        n += 1
idp = [(2 * i - 8, 0) for i in range(9)]
idmax = max(P.F_f(idp, a, b, CB)[1] for a, b in [(0.0, 1.0), (0.3, 0.9), (0.0, 1.5), (0.4, 1.3), (1.0, 1.6), (0.0, 2.0)])
print("checks %d; max |float(r9_iv) - float(r9_comp_scan)| and enclosure miss %.3e; reflection diff %.3e; identity max F %.3e"
      % (n, worst, worst_t, idmax))
