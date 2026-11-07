"""Control for r7_iv.py: at point bands the interval F must contain the float F of r7_comp_scan.py (composition 2221, best
assignment), computed there from cell integrals rather than septile means. Bands with a >= 1 are also checked through the
reflection used by family T (points (-u, v), composition 1222, band [2 - b, 2 - a]). The identity is negative throughout."""
from mpmath import iv, mpf
import r7_comp_scan as S
import r7_iv as P
import numpy as np
CB, CT = P.cell_forms((2, 2, 2, 1)), P.cell_forms((1, 2, 2, 2))
worst, worst_t, n = 0.0, 0.0, 0
for tau in [(1, 2, 3, 4, 5, 6, 0), (6, 0, 1, 2, 3, 4, 5), (1, 3, 2, 4, 5, 6, 0), (6, 5, 4, 0, 3, 2, 1), (3, 6, 0, 5, 1, 4, 2)]:
    pts = [(i + tau[i] - 6, i - tau[i]) for i in range(7)]
    for a, b in [(0.0, 1.0), (0.3, 0.9), (0.95, 1.0), (0.0, 1.5), (0.4, 1.3), (0.9, 1.1), (0.99, 1.01), (1.0, 1.6), (1.2, 1.9)]:
        C, M = S.cells_for(a, b, (2, 2, 2, 1))
        fs = (S.score(np.array(pts, float), C) - S.idv(a, b)) / (M / 7)
        asg, ff = P.F_f(pts, a, b, CB)
        z = P.F_iv(pts, mpf(a), mpf(a), mpf(b), mpf(b), asg, CB)
        worst = max(worst, abs(ff - fs), max(0.0, float(z.a) - fs - 1e-12), max(0.0, fs - float(z.b) - 1e-12))
        if a >= 1:
            pr = [(-u, v) for u, v in pts]
            asg2, ff2 = P.F_f(pr, 2 - b, 2 - a, CT)
            worst_t = max(worst_t, abs(ff2 - fs)); n += 1
        n += 1
idp = [(2 * i - 6, 0) for i in range(7)]
idmax = max(P.F_f(idp, a, b, CB)[1] for a, b in [(0.0, 1.0), (0.3, 0.9), (0.0, 1.5), (0.4, 1.3), (1.0, 1.6)])
print("checks %d; max |float(r7_iv) - float(r7_comp_scan)| and enclosure miss %.3e; reflection diff %.3e; identity max F %.3e"
      % (n, worst, worst_t, idmax))
