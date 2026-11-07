"""Dispersive-order test. On U = {a < S < b} (Lebesgue on the unit square), S has density L(s) = min(s, 2-s) on (a,b)
and v = x1-x2 has density g(v) = (1/2)|(a,b) n [|v|, 2-|v|]|, same total mass. If  L(Q_S(u)) >= g(Q_v(u))  for every
u in (0,1) (Q = quantile functions), then v is larger than S in the dispersive order on U, so every quantile-split gap of
v is >= that of S, i.e. H >= 0 for every split. Prints min over u of L(Q_S) - g(Q_v), and its ratio, over a grid of (a,b)."""
import sys
import numpy as np
N = int(sys.argv[1]) if len(sys.argv) > 1 else 161
NU = 2001
RES = 40001
s_fine = np.linspace(0, 2, RES); L_fine = np.minimum(s_fine, 2 - s_fine)
v = np.linspace(-1, 1, RES); av = np.abs(v)
u = (np.arange(NU) + 0.5) / NU
grid = np.linspace(0, 2, N)
worst = (9, None); worst_rel = (9, None)
for ia in range(N):
    for ib in range(ia + 1, N):
        a, b = grid[ia], grid[ib]
        # S quantiles
        msk = (s_fine >= a) & (s_fine <= b)
        ss = s_fine[msk]; Ls = L_fine[msk]
        cS = np.concatenate([[0], np.cumsum((Ls[1:] + Ls[:-1]) / 2 * np.diff(ss))])
        if cS[-1] <= 0: continue
        QS = np.interp(u * cS[-1], cS, ss); fS = np.minimum(QS, 2 - QS)
        g = 0.5 * np.clip(np.minimum(b, 2 - av) - np.maximum(a, av), 0, None)
        cv = np.concatenate([[0], np.cumsum((g[1:] + g[:-1]) / 2 * np.diff(v))])
        Qv = np.interp(u * cv[-1], cv, v); fv = 0.5 * np.clip(np.minimum(b, 2 - np.abs(Qv)) - np.maximum(a, np.abs(Qv)), 0, None)
        d = fS - fv; i = np.argmin(d); rel = d[i] / max(fS[i], 1e-12)
        if d[i] < worst[0]: worst = (d[i], (round(a, 4), round(b, 4), round(u[i], 4)))
        if rel < worst_rel[0]: worst_rel = (rel, (round(a, 4), round(b, 4), round(u[i], 4)))
print("N=%d  min L(Q_S)-g(Q_v) %.4e at (a,b,u)=%s" % (N, worst[0], worst[1]))
print("min relative %.4e at (a,b,u)=%s" % (worst_rel[0], worst_rel[1]))
