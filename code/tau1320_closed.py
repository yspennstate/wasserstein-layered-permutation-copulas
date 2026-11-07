"""tau = 1320 (points u' = -2,1,1,0; v' = -1,-2,0,3). Two cuts, closed forms by the v -> -v symmetry of the band:
'v then s' (each point its own v-quartile): F1 = 5A + B + (T_out - T_in) - 3(S4 - S1) - (S3 - S2),
  A, B = E[v | top, third v-quartile], T_out = E[s | |v| in its top half], T_in = E[s | |v| in its bottom half];
's then v' (u'-groups -2 | 0 | 1,1): F2 = 2C - (2 S4 - S1 - S2), C = E[|v| | s in its top half].
S_j = E[s | j-th s-quartile]. Checked against the polygon cuts of win_r.py (cutval) on a fine grid of the band."""
import numpy as np
from win_r import cutval
from cyc3_window import am, band
def closed(a, b, m=2000):
    g = (np.arange(m) + 0.5) / m; X1, X2 = np.meshgrid(g, g, indexing="ij")
    s = (X1 + X2).ravel(); v = (X1 - X2).ravel(); k = (s > a) & (s < b); s, v = s[k], v[k]; w = np.abs(v)
    qv = np.quantile(v, [0.5, 0.75]); qs = np.quantile(s, [0.25, 0.5, 0.75]); qw = np.median(w)
    A = v[v > qv[1]].mean(); B = v[(v > qv[0]) & (v <= qv[1])].mean()
    S = [s[s <= qs[0]].mean(), s[(s > qs[0]) & (s <= qs[1])].mean(), s[(s > qs[1]) & (s <= qs[2])].mean(), s[s > qs[2]].mean()]
    To, Ti = s[w > qw].mean(), s[w <= qw].mean(); C = w[s > qs[1]].mean()
    F1 = 5 * A + B + (To - Ti) - 3 * (S[3] - S[0]) - (S[2] - S[1]); F2 = 2 * C - (2 * S[3] - S[0] - S[1])
    return F1, F2
def poly(a, b, tau=(1, 3, 2, 0)):
    from win_r import slices
    U = band(a, b); M, _ = am(U); r = 4
    idv = sum((2 * j - 3) * am(St)[1][0] for j, St in enumerate(slices(U, M, r, 0)))
    pts = [(i + tau[i] - 3, i - tau[i]) for i in range(r)]
    return cutval(U, M, pts, 1, r) - idv, cutval(U, M, pts, 0, r) - idv
if __name__ == "__main__":
    out = []
    for a, b in [(0, 0.943), (0, 1.0), (0.3, 0.9), (0.5, 1.4), (0.2, 1.8), (1.057, 2.0), (1.2, 1.9)]:
        c, p = closed(a, b), poly(a, b)
        out.append("band (%.3f,%.3f)  F1 closed %+.5f poly %+.5f | F2 closed %+.5f poly %+.5f" % (a, b, c[0], p[0], c[1], p[1]))
        print(out[-1], flush=True)
    open("tau1320_closed.out", "w", newline="\n").write("\n".join(out) + "\n")
