"""tau = 12430, the cut G of tau12430_G.py on bands that do not cross the diagonal s = 1, proved positive in interval
arithmetic (mpmath.iv). G = 6A + B + 2E[s|V5] - E[s|V3] - E[s|V2] - sum_j (2j-4) S_j, with V1..V5 the v-quintile cells,
A = E[v|V5], B = E[v|V4], S_j the s-quintile means.
Bottom bands a < s < b <= 1: all quantities are lengths, so G(a,b) = b G(t), t = a/b, region t < s < 1, |v| < s.
The region is symmetric in v, so A = E[y|V1], B = E[y|V2] with y = -v, E[s|V5] = E[s|V1], E[s|V4] = E[s|V2], and V3 is
y in [0,y2] on one side. One side: the slice at y carries s in (max(t,y), 1).
Quintile points (fraction p of the mass below -y): y = 1 - sqrt(2p(1-t^2)) while p <= (1-t)/(2(1+t)), else
y = t - p(1+t) + (1-t)/2. Branches at t = 1/9 (p = 0.4) and t = 3/7 (p = 0.2); t = u/63 makes them u = 7, 27.
Top bands 1 <= a < s < b: s' = 2 - s is a bottom region, E[s|.] = 2 - E[s'|.], S_j = 2 - S'_{4-j}, so
  G_top = 6A + B - 2E'[s|V1] + E'[s|V3] + E'[s|V2] - sum_k (2k-4) S'_k,  again homogeneous of degree 1.
In the branch u >= 27 every integral is divided by (1-t) before forming ratios, so t -> 1 is not singular."""
from mpmath import iv, mpf
iv.dps = 30
ONE = None
def _tri(y0, y1):                      # one side, y in [y0,y1], y0 >= t: (mass, int y, int s)
    return (((1 - y0) ** 2 - (1 - y1) ** 2) / 2, (y1 ** 2 - y0 ** 2) / 2 - (y1 ** 3 - y0 ** 3) / 3,
            (y1 - y0) / 2 - (y1 ** 3 - y0 ** 3) / 6)
def _flat(y0, y1, c, t):               # one side, y in [y0,y1], y1 <= t; c = 1-t, or ONE when divided by 1-t
    k = 1 if c is ONE else c
    return k * (y1 - y0), k * (y1 ** 2 - y0 ** 2) / 2, k * (1 + t) * (y1 - y0) / 2
def _add(P, Q):
    return tuple(p + q for p, q in zip(P, Q))
def pieces(t, u1):
    ytri = lambda p: 1 - iv.sqrt(2 * iv.mpf(p) * (1 - t * t))
    yflat = lambda p: t - iv.mpf(p) * (1 + t) + (1 - t) / 2
    one_t, zero = 1 - t, iv.mpf(0)
    if u1 <= 7:
        y1, y2 = ytri("0.2"), ytri("0.4")
        return _tri(y1, iv.mpf(1)), _tri(y2, y1), _add(_flat(zero, t, one_t, t), _tri(t, y2))
    if u1 <= 27:
        y1, y2 = ytri("0.2"), yflat("0.4")
        return _tri(y1, iv.mpf(1)), _add(_flat(y2, t, one_t, t), _tri(t, y1)), _flat(zero, y2, one_t, t)
    y1, y2 = yflat("0.2"), yflat("0.4")
    T = (one_t / 2, one_t * (1 + 2 * t) / 6, one_t * (2 + t) / 6)   # full triangle [t,1], divided by 1-t
    return _add(T, _flat(y1, t, ONE, t)), _flat(y2, y1, ONE, t), _flat(zero, y2, ONE, t)
def Giv(u0, u1, top=False):
    t = iv.mpf([u0, u1]) / 63
    V1, V2, V3 = pieces(t, u1)
    E = lambda P: (P[1] / P[0], P[2] / P[0])          # (E[y], E[s])
    (A, s1), (B, s2), (_, s3) = E(V1), E(V2), E(V3)
    sp = [t] + [iv.sqrt(t * t + iv.mpf(p) * (1 - t * t)) for p in ("0.2", "0.4", "0.6", "0.8")] + [iv.mpf(1)]
    S = [(2 * (sp[j + 1] ** 2 + sp[j + 1] * sp[j] + sp[j] ** 2)) / (3 * (sp[j] + sp[j + 1])) for j in range(5)]
    rank = sum((2 * j - 4) * S[j] for j in range(5))
    if top:
        return 6 * A + B - 2 * s1 + s3 + s2 - rank
    return 6 * A + B + 2 * s1 - s3 - s2 - rank
def check(u0, u1, top, depth=0, acc=None):
    g = Giv(u0, u1, top)
    if g.a > 0:
        acc.append((u0, u1, g.a)); return True
    if depth > 40: return False
    m = (u0 + u1) / 2
    return check(u0, m, top, depth + 1, acc) and check(m, u1, top, depth + 1, acc)
def grid(a, b, n=1500):
    import numpy as np
    g = (np.arange(n) + 0.5) / n; X1, X2 = np.meshgrid(g, g, indexing="ij"); s = (X1 + X2).ravel(); v = (X1 - X2).ravel()
    k = (s > a) & (s < b); s, v = s[k], v[k]
    qv = np.quantile(v, [0.2, 0.4, 0.6, 0.8]); qs = np.quantile(s, [0.2, 0.4, 0.6, 0.8])
    c = [v <= qv[0]] + [(v > qv[j - 1]) & (v <= qv[j]) for j in range(1, 4)] + [v > qv[3]]
    es = [s[x].mean() for x in c]
    Sj = [s[s <= qs[0]].mean()] + [s[(s > qs[j - 1]) & (s <= qs[j])].mean() for j in range(1, 4)] + [s[s > qs[3]].mean()]
    return 6 * v[c[4]].mean() + v[c[3]].mean() + 2 * es[4] - es[2] - es[1] - sum((2 * j - 4) * Sj[j] for j in range(5))
if __name__ == "__main__":
    out = []
    for t in (0.0, 0.05, 0.3, 0.6, 0.9):
        for top in (False, True):
            u = mpf(t) * 63; gi = Giv(u, u, top); a, b = (t, 1.0) if not top else (1.0, 2 - t)
            out.append("check t=%.2f %-6s closed %s  grid %+.5f" % (t, "top" if top else "bottom", iv.nstr(gi, 8), grid(a, b)))
            print(out[-1], flush=True)
    for top in (False, True):
        for lo, hi in ((0, 7), (7, 27), (27, 63)):
            acc = []; ok = check(mpf(lo), mpf(hi), top, acc=acc); mn = min(acc, key=lambda z: z[2])
            out.append("%s bands, t in [%d/63, %d/63]: proved G > 0: %s; %d boxes; smallest lower bound %s at u in [%s, %s]" % (
                "top" if top else "bottom", lo, hi, ok, len(acc), iv.nstr(mn[2], 6), iv.nstr(mn[0], 6), iv.nstr(mn[1], 6)))
            print(out[-1], flush=True)
    open("tau12430_G_iv.out", "w", newline="\n").write("\n".join(out) + "\n")
