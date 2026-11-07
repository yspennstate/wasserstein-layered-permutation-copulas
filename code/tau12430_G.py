"""tau = 12430: a second admissible cut. Give V2 to the point (-3,-1) and V3 to (-1,-1) instead of splitting V2 u V3 at
its s-median; its value is G = F0 + E[s|V3] - E[s|V2] (F0 of tau12430_F0.py). On bands with s >= 1 a larger |v| means a
smaller s, so E[s|V3] > E[s|V2] there. Grid scan of max(F0, G)/m over bands of mass m <= 5/9."""
import numpy as np
g = (np.arange(1500) + 0.5) / 1500; X1, X2 = np.meshgrid(g, g, indexing="ij"); S0 = (X1 + X2).ravel(); V0 = (X1 - X2).ravel()
Q = lambda p: (2 * p) ** 0.5 if p <= 0.5 else 2 - (2 * (1 - p)) ** 0.5
def FG(a, b):
    k = (S0 > a) & (S0 < b); s, v = S0[k], V0[k]
    qv = np.quantile(v, [0.2, 0.4, 0.6, 0.8]); qs = np.quantile(s, [0.2, 0.4, 0.6, 0.8])
    c = [v <= qv[0]] + [(v > qv[j - 1]) & (v <= qv[j]) for j in range(1, 4)] + [v > qv[3]]
    A = v[c[4]].mean(); B = v[c[3]].mean(); es = [s[x].mean() for x in c]
    Sj = [s[s <= qs[0]].mean()] + [s[(s > qs[j - 1]) & (s <= qs[j])].mean() for j in range(1, 4)] + [s[s > qs[3]].mean()]
    f0 = 6 * A + B + 2 * (es[4] - es[2]) - sum((2 * j - 4) * Sj[j] for j in range(5))
    return f0, f0 + es[2] - es[1]
out, w0, wg, wm = [], (9,), (9,), (9,)
for m in np.linspace(5 / 9 / 12, 5 / 9, 12):
    for q in np.linspace(0, 1 - m, 13):
        a, b = Q(q), Q(min(q + m, 1)); f0, gg = FG(a, b)
        w0 = min(w0, (f0 / m, q, m)); wg = min(wg, (gg / m, q, m)); wm = min(wm, (max(f0, gg) / m, q, m))
for nm, w in (("F0", w0), ("G", wg), ("max(F0,G)", wm)):
    out.append("min %s/m over bands of mass <= 5/9: %+.4f at q=%.3f m=%.3f" % ((nm,) + w))
print("\n".join(out)); open("tau12430_G.out", "w", newline="\n").write("\n".join(out) + "\n")
