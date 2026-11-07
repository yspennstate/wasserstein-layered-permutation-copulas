"""tau = 1302: the 'v then s' cut puts each block point in its own v-quartile cell, and the s-terms cancel by the v -> -v
symmetry. Closed form F = 4A + 2B - 3(S4 - S1) - (S3 - S2): A, B = E[v | top, third v-quartile of U], S_j = E[s | j-th
s-quartile of U]. Check against win_r.F on a fine grid of the band."""
import numpy as np
from win_r import F
def closed(a, b, m=2000):
    g = (np.arange(m) + 0.5) / m; X1, X2 = np.meshgrid(g, g, indexing="ij")
    s = (X1 + X2).ravel(); v = (X1 - X2).ravel(); k = (s > a) & (s < b); s, v = s[k], v[k]
    qv = np.quantile(v, [0.5, 0.75]); qs = np.quantile(s, [0.25, 0.5, 0.75])
    A = v[v > qv[1]].mean(); B = v[(v > qv[0]) & (v <= qv[1])].mean()
    S = [s[s <= qs[0]].mean(), s[(s > qs[0]) & (s <= qs[1])].mean(), s[(s > qs[1]) & (s <= qs[2])].mean(), s[s > qs[2]].mean()]
    return 4 * A + 2 * B - 3 * (S[3] - S[0]) - (S[2] - S[1])
for a, b in [(0, 0.943), (0, 1.0), (0.3, 0.9), (0.5, 1.4), (0.2, 1.8), (0, 1.8), (1.057, 2.0)]:
    print("band (%.3f,%.3f)  closed %+.5f   win_r.F %+.5f" % (a, b, closed(a, b), F(a, b, (1, 3, 0, 2))))
