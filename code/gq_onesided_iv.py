"""r = 5, the v-quintile cut Gq (gq_scan.py) on bands that do not cross the diagonal, proved positive in interval
arithmetic for the representative of each orbit except 12403. By the symmetry v -> -v of a band, with A = E[v|c_4],
B = E[v|c_3] and s1, s2, s3 the s-means of the outer, second and middle |v|-cells,
  Gq = (v'_4 - v'_0) A + (v'_3 - v'_1) B + (u'_0 + u'_4) s1 + (u'_1 + u'_3) s2 + u'_2 s3 - sum_j (2j-4) S_j
on a bottom band, and on a top band (s' = 2 - s, using sum u' = 0) the s-coefficients change sign and the rank term
is the same function of the reflected band. The pieces and quantiles are those of tau12430_G_iv.py; a box is proved
when one of the cuts (ties in v' give several) has a positive lower bound."""
import sys
from mpmath import iv, mpf
from tau12430_G_iv import pieces
from gq_scan import cuts
iv.dps = 30
def coeffs(tau):
    out = []
    for u, w in cuts(tau):
        out.append((w[4] - w[0], w[3] - w[1], u[0] + u[4], u[1] + u[3], u[2]))
    return out
def Gq(u0, u1, top, C):
    t = iv.mpf([u0, u1]) / 63
    V1, V2, V3 = pieces(t, u1)
    E = lambda P: (P[1] / P[0], P[2] / P[0])
    (A, s1), (B, s2), (_, s3) = E(V1), E(V2), E(V3)
    sp = [t] + [iv.sqrt(t * t + iv.mpf(p) * (1 - t * t)) for p in ("0.2", "0.4", "0.6", "0.8")] + [iv.mpf(1)]
    S = [(2 * (sp[j + 1] ** 2 + sp[j + 1] * sp[j] + sp[j] ** 2)) / (3 * (sp[j] + sp[j + 1])) for j in range(5)]
    rank = sum((2 * j - 4) * S[j] for j in range(5))
    sg = -1 if top else 1
    return max((cA * A + cB * B + sg * (d1 * s1 + d2 * s2 + d3 * s3) - rank).a for cA, cB, d1, d2, d3 in C)
def check(u0, u1, top, C, acc, depth=0):
    g = Gq(u0, u1, top, C)
    if g > 0:
        acc.append((u0, u1, g)); return True
    if depth > 40: return False
    m = (u0 + u1) / 2
    return check(u0, m, top, C, acc, depth + 1) and check(m, u1, top, C, acc, depth + 1)
if __name__ == "__main__":
    reps = [l.split()[0].split(",")[0] for l in open("orbits_r5.out").read().splitlines()[1:] if "min F/m" in l]
    reps = [r for r in reps if r != "12403"] if len(sys.argv) < 2 else sys.argv[1:]
    out, allok = [], True
    for rep in reps:
        C = coeffs(tuple(int(c) for c in rep)); parts = []
        for top in (False, True):
            for lo, hi in ((0, 7), (7, 27), (27, 63)):
                acc = []; ok = check(mpf(lo), mpf(hi), top, C, acc); allok &= ok
                parts.append("%s[%d,%d] %s %d boxes min %s" % ("top" if top else "bot", lo, hi, "ok" if ok else "FAIL", len(acc),
                             iv.nstr(min(z[2] for z in acc), 4) if acc else "-"))
        out.append("tau=%s  %s" % (rep, "; ".join(parts))); print(out[-1], flush=True)
        open("gq_onesided_iv.out", "w", newline="\n").write("\n".join(out) + "\n")
    out.append("all one-sided bands proved for %d orbit representatives: %s" % (len(reps), allok))
    print(out[-1]); open("gq_onesided_iv.out", "w", newline="\n").write("\n".join(out) + "\n")
