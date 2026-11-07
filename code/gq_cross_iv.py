"""r = 5, the v-quintile cut Gq on bands a < s < b crossing the diagonal with mass M <= 5/9, in interval arithmetic,
for the representative of each orbit except 12403 (integrals, quantiles and variables as in tau12430_G_cross_iv.py).
With y1 = H^-1(3M/5), y2 = H^-1(M/5) and Gq = cA A + cB B + d1 s1 + d2 s2 + d3 s3 - rank (gq_onesided_iv.py),
  (M/5) Gq = cA/2 (N(1)-N(y1)) + cB/2 (N(y1)-N(y2)) + d1/2 (P(1)-P(y1)) + d2/2 (P(y1)-P(y2)) + d3 P(y2)
             - 4K(a) - 4K(b) + 2(K(x1)+K(x2)+K(x3)+K(x4)).
d/dy1 of (M/5)Gq is h(y1)/2 ((cB-cA) y1 + (d2-d1) sbar(y1)) and d/dy2 is h(y2)/2 (-cB y2 + (2d3-d2) sbar(y2)), with
sbar(y) = (2 - max(beta,y) + max(alpha,y))/2 the mean of s on the slice |v| = y; where the bracket has one sign on the
enclosure of y the matching end is used, otherwise the whole enclosure; the rank term
increases in x1..x4 and uses their lower ends. Boxes are bisected until one cut has a positive lower bound. The thin
corner alpha + beta > 2 - W0 is set aside: there Gq >= 0.6 cA (1-w) - (U + 6) w with w = b - a < W0 and U the sum of
the positive u' (checked below). Usage: gq_cross_iv.py W0 [reps...]."""
import sys
from mpmath import iv, mpf
from tau12430_G_cross_iv import q, K, N, P, yq
from gq_scan import cuts
def imax(x, c):
    x, c = iv.mpf(x), iv.mpf(c)
    return iv.mpf([max(x.a, c.a), max(x.b, c.b)])
iv.dps = 25
W0 = mpf(sys.argv[1]) if len(sys.argv) > 1 else mpf("0.05")
def coeffs(tau):
    return [((w[4] - w[0]), (w[3] - w[1]), (u[0] + u[4]), (u[1] + u[3]), u[2], sum(x for x in u if x > 0)) for u, w in cuts(tau)]
def lower(a0, a1, b0, b1, C):
    al = iv.mpf([a0, a1]); be = iv.mpf([b0, b1]); M = (2 - al * al - be * be) / 2
    xs = [q(iv.mpf(a0), iv.mpf(b1), iv.mpf(p)).a for p in ("0.2", "0.4", "0.6", "0.8")]
    xs = [min(x, q(iv.mpf(a1), iv.mpf(b0), iv.mpf(p)).a) for x, p in zip(xs, ("0.2", "0.4", "0.6", "0.8"))]
    y1 = yq(al, be, 3 * M / 5); y2 = yq(al, be, M / 5); one = iv.mpf(1)
    N1, P1 = N(one, al, be), P(one, al, be)
    sbar = lambda y: (2 - imax(be, y) + imax(al, y)) / 2
    cache = {}
    def NP(y, side):
        key = (id(y), side)
        if key not in cache:
            z = y if side == 0 else iv.mpf(y.a if side < 0 else y.b)
            cache[key] = (N(z, al, be), P(z, al, be))
        return cache[key]
    r = -4 * al ** 3 / 3 - 4 * K(2 - be) + 2 * sum(K(iv.mpf(x)) for x in xs)
    best = None
    for cA, cB, d1, d2, d3, _ in C:
        g1 = (cB - cA) * y1 + (d2 - d1) * sbar(y1); g2 = -cB * y2 + (2 * d3 - d2) * sbar(y2)
        e1 = 1 if g1.b < 0 else (-1 if g1.a > 0 else 0); e2 = 1 if g2.b < 0 else (-1 if g2.a > 0 else 0)
        Ny1, Py1 = NP(y1, e1); Ny2, Py2 = NP(y2, e2)
        v = (cA * (N1 - Ny1) + cB * (Ny1 - Ny2) + d1 * (P1 - Py1) + d2 * (Py1 - Py2)) / 2 + d3 * Py2 + r
        best = v.a if best is None else max(best, v.a)
    return best, M
def prove(C):
    st = dict(proved=0, outside=0, corner=0, failed=0, minlb=None)
    def run(a0, a1, b0, b1, d=0):
        if a1 * a1 + b1 * b1 < mpf(8) / 9: st["outside"] += 1; return
        if a0 + b0 > 2 - W0: st["corner"] += 1; return
        f, M = lower(a0, a1, b0, b1, C)
        if f > 0:
            st["proved"] += 1; lb = 5 * f / M.b
            if st["minlb"] is None or lb < st["minlb"]: st["minlb"] = lb
            return
        if d > 24: st["failed"] += 1; print("FAILED box", a0, a1, b0, b1, f, flush=True); return
        if a1 - a0 >= b1 - b0:
            m = (a0 + a1) / 2; run(a0, m, b0, b1, d + 1); run(m, a1, b0, b1, d + 1)
        else:
            m = (b0 + b1) / 2; run(a0, a1, b0, m, d + 1); run(a0, a1, m, b1, d + 1)
    for i in range(8):
        for j in range(8):
            run(mpf(i) / 8, mpf(i + 1) / 8, mpf(j) / 8, mpf(j + 1) / 8)
    return st
if __name__ == "__main__":
    reps = sys.argv[2:] or [l.split()[0].split(",")[0] for l in open("orbits_r5.out").read().splitlines()[1:] if "min F/m" in l and not l.startswith("  12403")]
    out, tag = [], iv.nstr(iv.mpf(W0), 3).strip("[]").split(",")[0]
    for rep in reps:
        C = coeffs(tuple(int(c) for c in rep))
        thin = max(mpf("0.6") * cA * (1 - W0) - (U + 6) * W0 for cA, cB, d1, d2, d3, U in C)
        st = prove(C)
        out.append("tau=%s  proved %d, outside %d, corner %d (thin bound %s), FAILED %d, min lower bound of Gq %s" % (
            rep, st["proved"], st["outside"], st["corner"], iv.nstr(iv.mpf(thin), 4), st["failed"],
            iv.nstr(iv.mpf(st["minlb"]), 5) if st["minlb"] is not None else "-"))
        print(out[-1], flush=True)
        open(__import__("os").environ.get("GQ_OUT") or "gq_cross_iv_w%s.out" % tag, "w", newline="\n").write("\n".join(out) + "\n")
