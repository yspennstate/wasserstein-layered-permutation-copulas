"""Rigorous positivity of F0 = 4A - 3(S4-S1) - (S3-S2) (a lower bound for the 'v then s' cut of tau = 1230) on bands crossing s = 1, mass <= 4/9, away from the thin corner. Variables
alpha = a, beta = 2 - b in [0,1]; mass M = (2 - alpha^2 - beta^2)/2 (the square has mass 1), so mass <= 4/9 is
alpha^2 + beta^2 >= 10/9. With K(x) = int_0^x s L(s) ds (x^3/3 for x <= 1, x^2 - x^3/3 - 1/3 for x >= 1), N(x) = int_0^x
y h(y) dy (h the |v| density), x1..x3 the s-quartiles and xm the |v|-median,
   M F0 = 2 (4 N(1) - 4 N(xm)) + 4 (2 K(x1) + 2 K(x2) + 2 K(x3) - 3 K(a) - 3 K(b)).
Enclosures: x_p is increasing in a and in b (dx/da = (1-p)L(a)/L(x), dx/db = p L(b)/L(x)), so its range on a box comes
from two corners; K and N are increasing in x; xm is located by bisection on the sign of H(x) - M/2 over the whole box
(H increasing in x). Everything in mpmath.iv (outward rounding). Boxes are bisected until M F > 0; boxes inside the thin
corner a + (2-b) > 2 - W0 are set aside. Reports the boxes, the smallest lower bound and the corner left."""
import sys
from mpmath import iv, mpf
iv.dps = 25
def hull(*xs):
    lo = min(x.a for x in xs); hi = max(x.b for x in xs); return iv.mpf([lo, hi])
def imin(x, c):
    return iv.mpf([min(x.a, c.a), min(x.b, c.b)])
def q(al, be, p):                          # point quantile (al, be point intervals)
    M = (2 - al * al - be * be) / 2; left = (1 - al * al) / 2
    c1 = iv.sqrt(al * al + 2 * p * M); c2 = 2 - iv.sqrt(be * be + 2 * (1 - p) * M)
    d = p * M - left
    if d.b <= 0: return c1
    if d.a >= 0: return c2
    return hull(c1, c2)
def K(x):                                  # x an interval; K increasing
    def k(y):
        Y = iv.mpf(y)
        return Y ** 3 / 3 if y <= 1 else Y * Y - Y ** 3 / 3 - iv.mpf(1) / 3
    return iv.mpf([k(x.a).a, k(x.b).b])
def H(x, al, be):                          # mass of |v| < x, x in [0,1]
    b = 2 - be; mb = imin(x, be); ma = imin(x, al)
    return (b * mb + (2 * x - x * x / 2) - (2 * mb - mb * mb / 2)) - (al * ma + x * x / 2 - ma * ma / 2)
def N(x, al, be):
    b = 2 - be; mb = imin(x, be); ma = imin(x, al)
    return (b * mb * mb / 2 + (x * x - x ** 3 / 3) - (mb * mb - mb ** 3 / 3)) - (al * ma * ma / 2 + (x ** 3 - ma ** 3) / 3)
def median(al, be, M):
    lo, hi = mpf(0), mpf(1)                # largest x with H < M/2 surely
    for _ in range(40):
        mid = (lo + hi) / 2
        if (H(iv.mpf(mid), al, be) - M / 2).b < 0: lo = mid
        else: hi = mid
    L0 = lo; lo, hi = mpf(0), mpf(1)       # smallest x with H > M/2 surely
    for _ in range(40):
        mid = (lo + hi) / 2
        if (H(iv.mpf(mid), al, be) - M / 2).a > 0: hi = mid
        else: lo = mid
    return iv.mpf([L0, hi])
def MF(a0, a1, b0, b1):                    # alpha in [a0,a1], beta in [b0,b1]
    al = iv.mpf([a0, a1]); be = iv.mpf([b0, b1]); M = (2 - al * al - be * be) / 2
    xs = [hull(q(iv.mpf(a0), iv.mpf(b1), p), q(iv.mpf(a1), iv.mpf(b0), p)) for p in (iv.mpf("0.25"), iv.mpf("0.5"), iv.mpf("0.75"))]
    xm = median(al, be, M)
    Nxm = hull(N(iv.mpf(xm.a), al, be), N(iv.mpf(xm.b), al, be))
    Ka = al ** 3 / 3; Kb = K(2 - be)
    return 2 * (4 * N(iv.mpf(1), al, be) - 4 * Nxm) + 4 * (2 * (K(xs[0]) + K(xs[1]) + K(xs[2])) - 3 * Ka - 3 * Kb), M
W0 = mpf(sys.argv[1]) if len(sys.argv) > 1 else mpf("0.05")
stats = dict(proved=0, outside=0, corner=0, failed=0, minlb=None)
def run(a0, a1, b0, b1, d=0):
    if a1 * a1 + b1 * b1 < mpf(10) / 9: stats["outside"] += 1; return
    if a0 + b0 > 2 - W0: stats["corner"] += 1; return
    f, M = MF(a0, a1, b0, b1)
    if f.a > 0:
        stats["proved"] += 1; lb = f.a / M.b
        if stats["minlb"] is None or lb < stats["minlb"][0]: stats["minlb"] = (lb, a0, a1, b0, b1)
        return
    if d > 22: stats["failed"] += 1; print("FAILED box", a0, a1, b0, b1, f); return
    if a1 - a0 >= b1 - b0:
        m = (a0 + a1) / 2; run(a0, m, b0, b1, d + 1); run(m, a1, b0, b1, d + 1)
    else:
        m = (b0 + b1) / 2; run(a0, a1, b0, m, d + 1); run(a0, a1, m, b1, d + 1)
if __name__ == "__main__":
    from tau_rest import closed
    out = []
    for a, be in [(0.5, 0.6), (0.9963, 2 - 1.6557), (0.7, 0.9)]:
        f, M = MF(mpf(a), mpf(a), mpf(be), mpf(be))
        out.append("point check alpha %.4f beta %.4f: interval F %s, float F0 %.10f" % (a, be, iv.nstr(f / M, 12), closed(a, 2 - be)["1302"] - 2 * closed(a, 2 - be)["B"]))
    print("\n".join(out), flush=True)
    G = 8
    for i in range(G):
        for j in range(G):
            run(mpf(i) / G, mpf(i + 1) / G, mpf(j) / G, mpf(j + 1) / G)
    out.append("W0 = %s: boxes proved %d, outside the mass range %d, thin-corner boxes set aside %d, FAILED %d; min lower bound of F/M %s on alpha [%s,%s] beta [%s,%s]" % (
        (iv.nstr(W0, 4), stats["proved"], stats["outside"], stats["corner"], stats["failed"]) + tuple(iv.nstr(z, 6) for z in stats["minlb"])))
    print(out[-1]); open("tau1230_cross_iv_%s.out" % iv.nstr(W0, 4), "w", newline="\n").write("\n".join(out) + "\n")
