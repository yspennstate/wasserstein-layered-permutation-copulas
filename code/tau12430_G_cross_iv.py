"""tau = 12430, the cut G of tau12430_G.py on bands a < s < b crossing s = 1 with mass M <= 5/9, proved positive in interval
arithmetic (mpmath.iv). Variables alpha = a, beta = 2 - b in [0,1]; M = (2 - alpha^2 - beta^2)/2, so M <= 5/9 is
alpha^2 + beta^2 >= 8/9. At |v| = y the band carries s in (max(alpha,y), 2 - max(beta,y)), of length
h(y) = 2 - max(beta,y) - max(alpha,y) (the square has mass 1, the density in (s,v) is 1/2, two signs of v). With
  H(x) = int_0^x h,  N(x) = int_0^x y h,  P(x) = int_0^x ((2 - max(beta,y))^2 - max(alpha,y)^2)/2 dy,
  K(x) = int_0^x s L(s) ds (L(s) = min(s, 2-s)),
the v-quintile points are y1 = H^-1(3M/5), y2 = H^-1(M/5), the s-quintile points x1..x4 (x0 = a, x5 = b), and
  (M/5) G = 3N(1) - (5/2)N(y1) - (1/2)N(y2) + P(1) - (3/2)P(y1) - (1/2)P(y2) - 4K(a) - 4K(b) + 2(K(x1)+K(x2)+K(x3)+K(x4)).
N, P, K, H are increasing in x, so (M/5)G is decreasing in y1, y2 and increasing in x1..x4: a lower bound uses the upper
ends of the y enclosures (bisection on the sign of H - level over the box) and the lower ends of the x enclosures (x_p is
increasing in a and in b). Boxes are bisected until the bound is positive; boxes in the thin corner alpha + beta > 2 - W0
are set aside. Usage: tau12430_G_cross_iv.py [W0]."""
import sys
from mpmath import iv, mpf
iv.dps = 25
def hull(*xs):
    return iv.mpf([min(x.a for x in xs), max(x.b for x in xs)])
def imin(x, c):
    return iv.mpf([min(x.a, c.a), min(x.b, c.b)])
def q(al, be, p):                          # s-quantile at point (al, be)
    M = (2 - al * al - be * be) / 2; left = (1 - al * al) / 2
    c1 = iv.sqrt(al * al + 2 * p * M); c2 = 2 - iv.sqrt(be * be + 2 * (1 - p) * M)
    d = p * M - left
    if d.b <= 0: return c1
    if d.a >= 0: return c2
    return hull(c1, c2)
def K(x):
    def k(y):
        Y = iv.mpf(y)
        return Y ** 3 / 3 if y <= 1 else Y * Y - Y ** 3 / 3 - iv.mpf(1) / 3
    return iv.mpf([k(x.a).a, k(x.b).b])
def _I1(x, c):                             # int_0^x max(c,y) dy
    m = imin(x, c); return c * m + (x * x - m * m) / 2
def _Iy(x, c):                             # int_0^x y max(c,y) dy
    m = imin(x, c); return c * m * m / 2 + (x ** 3 - m ** 3) / 3
def _I2(x, c):                             # int_0^x max(c,y)^2 dy
    m = imin(x, c); return c * c * m + (x ** 3 - m ** 3) / 3
def H(x, al, be): return 2 * x - _I1(x, be) - _I1(x, al)
def N(x, al, be): return x * x - _Iy(x, be) - _Iy(x, al)
def P(x, al, be): return (4 * x - 4 * _I1(x, be) + _I2(x, be) - _I2(x, al)) / 2
def yq(al, be, level):                     # enclosure of H^-1(level) over the box
    lo, hi = mpf(0), mpf(1)
    for _ in range(40):
        mid = (lo + hi) / 2
        if (H(iv.mpf(mid), al, be) - level).b < 0: lo = mid
        else: hi = mid
    L0 = lo; lo, hi = mpf(0), mpf(1)
    for _ in range(40):
        mid = (lo + hi) / 2
        if (H(iv.mpf(mid), al, be) - level).a > 0: hi = mid
        else: lo = mid
    return iv.mpf([L0, hi])
def MG(a0, a1, b0, b1):                    # enclosure (lower end valid) of (M/5) G over the box, and M
    al = iv.mpf([a0, a1]); be = iv.mpf([b0, b1]); M = (2 - al * al - be * be) / 2
    xs = [q(iv.mpf(a0), iv.mpf(b1), iv.mpf(p)).a for p in ("0.2", "0.4", "0.6", "0.8")]
    xs = [min(x, q(iv.mpf(a1), iv.mpf(b0), iv.mpf(p)).a) for x, p in zip(xs, ("0.2", "0.4", "0.6", "0.8"))]
    y1 = iv.mpf(yq(al, be, 3 * M / 5).b); y2 = iv.mpf(yq(al, be, M / 5).b)
    one = iv.mpf(1)
    v = (3 * N(one, al, be) - N(y1, al, be) * 5 / 2 - N(y2, al, be) / 2
         + P(one, al, be) - P(y1, al, be) * 3 / 2 - P(y2, al, be) / 2)
    r = -4 * al ** 3 / 3 - 4 * K(2 - be) + 2 * sum(K(iv.mpf(x)) for x in xs)
    return v + r, M
def MG_point(a, b):                        # exact-point version (for the checks)
    al, be = iv.mpf(a), iv.mpf(2 - b); M = (2 - al * al - be * be) / 2
    xs = [q(al, be, iv.mpf(p)) for p in ("0.2", "0.4", "0.6", "0.8")]
    y1 = yq(al, be, 3 * M / 5); y2 = yq(al, be, M / 5); one = iv.mpf(1)
    v = (3 * N(one, al, be) - N(y1, al, be) * 5 / 2 - N(y2, al, be) / 2
         + P(one, al, be) - P(y1, al, be) * 3 / 2 - P(y2, al, be) / 2)
    return (v - 4 * al ** 3 / 3 - 4 * K(2 - be) + 2 * sum(K(x) for x in xs)) * 5, M
W0 = mpf(sys.argv[1]) if len(sys.argv) > 1 else mpf("0.05")
stats = dict(proved=0, outside=0, corner=0, failed=0, minlb=None)
def run(a0, a1, b0, b1, d=0):
    if a1 * a1 + b1 * b1 < mpf(8) / 9: stats["outside"] += 1; return
    if a0 + b0 > 2 - W0: stats["corner"] += 1; return
    f, M = MG(a0, a1, b0, b1)
    if f.a > 0:
        stats["proved"] += 1; lb = 5 * f.a / M.b
        if stats["minlb"] is None or lb < stats["minlb"][0]: stats["minlb"] = (lb, a0, a1, b0, b1)
        return
    if d > 24: stats["failed"] += 1; print("FAILED box", a0, a1, b0, b1, f, flush=True); return
    if a1 - a0 >= b1 - b0:
        m = (a0 + a1) / 2; run(a0, m, b0, b1, d + 1); run(m, a1, b0, b1, d + 1)
    else:
        m = (b0 + b1) / 2; run(a0, a1, b0, m, d + 1); run(a0, a1, m, b1, d + 1)
if __name__ == "__main__":
    from tau12430_G_iv import grid
    out = []
    for a, b in [(0.9428, 2.0), (0.6, 1.5), (0.95, 1.2), (0.3, 1.7), (0.99, 1.02)]:
        g, M = MG_point(a, b)
        out.append("point check band (%.4f,%.4f): interval G %s, grid G %+.5f" % (a, b, iv.nstr(g, 10), grid(a, b)))
        print(out[-1], flush=True)
    Gd = 8
    for i in range(Gd):
        for j in range(Gd):
            run(mpf(i) / Gd, mpf(i + 1) / Gd, mpf(j) / Gd, mpf(j + 1) / Gd)
    mn = stats["minlb"]
    out.append("W0 = %s: boxes proved %d, outside the mass range %d, thin-corner boxes set aside %d, FAILED %d; min lower bound of G/M %s on alpha [%s,%s] beta [%s,%s]" % (
        (iv.nstr(W0, 4), stats["proved"], stats["outside"], stats["corner"], stats["failed"]) + tuple(iv.nstr(iv.mpf(z), 6) for z in mn)))
    print(out[-1]); open("tau12430_G_cross_iv_w%s.out" % iv.nstr(W0, 4).strip("[]").split(",")[0], "w", newline="\n").write("\n".join(out) + "\n")
