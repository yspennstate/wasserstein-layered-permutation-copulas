"""tau = 12403, the 's then v' cut
  Hs = 4 E[|v| : s in (x1, x3)] - (S_2 - S_1) - (S_4 - S_0),
with x0 = a < x1 < ... < x4 < x5 = b the s-quintile points of the band a < s < b and S_j the s-quintile means, proved
positive in interval arithmetic (mpmath.iv) on every band of mass <= 5/9. The square in (s, v) is |v| < L(s) = min(s, 2-s)
with density 1/2, so given s, |v| is uniform on (0, L(s)) and s has density L(s).

Bottom bands (b <= 1): every quantity is a length, Hs(a, b) = b Hs(t, 1) with t = a/b, x_p = sqrt(t^2 + p (1 - t^2)),
  E[|v| : s in (x1,x3)] = (x1^2 + x1 x3 + x3^2) / (3 (x1 + x3)),  S_j = 2 (x_j^2 + x_j x_{j+1} + x_{j+1}^2) / (3 (x_j + x_{j+1})).
Top bands (a >= 1): s' = 2 - s maps the band to the bottom band (t, 1) times 2 - a, t = (2 - b)/(2 - a); v is unchanged,
s-strip j is s'-strip 4 - j and S_j = 2 - S'_{4-j}, so Hs = (2 - a) [4 E'[|v| : s' in (x'2, x'4)] - (S'3 - S'2) - (S'4 - S'0)].
Both are closed forms in t on [0, 1] with no division by zero (x_j + x_{j+1} >= x1 > 0), so t is bisected directly.

Crossing bands (a < 1 < b), alpha = a, beta = 2 - b in [0, 1], M = (2 - alpha^2 - beta^2)/2: with
  Lam(x) = int_0^x L^2/2,  K(x) = int_0^x s L(s) ds,
  (M/5) Hs = 2 Lam(x3) - 2 Lam(x1) - K(x3) + 2 K(x2) + K(x4) - K(a) - K(b).
d/dx1 = -L(x1)^2 < 0, d/dx3 = L(x3)(L(x3) - x3) <= 0, d/dx2 = 2 x2 L(x2) > 0, d/dx4 = x4 L(x4) > 0, and -K(a) - K(b)
decreases in a and b, so a lower bound over a box takes x1, x3 at the upper ends of their enclosures, x2, x4 at the lower
ends (x_p increases in a and in b), and K(a), K(b) at the upper ends. Thin corner alpha + beta > 2 - W0: the band lies in
(1 - W0, 1 + W0) and has width < W0, so L >= 1 - W0 there, 4 E[|v| | .] >= 2 (1 - W0), and each S difference is < W0:
Hs >= 2 - 4 W0 > 0. Usage: hs12403_iv.py [W0]."""
import sys
from mpmath import iv, mpf
from tau12430_G_cross_iv import q, K
iv.dps = 25
P = ("0.2", "0.4", "0.6", "0.8")
def _bot(t):
    xs = [t] + [iv.sqrt(t * t + iv.mpf(p) * (1 - t * t)) for p in P] + [iv.mpf(1)]
    S = [2 * (xs[j] ** 2 + xs[j] * xs[j + 1] + xs[j + 1] ** 2) / (3 * (xs[j] + xs[j + 1])) for j in range(5)]
    Ev = lambda i, k: (xs[i] ** 2 + xs[i] * xs[k] + xs[k] ** 2) / (3 * (xs[i] + xs[k]))
    return xs, S, Ev
def hs_bottom(t):
    xs, S, Ev = _bot(t); return 4 * Ev(1, 3) - (S[2] - S[1]) - (S[4] - S[0])
def hs_top(t):
    xs, S, Ev = _bot(t); return 4 * Ev(2, 4) - (S[3] - S[2]) - (S[4] - S[0])
def check1(f, u0, u1, acc, d=0):
    g = f(iv.mpf([u0, u1]))
    if g.a > 0: acc.append((u0, u1, g.a)); return True
    if d > 40: return False
    m = (u0 + u1) / 2
    return check1(f, u0, m, acc, d + 1) and check1(f, m, u1, acc, d + 1)
def Lam(x):
    def l(y):
        Y = iv.mpf(y)
        return Y ** 3 / 6 if y <= 1 else (2 - (2 - Y) ** 3) / 6
    return iv.mpf([l(x.a).a, l(x.b).b])
def MH(a0, a1, b0, b1):                    # lower bound of (M/5) Hs over the box (alpha, beta), and M
    al = iv.mpf([a0, a1]); be = iv.mpf([b0, b1]); M = (2 - al * al - be * be) / 2
    lo = [q(iv.mpf(a0), iv.mpf(b1), iv.mpf(p)).a for p in P]
    hi = [q(iv.mpf(a1), iv.mpf(b0), iv.mpf(p)).b for p in P]
    x1, x2, x3, x4 = iv.mpf(hi[0]), iv.mpf(lo[1]), iv.mpf(hi[2]), iv.mpf(lo[3])
    v = 2 * Lam(x3) - 2 * Lam(x1) - K(x3) + 2 * K(x2) + K(x4) - K(iv.mpf(a1)) - K(iv.mpf(2 - b0))
    return v, M
def MH_point(a, b):
    al, be = iv.mpf(a), iv.mpf(2 - b); M = (2 - al * al - be * be) / 2
    x1, x2, x3, x4 = [q(al, be, iv.mpf(p)) for p in P]
    return (2 * Lam(x3) - 2 * Lam(x1) - K(x3) + 2 * K(x2) + K(x4) - K(al) - K(2 - be)) * 5 / M
W0 = mpf(sys.argv[1]) if len(sys.argv) > 1 else mpf("0.05")
stats = dict(proved=0, outside=0, corner=0, failed=0, minlb=None)
def run(a0, a1, b0, b1, d=0):
    if a1 * a1 + b1 * b1 < mpf(8) / 9: stats["outside"] += 1; return
    if a0 + b0 > 2 - W0: stats["corner"] += 1; return
    f, M = MH(a0, a1, b0, b1)
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
    out = []
    for a, b in ((0.2, 0.5), (1.7, 2.0), (0.6, 1.4)):
        if b <= 1: h = b * hs_bottom(iv.mpf(a / b))
        elif a >= 1: h = (2 - a) * hs_top(iv.mpf((2 - b) / (2 - a)))
        else: h = MH_point(a, b)
        out.append("point check band (%.2f,%.2f): interval Hs %s" % (a, b, iv.nstr(h, 8))); print(out[-1], flush=True)
    for name, f in (("bottom", hs_bottom), ("top", hs_top)):
        acc = []; ok = check1(f, mpf(0), mpf(1), acc); mn = min(acc, key=lambda z: z[2])
        out.append("%s bands, t in [0,1]: proved Hs > 0: %s; %d boxes; smallest lower bound %s at t in [%s, %s]" % (
            name, ok, len(acc), iv.nstr(mn[2], 6), iv.nstr(mn[0], 6), iv.nstr(mn[1], 6)))
        print(out[-1], flush=True)
    for i in range(8):
        for j in range(8):
            run(mpf(i) / 8, mpf(i + 1) / 8, mpf(j) / 8, mpf(j + 1) / 8)
    mn = stats["minlb"]
    out.append("crossing, W0 = %s: boxes proved %d, outside the mass range %d, thin-corner boxes %d (Hs >= 2 - 4 W0 = %s), FAILED %d; min lower bound of Hs/M %s" % (
        iv.nstr(iv.mpf(W0), 4), stats["proved"], stats["outside"], stats["corner"], iv.nstr(2 - 4 * iv.mpf(W0), 4),
        stats["failed"], iv.nstr(iv.mpf(mn[0]), 6) if mn else "-"))
    print(out[-1]); open("hs12403_iv.out", "w", newline="\n").write("\n".join(out) + "\n")
