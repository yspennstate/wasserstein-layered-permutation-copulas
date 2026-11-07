"""tau = 1302 on any band (a,b) of S = x1+x2 (density L(s) = min(s,2-s)): exact piecewise-polynomial integrals (Simpson on
pieces between breakpoints, exact for the cubics that occur), quantiles by bisection. F = 4A + 2B - 3(S4-S1) - (S3-S2).
Scans the bands crossing s = 1 with mass <= MMAX on a (q, m) grid and reports the minimum of F/m and of F."""
import sys
from math import sqrt
L = lambda s: min(s, 2 - s)
def integ(f, lo, hi, bps):
    pts = sorted({lo, hi} | {p for p in bps if lo < p < hi}); r = 0.0
    for x0, x1 in zip(pts, pts[1:]): r += (x1 - x0) / 6 * (f(x0) + 4 * f((x0 + x1) / 2) + f(x1))
    return r
def bis(g, lo, hi, target):
    for _ in range(100):
        mid = (lo + hi) / 2
        if g(mid) < target: lo = mid
        else: hi = mid
    return (lo + hi) / 2
def F(a, b):
    bs = [1.0]
    M = integ(L, a, b, bs)
    sq = [a] + [bis(lambda x: integ(L, a, x, bs), a, b, p * M) for p in (.25, .5, .75)] + [b]
    S = [integ(lambda s: s * L(s), sq[j], sq[j + 1], bs) / (M / 4) for j in range(4)]
    h = lambda x: max(0.0, min(b, 2 - x) - max(a, x))          # |v| density (both signs of v counted once each side)
    xb = [a, 2 - b, 1.0]; top = max(L(s) for s in (a, b, 1.0) if a <= s <= b)
    Mh = integ(h, 0, top, xb)
    xm = bis(lambda x: integ(h, 0, x, xb), 0, top, Mh / 2)
    B = integ(lambda x: x * h(x), 0, xm, xb) / (Mh / 2); A = integ(lambda x: x * h(x), xm, top, xb) / (Mh / 2)
    return 4 * A + 2 * B - 3 * (S[3] - S[0]) - (S[2] - S[1])
C = lambda s: s * s / 2 if s <= 1 else 1 - (2 - s) ** 2 / 2
Q = lambda p: sqrt(2 * p) if p <= 0.5 else 2 - sqrt(2 * (1 - p))
if __name__ == "__main__":
    from win_r import F as Fpoly
    import tau1302_lower as T
    out = ["checks: (0.5,1.0) exact-lower %.12f any %.12f | (0.5,1.4) poly %.10f any %.10f | (0.2,1.8) poly %.10f any %.10f" % (
        T.F(0.5)[0], F(0.5, 1.0), Fpoly(0.5, 1.4, (1, 3, 0, 2)), F(0.5, 1.4), Fpoly(0.2, 1.8, (1, 3, 0, 2)), F(0.2, 1.8))]
    MMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 4 / 9; G = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    best = (9, 0, 0); bestF = (9, 0, 0)
    for i in range(1, G + 1):
        m = MMAX * i / G
        for j in range(G + 1):
            q = 0.5 - m + m * j / G                             # crossing: q < 1/2 < q + m
            if q < 0 or q + m > 1: continue
            a, b = Q(q), Q(q + m)
            if not (a < 1 < b): continue
            f = F(a, b)
            if f / m < best[0]: best = (f / m, a, b)
            if f < bestF[0]: bestF = (f, a, b)
    out.append("crossing bands, mass <= %.4f, grid %d: min F/m %+.6f at (a,b) = (%.4f,%.4f); min F %+.6f at (%.4f,%.4f)" % (
        MMAX, G, best[0], best[1], best[2], bestF[0], bestF[1], bestF[2]))
    print("\n".join(out)); open("tau1302_any_%s.out" % sys.argv[1] if len(sys.argv) > 1 else "tau1302_any.out", "w", newline="\n").write("\n".join(out) + "\n")
