"""Window lemma for r = 3, the 3-cycle tau = (1,2,0) on values k, k+1, k+2 (sigma = id elsewhere).
Coordinates s = x1+x2, v = x1-x2; the window's strips form the band U = {a < s < b, |v| < L(s)}, L = min(s, 2-s),
uniform in (s, v). Points of tau in (u = p1+p2, v = p1-p2), offsets in units h = 1/n about the band's centre:
   (-1,-1), (+1,-1), (0,+2);   identity: (-2,0), (0,0), (+2,0).
Cut 'v then s': top cell {v > c} (a third of the band) -> (0,+2); the rest split at s = d into halves -> (-1,-1), (+1,-1).
Then n^2 D >= F/2 with F = sum_cells (u'.E[s|cell] + v'.E[v|cell]) - sum_j u'_j E[s|strip j] (each cell mass 1/n).
Polygon clipping in (s, v), exact up to bisection. Prints min F over a grid of bands (a, b) != (0, 2), and the corner band."""
import sys
import numpy as np

def clip(P, a, b, c):          # keep a*s + b*v <= c
    out = []
    for i in range(len(P)):
        p, q = P[i], P[(i + 1) % len(P)]
        fp, fq = a * p[0] + b * p[1] - c, a * q[0] + b * q[1] - c
        if fp <= 0: out.append(p)
        if fp * fq < 0:
            t = fp / (fp - fq); out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out

def am(P):                    # area, centroid
    A = cx = cy = 0.0
    for i in range(len(P)):
        (x0, y0), (x1, y1) = P[i], P[(i + 1) % len(P)]
        cr = x0 * y1 - x1 * y0; A += cr; cx += (x0 + x1) * cr; cy += (y0 + y1) * cr
    if abs(A) < 1e-300: return 0.0, (0.0, 0.0)
    return A / 2, (cx / (3 * A), cy / (3 * A))

SQ = [(0.0, 0.0), (1.0, -1.0), (2.0, 0.0), (1.0, 1.0)]     # the unit square in (s, v)

def band(a, b):
    return clip(clip(SQ, -1, 0, -a), 1, 0, b)

def bis(f, lo, hi, target, it=200):
    for _ in range(it):
        mid = (lo + hi) / 2
        if f(mid) < target: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def F(a, b):
    U = band(a, b); M, _ = am(U)
    # identity strips at band-mass thirds (s quantiles)
    q1 = bis(lambda x: am(clip(U, 1, 0, x))[0], a, b, M / 3)
    q2 = bis(lambda x: am(clip(U, 1, 0, x))[0], a, b, 2 * M / 3)
    st = [clip(U, 1, 0, q1), clip(clip(U, -1, 0, -q1), 1, 0, q2), clip(U, -1, 0, -q2)]
    Es = [am(P)[1][0] for P in st]
    idv = -2 * Es[0] + 2 * Es[2]
    vmax = max(abs(p[1]) for p in U)
    c = bis(lambda x: -am(clip(U, 0, -1, -x))[0], -vmax, vmax, -M / 3)    # mass of {v > x} decreasing in x
    top = clip(U, 0, -1, -c); rest = clip(U, 0, 1, c)
    d = bis(lambda x: am(clip(rest, 1, 0, x))[0], a, b, M / 3)
    lo = clip(rest, 1, 0, d); hi = clip(rest, -1, 0, -d)
    (mt, (st_, vt)), (ml, (sl, vl)), (mh, (sh, vh)) = am(top), am(lo), am(hi)
    err = max(abs(mt - M / 3), abs(ml - M / 3), abs(mh - M / 3)) / M
    tv = 0 * st_ + 2 * vt + (-1) * sl + (-1) * vl + 1 * sh + (-1) * vh
    return tv - idv, err

if __name__ == "__main__":
    G = int(sys.argv[1]) if len(sys.argv) > 1 else 41
    out = []
    best = (9, None)
    for a in np.linspace(0, 2, G):
        for b in np.linspace(0, 2, G):
            if b <= a + 1e-9 or (a < 1e-12 and b > 2 - 1e-12): continue
            f, e = F(a, b)
            # F scales with the band's width; report F / (b - a) too
            if f / (b - a) < best[0]: best = (f / (b - a), (a, b, f, e))
    out.append("grid G=%d: min F/(b-a) = %+.6f at a=%.4f b=%.4f (F=%+.3e, mass err %.1e)" % ((G, best[0]) + best[1]))
    for n in (4, 6, 8, 16, 64, 256):
        cq = lambda j: (2 * j / n) ** 0.5 if j <= n / 2 else 2 - (2 * (n - j) / n) ** 0.5
        rows = []
        for k in range(n - 2):
            f, e = F(cq(k), cq(k + 3))
            rows.append((f / 2, k))
        rows.sort()
        out.append("n=%d: min over k of n^2 D-bound (F/2) %+.5f at k=%d; corner k=0 %+.5f, k=n-3 %+.5f, middle %+.5f" % (
            n, rows[0][0], rows[0][1], [r for r in rows if r[1] == 0][0][0], [r for r in rows if r[1] == n - 3][0][0],
            [r for r in rows if r[1] == (n - 3) // 2][0][0]))
    print("\n".join(out))
    open(__file__.replace(".py", ".out"), "w", newline="\n").write("\n".join(out) + "\n")
