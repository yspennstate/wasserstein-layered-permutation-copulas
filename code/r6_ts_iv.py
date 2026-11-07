"""r = 6: the partition 'thirds of S, each split at V = 0' (r6_thirds_sign.py) with an assignment of the six points to the six
cells, proved F > 0 in interval arithmetic for every indecomposable non-reversal block of six values on every band of mass
at most 6/9.

Notation (mass of the square 1, density L(s) = min(s, 2-s) in s). For a band [x, y] write e = E[S] and l = E[L(S)];
given s, V is uniform on (-L(s), L(s)), so E[V | V > 0] = l/2. With x_0 = a < x_1 < ... < x_6 = b the sextiles of S on
U = {a < s < b}, e_j and l_j the means over the sextile [x_j, x_{j+1}], the cells are (third k, sign) with
E[S] = (e_{2k} + e_{2k+1})/2 and E[V] = +-(l_{2k} + l_{2k+1})/4, all of mass M/6. For an assignment pi of points to cells
  F = sum_j (A_j e_j + B_j l_j),   A_j = (u'_minus + u'_plus)/2 - (2j - 5),   B_j = (v'_plus - v'_minus)/4,
where minus / plus are the points given the lower / upper cell of third j // 2. Any assignment gives a valid partition, so a
positive lower bound for one assignment proves the box (the assignment is the best one at the box centre, in floats).

Families of bands. (B) b <= 1: the similarity (s, v) -> (s, v)/b maps U to {a/b < s < 1} and F scales by b, so it suffices
to take b = 1, a = t in [0, 1] (t = 1 is the thin limit, where all means tend to 1 and F stays finite). (T) a >= 1: the
reflection s -> 2 - s maps it to family B with u' replaced by -u' (Section 4 symmetries). (C) a < 1 < b: alpha = a,
beta = 2 - b in [0, 1], mass (2 - alpha^2 - beta^2)/2 <= 6/9, i.e. alpha^2 + beta^2 >= 2/3.

Means over [x, y]: on [x, y] within [0, 1], e = l = (2/3)(x^2 + xy + y^2)/(x + y); within [1, 2], with p = 2 - x, q = 2 - y,
e = 2 - (2/3)(p^2 + pq + q^2)/(p + q) and l = 2 - e; across 1, the mass-weighted mixture of the two pieces, weight
w = m1/(m1 + m2) in [0, 1] (m1 = (1 - x^2)/2, m2 = (1 - q^2)/2). Each enclosure is intersected with the a priori range of the mean
(e in [x, y], l between the min and max of L on [x, y]); on thin bands across s = 1 this is what carries the bound. Sextiles: x_j = Hinv((1 - j/6) H(a) + (j/6) H(b)) with
H(s) = s^2/2 for s <= 1 and 1 - (2-s)^2/2 for s >= 1; x_j increases in a and in b, so its enclosure over a box comes from the
two corners. Usage: r6_ts_iv.py [taus...] (default: all 460)."""
import sys, itertools
import numpy as np
from mpmath import iv, mpf
from scipy.optimize import linear_sum_assignment
iv.dps = 30
TWO3 = iv.mpf(2) / 3
ONE = iv.mpf(1)

def hull(*xs):
    return iv.mpf([min(x.a for x in xs), max(x.b for x in xs)])

def H_iv(s):                                  # s an mpf point
    S = iv.mpf(s)
    return S * S / 2 if s <= 1 else 1 - (2 - S) ** 2 / 2

def Hinv_iv(m):                               # m an interval
    if m.b <= 0.5: return iv.sqrt(2 * m)
    if m.a >= 0.5: return 2 - iv.sqrt(2 * (1 - m))
    return hull(iv.sqrt(2 * iv.mpf([m.a, 0.5])), 2 - iv.sqrt(2 * (1 - iv.mpf([0.5, m.b]))))

def sextiles(a0, a1, b0, b1):
    Ha0, Ha1, Hb0, Hb1 = H_iv(a0), H_iv(a1), H_iv(b0), H_iv(b1)
    xs = [hull(iv.mpf(a0), iv.mpf(a1))]
    for j in range(1, 6):
        lo = Hinv_iv(((6 - j) * Ha0 + j * Hb0) / 6).a
        hi = Hinv_iv(((6 - j) * Ha1 + j * Hb1) / 6).b
        xs.append(iv.mpf([lo, hi]))
    xs.append(hull(iv.mpf(b0), iv.mpf(b1)))
    return xs

def imin1(x): return iv.mpf([min(x.a, 1), min(x.b, 1)])
def imax1(x): return iv.mpf([max(x.a, 1), max(x.b, 1)])

def tri(x, y):                                # (2/3)(x^2+xy+y^2)/(x+y), the mean of s on [x,y] in [0,1]
    return TWO3 * (x * x + x * y + y * y) / (x + y)

def means(x, y):                              # enclosures of (e, l) over the band [x, y]
    if y.b <= 1:
        e = tri(x, y); return e, e
    if x.a >= 1:
        e = 2 - tri(2 - y, 2 - x); return e, 2 - e
    x1, y1 = imin1(x), imin1(y); x2, y2 = imax1(x), imax1(y)
    p, q = 2 - x2, 2 - y2
    m1 = (y1 * y1 - x1 * x1) / 2; m2 = (p * p - q * q) / 2
    E1 = tri(x1, y1) if (x1 + y1).a > 0 else iv.mpf([0, 1])
    E2 = 2 - tri(q, p) if (p + q).a > 0 else iv.mpf([1, 2])
    den = m1 + m2
    if den.a > 0:
        w = m1 / den; w = iv.mpf([max(w.a, 0), min(w.b, 1)])
    else:
        w = iv.mpf([0, 1])
    e = E2 + w * (E1 - E2); l = w * E1 + (1 - w) * (2 - E2)
    return e, l

def Lrange(p, q):                             # min and max of L = min(s, 2-s) over s in [p, q]
    lo = min(min(p, 2 - p), min(q, 2 - q)); hi = 1 if p <= 1 <= q else max(min(p, 2 - p), min(q, 2 - q))
    return lo, hi

def cap(z, lo, hi):                           # intersect the enclosure z with the a priori range [lo, hi]
    a, b = max(z.a, lo), min(z.b, hi)
    if a > b: raise ValueError("empty intersection")
    return iv.mpf([a, b])

def means_capped(x, y):                       # e lies in [x, y] (a mean of S over the strip), l between min L and max L
    e, l = means(x, y); p, q = x.a, y.b
    lo, hi = Lrange(p, q)
    return cap(e, p, q), cap(l, lo, hi)

# float twin, for choosing the assignment
def Hf(s): return s * s / 2 if s <= 1 else 1 - (2 - s) ** 2 / 2
def Hinvf(m): return (2 * m) ** 0.5 if m <= 0.5 else 2 - (2 * (1 - m)) ** 0.5
def Kf(s): return s ** 3 / 3 if s <= 1 else s * s - s ** 3 / 3 - 1 / 3
def Jf(s): return s ** 3 / 3 if s <= 1 else (2 - (2 - s) ** 3) / 3      # int_0^s L^2 dt  (= mass * E[L])
def sextile_means_f(a, b):
    ma, mb = Hf(a), Hf(b); xs = [a] + [Hinvf(ma + j * (mb - ma) / 6) for j in range(1, 6)] + [b]
    out = []
    for j in range(6):
        m = Hf(xs[j + 1]) - Hf(xs[j])
        out.append(((Kf(xs[j + 1]) - Kf(xs[j])) / m, (Jf(xs[j + 1]) - Jf(xs[j])) / m))
    return out

def coefs(pts, assign):                       # assign[c] = point index of cell c; cells 2k (minus), 2k+1 (plus)
    A, B = [], []
    for j in range(6):
        k = j // 2; im, ip = assign[2 * k], assign[2 * k + 1]
        A.append((pts[im][0] + pts[ip][0]) / 2 - (2 * j - 5)); B.append((pts[ip][1] - pts[im][1]) / 4)
    return A, B

def best_assign(pts, a, b):
    em = sextile_means_f(a, b)
    cells = []
    for k in range(3):
        es = (em[2 * k][0] + em[2 * k + 1][0]) / 2; ev = (em[2 * k][1] + em[2 * k + 1][1]) / 4
        cells += [(es, -ev), (es, ev)]
    W = np.array([[p[0] * c[0] + p[1] * c[1] for c in cells] for p in pts])
    ri, ci = linear_sum_assignment(-W)
    assign = [None] * 6
    for i, c in zip(ri, ci): assign[c] = i
    F = W[ri, ci].sum() - sum((2 * j - 5) * em[j][0] for j in range(6))
    return assign, F

def F_iv(pts, a0, a1, b0, b1, assign):
    xs = sextiles(a0, a1, b0, b1)
    A, B = coefs(pts, assign)
    tot = iv.mpf(0)
    for j in range(6):
        e, l = means_capped(xs[j], xs[j + 1]); tot += A[j] * e + B[j] * l
    return tot

def prove_box(pts, a0, a1, b0, b1, st, depth, split2):
    am, bm = float((a0 + a1) / 2), float((b0 + b1) / 2)
    assign, ff = best_assign(pts, am, bm)
    lb = F_iv(pts, a0, a1, b0, b1, assign).a
    if lb > 0:
        st["boxes"] += 1
        if st["min"] is None or lb < st["min"]: st["min"] = lb
        return
    if depth > 30:
        st["failed"] += 1; st["fails"].append((float(a0), float(a1), float(b0), float(b1), ff)); return
    if split2 and (b1 - b0) > (a1 - a0):
        m = (b0 + b1) / 2
        prove_box(pts, a0, a1, b0, m, st, depth + 1, split2); prove_box(pts, a0, a1, m, b1, st, depth + 1, split2)
    else:
        m = (a0 + a1) / 2
        prove_box(pts, a0, m, b0, b1, st, depth + 1, split2); prove_box(pts, m, a1, b0, b1, st, depth + 1, split2)

def new_st(): return dict(boxes=0, failed=0, min=None, fails=[], outside=0)

def family_B(pts, st):                        # b = 1, a = t in [0, 1]
    for i in range(16):
        prove_box(pts, mpf(i) / 16, mpf(i + 1) / 16, mpf(1), mpf(1), st, 0, False)

def family_C(pts, st, G=8):                   # a = alpha, b = 2 - beta, alpha^2 + beta^2 >= 2/3
    def run(al0, al1, be0, be1, d):
        if al1 * al1 + be1 * be1 < mpf(2) / 3: st["outside"] += 1; return
        prove_cross(al0, al1, be0, be1, d)
    def prove_cross(al0, al1, be0, be1, d):
        am, bm = float((al0 + al1) / 2), 2 - float((be0 + be1) / 2)
        assign, ff = best_assign(pts, am, bm)
        lb = F_iv(pts, al0, al1, 2 - be1, 2 - be0, assign).a
        if lb > 0:
            st["boxes"] += 1
            if st["min"] is None or lb < st["min"]: st["min"] = lb
            return
        if d > 24:
            st["failed"] += 1; st["fails"].append((float(al0), float(al1), float(be0), float(be1), ff)); return
        if al1 - al0 >= be1 - be0:
            m = (al0 + al1) / 2; run(al0, m, be0, be1, d + 1); run(m, al1, be0, be1, d + 1)
        else:
            m = (be0 + be1) / 2; run(al0, al1, be0, m, d + 1); run(al0, al1, m, be1, d + 1)
    for i in range(G):
        for j in range(G):
            run(mpf(i) / G, mpf(i + 1) / G, mpf(j) / G, mpf(j + 1) / G, 0)

indec = lambda t: all(set(t[:j]) != set(range(j)) for j in range(1, len(t)))
if __name__ == "__main__":
    TAUS = [t for t in itertools.permutations(range(6)) if indec(t) and t != (5, 4, 3, 2, 1, 0)]
    sel = [tuple(int(c) for c in s) for s in sys.argv[1:]] or TAUS
    tag = "_sel" if sys.argv[1:] else ""
    out, allok, gmin = [], True, None
    for tau in sel:
        pts = [(i + tau[i] - 5, i - tau[i]) for i in range(6)]
        parts = []
        for name, P, fam in (("B", pts, family_B), ("T", [(-u, v) for u, v in pts], family_B), ("C", pts, family_C)):
            st = new_st(); fam(P, st); allok &= st["failed"] == 0
            if st["min"] is not None: gmin = st["min"] if gmin is None else min(gmin, st["min"])
            parts.append("%s %d boxes%s min %s" % (name, st["boxes"], (" FAILED %d %s" % (st["failed"], st["fails"][:2])) if st["failed"] else "",
                                                 iv.nstr(iv.mpf(st["min"]), 4) if st["min"] is not None else "-"))
        out.append("tau=%s  %s" % ("".join(map(str, tau)), "; ".join(parts))); print(out[-1], flush=True)
        open("r6_ts_iv%s.out" % tag, "w", newline="\n").write("\n".join(out) + "\n")
    out.append("all %d blocks proved on families B, T, C: %s; smallest lower bound of F %s" % (len(sel), allok,
               iv.nstr(iv.mpf(gmin), 4) if gmin is not None else "-"))
    print(out[-1]); open("r6_ts_iv%s.out" % tag, "w", newline="\n").write("\n".join(out) + "\n")
