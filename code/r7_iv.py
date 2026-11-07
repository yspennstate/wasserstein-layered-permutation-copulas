"""r = 7: the partition 2221 (pieces of S of masses 2/7, 2/7, 2/7 each split at V = 0, and a last piece of mass 1/7 left
whole) with an assignment of the seven points to the seven cells, proved F > 0 in interval arithmetic for every
indecomposable non-reversal block of seven values on every band of mass at most 7/9.

With x_0 = a < ... < x_7 = b the septiles of S on the band and e_j, l_j the means of S and of L(S) over [x_j, x_{j+1}], a
piece made of septiles j, j+1 and split at V = 0 gives two cells with E[S] = (e_j + e_{j+1})/2 and
E[V] = +-(l_j + l_{j+1})/4; a piece made of one septile left whole gives E[S] = e_j, E[V] = 0. For an assignment pi
  F = sum_c (u_pi(c) E_S(c) + v_pi(c) E_V(c)) - sum_j (2j - 6) e_j = sum_j (A_j e_j + B_j l_j).
Families as in r6_ts_iv.py: (B) b = 1, a = t in [0, 1]; (T) a >= 1, mapped by s -> 2 - s to family B with u replaced by -u
and the composition reversed (2221 -> 1222); (C) a = alpha < 1 < b = 2 - beta with alpha^2 + beta^2 >= 4/9 (mass <= 7/9).
The means and their capped enclosures are those of r6_ts_iv.py.
Usage: r7_iv.py [--shard i n] [taus...]   (default: all 3446 blocks)."""
import sys, itertools
import numpy as np
from mpmath import iv, mpf
from scipy.optimize import linear_sum_assignment
from r6_ts_iv import hull, H_iv, Hinv_iv, means_capped, Hf, Hinvf, Kf, Jf
R = 7
COMP = (2, 2, 2, 1)

def quantiles(a0, a1, b0, b1):
    Ha0, Ha1, Hb0, Hb1 = H_iv(a0), H_iv(a1), H_iv(b0), H_iv(b1)
    xs = [hull(iv.mpf(a0), iv.mpf(a1))]
    for j in range(1, R):
        lo = Hinv_iv(((R - j) * Ha0 + j * Hb0) / R).a
        hi = Hinv_iv(((R - j) * Ha1 + j * Hb1) / R).b
        xs.append(iv.mpf([lo, hi]))
    xs.append(hull(iv.mpf(b0), iv.mpf(b1)))
    return xs

def cell_forms(comp):                         # each cell: (coefficients of e_j in E[S], coefficients of l_j in E[V])
    cells, s0 = [], 0
    for k in comp:
        if k == 1:
            cells.append(({s0: 1.0}, {}))
        elif k == 2:
            cells.append(({s0: .5, s0 + 1: .5}, {s0: -.25, s0 + 1: -.25}))
            cells.append(({s0: .5, s0 + 1: .5}, {s0: .25, s0 + 1: .25}))
        else:
            raise ValueError(comp)
        s0 += k
    assert s0 == R and len(cells) == R
    return cells

def coefs(pts, assign, cells):                # assign[c] = point index of cell c
    A = [-(2 * j - (R - 1)) for j in range(R)]; B = [0.0] * R
    for c, (fs, fv) in enumerate(cells):
        u, v = pts[assign[c]]
        for j, w in fs.items(): A[j] += w * u
        for j, w in fv.items(): B[j] += w * v
    return A, B

def qmeans_f(a, b):
    ma, mb = Hf(a), Hf(b); xs = [a] + [Hinvf(ma + j * (mb - ma) / R) for j in range(1, R)] + [b]
    out = []
    for j in range(R):
        m = Hf(xs[j + 1]) - Hf(xs[j])
        out.append(((Kf(xs[j + 1]) - Kf(xs[j])) / m, (Jf(xs[j + 1]) - Jf(xs[j])) / m))
    return out

def F_f(pts, a, b, cells, assign=None):       # float F; the best assignment when none is given
    em = qmeans_f(a, b)
    cm = [(sum(w * em[j][0] for j, w in fs.items()), sum(w * em[j][1] for j, w in fv.items())) for fs, fv in cells]
    W = np.array([[p[0] * c[0] + p[1] * c[1] for c in cm] for p in pts])
    if assign is None:
        ri, ci = linear_sum_assignment(-W)
        assign = [None] * R
        for i, c in zip(ri, ci): assign[c] = i
    F = sum(W[assign[c], c] for c in range(R)) - sum((2 * j - (R - 1)) * em[j][0] for j in range(R))
    return assign, F

def F_iv(pts, a0, a1, b0, b1, assign, cells):
    xs = quantiles(a0, a1, b0, b1)
    A, B = coefs(pts, assign, cells)
    tot = iv.mpf(0)
    for j in range(R):
        e, l = means_capped(xs[j], xs[j + 1]); tot += iv.mpf(A[j]) * e + iv.mpf(B[j]) * l
    return tot

def prove_box(pts, cells, a0, a1, b0, b1, st, depth):
    assign, ff = F_f(pts, float((a0 + a1) / 2), float((b0 + b1) / 2), cells)
    lb = F_iv(pts, a0, a1, b0, b1, assign, cells).a
    if lb > 0:
        st["boxes"] += 1
        if st["min"] is None or lb < st["min"]: st["min"] = lb
        return
    if depth > 30:
        st["failed"] += 1; st["fails"].append((float(a0), float(a1), float(b0), float(b1), ff)); return
    m = (a0 + a1) / 2
    prove_box(pts, cells, a0, m, b0, b1, st, depth + 1); prove_box(pts, cells, m, a1, b0, b1, st, depth + 1)

def new_st(): return dict(boxes=0, failed=0, min=None, fails=[], outside=0)

def family_B(pts, cells, st):                 # b = 1, a = t in [0, 1]
    for i in range(16):
        prove_box(pts, cells, mpf(i) / 16, mpf(i + 1) / 16, mpf(1), mpf(1), st, 0)

def family_C(pts, cells, st, G=9):            # a = alpha, b = 2 - beta, alpha^2 + beta^2 >= 4/9
    def run(al0, al1, be0, be1, d):
        if al1 * al1 + be1 * be1 < mpf(4) / 9: st["outside"] += 1; return
        am, bm = float((al0 + al1) / 2), 2 - float((be0 + be1) / 2)
        assign, ff = F_f(pts, am, bm, cells)
        lb = F_iv(pts, al0, al1, 2 - be1, 2 - be0, assign, cells).a
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
    args = sys.argv[1:]; shard = None
    if args[:1] == ["--shard"]:
        shard = (int(args[1]), int(args[2])); args = args[3:]
    TAUS = [t for t in itertools.permutations(range(R)) if indec(t) and t != tuple(range(R - 1, -1, -1))]
    sel = [tuple(int(c) for c in s) for s in args] or TAUS
    if shard: sel = sel[shard[0]::shard[1]]
    tag = ("_s%d" % shard[0]) if shard else ("_sel" if args else "")
    CB, CT = cell_forms(COMP), cell_forms(COMP[::-1])
    out, allok, gmin = [], True, None
    for tau in sel:
        pts = [(i + tau[i] - (R - 1), i - tau[i]) for i in range(R)]
        parts = []
        for name, P, cells, fam in (("B", pts, CB, family_B), ("T", [(-u, v) for u, v in pts], CT, family_B), ("C", pts, CB, family_C)):
            st = new_st(); fam(P, cells, st); allok &= st["failed"] == 0
            if st["min"] is not None: gmin = st["min"] if gmin is None else min(gmin, st["min"])
            parts.append("%s %d boxes%s min %s" % (name, st["boxes"], (" FAILED %d %s" % (st["failed"], st["fails"][:2])) if st["failed"] else "",
                                                 iv.nstr(iv.mpf(st["min"]), 4) if st["min"] is not None else "-"))
        out.append("tau=%s  %s" % ("".join(map(str, tau)), "; ".join(parts))); print(out[-1], flush=True)
        if len(out) % 20 == 0: open("r7_iv%s.out" % tag, "w", newline="\n").write("\n".join(out) + "\n")
    out.append("%d blocks proved on families B, T, C: %s; smallest lower bound of F %s" % (len(sel), allok,
               iv.nstr(iv.mpf(gmin), 4) if gmin is not None else "-"))
    print(out[-1]); open("r7_iv%s.out" % tag, "w", newline="\n").write("\n".join(out) + "\n")
