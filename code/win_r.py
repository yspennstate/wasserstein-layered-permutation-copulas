"""Window lemma for a general block tau of r consecutive values (identity elsewhere): two-step 'v then s' cut.
Points in units h = 1/n about the band centre: u'_i = i + tau(i) - (r-1), v'_i = i - tau(i); identity u' = 2i-(r-1), v' = 0.
Cut U by v at the cumulative quantiles of the v'-groups (lowest v' group gets the lowest v cells), then each v-group by s
into equal-mass cells, matched to the group's points in u' order. Also the transposed order ('s then v'). The bound is
n^2 D >= F/2, F = sum_cells (u'.E[s|cell] + v'.E[v|cell]) - sum_j u'_id,j E[s|strip j]. Reports min F/(b-a) over a band grid
and F/2 on the strips of a few n, per indecomposable non-layered tau."""
import sys, itertools
import numpy as np
from cyc3_window import clip, am, band

def bis(f, lo, hi, target, it=64):
    for _ in range(it):
        mid = (lo + hi) / 2
        if f(mid) < target: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def slices(P, M, k, axis):           # k equal-mass slices of polygon P along axis (0 = s, 1 = v)
    ex = [p[axis] for p in P]; lo, hi = min(ex), max(ex)
    A = (1, 0) if axis == 0 else (0, 1)
    cuts, out, rest = [], [], P
    for j in range(1, k):
        x = bis(lambda t: am(clip(P, A[0], A[1], t))[0], lo, hi, j * M / k); cuts.append(x)
    prev = None
    for j in range(k):
        Q = P
        if j > 0: Q = clip(Q, -A[0], -A[1], -cuts[j - 1])
        if j < k - 1: Q = clip(Q, A[0], A[1], cuts[j])
        out.append(Q)
    return out

def cutval(U, M, pts, first, r):
    """first = 1: group by v' then s; first = 0: group by u' rank in s... (s then v): s-slices sized by u'-groups."""
    key = (lambda p: p[1]) if first == 1 else (lambda p: p[0])
    other = (lambda p: p[0]) if first == 1 else (lambda p: p[1])
    groups = {}
    for p in pts: groups.setdefault(key(p), []).append(p)
    gs = sorted(groups)
    ex = [q[first] for q in U]; lo, hi = min(ex), max(ex); A = (1, 0) if first == 0 else (0, 1)
    tot, cum, val = len(pts), 0, 0.0
    bounds = []
    for g in gs[:-1]:
        cum += len(groups[g]); bounds.append(bis(lambda t: am(clip(U, A[0], A[1], t))[0], lo, hi, cum * M / tot))
    for gi, g in enumerate(gs):
        Q = U
        if gi > 0: Q = clip(Q, -A[0], -A[1], -bounds[gi - 1])
        if gi < len(gs) - 1: Q = clip(Q, A[0], A[1], bounds[gi])
        mem = sorted(groups[g], key=other)
        cells = slices(Q, M * len(mem) / tot, len(mem), 1 - first) if len(mem) > 1 else [Q]
        for p, C in zip(mem, cells):
            _, (es, ev) = am(C); val += p[0] * es + p[1] * ev
    return val

def F(a, b, tau):
    r = len(tau); U = band(a, b); M, _ = am(U)
    strips = slices(U, M, r, 0)
    idv = sum((2 * j - (r - 1)) * am(S)[1][0] for j, S in enumerate(strips))
    pts = [(i + tau[i] - (r - 1), i - tau[i]) for i in range(r)]
    return max(cutval(U, M, pts, 1, r), cutval(U, M, pts, 0, r)) - idv

indec = lambda t: all(set(t[:j]) != set(range(j)) for j in range(1, len(t)))
if __name__ == "__main__":
    r = int(sys.argv[1]) if len(sys.argv) > 1 else 4; G = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    taus = [t for t in itertools.permutations(range(r)) if indec(t) and t != tuple(range(r - 1, -1, -1))]
    out = []
    for tau in taus:
        best = (9, None)
        for a in np.linspace(0, 2, G):
            for b in np.linspace(0, 2, G):
                if b <= a + 1e-9 or (a < 1e-12 and b > 2 - 1e-12): continue
                f = F(a, b, tau)
                if f / (b - a) < best[0]: best = (f / (b - a), a, b)
        cq = lambda j, n: (2 * j / n) ** 0.5 if j <= n / 2 else 2 - (2 * (n - j) / n) ** 0.5
        nv = []
        for n in (r + 1, 8, 32):
            nv.append(min(F(cq(k, n), cq(k + r, n), tau) / 2 for k in range(n - r + 1)))
        line = "tau=%s  min F/(b-a) %+.4f at (%.2f,%.2f);  min_k n^2 D-bound n=%d %+.4f, n=8 %+.4f, n=32 %+.4f" % (
            "".join(map(str, tau)), best[0], best[1], best[2], r + 1, nv[0], nv[1], nv[2])
        out.append(line); print(line, flush=True)
        open("win_r%d.out" % r, "w", newline="\n").write("\n".join(out) + "\n")
