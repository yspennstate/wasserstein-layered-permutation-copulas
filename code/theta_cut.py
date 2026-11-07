"""Direction cuts for a block permutation tau of values k..k+r-1 (sigma = id elsewhere). U = the r quantile strips of S.
For an angle theta cut U into r equal-mass cells by quantiles of key = s cos(theta) + v sin(theta) (v = x1-x2), then
match cells to the r block points optimally (assignment on the cell means). theta = 0 is id's own cut; the layered proof
used theta = 90 deg. Lower bound d(tau) = max_theta [value - id value]. Reports, per (n, r, window), min over tau != id of
n^2 * max_theta d, the tau and the best theta. Grid of m x m cells of the unit square."""
import sys, itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
m = int(sys.argv[1]) if len(sys.argv) > 1 else 300
NS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "8").split(",")]
RS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "3,4").split(",")]
OUT = __file__.replace(".py", "_m%d_n%s_r%s.out" % (m, "_".join(map(str, NS)), "_".join(map(str, RS))))
g = (np.arange(m) + 0.5) / m
X1, X2 = np.meshgrid(g, g, indexing="ij"); X = np.stack([X1.ravel(), X2.ravel()], 1)
S = X.sum(1); V = X[:, 0] - X[:, 1]; order = np.argsort(S, kind="stable"); N = len(X)
TH = np.radians(np.arange(0, 180, 2.0))
lines = []
for n in NS:
    a = (np.arange(n) + 0.5) / n
    rank = np.empty(N, int); rank[order] = (np.arange(N) * n) // N
    for r in RS:
        for k in sorted(set((0, (n - r) // 2, n - r))):
            sel = (rank >= k) & (rank < k + r); P = X[sel]; s, v = S[sel], V[sel]
            K = len(P)
            cellmeans = []
            for th in TH:
                key = s * np.cos(th) + v * np.sin(th)
                o = np.argsort(key, kind="stable"); lab = np.empty(K, int); lab[o] = (np.arange(K) * r) // K
                cellmeans.append(np.array([P[lab == c].mean(0) for c in range(r)]))
            base_pts = np.stack([a[k:k + r], a[k:k + r]], 1)
            base = max(-linear_sum_assignment(-(cm @ base_pts.T))[1].size * 0 + (cm @ base_pts.T)[linear_sum_assignment(-(cm @ base_pts.T))].sum() for cm in cellmeans)
            res = []
            for tau in itertools.permutations(range(r)):
                if tau == tuple(range(r)): continue
                pts = np.stack([a[k:k + r], a[k + np.array(tau)]], 1)
                best = (-9, None)
                for th, cm in zip(TH, cellmeans):
                    C = cm @ pts.T; ri, ci = linear_sum_assignment(-C); val = C[ri, ci].sum()
                    if val > best[0]: best = (val, th)
                d = (best[0] - base) / n          # each cell has mass 1/n
                res.append((n * n * d, tau, round(np.degrees(best[1]))))
            res.sort(key=lambda t: t[0])
            indec = lambda tau: all(set(tau[:j]) != set(range(j)) for j in range(1, len(tau)))
            ri_ = [x for x in res if indec(x[1])]
            t2 = "   indecomposable: min %+.4f at %s theta=%d; #<=0 %d of %d; nonpositive (any): %s" % (
                ri_[0][0], ri_[0][1], ri_[0][2], sum(1 for x in ri_ if x[0] <= 0), len(ri_), [x[1] for x in res if x[0] <= 0])
            t ="n=%2d r=%d k=%2d  min_tau n^2 max_theta d %+.4f at tau=%s theta=%d; #tau<=0: %d of %d" % (
                n, r, k, res[0][0], res[0][1], res[0][2], sum(1 for x in res if x[0] <= 0), len(res))
            lines += [t, t2]; print(t); print(t2, flush=True)
            open(OUT, "w", newline="\n").write("\n".join(lines) + "\n")
