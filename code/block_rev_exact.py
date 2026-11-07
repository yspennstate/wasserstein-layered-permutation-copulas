"""Block reversal of values k..k+r-1: exact local re-cut bound. U = union of the r quantile strips of S = x1+x2,
s in (q_k, q_{k+r}). Re-cut U by global quantiles of v = x1-x2 (comonotone with p1-p2 = (2i+1-r)/n). Then
  D = -(1/n^2) sum_j (j - jbar) m_j  +  (1/(2n)) sum_i w_i E[v | R_i],  w_i = (2i+1-r)/n,
m_j = E[S | strip j]. v on U has density g(v) = (1/2) |(q_k, q_{k+r}) n [|v|, 2-|v|]|, so everything is 1-D.
Checks D > 0 for every (n, k, r), 2 <= r <= n-1. Two v-resolutions give the error. Usage: python block_rev_exact.py [nmax] [res]"""
import sys
import numpy as np
src = open(__file__.replace("block_rev_exact.py", "adj_swap_bound.py"), encoding="utf-8").read().split("\nworst = 9.0")[0]
ns = {}; exec(compile(src, "adj_swap_bound.py", "exec"), ns)
Finv, mom = ns["Finv"], ns["mom"]
NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 32
RES = int(sys.argv[2]) if len(sys.argv) > 2 else 40001
OUT = __file__.replace(".py", "_n%d.out" % NMAX)

def vpart(al, be, n, r, res):
    v = np.linspace(-1, 1, res)
    av = np.abs(v)
    g = 0.5 * np.clip(np.minimum(be, 2 - av) - np.maximum(al, av), 0, None)
    dv = v[1] - v[0]
    cm = np.concatenate([[0], np.cumsum((g[1:] + g[:-1]) / 2 * dv)])        # mass up to v
    cv = np.concatenate([[0], np.cumsum(((g * v)[1:] + (g * v)[:-1]) / 2 * dv)])
    tot = cm[-1]
    cuts = np.interp(np.arange(r + 1) / r * tot, cm, cv)                    # int v g up to each mass quantile
    Ev = np.diff(cuts) * n                                                   # E[v | R_i] (each R_i has mass 1/n)
    w = (2 * np.arange(r) + 1 - r) / n
    return (w * Ev).sum() / (2 * n), abs(tot - r / n)

lines = []; worst = (9, None); maxerr = 0
for n in range(3, NMAX + 1):
    q = [Finv(j / n) for j in range(n + 1)]; q[-1] = 2.0
    m = np.array([n * mom(q[j], q[j + 1], 1) for j in range(n)])
    rowmin = (9, None)
    for r in range(2, n):
        for k in range(0, n - r + 1):
            j = np.arange(k, k + r)
            upart = -((j - j.mean()) * m[k:k + r]).sum() / n ** 2
            vp, merr = vpart(q[k], q[k + r], n, r, RES)
            vp2, _ = vpart(q[k], q[k + r], n, r, RES // 2)
            D = upart + vp; maxerr = max(maxerr, abs(vp - vp2), merr)
            if n * n * D < rowmin[0]: rowmin = (n * n * D, (k, r))
            if n * n * D < worst[0]: worst = (n * n * D, (n, k, r))
    t = "n=%2d min n^2*D %+.5f at (k,r)=%s" % (n, rowmin[0], rowmin[1]); lines.append(t); print(t, flush=True)
    open(OUT, "w", newline="\n").write("\n".join(lines) + "\n")
lines.append("worst n^2*D %+.6f at (n,k,r)=%s; max resolution/mass error %.2e" % (worst[0], worst[1], maxerr))
open(OUT, "w", newline="\n").write("\n".join(lines) + "\n"); print(lines[-1])
