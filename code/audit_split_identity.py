"""Audit of the per-split identity: n^2 D (block_rev_exact.py, direct) against sum_t t(r-t)/r (gap_v_t - gap_S_t),
computed independently: S-gaps from strip means m_j, v-gaps from the v-quantile cells. Also checks gap_v_t >= gap_S_t."""
import numpy as np
src = open(__file__.replace("audit_split_identity.py", "block_rev_exact.py"), encoding="utf-8").read()
ns = {"__file__": __file__.replace("audit_split_identity.py", "block_rev_exact.py")}
exec(src[: src.index("lines = []")], ns)
Finv, mom, vpart = ns["Finv"], ns["mom"], ns["vpart"]
RES = 40001
v = np.linspace(-1, 1, RES); av = np.abs(v); dv = v[1] - v[0]
def vcells(al, be, r):
    g = 0.5 * np.clip(np.minimum(be, 2 - av) - np.maximum(al, av), 0, None)
    cm = np.concatenate([[0], np.cumsum((g[1:] + g[:-1]) / 2 * dv)])
    cv = np.concatenate([[0], np.cumsum(((g * v)[1:] + (g * v)[:-1]) / 2 * dv)])
    cuts = np.interp(np.arange(r + 1) / r * cm[-1], cm, cv)
    return np.diff(cuts) / (cm[-1] / r)
def split_sum(mu):
    r = len(mu); tot = 0.0; gaps = []
    for t in range(1, r):
        gap = mu[t:].mean() - mu[:t].mean(); gaps.append(gap); tot += t * (r - t) / r * gap
    return tot, np.array(gaps)
worst = 0; mingap = 9
for n in (5, 8, 13):
    q = [Finv(j / n) for j in range(n + 1)]; q[-1] = 2.0
    m = np.array([n * mom(q[j], q[j + 1], 1) for j in range(n)])
    for r in range(2, n):
        for k in range(0, n - r + 1):
            j = np.arange(k, k + r)
            direct = n * n * (-((j - j.mean()) * m[k:k + r]).sum() / n ** 2 + vpart(q[k], q[k + r], n, r, RES)[0])
            Ev = vcells(q[k], q[k + r], r)
            sv, gv = split_sum(Ev); ss, gs = split_sum(m[k:k + r])
            worst = max(worst, abs(direct - (sv - ss))); mingap = min(mingap, (gv - gs).min())
print("max |direct - split identity| = %.2e ; min over all (n,k,r,t) of gap_v - gap_S = %+.4e" % (worst, mingap))
