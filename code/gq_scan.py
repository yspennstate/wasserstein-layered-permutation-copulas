"""r = 5: the 'v-quintile' cut for every orbit. Cut the band into its five v-quintile cells c_0 < ... < c_4 and give the
cells to the block's points in increasing order of v' (ties: every order among tied points; the best is kept per band).
Value Gq = sum_k (u'_k E[s|c_k] + v'_k E[v|c_k]) - sum_j (2j-4) S_j. For 12430 this is the cut G of tau12430_G.py.
Grid scan of min Gq/m over bands of mass m <= 5/9, one representative per orbit (orbits_r5.out)."""
import itertools
import numpy as np
g = (np.arange(1500) + 0.5) / 1500; X1, X2 = np.meshgrid(g, g, indexing="ij"); S0 = (X1 + X2).ravel(); V0 = (X1 - X2).ravel()
Q = lambda p: (2 * p) ** 0.5 if p <= 0.5 else 2 - (2 * (1 - p)) ** 0.5
reps = [l.split()[0].split(",")[0] for l in open("orbits_r5.out").read().splitlines()[1:] if "min F/m" in l]
def cuts(tau):
    pts = sorted([(i - tau[i], i + tau[i] - 4) for i in range(5)])
    groups = [list(g) for _, g in itertools.groupby(pts, key=lambda p: p[0])]
    outs = []
    for combo in itertools.product(*[set(itertools.permutations(gr)) for gr in groups]):
        seq = [p for gr in combo for p in gr]
        outs.append(([p[1] for p in seq], [p[0] for p in seq]))
    return outs
if __name__ == "__main__":
    CUTS = {t: cuts(tuple(int(c) for c in t)) for t in reps}
    worst = {t: (9,) for t in reps}
    for m in np.linspace(5 / 9 / 12, 5 / 9, 12):
        for q in np.linspace(0, 1 - m, 13):
            a, b = Q(q), Q(min(q + m, 1)); k = (S0 > a) & (S0 < b); s, v = S0[k], V0[k]
            qv = np.quantile(v, [0.2, 0.4, 0.6, 0.8]); qs = np.quantile(s, [0.2, 0.4, 0.6, 0.8])
            c = [v <= qv[0]] + [(v > qv[j - 1]) & (v <= qv[j]) for j in range(1, 4)] + [v > qv[3]]
            Es = [s[x].mean() for x in c]; Ev = [v[x].mean() for x in c]
            Sj = [s[s <= qs[0]].mean()] + [s[(s > qs[j - 1]) & (s <= qs[j])].mean() for j in range(1, 4)] + [s[s > qs[3]].mean()]
            rank = sum((2 * j - 4) * Sj[j] for j in range(5))
            for t in reps:
                best = max(sum(u[i] * Es[i] + w[i] * Ev[i] for i in range(5)) for u, w in CUTS[t]) - rank
                worst[t] = min(worst[t], (best / m, q, m))
    out = ["tau=%s  cuts %d  min Gq/m %+.4f at q=%.3f m=%.3f" % ((t, len(CUTS[t])) + worst[t]) for t in reps]
    out.append("negative or zero: %d of %d" % (sum(1 for t in reps if worst[t][0] <= 0), len(reps)))
    print("\n".join(out)); open("gq_scan.out", "w", newline="\n").write("\n".join(out) + "\n")
