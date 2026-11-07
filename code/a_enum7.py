"""Check of (A_n) at n=7 (and n=8 if asked): A_c over one permutation per dihedral orbit (sigma^-1, n-1-sigma,
sigma(n-1-.) generate the 8-element group; A_c is invariant under it; we take a MIN
over representatives, never a sum). Same solver as a_enum.py.
Usage: python a_enum7.py [m] [eps] [n]; writes a_enum7_n<n>.out incrementally (lowest 12 levels)."""
import sys, itertools
import numpy as np
src = open(__file__.replace("a_enum7.py", "a_enum.py"), encoding="utf-8").read().split("\nlines = []")[0]
ARGV = list(sys.argv); sys.argv = ARGV[:3]
ns = {"__name__": "lib"}; exec(compile(src, "a_enum.py", "exec"), ns); Ac = ns["Ac"]
n = int(ARGV[3]) if len(ARGV) > 3 else 7
def orbit(s):
    out = set(); todo = [tuple(s)]
    while todo:
        t = todo.pop()
        if t in out: continue
        out.add(t)
        inv = [0] * n
        for i, v in enumerate(t): inv[v] = i
        todo += [tuple(inv), tuple(n - 1 - v for v in t), tuple(t[::-1])]
    return out
seen = set(); reps = []
for s in itertools.permutations(range(n)):
    if s in seen: continue
    o = orbit(s); seen |= o; reps.append(min(o))
a0 = Ac(list(range(n)))[0]
res = []; outp = __file__.replace(".py", "_n%d.out" % n)
for k, s in enumerate(reps):
    v, e = Ac(list(s)); res.append((v - a0, s, e))
    if k % 25 == 0 or k == len(reps) - 1:
        res.sort()
        L = ["n=%d m=%s eps=%s orbits done %d/%d  A_c(id) %.6f  #below id %d" % (n, ns["m"], ns["eps"], k + 1, len(reps), a0,
             sum(1 for d, _, _ in res if d < -1e-6))]
        L += ["  d %+.6f  n^2 d %+.4f  %s  margerr %.1e" % (d, n * n * d, s, e) for d, s, e in res[:12]]
        open(outp, "w", newline="\n").write("\n".join(L) + "\n")
print(open(outp).read())
