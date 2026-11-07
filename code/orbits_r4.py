"""Symmetry orbits of the twelve non-layered indecomposable blocks of four under inverse (transpose: same band) and
reverse-complement (band (a,b) -> (2-b,2-a)), checked against the window value F of win_r on a few bands."""
import itertools
from win_r import F
ind = lambda t: all(set(t[:j]) != set(range(j)) for j in range(1, len(t)))
inv = lambda t: tuple(sorted(range(4), key=lambda i: t[i]))
rc = lambda t: tuple(3 - t[3 - i] for i in range(4))
blocks = [t for t in itertools.permutations(range(4)) if ind(t) and t != (3, 2, 1, 0)]
s = lambda t: "".join(map(str, t))
seen, orbits = set(), []
for t in blocks:
    if t in seen: continue
    o = {t, inv(t), rc(t), inv(rc(t))}; seen |= o; orbits.append(sorted(o))
out = ["orbits: " + " | ".join(",".join(s(x) for x in o) for o in orbits)]
for t in blocks:
    for a, b in [(0.3, 0.9), (0.5, 1.4), (1.1, 1.7)]:
        f, fi, fr = F(a, b, t), F(a, b, inv(t)), F(2 - b, 2 - a, rc(t))
        out.append("%s (%.1f,%.1f): F %+.8f  inv %s %+.8f  rc %s on mirror %+.8f  maxdiff %.1e" % (
            s(t), a, b, f, s(inv(t)), fi, s(rc(t)), fr, max(abs(f - fi), abs(f - fr))))
open(__file__.replace(".py", ".out"), "w", newline="\n").write("\n".join(out) + "\n"); print("\n".join(out))
