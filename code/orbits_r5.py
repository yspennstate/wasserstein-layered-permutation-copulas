"""Symmetry orbits of the 70 non-layered indecomposable blocks of five under inverse (same band) and
reverse-complement (band (a,b) -> (2-b,2-a)); min F/m per orbit from win_r5_mass.out, and the invariance of F
along each orbit on three bands."""
import itertools, re
from win_r import F, indec
r = 5
inv = lambda t: tuple(sorted(range(r), key=lambda i: t[i]))
rc = lambda t: tuple(r - 1 - t[r - 1 - i] for i in range(r))
s = lambda t: "".join(map(str, t))
blocks = [t for t in itertools.permutations(range(r)) if indec(t) and t != tuple(range(r - 1, -1, -1))]
mins = {m.group(1): float(m.group(2)) for m in re.finditer(r"tau=(\d+)\s+min F/m ([+-][\d.]+)", open("win_r5_mass.out").read())}
seen, orbits = set(), []
for t in blocks:
    if t in seen: continue
    o = {t, inv(t), rc(t), inv(rc(t))}; seen |= o; orbits.append(sorted(o))
orbits.sort(key=lambda o: min(mins[s(x)] for x in o))
out = ["%d blocks, %d orbits (sorted by min F/m over bands of mass <= 5/9)" % (len(blocks), len(orbits))]
md = 0.0
for o in orbits:
    out.append("  %-27s min F/m %s" % (",".join(s(x) for x in o), " ".join("%+.4f" % mins[s(x)] for x in o)))
    for t in o:
        for a, b in [(0.3, 0.9), (0.5, 1.4), (1.1, 1.7)]:
            f = F(a, b, t)
            md = max(md, abs(f - F(a, b, inv(t))), abs(f - F(2 - b, 2 - a, rc(t))))
out.append("max |F difference| along orbits on three bands: %.1e" % md)
open("orbits_r5.out", "w", newline="\n").write("\n".join(out) + "\n"); print("\n".join(out))
