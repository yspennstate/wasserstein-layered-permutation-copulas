"""Merge the four shard outputs of r8_iv.py into r8_iv.out. The shards prove one block per orbit of {id, inverse, rc,
rc o inverse} (7,578 representatives); the merge checks that the representatives proved are exactly those orbits and that
the orbits cover all 29,092 indecomposable non-reversal blocks of eight (r8_orbit_ctrl.py checks that F is the same along an
orbit)."""
import re, itertools
from r8_iv import indec, R, rep, orbit
rows, finals = {}, []
for i in range(4):
    for line in open("r8_iv_s%d.out" % i, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("tau="): rows[line.split()[0][4:]] = line
        elif line: finals.append(line)
TAUS = [t for t in itertools.permutations(range(R)) if indec(t) and t != tuple(range(R - 1, -1, -1))]
REPS = sorted({rep(t) for t in TAUS})
key = lambda t: "".join(map(str, t))
assert len(TAUS) == 29092 and sorted(rows) == sorted(key(t) for t in REPS), (len(rows), len(REPS))
assert {u for t in REPS for u in orbit(t)} == set(TAUS)
assert all(": True;" in f for f in finals) and len(finals) == 4, finals
boxes = sum(int(n) for l in rows.values() for n in re.findall(r"(\d+) boxes", l))
assert not any("FAILED" in l for l in rows.values())
mins = [float(x) for l in rows.values() for x in re.findall(r"min \[([0-9.e+-]+),", l)]
out = [rows[key(t)] for t in REPS]
out.append("all %d orbit representatives proved on families B, T, C: True (covering all %d blocks); %d boxes; "
           "smallest lower bound of F %.4g" % (len(REPS), len(TAUS), boxes, min(mins)))
open("r8_iv.out", "w", newline="\n").write("\n".join(out) + "\n"); print(out[-1])
