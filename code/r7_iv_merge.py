"""Merge the four shard outputs of r7_iv.py into r7_iv.out, in the order of the full block list, with totals."""
import re, itertools
from r7_iv import indec, R
rows, finals = {}, []
for i in range(4):
    for line in open("r7_iv_s%d.out" % i, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("tau="): rows[line.split()[0][4:]] = line
        elif line: finals.append(line)
TAUS = ["".join(map(str, t)) for t in itertools.permutations(range(R)) if indec(t) and t != tuple(range(R - 1, -1, -1))]
assert sorted(rows) == sorted(TAUS) and len(TAUS) == 3446, (len(rows), len(TAUS))
assert all(f.endswith(tuple("0123456789]")) and ": True;" in f for f in finals) and len(finals) == 4, finals
boxes = sum(int(n) for l in rows.values() for n in re.findall(r"(\d+) boxes", l))
assert not any("FAILED" in l for l in rows.values())
mins = [float(x) for l in rows.values() for x in re.findall(r"min \[([0-9.e+-]+),", l)]
out = [rows[t] for t in TAUS]
out.append("all %d blocks proved on families B, T, C: True; %d boxes; smallest lower bound of F %.4g" % (len(TAUS), boxes, min(mins)))
open("r7_iv.out", "w", newline="\n").write("\n".join(out) + "\n"); print(out[-1])
