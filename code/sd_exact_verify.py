"""Check the sharded certificate outputs sd_exact_nN_sK_*.out: every shard finished, every line well formed, no
duplicate or missing class, the class count equal to Burnside's count of orbits of S_n under the 8 symmetries of the
square (less the class of id and rev), no FAIL, and report the smallest certified margin.
Usage: python sd_exact_verify.py N K   -> prints a summary (write it to sd_exact_nN.verify)."""
import sys, re, math

def involutions(n):
    a, b = 1, 1
    for k in range(2, n + 1): a, b = b, b + (k - 1) * a
    return b if n >= 1 else 1

def orbits(n):
    m = n // 2
    fix = math.factorial(n) + 2 * involutions(n) + (2 ** m) * math.factorial(m)
    if n % 4 in (0, 1): fix += 2 * math.factorial(2 * (n // 4)) // math.factorial(n // 4)
    assert fix % 8 == 0
    return fix // 8

n, K = int(sys.argv[1]), int(sys.argv[2])
pat = re.compile(r"^(\d{%d}) (ok|FAIL) (-?\d\.\d{6}e[-+]\d\d) excess (\d\.\de[-+]\d\d)$" % n)
keys = set(); bad = fail = 0; low = None; L = set()
for k in range(K):
    lines = open("sd_exact_n%d_s%d_%d.out" % (n, K, k), encoding="utf-8").read().split("\n")
    assert lines[-1] == "" and lines[-2] == "# done", "shard %d not finished" % k
    L.add(lines[0].split()[-1])
    for l in lines[1:-2]:
        m = pat.match(l)
        if not m: bad += 1; print("malformed", k, repr(l)); continue
        key = m.group(1)
        assert sorted(key) == sorted("".join(map(str, range(n)))), key
        if key in keys: print("duplicate", key)
        keys.add(key)
        if m.group(2) == "FAIL": fail += 1
        mg = float(m.group(3))
        if low is None or mg < low[0]: low = (mg, key)
want = orbits(n) - 1
print("n=%d shards=%d lower bound(s) %s" % (n, K, sorted(L)))
print("classes %d expected %d (Burnside %d less 1) malformed %d FAIL %d" % (len(keys), want, want + 1, bad, fail))
print("smallest margin %.6e at %s" % low)
print("VERIFIED" if len(keys) == want and bad == 0 and fail == 0 and len(L) == 1 else "NOT VERIFIED")
