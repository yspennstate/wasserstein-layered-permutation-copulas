"""Sharded driver for sd_exact.py (same certificate, same functions). The canonical representatives are taken in
itertools order and representative number j goes to shard j mod K.
Usage: python sd_exact_shard.py N K k   -> sd_exact_nN_sK_k.out (resumable)
       python sd_exact_shard.py N time  -> times the canon scan and a few classes."""
import sys, os, itertools, time
from sd_exact import lower_identity, upper
from sd_fast import canon

def reps(n):
    ident = tuple(range(n)); rev = canon(ident[::-1])
    for s in itertools.permutations(range(n)):
        if canon(s) != s or s == ident or s == rev: continue
        yield s

if __name__ == "__main__":
    n = int(sys.argv[1])
    if sys.argv[2] == "time":
        t = time.time(); c = 0
        for s in itertools.islice(itertools.permutations(range(n)), 200000): canon(s); c += 1
        print("canon per perm %.2e s" % ((time.time() - t) / c), flush=True)
        t = time.time(); L = lower_identity(n); print("lower %.1f s L=%.10f" % (time.time() - t, float(L)), flush=True)
        t = time.time(); m = 0
        for s in itertools.islice(reps(n), 1000, 1020):
            U, e = upper(list(s)); m += 1
        print("upper per class %.3f s" % ((time.time() - t) / m), flush=True)
        sys.exit()
    K, k = int(sys.argv[2]), int(sys.argv[3])
    out = "sd_exact_n%d_s%d_%d.out" % (n, K, k)
    L = lower_identity(n)
    done = set()
    if os.path.exists(out):
        for l in open(out, encoding="utf-8"):
            if l[:1].isdigit(): done.add(l.split()[0])
    f = open(out, "a", encoding="utf-8", newline="\n")
    if not done: f.write("# n=%d shard %d/%d exact lower bound for W2^2(identity) %.16f\n" % (n, k, K, float(L))); f.flush()
    for j, s in enumerate(reps(n)):
        if j % K != k: continue
        key = "".join(map(str, s))
        if key in done: continue
        U, e = upper(list(s))
        ok = U < L
        f.write("%s %s %.6e excess %.1e\n" % (key, "ok" if ok else "FAIL", float(L - U), float(e))); f.flush()
    f.write("# done\n"); f.close()
