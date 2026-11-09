"""Check that every key in the sharded outputs is the canonical representative of its symmetry class and is not
the class of id. Usage: python sd_canon_check.py N K"""
import sys, os, gzip
from sd_fast import canon
ALPH = "0123456789abcdefghijklmnopqrstuvwxyz"
n, K = int(sys.argv[1]), int(sys.argv[2])
ident = tuple(range(n)); idc = canon(ident)
c = bad = 0
for k in range(K):
    p = "sd_exact_n%d_s%d_%d.out" % (n, K, k)
    for l in (open(p, encoding="utf-8") if os.path.exists(p) else gzip.open(p + ".gz", "rt", encoding="utf-8")):
        if l[:1] == "#": continue
        s = tuple(ALPH.index(ch) for ch in l.split()[0])
        c += 1
        if canon(s) != s or canon(s) == idc: bad += 1
print("keys %d not canonical or id class %d" % (c, bad))
