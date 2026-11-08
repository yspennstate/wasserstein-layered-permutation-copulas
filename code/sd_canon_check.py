"""Check that every key in the sharded outputs is the canonical representative of its symmetry class and is not
the class of id. Usage: python sd_canon_check.py N K"""
import sys
from sd_fast import canon
n, K = int(sys.argv[1]), int(sys.argv[2])
ident = tuple(range(n)); idc = canon(ident)
c = bad = 0
for k in range(K):
    for l in open("sd_exact_n%d_s%d_%d.out" % (n, K, k), encoding="utf-8"):
        if not l[:1].isdigit(): continue
        s = tuple(int(ch) for ch in l.split()[0])
        c += 1
        if canon(s) != s or canon(s) == idc: bad += 1
print("keys %d not canonical or id class %d" % (c, bad))
