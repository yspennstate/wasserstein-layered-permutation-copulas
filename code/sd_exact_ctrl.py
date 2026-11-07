from sd_exact import lower_identity, upper
for n in (5, 7):
    L = lower_identity(n)
    for s in (list(range(n)), list(range(n))[::-1]):
        U, e = upper(s)
        print("n=%d %s U-L %+.3e (must be >= 0: the certifier cannot prove the diagonal beats itself)" % (n, "".join(map(str, s)), float(U - L)))
