"""Rigorous positivity of F(t) for tau = 1302 on bands inside s <= 1 (scaled to b = 1, t = a/b in [0,1]), by interval
arithmetic (mpmath.iv, outward rounding). Parametrise t = u/3, u in [0,3], so the median branch point t = 1/3 is u = 1
exactly. Formulas of tau1302_lower.py rewritten without cancellation:
  S_j = (2/3)(b^2 + ab + a^2)/(a + b) for the s-quartile [a,b], quantiles sqrt(t^2 + p(1-t^2));
  t <= 1/3: |v|-median x = 1 - sqrt((1-t^2)/2), M = (1-t^2)/2,
            B = [(1-t)t^2/2 + (x^2-t^2)/2 - (x^3-t^3)/3]/(M/2), A = [(1-x^2)/2 - (1-x^3)/3]/(M/2);
  t >= 1/3: x = (1+t)/4, B = (1+t)/8, A = [(t^2-x^2)/2 + (1-t)(1+2t)/6] * 4/(1+t).
F = 4A + 2B - 3(S4-S1) - (S3-S2). Since every quantity is a length, a band (a,b) with b <= 1 has F(a,b) = b F(a/b).
Bisects u-intervals until the enclosure of F is positive; prints the number of boxes and the smallest lower bound."""
from mpmath import iv, mpf
iv.dps = 30
def Fiv(u0, u1):
    u = iv.mpf([u0, u1]); t = u / 3
    sp = [iv.sqrt(t * t + iv.mpf(p) * (1 - t * t)) for p in ("0", "0.25", "0.5", "0.75", "1")]
    sp[0] = t; sp[4] = iv.mpf(1)
    S = [(2 * (sp[j + 1] ** 2 + sp[j + 1] * sp[j] + sp[j] ** 2)) / (3 * (sp[j] + sp[j + 1])) for j in range(4)]
    if u1 <= 1:
        M = (1 - t * t) / 2; half = M / 2
        x = 1 - iv.sqrt((1 - t) * (1 + t) / 2)          # (1-t)^2 - 2(M/2 - t(1-t)) = (1-t^2)/2
        B = ((1 - t) * t * t / 2 + (x * x - t * t) / 2 - (x ** 3 - t ** 3) / 3) / half
        A = ((1 - x * x) / 2 - (1 - x ** 3) / 3) / half
    else:
        assert u0 >= 1
        x = (1 + t) / 4; B = (1 + t) / 8
        A = ((t * t - x * x) / 2 + (1 - t) * (1 + 2 * t) / 6) * 4 / (1 + t)
    return 4 * A + 2 * B - 3 * (S[3] - S[0]) - (S[2] - S[1])
def check(u0, u1, depth=0, acc=None):
    f = Fiv(u0, u1)
    if f.a > 0:
        acc.append((u0, u1, f.a)); return True
    if depth > 40: return False
    m = (u0 + u1) / 2
    return check(u0, m, depth + 1, acc) and check(m, u1, depth + 1, acc)
if __name__ == "__main__":
    out = []
    for lo, hi in ((0, 1), (1, 3)):
        acc = []
        ok = check(mpf(lo), mpf(hi), acc=acc)
        mn = min(acc, key=lambda z: z[2])
        out.append("t in [%s/3, %s/3]: proved F > 0: %s; %d boxes; smallest lower bound %s on u in [%s, %s]" % (
            lo, hi, ok, len(acc), iv.nstr(mn[2], 8), iv.nstr(mn[0], 8), iv.nstr(mn[1], 8)))
        print(out[-1], flush=True)
    pt = [iv.nstr(Fiv(mpf(z), mpf(z)), 12) for z in (0, 1.5)]
    out.append("point enclosures: F(0) %s (float code 0.406911), F(0.5) %s (float code 1.515578)" % tuple(pt))
    print(out[-1]); open("tau1302_lower_iv.out", "w", newline="\n").write("\n".join(out) + "\n")
