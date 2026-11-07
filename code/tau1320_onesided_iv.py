"""Rigorous positivity of F1 for tau = 1320 on bands inside s <= 1 or inside s >= 1, by interval arithmetic.
Homogeneity: for b <= 1, F1(a,b) = b G+(a/b); for a >= 1, the reflection s -> 2-s keeps A, B, S4-S1, S3-S2 and flips
T_out - T_in, so F1(a,b) = (2-a) G-((2-b)/(2-a)). On the scaled triangle band {t < s < 1, |v| < s}:
  G± = 5A + B ± (T_out - T_in) - 3(S4-S1) - (S3-S2),
A, B, S_j as in tau1302_lower_iv.py; with x the |v|-median and M = (1-t^2)/2,
  t <= 1/3: T_out - T_in = [(1 - 2x^3 + t^3)/3 - x(1 - x^2)] / (M/2),
  t >= 1/3: T_out - T_in = 4(1+t+t^2)/(3(1+t)) - (1+t).
t = u/3, u in [0,3]; bisection until the enclosure is positive."""
from mpmath import iv, mpf
iv.dps = 30
def parts(u0, u1):
    u = iv.mpf([u0, u1]); t = u / 3
    sp = [t] + [iv.sqrt(t * t + iv.mpf(p) * (1 - t * t)) for p in ("0.25", "0.5", "0.75")] + [iv.mpf(1)]
    S = [(2 * (sp[j + 1] ** 2 + sp[j + 1] * sp[j] + sp[j] ** 2)) / (3 * (sp[j] + sp[j + 1])) for j in range(4)]
    if u1 <= 1:
        half = (1 - t * t) / 4
        x = 1 - iv.sqrt((1 - t) * (1 + t) / 2)
        B = ((1 - t) * t * t / 2 + (x * x - t * t) / 2 - (x ** 3 - t ** 3) / 3) / half
        A = ((1 - x * x) / 2 - (1 - x ** 3) / 3) / half
        D = ((1 - 2 * x ** 3 + t ** 3) / 3 - x * (1 - x * x)) / half
    else:
        assert u0 >= 1
        x = (1 + t) / 4; B = (1 + t) / 8
        A = ((t * t - x * x) / 2 + (1 - t) * (1 + 2 * t) / 6) * 4 / (1 + t)
        D = 4 * (1 + t + t * t) / (3 * (1 + t)) - (1 + t)
    return 5 * A + B - 3 * (S[3] - S[0]) - (S[2] - S[1]), D
def G(u0, u1, sgn):
    base, D = parts(u0, u1); return base + sgn * D
def check(u0, u1, sgn, acc, d=0):
    f = G(u0, u1, sgn)
    if f.a > 0: acc.append((u0, u1, f.a)); return True
    if d > 45: return False
    m = (u0 + u1) / 2
    return check(u0, m, sgn, acc, d + 1) and check(m, u1, sgn, acc, d + 1)
if __name__ == "__main__":
    import tau1320_any as T
    out = []
    for a, b in [(0.3, 0.9), (0.1, 0.5), (0.6, 0.95)]:
        t = a / b; f = G(mpf(3 * t), mpf(3 * t), 1)
        out.append("lower check (%.2f,%.2f): b G+ %s, float F1 %.10f" % (a, b, iv.nstr(b * f, 11), T.F1(a, b)))
    for a, b in [(1.1, 1.7), (1.05, 1.5), (1.4, 1.9)]:
        t = (2 - b) / (2 - a); f = G(mpf(3 * t), mpf(3 * t), -1)
        out.append("upper check (%.2f,%.2f): (2-a) G- %s, float F1 %.10f" % (a, b, iv.nstr((2 - a) * f, 11), T.F1(a, b)))
    print("\n".join(out), flush=True)
    for sgn, name in ((1, "G+ (bands in s <= 1)"), (-1, "G- (bands in s >= 1)")):
        for lo, hi in ((0, 1), (1, 3)):
            acc = []; ok = check(mpf(lo), mpf(hi), sgn, acc)
            mn = min(acc, key=lambda z: z[2]) if acc else (0, 0, mpf(-1))
            out.append("%s, t in [%d/3,%d/3]: proved > 0: %s; %d boxes; smallest lower bound %s" % (name, lo, hi, ok, len(acc), iv.nstr(mn[2], 6)))
            print(out[-1], flush=True)
    open("tau1320_onesided_iv.out", "w", newline="\n").write("\n".join(out) + "\n")
