"""Rigorous check, in exact rational arithmetic, that every permutation sigma of {0..n-1} other than the identity and
the reversal puts its grid points strictly closer to the uniform law than the diagonal points.
Lower bound for the identity: the semi-discrete dual g(w) = sum_i int_{Lag_i(w)} (|x-p_i|^2 - w_i) + sum_i w_i/n at a
rational w (any w is a lower bound; the cells are computed exactly). Upper bound for sigma: send each Laguerre cell to its
point (cost C, masses a_i exactly), then move the excess e = sum (a_i - 1/n)^+ at distance at most sqrt 2, so
W2 <= sqrt C + sqrt(2e) and W2^2 <= C + 2e + 2 r with r^2 >= 2 C e. Weights come from the float Newton solver and are
rounded to multiples of 2^-40; rounding only costs margin, never validity.
Usage: python sd_exact.py N  -> sd_exact_nN.out (one line per orbit representative: key, L - U as a float, resumable)."""
import sys, os, itertools, math
from fractions import Fraction as Fr
from sd_fast import w2 as w2float, pts as ptsfloat, canon, cell_data

def clip(poly, a, b, c):
    out = []
    m = len(poly)
    for k in range(m):
        P, Q = poly[k], poly[(k + 1) % m]
        fp = a * P[0] + b * P[1] - c; fq = a * Q[0] + b * Q[1] - c
        if fp <= 0:
            out.append(P)
            if fq > 0:
                t = fp / (fp - fq); out.append((P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])))
        elif fq <= 0:
            t = fp / (fp - fq); out.append((P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])))
    return out

def exact_cells(P, w):
    n = len(P); A = []; I = []
    for i in range(n):
        xi, yi = P[i]
        poly = [(Fr(0), Fr(0)), (Fr(1), Fr(0)), (Fr(1), Fr(1)), (Fr(0), Fr(1))]
        for j in range(n):
            if j == i: continue
            xj, yj = P[j]
            poly = clip(poly, 2 * (xj - xi), 2 * (yj - yi), xj * xj + yj * yj - xi * xi - yi * yi - w[j] + w[i])
            if not poly: break
        a = s = Fr(0); m = len(poly)
        for k in range(m):
            x0, y0 = poly[k][0] - xi, poly[k][1] - yi
            x1, y1 = poly[(k + 1) % m][0] - xi, poly[(k + 1) % m][1] - yi
            cr = x0 * y1 - x1 * y0
            a += cr; s += cr * (x0 * x0 + x0 * x1 + x1 * x1 + y0 * y0 + y0 * y1 + y1 * y1)
        A.append(a / 2); I.append(s / 12)
    return A, I

def float_weights(s):
    """Newton weights from sd_fast, re-run here to return w."""
    import numpy as np
    n = len(s); P = ptsfloat(s); w = [0.0] * n
    A, I, L = cell_data(P, w)
    for it in range(60):
        g = [1.0 / n - a for a in A]; gm = max(abs(x) for x in g)
        if gm < 1e-14: break
        H = np.zeros((n, n))
        for (i, j), l in L.items():
            d = math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]); H[i, j] -= l / (2 * d); H[i, i] += l / (2 * d)
        st = np.linalg.lstsq((H + H.T) / 2, np.array(g), rcond=None)[0]; st -= st.mean()
        t = 1.0
        while True:
            wn = [w[k] + t * st[k] for k in range(n)]; An, In, Ln = cell_data(P, wn)
            if (min(An) > 0 and max(abs(1.0 / n - a) for a in An) < gm) or t < 1e-8: break
            t /= 2
        w, A, L = wn, An, Ln
    return w

def exact_pts(s):
    n = len(s)
    return [(Fr(2 * i + 1, 2 * n), Fr(2 * s[i] + 1, 2 * n)) for i in range(n)]

def rat(x, D=2 ** 40):
    return Fr(round(x * D), D)

def lower_identity(n):
    s = list(range(n)); w = [rat(x) for x in float_weights(s)]
    A, I = exact_cells(exact_pts(s), w)
    return sum(I) - sum(wi * ai for wi, ai in zip(w, A)) + sum(w) / n

def upper(s):
    n = len(s); w = [rat(x) for x in float_weights(s)]
    A, I = exact_cells(exact_pts(s), w)
    assert sum(A) == 1
    C = sum(I); e = sum(max(a - Fr(1, n), Fr(0)) for a in A)
    q = 2 * C * e
    r = Fr(math.sqrt(float(q)) * (1 + 1e-6) + 1e-300) if q > 0 else Fr(0)
    while r * r < q: r *= 2
    return C + 2 * e + 2 * r, e

if __name__ == "__main__":
    n = int(sys.argv[1]); out = "sd_exact_n%d.out" % n
    L = lower_identity(n)
    done = set()
    if os.path.exists(out):
        for l in open(out, encoding="utf-8"):
            if l[:1].isdigit(): done.add(l.split()[0])
    f = open(out, "a", encoding="utf-8", newline="\n")
    if not done: f.write("# n=%d exact lower bound for W2^2(identity) %.16f (float of a rational)\n" % (n, float(L))); f.flush()
    ident = tuple(range(n)); rev = canon(ident[::-1])
    bad = 0
    for s in itertools.permutations(range(n)):
        if canon(s) != s or s == ident or s == rev: continue
        key = "".join(map(str, s))
        if key in done: continue
        U, e = upper(list(s))
        ok = U < L; bad += (not ok)
        f.write("%s %s %.6e excess %.1e\n" % (key, "ok" if ok else "FAIL", float(L - U), float(e))); f.flush()
    f.write("# done\n"); f.close()
