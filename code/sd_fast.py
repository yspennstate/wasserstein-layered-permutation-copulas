"""Fast exact semi-discrete W2^2 (grid points of a permutation vs uniform on the unit square): Laguerre cells clipped
to the square with edge labels, Newton on the concave dual with the analytic Hessian dA_i/dw_j = -L_ij/(2|p_i-p_j|).
Scan: python sd_fast.py N [out]  -> W2^2(sigma) - W2^2(id) for one representative of each orbit of the square's
symmetry group (inverse, reverse, complement), resumable, one line per rep."""
import sys, os, itertools, math

def clip(poly, a, b, c, lab):        # poly: list of (x, y, label of edge from this vertex to the next)
    out = []
    m = len(poly)
    for k in range(m):
        px, py, pl = poly[k]; qx, qy, _ = poly[(k + 1) % m]
        fp, fq = a * px + b * py - c, a * qx + b * qy - c
        if fp <= 0:
            if fq > 0:
                t = fp / (fp - fq)
                out.append((px, py, pl)); out.append((px + t * (qx - px), py + t * (qy - py), lab))
            else:
                out.append((px, py, pl))
        elif fq <= 0:
            t = fp / (fp - fq)
            out.append((px + t * (qx - px), py + t * (qy - py), pl))
    return out

def cell_data(P, w):
    n = len(P); A = [0.0] * n; I = [0.0] * n; L = {}
    for i in range(n):
        xi, yi = P[i]
        poly = [(0.0, 0.0, -1), (1.0, 0.0, -1), (1.0, 1.0, -1), (0.0, 1.0, -1)]
        for j in range(n):
            if j == i: continue
            xj, yj = P[j]
            poly = clip(poly, 2 * (xj - xi), 2 * (yj - yi), xj * xj + yj * yj - xi * xi - yi * yi - w[j] + w[i], j)
            if not poly: break
        a = s = 0.0
        m = len(poly)
        for k in range(m):
            x0, y0, lb = poly[k]; x1, y1, _ = poly[(k + 1) % m]
            x0 -= xi; y0 -= yi; x1 -= xi; y1 -= yi
            cr = x0 * y1 - x1 * y0
            a += cr; s += cr * (x0 * x0 + x0 * x1 + x1 * x1 + y0 * y0 + y0 * y1 + y1 * y1)
            if lb >= 0:
                L[(i, lb)] = L.get((i, lb), 0.0) + math.hypot(x1 - x0, y1 - y0)
        A[i] = a / 2; I[i] = s / 12
    return A, I, L

def w2(P, tol=1e-14, maxit=60):
    import numpy as np
    n = len(P); w = [0.0] * n
    A, I, L = cell_data(P, w)
    for it in range(maxit):
        g = [1.0 / n - a for a in A]
        gm = max(abs(x) for x in g)
        if gm < tol: break
        H = np.zeros((n, n))
        for (i, j), l in L.items():
            d = math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1])
            H[i, j] -= l / (2 * d); H[i, i] += l / (2 * d)
        H = (H + H.T) / 2
        st = np.linalg.lstsq(H, np.array(g), rcond=None)[0]   # A(w+st) ~ A + H st = 1/n
        st -= st.mean()
        t = 1.0
        while True:
            wn = [w[k] + t * st[k] for k in range(n)]
            An, In, Ln = cell_data(P, wn)
            if (min(An) > 0 and max(abs(1.0 / n - a) for a in An) < gm) or t < 1e-8: break
            t /= 2
        w, A, I, L = wn, An, In, Ln
    val = sum(I) - sum(w[k] * A[k] for k in range(n)) + sum(w) / n
    return val, max(abs(1.0 / n - a) for a in A), it

def pts(s):
    n = len(s)
    return [((i + .5) / n, (s[i] + .5) / n) for i in range(n)]

def canon(s):
    n = len(s)
    inv = [0] * n
    for i, v in enumerate(s): inv[v] = i
    best = None
    for t in (s, inv):
        for r in (False, True):
            for c in (False, True):
                u = list(t[::-1]) if r else list(t)
                if c: u = [n - 1 - v for v in u]
                u = tuple(u)
                if best is None or u < best: best = u
    return best

if __name__ == "__main__":
    n = int(sys.argv[1]); out = sys.argv[2] if len(sys.argv) > 2 else "sd_scan_n%d.out" % n
    base = w2(pts(list(range(n))))
    done = set()
    if os.path.exists(out):
        for l in open(out):
            if l[:1].isdigit(): done.add(l.split()[0])
    f = open(out, "a", newline="\n")
    if not done: f.write("# n=%d identity W2^2 %.16f grad %.1e\n" % (n, base[0], base[1])); f.flush()
    ident = tuple(range(n))
    for s in itertools.permutations(range(n)):
        if canon(s) != s or s == ident or s == canon(ident[::-1]): continue
        key = "".join(map(str, s))
        if key in done: continue
        v, g, it = w2(pts(s))
        f.write("%s %+.6e %.1e %d\n" % (key, v - base[0], g, it)); f.flush()
    f.write("# done\n"); f.close()
