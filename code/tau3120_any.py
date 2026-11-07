"""tau = 3120 on any band (a,b): closed forms of the two cuts of win_r.
Points (u',v'): (0,-3), (-1,0), (1,0), (0,3); identity u' = -3,-1,1,3.
'v then s': Fv = 6A + (T_hi - T_lo) - 3(S4-S1) - (S3-S2), A = E[|v| | |v| > xm] (xm = |v|-median), T_hi/T_lo = E[s] on the
  upper/lower s-half of {|v| < xm}.
's then v': Fs = 6E - 2(S4-S1) - (S3-S2), E = E[|v| | s in the middle two quarters] (E[|v| | s] = L(s)/2).
F = max(Fv, Fs). Checked against the polygon code win_r.F."""
from tau1302_any import integ, bis
L = lambda s: min(s, 2 - s)
def parts(a, b):
    bs = [1.0]
    M = integ(L, a, b, bs)
    sq = [a] + [bis(lambda x: integ(L, a, x, bs), a, b, p * M) for p in (.25, .5, .75)] + [b]
    S = [integ(lambda s: s * L(s), sq[j], sq[j + 1], bs) / (M / 4) for j in range(4)]
    h = lambda x: max(0.0, min(b, 2 - x) - max(a, x))
    xb = [a, 2 - b, 1.0]; top = max(L(s) for s in (a, b, 1.0) if a <= s <= b)
    Mh = integ(h, 0, top, xb)
    xm = bis(lambda x: integ(h, 0, x, xb), 0, top, Mh / 2)
    A = integ(lambda x: x * h(x), xm, top, xb) / (Mh / 2)
    g = lambda s: min(L(s), xm); gb = bs + [xm, 2 - xm]
    Mg = integ(g, a, b, gb)
    sm = bis(lambda x: integ(g, a, x, gb), a, b, Mg / 2)
    Tlo = integ(lambda s: s * g(s), a, sm, gb) / (Mg / 2); Thi = integ(lambda s: s * g(s), sm, b, gb) / (Mg / 2)
    E = integ(lambda s: L(s) ** 2 / 2, sq[1], sq[3], bs) / (M / 2)
    return S, A, Tlo, Thi, E
def Fv(a, b):
    S, A, Tlo, Thi, E = parts(a, b); return 6 * A + (Thi - Tlo) - 3 * (S[3] - S[0]) - (S[2] - S[1])
def Fs(a, b):
    S, A, Tlo, Thi, E = parts(a, b); return 6 * E - 2 * (S[3] - S[0]) - (S[2] - S[1])
if __name__ == "__main__":
    import sys
    from win_r import F as Fpoly, cutval, band, am, slices
    out = []
    tau = (3, 1, 2, 0)
    for a, b in [(0.3, 0.9), (0.0, 0.5), (0.5, 1.4), (0.8, 1.2), (0.2, 1.8), (0.05, 0.3)]:
        U = band(a, b); M, _ = am(U); st = slices(U, M, 4, 0)
        idv = sum((2 * j - 3) * am(Sx)[1][0] for j, Sx in enumerate(st))
        pts = [(i + tau[i] - 3, i - tau[i]) for i in range(4)]
        pv, ps = cutval(U, M, pts, 1, 4) - idv, cutval(U, M, pts, 0, 4) - idv
        out.append("(%.2f,%.2f): Fv poly %+.8f closed %+.8f | Fs poly %+.8f closed %+.8f | F/(b-a) %+.4f" % (
            a, b, pv, Fv(a, b), ps, Fs(a, b), max(pv, ps) / (b - a)))
    # scan of bands with a+b <= 2 and mass <= 4/9
    import numpy as np
    mv, ms, mx = (9, 0, 0), (9, 0, 0), (9, 0, 0)
    G = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    for a in np.linspace(0, 1, G + 1):
        for b in np.linspace(a, 2 - a, G + 1)[1:]:
            if b - a < 1e-9: continue
            m = integ(L, a, b, [1.0])
            if m > 4 / 9 + 1e-12: continue
            v, s = Fv(a, b), Fs(a, b)
            mv = min(mv, (v / m, a, b)); ms = min(ms, (s / m, a, b)); mx = min(mx, (max(v, s) / m, a, b))
    out.append("scan G=%d, a+b<=2, mass<=4/9: min Fv/m %+.4f at (%.3f,%.3f); min Fs/m %+.4f at (%.3f,%.3f); min max/m %+.4f at (%.3f,%.3f)" % ((G,) + mv + ms + mx))
    open(__file__.replace(".py", ".out"), "w", newline="\n").write("\n".join(out) + "\n"); print("\n".join(out))
