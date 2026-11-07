"""Closed forms of the two cuts of win_r for tau = 2301, 2310, 1230 (any band (a,b)), checked against the polygon code,
and the comparisons with F(1302) = 4A + 2B - 3(S4-S1) - (S3-S2).
Notation as tau3120_any: S_j = E[s | j-th s-quarter], A/B = E[|v| | |v| above/below its median xm], E = E[|v| | s in the
middle two quarters], Vbar = E|v| = (A+B)/2.
2301 points (-1,-2),(1,-2),(-1,2),(1,2): both cuts give 8 Vbar - 2(S4-S1) = F(1302) + 2B + (S4-S1) + (S3-S2).
2310 points (-1,-2),(1,-2),(0,1),(0,3): 'v then s' gives 5A + 3B - (5/2)(S4-S1) - (1/2)(S3-S2)
      = F(1302) + A + B + ((S4-S1) + (S3-S2))/2.
1230 points (-2,-1),(0,-1),(2,-1),(0,3): 'v then s' gives 4A + 2(R3-R1) - 3(S4-S1) - (S3-S2), R_j = E[s | j-th s-third of
      {v < xm}]; 's then v' gives 4E - (S4-S1) - (S3-S2)."""
import sys
import numpy as np
from tau1302_any import integ, bis, F as F1302
from tau3120_any import parts, L
def R13(a, b, xm):
    k = lambda s: L(s) + min(L(s), xm); kb = [1.0, xm, 2 - xm]
    Mk = integ(k, a, b, kb)
    c1 = bis(lambda x: integ(k, a, x, kb), a, b, Mk / 3); c2 = bis(lambda x: integ(k, a, x, kb), a, b, 2 * Mk / 3)
    R1 = integ(lambda s: s * k(s), a, c1, kb) / (Mk / 3); R3 = integ(lambda s: s * k(s), c2, b, kb) / (Mk / 3)
    return R1, R3
def xm_of(a, b):
    h = lambda x: max(0.0, min(b, 2 - x) - max(a, x)); xb = [a, 2 - b, 1.0]
    top = max(L(s) for s in (a, b, 1.0) if a <= s <= b); Mh = integ(h, 0, top, xb)
    return bis(lambda x: integ(h, 0, x, xb), 0, top, Mh / 2)
def AB(a, b):
    h = lambda x: max(0.0, min(b, 2 - x) - max(a, x)); xb = [a, 2 - b, 1.0]
    top = max(L(s) for s in (a, b, 1.0) if a <= s <= b); Mh = integ(h, 0, top, xb)
    xm = bis(lambda x: integ(h, 0, x, xb), 0, top, Mh / 2)
    return integ(lambda x: x * h(x), xm, top, xb) / (Mh / 2), integ(lambda x: x * h(x), 0, xm, xb) / (Mh / 2), xm
def closed(a, b):
    S, A, Tlo, Thi, E = parts(a, b); A2, B, xm = AB(a, b)
    d41, d32 = S[3] - S[0], S[2] - S[1]
    R1, R3 = R13(a, b, xm)
    return {"2301v": 4 * A + 4 * B - 2 * d41, "2301s": 4 * A + 4 * B - 2 * d41,
            "2310v": 5 * A + 3 * B - 2.5 * d41 - 0.5 * d32,
            "1230v": 4 * A + 2 * (R3 - R1) - 3 * d41 - d32, "1230s": 4 * E - d41 - d32,
            "1302": 4 * A + 2 * B - 3 * d41 - d32, "A-A2": A - A2, "B": B, "E": E, "R": R3 - R1, "d41": d41, "d32": d32}
if __name__ == "__main__":
    from win_r import cutval, band, am, slices
    out = []
    for a, b in [(0.3, 0.9), (0.0, 0.5), (0.5, 1.4), (0.8, 1.2), (0.2, 1.8), (0.05, 0.3), (0.6, 0.7)]:
        c = closed(a, b); U = band(a, b); M, _ = am(U); st = slices(U, M, 4, 0)
        idv = sum((2 * j - 3) * am(Sx)[1][0] for j, Sx in enumerate(st))
        line = "(%.2f,%.2f)" % (a, b)
        for name, tau in (("2301", (2, 3, 0, 1)), ("2310", (2, 3, 1, 0)), ("1230", (1, 2, 3, 0))):
            pts = [(i + tau[i] - 3, i - tau[i]) for i in range(4)]
            pv, ps = cutval(U, M, pts, 1, 4) - idv, cutval(U, M, pts, 0, 4) - idv
            line += " | %s v %+.7f/%+.7f" % (name, pv, c[name + "v"])
            if name + "s" in c: line += " s %+.7f/%+.7f" % (ps, c[name + "s"])
            else: line += " s(poly) %+.7f" % ps
        out.append(line)
    G = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    mins = {}
    for a in np.linspace(0, 1, G + 1):
        for b in np.linspace(a, 2 - a, G + 1)[1:]:
            if b - a < 1e-9: continue
            m = integ(L, a, b, [1.0])
            if m > 4 / 9 + 1e-12: continue
            c = closed(a, b)
            for k, v in (("1230v/m", c["1230v"] / m), ("1230s/m", c["1230s"] / m),
                         ("1230max/m", max(c["1230v"], c["1230s"]) / m), ("1230s-(2/3)3120s", 0),
                         ("R-B", c["R"] - c["B"]), ("E-A", c["E"] - c["A-A2"] * 0 - 0)):
                if k not in mins or v < mins[k][0]: mins[k] = (v, a, b)
    for k, v in mins.items(): out.append("scan G=%d a+b<=2 mass<=4/9: min %s %+.5f at (%.3f,%.3f)" % ((G, k) + v))
    open(__file__.replace(".py", ".out"), "w", newline="\n").write("\n".join(out) + "\n"); print("\n".join(out))
