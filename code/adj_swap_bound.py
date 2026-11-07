"""Adjacent swap (k,k+1) of id, diagonal re-cut of the union U of strips k,k+1.
Gain >= (1/(2n^2))(E[L(S)|U] - (m_{k+1}-m_k)), L(s)=min(s,2-s), m_j = E[S | strip j]; all exact via the density f(s)=L(s).
Checks the sign for every k and n<=400, and the n^2*gain profile."""
import numpy as np
from math import sqrt

def Finv(u):  # quantile of S = X1+X2
    return sqrt(2 * u) if u <= 0.5 else 2 - sqrt(2 * (1 - u))

def mom(a, b, p):  # int_a^b s^p L(s) ds
    def piece(lo, hi, left):
        if hi <= lo:
            return 0.0
        if left:  # L = s
            return (hi ** (p + 2) - lo ** (p + 2)) / (p + 2)
        # L = 2 - s
        return 2 * (hi ** (p + 1) - lo ** (p + 1)) / (p + 1) - (hi ** (p + 2) - lo ** (p + 2)) / (p + 2)
    return piece(a, min(b, 1), True) + piece(max(a, 1), b, False)

def momL(a, b):  # int_a^b L(s)^2 ds
    def piece(lo, hi, left):
        if hi <= lo:
            return 0.0
        if left:
            return (hi ** 3 - lo ** 3) / 3
        return ((2 - lo) ** 3 - (2 - hi) ** 3) / 3
    return piece(a, min(b, 1), True) + piece(max(a, 1), b, False)

worst = 9.0; lines = []
for n in range(2, 401):
    q = [Finv(j / n) for j in range(n + 1)]; q[-1] = 2.0
    m = [n * mom(q[j], q[j + 1], 1) for j in range(n)]
    g = []
    for k in range(n - 1):
        EL = (n / 2) * momL(q[k], q[k + 2])
        gain = (EL - (m[k + 1] - m[k])) / (2 * n * n)
        g.append(gain)
        if n >= 3: worst = min(worst, gain * n ** 2.5)
    if n in (4, 8, 16, 64, 400):
        lines.append("n=%3d min n^2.5*gain %.4f (k=%d)  corner n^2.5*gain %.4f  middle n^2*gain %.4f" % (
            n, min(g) * n ** 2.5, int(np.argmin(g)), g[-1] * n ** 2.5, g[(n - 1) // 2] * n ** 2))
lines.append("min over 3<=n<=400, all k, of n^2.5*gain: %.5f  (corner closed form %.5f)" % (worst, (2 * sqrt(2) - 2) / 3))
print("\n".join(lines))
open(__file__.replace(".py", ".out"), "w", newline="\n").write("\n".join(lines) + "\n")
