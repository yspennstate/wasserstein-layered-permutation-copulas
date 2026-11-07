from mpmath import mpf, iv
from r6_ts_iv import best_assign, F_iv
from r6_thirds_sign import F6
for tau in [(0,1,2,3,4,5),(1,3,0,5,4,2),(1,2,3,4,5,0)]:
    pts = [(i + tau[i] - 5, i - tau[i]) for i in range(6)]
    for a, b in ((0.3, 0.9), (0.5, 1.6), (1.2, 1.7), (0.0, 1.15)):
        asg, ff = best_assign(pts, a, b)
        fi = F_iv(pts, mpf(a), mpf(a), mpf(b), mpf(b), asg)
        print(tau, a, b, "float %+.6f  F6 %+.6f  iv [%s, %s]" % (ff, F6(a, b, tau)[0], iv.nstr(iv.mpf(fi.a), 8), iv.nstr(iv.mpf(fi.b), 8)))
