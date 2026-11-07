from r6_thirds_sign import F6, Hinv
from win_r import F
for t in [(0,1,2,3,4,5),(5,4,3,2,1,0),(1,3,0,5,4,2),(1,4,0,5,3,2)]:
    for q, m in ((0.1, 0.3), (0.3, 0.5), (0.9, 0.05)):
        a, b = Hinv(q), Hinv(q + m)
        print(t, q, m, "F6/m %+.4f" % (F6(a, b, t)[0] / m), "cut/m %+.4f" % (F(a, b, t) / m))
