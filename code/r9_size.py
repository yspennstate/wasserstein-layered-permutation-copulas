"""Size the blocks-of-nine step: number of indecomposable non-reversal blocks of nine and of orbits under {inverse, rc}."""
import itertools
R = 9
indec = lambda t: all(set(t[:j]) != set(range(j)) for j in range(1, len(t)))
inv = lambda t: tuple(sorted(range(len(t)), key=lambda i: t[i]))
rc = lambda t: tuple(len(t) - 1 - t[len(t) - 1 - i] for i in range(len(t)))
rev = tuple(range(R - 1, -1, -1))
TAUS = [t for t in itertools.permutations(range(R)) if indec(t) and t != rev]
REPS = {min({t, inv(t), rc(t), rc(inv(t))}) for t in TAUS}
print("r=%d: %d blocks, %d orbit representatives" % (R, len(TAUS), len(REPS)))
