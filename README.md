# Layered permutation copulas and the Wasserstein distance to the uniform law

Yitzchak Shmalo

Paper: [paper/main.pdf](paper/main.pdf) (source [paper/main.tex](paper/main.tex)).

Schrott (arXiv:2609.10467, Conjecture 1.1) conjectured that every copula $\pi$ on $[0,1]^2$ satisfies
$W_2^2(\pi,\lambda)\le 1/10$, where $\lambda$ is the uniform law and $1/10$ is attained by the comonotone copula. By an
empirical-rank argument the conjecture is equivalent to a statement about the $n$-point permutation measures on the grid
points $((i+\frac12)/n,(\sigma(i)+\frac12)/n)$: the identity permutation is the farthest from $\lambda$ for every $n$.
The note proves this for every layered permutation (a concatenation of decreasing runs of consecutive values) and its
images under the symmetries of the square: each such permutation other than the identity and the reversal is strictly
closer to $\lambda$ than the identity. Section 4 extends this, for $n\ge 9$, to permutations that split into layers and
blocks of at most four consecutive values, with a computer-assisted proof in interval arithmetic for the blocks of four.
The general case remains open; the computations of Section 5 find no counterexample among all permutations with
$n\le 8$.

## Code

Python 3 with numpy and scipy. Each script writes its output next to itself.

- `a_enum7.py 40 0.001 8` (grid, smoothing, $n$; solver in `a_enum.py`): every permutation of size $n$ up to the symmetries of the square (5,282 classes at $n=8$), with the
  optimal-transport value computed by a smoothed semi-discrete dual. Outputs `a_enum7_n7.out`, `a_enum7_n8.out`.
- `block_rev_exact.py 48` (closed forms in `adj_swap_bound.py`): the lower bound $D$ for every block reversal, $3\le n\le 48$. Output `block_rev_exact_n48.out`.
- `audit_split_identity.py`: the per-split identity of Lemma 2.2 against the direct evaluation of $D$ for
  $n=5,8,13$. Output `audit_split_identity.out`.
- `dispersive_check.py`, `two_cell_check2.py`: grid checks of Lemma 3.1 and of the per-split inequality on all bands.
- `theta_cut.py`: cuts of a band along a fixed direction, the computation behind Section 6.
- `cyc3_window.py`, `win_r.py`: the window bound $F$ of Lemma 4.1 for the 3-cycles and for any block, on a grid of bands
  (outputs `cyc3_window.out`, `win_r3.out`, `win_r4.out`, `win_r4_mass.out`); `orbits_r4.py`: the six orbits of the
  twelve blocks of four and the invariance of $F$ along each orbit.
- `tau1302_closed.py`, `tau1320_closed.py`, `tau_rest.py` (with `tau1302_any.py`, `tau3120_any.py`): the closed forms of
  Lemma 4.3 against the direct evaluation of the cut.
- Interval arithmetic (mpmath): `tau1230_lower_iv.py` and `tau1320_onesided_iv.py` (bands on one side of the diagonal),
  `tau1230_cross_iv.py 0.05` and `tau1320_cross_iv.py 0.25` (bands crossing it, mass at most 4/9), the proof of
  Proposition 4.4; `tau1302_lower_iv.py` and `tau1302_cross_iv.py 0.05` prove $F(1302)>0$ directly as a check.

## Funding

The research presented in this paper was supported by the European Research Council (ERC) under the European
Union's Horizon Europe research and innovation programme (grant agreement No. 101041711), by the Simons Foundation
as part of the Collaboration on the Mathematical and Scientific Foundations of Deep Learning, by Heights Labs, by
Convex Nexus Capital, by the Israel Science Foundation (grant number 2258/19), and by the Israel Science Foundation
(ISF Grant 4101/25). Computational resources were provided in part by Google Cloud through its education credits
program (credit 530550103).
