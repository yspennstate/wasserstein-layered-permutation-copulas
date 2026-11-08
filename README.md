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
blocks of at most eight consecutive values, and of at most nine when $n\ge10$, with a computer-assisted proof in interval
arithmetic for the blocks of four to nine.
For $n\le10$ the statement is proved for every permutation (Proposition 4.11) by an exact rational computation with
Laguerre cells. The general case remains open.

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
- Blocks of five: `win_r5_mass.py` (the window bound for all 70 blocks on a grid of bands of mass at most 5/9),
  `orbits_r5.py` (the 25 orbits), `gq_scan.py` (the quintile partition on the same grid). Interval arithmetic for
  Proposition 4.5: `gq_onesided_iv.py` and `gq_cross_iv.py 0.05` (the quintile partition for the 24 orbits other than that
  of 12403; `gq_cross_iv_all.out` collects the crossing runs), `hs12403_iv.py 0.05` (the $s$ then $v$ partition for
  12403); `tau12430_G.py`, `tau12430_G_iv.py` and `tau12430_G_cross_iv.py 0.05` treat 12430, the orbit with the
  smallest margin, separately as a check. `gq_cross_formula_res.py` compares the crossing-band formula used in
  `gq_cross_iv.py` with a direct evaluation of the quintile partition on jittered grids of 500, 1000 and 2000 points a
  side; the differences shrink with the grid.
- Blocks of six: `r6_thirds_sign.py` (the partition of Proposition 4.6, thirds of $S$ each split at $V=0$, with the
  best assignment, on a grid of bands of mass at most 6/9; positive for all 460 blocks, output `r6_thirds_sign.out`) and
  `r6_ts_ctrl.py` (the identity is negative and the reversal positive under the same partition). `r6_ts_iv.py` is the
  interval-arithmetic proof of Proposition 4.6 for all 460 blocks (31,694 boxes, output `r6_ts_iv.out`);
  `r6_ts_iv_ctrl.py` checks its enclosures against the floating-point evaluation at point bands (output `ctrl_ts.out`).
- Blocks of seven: `r7_comp_scan.py` evaluates partitions made of pieces of one or two septile strips of $S$, the pairs
  split at $V=0$, from the cell integrals, on a grid of bands of mass at most 7/9. The partition 2221 of Proposition 4.7
  alone is positive for all 3,446 blocks (output `r7_comp_2221.out`; with the best of 2221, 2212, 2122, 1222 per band,
  `r7_comp_2221_2212_2122_1222.out`); `r7_comp_ctrl6.py` checks the same code at six values against `r6_thirds_sign.py`
  (output `ctrl6.out`). `r7_iv.py` is the interval-arithmetic proof of Proposition 4.7 for all 3,446 blocks, run in four
  shards (`--shard i 4`) and merged by `r7_iv_merge.py` (315,613 boxes, output `r7_iv.out`); `r7_iv_ctrl.py` checks its
  enclosures at point bands against `r7_comp_scan.py`, including the reflection used for bands above the diagonal
  (output `r7_iv_ctrl.out`).
- Blocks of eight: `r8_comp_scan.py` evaluates the partition 2222 of Proposition 4.8 (quartiles of $S$, each split at
  $V=0$) from the cell integrals on a grid of bands of mass at most 8/9; it is positive for all 29,092 blocks (output
  `r8_comp_2222.out`). `r8_iv.py` is the interval-arithmetic proof. It runs on one block from each of the 7,578 orbits
  of the inverse and of $i\mapsto7-\tau(7-i)$, in four shards merged by `r8_iv_merge.py` (output `r8_iv.out`), which
  checks that the orbits cover all 29,092 blocks. `r8_orbit_ctrl.py` checks from the cell integrals that $F$ agrees
  across an orbit (output `r8_orbit_ctrl.out`); `r8_iv_ctrl.py` checks the enclosures at point bands, including the
  reflection (output `r8_iv_ctrl.out`).
- Blocks of nine: `r9_size.py` counts the 273,342 blocks and their 69,224 orbits (output `r9_size.out`).
  `r9_comp_scan.py` evaluates the partition 12222 of Proposition 4.9 (one ninth of $S$ left whole, then four pieces of
  two ninths, each split at $V=0$) from the cell integrals on a grid of bands of mass up to 1, the full square included
  (`R9_MMAX=1.0`, output `r9_comp_12222.out`): it is negative for 1,253 blocks, on the full square only, and these are
  positive on bands of mass up to 9/10 (`R9_MMAX=0.9`, output `r9_comp_12222_neg_m09.out`), which is what $n\ge10$ needs.
  `r9_iv.py` is the interval-arithmetic proof of the partition 12222 on bands of mass up to 9/10 (`R9_MASS=9/10`) for one
  block from each orbit of the inverse and of $i\mapsto8-\tau(8-i)$, in six shards merged by `r9_iv_merge.py`
  (7,382,553 boxes, output `r9_iv.out`), which checks that the orbits cover all 273,342 blocks. The map
  $i\mapsto8-\tau(8-i)$ carries the partition 12222 for $\tau$ to the mirror partition 22221 for its image;
  `r9_orbit_ctrl.py` checks this relation and the invariance under the inverse from the cell integrals (output
  `r9_orbit_ctrl.out`), and `r9_iv_ctrl.py` checks the enclosures at point bands, including the reflection (output
  `r9_iv_ctrl.out`).
- Every permutation for $n\le10$: `sd_fast.py N` solves the semi-discrete problem between the grid points and $\lambda$
  (Laguerre cells clipped to the square, Newton's method on the dual) for one permutation in each class under the
  symmetries of the square (outputs `sd_scan_n3.out` to `sd_scan_n9.out`). `sd_exact.py N` is the proof of
  Proposition 4.11: with the weights rounded to multiples of $2^{-40}$ it evaluates, in exact rational arithmetic, a dual
  lower bound for the identity and a primal upper bound for each class (outputs `sd_exact_n3.out` to `sd_exact_n9.out`,
  one line per class, `ok` when the upper bound lies below the lower bound). For $n=10$ `sd_exact_shard.py 10 4 k`,
  $k=0,\dots,3$, runs the same certificate in four shards (outputs `sd_exact_n10_s4_0.out` to `sd_exact_n10_s4_3.out`),
  and `sd_exact_verify.py 10 4` checks that the shards finished, that every class appears once, that their number is
  the Burnside count 456,453 and that none fails (output `sd_exact_n10.verify`); `sd_canon_check.py 10 4` checks that each
  listed permutation is the canonical representative of its class, so that distinct lines are distinct classes
  (output `sd_canon_check_n10.out`). `sd_exact_ctrl.py` runs the same bounds on
  the identity and the reversal, where they must not separate (output `sd_exact_ctrl.out`).

## Funding

The research presented in this paper was supported by the European Research Council (ERC) under the European
Union's Horizon Europe research and innovation programme (grant agreement No. 101041711), by the Simons Foundation
as part of the Collaboration on the Mathematical and Scientific Foundations of Deep Learning, by Heights Labs, by
Convex Nexus Capital, by the Israel Science Foundation (grant number 2258/19), and by the Israel Science Foundation
(ISF Grant 4101/25). Computational resources were provided in part by Google Cloud through its education credits
program (credit 530550103).
