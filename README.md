# CT formula derivation: the transcorrelated Hamiltonian through O(lambda^2)

This repo derives, step by step and with a numerical check at every step,
the transcorrelated Hamiltonian

    H_bar = e^{-A} H e^A,   H = F + V,   A = T1 + T2,

through O(lambda^2) in Moller--Plesset counting (F ~ O(1), V, T ~ O(lambda)):

    H_bar = H + [F, A] + [V, A] + (1/2)[[F, A], A]  +  O(lambda^3)

(the valence double commutator (1/2)[[V, A], A] is O(lambda^3) and is
dropped).  The final result is the twelve commutator blocks with all
three-body operators folded back into two-body, one-body and scalar parts
by the cumulant expansion, collected per operator, ready for
substitution: see `cumulant_expansion/cumulant_expansion.pdf`, section
"All final blocks, grouped by operator".

## Repo map: prerequisites vs process

The first, second and fifth directories are **prerequisites** -- tools and
input formulas, containing no derivation choices of their own.  The third,
fourth and sixth are **the derivation process**, applied in order.

### Prerequisites

1. `generalized_wick_theorem/` -- the commutator engine.  Expands
   commutators of number-conserving excitation operators E with the
   generalized Wick theorem (Kutzelnigg--Mukherjee, paired-order form, no
   sign rule).  Self test: every printed rule of the reference note
   reproduced term by term, plus exhaustive Jordan--Wigner matrix sweeps.

2. `commutator/` -- the CT commutators themselves.  `generate_commutators.py`
   builds the 12 symbolic blocks (4 first-order [F/V, T_r] and 8
   second-order Fock double-commutator blocks) with sympy's Commutator and
   matrix-checks them; `ct_*.tex/.pdf` are the derivation notes.

5. `cumulant_decomposition/spin_free/` -- the input formula: the 51-term
   spin-free cumulant decomposition of the three-body operator
   E^{pqr}_{stu} into density matrices times two-/one-body operators plus
   scalar terms (`spin_free.tex`).  The formula is transcribed as
   `TEMPLATE` in stage 6 and verified there numerically from scratch
   (exact on random index assignments against Jordan--Wigner matrices), so
   the pipeline does not rely on trusting the notebook that produced it.

### The derivation process (in order)

3. `many_body_equation/` -- expand all 12 blocks with the generalized Wick
   theorem; apply the index-space delta rules (numbered indices:
   p1,p2,... whole space; i1,i2,... occupied; a1,a2,... virtual);
   merge equal terms over every exact identity (merge: index
   renaming, pair symmetries, diagonal pair permutation of E).
   Checks: Jordan--Wigner matrix check per block, both with
   spin-orbital operators and with genuinely spin-summed operators.

4. `space_restriction/` -- the CF-F12 (Yanai) OBS-virtual restriction: a
   term is zero iff both virtual indices of ONE AND THE SAME t2 amplitude
   sit on the E operator (contracted virtuals are fine).  Killed terms are
   underlined in the pdf, survivors gathered.  `full_expression()` then
   assembles the restricted blocks into the full transcorrelated
   Hamiltonian (blocks + h.c., 1/2 on the Fock double commutator, merged);
   its doubles-only part is eq. (S1) of the Supplemental Material of
   Masteran et al., J. Chem. Phys. 158, 057101 (2023)
   (`reference/supporting.pdf`).  Check (`test/check_reference_s1.py`):
   the assembled doubles expression equals the printed 16-term (S1)
   exactly (spin-summed Jordan--Wigner matrices, OBS-vir/CABS split), the
   killed terms vanish, and the assembly equals the direct nested
   commutator with the anti-Hermitian generator A = T - T^dagger.

6. `cumulant_expansion/` -- decompose the surviving three-body terms with
   the grouped template.  Every template index appears in exactly one
   factor of each piece, so a whole operator group survives or dies
   together: a group survives iff its operator carries every virtual index
   of the pattern (a density matrix with a virtual index is zero; the
   pure-density-matrix scalar part always vanishes).  Checks: the template
   identity itself (exact, SD reference, random assignments), the template
   grouping, the group-vs-flat decomposition equivalence, and the final
   block grouping.  Output: the final blocks, flat and grouped by
   operator.

7. `f12_special_term/` -- the final piece: the contribution of the fixed
   F12 geminal amplitudes (CABS space, SP/cusp coefficients) that
   compensates the stage-4 space restriction.  The stage-4 killed terms
   are resurrected, folded with the same cumulant template, and
   substituted into the stage-6 blocks; the assembled special term (32
   terms, 24 doubles-only) and the substituted final formula (166 terms,
   126 doubles-only) are written to `f12_special_term.tex/.pdf`.
   Checks: the killed terms are exactly the complement of the stage-4
   survivors; the cumulant folding of the killed three-body patterns is
   exact (SD reference, Jordan--Wigner); the assembled special term
   vanishes in the F12 setting (so (S1) is untouched).  The substituted
   doubles formula is compared with eqs. (S3)/(S4) of the reference SI
   (`test/check_reference_s34.py`, parser `test/si_parse.py` on
   `reference/supporting.tex`): the bare and all first-order content
   match EXACTLY; the second-order f.G.G residual (the SI's
   F12-intermediate/screening content) is reported in three tensor
   setups.  See its readme.md for the details.

## Conventions in force

- Spin-integrated formalism: NO index antisymmetry of t or g anywhere;
  the only symmetries are the simultaneous pair swaps
  t^{ab}_{ij} = t^{ba}_{ji}, g^{pq}_{rs} = g^{qp}_{sr}.
- Real coefficients: stars dropped, f symmetric, g symmetric under pair
  exchange, g^{pq}_{rs} = g^{rs}_{pq}.
- Density matrices D^i_j, D^{ij}_{kl}, ... are kept symbolic (no
  delta-function substitution); any D carrying a virtual index is zero.
- The stages up to and including space restriction are operator
  identities; the reference state enters only in the cumulant stage
  through the density matrices.
- The truncation is by perturbation order: O(lambda^2) everywhere.

## How to verify

Run the whole pipeline (a few minutes, all checks printed):

    ./run_all.sh

or each stage separately, in order:

    cd generalized_wick_theorem && python3 generalized_wick_commutator.py
    cd commutator               && python3 generate_commutators.py
    cd many_body_equation       && python3 expand_commutators.py
    cd space_restriction        && python3 space_restriction.py
    cd cumulant_expansion       && python3 cumulant_expansion.py
    cd f12_special_term         && python3 f12_special_term.py

Each stage regenerates its .tex and builds the .pdf (latexmk required for
the pdf; the checks themselves need only python3 with sympy and numpy).
The check code lives in `test/check_<stage>.py` (imported and run by each
stage script, also runnable standalone) and the shared pdf-build and
index-LaTeX helpers in `utils/latex_utils.py`.

Expected block term counts (blocks: 1-4 first order [F,T1] [F,T2] [V,T1]
[V,T2]; 5-12 the eight second-order Fock blocks, [[F,T_r],T_s] then
[[F,T_r]^dag,T_s]):

| stage | counts |
|---|---|
| 3. expanded | 2, 2, 2, 4 / 1, 4, 2, 4, 2, 8, 1, 12 |
| 4. survivors | 2, 1, 2, 2 / 1, 4, 1, 2, 1, 6, 0, 6 |
| 6. final | 2, 1, 2, 16 / 1, 4, 1, 2, 1, 6, 0, 38 |

In stage 6 the single three-body term of block 4 decomposes into 15 terms
(9 of the 19 template groups survive: 6 two-body + 3 one-body), the four
three-body terms of block 12 into 36 (9 groups for the one-virtual
pattern, 5 for the two-virtual patterns), and the four
two-upper-virtual three-body terms of block 12 (E^{abi}_{cjp} twice,
E^{abi}_{cdj}, E^{abp}_{cdi}) are already zero from stage 4 -- each
carries one t2 amplitude with both of its virtual indices on the operator.
