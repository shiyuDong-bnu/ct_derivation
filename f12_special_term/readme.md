# f12_special_term -- the final piece

The CT-F12 correlator is

    A = T1 + T2(OBS, optimized) + T_F12(fixed geminal, CABS)

where the F12 geminal amplitudes are fixed by the cusp condition (SP
ansatz) and live in the CABS virtual space:

    t~^{ab}_{ij} = (3/8) <ab|Q12 f12|ij> + (1/8) <ab|Q12 f12|ji>

The stage-4 space restriction killed exactly the terms whose t2 has both
OBS virtuals on the E operator; the F12 geminal is what compensates that
restriction, and its contribution to the transcorrelated Hamiltonian is
the "F12 special term" derived here.

## What the (S1)/(S3)/(S4) comparison established (2026-10-13)

Reference: Supplemental Material of Masteran et al., J. Chem. Phys. 158,
057101 (2023) (`../reference/supporting.pdf`; arXiv:2211.09685 source
used for exact transcription) and Yanai--Shiozaki, J. Chem. Phys. 136,
084107 (2012) (`../reference/084107_1_online.pdf`).

* Eq. (S1) (BCH through O(lambda^2), WITHOUT cumulant decomposition):
  our stage-4 expression (restricted blocks + h.c., 1/2 on the Fock
  double commutator) reproduces the printed 16 terms EXACTLY -- verified
  term by term (canonical forms) and numerically (1e-13) in
  `test/check_reference_s1.py`.
* The cumulant template of stage 6 is EXACTLY the SI's own 51-term
  formula (their eq. for the E3 decomposition): both satisfy
  E3 - (Fermi-vacuum normal-ordered E3) = template on random index
  assignments with SD-reference RDMs (max deviation 0.0).
* Therefore our stage-6 output = (S1) + cumulant truncation, EXACTLY
  (8.5e-14).

* Eqs. (S3)/(S4) (the dressed 1-/2-body matrix elements h_bar, g_bar):
  the bare h/g content and ALL first-order content (the h.G and g.G
  classes, with RDMs) match our stage-6 result EXACTLY (canonical forms,
  coefficients, and numerics).  The SECOND-ORDER f.G.G content does NOT
  match the printed (S3)/(S4): those contain f^{i}_{j}
  (occupied-occupied) contractions and f^i_i trace pieces that the plain
  truncation of (S1) cannot produce (the cumulant template never rewrites
  the coefficient tensors), plus the X.f / B intermediate pieces.

  The extra content is the F12-intermediate machinery of Yanai--Shiozaki
  (their eqs. (23)-(27), with eq. (27)'s spurious 1/2 removed per the
  Comment):

      V^{pq}_{ij}  = g^{pq}_{ab} G^{ab}_{ij}              (23)
      V~^{ij}_{pq} = g^{ab}_{pq} G^{ij}_{ab}              (adjoint of V)
      X^{kl}_{ij}  = G^{ab}_{ij} G^{ab}_{kl}              (24)
      B^{kl}_{ij}  = G^{ab}_{ij} f^a_c G^{cb}_{kl}
                     + G^{ab}_{ij} f^b_c G^{ac}_{kl}      (25)
      U^{prs}_{ijb}= g^{pr}_{xs} G^{xb}_{ij}              (26)
      S^{klb}_{ija}= G^{xa}_{ij} G^{yb}_{kl}
                     [f^x_y + delta^x_y (f^a_a - f^i_i - f^j_j)]   (27,
                                                     corrected: no 1/2)

  (a,b,x,y over the complete virtual space; the f^a_a / f^i_i pieces are
  virtual/occupied Fock traces -- the source of the f(oo) content.)

  The SI authors' SeQuant driver (examples/uccf12/uccf12.cpp in
  ValeevGroup/SeQuant) applies, after Wick and before the cumulant
  substitution, extra F12-specific simplification steps
  (`overlap_with_obs`, `screen_F12_proj`) and afterwards
  `hamiltonian_based_projector_2` / `fock_based_projector_2`.  The
  printed (S3)/(S4) are the result of those steps, not of the plain
  truncation.

  Numerically (4 spatial orbitals {2 occ, 1 OBS-vir, 1 CABS}, random
  tensors, SD reference): the residual |ours - SI| is ~24 with generic
  geminal amplitudes, ~5.6 with true SP-form amplitudes, ~0.5 adding a
  canonical (diagonal) Fock matrix -- i.e. the SI's printed formulas
  lean on the F12 geminal/orbital structure, and a residue remains that
  is presumably the SeQuant screening steps.  The first-order match is
  assumption-free and exact.

  Scratch code of the comparison: /tmp/s34_check/ (`compare_abc.py`,
  `structural_s34.py`, `arbiter.py`, `per_class.py`).  The SI tex parser
  and the final matrix-level comparison now live in the repo:
  `test/si_parse.py` and `test/check_reference_s34.py`.

## Consequences for this stage

The stage-6 result is exactly (S1) + cumulant truncation; the resurrected
geminal commutators (the stage-4 killed terms, folded the same way) are
the F12 special term restored below.  What still separates the
substituted result from the printed (S3)/(S4) is the second-order
F12-intermediate content (X.f, B, and the S-intermediate delta.f pieces)
produced by the SeQuant screening steps -- a reorganization of the
second-order terms using the F12 geminal/orbital structure, not a plain
BCH contribution.

## The implemented formula (2026-10-13)

The resurrected geminal commutators.  `f12_special_term.py` takes the stage-4 killed terms of
every block (`space_restriction.build()`, field `zero`), folds the
three-body ones back with the grouped cumulant template of stage 6
(`cumulant_expansion.decompose`), adds them to the stage-6 final blocks,
and assembles the result the way the commutator stage defines (blocks +
h.c., 1/2 on the Fock double commutator, merged once --
`space_restriction.full_expression`).

Counts: blocks 2, 4, 6, 7, 8, 10, 11, 12 carry 1, 2, 1, 1, 1, 2, 2, 6
killed terms; after the cumulant folding the special blocks have 1, 4, 1,
1, 0, 2, 2, 8 terms (the all-virtual-upper E^{abc}_{ijk} of block 8 folds
to zero: no template group carries all three virtual indices).  The
assembled special term has 32 terms (24 doubles-only); the substituted
final formula (stage 6 + special) has 166 terms (126 doubles-only).
Output: `f12_special_term.tex`/`.pdf`.

Checks (`../test/check_f12_special_term.py`, run by the stage script):

* complement: killed + survivors = the stage-3 block, per block;
* the cumulant folding of the killed three-body patterns (the template
  identity with the virtual rule, spin-summed Jordan--Wigner matrices, SD
  reference; max deviation 0.0);
* the assembled special expression vanishes exactly in the F12 setting
  (free virtuals OBS-vir, geminal with zero both-OBS-vir block):
  substituting it back does not touch the restricted result, eq. (S1)
  compatibility.

## The (S3)/(S4) comparison (`../test/check_reference_s34.py`)

The substituted final formula (doubles part + bare H) is compared with
the dressed one-/two-body matrix elements hbar/gbar of eqs. (S3)/(S4),
parsed from the SI tex source (`../reference/supporting.tex`, parser
`../test/si_parse.py`, the V, V~, X, B intermediates expanded by their
definitions), as spin-summed Jordan--Wigner matrices on 4 spatial
orbitals {2 occ, 1 OBS-vir, 1 CABS} with the SD-reference RDMs, split by
geminal order:

* bare H and ALL first-order (h.G, g.G) content: EXACT match
  (max deviation ~1e-14), assumption-free;
* the second-order f.G.G content does NOT match the printed (S3)/(S4):
  max|Delta| ~ 42 with a random Fock and a plain SP-restricted geminal,
  ~3.1 with the geminal in true SP (cusp-condition) form, ~0.85 adding a
  canonical (diagonal) Fock matrix (the absolute scale depends on the
  random tensors; the first-order match is exact in every setup).  The residual is the F12-intermediate
  machinery (X.f, B, S-intermediate delta.f pieces with f^a_a / f^i_i
  traces) produced by the SeQuant driver's F12 screening steps
  (`overlap_with_obs`, `screen_F12_proj`, `hamiltonian_based_projector_2`,
  `fock_based_projector_2` in examples/uccf12/uccf12.cpp of
  ValeevGroup/SeQuant), which the plain truncation of (S1) cannot
  produce.  See the comparison record above for the details.

## How this stage was built

Same machinery as stages 3-6: the killed terms come from the stage-4
build, the three-body pieces are folded with the grouped cumulant
template (stage 6), and the assembly is the stage-4 one; with
Jordan--Wigner matrix checks at each step; the result is written to
f12_special_term.tex/.pdf.
