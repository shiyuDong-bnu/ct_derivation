# f12_special_term -- the final piece (PENDING DEFINITION)

The CT-F12 correlator is

    A = T1 + T2(OBS, optimized) + T_F12(fixed geminal, CABS)

where the F12 geminal amplitudes are fixed by the cusp condition (SP
ansatz) and live in the CABS virtual space:

    t~^{ab}_{ij} = (3/8) <ab|Q12 f12|ij> + (1/8) <ab|Q12 f12|ji>

The stage-4 space restriction killed exactly the terms whose t2 has both
OBS virtuals on the E operator; the F12 geminal is what compensates that
restriction, and its contribution to the transcorrelated Hamiltonian is
the "F12 special term" to be derived here.

## Open question: which formula

Candidates (to be decided before implementation):

1. The resurrected geminal commutators: the stage-4 killed terms
   ([F, T], [V, T], (1/2)[[F, T], T] with both virtuals on the operator)
   re-enter with the fixed geminal coefficients over CABS virtuals.
2. The Yanai--Shiozaki extra intermediates: explicit F12 contributions
   built from the V, X, B intermediates (<f12 Q12 f12>, <f12^2>,
   <f12 T f12>), following the corrected expressions of the Kumar--Valeev
   comment (J. Chem. Phys. 158, 057101 (2023)) -- the original eq. (27)
   has a spurious 1/2.
3. The deexcitation (anti-Hermitian) part of A^F12.

## How this stage will be built

Same machinery as stages 3-6: expand with the generalized Wick theorem
(stage 1), apply the index-space and delta rules (stage 3), the space
rules (stage 4), fold three-body pieces back with the grouped cumulant
template (stage 6), with Jordan--Wigner matrix checks at each step, and
write the result to f12_special_term.tex/.pdf.
