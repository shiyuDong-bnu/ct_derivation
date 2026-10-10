# generalized_wick_commutator.py

A sympy-based script that expands commutators of number-conserving
excitation operators

    E^{u1..un}_{l1..ln} = a^+_{u1} ... a^+_{un} a_{ln} ... a_{l1}

with the generalized Wick theorem (Kutzelnigg and Mukherjee, J. Chem.
Phys. 107, 432 (1997), eqs. (9a)/(9b)), in the paired-order form proved
in `single_amplitude_formula/generalized_wick.pdf`.

There is no sign rule: every coefficient of the expansion is +1 once the
residual is written in the paired order (the parity lemma of the note) --
only the pairing of upper to lower indices has to be kept.  The module
therefore contains no permutation-sign computation at all:

* `product_paired(op1, op2)` -- the sum over joins, eq. (4)/(12) of the
  note, all coefficients +1, residual in paired order.
* `commutator(op1, op2)` -- AB and BA both expanded by the theorem and
  canonicalised; the two no-contraction terms are the same operator with
  its pairs permuted, so they cancel automatically.  Returns
  `{(deltas, upper, lower): coeff}`.
* `Operator.canonical()` / `collect(terms)` -- the canonical order: sort
  the upper indices and apply the same permutation to the lower ones, so
  every (upper, lower) pair is kept together; a simultaneous permutation
  of both rows carries no sign (the (8)->(9) transition of the note), so
  no sign is needed here either.  Written forms related by a re-pairing
  (the section 3.4/8 caveat) are not identified; compare those
  numerically.

Run

    python3 generalized_wick_commutator.py

for the self test.  It reproduces every printed equation of the note --
the product rules (15), (16), (19), (20) and the commutators (14), (17),
(18), (21) -- term by term in the written paired order, checks the term
count of eq. (6) and antisymmetry, and verifies all commutators and the
four product rules of `single_amplitude_formula/gen_commutators.py`
numerically against exact Jordan-Wigner matrices.
