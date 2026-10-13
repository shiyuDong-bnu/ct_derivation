After we get those many body equation there can be furthur simplification
one is the restriction of space.
here we should use some new information from CF-F12 of yanai.
here the virtual space is split into two space one is OBS-vir and another is CABS+\infty vir
we will now restrict our virtual space into OBS-vir
so for every operator we index is OBS-vir now
and the amplitude $t_{ij}^{ab}$ should be zero if all two space  is OBS-vir
if in the many body equation the virtual index is contracted with other term that is ok
if the two virtual index which appeares in operator then this should be zero.

here is what you should do,first of all copy original term
then find all those terms that should be zero , mark them by underline
and finally gather all nonzero terms

we will do cumulant expansion later.

## The full expression (blocks + h.c.) and eq. (S1)

After the restriction the stage also assembles the surviving blocks into
the full transcorrelated Hamiltonian through O(lambda^2), the way the
commutator stage defines it:

    H_bar = H + [H_b, A] + (1/2)[[F, A], A],   A = T - T^dagger
          = H + (first-order blocks) + h.c.
            + (1/2) (the 8 second-order Fock blocks) + h.c.

`full_expression(blocks, doubles_only=False, field="keep")` does the
assembly: every block is added together with its Hermitian conjugate
(`mbe.dagger_terms`), the second-order Fock blocks carry the 1/2 of the
double commutator, and the assembled sum is merged once (`mbe.merge`) --
Hermitian-conjugate pairs with equal canonical form fold into one term,
self-conjugate terms keeping coefficient 1 through the 1/2.
`field="terms"` assembles the unrestricted stage-3 expansion instead of
the survivors.

With `doubles_only=True` only the pure-T2 blocks are kept (the F12
geminal operator A^F12 = T2 - T2^dagger); the assembled restricted
expression is then **eq. (S1) of the Supplemental Material of Masteran,
Kumar, Teke, Gaudel, Yanai and Valeev, J. Chem. Phys. 158, 057101
(2023)** (`../reference/supporting.pdf`), the complete transcorrelated
Hamiltonian without the cumulant decomposition -- 16 terms.  The
correspondence needs the F12 conventions: indices on the E operator run
over the OBS spaces, contracted virtuals over the complete virtual
(OBS-vir + CABS), and the geminal amplitudes are pair-swap AND
pair-exchange symmetric (G^{alpha beta}_{ij} = G^{ij}_{alpha beta}, which
is what lets (S1) write G with either index orientation) and vanish with
both indices OBS-vir.

Check: `../test/check_reference_s1.py` (run by this script, or
standalone) verifies

1. check_s1 -- the assembled doubles expression == the printed 16 terms
   of (S1) exactly (max deviation ~1e-13 on random tensors with
   genuinely spin-summed Jordan--Wigner operators, 2 occupied + 1
   OBS-vir + 1 CABS spatial orbitals), the assembled expression has
   exactly 16 terms, and the killed terms of the restriction rule vanish
   exactly in this setting;
2. check_full_expression -- the same assembly over the unrestricted
   stage-3 blocks == the direct nested matrix commutator
   [H_b, A] + (1/2)[[F, A], A] with A = T - T^dagger (full virtual
   ranges), i.e. the "+ h.c., 1/2 on the double commutator" assembly is
   exactly the anti-Hermitian generator of the commutator stage.
