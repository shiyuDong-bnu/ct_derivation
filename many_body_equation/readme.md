# many_body_equation — expansion and canonicalization of the commutator blocks

## Commutator expansion

In this section we expand all commutators.  The commutator blocks are
those of the commutator stage (`../commutator/generate_commutators.py`,
`commutator_blocks()`): the 4 first-order blocks [h, T_r], [V, T_r] and
the 8 second-order Fock blocks [[F, T_r], T_s], [[F, T_r]^dagger, T_s]
(r, s = 1, 2).  Each block is expanded with the generalized Wick theorem
(`../generalized_wick_theorem/generalized_wick_commutator.py`) into
many-body terms (coefficient x tensors x excitation operator E); the
excitations of a nested commutator carry fresh dummy letters (a1,a2 /
i1,i2 first, then a3,a4 / i3,i4).

## Exact symmetries of the operator and the tensors

The only identities used by the simplification steps below (all exact in
the spin-integrated, real-coefficient convention; no index antisymmetry
anywhere):

    operator:  the spin-summed E is invariant under the DIAGONAL
               (simultaneous) permutation of its index pairs, with NO
               sign, E^{qp..}_{rs..} = E^{pq..}_{sr..};  a single row
               alone is NOT a symmetry of the spin-summed operator
    tensors:   t^{ab}_{ij} = t^{ba}_{ji}  and
               g^{pq}_{rs} = g^{qp}_{sr}  (simultaneous swap of both
               index pairs);  real coefficients: h^q_p = h^p_q,
               f^q_p = f^p_q (upper/lower swap), and g symmetric under
               pair exchange, g^{pq}_{rs} = g^{rs}_{pq}

## Simplification steps

Applied to every expanded term, in order:  first the deltas are resolved,
then the operator is put in canonical form, then the tensor product.

### Resolve the deltas

1. Resolve the deltas according to the current delta rules.  Indices
   are numbered per space: p1,p2,... whole space; i1,i2,... occupied;
   a1,a2,... virtual (bare letters p,q,r,s etc. denote the same spaces;
   the space of a symbol is read off its base letter).  The occupied and
   virtual spaces are disjoint subspaces of the whole space, so a
   Kronecker delta resolves as follows:

       delta(p, a)  (whole, vir):  the sum over the whole-space index p
            collapses onto the virtual value a -- p is replaced by a;
       delta(p, i)  (whole, occ):  likewise, p is replaced by i;
       delta(i, a)  (occ, vir):  no orbital is both occupied and virtual,
            so the delta is never satisfied -- the term vanishes;
       delta(x, y)  with x, y in the same space:  substitute one index
            by the other (every occurrence of y is replaced by x, in the
            tensors and in the operator) and drop the delta -- the two
            same-space dummies contract (the einsum convention);
       delta(whole, whole) never occurs in this truncation.

### Operator canonicalization

2. Reorder the operator according to the symmetry of the operator: the
   index pairs of E are sorted, each (upper, lower) pair kept together,
   based on space and lexical order (the space order of step 4, lexical
   within a space).

### Tensor canonicalization

3. Reorder the tensor product by the tensor head name (the tensors are
   scalars and commute: the product is a multiset, sorted by name).
   Ties (same head name) are broken by the canonical index arrangement
   of step 4, so the order does not depend on the written arrangement
   of the tensors.

4. Reorder the index of each tensor according to its symmetry (the Exact
   symmetries section above), taking the minimal arrangement under the
   order:

       space:  whole (total) > occupied > virtual
       then:   outer index > dummy index
       then:   lexical order

   (an outer index is one that also sits on the operator; a dummy index
   is one that is summed out inside the tensor product).

### Merging

Steps 2-4 alone are NOT letter-invariant:  steps 2 and 4 sort by the
index order of the term's current letters, which presupposes canonical
letters.  All renaming is therefore done at once:  the canonical
representative of a term is the MINIMUM of the steps-2-4 form over the
renaming orbit (`orbit_canonical`):  the term's letters are first
brought onto the numbered pool prefix (a1,a2,... virtual; i1,i2,...
occupied; p1,p2,... whole space), then every within-space permutation
of the letters is tried and the minimum under the index order (then the
tensor signatures) wins.  No separate substitution step is needed for
the outer or the dummy indices:  the orbit already enumerates every
within-space lettering, and a re-lettered candidate is just another
orbit member.  The orbit minimum is letter-invariant by construction:
two terms that differ by a renaming have the same renaming orbit, hence
the same minimum.

Two terms are equal -- and merge, their coefficients summed (`merge`,
applied once per block in `expanded_blocks`) -- iff their canonical
representatives agree.  All moves carry sign +1.

## Public API for later stages (`expand_commutators.py`)

* `expanded_blocks()` -- the 12 expanded blocks as a list of dicts:
  `kind` ("first"/"second"), `expr` (the commutator-stage sympy
  expression), `lhs` (LaTeX), `x_name`, `r`, `s`, `is_dag`, `terms`
  (merged `(coeff, tensors, op, leftover)` list), `ranks`.
* `build_blocks()` / `BLOCKS` -- the legacy tuple view
  (kind, x_name, r, s, is_dag, lhs, terms).
* the term machinery: `Tensor`, `Operator`, `SPACE`, `FAMILY`,
  `merge`, `canonical_form`, `orbit_canonical`, `term_latex`,
  `terms_latex`, `pack_rows`, `ranks_of`.

Used by `space_restriction/space_restriction.py` and
`cumulant_expansion/cumulant_expansion.py`.

The numerical checks live in `../test/check_many_body_equation.py` and
are run by the stage script.

we will do cumulant expansion of three body operator later.
