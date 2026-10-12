In this section we need to expand all commutator out using
generalized wick theorems.

the all comutator is in comutator directory
the generalized theorem is in generalized wick theorem

The block inventory is taken from the commutator stage API
(`commutator/generate_commutators.py`, `commutator_blocks()`): the 4
first-order blocks [h, T_r], [V, T_r] and the 8 second-order Fock blocks
[[F, T_r], T_s], [[F, T_r]^dagger, T_s] (r, s = 1, 2).  Each nested
sympy expression is evaluated with the generalized Wick theorem
(`expand_expr`); the excitations of a nested commutator carry fresh
dummy letters (a1,a2 / i1,i2 first, then a3,a4 / i3,i4).

Public API for later stages (`expand_commutators.py`):

* `expanded_blocks()` -- the 12 expanded blocks as a list of dicts:
  `kind` ("first"/"second"), `expr` (the commutator-stage sympy
  expression), `lhs` (LaTeX), `x_name`, `r`, `s`, `is_dag`, `terms`
  (merged `(coeff, tensors, op, leftover)` list), `ranks`.
* `build_blocks()` / `BLOCKS` -- the legacy tuple view
  (kind, x_name, r, s, is_dag, lhs, terms).
* the term machinery: `Tensor`, `Operator`, `SPACE`, `FAMILY`,
  `merge`, `dummy_index`, `term_latex`, `terms_latex`, `pack_rows`,
  `ranks_of`.

Used by `space_restriction/space_restriction.py` and
`cumulant_expansion/cumulant_expansion.py`.

simplify rules, as implemented in `expand_commutators.py`:

1. resolve_delta -- the index-space delta rules.  Indices are numbered
   per space: p1,p2,... whole space; i1,i2,... occ; a1,a2,... vir
   (bare letters p,q,r,s etc. denote the same spaces; the space of a
   symbol is read off its base letter):
   delta(whole, vir) -> the whole-space index becomes the virtual one;
   delta(whole, occ) -> the whole-space index becomes the occupied one;
   delta(occ, vir)   -> 0 (the term vanishes);
   same-space dummies contract (the einsum convention);
   delta(whole, whole) raises (unreachable with these three spaces).

2. canonicalization -- the operator is invariant under a simultaneous
   (diagonal) permutation of its index pairs,

       E^{p_1 p_2 ... p_n}_{q_1 q_2 ... q_n}
         = E^{p_{s(1)} ... p_{s(n)}}_{q_{s(1)} ... q_{s(n)}}
         for any permutation s, with NO sign;

   the canonical written order sorts the pairs by the upper index under
   the space-based index order (general > occ > vir, lexical within a
   space; `index_key`, applied by `op_canonical`).  The
   coefficient tensors are likewise canonicalized: each
   tensor's indices are reordered under its exact symmetries (h, f
   symmetric; t, g simultaneous pair swap; g also pair exchange), the
   canonical arrangement being the minimum under `index_key`
   (`Tensor.canonical`), and the tensors are then sorted as a multiset
   (`tensors_canonical`).

3. dummy indices -- the summation indices follow the einsum convention
   and may be renamed by an arbitrary unused letter of the same space;
   the free indices of the operator are equally arbitrary, because the
   operator basis sum makes them summation labels.

4. merging (`merge`, via `dummy_index`) -- two terms are equal, and
   merge, iff they are related by identities that are exact in the
   spin-integrated, real-coefficient convention:
   a. renaming of any summed index within its space (rule 3);
   b. the pair symmetry of the amplitudes, t^{ab}_{ij} = t^{ba}_{ji} and
      g^{pq}_{rs} = g^{qp}_{sr} (simultaneous swap of both index pairs)
      -- no index antisymmetry of t or g is used anywhere;
   c. real coefficients: h and f symmetric (h^q_p = h^p_q,
      f^q_p = f^p_q) and g symmetric under pair exchange
      (g^{pq}_{rs} = g^{rs}_{pq}); the density matrices D (appearing in
      the cumulant stage) carry the same symmetries;
   d. the diagonal pair-permutation symmetry of the spin-summed E
      operator: the two index rows may only be permuted TOGETHER, with
      NO sign (E^{qp..}_{rs..} = E^{pq..}_{sr..});  a single row alone
      is NOT a symmetry of the spin-summed operator (single-row
      antisymmetry belongs to the spin-orbital string, not to the
      spin-integrated E);
   e. merge all equal operators (the representative is the minimum, under
      `index_key`, over the renaming orbit of the canonical written form
      given by b--d; all orbit moves carry sign +1).

we will do cumulant expansion of three body operator later.