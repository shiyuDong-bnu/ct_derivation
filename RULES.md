# Rules and reasons at each stage

Every rule the derivation uses, where it is used, and why it is valid.
Stages 1, 2, 5 are prerequisites (tools and input formulas); stages 3, 4,
6 are the derivation process, applied in order.

## Stage 1 -- generalized_wick_theorem (tool)

- Operators are number-conserving excitation operators
  E^{u1..un}_{l1..ln} = a+_{u1} ... a+_{un} a_{ln} ... a_{l1}, written in
  paired order.
- Commutators are expanded with the generalized Wick theorem
  (Kutzelnigg--Mukherjee, J. Chem. Phys. 107, 432 (1997), eqs. (9a)/(9b)).
  In paired order there is NO sign rule: every coefficient is +1 and only
  the pairing of upper to lower indices (the delta contractions) has to be
  tracked.
- Canonical order of a written operator: sort the upper indices and apply
  the same permutation to the lower ones (diagonal pair permutation
  carries no sign).
- Verified: reproduces every printed rule of the reference note term by
  term; all commutator rules checked exhaustively on Jordan--Wigner
  matrices.

## Stage 2 -- commutator (input: which blocks are needed and why)

- Baker--Campbell--Hausdorff: e^{-A} H e^{A} = H + [H, A] + (1/2)[[H, A],
  A] + ..., with H = F + V and A = T1 + T2.
- Perturbation-order truncation (Moller--Plesset counting F ~ O(1),
  V ~ O(lambda), T ~ O(lambda)): through O(lambda^2) keep
  [F, A] + [V, A] + (1/2)[[F, A], A].  REASON to drop (1/2)[[V, A], A]:
  it is O(lambda^3) -- deleted, not just flagged.
- Block split: A = T1 + T2 gives 4 first-order blocks [F,T1], [F,T2],
  [V,T1], [V,T2] and 8 second-order Fock blocks [[F,Tr],Ts],
  [[F,Tr]+,Ts] (r,s = 1,2); the assembled result gets + h.c.
- Each block is matrix-checked.

## Stage 3 -- many_body_equation (expansion)

Substitution rules (operator definitions):
- F = f^q_p E^p_q,  V = (1/2) g^{pq}_{rs} E^{rs}_{pq},
  T1 = t^a_i E^a_i,  T2 = (1/2) t^{ab}_{ij} E^{ab}_{ij};
  the second excitation of a nested commutator carries dummy indices
  c,d / k,l.

Index-space and delta rules:
- p,q,r,s whole space;  i,j,k,l occupied;  a,b,c,d virtual.
- delta(whole, vir) -> the whole-space index becomes the virtual one;
  delta(whole, occ) -> occupied;  delta(occ, vir) = 0;
  same-space dummies contract;  delta(whole, whole) is kept.
- Pauli rule: a repeated index within one row of E makes the term vanish.

Conventions in force:
- Spin-integrated formalism: NO index antisymmetry of t or g anywhere;
  the only symmetries are the simultaneous pair swaps
  t^{ab}_{ij} = t^{ba}_{ji} and g^{pq}_{rs} = g^{qp}_{sr}.
- Real coefficients: stars dropped; f symmetric (f^q_p = f^p_q); g
  symmetric under pair exchange (g^{pq}_{rs} = g^{rs}_{pq}).
- The exact fermionic row antisymmetry of E IS used (permuting one row
  gives the parity sign).

Merging rule (orbit minimum): two terms are equal -- and merge -- iff
related by (a) renaming of any summed index within its space (the free
indices of the operator are summation labels of the operator basis sum,
so they rename like tensor dummies), (b) the pair swaps above, (c) the
real-coefficient symmetries above, (d) E-row antisymmetry with parity
signs.  REASON this is legitimate: every identity used is exact in this
convention; verified by the coefficient-tensor route cross-check (raw vs
merged expansion agree after row antisymmetrization).

Conjugate blocks: [X_r]+ is expanded by conjugating term by term --
E^{u..}_{l..} -> E^{l..}_{u..}, f and g Hermitian-swapped, t real
unchanged.

Everything up to here is an OPERATOR IDENTITY: no reference-state
information enters.

## Stage 4 -- space_restriction (kill rule)

- From CF-F12 (Yanai): the virtual space is split into OBS-virtual and
  CABS; the amplitudes live outside the OBS virtual space, so a t2
  amplitude t^{ab}_{ij} whose two virtual indices BOTH appear on the E
  operator is zero.
- KILL RULE (provenance form): a term is zero iff both virtual indices of
  ONE AND THE SAME t2 amplitude sit on the E operator.  A virtual index
  contracted with another tensor is fine; the two virtuals must come from
  the same amplitude.
- Applied term by term: killed terms are underlined in the pdf, survivors
  gathered.  Still an operator identity -- no reference state yet.
- Consequences: block [[F,T2],T2] = 0 entirely; in [[F,T2]+,T2] the four
  two-upper-virtual three-body terms (E^{abi}_{cjp} twice, E^{abi}_{cdj},
  E^{abp}_{cdi}) and the two-body terms E^{ab}_{cp}, E^{ab}_{cd} die.

## Stage 5 -- cumulant_decomposition/spin_free (input formula)

- The spin-free cumulant decomposition of the three-body operator:
  E^{pqr}_{stu} = [51 terms: 1-RDM D times E2, D2 times E1, D.D times E1,
  D times D2, D.D.D] + (normal-ordered three-body remainder) +
  lambda^{pqr}_{stu} (three-cumulant).
- REASON the last two pieces are dropped: for a single-determinant
  reference lambda_3 = 0 identically, and the [..]_{1,2} truncation drops
  the normal-ordered three-body remainder (for a multireference state
  dropping lambda_3 is an approximation).
- This is an OPERATOR IDENTITY with fixed external labels: no index may
  be renamed, only operator rows may be reordered (parity sign into the
  coefficient).  It is verified numerically from scratch in stage 6, so
  the notebook that produced it is not trusted blindly.

## Stage 6 -- cumulant_expansion (decomposition and final form)

- The reference state enters ONLY here, through the density matrices.
- VIRTUAL RULE: a density matrix carrying a virtual index is zero.
  Occupied/whole-space DMs are kept symbolic as D^i_j, D^{ij}_{kl}, ...
  (no delta-function substitution).
- Grouped template: the 51 terms collect into 19 operator groups.  Every
  template index appears in exactly one factor of each piece, so a whole
  group survives or dies together: a group survives IFF its operator
  carries EVERY virtual index of the pattern.  The pure-density-matrix
  scalar group always vanishes.
- Resulting counts: one-virtual patterns (E^{apq}_{ijr} of [V,T2],
  E^{aij}_{klp} of [[F,T2]+,T2]) keep 6 two-body + 3 one-body groups;
  two-virtual patterns (E^{aij}_{bkl}, E^{aip}_{bjk}) keep 4 two-body +
  1 one-body group; the E^{ab...} patterns never reach this stage (stage
  4 killed them).
- The decomposed pieces are merged back into the block with the stage-3
  orbit-min merging (summed dummies may be renamed and canonicalized
  here, unlike in the template identity).
- Final presentation: coefficient pieces collected per operator (grouped
  blocks), ready for substitution; grouping is verified presentational
  only (flatten = block).
- Checks: template identity exact (SD reference, random assignments, with
  and without the virtual rule); template grouping flatten check;
  group-vs-flat decomposition equivalence per three-body term; block
  grouping check.

## Stage 7 -- f12_special_term (PENDING DEFINITION)

- The final piece: the contribution of the fixed F12 geminal amplitudes
  (CABS space, SP/cusp coefficients t~^{ab}_{ij} = (3/8)<ab|Q12 f12|ij> +
  (1/8)<ab|Q12 f12|ji>) that compensates the stage-4 space restriction.
- The exact formula is not yet decided; see f12_special_term/readme.md
  for the candidate forms (resurrected geminal commutators /
  Yanai--Shiozaki V,X,B intermediates / deexcitation part of A^F12).
