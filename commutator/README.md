# The commutators of canonical transcorrelation

Everything needed to build the commutators of the CT expansion
$\bar H=e^{-\hat A}\hat H e^{\hat A}$, in one place: the derivation notes, the
generator that produces the blocks, and the checks that keep them honest.

This directory is about the **commutators themselves**. What is *not* here is
the step that turns a commutator into a working equation — the expansion
through the generalized Wick theorem, the cumulant decomposition, and the
fold of the three-body pieces back into one- and two-body operators
($[\dots]_{1,2}$). That is done in `../single_amplitude_formula/`
(generalized Wick) and `../ctformula/` (the algebra engine).

## Contents

### Derivation notes, in the order they were built

| file | pages | what it establishes |
|---|---|---|
| [`ct_structure.tex`](ct_structure.tex) | 4 | the ansatz, the second-order truncation, the decomposition into blocks, and the block inventory |
| [`ct_F_A.tex`](ct_F_A.tex) | 5 | $[\hat F,\hat A]$: both blocks, the coefficients $C^q_p$ and $C^{rs}_{pq}$, the result |
| [`ct_V_A.tex`](ct_V_A.tex) | 3 | $[\hat V,\hat A]$: the $(2,1)$ and $(2,2)$ blocks and their contracted forms |
| [`ct_f_a_a.tex`](ct_f_a_a.tex) | 3 | $\tfrac12[[\hat F,\hat A],\hat A]$: the eight blocks, none redundant |
| [`ct_commutators.tex`](ct_commutators.tex) | 5 | the gathering document: all three commutators, the conventions, the check table, and the three scans |

Each is standalone and built with `pdflatex <file>.tex` from this directory.

### The generator

* [`generate_commutators.py`](generate_commutators.py) — builds every
  commutator the CT theory needs, symbolically with sympy's `Commutator`, and
  writes [`commutators.tex`](commutators.tex) / `commutators.pdf`.
  First order: $[\hat F,\hat T_r]$, $[\hat V,\hat T_r]$. Second order:
  $[[\hat F,\hat T_r],\hat T_s]$ and $[[\hat F,\hat T_r]^\dagger,\hat T_s]$,
  and the same pair for $\hat V$ — eight blocks, plus the conjugate halves.
  It keeps them in **nested-operator form**; it does not expand them into
  many-body equations.

  **Convention warning.** The docstring of the script writes
  $\hat F=f^q_p\E{q}{p}$ while the header of `commutators.tex` calls the same
  line the *crossed* Fock order. Those are two different conventions — crossed
  is $\hat F=f^p_q\E{q}{p}$, the one used by all the notes above — and they
  differ by transposing $f$. The two files must be brought into agreement
  before the generator's output is used with the notes; picking the crossed
  order is the consistent choice, since $\hat F$ is Hermitian there for
  symmetric $f$ and the notes are written that way.

### Conventions the notes are written in

Crossed Fock order, as on the handwritten sheets and in `../formula.tex`;
excitation operators; general indices ranging over the whole space
$\{\occ\}\cup\{\vir\}$:

$$\hat F=f^p_q\E{q}{p},\qquad
  \hat V=\tfrac12 g^{pq}_{rs}\E{rs}{pq},\qquad
  \hat T_1=t^\vir_\occ\E{\vir}{\occ},\qquad
  \hat T_2=\tfrac12 t^{\vir\beta}_{\occ j}\E{\vir\beta}{\occ j},\qquad
  \hat A=\hat T-\hat T^\dagger .$$

Two facts the checks depend on, and which are easy to lose:

* **$\hat F$ is Hermitian only for symmetric (real) $f$.** The step
  $[\hat F,\hat A]=[\hat F,\hat T]+[\hat F,\hat T]^\dagger$ uses that
  Hermiticity. A non-symmetric $f$ makes the identity fail, and every
  canonical-Fock sanity check still passes.
* **A general index is occupied or virtual and nothing else.** So a delta
  between an occupied and a virtual label is unsatisfiable, and terms
  disappear for that reason — which is *not* the same as cancelling. The block
  $[[\hat F,\hat T_1],\hat T_1]$ loses two of its four delta terms that way.

### What is checked

The checks live in `../tests/` and each is a matrix sweep, not a re-reading:

| quantity | script | relative |
|---|---|---|
| $[\hat F,\hat T_1]$, $[\hat F,\hat T_2]$ (four terms, uncancelled) | `test_F_A_commutator.py` | $2.6\times10^{-16}$, $1.5\times10^{-16}$ |
| $[\hat F,\hat A]$ and the closed form of $C^q_p$ | `test_F_A_result.py` | $6.5\times10^{-16}$, $2.5\times10^{-16}$ |
| $[\hat V,\hat T_1]$ (four terms), $[\hat V,\hat T_2]$ (twelve terms) | `test_V_A_commutator.py` | $6.1\times10^{-16}$, $8.9\times10^{-16}$ |
| the eight second-order blocks, and $[[\hat F,\hat T_1],\hat T_1]$ | `test_ct_structure.py` | $10^{-13}$ |

Two of them also assert that a *wrong* form fails, so the check cannot pass by
accident: each Fock block under the other convention's $\hat F$, and the two
mixed nested commutators against each other.

Broadly, `../tests/test_ct.py` and `../tests/test_identities.py` run the
algebra engine over the CT fragments; three blocks fail there
($[\hat V,\hat T_2]$, $\tfrac12[[\hat F,\hat A_2],\hat T_1]$,
$\tfrac12[[\hat F,\hat A_2],\hat T_2]$) because the engine's canonicalisation
sorts the two index rows of an $\hat E$ independently, which changes the spin
pairing of a spin-free operator. The notes above are checked against matrices
directly and are not affected.

### The scans

The sheets the notes were read from are in
[`../single_amplitude_formula/pictures/`](../single_amplitude_formula/pictures/):
`ct_F_A_scan.png`, `ct_V_A_scan.png`, `ct_f_a_a_scan.png`, with the findings
recorded in that directory's `README.md`. `ct_commutators.tex` embeds all three.

## Traps that have bitten this derivation

Recorded because each cost a wrong turn, and each was caught by a matrix check
rather than by re-reading the algebra.

1. **The transposed Fock tensor.** $f^p_a$ against $f^a_p$: invisible for
   symmetric $f$, and therefore invisible in every canonical-Fock check.
2. **Folding a pair that cannot be folded term by term.** The two terms of a
   pair are related by a relabelling of the *sum*, not term by term, and it is
   term by term that the coefficient is read off.
3. **The short term list.** The $(2,2)$ rule has twelve terms, not eight; the
   four double-delta ones are 41% of the operator. Likewise
   $[[\hat F,\hat T_1],\hat T_1]\neq0$, against the intuition that operators
   of the same shape commute.
4. **Hermiticity assumed rather than checked.** See the first convention point
   above.

## Open

* **$C^{rs}_{pq}$ has no compact closed form written down.** The coefficient is
  verified as a collection of index patterns (`test_F_A_result.py` reconstructs
  $[\hat F,\hat A]$ from it), but the explicit formula is still owed.
* **The three-body reduction.** The $(2,2)$-derived blocks produce three-body
  operators; folding them back with the extended normal ordering of
  Kutzelnigg and Mukherjee is done by hand on the operator side only.
* **The generator's convention** must be reconciled with these notes (see
  above).
* **The multireference case.** Everything here is vacuum normal ordering; a
  general reference needs the density and cumulant terms.
