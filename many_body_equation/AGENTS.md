# AGENTS.md — many_body_equation

## Project overview

This directory is **stage 3 (many-body expansion) of the `ct_derivation`
pipeline** (repo root: `../`, see `../README.md` and `../RULES.md`). The
pipeline derives, step by step and with a numerical check at every step,
the transcorrelated Hamiltonian

    H_bar = e^{-A} H e^{A},   H = F + V,   A = T1 + T2,

through O(lambda^2) in Moller--Plesset counting (h ~ O(1), V, T ~ O(lambda)):
the full first-order term [H_b, A] with H_b = h + V, and the Fock half of
the second order, (1/2)[[F, A], A]. The valence double commutator
(1/2)[[V, A], A] is O(lambda^3) and is NOT expanded.

This stage takes the 12 commutator blocks built by the previous stage —
the 4 first-order blocks [h, T_r], [V, T_r] and the 8 second-order Fock
blocks [[F, T_r], T_s], [[F, T_r]^dagger, T_s] (r, s = 1, 2) — and expands
each one with the generalized Wick theorem into many-body terms
(coefficient x tensors x excitation operator E), applying the index-space
delta rules and merging equal terms over every exact identity. The result
is written to `many_body_equations.tex`/`.pdf` and exposed
programmatically to the later stages.

The physics conventions (spin-integrated formalism, real coefficients,
exact symmetries used for merging) are documented in `readme.md` — read it
before touching the term machinery. Everything in this stage is an
operator identity; no reference-state information enters here.

## Repository layout

- `expand_commutators.py` — the entire stage: expansion engine, term
  machinery, LaTeX writer, and numerical checks, in one file (~1000
  lines). Running it executes the checks and regenerates the tex/pdf.
- `readme.md` — the authoritative statement of the simplification steps
  (1. resolve_delta, 2. op_canonical, 3. substitute_outer,
  4. sort_tensors, 5. Tensor.canonical, 6. substitute_dummies, then
  merging via the renaming-orbit minimum) and the public API. Keep it in
  sync with the code.
- `many_body_equations.tex` / `.pdf` — GENERATED output (by `write_tex()`);
  never edit by hand. `.aux`, `.fdb_latexmk`, `.fls`, `.log` are latexmk
  build artifacts.

Cross-stage imports (via `sys.path` insertion of the sibling directories):
- `../generalized_wick_theorem/generalized_wick_commutator.py` — the Wick
  engine: `Operator`, `commutator`, and the Jordan-Wigner helpers
  `_jw_matrices`, `_matmul`.
- `../commutator/generate_commutators.py` (imported as `gen_commutator`) — the block
  inventory: `commutator_blocks()`, the symbols `H, F, V, T1, T2`,
  `is_dagger()`, `ct_latex()`.
- `../utils/latex_utils.py` — the shared LaTeX helpers: `compile_tex`
  (the pdf build of every stage) and `idx_latex`.

The numerical checks of every stage live in `../test/check_<stage>.py`
(this stage: `../test/check_many_body_equation.py`); each stage script
imports and runs them from its `__main__`, so the workflow below is
unchanged. The `sys.modules.setdefault(...)` line in each `__main__`
registers the running script under its module name so the check module
imports the same instance — the term machinery relies on `Tensor`/
`Operator` class identity. The check modules are also runnable
standalone.

Downstream consumers of this stage's API: `../space_restriction/` and
`../cumulant_expansion/`.

## Technology stack and dependencies

- Plain Python 3 scripts; **no package manifest** (no pyproject.toml,
  setup.py, requirements.txt) and no test framework.
- Runtime deps: `sympy` (symbols, Rational, Commutator) and `numpy`
  (numerical checks only). Verified working with sympy 1.13 / numpy 2.5.
- `latexmk` (TeX Live) for the pdf build; if it is absent the script
  skips the pdf but still runs all checks and writes the tex.

## Build, run and test commands

There is no separate test suite — **running the stage IS the test**:

    python3 expand_commutators.py

This (takes ~25 s):
1. expands all 12 blocks and prints their term counts and operator ranks;
2. `check_blocks()` — compares every expanded block against the direct
   nested matrix commutator with random real tensors on Jordan-Wigner
   spin-orbital matrices (4 modes, 2 occupied + 2 virtual);
3. `check_blocks_spin_summed()` — the same comparison with genuinely
   spin-summed operators (4 spatial orbitals = 8 spin orbitals); this is
   the check that sees the diagonal-pair-permutation merging rule, which
   the spin-orbital check cannot distinguish;
4. regenerates `many_body_equations.tex` and builds the pdf via latexmk.

A failure exits nonzero (`SystemExit`). The whole pipeline, this stage
included, is driven by `../run_all.sh`.

## Code organization of expand_commutators.py

- Index machinery: `SPACE`, `FAMILY`, `space_of`, `index_key` — every
  index symbol belongs to a space (g = whole, o = occupied, v = virtual),
  read off its base letter (p, p1, p2, ... are all whole-space).
- `Tensor` — a coefficient tensor (h, f, g, t); `Tensor.canonical()` is
  the minimum arrangement under the exact tensor symmetries.
- Simplify steps: `resolve_delta` (step 1), `op_canonical` (2),
  `substitute_outer` (3), `sort_tensors` (4), `Tensor.canonical` (5),
  `substitute_dummies` (6), gathered per term in `canonical_form`; the
  canonical representative is the renaming-orbit minimum
  `orbit_canonical`; `merge()` merges terms whose representatives agree.
- Expansion: `comm_terms`, `dagger_terms`, `expand_expr`, plus the
  operator pools `H_OP`, `F_OP`, `V_OP`, `_excitation_pool`.
- Public API for later stages: `expanded_blocks()` (list of 12 dicts with
  keys `kind, expr, lhs, x_name, r, s, is_dag, terms, ranks`),
  `build_blocks()`/`BLOCKS` (legacy tuple view), and the term machinery
  (`Tensor`, `Operator`, `SPACE`, `FAMILY`, `merge`, `canonical_form`,
  `orbit_canonical`, `term_latex`, `terms_latex`, `pack_rows`,
  `ranks_of`).
- LaTeX: `term_latex`, `terms_latex`, `pack_rows`, `write_tex`
  (`compile_tex` and `idx_latex` are shared, from
  `../utils/latex_utils.py`).
- Verification: `check_blocks`, `check_blocks_spin_summed` — in
  `../test/check_many_body_equation.py`, imported and run by the stage
  script's `__main__`.

A term is a tuple `(coeff, tensors, op, leftover)`: `coeff` a sympy
Rational, `tensors` a tuple of `Tensor`, `op` an `Operator`, `leftover`
unresolved Kronecker deltas (empty for every block of this truncation).

## Conventions in force (do not violate)

- Spin-integrated formalism: **NO index antisymmetry of t or g anywhere**.
  The only symmetries are the simultaneous pair swaps
  t^{ab}_{ij} = t^{ba}_{ji} and g^{pq}_{rs} = g^{qp}_{sr}.
- Real coefficients: h and f symmetric (h^q_p = h^p_q), g symmetric under
  pair exchange (g^{pq}_{rs} = g^{rs}_{pq}); conjugate blocks are obtained
  by conjugating term by term (E rows swapped, h/f/g Hermitian-swapped,
  t unchanged).
- The spin-summed E operator is invariant only under the DIAGONAL pair
  permutation of its index rows, with NO sign (E^{qp..}_{rs..} =
  E^{pq..}_{sr..}); a single row alone is NOT a symmetry. All merging
  orbit moves carry sign +1.
- Einsum convention: repeated indices are summed over their space; dummy
  letters are numbered per space (a1,a2,... virtual; i1,i2,... occupied;
  p1,p2,... whole space) and excitations of one nested commutator carry
  fresh dummy letters.
- Merging must use ONLY identities exact in this convention; both
  numerical checks must stay green after any change.

## Interaction rules

- **Ask before you edit.** Before changing any code, grill the user to
  pinpoint exactly what they want. If there is ANY ambiguity — scope,
  naming, which of several valid approaches, whether docs or downstream
  stages should change too — ask, and keep asking until everything is
  clear. Do not start editing on a guess.

## Code style

- Single-file stages for the derivation logic (shared LaTeX helpers in
  `../utils/latex_utils.py`, numerical checks in `../test/`), stdlib +
  sympy + numpy only; dense docstrings and section banner comments
  (`# ---...---`) stating the physics reason for each rule. Follow the
  existing style: lowercase prose comments that explain WHY a rule is
  valid, not what the code does.
- When a simplify rule or the public API changes, update `readme.md`, the
  module docstring, and (if the tex prose mentions it) the strings in
  `write_tex()` — the generated tex repeats the rules for the reader.
- No CI, no linter config, no deployment: the deliverable is the verified
  expansion and the generated pdf.
