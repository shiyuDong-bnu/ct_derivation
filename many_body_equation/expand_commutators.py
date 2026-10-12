#!/usr/bin/env python3
"""Expand all CT commutators into many-body equations.

The block inventory is taken from the commutator stage API
(``commutator/generate_commutators.py``, ``commutator_blocks()``): the 4
first-order blocks [h, T_r], [V, T_r] and the 8 second-order Fock blocks
[[F, T_r], T_s], [[F, T_r]^dagger, T_s] (r, s = 1, 2).  Every block is
expanded with the generalized Wick theorem of
``generalized_wick_theorem/generalized_wick_commutator.py``, substituting

    h  = h^q_p E^p_q                    (p, q general)
    F  = f^q_p E^p_q                    (p, q general)
    V  = (1/2) g^{pq}_{rs} E^{rs}_{pq}  (p, q, r, s general)
    T1 = t^a_i E^a_i                    (a virtual, i occupied)
    T2 = (1/2) t^{ab}_{ij} E^{ab}_{ij}

and applying the simplification steps of readme.md:

    1. resolve_delta:  the index-space delta rules (indices are numbered
       per space: p1,p2,... whole space; i1,i2,... occupied; a1,a2,...
       virtual; bare letters p,q,r,s etc. denote the same spaces;
       delta(whole, vir) -> the whole-space index becomes the virtual one;
       delta(whole, occ) -> occupied;  delta(occ, vir) -> 0;
       same-space dummies contract;  delta(whole, whole) raises
       (unreachable with the three spaces of this truncation));
    2. op_canonical:  the operator's index pairs are sorted, each
       (upper, lower) pair kept together, with NO sign (the diagonal
       pair-permutation symmetry of the spin-summed E);
    3. substitute_outer:  the operator's (outer) indices are re-lettered
       onto the numbered pool of their space in appearance order, the
       same substitution applied to the tensor part simultaneously;
    4. sort_tensors:  the tensor product is sorted by the tensor head
       name (the tensors are scalars and commute -- a multiset);
    5. Tensor.canonical:  each tensor's indices are reordered to the
       minimum over its symmetry orbit under the index order:  space
       (whole > occ > vir), outer index > dummy index, lexical;
    6. substitute_dummies:  the dummy indices are re-lettered onto the
       numbered pool of their space, continuing after the outer letters.

    Steps 2-6 alone are not letter-invariant (steps 2 and 5 sort by the
    index order of the term's current letters);  the canonical
    representative of a term is the minimum of the steps-2-6 form over
    the renaming orbit (``orbit_canonical``).  Terms whose canonical
    representatives agree are equal and are merged (``merge``, applied
    once per block in ``expanded_blocks``).

The truncation is by perturbation order (Moller-Plesset counting
h ~ O(1), V ~ O(lambda), T ~ O(lambda); the Fock operator F differs from
h only at O(lambda), so it is interchangeable with h inside the nested
blocks), correct through O(lambda^2): the full first-order term
[H_b, A] with H_b = h + V and the Fock half of the second order,
(1/2)[[F, A], A].  The valence double commutator blocks [[V, Tr], Ts] are
O(lambda^3) and are NOT expanded.

For the conjugate blocks [X_r^dagger, T_s] the expanded first-order block
is conjugated term by term: E^{u..}_{l..} -> E^{l..}_{u..}, h, f and g
Hermitian (h^q_p -> h^p_q, f^q_p -> f^p_q, g^{pq}_{rs} -> g^{rs}_{pq});
the coefficients are real, so t is unchanged.

Terms are merged over every identity that is exact in this convention:
renaming of any summed index (the free indices of the operator are the
summation labels of the operator basis sum, so they can be renamed like
tensor dummies), the spin-integrated pair symmetry t^{ab}_{ij} = t^{ba}_{ji}
and g^{pq}_{rs} = g^{qp}_{sr}, real symmetric h and f, real g with
g^{pq}_{rs} = g^{rs}_{pq}, and the diagonal pair-permutation symmetry of
the spin-summed E (the two rows may only be permuted together, with no
sign; a single row alone is NOT a symmetry of the spin-summed operator).
No index antisymmetry of t or g is used anywhere.

Public API for later stages (used by space_restriction and
cumulant_expansion):

    expanded_blocks()  -- the 12 expanded blocks as a list of dicts:
                          kind, expr, lhs, x_name, r, s, is_dag, terms,
                          ranks
    build_blocks()     -- fills the legacy BLOCKS tuple view
                          (kind, x_name, r, s, is_dag, lhs, terms)
    Tensor, Operator, SPACE, FAMILY, merge, term_latex,
    terms_latex, pack_rows, ranks_of -- the term machinery

Every expanded block is verified numerically against the direct (nested)
matrix commutator with random real tensors of these symmetry classes,
twice: on Jordan-Wigner spin-orbital matrices (4 modes, 2 occupied + 2
virtual) and with genuinely spin-summed operators (4 spatial orbitals =
8 spin orbitals) -- the spin-orbital check alone cannot see the merging
over the diagonal pair permutation of the spin-summed E,
because single-row antisymmetry IS exact for a spin-orbital string.  The
checks live in test/check_many_body_equation.py at the repo root and are
run by this script.  The blocks are then written to
many_body_equations.tex.
"""

import itertools
import sys
from pathlib import Path

from sympy import Rational, symbols
from sympy.physics.quantum import Commutator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "generalized_wick_theorem"))
sys.path.insert(0, str(HERE.parent / "commutator"))
sys.path.insert(0, str(HERE.parent / "utils"))
from generalized_wick_commutator import Operator, commutator  # noqa: E402
import generate_commutators as gen_commutator  # noqa: E402
from latex_utils import compile_tex, idx_latex  # noqa: E402

TEX_PATH = HERE / "many_body_equations.tex"

# ---------------------------------------------------------------------------
# index symbols and their spaces
# ---------------------------------------------------------------------------
p, q, r, s = symbols("p q r s")          # whole space
i, j, k, l = symbols("i j k l")          # occupied
a, b, c, d = symbols("a b c d")          # virtual

SPACE = {p: "g", q: "g", r: "g", s: "g",
         i: "o", j: "o", k: "o", l: "o",
         a: "v", b: "v", c: "v", d: "v"}

OCC = (0, 1)
VIR = (2, 3)
ALL = (0, 1, 2, 3)
RANGE = {"o": OCC, "v": VIR, "g": ALL}

# numbered dummy letters per space, handed out by _excitation_pool and
# the substitution steps (substitute_outer / substitute_dummies); the
# space of a symbol is read off its base letter
# by space_of, so p, p1, p2, ... are all whole-space
FAMILY = {"g": symbols("p1:9"),
          "o": symbols("i1:9"),
          "v": symbols("a1:9")}

_BASE_SPACE = {str(x): sp for x, sp in SPACE.items()}


def space_of(x):
    """The index space of a symbol: the space of its base letter, ignoring
    any numeric suffix (p, p1, p2, ... are all whole-space)."""
    return _BASE_SPACE[str(x).rstrip("0123456789")]

# canonical index order: general > occupied > virtual, lexical within a
# space (flip the ranks here to change the convention)
SPACE_ORDER = {"v": 0, "o": 1, "g": 2}


def index_key(x):
    """Total order on index symbols: orbital space first, then the base
    letter, then the number (p < p1 < p2 < ... < p10 < q)."""
    name = str(x)
    base = name.rstrip("0123456789")
    num = name[len(base):]
    return (SPACE_ORDER[space_of(x)], base, int(num) if num else 0)


class Tensor:
    """A coefficient tensor, e.g. f^q_p, g^{pq}_{rs}, t^{ab}_{ij}.

    Coefficients are real; the star field is legacy and stays False."""

    def __init__(self, name, upper, lower, star=False):
        self.name = name
        self.upper = tuple(upper)
        self.lower = tuple(lower)
        self.star = star

    def substitute(self, subs):
        return Tensor(self.name,
                      [subs.get(x, x) for x in self.upper],
                      [subs.get(x, x) for x in self.lower], self.star)

    def dagger(self):
        if self.name in ("f", "g", "h"):  # Hermitian: swap the index pair(s)
            return Tensor(self.name, self.lower, self.upper)
        return Tensor(self.name, self.upper, self.lower)   # t is real

    def signature(self):
        return (self.name, self.star, self.upper, self.lower)

    def __eq__(self, other):
        return (isinstance(other, Tensor)
                and self.signature() == other.signature())

    def __hash__(self):
        return hash(self.signature())

    def canonical(self, outer=frozenset()):
        """Canonical index arrangement under the exact symmetries of the
        tensor (step 5 of readme.md).  The canonical arrangement is the
        minimum over the symmetry orbit under the index order

            space:  whole (total) > occupied > virtual
            then:   outer index > dummy index
            then:   lexical order (base letter, then number)

        where an "outer" index is one that also sits on the operator
        (``outer`` is the operator's index set; the default empty set
        recovers the plain ``index_key`` order).  The symmetry orbits:

            tensor          pair swap            pair exchange
                            (within pairs)       (upper <-> lower)
            h, f, 1-RDM D   trivial              yes -- Hermiticity
                            (1 index per pair)
            t2              yes -- diagonal      no -- not Hermitian
                            pair symmetry of E        (vir/occ differ)
            g, 2-RDM D      yes                  yes -- Hermiticity

        Pair swap is inherited from the diagonal pair symmetry of the
        spin-summed E (E^{rs}_{pq} = E^{sr}_{qp}); pair exchange is
        Hermiticity with real coefficients.  For a rank-1 tensor pair
        exchange is the upper/lower swap (h^q_p = h^p_q).  No index
        antisymmetry is used anywhere."""
        def key(x):
            name = str(x)
            base = name.rstrip("0123456789")
            num = name[len(base):]
            return (SPACE_ORDER[space_of(x)], x in outer, base,
                    int(num) if num else 0)

        opts = [(self.upper, self.lower)]
        if self.name in ("f", "h", "D") and len(self.upper) == 1:
            opts.append((self.lower, self.upper))
        elif len(self.upper) == 2:
            sw = ((self.upper[1], self.upper[0]),
                  (self.lower[1], self.lower[0]))
            opts.append(sw)
            if self.name in ("g", "D"):
                opts += [(self.lower, self.upper), (sw[1], sw[0])]
        up, lo = min(opts, key=lambda ul: tuple(key(x) for x in ul[0] + ul[1]))
        return Tensor(self.name, up, lo, self.star)

    def latex(self):
        up = "".join(idx_latex(x) for x in self.upper)
        lo = "".join(idx_latex(x) for x in self.lower)
        if self.star:
            up += " *"
        return "%s^{%s}_{%s}" % (self.name, up, lo)


# ---------------------------------------------------------------------------
# terms: (Rational coeff, (Tensor, ...), Operator, leftover deltas)
# ---------------------------------------------------------------------------
def resolve_delta(coeff, tensors, deltas, op):
    """Simplification step 1 of readme.md: the index-space delta rules.
    Returns the term or None (zero)."""
    subs = {}
    leftover = []
    for x, y in deltas:
        sx, sy = space_of(x), space_of(y)
        if {sx, sy} == {"o", "v"}:
            return None                      # delta(occ, vir) = 0
        if sx == "g" and sy != "g":
            subs[x] = y
        elif sy == "g" and sx != "g":
            subs[y] = x
        elif sx == sy:                       # same-space dummies: contract
            subs[y] = x
        else:                                # unreachable with three spaces
            raise ValueError(f"unresolved delta ({x}, {y})")
    if subs:
        tensors = tuple(t.substitute(subs) for t in tensors)
        op = Operator([subs.get(x, x) for x in op.upper_index],
                      [subs.get(x, x) for x in op.lower_index])
    return (coeff, tensors, op, tuple(leftover))


def op_canonical(op):
    """Canonical pair order (step 2): the index pairs sorted by the upper
    index under ``index_key``, each (upper, lower) pair kept together, no
    sign -- the full pair-level symmetry of the spin-summed E."""
    order = sorted(range(op.rank), key=lambda k: index_key(op.upper_index[k]))
    return Operator([op.upper_index[k] for k in order],
                    [op.lower_index[k] for k in order])


def comm_terms(terms_a, terms_b):
    """Commutator of two sums of terms, expanded with the generalized Wick
    theorem and simplified with the index-space rules (step 1,
    ``resolve_delta``).  NOT merged: merging (``merge``) happens once per
    block in ``expanded_blocks``, after the full commutator is expanded."""
    out = []
    for ca, ta, opa, _ in terms_a:
        for cb, tb, opb, _ in terms_b:
            for (deltas, up, lo), c in commutator(opa, opb).items():
                term = resolve_delta(ca * cb * c, ta + tb, deltas,
                                     Operator(up, lo))
                if term is not None:
                    out.append(term)
    return out


def dagger_terms(terms):
    """Term-by-term Hermitian conjugate of an expanded block.  NOT merged:
    merging (``merge``) happens once per block in ``expanded_blocks``."""
    out = []
    for coeff, tensors, op, leftover in terms:
        if leftover:
            raise ValueError("dagger with unresolved deltas")
        out.append((coeff, tuple(t.dagger() for t in tensors),
                    Operator(op.lower_index, op.upper_index), ()))
    return out


def ranks_of(terms):
    return sorted({op.rank for _, _, op, _ in terms})


# ---------------------------------------------------------------------------
# canonicalization and merging (steps 2-6 of readme.md): two terms are
# equal -- and merge -- iff they reach the same canonical form by moves
# that are exact identities in the spin-integrated, real-coefficient
# convention (the "Exact symmetries" section of readme.md):
#   * the diagonal pair-permutation symmetry of the spin-summed E: the two
#     index rows may only be permuted TOGETHER, with NO sign
#     (E^{qp...}_{rs...} = E^{pq...}_{sr...}).  A single row alone is NOT a
#     symmetry of the spin-summed operator (that antisymmetry belongs to
#     the spin-orbital string, not to the spin-integrated E)
#     -- step 2, op_canonical;
#   * the pair symmetry of the amplitudes, t^{ab}_{ij} = t^{ba}_{ji} and
#     g^{pq}_{rs} = g^{qp}_{sr} (simultaneous swap of both index pairs),
#     and real symmetric h, f and pair-exchange-symmetric g
#     -- step 5, Tensor.canonical;
#   * renaming of any summed index within its space.  ALL indices are
#     summed: the free (outer) indices of the operator are the summation
#     labels of the operator basis sum, so they can be renamed like tensor
#     dummies -- steps 3 and 6, substitute_outer / substitute_dummies.
# No index antisymmetry of t or g is used.  All moves carry sign +1.
# ---------------------------------------------------------------------------
def substitute_outer(tensors, op):
    """Step 3: re-letter the outer indices (the indices of the operator)
    onto the numbered pool of their space, in order of appearance (upper
    row, then lower row); the same substitution is applied to the tensor
    part simultaneously, since the tensors share indices with the
    operator.  A tensor-only (dummy) letter that collides with a new
    outer letter is first moved to a scratch letter of its space; all
    dummy indices are re-lettered canonically in step 6
    (substitute_dummies)."""
    mapping, count = {}, {}
    for x in list(op.upper_index) + list(op.lower_index):
        if x not in mapping:
            sp = space_of(x)
            mapping[x] = FAMILY[sp][count.get(sp, 0)]
            count[sp] = count.get(sp, 0) + 1
    targets = set(mapping.values())
    aside = {}
    for t in tensors:
        for x in t.upper + t.lower:
            if x not in mapping and x in targets and x not in aside:
                base = str(x).rstrip("0123456789")
                aside[x] = symbols("%s%d" % (base, 1000 + len(aside)))
    sub = {**aside, **mapping}
    return (tuple(t.substitute(sub) for t in tensors),
            Operator([mapping[x] for x in op.upper_index],
                     [mapping[x] for x in op.lower_index]))


def sort_tensors(tensors, outer):
    """Step 4: order the tensor product by the tensor head name (the
    tensors are scalars and commute -- the product is a multiset).  Ties
    (same head name) are broken by the canonical index arrangement of
    step 5, so the order does not depend on the written arrangement of
    the tensors."""
    return tuple(sorted(
        tensors, key=lambda t: (t.name, str(t.canonical(outer).signature()))))


def substitute_dummies(tensors, op):
    """Step 6: re-letter the dummy indices (those not on the operator)
    onto the numbered pool of their space, skipping the outer letters, in
    order of appearance over the canonical tensor product.  The
    operator's letters are already canonical (step 3) and are not
    touched."""
    outer = set(op.upper_index) | set(op.lower_index)
    used = {}
    for x in outer:
        used.setdefault(space_of(x), set()).add(x)
    free = {sp: iter([y for y in FAMILY[sp] if y not in used.get(sp, ())])
            for sp in FAMILY}
    mapping = {}
    for t in tensors:
        for x in t.upper + t.lower:
            if x not in outer and x not in mapping:
                mapping[x] = next(free[space_of(x)])
    return tuple(t.substitute(mapping) for t in tensors)


def canonical_form(tensors, op):
    """The canonical written form of one term: steps 2-6 of readme.md
    applied in order (op_canonical, substitute_outer, sort_tensors,
    Tensor.canonical with the operator's index set, substitute_dummies).
    Returns (tensors, op).  NOT letter-invariant by itself: steps 2 and 5
    sort by the index order of the term's current letters, so two
    renamings of the same term can give different forms -- the canonical
    representative is the minimum over the renaming orbit
    (``orbit_canonical``)."""
    op = op_canonical(op)                        # step 2
    tensors, op = substitute_outer(tensors, op)  # step 3
    outer = set(op.upper_index) | set(op.lower_index)
    tensors = sort_tensors(tensors, outer)       # step 4
    tensors = tuple(t.canonical(outer) for t in tensors)  # step 5
    return substitute_dummies(tensors, op), op   # step 6


def orbit_canonical(tensors, op):
    """The canonical representative of a term: the minimum of
    ``canonical_form`` (steps 2-6) over the renaming orbit -- every
    within-space renaming of the term's indices is tried and the minimum
    under the index order (then the tensor signatures) wins.  The term's
    letters are first brought onto the numbered pool prefix (any
    deterministic injective map), so that two renamings of the same term
    enumerate permutations of the SAME letter set; then the orbit minimum
    is letter-invariant by construction: the renaming orbit of a term is
    the same set for two terms that differ by a renaming, so their
    minima agree.  Returns the minimizing (tensors, op)."""
    by_space = {}
    for x in sorted({x for t in tensors for x in t.upper + t.lower}
                    | set(op.upper_index) | set(op.lower_index),
                    key=str):
        by_space.setdefault(space_of(x), []).append(x)
    norm = {}
    for sp, xs in by_space.items():
        norm.update(zip(xs, FAMILY[sp]))
    tensors = tuple(t.substitute(norm) for t in tensors)
    op = Operator([norm[x] for x in op.upper_index],
                  [norm[x] for x in op.lower_index])
    pools = {sp: [norm[x] for x in xs] for sp, xs in by_space.items()}
    best_key = None
    best = None
    for combo in itertools.product(
            *(itertools.permutations(xs) for xs in pools.values())):
        sub = {}
        for xs, perm in zip(pools.values(), combo):
            sub.update(zip(xs, perm))
        ts = tuple(t.substitute(sub) for t in tensors)
        o = Operator([sub.get(x, x) for x in op.upper_index],
                     [sub.get(x, x) for x in op.lower_index])
        cand_t, cand_o = canonical_form(ts, o)
        key = (tuple(index_key(x) for x in cand_o.upper_index),
               tuple(index_key(x) for x in cand_o.lower_index),
               tuple(str(t.signature()) for t in cand_t))
        if best_key is None or key < best_key:
            best_key = key
            best = (cand_t, cand_o)
    return best


def merge(terms):
    """Merge equal terms -- the merging entry point of the stage,
    applied once per block (``expanded_blocks``) after the full
    commutator is expanded.  Every term is put in canonical form by the
    steps of readme.md: the deltas are resolved during the expansion
    (step 1, ``resolve_delta``), then the operator is canonicalized
    (step 2, ``op_canonical``), the outer indices are substituted (step
    3, ``substitute_outer``), the tensor product is sorted (step 4,
    ``sort_tensors``), each tensor's indices are canonicalized (step 5,
    ``Tensor.canonical``), and the dummy indices are substituted (step 6,
    ``substitute_dummies``); the canonical representative of the term is
    the minimum of steps 2-6 over the renaming orbit
    (``orbit_canonical``).  Two terms merge iff the canonical
    representatives agree."""
    merged = {}
    keep = {}
    for coeff, tensors, op, leftover in terms:
        if leftover:
            raise ValueError("merge with unresolved deltas")
        canon_t, canon_op = orbit_canonical(tensors, op)
        key = (canon_op, canon_t)
        if key not in merged:
            merged[key] = Rational(0)
            keep[key] = (canon_t, canon_op)
        merged[key] += coeff
    return [(coeff, keep[key][0], keep[key][1], ())
            for key, coeff in merged.items() if coeff]


# ---------------------------------------------------------------------------
# the operator pool
# ---------------------------------------------------------------------------
def op_term(coeff, tensors, upper, lower):
    return [(Rational(coeff), tuple(tensors), Operator(upper, lower), ())]


H_OP = op_term(1, [Tensor("h", (q,), (p,))], [p], [q])
F_OP = op_term(1, [Tensor("f", (q,), (p,))], [p], [q])
V_OP = op_term(Rational(1, 2), [Tensor("g", (p, q), (r, s))], [r, s], [p, q])

STATIC_POOL = {gen_commutator.H: H_OP, gen_commutator.F: F_OP, gen_commutator.V: V_OP}


def _excitation_pool(rank, slot):
    """The (slot + 1)-th excitation operator of a block, with fresh dummy
    letters: slot 0 carries a,b / i,j, slot 1 c,d / k,l, and so on -- the
    excitations of one block are independent summations, so their dummy
    letters must differ."""
    vu, vv = FAMILY["v"][2 * slot], FAMILY["v"][2 * slot + 1]
    ou, ov = FAMILY["o"][2 * slot], FAMILY["o"][2 * slot + 1]
    if rank == 1:
        return op_term(1, [Tensor("t", (vu,), (ou,))], [vu], [ou])
    return op_term(Rational(1, 2), [Tensor("t", (vu, vv), (ou, ov))],
                   [vu, vv], [ou, ov])


def expand_expr(expr, counter=None):
    """Expand a commutator-stage expression into many-body terms.

    Walks the nested Commutator/Dagger expression returned by
    ``commutator_blocks()`` and evaluates it with the generalized Wick
    theorem; ``counter`` hands out fresh dummy letters for the excitation
    operators (inner excitation first).  No merging happens during the
    walk: the fully expanded block is merged once (``merge``) in
    ``expanded_blocks``.
    """
    if counter is None:
        counter = [0]
    if isinstance(expr, Commutator):
        return comm_terms(expand_expr(expr.args[0], counter),
                          expand_expr(expr.args[1], counter))
    if gen_commutator.is_dagger(expr):
        return dagger_terms(expand_expr(expr.args[0], counter))
    if expr == gen_commutator.T1 or expr == gen_commutator.T2:
        rank = 1 if expr == gen_commutator.T1 else 2
        pool = _excitation_pool(rank, counter[0])
        counter[0] += 1
        return pool
    return STATIC_POOL[expr]


# ---------------------------------------------------------------------------
# all commutator blocks, driven by the commutator-stage API
# ---------------------------------------------------------------------------
BLOCKS = []        # legacy view: (kind, x_name, r, s, is_dag, lhs, terms)
_EXPANDED = None


def _block_meta(kind, expr):
    """(x_name, r, s, is_dag) of a block of commutator_blocks()."""
    def t_rank(t):
        return 1 if t == gen_commutator.T1 else 2
    if kind == "first":
        x, t = expr.args
        return ("h" if x == gen_commutator.H else "V", t_rank(t), None, False)
    inner, ts = expr.args
    is_dag = gen_commutator.is_dagger(inner)
    base = inner.args[0] if is_dag else inner   # the [F, T_r] commutator
    return ("F", t_rank(base.args[1]), t_rank(ts), is_dag)


def expanded_blocks():
    """All expanded commutator blocks through O(lambda^2).

    One dict per block of ``commutator.commutator_blocks()`` (4
    "first_order" + 8 "second_order", in the same order):

        kind    "first" | "second"
        expr    the commutator-stage sympy expression of the block
        lhs     its LaTeX rendering
        x_name  "h" | "V" (first order), "F" (second order)
        r, s    excitation ranks (s is None at first order)
        is_dag  True for the [[F, T_r]^dagger, T_s] blocks
        terms   [(coeff, tensors, op, leftover), ...], merged over steps
                1-6 of readme.md: ``resolve_delta`` is applied term by
                term during the expansion, and the fully expanded block
                is merged once by ``merge``
        ranks   sorted operator ranks present

    A term is an exact operator identity: coeff a Rational, tensors a
    tuple of Tensor, op an Operator; the leftover Kronecker deltas are
    empty for every block of this truncation.
    """
    global _EXPANDED
    if _EXPANDED is not None:
        return _EXPANDED
    out = []
    for kind, key in (("first", "first_order"), ("second", "second_order")):
        for expr in gen_commutator.commutator_blocks()[key]:
            x_name, rr, ss, is_dag = _block_meta(kind, expr)
            terms = merge(expand_expr(expr))
            out.append({"kind": kind, "expr": expr, "lhs": gen_commutator.ct_latex(expr),
                        "x_name": x_name, "r": rr, "s": ss, "is_dag": is_dag,
                        "terms": terms, "ranks": ranks_of(terms)})
    _EXPANDED = out
    return out


def build_blocks():
    """Fill the legacy BLOCKS list from expanded_blocks()."""
    BLOCKS[:] = [(b["kind"], b["x_name"], b["r"], b["s"], b["is_dag"],
                  b["lhs"], b["terms"]) for b in expanded_blocks()]


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------
def term_latex(coeff, tensors, op, leftover, first_term):
    body = ""
    mag = abs(coeff)
    if mag != 1:
        body += (r"\tfrac{%d}{%d}\, " % (mag.numerator, mag.denominator)
                 if mag.denominator != 1 else "%d\\, " % mag.numerator)
    body += " ".join(t.latex() for t in tensors) + " "
    for x, y in leftover:
        body += r"\delta^{%s}_{%s} " % (x, y)
    if op.rank:
        body += (r"\hat{E}^{%s}_{%s}"
                 % ("".join(idx_latex(u) for u in op.upper_index),
                    "".join(idx_latex(l) for l in op.lower_index)))
    if coeff < 0:
        return "- " + body
    if first_term:
        return body
    return "+ " + body


def terms_latex(terms):
    def key(term):
        _, tensors, op, _ = term
        return (op.rank,
                [index_key(u) for u in op.upper_index],
                [index_key(l) for l in op.lower_index],
                [t.latex() for t in tensors])
    ordered = sorted(terms, key=key)
    return [term_latex(c, t, op, lo, n == 0)
            for n, (c, t, op, lo) in enumerate(ordered)]


ROW_WIDTH = 170     # approximate source characters per align row


def pack_rows(lhs, lines):
    """Pack several terms on each align row, up to ROW_WIDTH characters."""
    rows = []
    current = lhs + " &= " + lines[0]
    for line in lines[1:]:
        if len(current) + 1 + len(line) > ROW_WIDTH:
            rows.append(current)
            current = "&\\quad " + line
        else:
            current += " " + line
    rows.append(current)
    return rows


def write_tex():
    L = []
    add = L.append
    add(r"\documentclass[11pt]{article}")
    add(r"\usepackage{amsmath,amssymb}")
    add(r"\usepackage[margin=2.5cm]{geometry}")
    add(r"\allowdisplaybreaks")
    add(r"\title{Many-body expansion of the CT commutators}")
    add(r"\date{generated by \texttt{expand\_commutators.py}}")
    add(r"\begin{document}")
    add(r"\maketitle")
    add("")
    add(r"All commutator blocks of the \texttt{commutator} stage, taken from"
        r" its API (\texttt{commutator\_blocks()} of"
        r" \texttt{commutator/generate\_commutators.py}) and expanded with"
        r" the generalized Wick theorem"
        r" (\texttt{generalized\_wick\_theorem/generalized\_wick\_commutator.py})."
        r" The truncation is by perturbation order (M\o ller--Plesset counting"
        r" $\hat{h} \sim O(1)$, $\hat{V} \sim O(\lambda)$,"
        r" $\hat{T} \sim O(\lambda)$; the Fock operator differs from"
        r" $\hat{h}$ only at $O(\lambda)$), correct through $O(\lambda^2)$:"
        r" the full first-order term $[\hat{H}_b, \hat{A}]$ with"
        r" $\hat{H}_b = \hat{h} + \hat{V}$, and the Fock half of the"
        r" second order, $\tfrac{1}{2}[[\hat{F}, \hat{A}], \hat{A}]$"
        r" (writing $\hat{F}$ for $\hat{h}$ there changes the result only"
        r" at $O(\lambda^3)$).  The valence double"
        r" commutator is $O(\lambda^3)$ and is not expanded."
        r" Repeated indices are summed over their space.  The index spaces and"
        r" the delta rules are those of \texttt{readme.md}:")
    add(r"\begin{align}")
    add(r"p_1,p_2,\ldots &:\ \text{whole space}, &")
    add(r"i_1,i_2,\ldots &:\ \text{occupied}, &")
    add(r"a_1,a_2,\ldots &:\ \text{virtual}, \\")
    add(r"\delta(\text{whole},\text{vir}) &\to \text{virtual}, &")
    add(r"\delta(\text{whole},\text{occ}) &\to \text{occupied}, &")
    add(r"\delta(\text{occ},\text{vir}) &= 0.")
    add(r"\end{align}")
    add(r"The operators are")
    add(r"\begin{align}")
    add(r"\hat{h} &= h^q_p\, \hat{E}^p_q, &")
    add(r"\hat{V} &= \tfrac{1}{2} g^{pq}_{rs}\, \hat{E}^{rs}_{pq}, \\")
    add(r"\hat{F} &= f^q_p\, \hat{E}^p_q, &")
    add(r"\hat{T}_1 &= t^a_i\, \hat{E}^a_i, \\")
    add(r"\hat{T}_2 &= \tfrac{1}{2} t^{ab}_{ij}\, \hat{E}^{ab}_{ij},")
    add(r"\end{align}")
    add(r"with $\hat{h}$, $\hat{F}$ and $\hat{V}$ Hermitian.  The dummy"
        r" indices are numbered per space ($a_1,a_2,\ldots$ virtual,"
        r" $i_1,i_2,\ldots$ occupied, $p_1,p_2,\ldots$ whole space); the"
        r" second excitation of the nested commutators carries the next"
        r" pair ($a_3,a_4$ and $i_3,i_4$)."
        r" The coefficients are real and follow the spin-integrated"
        r" convention: no index antisymmetry is used anywhere,"
        r" $g^{pq}_{rs}$ and $t^{ab}_{ij}$ carry only the simultaneous pair"
        r" swap $g^{pq}_{rs} = g^{qp}_{sr}$, $t^{ab}_{ij} = t^{ba}_{ji}$, and"
        r" $h^q_p = h^p_q$, $f^q_p = f^p_q$, $g^{pq}_{rs} = g^{rs}_{pq}$."
        r" Conjugate blocks are expanded by conjugating the first-order"
        r" result term by term ($\hat{E}^{u\ldots}_{l\ldots} \to"
        r" \hat{E}^{l\ldots}_{u\ldots}$, $f^q_p \to f^p_q$,"
        r" $g^{pq}_{rs} \to g^{rs}_{pq}$; $t$ is real and unchanged)."
        r" Blocks containing operators of rank higher than two are marked;"
        r" their cumulant expansion (and the $[\ldots]_{1,2}$ truncation)"
        r" is done separately.  The expanded blocks are available"
        r" programmatically through \texttt{expanded\_blocks()}.")
    add("")
    add(r"Equal terms are merged once per block, after the full expansion"
        r" (\texttt{merge}), over every"
        r" identity that is exact in this convention (\texttt{readme.md},"
        r" simplification steps 2--6): the diagonal pair-permutation symmetry of the"
        r" spin-summed"
        r" $\hat{E}$ (the two index rows may only be permuted together,"
        r" with no sign,"
        r" $\hat{E}^{qp\ldots}_{rs\ldots} = \hat{E}^{pq\ldots}_{sr\ldots}$;"
        r" a single row alone is \emph{not} a symmetry of the spin-summed"
        r" operator), the pair symmetries of $g$ and $t$ above, and the"
        r" renaming of summed indices --- the free indices of the operator"
        r" are the summation labels of the operator basis sum, so they can"
        r" be renamed like tensor dummies.")
    add("")
    add(r"Every block is verified numerically against the direct nested"
        r" commutator on Jordan--Wigner matrices, twice: with spin-orbital"
        r" operators (4 modes, 2 occupied $+$ 2 virtual) and with the"
        r" genuinely spin-summed operators (4 spatial orbitals $=$ 8 spin"
        r" orbitals); the second check is the one that sees the merging"
        r" steps.")
    add("")

    sections = [("First-order commutators", "first"),
                ("Second order: Fock blocks", "second")]
    for title, want_kind in sections:
        add(r"\section{%s}" % title)
        for blk in expanded_blocks():
            if blk["kind"] != want_kind:
                continue
            terms = blk["terms"]
            lines = terms_latex(terms)
            big = [n for n in blk["ranks"] if n > 2]
            marker = ""
            if big:
                names = {3: "three-body", 4: "four-body"}
                marker = (r"\quad (\text{%s terms})"
                          % " and ".join(names[n] for n in big))
            add(r"\begin{align}")
            rows = pack_rows(blk["lhs"], lines)
            for row in rows[:-1]:
                add(row + r" \\")
            add(rows[-1] + marker)
            add(r"\end{align}")
        add("")

    add(r"\section{Assembly}")
    add(r"With the blocks above, the transcorrelated Hamiltonian through"
        r" $O(\lambda^2)$ is")
    add(r"\begin{align}")
    add(r"\left[\hat{H}_b,\, \hat{A}\right] &= \left[\hat{h},\, \hat{A}\right]"
        r" + \left[\hat{V},\, \hat{A}\right], \\")
    add(r"\left[\hat{h},\, \hat{A}\right] &= \left[\hat{h},\, \hat{T}_1\right]"
        r" + \left[\hat{h},\, \hat{T}_2\right] + \mathrm{h.c.}, \\")
    add(r"\left[\hat{V},\, \hat{A}\right] &= \left[\hat{V},\, \hat{T}_1\right]"
        r" + \left[\hat{V},\, \hat{T}_2\right] + \mathrm{h.c.}, \\")
    add(r"\tfrac{1}{2}\left[\left[\hat{F},\, \hat{A}\right],\, \hat{A}\right]"
        r" &= \tfrac{1}{2}\sum_{r,s=1}^{2}\left("
        r" \left[\left[\hat{F},\, \hat{T}_r\right],\, \hat{T}_s\right]"
        r" + \left[\left[\hat{F},\, \hat{T}_r\right]^{\dagger},\,"
        r" \hat{T}_s\right]\right) + \mathrm{h.c.},")
    add(r"\end{align}")
    add(r"the valence double commutator"
        r" $\tfrac{1}{2}[[\hat{V}, \hat{A}], \hat{A}] \sim O(\lambda^3)$"
        r" being dropped by the perturbation-order truncation.")
    add(r"\end{document}")

    TEX_PATH.write_text("\n".join(L) + "\n")
    print(f"wrote {TEX_PATH}")


if __name__ == "__main__":
    # register the running script under its module name, so that the check
    # module imports this very instance and not a second copy (the term
    # machinery relies on Tensor/Operator class identity)
    sys.modules.setdefault("expand_commutators", sys.modules["__main__"])
    sys.path.insert(0, str(HERE.parent / "test"))
    from check_many_body_equation import (  # noqa: E402
        check_blocks, check_blocks_spin_summed)
    build_blocks()
    print("block term counts:")
    for n_, (kind, xn, rr, ss, is_dag, lhs, terms) in enumerate(BLOCKS, 1):
        print(f"  {n_:2d}. {len(terms):3d} terms, ranks {ranks_of(terms)}"
              f"  {lhs[:70]}")
    print()
    check_blocks()
    check_blocks_spin_summed()
    write_tex()
    compile_tex(TEX_PATH)
