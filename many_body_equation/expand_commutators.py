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

and applying the simplify rules of readme.md:

    1. resolve_delta:  the index-space delta rules
       (p,q,r,s whole space;  i,j,k,l occupied;  a,b,c,d virtual;
       delta(whole, vir) -> the whole-space index becomes the virtual one;
       delta(whole, occ) -> occupied;  delta(occ, vir) -> 0;
       same-space dummies contract;  delta(whole, whole) is kept);
    2. canonicalization:  the operator is invariant under a simultaneous
       permutation of its index pairs, with NO sign (Operator.canonical,
       applied in merge);
    3. dummy indices:  every summed index (including the operator's free
       indices, which are the summation labels of the operator basis sum)
       may be renamed by an unused letter of the same space;
    4. merging:  two terms are equal iff related by exact identities
       (renaming, the pair symmetries of t and g, real symmetric h/f and
       pair-exchange-symmetric g, and the diagonal pair-permutation
       symmetry of the spin-summed E -- the two rows may only be
       permuted together, with NO sign);  the canonical form is the
       orbit minimum (canonicalize).

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
    Tensor, Operator, SPACE, FAMILY, canonicalize, term_latex,
    terms_latex, pack_rows, ranks_of -- the term machinery

Every expanded block is verified numerically against the direct (nested)
matrix commutator with random real tensors of these symmetry classes,
twice: on Jordan-Wigner spin-orbital matrices (4 modes, 2 occupied + 2
virtual) and with genuinely spin-summed operators (4 spatial orbitals =
8 spin orbitals) -- the spin-orbital check alone cannot see rule 4d,
because single-row antisymmetry IS exact for a spin-orbital string.  The
blocks are then written to many_body_equations.tex.
"""

import itertools
import shutil
import subprocess
import sys
from pathlib import Path

from sympy import Rational, symbols
from sympy.physics.quantum import Commutator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "generalized_wick_theorem"))
sys.path.insert(0, str(HERE.parent / "commutator"))
from generalized_wick_commutator import (  # noqa: E402
    Operator, _jw_matrices, _matmul, commutator)
import generate_commutators as gc  # noqa: E402

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

# extra letters per space, in case an expansion needs more than the twelve
# names above (their space must be known for the index-space rules)
FAMILY = {"g": symbols("p q r s u v w x"),
          "o": symbols("i j k l m n"),
          "v": symbols("a b c d e f")}
for sp, letters in FAMILY.items():
    for x in letters:
        SPACE[x] = sp


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

    def latex(self):
        up = "".join(str(x) for x in self.upper)
        lo = "".join(str(x) for x in self.lower)
        if self.star:
            up += " *"
        return "%s^{%s}_{%s}" % (self.name, up, lo)


# ---------------------------------------------------------------------------
# terms: (Rational coeff, (Tensor, ...), Operator, leftover deltas)
# ---------------------------------------------------------------------------
def resolve_delta(coeff, tensors, deltas, op):
    """Simplify rule 1 of readme.md: the index-space delta rules.
    Returns the term or None (zero)."""
    subs = {}
    leftover = []
    for x, y in deltas:
        sx, sy = SPACE[x], SPACE[y]
        if {sx, sy} == {"o", "v"}:
            return None                      # delta(occ, vir) = 0
        if sx == "g" and sy != "g":
            subs[x] = y
        elif sy == "g" and sx != "g":
            subs[y] = x
        elif sx == sy:                       # same-space dummies: contract
            subs[y] = x
        else:                                # delta(whole, whole): kept
            leftover.append((x, y))
    if subs:
        tensors = tuple(t.substitute(subs) for t in tensors)
        op = Operator([subs.get(x, x) for x in op.upper_index],
                      [subs.get(x, x) for x in op.lower_index])
    # Pauli: a repeated index within one row makes the operator vanish
    if (len(set(op.upper_index)) != op.rank
            or len(set(op.lower_index)) != op.rank):
        return None
    return (coeff, tensors, op, tuple(leftover))


def merge(terms):
    """Merge like terms (simplify rule 2 of readme.md: the operator is
    invariant under a simultaneous permutation of its index pairs, with
    no sign; canonical order = pairs sorted by the upper index, via
    ``Operator.canonical``; tensors as a multiset)."""
    merged = {}
    keep = {}
    for coeff, tensors, op, leftover in terms:
        op = op.canonical()
        sigs = tuple(sorted((t.signature() for t in tensors), key=str))
        dlts = tuple(sorted(leftover, key=str))
        key = (op.upper_index, op.lower_index, sigs, dlts)
        merged[key] = merged.get(key, Rational(0)) + coeff
        keep.setdefault(key, (tensors, op, tuple(leftover)))
    out = []
    for key, coeff in merged.items():
        if coeff:
            tensors, op, leftover = keep[key]
            out.append((coeff, tensors, op, leftover))
    return out


def comm_terms(terms_a, terms_b):
    """Commutator of two sums of terms, expanded with the generalized Wick
    theorem and simplified with the index-space rules (rule 1) and the
    pair-order canonicalization (rule 2, inside ``merge``)."""
    out = []
    for ca, ta, opa, _ in terms_a:
        for cb, tb, opb, _ in terms_b:
            for (deltas, up, lo), c in commutator(opa, opb).items():
                term = resolve_delta(ca * cb * c, ta + tb, deltas,
                                     Operator(up, lo))
                if term is not None:
                    out.append(term)
    return merge(out)


def dagger_terms(terms):
    """Term-by-term Hermitian conjugate of an expanded block."""
    out = []
    for coeff, tensors, op, leftover in terms:
        if leftover:
            raise ValueError("dagger with unresolved deltas")
        out.append((coeff, tuple(t.dagger() for t in tensors),
                    Operator(op.lower_index, op.upper_index), ()))
    return merge(out)


def ranks_of(terms):
    return sorted({op.rank for _, _, op, _ in terms})


# ---------------------------------------------------------------------------
# merging (simplify rules 3 and 4 of readme.md): two terms are equal -- and
# merge -- iff they are related by identities that are exact in the
# spin-integrated, real-coefficient convention:
#   * renaming of any summed index within its space (rule 3).  ALL indices
#     are summed: the free indices of the operator are the summation labels
#     of the operator basis sum, so they can be renamed like tensor dummies;
#   * the pair symmetry of the amplitudes, t^{ab}_{ij} = t^{ba}_{ji} and
#     g^{pq}_{rs} = g^{qp}_{sr} (simultaneous swap of both index pairs);
#   * real coefficients: h and f symmetric (h^q_p = h^p_q, f^q_p = f^p_q)
#     and g symmetric under pair exchange (g^{pq}_{rs} = g^{rs}_{pq});
#   * the diagonal pair-permutation symmetry of the spin-summed E: the two
#     index rows may only be permuted TOGETHER, with NO sign
#     (E^{qp...}_{rs...} = E^{pq...}_{sr...}).  A single row alone is NOT a
#     symmetry of the spin-summed operator (that antisymmetry belongs to
#     the spin-orbital string, not to the spin-integrated E).
# No index antisymmetry of t or g is used.  The canonical form of a term is
# the minimum over this orbit; all orbit moves carry sign +1.
# ---------------------------------------------------------------------------
def _term_orbit_min(tensors, op):
    """Canonical (min) form of a term over the exact-identity orbit.

    Returns (key, sign, (canonical tensors, canonical operator)) where key
    is hashable and sign converts the original term into the canonical
    representative.  sign is always +1: no orbit move carries a sign in
    the spin-integrated convention."""
    allidx = sorted({x for t in tensors for x in t.upper + t.lower}
                    | set(op.upper_index) | set(op.lower_index), key=str)
    by_space = {}
    for x in allidx:
        by_space.setdefault(SPACE[x], []).append(x)
    n = op.rank
    # diagonal pair permutation of the two rows of E: exact, sign +1
    op_opts = [(1, perm, perm) for perm in itertools.permutations(range(n))]
    t_opts = []
    for t in tensors:
        if t.name in ("f", "h", "D") and len(t.upper) == 1:
            # h, f symmetric; the 1-RDM of a real reference is symmetric
            t_opts.append(((t.upper, t.lower), (t.lower, t.upper)))
        elif len(t.upper) == 2:
            sw = ((t.upper[1], t.upper[0]), (t.lower[1], t.lower[0]))
            if t.name in ("g", "D"):
                # pair swap + pair exchange (real, Hermitian)
                t_opts.append(((t.upper, t.lower), sw, (t.lower, t.upper),
                               (sw[1], sw[0])))
            else:
                t_opts.append(((t.upper, t.lower), sw))
        else:
            t_opts.append(((t.upper, t.lower),))
    best = None
    best_sign = 1
    best_term = None
    for combo in itertools.product(
            *(itertools.permutations(xs) for xs in by_space.values())):
        mapping = {}
        for xs, tgt in zip(by_space.values(), combo):
            mapping.update(zip(xs, tgt))
        for osign, uperm, lperm in op_opts:
            mup = tuple(mapping[op.upper_index[k]] for k in uperm)
            mlo = tuple(mapping[op.lower_index[k]] for k in lperm)
            for tcombo in itertools.product(*t_opts):
                ts = tuple(sorted(
                    (t.name,
                     tuple(str(mapping[x]) for x in u),
                     tuple(str(mapping[x]) for x in l))
                    for t, (u, l) in zip(tensors, tcombo)))
                key = ((tuple(str(x) for x in mup),
                        tuple(str(x) for x in mlo)), ts)
                if best is None or key < best:
                    best = key
                    best_sign = osign
                    best_term = (
                        tuple(Tensor(t.name, [mapping[x] for x in u],
                                     [mapping[x] for x in l])
                              for t, (u, l) in zip(tensors, tcombo)),
                        Operator(list(mup), list(mlo)))
    return best, best_sign, best_term


def _normalize_letters(tensors, op):
    """Replace every index by the next canonical letter of its space in
    encounter order (operator upper, operator lower, then tensors), so that
    all terms draw on the same letter pool before the orbit minimum."""
    scan = list(op.upper_index) + list(op.lower_index)
    for t in tensors:
        scan += list(t.upper) + list(t.lower)
    mapping, count = {}, {}
    for x in scan:
        if x not in mapping:
            sp = SPACE[x]
            mapping[x] = FAMILY[sp][count.get(sp, 0)]
            count[sp] = count.get(sp, 0) + 1
    return (tuple(t.substitute(mapping) for t in tensors),
            Operator([mapping[x] for x in op.upper_index],
                     [mapping[x] for x in op.lower_index]))


def canonicalize(terms):
    """Merge equal terms (simplify rule 4 of readme.md) over the
    exact-identity orbit; summed indices are renamed onto the same letter
    pool first (rule 3, ``_normalize_letters``)."""
    merged = {}
    keep = {}
    for coeff, tensors, op, leftover in terms:
        if leftover:
            raise ValueError("canonicalize with unresolved deltas")
        tensors, op = _normalize_letters(tensors, op)
        key, sign, canon = _term_orbit_min(tensors, op)
        if key not in merged:
            merged[key] = Rational(0)
            keep[key] = canon
        merged[key] += coeff * sign
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

STATIC_POOL = {gc.H: H_OP, gc.F: F_OP, gc.V: V_OP}


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
    operators (inner excitation first).  The result is merged (``merge``)
    but not canonicalized.
    """
    if counter is None:
        counter = [0]
    if isinstance(expr, Commutator):
        return comm_terms(expand_expr(expr.args[0], counter),
                          expand_expr(expr.args[1], counter))
    if gc.is_dagger(expr):
        return dagger_terms(expand_expr(expr.args[0], counter))
    if expr == gc.T1 or expr == gc.T2:
        rank = 1 if expr == gc.T1 else 2
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
        return 1 if t == gc.T1 else 2
    if kind == "first":
        x, t = expr.args
        return ("h" if x == gc.H else "V", t_rank(t), None, False)
    inner, ts = expr.args
    is_dag = gc.is_dagger(inner)
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
        terms   [(coeff, tensors, op, leftover), ...], canonicalized
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
        for expr in gc.commutator_blocks()[key]:
            x_name, rr, ss, is_dag = _block_meta(kind, expr)
            terms = canonicalize(expand_expr(expr))
            out.append({"kind": kind, "expr": expr, "lhs": gc.ct_latex(expr),
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
                 % ("".join(str(u) for u in op.upper_index),
                    "".join(str(l) for l in op.lower_index)))
    if coeff < 0:
        return "- " + body
    if first_term:
        return body
    return "+ " + body


def terms_latex(terms):
    def key(term):
        _, tensors, op, _ = term
        return (op.rank,
                [str(u) for u in op.upper_index],
                [str(l) for l in op.lower_index],
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
    add(r"p,q,r,s &:\ \text{whole space}, &")
    add(r"i,j,k,l &:\ \text{occupied}, &")
    add(r"a,b,c,d &:\ \text{virtual}, \\")
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
    add(r"with $\hat{h}$, $\hat{F}$ and $\hat{V}$ Hermitian; the second"
        r" excitation of the nested commutators carries the dummy indices"
        r" $c,d$ and $k,l$."
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
    add(r"Equal terms are merged (\texttt{readme.md}, rule 4) over every"
        r" identity that is exact in this convention: renaming of any summed"
        r" index within its space --- the free indices of the operator are"
        r" the summation labels of the operator basis sum, so they can be"
        r" renamed like tensor dummies --- the pair symmetries of $g$ and"
        r" $t$ above, and the diagonal pair-permutation symmetry of the"
        r" spin-summed $\hat{E}$: the two index rows may only be permuted"
        r" together, with no sign"
        r" ($\hat{E}^{qp\ldots}_{rs\ldots} = \hat{E}^{pq\ldots}_{sr\ldots}$);"
        r" a single row alone is \emph{not} a symmetry of the spin-summed"
        r" operator.")
    add("")
    add(r"Every block is verified numerically against the direct nested"
        r" commutator on Jordan--Wigner matrices, twice: with spin-orbital"
        r" operators (4 modes, 2 occupied $+$ 2 virtual) and with the"
        r" genuinely spin-summed operators (4 spatial orbitals $=$ 8 spin"
        r" orbitals); the second check is the one that sees the merging"
        r" rules.")
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


# ---------------------------------------------------------------------------
# numerical verification: Jordan-Wigner matrices, 4 modes (2 occ + 2 vir)
# ---------------------------------------------------------------------------
def check_blocks():
    import itertools

    import numpy as np

    rng = np.random.default_rng(20261010)
    cre, ann = _jw_matrices(4)

    def mat_sub(ma, mb):
        out = dict(ma)
        for key, v in mb.items():
            out[key] = out.get(key, 0) - v
        return {k: v for k, v in out.items() if v}

    cache = {}

    def e_matrix(up_vals, lo_vals):
        key = (tuple(up_vals), tuple(lo_vals))
        if key not in cache:
            m = {(n_, n_): 1 for n_ in range(16)}
            for u in up_vals:
                m = _matmul(m, cre[u])
            for lo_ in reversed(lo_vals):
                m = _matmul(m, ann[lo_])
            cache[key] = m
        return cache[key]

    def comm_mat(ma, mb):
        return mat_sub(_matmul(ma, mb), _matmul(mb, ma))

    def dagger_mat(m):
        return {(c_, r_): np.conj(v) for (r_, c_), v in m.items()}

    # random real tensors, spin-integrated convention: h and f symmetric; g
    # symmetric under the simultaneous swap of both index pairs and under
    # pair exchange, g^{pq}_{rs} = g^{qp}_{sr} = g^{rs}_{pq}; t2 symmetric
    # under the simultaneous swap t^{ab}_{ij} = t^{ba}_{ji}.  No index
    # antisymmetry is assumed here or anywhere in the expansion.
    h1 = rng.normal(size=(4, 4))
    h_val = h1 + h1.T
    h2 = rng.normal(size=(4, 4))
    f_val = h2 + h2.T
    g_val = rng.normal(size=(4, 4, 4, 4))
    g_val = g_val + g_val.transpose(1, 0, 3, 2)
    g_val = g_val + g_val.transpose(2, 3, 0, 1)
    t1_val = rng.normal(size=(4, 4))
    t2_val = rng.normal(size=(4, 4, 4, 4))
    t2_val = t2_val + t2_val.transpose(1, 0, 3, 2)

    def tensor_value(t, val):
        up = [val[x] for x in t.upper]
        lo = [val[x] for x in t.lower]
        if t.name == "h":
            out = h_val[up[0], lo[0]]
        elif t.name == "f":
            out = f_val[up[0], lo[0]]
        elif t.name == "g":
            out = g_val[up[0], up[1], lo[0], lo[1]]
        elif len(t.upper) == 1:
            out = t1_val[up[0], lo[0]]
        else:
            out = t2_val[up[0], up[1], lo[0], lo[1]]
        return np.conj(out) if t.star else out

    def op_matrix_direct(pool):
        """Matrix of a one-term operator pool (F_OP, V_OP, ...)."""
        coeff, tensors, op, _ = pool[0]
        syms = list(op.upper_index + op.lower_index)
        m = {}
        for vals in itertools.product(*(RANGE[SPACE[x]] for x in syms)):
            val = dict(zip(syms, vals))
            factor = complex(coeff)
            for t in tensors:
                factor = factor * tensor_value(t, val)
            for key, v in e_matrix([val[x] for x in op.upper_index],
                                   [val[x] for x in op.lower_index]).items():
                m[key] = m.get(key, 0) + factor * v
        return {k: v for k, v in m.items() if abs(v) > 1e-12}

    def terms_matrix(terms):
        """Matrix of an expanded block (sum of resolved terms)."""
        m = {}
        for coeff, tensors, op, leftover in terms:
            free = list(op.upper_index + op.lower_index)
            dummies = sorted({x for t in tensors
                              for x in t.upper + t.lower} - set(free),
                             key=str)
            accum = {}
            for vals in itertools.product(
                    *(RANGE[SPACE[x]] for x in free + dummies)):
                val = dict(zip(free + dummies, vals))
                if any(val[x] != val[y] for x, y in leftover):
                    continue
                factor = complex(coeff)
                for t in tensors:
                    factor = factor * tensor_value(t, val)
                if factor:
                    key = tuple(vals[:len(free)])
                    accum[key] = accum.get(key, 0) + factor
            for key, factor in accum.items():
                em = e_matrix(key[:op.rank], key[op.rank:])
                for pos, v in em.items():
                    m[pos] = m.get(pos, 0) + factor * v
        return {k: v for k, v in m.items() if abs(v) > 1e-9}

    def max_dev(ma, mb):
        keys = set(ma) | set(mb)
        return max((abs(ma.get(k, 0) - mb.get(k, 0)) for k in keys),
                   default=0.0)

    def block_matrix(expr):
        """Direct matrix of a commutator-stage expression (nested matrix
        commutators and daggers of the elementary-operator matrices)."""
        if isinstance(expr, Commutator):
            return comm_mat(block_matrix(expr.args[0]),
                            block_matrix(expr.args[1]))
        if gc.is_dagger(expr):
            return dagger_mat(block_matrix(expr.args[0]))
        if expr == gc.T1 or expr == gc.T2:
            return op_matrix_direct(
                _excitation_pool(1 if expr == gc.T1 else 2, 0))
        return op_matrix_direct(STATIC_POOL[expr])

    print("numerical checks (Jordan-Wigner, 4 modes, occ={0,1} vir={2,3}):")
    ok = True
    for n, blk in enumerate(expanded_blocks(), 1):
        dev = max_dev(terms_matrix(blk["terms"]), block_matrix(blk["expr"]))
        status = "OK " if dev < 1e-9 else "FAIL"
        ok &= dev < 1e-9
        print(f"  [{status}] block {n:2d}: max|Delta| = {dev:.3e}")
    if not ok:
        raise SystemExit("many-body expansion check failed")
    print()


# ---------------------------------------------------------------------------
# route cross-check: coefficient tensors, raw expansion vs merged expansion
# ---------------------------------------------------------------------------
def check_tensor_routes():
    """For every block, build the coefficient tensor of each operator rank
    by explicit contraction with random real tensors, twice: from the raw
    expansion (no merging at any stage) and from the merged terms above.
    The coefficient of a basis operator is only defined up to the diagonal
    pair-permutation symmetry of the spin-summed E, so both tensors are
    symmetrized over the simultaneous permutation of the index pairs
    before the entry-by-entry comparison."""
    import math

    import numpy as np

    rng = np.random.default_rng(20261011)
    h1 = rng.normal(size=(4, 4))
    h_val = h1 + h1.T
    h2 = rng.normal(size=(4, 4))
    f_val = h2 + h2.T
    g_val = rng.normal(size=(4, 4, 4, 4))
    g_val = g_val + g_val.transpose(1, 0, 3, 2)
    g_val = g_val + g_val.transpose(2, 3, 0, 1)
    t1_val = rng.normal(size=(4, 4))
    t2_val = rng.normal(size=(4, 4, 4, 4))
    t2_val = t2_val + t2_val.transpose(1, 0, 3, 2)

    def tensor_value(t, val):
        up = [val[x] for x in t.upper]
        lo = [val[x] for x in t.lower]
        if t.name == "h":
            return h_val[up[0], lo[0]]
        if t.name == "f":
            return f_val[up[0], lo[0]]
        if t.name == "g":
            return g_val[up[0], up[1], lo[0], lo[1]]
        if len(t.upper) == 1:
            return t1_val[up[0], lo[0]]
        return t2_val[up[0], up[1], lo[0], lo[1]]

    def coeff_tensor(terms, rank):
        C = np.zeros((4,) * (2 * rank))
        for coeff, tensors, op, _ in terms:
            if op.rank != rank:
                continue
            free = list(op.upper_index + op.lower_index)
            dums = sorted({x for t in tensors for x in t.upper + t.lower}
                          - set(free), key=str)
            for vals in itertools.product(
                    *(RANGE[SPACE[x]] for x in free + dums)):
                val = dict(zip(free + dums, vals))
                factor = float(coeff)
                for t in tensors:
                    factor *= tensor_value(t, val)
                if factor:
                    C[tuple(vals[:len(free)])] += factor
        perms = list(itertools.permutations(range(rank)))
        A = np.zeros_like(C)
        for perm in perms:
            A = A + C.transpose(list(perm) + [rank + k for k in perm])
        return A / math.factorial(rank)

    print("route cross-check (coefficient tensors, raw vs merged,"
          " symmetrized over the diagonal pair permutations):")
    ok = True
    for n, blk in enumerate(expanded_blocks(), 1):
        raw = expand_expr(blk["expr"])   # fresh expansion, never canonicalized
        terms = blk["terms"]
        for rank in ranks_of(raw + terms):
            dev = float(np.max(np.abs(
                coeff_tensor(raw, rank) - coeff_tensor(terms, rank))))
            status = "OK " if dev < 1e-9 else "FAIL"
            ok &= dev < 1e-9
            print(f"  [{status}] block {n:2d} rank {rank}: "
                  f"max|Delta| = {dev:.3e}")
    if not ok:
        raise SystemExit("route cross-check failed")
    print()


# ---------------------------------------------------------------------------
# numerical verification with the spin-summed E (unitary group generator):
# the merging must be exact in the spin-integrated formalism itself
# ---------------------------------------------------------------------------
def check_blocks_spin_summed():
    """Rebuild every block with genuinely spin-summed operators

        Ef^{u1..un}_{l1..ln} = sum_{s1..sn} a+_{u1 s1} ... a+_{un sn}
                               a_{ln sn} ... a_{l1 s1}

    (4 spatial orbitals = 8 spin orbitals, Jordan-Wigner, dense matrices)
    and compare against the direct nested commutator of the spin-summed
    elementary operators.

    The spin-orbital check (check_blocks) cannot see the difference: the
    single-row antisymmetry of a spin-orbital string is exact there.  For
    the spin-summed operator the only row-level symmetry is the diagonal
    pair permutation (E^{qp..}_{rs..} = E^{pq..}_{sr..}, no sign); the
    sanity assertions below verify both statements.  The raw expansion
    (rules 1 and 2 only, never canonicalized) is checked alongside every
    block, so a failure is attributed to the merging (rule 4)
    unambiguously.
    """
    import numpy as np

    rng = np.random.default_rng(20261012)
    cre, ann = _jw_matrices(8)
    dim = 256
    eye = np.eye(dim)

    def dense(m):
        a = np.zeros((dim, dim))
        for (r_, c_), v in m.items():
            a[r_, c_] = v
        return a

    cre = [dense(m) for m in cre]
    ann = [dense(m) for m in ann]

    e_cache = {}

    def e_sf(up_vals, lo_vals):
        """Spin-summed E matrix: sum over spins of the paired string."""
        key = (tuple(up_vals), tuple(lo_vals))
        if key not in e_cache:
            m = np.zeros((dim, dim))
            rank = len(up_vals)
            for spins in itertools.product((0, 1), repeat=rank):
                mm = eye
                for u, sg in zip(up_vals, spins):
                    mm = mm @ cre[2 * u + sg]
                for l_, sg in zip(reversed(lo_vals), reversed(spins)):
                    mm = mm @ ann[2 * l_ + sg]
                m += mm
            e_cache[key] = m
        return e_cache[key]

    # the exact symmetry of the spin-summed E^{pq}_{rs}: the diagonal pair
    # swap only; a single row alone is NOT (anti)symmetric
    diag = np.max(np.abs(e_sf((1, 0), (0, 1)) - e_sf((0, 1), (1, 0))))
    anti = np.max(np.abs(e_sf((1, 0), (0, 1)) + e_sf((0, 1), (0, 1))))
    print("spin-summed check (4 spatial orbitals, 8 spin orbitals):")
    print(f"  E^qp_rs = +E^pq_sr (diagonal pair swap): "
          f"max|Delta| = {diag:.3e}")
    print(f"  E^qp_rs = -E^pq_rs (row antisymmetry):   "
          f"max|Delta| = {anti:.3e}  (must NOT vanish)")
    if diag > 1e-12 or anti < 1e-3:
        raise SystemExit("spin-summed E symmetry sanity check failed")

    # random real spin-integrated tensors, same symmetry classes as in
    # check_blocks
    h1 = rng.normal(size=(4, 4))
    h_val = h1 + h1.T
    h2 = rng.normal(size=(4, 4))
    f_val = h2 + h2.T
    g_val = rng.normal(size=(4, 4, 4, 4))
    g_val = g_val + g_val.transpose(1, 0, 3, 2)
    g_val = g_val + g_val.transpose(2, 3, 0, 1)
    t1_val = rng.normal(size=(4, 4))
    t2_val = rng.normal(size=(4, 4, 4, 4))
    t2_val = t2_val + t2_val.transpose(1, 0, 3, 2)

    t_cache = {}

    def tensor_array(t):
        """Dense array of one coefficient tensor over its index ranges."""
        if t.signature() not in t_cache:
            shape = tuple(len(RANGE[SPACE[x]]) for x in t.upper + t.lower)
            arr = np.zeros(shape)
            for pos in itertools.product(*(range(n_) for n_ in shape)):
                val = {}
                for k, x in enumerate(t.upper + t.lower):
                    val[x] = RANGE[SPACE[x]][pos[k]]
                up = [val[x] for x in t.upper]
                lo = [val[x] for x in t.lower]
                if t.name == "h":
                    arr[pos] = h_val[up[0], lo[0]]
                elif t.name == "f":
                    arr[pos] = f_val[up[0], lo[0]]
                elif t.name == "g":
                    arr[pos] = g_val[up[0], up[1], lo[0], lo[1]]
                elif len(t.upper) == 1:
                    arr[pos] = t1_val[up[0], lo[0]]
                else:
                    arr[pos] = t2_val[up[0], up[1], lo[0], lo[1]]
            t_cache[t.signature()] = arr
        return t_cache[t.signature()]

    letters = list(FAMILY["g"] + FAMILY["o"] + FAMILY["v"])
    sub = {x: chr(ord("A") + k) for k, x in enumerate(letters)}

    def term_coeff_array(tensors, free):
        """Coefficient array over the operator's free indices, dummies
        contracted (einsum over the tensor product)."""
        ops, subs = [], []
        for t in tensors:
            subs.append("".join(sub[x] for x in t.upper + t.lower))
            ops.append(tensor_array(t))
        out = "".join(sub[x] for x in free)
        return np.einsum(",".join(subs) + "->" + out, *ops)

    def terms_matrix_sf(terms):
        m = np.zeros((dim, dim))
        for coeff, tensors, op, leftover in terms:
            assert not leftover
            free = list(op.upper_index + op.lower_index)
            C = float(coeff) * term_coeff_array(tensors, free)
            for pos in itertools.product(
                    *(range(len(RANGE[SPACE[x]])) for x in free)):
                if abs(C[pos]) < 1e-14:
                    continue
                vals = [RANGE[SPACE[x]][pos[k]] for k, x in enumerate(free)]
                m += C[pos] * e_sf(vals[:op.rank], vals[op.rank:])
        return m

    mats = {gc.H: terms_matrix_sf(H_OP), gc.F: terms_matrix_sf(F_OP),
            gc.V: terms_matrix_sf(V_OP),
            gc.T1: terms_matrix_sf(_excitation_pool(1, 0)),
            gc.T2: terms_matrix_sf(_excitation_pool(2, 0))}

    def block_matrix_sf(expr):
        if isinstance(expr, Commutator):
            a = block_matrix_sf(expr.args[0])
            b = block_matrix_sf(expr.args[1])
            return a @ b - b @ a
        if gc.is_dagger(expr):
            return block_matrix_sf(expr.args[0]).conj().T
        return mats[expr]

    ok = True
    for n, blk in enumerate(expanded_blocks(), 1):
        direct = block_matrix_sf(blk["expr"])
        dev = float(np.max(np.abs(terms_matrix_sf(blk["terms"]) - direct)))
        raw = expand_expr(blk["expr"])
        dev_raw = float(np.max(np.abs(terms_matrix_sf(raw) - direct)))
        status = "OK " if dev < 1e-9 else "FAIL"
        status_raw = "OK " if dev_raw < 1e-9 else "FAIL"
        ok &= dev < 1e-9 and dev_raw < 1e-9
        print(f"  [{status}] block {n:2d} merged max|Delta| = {dev:9.3e}"
              f"   [{status_raw}] raw max|Delta| = {dev_raw:9.3e}")
    if not ok:
        raise SystemExit("spin-summed check failed")
    print()


# ---------------------------------------------------------------------------
def compile_tex():
    if shutil.which("latexmk") is None:
        print("latexmk not found, skipping PDF build")
        return
    proc = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", TEX_PATH.name],
        cwd=HERE, capture_output=True, text=True)
    if proc.returncode == 0:
        print(f"built {TEX_PATH.with_suffix('.pdf')}")
    else:
        print("latexmk failed:")
        print("\n".join(proc.stdout.splitlines()[-15:]))
        raise SystemExit(1)


if __name__ == "__main__":
    build_blocks()
    print("block term counts:")
    for n_, (kind, xn, rr, ss, is_dag, lhs, terms) in enumerate(BLOCKS, 1):
        print(f"  {n_:2d}. {len(terms):3d} terms, ranks {ranks_of(terms)}"
              f"  {lhs[:70]}")
    print()
    check_blocks()
    check_tensor_routes()
    check_blocks_spin_summed()
    write_tex()
    compile_tex()
