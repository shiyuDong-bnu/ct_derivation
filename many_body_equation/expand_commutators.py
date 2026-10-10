#!/usr/bin/env python3
"""Expand all CT commutators into many-body equations.

Every commutator block of the ``commutator`` stage (4 first-order blocks and
16 second-order blocks) is expanded with the generalized Wick theorem of
``generalized_wick_theorem/generalized_wick_commutator.py``, substituting

    F  = f^q_p E^p_q                    (p, q general)
    V  = (1/2) g^{pq}_{rs} E^{rs}_{pq}  (p, q, r, s general)
    T1 = t^a_i E^a_i                    (a virtual, i occupied)
    T2 = (1/2) t^{ab}_{ij} E^{ab}_{ij}

and applying the index-space rules of readme.md:

    p,q,r,s  whole space;  i,j,k,l  occupied;  a,b,c,d  virtual
    delta(whole, vir) -> the whole-space index becomes the virtual one
    delta(whole, occ) -> the whole-space index becomes the occupied one
    delta(occ, vir)   -> 0

The truncation is by perturbation order (Moller-Plesset counting F ~ O(1),
V ~ O(lambda), T ~ O(lambda)), correct through O(lambda^2): the full
first-order commutators [F, A], [V, A] and the Fock half of the second
order, (1/2)[[F, A], A].  The valence double commutator blocks
[[V, Tr], Ts] are O(lambda^3) and are NOT expanded.

For the conjugate blocks [X_r^dagger, T_s] the expanded first-order block is
conjugated term by term: E^{u..}_{l..} -> E^{l..}_{u..}, f and g Hermitian
(f^q_p -> f^p_q, g^{pq}_{rs} -> g^{rs}_{pq}); the coefficients are real, so
t is unchanged.

Terms are merged over every identity that is exact in this convention:
renaming of any summed index (the free indices of the operator are the
summation labels of the operator basis sum, so they can be renamed like
tensor dummies), the spin-integrated pair symmetry t^{ab}_{ij} = t^{ba}_{ji}
and g^{pq}_{rs} = g^{qp}_{sr}, real symmetric f, real g with
g^{pq}_{rs} = g^{rs}_{pq}, and the exact fermionic row antisymmetry of the
E operator.  No index antisymmetry of t or g is used anywhere.

Every expanded block is verified numerically against the direct (nested)
matrix commutator on Jordan-Wigner matrices (4 modes, 2 occupied + 2
virtual) with random real tensors of these symmetry classes, then written
to many_body_equations.tex.
"""

import itertools
import shutil
import subprocess
import sys
from pathlib import Path

from sympy import Rational, symbols

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "generalized_wick_theorem"))
from generalized_wick_commutator import (  # noqa: E402
    Operator, _jw_matrices, _matmul, commutator)

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
        if self.name in ("f", "g"):      # Hermitian: swap the index pair(s)
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
def resolve(coeff, tensors, deltas, op):
    """Apply the index-space delta rules.  Returns the term or None (zero)."""
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
    """Merge like terms (canonical operator order, tensors as a multiset)."""
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
    theorem and simplified with the index-space rules."""
    out = []
    for ca, ta, opa, _ in terms_a:
        for cb, tb, opb, _ in terms_b:
            for (deltas, up, lo), c in commutator(opa, opb).items():
                term = resolve(ca * cb * c, ta + tb, deltas, Operator(up, lo))
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
# canonical merging (readme rule e): two terms are equal -- and merge -- iff
# they are related by identities that are exact in the spin-integrated,
# real-coefficient convention:
#   * renaming of any summed index within its space.  ALL indices are
#     summed: the free indices of the operator are the summation labels of
#     the operator basis sum, so they can be renamed like tensor dummies;
#   * the pair symmetry of the amplitudes, t^{ab}_{ij} = t^{ba}_{ji} and
#     g^{pq}_{rs} = g^{qp}_{sr} (simultaneous swap of both index pairs);
#   * real coefficients: f symmetric (f^q_p = f^p_q) and g symmetric under
#     pair exchange (g^{pq}_{rs} = g^{rs}_{pq});
#   * the exact fermionic row antisymmetry of the E operator.
# No index antisymmetry of t or g is used.  The canonical form of a term is
# the minimum over this orbit; merged coefficients carry the orbit sign.
# ---------------------------------------------------------------------------
def _parity(perm):
    inv = sum(1 for x in range(len(perm)) for y in range(x + 1, len(perm))
              if perm[x] > perm[y])
    return -1 if inv % 2 else 1


def _term_orbit_min(tensors, op):
    """Canonical (min) form of a term over the exact-identity orbit.

    Returns (key, sign, (canonical tensors, canonical operator)) where key
    is hashable and sign converts the original term into the canonical
    representative."""
    allidx = sorted({x for t in tensors for x in t.upper + t.lower}
                    | set(op.upper_index) | set(op.lower_index), key=str)
    by_space = {}
    for x in allidx:
        by_space.setdefault(SPACE[x], []).append(x)
    n = op.rank
    op_opts = [(_parity(up) * _parity(lo), up, lo)
               for up in itertools.permutations(range(n))
               for lo in itertools.permutations(range(n))]
    t_opts = []
    for t in tensors:
        if t.name in ("f", "D") and len(t.upper) == 1:
            # f symmetric; the 1-RDM of a real reference is symmetric
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
    """Merge equal terms (readme rule e) over the exact-identity orbit."""
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


F_OP = op_term(1, [Tensor("f", (q,), (p,))], [p], [q])
V_OP = op_term(Rational(1, 2), [Tensor("g", (p, q), (r, s))], [r, s], [p, q])
T1_OP = op_term(1, [Tensor("t", (a,), (i,))], [a], [i])
T2_OP = op_term(Rational(1, 2), [Tensor("t", (a, b), (i, j))], [a, b], [i, j])
T1B_OP = op_term(1, [Tensor("t", (c,), (k,))], [c], [k])
T2B_OP = op_term(Rational(1, 2), [Tensor("t", (c, d), (k, l))], [c, d],
                 [k, l])

TS = {1: T1_OP, 2: T2_OP}         # first excitation: a,b / i,j
TS_B = {1: T1B_OP, 2: T2B_OP}     # second excitation: c,d / k,l

# ---------------------------------------------------------------------------
# all commutator blocks
# ---------------------------------------------------------------------------
BLOCKS = []        # (kind, x_name, r, s, is_dag, lhs latex, expanded terms)


def build_blocks():
    first = {}
    for x_name, x_op in (("F", F_OP), ("V", V_OP)):
        for rr in (1, 2):
            first[x_name, rr] = canonicalize(comm_terms(x_op, TS[rr]))
            lhs = r"\left[\hat{%s},\, \hat{T}_{%d}\right]" % (x_name, rr)
            BLOCKS.append(("first", x_name, rr, None, False,
                           lhs, first[x_name, rr]))

    # second order: only the Fock blocks are O(lambda^2); the valence
    # double commutator [[V, Tr], Ts] is O(lambda^3) and is dropped by the
    # perturbation-order truncation.
    for x_name in ("F",):
        for rr in (1, 2):
            inner = first[x_name, rr]
            inner_dag = dagger_terms(inner)
            for ss in (1, 2):
                blk = canonicalize(comm_terms(inner, TS_B[ss]))
                lhs = (r"\left[\left[\hat{%s},\, \hat{T}_{%d}\right],\, "
                       r"\hat{T}_{%d}\right]" % (x_name, rr, ss))
                BLOCKS.append(("second", x_name, rr, ss, False, lhs, blk))
                blk_dag = canonicalize(comm_terms(inner_dag, TS_B[ss]))
                lhs = (r"\left[\left[\hat{%s},\, \hat{T}_{%d}\right]"
                       r"^{\dagger},\, \hat{T}_{%d}\right]"
                       % (x_name, rr, ss))
                BLOCKS.append(("second", x_name, rr, ss, True, lhs, blk_dag))


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
    add(r"All commutator blocks of the \texttt{commutator} stage expanded with"
        r" the generalized Wick theorem"
        r" (\texttt{generalized\_wick\_theorem/generalized\_wick\_commutator.py})."
        r" The truncation is by perturbation order (M\o ller--Plesset counting"
        r" $\hat{F} \sim O(1)$, $\hat{V} \sim O(\lambda)$,"
        r" $\hat{T} \sim O(\lambda)$), correct through $O(\lambda^2)$: the"
        r" first-order commutators $[\hat{F}, \hat{A}]$, $[\hat{V}, \hat{A}]$"
        r" and the Fock half of the second order,"
        r" $\tfrac{1}{2}[[\hat{F}, \hat{A}], \hat{A}]$.  The valence double"
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
    add(r"\hat{F} &= f^q_p\, \hat{E}^p_q, &")
    add(r"\hat{V} &= \tfrac{1}{2} g^{pq}_{rs}\, \hat{E}^{rs}_{pq}, &")
    add(r"\hat{T}_1 &= t^a_i\, \hat{E}^a_i, &")
    add(r"\hat{T}_2 &= \tfrac{1}{2} t^{ab}_{ij}\, \hat{E}^{ab}_{ij},")
    add(r"\end{align}")
    add(r"with $\hat{F}$, $\hat{V}$ Hermitian; the second excitation of the"
        r" nested commutators carries the dummy indices $c,d$ and $k,l$."
        r" The coefficients are real and follow the spin-integrated"
        r" convention: no index antisymmetry is used anywhere,"
        r" $g^{pq}_{rs}$ and $t^{ab}_{ij}$ carry only the simultaneous pair"
        r" swap $g^{pq}_{rs} = g^{qp}_{sr}$, $t^{ab}_{ij} = t^{ba}_{ji}$, and"
        r" $f^q_p = f^p_q$, $g^{pq}_{rs} = g^{rs}_{pq}$."
        r" Conjugate blocks are expanded by conjugating the first-order"
        r" result term by term ($\hat{E}^{u\ldots}_{l\ldots} \to"
        r" \hat{E}^{l\ldots}_{u\ldots}$, $f^q_p \to f^p_q$,"
        r" $g^{pq}_{rs} \to g^{rs}_{pq}$; $t$ is real and unchanged)."
        r" Blocks containing operators of rank higher than two are marked;"
        r" their cumulant expansion (and the $[\ldots]_{1,2}$ truncation)"
        r" is done separately.")
    add("")
    add(r"Equal terms are merged (\texttt{readme.md}, rule e) over every"
        r" identity that is exact in this convention: renaming of any summed"
        r" index within its space --- the free indices of the operator are"
        r" the summation labels of the operator basis sum, so they can be"
        r" renamed like tensor dummies --- the pair symmetries of $g$ and"
        r" $t$ above, and the exact fermionic row antisymmetry of"
        r" $\hat{E}$.")
    add("")

    sections = [("First-order commutators", 0, 4),
                ("Second order: Fock blocks", 4, 12)]
    for title, lo_i, hi_i in sections:
        add(r"\section{%s}" % title)
        for kind, x_name, rr, ss, is_dag, lhs, terms in BLOCKS[lo_i:hi_i]:
            lines = terms_latex(terms)
            big = [n for n in ranks_of(terms) if n > 2]
            marker = ""
            if big:
                names = {3: "three-body", 4: "four-body"}
                marker = (r"\quad (\text{%s terms})"
                          % " and ".join(names[n] for n in big))
            add(r"\begin{align}")
            rows = pack_rows(lhs, lines)
            for row in rows[:-1]:
                add(row + r" \\")
            add(rows[-1] + marker)
            add(r"\end{align}")
        add("")

    add(r"\section{Assembly}")
    add(r"With the blocks above, the transcorrelated Hamiltonian through"
        r" $O(\lambda^2)$ is")
    add(r"\begin{align}")
    add(r"\left[\hat{F},\, \hat{A}\right] &= \left[\hat{F},\, \hat{T}_1\right]"
        r" + \left[\hat{F},\, \hat{T}_2\right] + \mathrm{h.c.}, \\")
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

    # random real tensors, spin-integrated convention: f symmetric; g
    # symmetric under the simultaneous swap of both index pairs and under
    # pair exchange, g^{pq}_{rs} = g^{qp}_{sr} = g^{rs}_{pq}; t2 symmetric
    # under the simultaneous swap t^{ab}_{ij} = t^{ba}_{ji}.  No index
    # antisymmetry is assumed here or anywhere in the expansion.
    h = rng.normal(size=(4, 4))
    f_val = h + h.T
    g_val = rng.normal(size=(4, 4, 4, 4))
    g_val = g_val + g_val.transpose(1, 0, 3, 2)
    g_val = g_val + g_val.transpose(2, 3, 0, 1)
    t1_val = rng.normal(size=(4, 4))
    t2_val = rng.normal(size=(4, 4, 4, 4))
    t2_val = t2_val + t2_val.transpose(1, 0, 3, 2)

    def tensor_value(t, val):
        up = [val[x] for x in t.upper]
        lo = [val[x] for x in t.lower]
        if t.name == "f":
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

    mats = {"F": op_matrix_direct(F_OP), "V": op_matrix_direct(V_OP)}
    t_mats = {1: op_matrix_direct(T1_OP), 2: op_matrix_direct(T2_OP)}
    tb_mats = {1: op_matrix_direct(T1B_OP), 2: op_matrix_direct(T2B_OP)}
    first_mat = {(xn, rr): comm_mat(mats[xn], t_mats[rr])
                 for xn in ("F", "V") for rr in (1, 2)}

    print("numerical checks (Jordan-Wigner, 4 modes, occ={0,1} vir={2,3}):")
    ok = True
    for n, (kind, x_name, rr, ss, is_dag, lhs, terms) in enumerate(BLOCKS, 1):
        if kind == "first":
            want = first_mat[x_name, rr]
        else:
            base = first_mat[x_name, rr]
            if is_dag:
                base = dagger_mat(base)
            want = comm_mat(base, tb_mats[ss])
        dev = max_dev(terms_matrix(terms), want)
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
    The coefficient of a basis operator is only defined up to the row
    antisymmetry of E, so both tensors are antisymmetrized in each index
    row before the entry-by-entry comparison."""
    import math

    import numpy as np

    rng = np.random.default_rng(20261011)
    h = rng.normal(size=(4, 4))
    f_val = h + h.T
    g_val = rng.normal(size=(4, 4, 4, 4))
    g_val = g_val + g_val.transpose(1, 0, 3, 2)
    g_val = g_val + g_val.transpose(2, 3, 0, 1)
    t1_val = rng.normal(size=(4, 4))
    t2_val = rng.normal(size=(4, 4, 4, 4))
    t2_val = t2_val + t2_val.transpose(1, 0, 3, 2)

    def tensor_value(t, val):
        up = [val[x] for x in t.upper]
        lo = [val[x] for x in t.lower]
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
        for up in perms:
            for lo in perms:
                A = A + _parity(up) * _parity(lo) * C.transpose(
                    list(up) + [rank + k for k in lo])
        return A / math.factorial(rank) ** 2

    def raw_block(kind, x_name, rr, ss, is_dag):
        x_op = {"F": F_OP, "V": V_OP}[x_name]
        if kind == "first":
            return comm_terms(x_op, TS[rr])
        inner = comm_terms(F_OP, TS[rr])
        if is_dag:
            inner = dagger_terms(inner)
        return comm_terms(inner, TS_B[ss])

    print("route cross-check (coefficient tensors, raw vs merged,"
          " rows antisymmetrized):")
    ok = True
    for n, (kind, x_name, rr, ss, is_dag, lhs, terms) in enumerate(BLOCKS, 1):
        raw = raw_block(kind, x_name, rr, ss, is_dag)
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
    write_tex()
    compile_tex()
