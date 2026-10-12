#!/usr/bin/env python3
"""Cumulant expansion of the surviving three-body terms (final stage).

Following readmd.md: the three-body operators that survive the space
restriction are decomposed into two-body and one-body operators (plus
scalar terms) with density-matrix coefficients, using the spin-free
cumulant decomposition of
``cumulant_decomposition/spin_free/spin_free.tex``:

    E^{pqr}_{stu}  ≈  [51 terms with D (1-RDM) and D (2-RDM)]

The ``≈'' drops the normal-ordered three-body remainder and the
three-cumulant lambda^{pqr}_{stu}; for a single-determinant reference
lambda_3 = 0 and only the [..]_{1,2} truncation of the three-body
remainder remains.  Every density matrix carrying a virtual index
vanishes; density matrices with occupied or whole-space indices are kept
symbolic as D^i_j, D^{ij}_{kl}, ... (no delta-function substitution).

The formula is verified numerically before use: for a closed-shell
single-determinant reference on Jordan-Wigner matrices (4 spatial
orbitals, 8 spin orbitals), the identity

    E^{pqr}_{stu} - (Fermi-vacuum normal-ordered E^{pqr}_{stu})
        = [the 51 terms with the reference's density matrices]

is checked on random index assignments, and again per surviving index
pattern with the virtual rule applied (the killed terms vanish
identically for the single-determinant reference).  The checks live in
test/check_cumulant_expansion.py at the repo root and are run by this
script.

Blocks 4 ([V, T2]) and 12 ([[F, T2]^dag, T2]) contain the only surviving
three-body terms; their final form gathers the surviving <=2-body terms
plus the decomposed three-body pieces, merged over the exact-identity
orbit of ``expand_commutators``.  The other ten blocks are unchanged.
"""

import itertools
import sys
from pathlib import Path

from sympy import Rational, symbols

HERE = Path(__file__).resolve().parent
MBE = HERE.parent / "many_body_equation"
sys.path.insert(0, str(MBE))
sys.path.insert(0, str(HERE.parent / "space_restriction"))
sys.path.insert(0, str(HERE.parent / "utils"))

import expand_commutators as mbe  # noqa: E402
import space_restriction as sr  # noqa: E402
from latex_utils import compile_tex  # noqa: E402

TEX_PATH = HERE / "cumulant_expansion.tex"

Tensor = mbe.Tensor
Operator = mbe.Operator

# template symbols for E^{w1 w2 w3}_{w4 w5 w6}
W1, W2, W3, W4, W5, W6 = symbols("w1 w2 w3 w4 w5 w6")
H = Rational(1, 2)


def D1(u, l):
    return Tensor("D", (u,), (l,))


def D2(u1, u2, l1, l2):
    return Tensor("D", (u1, u2), (l1, l2))


# the 51-term spin-free decomposition of E^{w1 w2 w3}_{w4 w5 w6},
# transcribed from cumulant_decomposition/spin_free/spin_free.tex
# (coefficient, density-matrix factors, (E upper, E lower)); empty E = scalar
TEMPLATE = [
    # D . E2
    (1, (D1(W1, W4),), ((W2, W3), (W5, W6))),
    (-H, (D1(W1, W5),), ((W2, W3), (W4, W6))),
    (-H, (D1(W1, W6),), ((W2, W3), (W5, W4))),
    (-H, (D1(W2, W4),), ((W1, W3), (W5, W6))),
    (1, (D1(W2, W5),), ((W1, W3), (W4, W6))),
    (-H, (D1(W2, W6),), ((W1, W3), (W4, W5))),
    (-H, (D1(W3, W4),), ((W2, W1), (W5, W6))),
    (-H, (D1(W3, W5),), ((W1, W2), (W4, W6))),
    (1, (D1(W3, W6),), ((W1, W2), (W4, W5))),
    # D2 . E1
    (1, (D2(W1, W2, W4, W5),), ((W3,), (W6,))),
    (-H, (D2(W1, W2, W4, W6),), ((W3,), (W5,))),
    (-H, (D2(W1, W2, W6, W5),), ((W3,), (W4,))),
    (-H, (D2(W1, W3, W4, W5),), ((W2,), (W6,))),
    (1, (D2(W1, W3, W4, W6),), ((W2,), (W5,))),
    (-H, (D2(W1, W3, W5, W6),), ((W2,), (W4,))),
    (-H, (D2(W2, W3, W4, W6),), ((W1,), (W5,))),
    (1, (D2(W2, W3, W5, W6),), ((W1,), (W4,))),
    (-H, (D2(W3, W2, W4, W5),), ((W1,), (W6,))),
    # D . D . E1
    (-2, (D1(W1, W4), D1(W2, W5)), ((W3,), (W6,))),
    (-2, (D1(W1, W4), D1(W3, W6)), ((W2,), (W5,))),
    (-H, (D1(W1, W5), D1(W2, W6)), ((W3,), (W4,))),
    (-H, (D1(W1, W5), D1(W3, W4)), ((W2,), (W6,))),
    (-H, (D1(W1, W6), D1(W2, W4)), ((W3,), (W5,))),
    (-H, (D1(W1, W6), D1(W3, W5)), ((W2,), (W4,))),
    (-H, (D1(W2, W4), D1(W3, W5)), ((W1,), (W6,))),
    (-2, (D1(W2, W5), D1(W3, W6)), ((W1,), (W4,))),
    (-H, (D1(W2, W6), D1(W3, W4)), ((W1,), (W5,))),
    (1, (D1(W1, W4), D1(W2, W6)), ((W3,), (W5,))),
    (1, (D1(W1, W4), D1(W3, W5)), ((W2,), (W6,))),
    (1, (D1(W1, W5), D1(W2, W4)), ((W3,), (W6,))),
    (1, (D1(W1, W5), D1(W3, W6)), ((W2,), (W4,))),
    (1, (D1(W1, W6), D1(W2, W5)), ((W3,), (W4,))),
    (1, (D1(W1, W6), D1(W3, W4)), ((W2,), (W5,))),
    (1, (D1(W2, W4), D1(W3, W6)), ((W1,), (W5,))),
    (1, (D1(W2, W5), D1(W3, W4)), ((W1,), (W6,))),
    (1, (D1(W2, W6), D1(W3, W5)), ((W1,), (W4,))),
    # D . D2  (scalar)
    (-1, (D1(W1, W4), D2(W2, W3, W5, W6)), ((), ())),
    (H, (D1(W1, W5), D2(W2, W3, W4, W6)), ((), ())),
    (H, (D1(W1, W6), D2(W3, W2, W4, W5)), ((), ())),
    (H, (D1(W2, W4), D2(W1, W3, W5, W6)), ((), ())),
    (-1, (D1(W2, W5), D2(W1, W3, W4, W6)), ((), ())),
    (H, (D1(W2, W6), D2(W1, W3, W4, W5)), ((), ())),
    (H, (D1(W3, W4), D2(W1, W2, W6, W5)), ((), ())),
    (H, (D1(W3, W5), D2(W1, W2, W4, W6)), ((), ())),
    (-1, (D1(W3, W6), D2(W1, W2, W4, W5)), ((), ())),
    # D . D . D  (scalar)
    (4, (D1(W1, W4), D1(W2, W5), D1(W3, W6)), ((), ())),
    (-2, (D1(W1, W4), D1(W2, W6), D1(W3, W5)), ((), ())),
    (-2, (D1(W1, W5), D1(W2, W4), D1(W3, W6)), ((), ())),
    (1, (D1(W1, W5), D1(W2, W6), D1(W3, W4)), ((), ())),
    (1, (D1(W1, W6), D1(W2, W4), D1(W3, W5)), ((), ())),
    (-2, (D1(W1, W6), D1(W2, W5), D1(W3, W4)), ((), ())),
]

WS = (W1, W2, W3, W4, W5, W6)


def decompose(term):
    """Decompose one three-body term with the template and apply the
    virtual rule: a density matrix carrying a virtual index is zero."""
    coeff, tensors, op, leftover = term
    assert op.rank == 3 and not leftover
    mapping = dict(zip(WS, op.upper_index + op.lower_index))
    out = []
    for tcoef, dms, (up, lo) in TEMPLATE:
        dm_sub = tuple(t.substitute(mapping) for t in dms)
        if any(mbe.space_of(x) == "v" for t in dm_sub for x in t.upper + t.lower):
            continue                      # density matrix with virtual index
        out.append((coeff * Rational(tcoef), tensors + dm_sub,
                    Operator([mapping[x] for x in up],
                             [mapping[x] for x in lo]), ()))
    return out


_TEMPLATE_GROUPS_W = None


def template_groups_W():
    """The grouped template in the w1..w6 symbols (pair-canonical
    operators, all signs +1)."""
    global _TEMPLATE_GROUPS_W
    if _TEMPLATE_GROUPS_W is None:
        _TEMPLATE_GROUPS_W = group_fixed_letters(_template_terms(WS))
    return _TEMPLATE_GROUPS_W


def decompose_grouped(term):
    """Decompose one three-body term with the GROUPED template.

    Every template index appears in exactly one factor of each piece, so
    within a group either every piece survives or none does: a group
    survives iff its operator carries every virtual index of the pattern
    (equivalently, none of its density-matrix pieces carries a virtual
    index).  Returns (surviving groups, total groups), where a surviving
    group is (operator, [(coeff, tensors), ...]) with the outer
    coefficient and tensors of the term already included."""
    coeff, tensors, op, leftover = term
    assert op.rank == 3 and not leftover
    mapping = dict(zip(WS, op.upper_index + op.lower_index))
    groups = template_groups_W()
    surviving = []
    for op_t, pieces in groups:
        kept = []
        for c, dms in pieces:
            dm_sub = tuple(t.substitute(mapping) for t in dms)
            if any(mbe.space_of(x) == "v" for t in dm_sub
                   for x in t.upper + t.lower):
                continue                  # density matrix with virtual index
            kept.append((coeff * c, tensors + dm_sub))
        if kept:
            surviving.append((Operator([mapping[x] for x in op_t.upper_index],
                                       [mapping[x] for x in op_t.lower_index]),
                              kept))
    return surviving, len(groups)


# ---------------------------------------------------------------------------
# the final blocks
# ---------------------------------------------------------------------------
_FINAL = None


def final_blocks():
    global _FINAL
    if _FINAL is not None:
        return _FINAL
    mbe.build_blocks()
    out = []
    for n, (kind, xn, rr, ss, is_dag, lhs, terms) in enumerate(mbe.BLOCKS, 1):
        keep = [t for t in terms if not sr.is_zero(t)]
        three = [t for t in keep if t[2].rank == 3]
        small = [t for t in keep if t[2].rank < 3]
        decomposed = []
        for t in three:
            decomposed.extend(decompose(t))
        final = mbe.merge(small + decomposed) if three else keep
        out.append({"n": n, "lhs": lhs, "keep": keep, "three_body": three,
                    "decomposed": mbe.merge(decomposed)
                    if decomposed else [],
                    "final": final})
    _FINAL = out
    return out


# ---------------------------------------------------------------------------
# grouping by operator (for substitution)
# ---------------------------------------------------------------------------
def _op_pattern_key(op):
    """Renaming-independent key of the operator over its exact symmetries:
    the index pairs are permuted together (the diagonal pair-permutation
    symmetry of the spin-summed E; no sign -- a single row alone is not a
    symmetry).  Returns (key, sign, (upper, lower)) with sign always +1,
    where (upper, lower) is the achieving arrangement in the ORIGINAL
    letters -- the operator shown in a group is one that literally appears
    in the block, so the labels match the ungrouped equation."""
    n = op.rank
    best = None
    for perm in itertools.permutations(range(n)):
        cu = [op.upper_index[k] for k in perm]
        cl = [op.lower_index[k] for k in perm]
        tag = {}
        pattern = []
        for x in cu + cl:
            if x not in tag:
                tag[x] = len(tag)
            pattern.append((mbe.space_of(x), tag[x]))
        key = tuple(pattern)
        if best is None or key < best[0]:
            best = (key, 1, (cu, cl))
    return best


def group_by_operator(terms):
    """Collect the coefficient pieces multiplying the same operator.

    Two terms land in the same group iff their operators agree up to the
    exact operator identities (the diagonal pair permutation and renaming
    of the summation labels).  The group's operator keeps the letters of
    the first term that defines the group (no fresh normalization: the
    operator appears on the left of the bracket and its labels must match
    the ungrouped block); the other members are renamed onto those
    letters, with tensor dummies keeping their own letters unless they
    collide with an operator letter.  Returns a list of
    (operator, [(coeff, tensors), ...]) sorted by rank and letters."""
    groups = {}
    for coeff, tensors, op, leftover in terms:
        if leftover:
            raise ValueError("grouping with unresolved deltas")
        key, sign, (cu, cl) = _op_pattern_key(op)
        entry = groups.get(key)
        if entry is None:
            entry = groups[key] = {"op": Operator(cu, cl), "pieces": {}}
        rep = entry["op"]
        mapping = dict(zip(cu + cl, rep.upper_index + rep.lower_index))
        used = set(rep.upper_index) | set(rep.lower_index)
        for t in tensors:
            for x in t.upper + t.lower:
                if x in mapping:
                    continue
                y = x
                if y in used:
                    # dummy letter collides with an operator letter of the
                    # representative: take a fresh letter of its space
                    y = next(z for z in mbe.FAMILY[mbe.space_of(x)]
                             if z not in used)
                mapping[x] = y
                used.add(y)
        tensors_n = tuple(t.substitute(mapping) for t in tensors)
        sig = tuple(sorted((t.signature() for t in tensors_n), key=str))
        prev = entry["pieces"].get(sig)
        entry["pieces"][sig] = ((prev[0] if prev else Rational(0))
                                + sign * coeff, tensors_n)
    out = []
    for key, entry in groups.items():
        pieces = sorted(((c, t) for (c, t) in entry["pieces"].values() if c),
                        key=lambda ct: [x.latex() for x in ct[1]])
        out.append((entry["op"], pieces))
    out.sort(key=lambda gp: (gp[0].rank,
                             [str(x) for x in gp[0].upper_index],
                             [str(x) for x in gp[0].lower_index]))
    return out


def _groups_to_rows(lhs, groups, width):
    """Render groups as align rows: each group is the operator times the
    bracketed sum of its coefficient pieces, broken over rows with
    \\left(...\\right. continuation pairs when it is too long."""
    rows = []
    first_row = True
    for op, pieces in groups:
        if op.rank:
            op_tex = (r"\hat{E}^{%s}_{%s}\,\left("
                      % ("".join(str(x) for x in op.upper_index),
                         "".join(str(x) for x in op.lower_index)))
        else:
            op_tex = ""
        piece_tex = [mbe.term_latex(c, t, Operator([], []), (), n == 0)
                     for n, (c, t) in enumerate(pieces)]
        cur = op_tex + piece_tex[0]
        group_rows = []
        for piece in piece_tex[1:]:
            if op_tex and len(cur) + 1 + len(piece) > width:
                group_rows.append(cur + r"\right.")
                cur = r"\left.\quad " + piece
            elif not op_tex and len(cur) + 1 + len(piece) > width:
                group_rows.append(cur)
                cur = r"\quad " + piece
            else:
                cur += " " + piece
        group_rows.append(cur + (r"\right)" if op_tex else ""))
        for row in group_rows:
            if first_row:
                rows.append(lhs + " &= " + row)
                first_row = False
            else:
                rows.append("&" + row)
    return rows


def grouped_rows(lhs, terms, width=140):
    """Grouped rows of a commutator block (summed dummies: renaming and
    canonicalization allowed)."""
    return _groups_to_rows(lhs, group_by_operator(terms), width)


# ---------------------------------------------------------------------------
# grouping of the spin_free template (operator identity: fixed labels)
# ---------------------------------------------------------------------------
def _canonical_rows(op):
    """Min form of the operator over diagonal pair permutations only,
    letters FIXED (operator identity: no renaming); no sign is involved:
    the two rows of the spin-summed E may only be permuted together."""
    n = op.rank
    best = None
    best_op = op
    for perm in itertools.permutations(range(n)):
        cu = [op.upper_index[k] for k in perm]
        cl = [op.lower_index[k] for k in perm]
        key = (tuple(str(x) for x in cu), tuple(str(x) for x in cl))
        if best is None or key < best:
            best = key
            best_op = Operator(cu, cl)
    return 1, best_op


def _template_terms(six):
    """The 51 template terms as (coeff, tensors, operator) with the six
    given symbols standing for w1..w6."""
    disp = dict(zip(WS, six))
    out = []
    for tcoef, dms, (up, lo) in TEMPLATE:
        tensors = tuple(t.substitute(disp) for t in dms)
        op = Operator([disp[x] for x in up], [disp[x] for x in lo])
        out.append((Rational(tcoef), tensors, op))
    return out


def template_terms():
    """The template in the display letters p,q,r,s,t,u."""
    return _template_terms(symbols("p q r s t u"))


def group_fixed_letters(terms):
    """Group terms by their operator WITHOUT renaming any index (operator
    identity: the labels are fixed on both sides); only the index pairs of
    each operator are permuted together (diagonal pair permutation, no
    sign).  Pieces with identical tensor multisets are merged."""
    groups = {}
    for coeff, tensors, op in terms:
        sign, op_c = _canonical_rows(op)
        key = (op_c.upper_index, op_c.lower_index)
        entry = groups.setdefault(key, {"op": op_c, "pieces": {}})
        sig = tuple(sorted((t.signature() for t in tensors), key=str))
        prev = entry["pieces"].get(sig)
        entry["pieces"][sig] = ((prev[0] if prev else Rational(0))
                                + sign * coeff, tensors)
    out = []
    for key, entry in groups.items():
        pieces = sorted(((c, t) for (c, t) in entry["pieces"].values() if c),
                        key=lambda ct: [x.latex() for x in ct[1]])
        out.append((entry["op"], pieces))
    out.sort(key=lambda gp: (-gp[0].rank,
                             [str(x) for x in gp[0].upper_index],
                             [str(x) for x in gp[0].lower_index]))
    return out


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------
def template_latex():
    """The template rendered with display letters p,q,r,s,t,u."""
    return [mbe.term_latex(c, t, op, (), n == 0)
            for n, (c, t, op) in enumerate(template_terms())]


def write_tex(blocks):
    L = []
    add = L.append
    add(r"\documentclass[11pt]{article}")
    add(r"\usepackage{amsmath,amssymb}")
    add(r"\usepackage[margin=2.5cm]{geometry}")
    add(r"\allowdisplaybreaks")
    add(r"\title{Cumulant expansion of the three-body terms}")
    add(r"\date{generated by \texttt{cumulant\_expansion.py}}")
    add(r"\begin{document}")
    add(r"\maketitle")
    add("")
    add(r"The three-body operators surviving the space restriction are"
        r" decomposed into two-body and one-body operators (plus scalar"
        r" terms) with density-matrix coefficients, using the spin-free"
        r" cumulant decomposition of"
        r" \texttt{cumulant\_decomposition/spin\_free/spin\_free.tex}:")
    rows = mbe.pack_rows(
        r"\hat{E}^{pqr}_{stu} \approx", template_latex())
    add(r"\begin{align}")
    for row in rows[:-1]:
        add(row + r" \\")
    add(rows[-1])
    add(r"\end{align}")
    add(r"This is an operator identity: the labels $p,q,r,s,t,u$ are fixed"
        r" on both sides, so no index may be renamed; only the index pairs"
        r" of each operator can be permuted together (the diagonal"
        r" pair-permutation symmetry of the spin-summed $\hat{E}$, no"
        r" sign).  With the terms grouped by the operator in this"
        r" sense, the decomposition reads")
    tgroups = group_fixed_letters(template_terms())
    rows = _groups_to_rows(r"\hat{E}^{pqr}_{stu}", tgroups, 140)
    rows[0] = rows[0].replace("&=", r"&\approx", 1)
    add(r"\begin{align}")
    for row in rows[:-1]:
        add(row + r" \\")
    add(rows[-1])
    add(r"\end{align}")
    add(r"(%d operator groups; a group with no $\hat{E}$ shown is the"
        r" scalar part.)" % len(tgroups))
    add(r"The $\approx$ drops the normal-ordered three-body remainder and"
        r" the three-cumulant $\lambda^{pqr}_{stu}$; for a"
        r" single-determinant reference $\lambda_3 = 0$ identically."
        r" Every density matrix carrying a virtual index vanishes;"
        r" density matrices with occupied or whole-space indices are kept"
        r" symbolic as $D^i_j$, $D^{ij}_{kl}$, \dots\ (no delta-function"
        r" substitution).  The formula is verified numerically for a"
        r" closed-shell single-determinant reference on Jordan--Wigner"
        r" matrices (random index assignments with and without the"
        r" virtual rule).")
    add("")
    add(r"\section{Final blocks}")
    add(r"Only the blocks containing surviving three-body terms change;"
        r" the three-body pieces are decomposed and merged back into the"
        r" block.  Blocks not shown are unchanged from"
        r" \texttt{space\_restriction.tex}.")
    for blk in blocks:
        if not blk["three_body"]:
            continue
        add(r"\subsection*{%s}" % ("$" + blk["lhs"] + "$"))
        add(r"after the space restriction (%d terms, %d three-body):"
            % (len(blk["keep"]), len(blk["three_body"])))
        rows = mbe.pack_rows(blk["lhs"], mbe.terms_latex(blk["keep"]))
        add(r"\begin{align}")
        for row in rows[:-1]:
            add(row + r" \\")
        add(rows[-1])
        add(r"\end{align}")
        add(r"each three-body term is decomposed with the grouped identity."
            r" Every template index appears in exactly one factor of each"
            r" piece, so a whole operator group survives or dies together:"
            r" a group survives iff its operator carries every virtual"
            r" index of the pattern; in particular the scalar part (pure"
            r" density matrices) always vanishes.  The surviving groups"
            r" are:")
        for term in blk["three_body"]:
            surviving, total = decompose_grouped(term)
            n2 = sum(1 for op, _ in surviving if op.rank == 2)
            n1 = sum(1 for op, _ in surviving if op.rank == 1)
            add(r"%d of the %d groups survive (%d two-body, %d one-body):"
                % (len(surviving), total, n2, n1))
            t_lhs = mbe.term_latex(*term[:3], (), True)
            rows = _groups_to_rows(t_lhs, surviving, 140)
            rows[0] = rows[0].replace("&=", r"&\approx", 1)
            add(r"\begin{align}")
            for row in rows[:-1]:
                add(row + r" \\")
            add(rows[-1])
            add(r"\end{align}")
        add(r"the decomposition of the three-body part produces %d terms;"
            r" the final block (surviving terms and decomposition"
            r" merged) has %d terms:"
            % (len(blk["decomposed"]), len(blk["final"])))
        rows = mbe.pack_rows(blk["lhs"], mbe.terms_latex(blk["final"]))
        add(r"\begin{align}")
        for row in rows[:-1]:
            add(row + r" \\")
        add(rows[-1])
        add(r"\end{align}")
        add(r"grouped by the operator, so that all coefficient pieces"
            r" multiplying the same $\hat{E}$ can be substituted together"
            r" (%d operator groups):"
            % len(group_by_operator(blk["final"])))
        rows = grouped_rows(blk["lhs"], blk["final"])
        add(r"\begin{align}")
        for row in rows[:-1]:
            add(row + r" \\")
        add(rows[-1])
        add(r"\end{align}")
    add("")
    add(r"\section{All final blocks}")
    for blk in blocks:
        add(r"\subsection*{%s\quad(%d terms)}"
            % ("$" + blk["lhs"] + "$", len(blk["final"])))
        add(r"\begin{align}")
        if blk["final"]:
            rows = mbe.pack_rows(blk["lhs"], mbe.terms_latex(blk["final"]))
            for row in rows[:-1]:
                add(row + r" \\")
            add(rows[-1])
        else:
            add(blk["lhs"] + r" &= 0")
        add(r"\end{align}")
    add(r"\section{All final blocks, grouped by operator}")
    add(r"Every block with the coefficient pieces collected per operator,"
        r" ready for substitution.")
    for blk in blocks:
        n_groups = len(group_by_operator(blk["final"]))
        add(r"\subsection*{%s\quad(%d terms, %d operator groups)}"
            % ("$" + blk["lhs"] + "$", len(blk["final"]), n_groups))
        add(r"\begin{align}")
        if blk["final"]:
            rows = grouped_rows(blk["lhs"], blk["final"])
            for row in rows[:-1]:
                add(row + r" \\")
            add(rows[-1])
        else:
            add(blk["lhs"] + r" &= 0")
        add(r"\end{align}")
    add(r"\end{document}")
    TEX_PATH.write_text("\n".join(L) + "\n")
    print(f"wrote {TEX_PATH}")


if __name__ == "__main__":
    # register the running script under its module name, so that the check
    # module imports this very instance and not a second copy (the term
    # machinery relies on Tensor/Operator class identity)
    sys.modules.setdefault("cumulant_expansion", sys.modules["__main__"])
    sys.path.insert(0, str(HERE.parent / "test"))
    from check_cumulant_expansion import (  # noqa: E402
        check_grouped_decomposition, check_grouping, check_template,
        check_template_grouping)
    blocks = final_blocks()
    print("final blocks (after cumulant expansion of the three-body part):")
    for blk in blocks:
        extra = (f"  [{len(blk['three_body'])} three-body -> "
                 f"{len(blk['decomposed'])} decomposed]"
                 if blk["three_body"] else "")
        print(f"  {blk['n']:2d}. {len(blk['keep']):3d} survivors -> "
              f"{len(blk['final']):3d} final{extra}  {blk['lhs'][:60]}")
    print()
    check_template()
    check_template_grouping()
    check_grouping(blocks)
    check_grouped_decomposition(blocks)
    write_tex(blocks)
    compile_tex(TEX_PATH)
