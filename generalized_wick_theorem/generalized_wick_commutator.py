"""Generalized Wick theorem for commutators of excitation operators.

The operators are normal-ordered products of (spin-free) excitation
operators,

    E^{u1..un}_{l1..ln} = a^+_{u1} ... a^+_{un} a_{ln} ... a_{l1},

i.e. the written order of the lower indices is the reverse of the order of
the annihilation factors.  Indices are sympy symbols.

For two such operators the generalized Wick theorem (Kutzelnigg and
Mukherjee, J. Chem. Phys. 107, 432 (1997), eqs. (9a)/(9b), proved in
paired-order form in ``single_amplitude_formula/generalized_wick.pdf``)
expands the product as a sum over joins: a join phi pairs k lower indices
of the left factor with k upper indices of the right one, each joined pair
(l, u) contributing a Kronecker delta delta_lu, and the residual is one
operator whose slots are the repaired pairs -- the joined left upper q_i
paired with r_{phi(i)}, the partner of the right lower it contracted
against, the untouched pairs unchanged:

    E^[q]_[p] E^[s]_[r] = sum_phi  Delta_phi  E^{[q_i][q_j][s_k]}_{[r_phi(i)][p_j][r_k]}

Every coefficient is +1: there is no sign rule, only the pairing of upper
to lower indices has to be kept (the parity lemma of the note, section 3).
The commutator is AB minus BA with both products expanded by the theorem
and canonicalised; the two no-contraction terms are the same operator with
its pairs permuted, so they cancel automatically.

Only sympy (and the standard library) is used.  Run

    python3 generalized_wick_commutator.py

for the self test: every printed equation of the note -- the product rules
(15), (16), (19), (20) and the commutators (14), (17), (18), (21) -- is
reproduced term by term in the written paired order, the term count of
eq. (6) is checked, and all results are verified numerically against
exact Jordan-Wigner matrices.
"""

import itertools
import random
import sys
from itertools import combinations, permutations

from sympy import Function, KroneckerDelta, Mul, Tuple, symbols

__all__ = ["Operator", "commutator", "product_paired", "collect",
           "to_sympy", "latex"]


class Operator:
    """Normal-ordered monomial ``E^{upper}_{lower}`` with sympy indices."""

    def __init__(self, upper_index, lower_index):
        upper_index = tuple(upper_index)
        lower_index = tuple(lower_index)
        if len(upper_index) != len(lower_index):
            raise ValueError("unbalanced operator: %d upper vs %d lower"
                             % (len(upper_index), len(lower_index)))
        self.upper_index = upper_index
        self.lower_index = lower_index

    @property
    def rank(self):
        return len(self.upper_index)

    def canonical(self):
        """Canonical order: the pairs sorted by their upper index.

        The permutation that sorts the upper row is applied to the lower
        row unchanged, so every (upper, lower) pair is kept together.
        This diagonal permutation of the two rows is the *full* symmetry
        of the spin-summed operator -- and it carries no sign.  Nothing
        stronger is used: permuting one row alone gives a different
        operator (E^{tq}_{rs} is not E^{qt}_{rs}), and no sympy
        canonicalisation is ever applied to the indices.
        """
        order = sorted(range(self.rank),
                       key=lambda i: self.upper_index[i].sort_key())
        return Operator([self.upper_index[i] for i in order],
                        [self.lower_index[i] for i in order])

    def factors(self):
        """Creation/annihilation factors in their actual order."""
        return ([("c", i) for i in self.upper_index]
                + [("a", i) for i in reversed(self.lower_index)])

    def __repr__(self):
        return "E^{%s}_{%s}" % ("".join(str(i) for i in self.upper_index),
                                "".join(str(i) for i in self.lower_index))

    def __eq__(self, other):
        return (isinstance(other, Operator)
                and self.upper_index == other.upper_index
                and self.lower_index == other.lower_index)

    def __hash__(self):
        return hash((self.upper_index, self.lower_index))


# ----------------------------------------------------------------------
# canonical form
# ----------------------------------------------------------------------
def collect(terms):
    """Merge ``(coeff, deltas, op)`` terms into a canonical dict.

    Every operator is brought to its canonical order first (see
    ``Operator.canonical``: the pairs are sorted by their upper index,
    which is sign-free), then like terms are merged.  Returns
    ``{(frozenset(deltas), upper, lower): coeff}``.  Terms whose operator
    repeats an index vanish by Pauli exclusion and are dropped.

    Note that this identifies operators up to permutation of their pairs.
    Two written forms related by a *re*-pairing (e.g. E^{tq}_{rs} vs
    E^{qt}_{rs}, which differ by a fermionic sign) are not identified --
    that is the caveat of section 3.4/8 of the note; compare such
    expressions numerically (the Jordan-Wigner check below).
    """
    merged = {}
    for coeff, deltas, op in terms:
        if coeff == 0:
            continue
        if (len(set(op.upper_index)) != op.rank
                or len(set(op.lower_index)) != op.rank):
            continue
        op = op.canonical()
        key = (frozenset(deltas), op.upper_index, op.lower_index)
        merged[key] = merged.get(key, 0) + coeff
    return {key: coeff for key, coeff in merged.items() if coeff}


# ----------------------------------------------------------------------
# generalized Wick theorem: the sum over joins, every coefficient +1
# ----------------------------------------------------------------------
def _joins(m, n):
    """All partial injective maps phi: {1..m} ,-> {1..n} (0-based).

    Yields every join as a tuple of pairs ``(i, phi(i))`` with i
    ascending, including the empty join.  The count is
    sum_k C(m,k) C(n,k) k!  -- eq. (6) of the note.
    """
    for k in range(0, min(m, n) + 1):
        for dom in combinations(range(m), k):
            for img in permutations(range(n), k):
                yield tuple(zip(dom, img))


def product_paired(first, second):
    """The generalized Wick product ``first * second`` in paired order.

    This is eq. (4)/(12) of ``single_amplitude_formula/generalized_wick.pdf``:
    the sum runs over all joins phi, and every coefficient is +1.  A joined
    pair (i, phi(i)) contracts the lower index p_i of ``first`` with the
    upper index s_{phi(i)} of ``second`` -- contributing delta_{p_i
    s_{phi(i)}} -- and leaves their partners to close up into one slot of
    the residual: q_i paired with r_{phi(i)}.  The untouched pairs are
    carried over unchanged.  Returns the raw term list (not canonicalised:
    the point of the paired order is the written form).
    """
    r1, r2 = first.rank, second.rank
    terms = []
    for join in _joins(r1, r2):
        dom = {i for i, _ in join}
        img = {j for _, j in join}
        deltas = tuple((first.lower_index[i], second.upper_index[j])
                       for i, j in join)
        upper = (tuple(first.upper_index[i] for i, _ in join)
                 + tuple(first.upper_index[j]
                         for j in range(r1) if j not in dom)
                 + tuple(second.upper_index[j]
                         for j in range(r2) if j not in img))
        lower = (tuple(second.lower_index[j] for _, j in join)
                 + tuple(first.lower_index[j]
                         for j in range(r1) if j not in dom)
                 + tuple(second.lower_index[j]
                         for j in range(r2) if j not in img))
        terms.append((+1, deltas, Operator(upper, lower)))
    return terms


def commutator(op1, op2):
    """``[op1, op2]`` via the generalized Wick theorem.

    Both products are expanded by the theorem and canonicalised; the two
    no-contraction terms then cancel automatically, since they are the
    same operator with its pairs permuted.  Every coefficient of the
    expansion is +1, the only minus is the commutator's own.  Returns the
    canonical dict ``{(deltas, upper, lower): coeff}``.
    """
    terms = (product_paired(op1, op2)
             + [(-c, d, op) for c, d, op in product_paired(op2, op1)])
    return collect(terms)


# ----------------------------------------------------------------------
# output
# ----------------------------------------------------------------------
_E = Function("E")


def to_sympy(result):
    """Sympy expression of a canonical commutator dict."""
    total = 0
    for (deltas, upper, lower), coeff in result.items():
        d = Mul(*[KroneckerDelta(l, u) for l, u in sorted(
            deltas, key=lambda pr: (pr[0].sort_key(), pr[1].sort_key()))])
        total += coeff * d * _E(Tuple(*upper), Tuple(*lower))
    return total


def latex(result):
    """LaTeX fragment of a canonical commutator dict, one term per line."""
    def sort_key(item):
        (deltas, upper, lower), _ = item
        return (len(upper),
                [str(i) for i in upper], [str(i) for i in lower],
                sorted((str(a), str(b)) for a, b in deltas))

    lines = []
    for (deltas, upper, lower), coeff in sorted(result.items(), key=sort_key):
        parts = ["\\delta_{%s}^{%s}" % (a, b) for a, b in sorted(
            deltas, key=lambda pr: (str(pr[0]), str(pr[1])))]
        parts.append("E^{%s}_{%s}" % (" ".join(str(i) for i in upper),
                                      " ".join(str(i) for i in lower)))
        sign = "- " if coeff < 0 else "+ "
        mag = abs(coeff)
        lines.append(sign + ("" if mag == 1 else "%d " % mag)
                     + " ".join(parts))
    return "\\\\\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# numeric verification: exact Jordan-Wigner matrices, stored sparse
# (every monomial matrix has at most one entry per column)
# ----------------------------------------------------------------------
def _jw_matrices(n_modes):
    """``{(row, col): +-1}`` matrices of a_p and a^+_p, Jordan-Wigner."""
    cre, ann = [], []
    for mode in range(n_modes):
        a, c = {}, {}
        below = (1 << mode) - 1
        for state in range(1 << n_modes):
            if (state >> mode) & 1:
                parity = bin(state & below).count("1") & 1
                flipped = state ^ (1 << mode)
                a[(flipped, state)] = -1 if parity else 1
                c[(state, flipped)] = -1 if parity else 1
        ann.append(a)
        cre.append(c)
    return cre, ann


def _matmul(A, B):
    by_row = {}
    for (r, k), v in A.items():
        by_row.setdefault(k, []).append((r, v))
    out = {}
    for (k, c), v2 in B.items():
        for r, v1 in by_row.get(k, ()):
            out[(r, c)] = out.get((r, c), 0) + v1 * v2
    return {key: v for key, v in out.items() if v}


def _numeric_sweep(op1, op2, result, labels, n_modes, palette,
                   sample=None, seed=1, commutator=True):
    """Check the expansion against explicit matrices.

    ``result`` is the canonical dict of the expansion of ``op1 * op2`` (or
    of ``[op1, op2]`` when ``commutator`` is true).  Returns
    ``(bad, total)``: how many of the ``total`` index assignments violate
    the identity.
    """
    cre, ann = _jw_matrices(n_modes)
    dim = 1 << n_modes
    identity = {(i, i): 1 for i in range(dim)}
    cache = {}

    def matrix(upper, lower, v):
        key = (tuple(v[i] for i in upper), tuple(v[i] for i in lower))
        if key not in cache:
            m = identity
            for i in upper:
                m = _matmul(m, cre[v[i]])
            for i in reversed(lower):
                m = _matmul(m, ann[v[i]])
            cache[key] = m
        return cache[key]

    if sample is None:
        assignments = itertools.product(range(palette), repeat=len(labels))
        total = palette ** len(labels)
    else:
        rng = random.Random(seed)
        assignments = (tuple(rng.randrange(palette) for _ in labels)
                       for _ in range(sample))
        total = sample

    bad = 0
    for tup in assignments:
        v = dict(zip(labels, tup))
        lhs = _matmul(matrix(op1.upper_index, op1.lower_index, v),
                      matrix(op2.upper_index, op2.lower_index, v))
        if commutator:
            for (r, c), val in _matmul(
                    matrix(op2.upper_index, op2.lower_index, v),
                    matrix(op1.upper_index, op1.lower_index, v)).items():
                lhs[r, c] = lhs.get((r, c), 0) - val
            lhs = {key: val for key, val in lhs.items() if val}
        rhs = {}
        for (deltas, upper, lower), coeff in result.items():
            if all(v[a] == v[b] for a, b in deltas):
                for key, val in matrix(upper, lower, v).items():
                    rhs[key] = rhs.get(key, 0) + coeff * val
        rhs = {key: val for key, val in rhs.items() if val}
        if lhs != rhs:
            bad += 1
    return bad, total


# ----------------------------------------------------------------------
# self test
# ----------------------------------------------------------------------
FAILED = []


def _check(label, ok, detail=""):
    if not ok:
        FAILED.append(label)
    print("  [%s] %-58s %s" % ("ok" if ok else "FAIL", label, detail))


def _as_operators(raw):
    """Turn the tuple-based product-rule output into Operator terms."""
    return [(coeff, tuple(deltas), Operator(op[1], op[2]))
            for coeff, deltas, op in raw]


def _p11(u1, l1, u2, l2):
    """E^{u1}_{l1} E^{u2}_{l2} (verified rule (1)); labels are 1-tuples."""
    return [(+1, (), ("E", u1 + u2, l1 + l2)),
            (+1, ((l1[0], u2[0]),), ("E", u1, l2))]


def _p12(u, l, u1, u2, l1, l2):
    """E^{u}_{l} E^{u1 u2}_{l1 l2} (verified rule (2)); labels are 1-tuples."""
    return [(+1, (), ("E", u + u1 + u2, l + l1 + l2)),
            (+1, ((l[0], u1[0]),), ("E", u + u2, l1 + l2)),
            (+1, ((l[0], u2[0]),), ("E", u1 + u, l1 + l2))]


def _p21(u1, u2, l1, l2, u, l):
    """E^{u1 u2}_{l1 l2} E^{u}_{l} (verified rule (2')); 1-tuple labels."""
    return [(+1, (), ("E", u1 + u2 + u, l + l1 + l2)),
            (-1, ((l1[0], u[0]),), ("E", u1 + u2, l2 + l)),
            (+1, ((l2[0], u[0]),), ("E", u1 + u2, l1 + l))]


def _p22(u1, u2, l1, l2, v1, v2, m1, m2):
    """E^{u1 u2}_{l1 l2} E^{v1 v2}_{m1 m2} (verified rule (3)); 1-tuples."""
    return [(+1, (), ("E", u1 + u2 + v1 + v2, l1 + l2 + m1 + m2)),
            (-1, ((l1[0], v1[0]),), ("E", u1 + u2 + v2, l2 + m1 + m2)),
            (+1, ((l1[0], v2[0]),), ("E", u1 + u2 + v1, l2 + m1 + m2)),
            (+1, ((l2[0], v1[0]),), ("E", u1 + u2 + v2, l1 + m1 + m2)),
            (-1, ((l2[0], v2[0]),), ("E", u1 + u2 + v1, l1 + m1 + m2)),
            (+1, ((l1[0], v1[0]), (l2[0], v2[0])), ("E", u1 + u2, m1 + m2)),
            (-1, ((l1[0], v2[0]), (l2[0], v1[0])), ("E", u1 + u2, m1 + m2))]


def test():
    p, q, r, s, t, u, v, w = symbols("p q r s t u v w")
    op_q_p = Operator([q], [p])
    op_s_r = Operator([s], [r])
    op_tu_rs = Operator([t, u], [r, s])
    op_rs_pq = Operator([r, s], [p, q])
    op_uw_tv = Operator([u, w], [t, v])

    print("1. the note's equations, as printed (generalized_wick.pdf)")
    from math import comb, factorial

    # term count of the product rule, eq. (6) of the note
    for m, n in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 3)]:
        op_m = Operator(symbols("a0:%d" % m), symbols("b0:%d" % m))
        op_n = Operator(symbols("c0:%d" % n), symbols("d0:%d" % n))
        want_count = sum(comb(m, k) * comb(n, k) * factorial(k)
                         for k in range(min(m, n) + 1))
        _check("term count (%d, %d) == sum_k C(m,k) C(n,k) k!" % (m, n),
               len(product_paired(op_m, op_n)) == want_count)

    # every equation of the note, transcribed term by term in the written
    # (paired) order, compared against the raw output -- this checks the
    # written index placement, not only the operator identity
    def raw(terms):
        return [(c, tuple(d), op.upper_index, op.lower_index)
                for c, d, op in terms]

    prod_15 = [  # E^q_p E^{tu}_{rs}
        (+1, (), (q, t, u), (p, r, s)),
        (+1, ((p, t),), (q, u), (r, s)),
        (+1, ((p, u),), (q, t), (s, r))]
    prod_16 = [  # E^{tu}_{rs} E^q_p
        (+1, (), (t, u, q), (r, s, p)),
        (+1, ((r, q),), (t, u), (p, s)),
        (+1, ((s, q),), (u, t), (p, r))]
    prod_19 = [  # E^{rs}_{pq} E^{uw}_{tv}
        (+1, (), (r, s, u, w), (p, q, t, v)),
        (+1, ((p, u),), (r, s, w), (t, q, v)),
        (+1, ((p, w),), (r, s, u), (v, q, t)),
        (+1, ((q, u),), (s, r, w), (t, p, v)),
        (+1, ((q, w),), (s, r, u), (v, p, t)),
        (+1, ((p, u), (q, w)), (r, s), (t, v)),
        (+1, ((p, w), (q, u)), (r, s), (v, t))]
    prod_20 = [  # E^{uw}_{tv} E^{rs}_{pq}
        (+1, (), (u, w, r, s), (t, v, p, q)),
        (+1, ((t, r),), (u, w, s), (p, v, q)),
        (+1, ((t, s),), (u, w, r), (q, v, p)),
        (+1, ((v, r),), (w, u, s), (p, t, q)),
        (+1, ((v, s),), (w, u, r), (q, t, p)),
        (+1, ((t, r), (v, s)), (u, w), (p, q)),
        (+1, ((t, s), (v, r)), (u, w), (q, p))]

    comm_14 = [  # [E^q_p, E^s_r]
        (+1, ((p, s),), (q,), (r,)),
        (-1, ((r, q),), (s,), (p,))]
    comm_17 = [  # [E^q_p, E^{tu}_{rs}]
        (+1, ((p, t),), (q, u), (r, s)),
        (+1, ((p, u),), (q, t), (s, r)),
        (-1, ((r, q),), (t, u), (p, s)),
        (-1, ((s, q),), (u, t), (p, r))]
    comm_18 = [  # [E^{tu}_{rs}, E^q_p]
        (+1, ((r, q),), (t, u), (p, s)),
        (+1, ((s, q),), (u, t), (p, r)),
        (-1, ((p, t),), (q, u), (r, s)),
        (-1, ((p, u),), (q, t), (s, r))]
    comm_21 = (  # [E^{rs}_{pq}, E^{uw}_{tv}]
        prod_19[1:]
        + [(-c, d, up, lo) for c, d, up, lo in prod_20[1:]])

    def as_terms(raw_terms):
        return [(c, d, Operator(up, lo)) for c, d, up, lo in raw_terms]

    _check("eq. (15): E^q_p E^{tu}_{rs} as printed",
           raw(product_paired(op_q_p, op_tu_rs)) == prod_15)
    _check("eq. (16): E^{tu}_{rs} E^q_p as printed",
           raw(product_paired(op_tu_rs, op_q_p)) == prod_16)
    _check("eq. (19): E^{rs}_{pq} E^{uw}_{tv} as printed",
           raw(product_paired(op_rs_pq, op_uw_tv)) == prod_19)
    _check("eq. (20): E^{uw}_{tv} E^{rs}_{pq} as printed",
           raw(product_paired(op_uw_tv, op_rs_pq)) == prod_20)
    _check("eq. (14): the 1-1 commutator as printed",
           collect(as_terms(comm_14)) == commutator(op_q_p, op_s_r))
    _check("eq. (17): the 1-2 commutator as printed",
           collect(as_terms(comm_17)) == commutator(op_q_p, op_tu_rs))
    _check("eq. (18): the 2-1 commutator as printed",
           collect(as_terms(comm_18)) == commutator(op_tu_rs, op_q_p))
    _check("eq. (21): the 2-2 commutator as printed",
           collect(as_terms(comm_21)) == commutator(op_rs_pq, op_uw_tv))

    print("1b. the canonical order keeps the pairs together")
    _check("E^{tuq}_{rsp} canonicalises to E^{qtu}_{prs}",
           Operator([t, u, q], [r, s, p]).canonical()
           == Operator([q, t, u], [p, r, s]))
    # the diagonal permutation is the full symmetry of the spin-summed
    # operator: swapping both rows together is identified, with no sign...
    _check("E^{sq}_{rp} == E^{qs}_{pr} (both rows swapped, no sign)",
           Operator([s, q], [r, p]).canonical()
           == Operator([q, s], [p, r]).canonical())
    # ...but permuting one row alone is NOT an identity -- the spin-summed
    # operator has no single-row (anti)symmetry, so the canonical order
    # must keep E^{tq}_{rs} and E^{qt}_{rs} apart
    _check("E^{tq}_{rs} != E^{qt}_{rs} (one row alone is not a symmetry)",
           Operator([t, q], [r, s]).canonical()
           != Operator([q, t], [r, s]).canonical())
    for a, b in [(op_q_p, op_s_r), (op_q_p, op_tu_rs),
                 (op_rs_pq, op_uw_tv)]:
        _check("no-join terms of %s x %s coincide in canonical order"
               % (a, b),
               product_paired(a, b)[0][2].canonical()
               == product_paired(b, a)[0][2].canonical())

    print("2. the commutators, canonicalised")
    cases = [("1-1", op_q_p, op_s_r, (p, q, r, s)),
             ("1-2", op_q_p, op_tu_rs, (p, q, r, s, t, u)),
             ("2-2", op_rs_pq, op_uw_tv, (p, q, r, s, t, u, v, w))]
    results = {}
    for name, a, b, _ in cases:
        results[name] = commutator(a, b)
        print(latex(results[name]))
    for name, a, b, _ in cases:
        got = results[name]
        rev = commutator(b, a)
        _check("%s antisymmetry [B, A] == -[A, B]" % name,
               all(rev.get(key, 0) == -coeff for key, coeff in got.items())
               and len(rev) == len(got))

    print("3. numeric: the commutators, Jordan-Wigner matrices, 4 modes")
    _check("1-1, exhaustive over 4^4 assignments",
           _numeric_sweep(op_q_p, op_s_r, results["1-1"],
                          (p, q, r, s), n_modes=4, palette=4) == (0, 4 ** 4))
    _check("1-2, exhaustive over 3^6 assignments",
           _numeric_sweep(op_q_p, op_tu_rs, results["1-2"],
                          (p, q, r, s, t, u), n_modes=4, palette=3)
           == (0, 3 ** 6))
    _check("2-2, exhaustive over 3^8 assignments",
           _numeric_sweep(op_rs_pq, op_uw_tv, results["2-2"],
                          (p, q, r, s, t, u, v, w), n_modes=4, palette=3)
           == (0, 3 ** 8))
    _check("2-2, random sample of 2000 over 4^8 assignments",
           _numeric_sweep(op_rs_pq, op_uw_tv, results["2-2"],
                          (p, q, r, s, t, u, v, w), n_modes=4, palette=4,
                          sample=2000) == (0, 2000))

    print("4. numeric: the four product rules, Jordan-Wigner matrices")
    # the printed rules of single_amplitude_formula/gen_commutators.py use
    # a different index order than the paired one, so they are compared as
    # matrices, not as written expressions (section 8 caveat of the note)
    rules = [
        ("(1)  1-1", op_q_p, op_s_r,
         _p11((q,), (p,), (s,), (r,)), (p, q, r, s), 4),
        ("(2)  1x2", op_q_p, op_tu_rs,
         _p12((q,), (p,), (t,), (u,), (r,), (s,)), (p, q, r, s, t, u), 3),
        ("(2') 2x1", op_tu_rs, op_q_p,
         _p21((t,), (u,), (r,), (s,), (q,), (p,)), (p, q, r, s, t, u), 3),
        ("(3)  2-2", op_rs_pq, op_uw_tv,
         _p22((r,), (s,), (p,), (q,), (u,), (w,), (t,), (v,)),
         (p, q, r, s, t, u, v, w), 3),
    ]
    for name, a, b, raw_terms, labels, palette in rules:
        terms = collect(_as_operators(raw_terms))
        _check("rule %s, exhaustive over %d^%d assignments"
               % (name, palette, len(labels)),
               _numeric_sweep(a, b, terms, labels, n_modes=4,
                              palette=palette, commutator=False)
               == (0, palette ** len(labels)))

    print("=====")
    if FAILED:
        print("FAILED: %s" % FAILED)
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    test()
