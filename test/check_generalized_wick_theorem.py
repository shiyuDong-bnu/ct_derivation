#!/usr/bin/env python3
"""Self test of the generalized Wick engine (stage 1).

Moved out of ``generalized_wick_theorem/generalized_wick_commutator.py``;
run either through the stage script

    python3 generalized_wick_theorem/generalized_wick_commutator.py

or standalone (python3 test/check_generalized_wick_theorem.py).  Every
printed equation of the reference note -- the product rules (15), (16),
(19), (20) and the commutators (14), (17), (18), (21) -- is reproduced
term by term in the written paired order, the term count of eq. (6) is
checked, and all results are verified numerically against exact
Jordan-Wigner matrices (the sparse ``_jw_matrices``/``_matmul`` helpers
of the stage).
"""

import itertools
import random
import sys
from pathlib import Path

from sympy import symbols

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "generalized_wick_theorem"))

from generalized_wick_commutator import (  # noqa: E402
    Operator, _jw_matrices, _matmul, collect, commutator, product_paired)


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
