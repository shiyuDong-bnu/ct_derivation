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

for the self test (it lives in test/check_generalized_wick_theorem.py at
the repo root): every printed equation of the note -- the product rules
(15), (16), (19), (20) and the commutators (14), (17), (18), (21) -- is
reproduced term by term in the written paired order, the term count of
eq. (6) is checked, and all results are verified numerically against
exact Jordan-Wigner matrices.
"""

import sys
from itertools import combinations, permutations
from pathlib import Path

from sympy import Function, KroneckerDelta, Mul, Tuple, symbols

HERE = Path(__file__).resolve().parent

__all__ = ["Operator", "commutator", "product_paired", "collect",
           "to_sympy"]


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


# ----------------------------------------------------------------------
# exact Jordan-Wigner matrices, stored sparse (every monomial matrix has
# at most one entry per column) -- the shared matrix helpers of every
# stage's numerical checks
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


if __name__ == "__main__":
    # register the running script under its module name, so that the check
    # module imports this very instance and not a second copy
    sys.modules.setdefault("generalized_wick_commutator",
                           sys.modules["__main__"])
    sys.path.insert(0, str(HERE.parent / "test"))
    from check_generalized_wick_theorem import test
    test()
