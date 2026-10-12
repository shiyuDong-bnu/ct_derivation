#!/usr/bin/env python3
"""Numerical check of the commutator stage (stage 2).

Moved out of ``commutator/generate_commutators.py``; run either through
the stage script

    python3 commutator/generate_commutators.py

or standalone (python3 test/check_commutator.py).  Every reduction
identity behind the block inventory is checked on random 6x6 matrices:
[X, A] = [X, T1] + [X, T2] + h.c. for X = h, V, F, the split of the
composite [H_b, A], the assembly of (1/2)[[F, A], A] from the 8 nested
blocks, and the nested vs expanded forms of bar H.  The two mixed blocks
[[F, T1], T2] and [[F, T2], T1] are checked to be DIFFERENT operators
(the Jacobi shortcut is not available, so both are kept).
"""

import sys
from pathlib import Path

from sympy import Add, Mul
from sympy.core.parameters import evaluate
from sympy.physics.quantum import Commutator, Dagger, Operator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "commutator"))

from generate_commutators import (  # noqa: E402
    A, B, B1, B2, BAR_H, BAR_H_EXPANDED, C1, C2, CT_second_F, D, D1, D2, F,
    H, HB, NCommutator, T1, T2, V, is_dagger, second_order_F, split_mul)


def check_identities():
    import numpy as np

    rng = np.random.default_rng(20260409)
    n = 6

    def rand_matrix():
        return (rng.uniform(-1, 1, (n, n))
                + 1j * rng.uniform(-1, 1, (n, n)))

    def rand_hermitian():
        m = rand_matrix()
        return m + m.conj().T

    subs = {H: rand_hermitian(), V: rand_hermitian(), F: rand_hermitian(),
            T1: rand_matrix(), T2: rand_matrix()}

    def to_matrix(expr):
        if isinstance(expr, Commutator):
            a, b = (to_matrix(x) for x in expr.args)
            return a @ b - b @ a
        if is_dagger(expr):
            return to_matrix(expr.args[0]).conj().T
        if isinstance(expr, Operator):
            return subs[expr]
        if isinstance(expr, Add):
            out = np.zeros((n, n), dtype=complex)
            for term in expr.args:
                out = out + to_matrix(term)
            return out
        if isinstance(expr, Mul):
            coeff, rest = split_mul(expr)
            return float(coeff) * to_matrix(rest)
        raise TypeError(f"cannot evaluate {expr!r}")

    def deviation(lhs, rhs):
        return float(np.max(np.abs(to_matrix(lhs) - to_matrix(rhs))))

    checks = []

    # [X, A] = [X, T1] + [X, T2] + h.c.  (X elementary and Hermitian)
    with evaluate(False):
        for X, blocks, tag in ((H, (B1, B2), "h"),
                               (V, (D1, D2), "V"),
                               (F, (C1, C2), "F")):
            lhs = NCommutator(X, A)
            rhs = blocks[0] + blocks[1] + Dagger(blocks[0]) + Dagger(blocks[1])
            checks.append((f"[{tag}, A] = [{tag}, T1] + [{tag}, T2] + h.c.",
                           deviation(lhs, rhs)))

        # [H_b, A] = [h, A] + [V, A]  (composite bare Hamiltonian)
        checks.append(("[H_b, A] = [h, A] + [V, A]",
                       deviation(NCommutator(HB, A), B + D)))

        # (1/2)[[F, A], A] = (1/2)([C, T] + [C, T]^dagger) over the 8 blocks
        lhs = CT_second_F
        s8 = np.zeros((n, n), dtype=complex)
        for b in second_order_F:
            s8 = s8 + to_matrix(b)
        rhs_mat = 0.5 * (s8 + s8.conj().T)
        diff = to_matrix(lhs) - rhs_mat
        checks.append(("(1/2)[[F, A], A] = (1/2)(sum of 8 blocks + h.c.)",
                       float(np.max(np.abs(diff)))))

        # the expanded target operator agrees with the nested definition
        checks.append(("bar H (nested) = bar H (expanded elementary blocks)",
                       deviation(BAR_H, BAR_H_EXPANDED)))

        # the two mixed blocks are different operators (no Jacobi shortcut)
        diff = to_matrix(NCommutator(C1, T2)) - to_matrix(NCommutator(C2, T1))
        mixed = float(np.max(np.abs(diff)))
        checks.append(("[[F, T1], T2] != [[F, T2], T1] (mixed blocks differ)",
                       mixed))

    print("numerical checks (random 6x6 matrices, h, V and F Hermitian):")
    ok = True
    for name, dev in checks:
        if name.startswith("[["):
            status = "OK " if dev > 1e-6 else "FAIL"
            ok &= dev > 1e-6
        else:
            status = "OK " if dev < 1e-10 else "FAIL"
            ok &= dev < 1e-10
        print(f"  [{status}] {name}: {dev:.3e}")
    if not ok:
        raise SystemExit("identity check failed")
    print()


if __name__ == "__main__":
    check_identities()
