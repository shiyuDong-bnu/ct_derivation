#!/usr/bin/env python3
"""Numerical verification of the many-body expansion (stage 3).

Moved out of ``many_body_equation/expand_commutators.py``; run either
through the stage script

    python3 many_body_equation/expand_commutators.py

or standalone (python3 test/check_many_body_equation.py).  Every expanded
block is compared against the direct nested matrix commutator with random
real tensors of the spin-integrated symmetry classes, twice:

* check_blocks -- Jordan-Wigner spin-orbital matrices (4 modes, 2
  occupied + 2 virtual);
* check_blocks_spin_summed -- genuinely spin-summed operators (4 spatial
  orbitals = 8 spin orbitals).  This is the check that sees the
  diagonal-pair-permutation merging rule: for a spin-orbital string the
  single-row antisymmetry is exact, so the first check cannot
  distinguish it.
"""

import itertools
import sys
from pathlib import Path

from sympy.physics.quantum import Commutator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "many_body_equation"))
sys.path.insert(0, str(HERE.parent / "commutator"))
sys.path.insert(0, str(HERE.parent / "generalized_wick_theorem"))

from expand_commutators import (  # noqa: E402
    FAMILY, F_OP, H_OP, RANGE, SPACE, STATIC_POOL, V_OP, _excitation_pool,
    expand_expr, expanded_blocks, space_of)
from generate_commutators import F, H, T1, T2, V, is_dagger  # noqa: E402
from generalized_wick_commutator import _jw_matrices, _matmul  # noqa: E402


def check_blocks():
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
        for vals in itertools.product(*(RANGE[space_of(x)] for x in syms)):
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
                    *(RANGE[space_of(x)] for x in free + dummies)):
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
        if is_dagger(expr):
            return dagger_mat(block_matrix(expr.args[0]))
        if expr == T1 or expr == T2:
            return op_matrix_direct(
                _excitation_pool(1 if expr == T1 else 2, 0))
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
    sanity assertions below verify both statements.  A fresh re-expansion
    of every block (``expand_expr``, never merged) is checked alongside
    the merged terms, so a failure is attributed to the merging
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
            shape = tuple(len(RANGE[space_of(x)]) for x in t.upper + t.lower)
            arr = np.zeros(shape)
            for pos in itertools.product(*(range(n_) for n_ in shape)):
                val = {}
                for k, x in enumerate(t.upper + t.lower):
                    val[x] = RANGE[space_of(x)][pos[k]]
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

    letters = list(dict.fromkeys(
        list(SPACE) + list(FAMILY["g"] + FAMILY["o"] + FAMILY["v"])))
    labels = ([chr(ord("a") + k) for k in range(26)]
              + [chr(ord("A") + k) for k in range(26)])
    sub = {x: labels[k] for k, x in enumerate(letters)}

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
                    *(range(len(RANGE[space_of(x)])) for x in free)):
                if abs(C[pos]) < 1e-14:
                    continue
                vals = [RANGE[space_of(x)][pos[k]] for k, x in enumerate(free)]
                m += C[pos] * e_sf(vals[:op.rank], vals[op.rank:])
        return m

    mats = {H: terms_matrix_sf(H_OP), F: terms_matrix_sf(F_OP),
            V: terms_matrix_sf(V_OP),
            T1: terms_matrix_sf(_excitation_pool(1, 0)),
            T2: terms_matrix_sf(_excitation_pool(2, 0))}

    def block_matrix_sf(expr):
        if isinstance(expr, Commutator):
            a = block_matrix_sf(expr.args[0])
            b = block_matrix_sf(expr.args[1])
            return a @ b - b @ a
        if is_dagger(expr):
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


if __name__ == "__main__":
    check_blocks()
    check_blocks_spin_summed()
