#!/usr/bin/env python3
"""Numerical checks of the cumulant expansion (stage 6).

Moved out of ``cumulant_expansion/cumulant_expansion.py``; run either
through the stage script

    python3 cumulant_expansion/cumulant_expansion.py

or standalone (python3 test/check_cumulant_expansion.py).  Four checks:

* check_template -- the 51-term spin-free cumulant identity itself, for a
  closed-shell single-determinant reference on Jordan-Wigner matrices (4
  spatial orbitals = 8 spin orbitals), on random index assignments, with
  and without the virtual rule;
* check_template_grouping -- the fixed-letter grouping of the template
  flattens back to the 51 terms exactly (operator identity: no renaming,
  only the diagonal pair permutation of the spin-summed E, no sign);
* check_grouping -- the per-operator grouping of the final blocks is
  presentational only (flatten = block);
* check_grouped_decomposition -- the group-level decomposition (virtual
  rule applied to whole template groups) reproduces the flat term-by-term
  decomposition exactly.
"""

import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "many_body_equation"))
sys.path.insert(0, str(HERE.parent / "space_restriction"))
sys.path.insert(0, str(HERE.parent / "generalized_wick_theorem"))
sys.path.insert(0, str(HERE.parent / "cumulant_expansion"))

import expand_commutators as mbe  # noqa: E402
from cumulant_expansion import (  # noqa: E402
    TEMPLATE, WS, _canonical_rows, decompose, decompose_grouped,
    final_blocks, group_by_operator, group_fixed_letters, template_terms)
from generalized_wick_commutator import _jw_matrices  # noqa: E402


def check_grouping(blocks):
    """The grouping is presentational only: flattening the groups and
    re-canonicalizing must reproduce the final block term for term."""
    print("grouping check (flattened groups = final block):")
    ok = True
    for blk in blocks:
        if not blk["three_body"]:
            continue
        flat = []
        for op, pieces in group_by_operator(blk["final"]):
            for c, t in pieces:
                flat.append((c, t, op, ()))
        a = sorted(str((c, [x.signature() for x in t], op.upper_index,
                        op.lower_index))
                   for c, t, op, _ in mbe.merge(flat))
        b = sorted(str((c, [x.signature() for x in t], op.upper_index,
                        op.lower_index))
                   for c, t, op, _ in mbe.merge(list(blk["final"])))
        same = a == b
        ok &= same
        print(f"  [{'OK ' if same else 'FAIL'}] block {blk['n']:2d}: "
              f"{len(blk['final'])} terms -> "
              f"{len(group_by_operator(blk['final']))} operator groups")
    if not ok:
        raise SystemExit("grouping check failed")
    print()


def check_grouped_decomposition(blocks):
    """The group-level decomposition (virtual rule applied to whole
    template groups) must reproduce the flat term-by-term decomposition
    exactly; report the surviving/killed group counts per pattern."""
    def norm(c, t, op):
        sign, op_c = _canonical_rows(op)
        sig = tuple(sorted((x.signature() for x in t), key=str))
        return str((sign * c, sig, op_c.upper_index, op_c.lower_index))

    print("grouped decomposition check (grouped = flat, per three-body"
          " term):")
    ok = True
    for blk in blocks:
        for term in blk["three_body"]:
            surviving, total = decompose_grouped(term)
            flat_g = sorted(norm(c, t, op) for op, pieces in surviving
                            for c, t in pieces)
            flat_f = sorted(norm(c, t, op) for c, t, op, _ in decompose(term))
            same = flat_g == flat_f
            ok &= same
            n2 = sum(1 for op, _ in surviving if op.rank == 2)
            n1 = sum(1 for op, _ in surviving if op.rank == 1)
            print(f"  [{'OK ' if same else 'FAIL'}] block {blk['n']:2d} "
                  f"{mbe.term_latex(*term[:3], (), True)}: "
                  f"{len(surviving)}/{total} groups survive "
                  f"({n2} two-body, {n1} one-body)")
    if not ok:
        raise SystemExit("grouped decomposition check failed")
    print()


def check_template():
    import numpy as np

    rng = np.random.default_rng(20260610)
    cre, ann = _jw_matrices(8)          # 4 spatial orbitals x 2 spins
    dim = 256
    occ_spatial = (0, 1)

    def dense(m):
        a = np.zeros((dim, dim))
        for (r_, c_), v in m.items():
            a[r_, c_] = v
        return a

    cre_d = [dense(m) for m in cre]
    ann_d = [dense(m) for m in ann]

    def e_sf(up_vals, lo_vals):
        """Spin-free E matrix: sum over spins of the paired string."""
        m = np.zeros((dim, dim))
        rank = len(up_vals)
        for spins in itertools.product((0, 1), repeat=rank):
            mm = np.eye(dim)
            for u, sg in zip(up_vals, spins):
                mm = mm @ cre_d[2 * u + sg]
            for l_, sg in zip(reversed(lo_vals), reversed(spins)):
                mm = mm @ ann_d[2 * l_ + sg]
            m += mm
        return m

    # reference: spatial orbitals 0, 1 doubly occupied
    ref = np.zeros(dim)
    ref[sum(1 << o for o in (0, 1, 2, 3))] = 1.0

    # reference density matrices, computed from the reference state
    d1 = np.zeros((4, 4))
    for p_ in range(4):
        for q_ in range(4):
            d1[p_, q_] = ref @ (e_sf((p_,), (q_,)) @ ref)
    d2 = np.zeros((4, 4, 4, 4))
    for pq in itertools.product(range(4), repeat=2):
        for rs in itertools.product(range(4), repeat=2):
            d2[pq + rs] = ref @ (e_sf(pq, rs) @ ref)

    # sanity: closed-shell values and single-determinant factorization
    assert np.allclose(d1, 2 * np.diag([1, 1, 0, 0]))
    assert np.allclose(
        d2, np.einsum("pr,qs->pqrs", d1, d1)
        - 0.5 * np.einsum("ps,qr->pqrs", d1, d1))

    def e3_no(up_vals, lo_vals):
        """Fermi-vacuum normal-ordered spin-free E3."""
        m = np.zeros((dim, dim))
        for spins in itertools.product((0, 1), repeat=3):
            ops = ([("c", 2 * u + sg) for u, sg in zip(up_vals, spins)]
                   + [("a", 2 * l_ + sg)
                      for l_, sg in zip(reversed(lo_vals),
                                        reversed(spins))])

            def qpc(op):
                kind, x = op
                occ = (x // 2) in occ_spatial
                return (kind == "c") != occ

            sign = (-1) ** sum(1 for x in range(6) for y in range(x + 1, 6)
                               if not qpc(ops[x]) and qpc(ops[y]))
            mm = np.eye(dim)
            for kind, x in [o for o in ops if qpc(o)] + \
                           [o for o in ops if not qpc(o)]:
                mm = mm @ (cre_d[x] if kind == "c" else ann_d[x])
            m += sign * mm
        return m

    def dm_value(t, val):
        up = [val[x] for x in t.upper]
        lo = [val[x] for x in t.lower]
        if len(up) == 1:
            return d1[up[0], lo[0]]
        return d2[up[0], up[1], lo[0], lo[1]]

    def template_matrix(assignment, virtual_rule):
        val = dict(zip(WS, assignment))
        m = np.zeros((dim, dim))
        for tcoef, dms, (up, lo) in TEMPLATE:
            if virtual_rule and any(
                    val[x] not in occ_spatial
                    for t in dms for x in t.upper + t.lower):
                continue
            factor = float(tcoef)
            for t in dms:
                factor *= dm_value(t, val)
            if not factor:
                continue
            up_v = tuple(val[x] for x in up)
            lo_v = tuple(val[x] for x in lo)
            if up_v:
                m += factor * e_sf(up_v, lo_v)
            else:
                m += factor * np.eye(dim)
        return m

    def run_check(assignments, virtual_rule):
        max_dev = 0.0
        for assignment in assignments:
            dev = float(np.max(np.abs(
                e_sf(assignment[:3], assignment[3:])
                - e3_no(assignment[:3], assignment[3:])
                - template_matrix(assignment, virtual_rule))))
            max_dev = max(max_dev, dev)
        return max_dev

    all_assignments = list(itertools.product(range(4), repeat=6))
    sample = [all_assignments[k] for k in
              rng.choice(len(all_assignments), size=150, replace=False)]
    dev = run_check(sample, False)
    print("template check (E3 - normal-ordered E3 = 51 terms, "
          "single-determinant reference):")
    print(f"  150 random assignments, no virtual rule: "
          f"max|Delta| = {dev:.3e}")
    if dev > 1e-9:
        raise SystemExit("template check failed")

    print("per-pattern check (virtual rule applied; killed terms vanish "
          "for the reference):")
    occ_v, vir_v, all_v = (0, 1), (2, 3), (0, 1, 2, 3)
    ok = True
    for blk in final_blocks():
        for coeff, tensors, op, _ in blk["three_body"]:
            ranges = [{"o": occ_v, "v": vir_v, "g": all_v}[mbe.space_of(x)]
                      for x in op.upper_index + op.lower_index]
            assignments = list(itertools.product(*ranges))
            sample = [assignments[k] for k in
                      rng.choice(len(assignments),
                                 size=min(60, len(assignments)),
                                 replace=False)]
            dev = run_check(sample, True)
            status = "OK " if dev < 1e-9 else "FAIL"
            ok &= dev < 1e-9
            print(f"  [{status}] "
                  f"{mbe.term_latex(coeff, tensors, op, (), True)}"
                  f": max|Delta| = {dev:.3e}")
    if not ok:
        raise SystemExit("pattern check failed")
    print()


def check_template_grouping():
    """The fixed-letter grouping of the template (diagonal pair
    permutation only, no renaming, no signs) flattened again must
    reproduce the 51 terms exactly."""
    def norm(c, t, op):
        sign, op_c = _canonical_rows(op)
        sig = tuple(sorted((x.signature() for x in t), key=str))
        return str((sign * c, sig, op_c.upper_index, op_c.lower_index))

    groups = group_fixed_letters(template_terms())
    flat = sorted(norm(c, t, op) for op, pieces in groups
                  for c, t in pieces)
    orig = sorted(norm(c, t, op) for c, t, op in template_terms())
    n_pieces = sum(len(p) for _, p in groups)
    same = flat == orig
    print(f"template grouping check (flattened = 51 terms, "
          f"{n_pieces} pieces in {len(groups)} operator groups): "
          f"{'OK' if same else 'FAIL'}")
    if not same:
        raise SystemExit("template grouping check failed")
    print()


if __name__ == "__main__":
    blocks = final_blocks()
    check_template()
    check_template_grouping()
    check_grouping(blocks)
    check_grouped_decomposition(blocks)
