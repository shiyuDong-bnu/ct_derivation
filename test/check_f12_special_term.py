#!/usr/bin/env python3
"""Numerical checks of the f12_special_term stage (stage 7).

Run through the stage script (python3 f12_special_term/f12_special_term.py)
or standalone (python3 test/check_f12_special_term.py).  Three checks:

* check_complement -- the F12 special terms are exactly the complement of
  the stage-4 survivors: killed + surviving terms merge back to the
  stage-3 block, per block;
* check_killed_three_body -- the cumulant folding of the killed three-body
  terms: the template identity with the virtual rule, verified on the
  killed index patterns (spin-summed Jordan-Wigner matrices, 4 spatial
  orbitals, single-determinant reference), the same check stage 6 ran on
  the surviving patterns;
* check_special_vanishes -- the assembled special expression (blocks +
  h.c., 1/2 on the Fock double commutator, cumulant-folded) vanishes in
  the F12 setting: free virtuals OBS-vir, contracted virtuals complete,
  geminal amplitudes pair-swap and pair-exchange symmetric with zero
  both-OBS-vir block.  Substituting the special term back therefore does
  not touch the restricted result (eq. (S1) compatibility).
"""

import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "many_body_equation"))
sys.path.insert(0, str(HERE.parent / "space_restriction"))
sys.path.insert(0, str(HERE.parent / "cumulant_expansion"))
sys.path.insert(0, str(HERE.parent / "f12_special_term"))

import expand_commutators as mbe  # noqa: E402
import space_restriction as sr  # noqa: E402
import cumulant_expansion as ce  # noqa: E402
from f12_special_term import special_blocks  # noqa: E402


def check_complement(blocks):
    """killed + survivors = the stage-3 block (merged), per block."""
    print("complement check (killed + survivors = stage-3 block):")
    ok = True

    def norm(terms):
        return sorted(str((c, [t.signature() for t in ts],
                           op.upper_index, op.lower_index))
                      for c, ts, op, _ in mbe.merge(terms))

    for blk, src in zip(blocks, sr.build()):
        same = norm(list(src["zero"]) + list(src["keep"])) \
            == norm(list(src["terms"]))
        ok &= same
        if blk["killed"]:
            print(f"  [{'OK ' if same else 'FAIL'}] block {blk['n']:2d}: "
                  f"{len(src['keep'])} survivors + {len(src['zero'])} killed"
                  f" = {len(src['terms'])} stage-3 terms")
    if not ok:
        raise SystemExit("complement check failed")
    print()


def check_killed_three_body(blocks):
    """The cumulant template with the virtual rule, verified on the killed
    three-body patterns: E3 - (Fermi-vacuum normal-ordered E3) = the
    surviving template pieces, for a single-determinant reference on
    spin-summed Jordan-Wigner matrices (4 spatial orbitals)."""
    from generalized_wick_commutator import _jw_matrices
    rng = np.random.default_rng(20260610)
    dim = 256
    occ_spatial = (0, 1)

    def dense(m):
        a = np.zeros((dim, dim))
        for (r_, c_), v in m.items():
            a[r_, c_] = v
        return a

    cre, ann = _jw_matrices(8)
    cre_d = [dense(m) for m in cre]
    ann_d = [dense(m) for m in ann]
    eye = np.eye(dim)

    def e_sf(up_vals, lo_vals):
        m = np.zeros((dim, dim))
        for spins in itertools.product((0, 1), repeat=len(up_vals)):
            mm = eye
            for u, sg in zip(up_vals, spins):
                mm = mm @ cre_d[2 * u + sg]
            for l_, sg in zip(reversed(lo_vals), reversed(spins)):
                mm = mm @ ann_d[2 * l_ + sg]
            m += mm
        return m

    def e3_no(up_vals, lo_vals):
        """Fermi-vacuum normal-ordered spin-free E3."""
        m = np.zeros((dim, dim))
        for spins in itertools.product((0, 1), repeat=3):
            ops = ([("c", 2 * u + sg) for u, sg in zip(up_vals, spins)]
                   + [("a", 2 * l_ + sg)
                      for l_, sg in zip(reversed(lo_vals),
                                        reversed(spins))])

            def qp(op):
                kind, x = op
                return (kind == "c") != ((x // 2) in occ_spatial)

            sign = (-1) ** sum(1 for x in range(6) for y in range(x + 1, 6)
                               if not qp(ops[x]) and qp(ops[y]))
            mm = eye
            for kind, x in [o for o in ops if qp(o)] \
                    + [o for o in ops if not qp(o)]:
                mm = mm @ (cre_d[x] if kind == "c" else ann_d[x])
            m += sign * mm
        return m

    # SD reference (orbitals 0,1 doubly occupied): exact RDMs
    d1 = 2.0 * np.diag([1.0, 1.0, 0.0, 0.0])
    d2 = (np.einsum("pr,qs->pqrs", d1, d1)
          - 0.5 * np.einsum("ps,qr->pqrs", d1, d1))

    def template_matrix(assignment):
        val = dict(zip(ce.WS, assignment))
        m = np.zeros((dim, dim))
        for tcoef, dms, (up, lo) in ce.TEMPLATE:
            if any(val[x] not in occ_spatial
                   for t in dms for x in t.upper + t.lower):
                continue                    # density matrix with virtual index
            factor = float(tcoef)
            for t in dms:
                up_v = [val[x] for x in t.upper]
                lo_v = [val[x] for x in t.lower]
                factor *= (d1[up_v[0], lo_v[0]] if len(up_v) == 1
                           else d2[up_v[0], up_v[1], lo_v[0], lo_v[1]])
            if not factor:
                continue
            up_v = tuple(val[x] for x in up)
            lo_v = tuple(val[x] for x in lo)
            m += factor * (e_sf(up_v, lo_v) if up_v else eye)
        return m

    print("killed three-body pattern check (E3 - normal-ordered E3 = "
          "surviving template pieces, virtual rule, SD reference):")
    ranges_of = {"o": (0, 1), "v": (2, 3), "g": (0, 1, 2, 3)}
    ok = True
    for blk in blocks:
        for coeff, tensors, op, _ in blk["three_body"]:
            ranges = [ranges_of[mbe.space_of(x)]
                      for x in op.upper_index + op.lower_index]
            assignments = list(itertools.product(*ranges))
            dev = 0.0
            for assignment in assignments:
                dev = max(dev, float(np.max(np.abs(
                    e_sf(assignment[:3], assignment[3:])
                    - e3_no(assignment[:3], assignment[3:])
                    - template_matrix(assignment)))))
            status = "OK " if dev < 1e-9 else "FAIL"
            ok &= dev < 1e-9
            print(f"  [{status}] "
                  f"{mbe.term_latex(coeff, tensors, op, (), True)[:60]}"
                  f": max|Delta| = {dev:.3e}")
    if not ok:
        raise SystemExit("killed three-body pattern check failed")
    print()


# ---------------------------------------------------------------------------
# spin-summed Jordan-Wigner evaluation in the F12 setting
# ---------------------------------------------------------------------------
DIM = 256
_E_CACHE = {}
_CRE_ANN = None


def _jw():
    from generalized_wick_commutator import _jw_matrices
    cre, ann = _jw_matrices(8)

    def dense(m):
        a = np.zeros((DIM, DIM))
        for (r_, c_), v in m.items():
            a[r_, c_] = v
        return a

    return [dense(m) for m in cre], [dense(m) for m in ann]


def e_sf(up_vals, lo_vals):
    """Spin-summed E matrix: sum over spins of the paired string."""
    global _CRE_ANN
    if _CRE_ANN is None:
        _CRE_ANN = _jw()
    key = (tuple(up_vals), tuple(lo_vals))
    if key not in _E_CACHE:
        cre, ann = _CRE_ANN
        m = np.zeros((DIM, DIM))
        for spins in itertools.product((0, 1), repeat=len(up_vals)):
            mm = np.eye(DIM)
            for u, sg in zip(up_vals, spins):
                mm = mm @ cre[2 * u + sg]
            for l_, sg in zip(reversed(lo_vals), reversed(spins)):
                mm = mm @ ann[2 * l_ + sg]
            m += mm
        _E_CACHE[key] = m
    return _E_CACHE[key]


def range_f12(x, free):
    """The F12 reading: indices on the E operator run over the OBS spaces
    (free virtual = OBS-vir {2}), contracted virtuals over the complete
    virtual {2,3}."""
    sp = mbe.space_of(x)
    if sp == "o":
        return (0, 1)
    if sp == "g":
        return (0, 1, 2, 3)
    return (2,) if free else (2, 3)


def eval_terms(terms, arrs, range_of=range_f12):
    """Matrix of a sum of terms, free letters (on the operator) and
    dummies summed over the ranges of range_of(letter, is_free); the
    density matrices come from arrs "D1"/"D2"."""
    m = np.zeros((DIM, DIM))
    for coeff, tensors, op, _ in terms:
        oup, olo = tuple(op.upper_index), tuple(op.lower_index)
        free = list(dict.fromkeys(list(oup) + list(olo)))
        fset = set(free)
        dummies = sorted({x for t in tensors for x in t.upper + t.lower}
                         - fset, key=str)
        letters = free + dummies
        ranges = [range_of(x, x in fset) for x in letters]
        accum = {}
        for vals in itertools.product(*ranges):
            val = dict(zip(letters, vals))
            factor = float(coeff)
            for t in tensors:
                if t.name == "D":
                    arr = arrs["D1"] if len(t.upper) == 1 else arrs["D2"]
                elif t.name == "t":
                    arr = arrs["t1"] if len(t.upper) == 1 else arrs["t"]
                else:
                    arr = arrs[t.name]
                factor *= arr[tuple(val[x] for x in t.upper + t.lower)]
            if abs(factor) > 1e-15:
                key = tuple(vals[:len(free)])
                accum[key] = accum.get(key, 0.0) + factor
        for key, factor in accum.items():
            m += factor * e_sf(key[:len(oup)], key[len(oup):])
    return m


def f12_tensors(seed=20261013, canonical_fock=False):
    """Random real tensors of the symmetry classes of the derivation, with
    the F12 geminal: pair-swap AND pair-exchange symmetric, zero
    both-OBS-vir block (the SP restriction); plus the SD-reference RDMs."""
    rng = np.random.default_rng(seed)
    h1 = rng.normal(size=(4, 4))
    h_val = h1 + h1.T
    if canonical_fock:
        f_val = np.diag(rng.normal(size=4))
    else:
        h2 = rng.normal(size=(4, 4))
        f_val = h2 + h2.T
    g2 = rng.normal(size=(4, 4, 4, 4))
    g_val = g2 + g2.transpose(1, 0, 3, 2) + g2.transpose(2, 3, 0, 1) \
        + g2.transpose(3, 2, 1, 0)
    gem = rng.normal(size=(4, 4, 4, 4))
    gem = gem + gem.transpose(1, 0, 3, 2)          # pair swap
    gem = gem + gem.transpose(2, 3, 0, 1)          # pair exchange
    gem[2, 2, :, :] = 0.0                          # SP restriction
    gem[:, :, 2, 2] = 0.0
    d1 = 2.0 * np.diag([1.0, 1.0, 0.0, 0.0])
    d2 = (np.einsum("pr,qs->pqrs", d1, d1)
          - 0.5 * np.einsum("ps,qr->pqrs", d1, d1))
    return {"h": h_val, "f": f_val, "g": g_val, "t": gem,
            "t1": rng.normal(size=(4, 4)), "D1": d1, "D2": d2}


def check_special_vanishes():
    """The assembled special expression vanishes in the F12 setting."""
    from f12_special_term import assembled
    arrs = f12_tensors()
    m_all = eval_terms(assembled("special"), arrs)
    m_dbl = eval_terms(assembled("special", doubles_only=True), arrs)
    d_all = float(np.max(np.abs(m_all)))
    d_dbl = float(np.max(np.abs(m_dbl)))
    print("special term vanishes in the F12 setting (free virtuals OBS-vir,"
          " geminal both-OBS-vir block zero):")
    print(f"  max|assembled special|            = {d_all:.3e}")
    print(f"  max|assembled special (doubles)|  = {d_dbl:.3e}")
    ok = d_all < 1e-12 and d_dbl < 1e-12
    print(f"  [{'OK' if ok else 'FAIL'}]")
    print()
    if not ok:
        raise SystemExit("special-vanishing check failed")


if __name__ == "__main__":
    blocks = special_blocks()
    check_complement(blocks)
    check_killed_three_body(blocks)
    check_special_vanishes()
