#!/usr/bin/env python3
"""Numerical verification against the literature reference, eq. (S1) of the
Supplemental Material of Masteran, Kumar, Teke, Gaudel, Yanai and Valeev,
"Comment on 'Canonical transcorrelated theory with projected Slater-type
geminals'", J. Chem. Phys. 158, 057101 (2023) (the pdf lives in
reference/supporting.pdf).  Equation (S1) is "a complete expression for
the transcorrelated Hamiltonian obtained without the use of the cumulant
decomposition": 16 terms for

    H + [H, A^F12] + (1/2)[[F, A^F12], A^F12],

with the anti-Hermitian geminal operator (the RHS of (S1) contains only
the G-dependent terms; the bare H does not reappear).

Two checks:

* check_s1 -- the doubles part of the full expression of the
  space_restriction stage (``full_expression(doubles_only=True)``: the
  surviving blocks plus their Hermitian conjugates, the second-order
  blocks carrying the 1/2 of the double commutator, merged once) equals
  the 16 terms of (S1) exactly, as operators.  The comparison is done
  with genuinely spin-summed Jordan-Wigner operators on 4 spatial
  orbitals {0,1 occupied; 2 OBS-vir; 3 CABS}, random real tensors, and
  the F12 reading of both sides: indices ON the E operator run over the
  OBS spaces (free virtual = {2}), contracted virtuals over the complete
  virtual {2,3}, and the geminal amplitudes G (= our t2) are pair-swap
  AND pair-exchange symmetric with G[2,2,i,j] = 0 (the SP-geminal
  restriction; the pair-exchange symmetry is the convention that lets
  (S1) write G with either index orientation -- without it the two
  sides do NOT agree).  Also verified: the terms killed by the
  space-restriction rule vanish exactly in this setting, and the
  assembled doubles expression has exactly 16 terms.

* check_full_expression -- internal consistency: the same assembly over
  the UNRESTRICTED (stage-3) blocks equals the direct nested matrix
  commutator [H_b, A] + (1/2)[[F, A], A] with A = T1 + T2 - (T1 + T2)^+
  (full virtual ranges, no space restriction), proving the "+ h.c., 1/2
  on the double commutator" assembly is exactly the anti-Hermitian
  generator of the commutator stage.

Run standalone (python3 test/check_reference_s1.py) or through the stage
script (python3 space_restriction/space_restriction.py).
"""

import itertools
import sys
from pathlib import Path

import numpy as np
from sympy import Rational

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "many_body_equation"))
sys.path.insert(0, str(HERE.parent / "space_restriction"))
sys.path.insert(0, str(HERE.parent / "commutator"))

import expand_commutators as mbe  # noqa: E402
import space_restriction as sr  # noqa: E402
from generalized_wick_commutator import Operator, _jw_matrices  # noqa: E402

# ---------------------------------------------------------------------------
# equation (S1), term by term (transcribed from the pdf; G -> Tensor("t")).
# Letters: i1..i4 occupied (mbe i1..i4), p1'..p3' whole (mbe p1..p3), and
# virtual mbe letters a1..a8 standing for the (S1) virtuals as
#   a1, a2 <- a1, a2 (OBS-vir)      a3, a4 <- x1, x2 (CABS)
#   a5..a8 <- alpha1..alpha4 (complete virtual)
# ---------------------------------------------------------------------------
_io = mbe.FAMILY["o"]
_pg = mbe.FAMILY["g"]
_va = mbe.FAMILY["v"]
L = {"i1": _io[0], "i2": _io[1], "i3": _io[2], "i4": _io[3],
     "p1": _pg[0], "p2": _pg[1], "p3": _pg[2],
     "a1": _va[0], "a2": _va[1], "x1": _va[2], "x2": _va[3],
     "al1": _va[4], "al2": _va[5], "al3": _va[6], "al4": _va[7]}


def _t(name, up, lo):
    return mbe.Tensor(name, tuple(L[x] for x in up), tuple(L[x] for x in lo))


def _op(up, lo):
    return Operator([L[x] for x in up], [L[x] for x in lo])


S1_TERMS = [
    # first order, [H, A^F12]
    (Rational(1, 2), (_t("g", ("p1", "p2"), ("al1", "al2")),
                      _t("t", ("al1", "al2"), ("i1", "i2"))),
     _op(("i1", "i2"), ("p1", "p2"))),
    (Rational(1), (_t("h", ("p1",), ("x1",)),
                   _t("t", ("a1", "x1"), ("i1", "i2"))),
     _op(("i1", "i2"), ("a1", "p1"))),
    (Rational(1, 2), (_t("g", ("al1", "al2"), ("p1", "p2")),
                      _t("t", ("i1", "i2"), ("al1", "al2"))),
     _op(("p1", "p2"), ("i1", "i2"))),
    (Rational(1), (_t("h", ("x1",), ("p1",)),
                   _t("t", ("i1", "i2"), ("a1", "x1"))),
     _op(("a1", "p1"), ("i1", "i2"))),
    (Rational(1), (_t("g", ("p1", "x1"), ("p2", "p3")),
                   _t("t", ("i2", "i1"), ("a1", "x1"))),
     _op(("p3", "a1", "p2"), ("i1", "i2", "p1"))),
    (Rational(1), (_t("g", ("p1", "p3"), ("p2", "x1")),
                   _t("t", ("x1", "a1"), ("i1", "i2"))),
     _op(("i2", "p2", "i1"), ("a1", "p1", "p3"))),
    # second order, (1/2)[[F, A^F12], A^F12]
    (Rational(1), (_t("f", ("x2",), ("x1",)),
                   _t("t", ("a1", "x1"), ("i1", "i2")),
                   _t("t", ("i4", "i3"), ("a2", "x2"))),
     _op(("i2", "a2", "i1"), ("i3", "i4", "a1"))),
    (Rational(-1, 2), (_t("f", ("i1",), ("p1",)),
                       _t("t", ("al1", "al2"), ("i1", "i2")),
                       _t("t", ("i3", "i4"), ("al1", "al2"))),
     _op(("p1", "i2"), ("i3", "i4"))),
    (Rational(-1, 2), (_t("f", ("p1",), ("i1",)),
                       _t("t", ("al1", "al2"), ("i2", "i3")),
                       _t("t", ("i4", "i1"), ("al1", "al2"))),
     _op(("i2", "i3"), ("i4", "p1"))),
    (Rational(1, 2), (_t("f", ("al1",), ("p1",)),
                      _t("t", ("a1", "x1"), ("i1", "i2")),
                      _t("t", ("i3", "i4"), ("x1", "al1"))),
     _op(("i2", "p1", "i1"), ("i3", "i4", "a1"))),
    (Rational(1, 2), (_t("f", ("p1",), ("al1",)),
                      _t("t", ("i1", "i2"), ("a1", "x1")),
                      _t("t", ("x1", "al1"), ("i3", "i4"))),
     _op(("a1", "i3", "i4"), ("i1", "i2", "p1"))),
    (Rational(-1, 2), (_t("f", ("p1",), ("i2",)),
                       _t("t", ("i1", "i2"), ("a1", "x1")),
                       _t("t", ("a2", "x1"), ("i3", "i4"))),
     _op(("a1", "i3", "i4"), ("i1", "a2", "p1"))),
    (Rational(-1, 2), (_t("f", ("i2",), ("p1",)),
                       _t("t", ("a1", "x1"), ("i1", "i3")),
                       _t("t", ("i2", "i4"), ("a2", "x1"))),
     _op(("i3", "i1", "a2"), ("i4", "a1", "p1"))),
    (Rational(-1, 2), (_t("f", ("i2",), ("p1",)),
                       _t("t", ("a1", "x1"), ("i1", "i2")),
                       _t("t", ("i3", "i4"), ("a2", "x1"))),
     _op(("a2", "p1", "i1"), ("i3", "i4", "a1"))),
    (Rational(-1, 2), (_t("f", ("i2",), ("p1",)),
                       _t("t", ("i1", "i3"), ("a1", "x1")),
                       _t("t", ("a2", "x1"), ("i2", "i4"))),
     _op(("a1", "i4", "p1"), ("i1", "i3", "a2"))),
    (Rational(1), (_t("f", ("al2",), ("al1",)),
                   _t("t", ("al1", "al3"), ("i1", "i2")),
                   _t("t", ("i4", "i3"), ("al2", "al3"))),
     _op(("i2", "i1"), ("i3", "i4"))),
]

# the (S1) virtual letters -> summation ranges (see the letter table above)
_S1_VRANGE = {1: (2,), 2: (2,), 3: (3,), 4: (3,),
              5: (2, 3), 6: (2, 3), 7: (2, 3), 8: (2, 3)}


def _range_s1(x, _free):
    name = str(x)
    base = name.rstrip("0123456789")
    if base == "i":
        return (0, 1)
    if base == "p":
        return (0, 1, 2, 3)
    return _S1_VRANGE[int(name[len(base):])]


def _range_f12(x, free):
    """The F12 reading of our terms: indices on the E operator run over
    the OBS spaces (free virtual = OBS-vir {2}), contracted virtuals over
    the complete virtual {2,3}."""
    sp = mbe.space_of(x)
    if sp == "o":
        return (0, 1)
    if sp == "g":
        return (0, 1, 2, 3)
    return (2,) if free else (2, 3)


def _range_full(x, _free):
    sp = mbe.space_of(x)
    return {"o": (0, 1), "g": (0, 1, 2, 3), "v": (2, 3)}[sp]


# ---------------------------------------------------------------------------
# spin-summed Jordan-Wigner matrices (4 spatial orbitals = 8 spin orbitals)
# and a generic term evaluator (pattern of check_many_body_equation.py)
# ---------------------------------------------------------------------------
DIM = 256


def _dense(m):
    a = np.zeros((DIM, DIM))
    for (r_, c_), v in m.items():
        a[r_, c_] = v
    return a


_cre, _ann = _jw_matrices(8)
CRE = [_dense(m) for m in _cre]
ANN = [_dense(m) for m in _ann]
EYE = np.eye(DIM)
_E_CACHE = {}


def e_sf(up_vals, lo_vals):
    """Spin-summed E matrix: sum over spins of the paired string."""
    key = (tuple(up_vals), tuple(lo_vals))
    if key not in _E_CACHE:
        m = np.zeros((DIM, DIM))
        rank = len(up_vals)
        for spins in itertools.product((0, 1), repeat=rank):
            mm = EYE
            for u, sg in zip(up_vals, spins):
                mm = mm @ CRE[2 * u + sg]
            for l_, sg in zip(reversed(lo_vals), reversed(spins)):
                mm = mm @ ANN[2 * l_ + sg]
            m += mm
        _E_CACHE[key] = m
    return _E_CACHE[key]


def eval_terms(enc_terms, range_of, arrs):
    """Matrix of a sum of encoded terms (coeff, [(name, up, lo), ...],
    (op_up, op_lo)); free letters (on the operator) and dummies are summed
    over the ranges given by range_of(letter, is_free)."""
    m = np.zeros((DIM, DIM))
    for coeff, tensors, (oup, olo) in enc_terms:
        free = list(dict.fromkeys(list(oup) + list(olo)))
        fset = set(free)
        dummies = sorted({x for _, u, l in tensors for x in u + l} - fset,
                         key=str)
        letters = free + dummies
        ranges = [range_of(x, x in fset) for x in letters]
        accum = {}
        for vals in itertools.product(*ranges):
            val = dict(zip(letters, vals))
            factor = float(coeff)
            for name, u, l in tensors:
                factor *= arrs[name][tuple(val[x] for x in u + l)]
            if abs(factor) > 1e-15:
                key = tuple(vals[:len(free)])
                accum[key] = accum.get(key, 0.0) + factor
        for key, factor in accum.items():
            m += factor * e_sf(key[:len(oup)], key[len(oup):])
    return m


def enc(term):
    coeff, tensors, op, _ = term
    return (float(coeff), [(t.name, t.upper, t.lower) for t in tensors],
            (tuple(op.upper_index), tuple(op.lower_index)))


def _random_tensors(seed=20261012):
    rng = np.random.default_rng(seed)
    h1 = rng.normal(size=(4, 4))
    h_val = h1 + h1.T
    h2 = rng.normal(size=(4, 4))
    f_val = h2 + h2.T
    g2 = rng.normal(size=(4, 4, 4, 4))
    g_val = g2 + g2.transpose(1, 0, 3, 2) + g2.transpose(2, 3, 0, 1) \
        + g2.transpose(3, 2, 1, 0)
    t1_val = rng.normal(size=(4, 4))
    # geminal amplitudes: pair-swap and pair-exchange symmetric (the (S1)
    # convention), both-OBS-vir amplitudes zero (the SP-geminal
    # restriction of the F12 construction)
    t2_val = rng.normal(size=(4, 4, 4, 4))
    t2_val = t2_val + t2_val.transpose(1, 0, 3, 2)
    t2_val = t2_val + t2_val.transpose(2, 3, 0, 1)
    t2_val[2, 2, :, :] = 0.0
    t2_val[:, :, 2, 2] = 0.0
    return {"h": h_val, "f": f_val, "g": g_val,
            "t1": t1_val, "t2": t2_val}


def _arrs(vals, t2):
    return {"h": vals["h"], "f": vals["f"], "g": vals["g"], "t": t2}


def check_s1():
    """The doubles part of the stage-4 full expression == eq. (S1)."""
    vals = _random_tensors()
    arrs = _arrs(vals, vals["t2"])
    blocks = sr.build()
    doubles = sr.full_expression(blocks, doubles_only=True)
    zero = [t for bi in (1, 3, 7, 11) for t in blocks[bi]["zero"]]

    s1_as_terms = [(c, ts, op, ()) for c, ts, op in S1_TERMS]
    m_s1 = eval_terms([enc(t) for t in s1_as_terms], _range_s1, arrs)
    m_ours = eval_terms([enc(t) for t in doubles], _range_f12, arrs)
    m_zero = eval_terms([enc(t) for t in zero], _range_f12, arrs)

    dev = float(np.max(np.abs(m_s1 - m_ours)))
    zmax = float(np.max(np.abs(m_zero))) if zero else 0.0
    print("reference check (eq. (S1) of Masteran et al., JCP 158, 057101"
          " (2023)):")
    print(f"  assembled doubles expression: {len(doubles)} terms"
          f" (must be 16)")
    print(f"  max|(S1) - full_expression(doubles_only)| = {dev:.3e}")
    print(f"  max|space-restriction killed terms|       = {zmax:.3e}")
    ok = len(doubles) == 16 and dev < 1e-9 and zmax < 1e-12
    print(f"  [{'OK' if ok else 'FAIL'}]")
    print()
    if not ok:
        raise SystemExit("reference (S1) check failed")


def check_full_expression():
    """The stage-3 assembly (blocks + h.c., 1/2 on the Fock double
    commutator) == the direct nested commutator with the anti-Hermitian
    generator A = T - T^dagger (full virtual ranges, no restriction)."""
    vals = _random_tensors()
    # the internal identity holds for any t2; use an unrestricted one
    rng = np.random.default_rng(7)
    t2_free = rng.normal(size=(4, 4, 4, 4))
    t2_free = t2_free + t2_free.transpose(1, 0, 3, 2)
    arrs = _arrs(vals, t2_free)
    arrs["t1"] = vals["t1"]

    blocks = sr.build()
    assembled = sr.full_expression(blocks, field="terms")

    def enc_t(term):
        coeff, tensors, op, _ = term
        out = []
        for t in tensors:
            name = "t1" if t.name == "t" and len(t.upper) == 1 else t.name
            out.append((name, t.upper, t.lower))
        return (float(coeff), out,
                (tuple(op.upper_index), tuple(op.lower_index)))

    ev = lambda terms: eval_terms([enc_t(t) for t in terms],
                                  _range_full, arrs)
    m_ours = ev(assembled)

    def pool_mat(pool):
        return ev([pool[0]])

    h_mat = pool_mat(mbe.H_OP) + pool_mat(mbe.V_OP)
    f_mat = pool_mat(mbe.F_OP)
    a_mat = pool_mat(mbe._excitation_pool(1, 0)) \
        + pool_mat(mbe._excitation_pool(2, 0))
    a_mat = a_mat - a_mat.T

    def comm(x, y):
        return x @ y - y @ x

    direct = comm(h_mat, a_mat) + 0.5 * comm(comm(f_mat, a_mat), a_mat)
    dev = float(np.max(np.abs(m_ours - direct)))
    print("full-expression assembly vs direct [H_b, A] + 1/2[[F, A], A],"
          " A = T - T^dagger:")
    print(f"  max|Delta| = {dev:.3e}  [{'OK' if dev < 1e-9 else 'FAIL'}]")
    print()
    if dev >= 1e-9:
        raise SystemExit("full-expression assembly check failed")


if __name__ == "__main__":
    check_s1()
    check_full_expression()
