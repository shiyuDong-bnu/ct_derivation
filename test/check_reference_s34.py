#!/usr/bin/env python3
"""Numerical comparison with eqs. (S3)/(S4) of the Supplemental Material of
Masteran, Kumar, Teke, Gaudel, Yanai and Valeev, "Comment on 'Canonical
transcorrelated theory with projected Slater-type geminals'", J. Chem.
Phys. 158, 057101 (2023) -- the dressed one- and two-body matrix elements

    H_bar = hbar^{p'}_{q'} E^{q'}_{p'} + (1/2) gbar^{p'q'}_{r's'} E^{r's'}_{p'q'}
            (+ scalar),

parsed from the SI tex source (reference/supporting.tex, parser:
test/si_parse.py; the V, V~, X, B intermediates are expanded inline by
their definitions).

Our side: the substituted final formula of the f12_special_term stage
(stage-6 cumulant-expanded blocks + the F12 special term), doubles part,
assembled blocks + h.c. with 1/2 on the Fock double commutator, plus the
bare H = h + V.

Matrix-level comparison on 4 spatial orbitals {0,1 occupied; 2 OBS-vir;
3 CABS} with genuinely spin-summed Jordan-Wigner operators and the
single-determinant reference density matrices.  The terms are compared
per perturbation order of the geminal (0 = bare, 1 = first order
h.G / g.G, 2 = second order f.G.G): the bare and first-order content must
match EXACTLY (assumption-free); the second-order content of the printed
(S3)/(S4) carries the F12-intermediate/screening structure of the SI's
SeQuant driver and is NOT reproduced by the plain truncation -- the
residual is reported, in three tensor setups:

  * random Fock + SP-restricted geminal (pair-swap and pair-exchange
    symmetric, zero both-OBS-vir block),
  * the same with the geminal in true SP form (cusp-condition
    coefficients of a random f12 integral tensor),
  * the same with a canonical (diagonal) Fock matrix.
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
sys.path.insert(0, str(HERE))

import expand_commutators as mbe  # noqa: E402
from f12_special_term import assembled  # noqa: E402
from check_f12_special_term import e_sf, eval_terms, f12_tensors, DIM  # noqa: E402
from si_parse import parse_equation  # noqa: E402


# ---------------------------------------------------------------------------
# their side: (S3) -> hbar[P,Q], (S4) -> gbar[P,Q,R,S]
# ---------------------------------------------------------------------------
def atom_range(a):
    kind, _ = a
    if kind in ("P", "Q", "R", "S"):
        return (0, 1, 2, 3)
    return {"i": (0, 1), "a": (2,), "x": (3,), "al": (2, 3),
            "p": (0, 1, 2, 3)}[kind]


def n_geminal(tensors):
    """Perturbation order of a parsed SI term: its number of geminal
    factors (after the intermediates are expanded)."""
    return sum(1 for _t, name, _u, _l in tensors if name == "G")


def eval_si(terms, nfree, arrs):
    """Evaluate the coefficient array of (S3) (nfree=2) or (S4) (nfree=4):
    the free indices p',q'(,r',s') are tied into each term by its deltas."""
    shape = (4,) * nfree
    out = np.zeros(shape)
    for coeff, tensors, deltas, e in terms:
        assert e is None
        ties = {}
        if nfree == 2:
            for _d, up, lo in deltas:
                if lo == ("Q", 0):
                    ties[up] = 1      # delta^{X}_{q'}: X = q' (axis 1)
                elif up == ("P", 0):
                    ties[lo] = 0      # delta^{p'}_{Y}: Y = p' (axis 0)
                else:
                    raise ValueError(deltas)
        else:
            for _d, up, lo in deltas:
                if lo == ("R", 0):
                    ties[up] = 2
                elif lo == ("S", 0):
                    ties[up] = 3
                elif up == ("P", 0):
                    ties[lo] = 0
                elif up == ("Q", 0):
                    ties[lo] = 1
                else:
                    raise ValueError(deltas)
        # atoms may also BE the free indices directly (the bare-h term)
        for atom, axis in ((("P", 0), 0), (("Q", 0), 1),
                           (("R", 0), 2), (("S", 0), 3)):
            if axis < nfree:
                ties.setdefault(atom, axis)
        atoms = []
        for _t, _n, up, lo in tensors:
            atoms.extend(up + lo)
        dummies = sorted({a for a in atoms if a not in ties})
        ranges = [atom_range(a) for a in dummies]
        # a tied dummy only contributes when the free value lies in ITS
        # range (delta^{a1}_{q'} vanishes for q' outside OBS-vir, etc.)
        tie_atoms = sorted(ties, key=lambda a: ties[a])
        tie_ok = [atom_range(a) if a[0] not in "PQRS" else (0, 1, 2, 3)
                  for a in tie_atoms]
        for free_vals in itertools.product(range(4), repeat=nfree):
            if any(free_vals[ties[a]] not in ok
                   for a, ok in zip(tie_atoms, tie_ok)):
                continue
            fv = {atom: free_vals[axis] for atom, axis in ties.items()}
            tot = 0.0
            for vals in itertools.product(*ranges):
                val = dict(zip(dummies, vals))
                val.update(fv)
                factor = float(coeff)
                for _t, name, up, lo in tensors:
                    idx = tuple(val[a] for a in up + lo)
                    if name == "D":
                        arr = arrs["D1"] if len(up) == 1 else arrs["D2"]
                    else:
                        arr = arrs[name]
                    factor *= arr[idx]
                tot += factor
            out[free_vals] += tot
    return out


def si_matrix(hbar, gbar):
    """hbar^{p'}_{q'} E^{q'}_{p'} + (1/2) gbar^{p'q'}_{r's'} E^{r's'}_{p'q'}."""
    m = np.zeros((DIM, DIM))
    for P, Q in itertools.product(range(4), repeat=2):
        if abs(hbar[P, Q]) > 1e-15:
            m += hbar[P, Q] * e_sf([Q], [P])
    for P, Q, R, S in itertools.product(range(4), repeat=4):
        if abs(gbar[P, Q, R, S]) > 1e-15:
            m += 0.5 * gbar[P, Q, R, S] * e_sf([R, S], [P, Q])
    return m


# ---------------------------------------------------------------------------
# tensor setups
# ---------------------------------------------------------------------------
def sp_form_geminal(rng):
    """The geminal in true SP form: cusp-condition coefficients
    (3/8) <ab|Q12 f12|ij> + (1/8) <ab|Q12 f12|ji> of a random f12 integral
    tensor (g-like symmetries), Q12 removing the both-OBS-vir block."""
    f12 = rng.normal(size=(4, 4, 4, 4))
    f12 = f12 + f12.transpose(1, 0, 3, 2) + f12.transpose(2, 3, 0, 1) \
        + f12.transpose(3, 2, 1, 0)
    f12[2, 2, :, :] = 0.0
    f12[:, :, 2, 2] = 0.0
    return (3.0 / 8.0) * f12 + (1.0 / 8.0) * f12.transpose(0, 1, 3, 2)


def setups():
    base = f12_tensors()
    rng = np.random.default_rng(20261013)
    gem_sp = sp_form_geminal(rng)
    f_can = np.diag(np.random.default_rng(20261014).normal(size=4))
    yield "random Fock, SP-restricted geminal", dict(base)
    yield "random Fock, SP-form geminal", {**base, "t": gem_sp,
                                           "G": gem_sp}
    yield "canonical Fock, SP-form geminal", {**base, "t": gem_sp,
                                              "G": gem_sp, "f": f_can}


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------
def our_matrices(arrs):
    """Our side, per geminal order: bare (H), first (one t), second
    (two t) -- the substituted final formula, doubles part, assembled."""
    total = assembled("total", doubles_only=True)
    by_order = {1: [], 2: []}
    for term in total:
        n_t = sum(1 for t in term[1] if t.name == "t")
        by_order[n_t].append(term)
    bare = eval_terms(list(mbe.H_OP) + list(mbe.V_OP), arrs)
    first = eval_terms(by_order[1], arrs)
    second = eval_terms(by_order[2], arrs)
    ranks = {}
    for term in total:
        ranks.setdefault(term[2].rank, []).append(term)
    scalar = float(np.max(np.abs(eval_terms(ranks.get(0, []), arrs)))) \
        if 0 in ranks else 0.0
    return {0: bare, 1: first, 2: second}, ranks, scalar


def their_matrices(arrs):
    """The parsed (S3)/(S4), per geminal order."""
    si_h = parse_equation("eq:h")
    si_g = parse_equation("eq:g")
    out = {}
    for order in (0, 1, 2):
        h_terms = [t for t in si_h if n_geminal(t[1]) == order]
        g_terms = [t for t in si_g if n_geminal(t[1]) == order]
        hbar = eval_si(h_terms, 2, arrs)
        gbar = eval_si(g_terms, 4, arrs)
        out[order] = si_matrix(hbar, gbar)
    return out


def check_s34():
    print("reference check (eqs. (S3)/(S4) of Masteran et al., JCP 158,"
          " 057101 (2023); dressed hbar/gbar):")
    print("  ours = substituted final formula (stage 6 + F12 special term),"
          " doubles part + bare H")
    si_h = parse_equation("eq:h")
    si_g = parse_equation("eq:g")
    print(f"  parsed: (S3) {len(si_h)} terms, (S4) {len(si_g)} terms "
          "(intermediates V, V~, X, B expanded)")
    all_ok = True
    for label, arrs in setups():
        arrs = {**arrs, "G": arrs["t"]}
        ours, ranks, scalar = our_matrices(arrs)
        theirs = their_matrices(arrs)
        dev = {o: float(np.max(np.abs(ours[o] - theirs[o])))
               for o in (0, 1, 2)}
        tot = float(np.max(np.abs(sum(ours.values())
                                  - sum(theirs.values()))))
        print(f"\n  setup: {label}")
        print(f"    bare H (no geminal):      max|Delta| = {dev[0]:.3e}")
        print(f"    first order (h.G, g.G):   max|Delta| = {dev[1]:.3e}")
        print(f"    second order (f.G.G):     max|Delta| = {dev[2]:.3e}"
              "   <- known residual: the SI's F12-intermediate/screening"
              " content")
        print(f"    total:                    max|Delta| = {tot:.3e}")
        print(f"    scalar part on our side:  max|.|     = {scalar:.3e}"
              " (should vanish)")
        ok = dev[0] < 1e-9 and dev[1] < 1e-9 and scalar < 1e-9
        all_ok &= ok
        print(f"    bare + first order + no scalar: "
              f"[{'OK' if ok else 'FAIL'}]")
    print()
    if not all_ok:
        raise SystemExit("reference (S3)/(S4) check failed: the bare or"
                         " first-order content does not match exactly")


if __name__ == "__main__":
    check_s34()
