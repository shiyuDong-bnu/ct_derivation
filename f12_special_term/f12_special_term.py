#!/usr/bin/env python3
"""F12 special term: substitute the fixed-geminal doubles back into the
cumulant-expanded result (final stage).

The stage-4 space restriction killed every term carrying a t2 amplitude
whose two virtual indices BOTH sit on the E operator (the free, external
indices of the equation): in the CF-F12 construction those amplitudes are
not independent -- they are the fixed F12 geminal amplitudes of the SP
ansatz,

    t~^{ab}_{ij} = (3/8) <ab|Q12 f12|ij> + (1/8) <ab|Q12 f12|ji>

(a, b over the complete virtual space;  Q12 the strong-orthogonality
projector, so the geminal vanishes with both indices OBS-vir).  The killed
terms are exactly the contribution of those fixed geminal amplitudes to
the transcorrelated Hamiltonian: the "F12 special term".  This stage
brings them back and substitutes them into the cumulant-expanded result
of stage 6:

    * the killed terms of every block are taken from the stage-4 build
      (``space_restriction.build()``, field ``zero``);
    * the three-body ones are folded back to two-/one-body pieces with the
      same grouped cumulant template as stage 6
      (``cumulant_expansion.decompose``;  the all-virtual-upper pattern
      E^{abc}_{ijk} of [[F,T2],T2] has no surviving group: every piece of
      the decomposition would carry a virtual density matrix);
    * the special terms are added to the stage-6 final blocks and the
      blocks are assembled the way the commutator stage defines: blocks +
      Hermitian conjugates, the second-order Fock blocks carrying the 1/2
      of the double commutator, merged once
      (``space_restriction.full_expression``).

Checks (test/check_f12_special_term.py, run by this script):

    * complement: killed + survivors = the stage-3 block, per block;
    * the cumulant folding of the killed three-body patterns (the template
      identity with the virtual rule, Jordan--Wigner matrices, SD
      reference);
    * the assembled special expression vanishes in the F12 setting
      (free virtuals OBS-vir, geminal amplitudes with both OBS-vir zero):
      substituting it back does not touch the restricted result -- eq.
      (S1) compatibility.

The SI comparison against eqs. (S3)/(S4) of Masteran et al., J. Chem.
Phys. 158, 057101 (2023) lives in test/check_reference_s34.py and is also
run by this script.  The result is written to f12_special_term.tex/.pdf.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "many_body_equation"))
sys.path.insert(0, str(HERE.parent / "space_restriction"))
sys.path.insert(0, str(HERE.parent / "cumulant_expansion"))
sys.path.insert(0, str(HERE.parent / "utils"))

import expand_commutators as mbe  # noqa: E402
import space_restriction as sr  # noqa: E402
import cumulant_expansion as ce  # noqa: E402
from latex_utils import compile_tex  # noqa: E402

TEX_PATH = HERE / "f12_special_term.tex"

_BLOCKS = None


def special_blocks():
    """Per commutator block: the F12 special terms (the terms the stage-4
    space restriction killed), the three-body ones folded back with the
    cumulant template, and the block total (stage-6 final + special)."""
    global _BLOCKS
    if _BLOCKS is not None:
        return _BLOCKS
    out = []
    for n, (blk, fin) in enumerate(zip(sr.build(), ce.final_blocks()), 1):
        killed = list(blk["zero"])
        three = [t for t in killed if t[2].rank == 3]
        small = [t for t in killed if t[2].rank < 3]
        decomposed = []
        for t in three:
            decomposed.extend(ce.decompose(t))
        special = mbe.merge(small + decomposed) if three else killed
        total = mbe.merge(list(fin["final"]) + special)
        out.append({"n": n, "kind": blk["kind"], "lhs": blk["lhs"],
                    "killed": killed, "three_body": three,
                    "decomposed": mbe.merge(decomposed) if decomposed else [],
                    "special": special, "final": fin["final"],
                    "total": total})
    _BLOCKS = out
    return out


def assembled(field, doubles_only=False):
    """The blocks assembled into the transcorrelated Hamiltonian through
    O(lambda^2) (blocks + h.c., 1/2 on the second-order Fock blocks,
    merged once -- the assembly of the space-restriction stage), with
    ``field`` picking the per-block term list: "special" for the F12
    special term alone, "total" for the substituted final result."""
    return sr.full_expression(special_blocks(), doubles_only=doubles_only,
                              field=field)


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------
def _align(rows):
    L = [r"\begin{align}"]
    for row in rows[:-1]:
        L.append(row + r" \\")
    L.append(rows[-1])
    L.append(r"\end{align}")
    return L


def write_tex(blocks):
    special_all = assembled("special")
    special_dbl = assembled("special", doubles_only=True)
    total_all = assembled("total")
    L = []
    add = L.append
    add(r"\documentclass[11pt]{article}")
    add(r"\usepackage{amsmath,amssymb}")
    add(r"\usepackage[margin=2.5cm]{geometry}")
    add(r"\allowdisplaybreaks")
    add(r"\title{The F12 special term and the substituted final formula}")
    add(r"\date{generated by \texttt{f12\_special\_term.py}}")
    add(r"\begin{document}")
    add(r"\maketitle")
    add("")
    add(r"The stage-4 space restriction killed every term carrying a $t_2$"
        r" amplitude whose two virtual indices both sit on the operator"
        r" $\hat E$ (the free, external indices of the equation).  In the"
        r" CF-F12 construction those amplitudes are not independent: they"
        r" are the fixed F12 geminal amplitudes of the SP ansatz,")
    add(r"\begin{align}")
    add(r"\tilde t^{ab}_{ij} = \tfrac{3}{8}\langle ab|\hat Q_{12} f_{12}"
        r"|ij\rangle + \tfrac{1}{8}\langle ab|\hat Q_{12} f_{12}|ji\rangle,")
    add(r"\end{align}")
    add(r"with $a,b$ over the complete virtual space and $\hat Q_{12}$ the"
        r" strong-orthogonality projector (the geminal vanishes with both"
        r" indices OBS-vir, which is what made the stage-4 rule exact).  The"
        r" killed terms are the contribution of these fixed geminal"
        r" amplitudes to the transcorrelated Hamiltonian -- the \emph{F12"
        r" special term}.  This stage brings them back: the killed"
        r" three-body terms are folded back to two-/one-body pieces with"
        r" the grouped cumulant template of stage 6 (a density matrix with"
        r" a virtual index is zero; the pure-density scalar part always"
        r" vanishes), and the result is substituted into the"
        r" cumulant-expanded blocks of stage 6.")
    add("")
    add(r"\section{The F12 special terms of each block}")
    add(r"Only the blocks with killed terms are shown.  The killed terms"
        r" come first; when a block has killed three-body terms, their"
        r" cumulant-folded form follows, then the special block (killed"
        r" terms and folded pieces merged).")
    for blk in blocks:
        if not blk["killed"]:
            continue
        add(r"\subsection*{%s\quad(%d killed terms)}"
            % ("$" + blk["lhs"] + "$", len(blk["killed"])))
        add(r"killed by the space restriction:")
        L.extend(_align(mbe.pack_rows(blk["lhs"],
                                        mbe.terms_latex(blk["killed"]))))
        if blk["three_body"]:
            add(r"the %d three-body terms fold into %d terms by the"
                r" cumulant template:"
                % (len(blk["three_body"]), len(blk["decomposed"])))
            if blk["decomposed"]:
                L.extend(_align(mbe.pack_rows(
                    "3\\text{-body part}",
                    mbe.terms_latex(blk["decomposed"]))))
            else:
                add(r"\begin{align}")
                add(r"3\text{-body part} &= 0 \quad (\text{no template group"
                    r" carries all three virtual indices})")
                add(r"\end{align}")
            add(r"the special block (killed terms and folded pieces merged,"
                r" %d terms):" % len(blk["special"]))
            if blk["special"]:
                L.extend(_align(mbe.pack_rows(
                    blk["lhs"] + r"^{F12}", mbe.terms_latex(blk["special"]))))
            else:
                add(r"\begin{align}")
                add(blk["lhs"] + r"^{F12} &= 0")
                add(r"\end{align}")
    add("")
    add(r"\section{The assembled F12 special term}")
    add(r"Assembled the way the commutator stage defines the"
        r" transcorrelated Hamiltonian: the blocks plus their Hermitian"
        r" conjugates (the anti-Hermitian generator $\hat A = \hat T - \hat"
        r" T^\dagger$), the second-order Fock blocks carrying the factor"
        r" $1/2$, merged once (%d terms):" % len(special_all))
    L.extend(_align(mbe.pack_rows(r"\bar H^{F12}_{\mathrm{special}}",
                                    mbe.terms_latex(special_all))))
    add(r"Every term carries a geminal amplitude with both virtual indices"
        r" on the operator, so the whole expression vanishes when the free"
        r" virtuals are OBS-vir (the SP geminal has no both-OBS-vir"
        r" amplitudes) -- substituting it back does not touch the"
        r" restricted result, eq. (S1) of the reference.  The pure-$T_2$"
        r" part (the doubles contribution of the geminal operator $\hat"
        r" A^{F12}$, %d terms):" % len(special_dbl))
    L.extend(_align(mbe.pack_rows(r"\bar H^{F12}_{\mathrm{special}}"
                                    r"\Big|_{\mathrm{doubles}}",
                                    mbe.terms_latex(special_dbl))))
    add("")
    add(r"\section{The substituted final formula}")
    add(r"The stage-6 result with the F12 special term substituted back,"
        r" assembled (blocks + h.c., $1/2$ on the Fock double commutator;"
        r" %d terms) -- the complete transcorrelated Hamiltonian through"
        r" $O(\lambda^2)$ with the cumulant-folded three-body part and the"
        r" fixed geminal amplitudes:" % len(total_all))
    L.extend(_align(mbe.pack_rows(r"\bar H - \hat H",
                                    mbe.terms_latex(total_all))))
    add(r"\end{document}")
    TEX_PATH.write_text("\n".join(L) + "\n")
    print(f"wrote {TEX_PATH}")


if __name__ == "__main__":
    # register the running script under its module name, so that the check
    # modules import this very instance and not a second copy (the term
    # machinery relies on Tensor/Operator class identity)
    sys.modules.setdefault("f12_special_term", sys.modules["__main__"])
    sys.path.insert(0, str(HERE.parent / "test"))
    from check_f12_special_term import (  # noqa: E402
        check_complement, check_killed_three_body, check_special_vanishes)
    blocks = special_blocks()
    print("F12 special term (stage-4 killed terms, cumulant-folded):")
    for blk in blocks:
        if not blk["killed"]:
            continue
        extra = (f"  [{len(blk['three_body'])} three-body -> "
                 f"{len(blk['decomposed'])} folded]" if blk["three_body"]
                 else "")
        print(f"  {blk['n']:2d}. {len(blk['killed']):3d} killed -> "
              f"{len(blk['special']):3d} special;  final "
              f"{len(blk['final']):3d} -> total {len(blk['total']):3d}"
              f"{extra}  {blk['lhs'][:50]}")
    print()
    print(f"assembled special term: {len(assembled('special'))} terms "
          f"(doubles only: {len(assembled('special', doubles_only=True))})")
    print(f"substituted final formula: {len(assembled('total'))} terms "
          f"(doubles only: {len(assembled('total', doubles_only=True))})")
    print()
    check_complement(blocks)
    check_killed_three_body(blocks)
    check_special_vanishes()
    from check_reference_s34 import check_s34  # noqa: E402
    check_s34()
    write_tex(blocks)
    compile_tex(TEX_PATH)
