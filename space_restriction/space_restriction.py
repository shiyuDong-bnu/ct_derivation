#!/usr/bin/env python3
"""Space restriction to the OBS virtual space (CF-F12 of Yanai).

Following readme.md: in the CF-F12 construction the virtual space splits
into OBS-vir and CABS+infinity, and we restrict to OBS-vir.  The amplitude
t^{ab}_{ij} vanishes when both of its indices are OBS-vir, so in the
many-body expansions of ``many_body_equation`` (taken from that stage's
public API, ``expand_commutators.expanded_blocks()``):

    * a virtual index that is contracted with another tensor is OK -- the
      contraction sum still runs over the full virtual space;
    * a term is zero iff both virtual indices of ONE AND THE SAME t2
      amplitude sit on the E operator -- the free indices of the operator
      are the external (OBS-vir) indices of the equation, so the term
      carries the vanishing amplitude.  Two free virtual indices that
      stem from two DIFFERENT amplitudes do not kill the term.

For every commutator block this script writes the original expansion with
the vanishing terms underlined, then the restricted expansion gathering the
nonzero terms, and reports per block how many terms vanish.

``full_expression()`` then assembles the surviving blocks into the full
transcorrelated Hamiltonian through O(lambda^2) the way the commutator
stage defines it: blocks + Hermitian conjugates (the anti-Hermitian
generator A = T - T^dagger), the second-order Fock blocks carrying the
1/2 of the double commutator, merged once.  Its doubles-only part is eq.
(S1) of the Supplemental Material of Masteran et al., J. Chem. Phys. 158,
057101 (2023) (reference/supporting.pdf);  the identity is verified
numerically in test/check_reference_s1.py, which this script runs before
writing the tex.
"""

import sys
from pathlib import Path

from sympy import Rational

HERE = Path(__file__).resolve().parent
MBE = HERE.parent / "many_body_equation"
sys.path.insert(0, str(MBE))
sys.path.insert(0, str(HERE.parent / "utils"))

import expand_commutators as mbe  # noqa: E402
from latex_utils import compile_tex  # noqa: E402

TEX_PATH = HERE / "space_restriction.tex"


# ---------------------------------------------------------------------------
# the restriction rule
# ---------------------------------------------------------------------------
def is_zero(term):
    """Restriction rule: both virtual indices of one and the same t2
    amplitude sit on the E operator (as free, hence OBS-vir, external
    indices), so the term carries the vanishing amplitude.  Contracted
    virtuals and virtuals from different amplitudes are fine.

    The t2 amplitude is identified by its index spaces (a ``t`` tensor
    carrying exactly two virtual indices, per the ``SPACE`` table of the
    many_body_equation stage), wherever they sit on the tensor; a virtual
    index sits on the operator when it appears in either row of E."""
    _, tensors, op, _ = term
    op_idx = set(op.upper_index) | set(op.lower_index)
    for t in tensors:
        if t.name != "t":
            continue
        vir = [x for x in t.upper + t.lower if mbe.space_of(x) == "v"]
        if len(vir) == 2 and all(x in op_idx for x in vir):
            return True
    return False


# ---------------------------------------------------------------------------
# build the blocks on the many_body_equation API
# ---------------------------------------------------------------------------
def build():
    """The blocks of ``expand_commutators.expanded_blocks()``, each with
    its terms split into vanishing (``zero``) and surviving (``keep``)
    ones by the restriction rule."""
    blocks = []
    for blk in mbe.expanded_blocks():
        terms = blk["terms"]
        zero = [t for t in terms if is_zero(t)]
        keep = [t for t in terms if not is_zero(t)]
        blocks.append({"kind": blk["kind"], "lhs": blk["lhs"],
                       "terms": terms, "zero": zero, "keep": keep})
    return blocks


# ---------------------------------------------------------------------------
# the full expression: restricted blocks + h.c.
# ---------------------------------------------------------------------------
def full_expression(blocks, doubles_only=False, field="keep"):
    """The transcorrelated Hamiltonian through O(lambda^2), assembled from
    the blocks the way the commutator stage defines it

        H_bar = H + [H_b, A] + (1/2)[[F, A], A],   A = T - T^dagger
              = H + sum(first-order blocks) + h.c.
                + (1/2) (sum of the 8 second-order Fock blocks) + h.c.,

    i.e. every block is added together with its Hermitian conjugate (the
    anti-Hermitian generator of the commutator stage), the second-order
    Fock blocks carrying the 1/2 of the double commutator.  ``field``
    picks which term list of each block dict is assembled ("keep" for the
    restricted expression, "terms" for the unrestricted stage-3 one).  The
    assembled sum is merged once (``mbe.merge``): Hermitian-conjugate
    pairs with equal canonical form fold into one term, self-conjugate
    terms keeping coefficient 1 through the 1/2 -- exactly the term
    structure of eq. (S1) of Masteran et al. (see below).

    ``doubles_only=True`` keeps only the pure-T2 blocks (r = 2, s in
    (None, 2)) -- the F12 geminal operator A^F12 = T2 - T2^dagger;  the
    assembled restricted expression is then eq. (S1) of the Supplemental
    Material of Masteran, Kumar, Teke, Gaudel, Yanai and Valeev,
    J. Chem. Phys. 158, 057101 (2023) (reference/supporting.pdf), the
    complete transcorrelated Hamiltonian without the cumulant
    decomposition.  Verified term by term and numerically in
    test/check_reference_s1.py.
    """
    assembled = []
    for blk, meta in zip(blocks, mbe.expanded_blocks()):
        if doubles_only and not (meta["r"] == 2 and meta["s"] in (None, 2)):
            continue
        mult = Rational(1, 2) if blk["kind"] == "second" else Rational(1)
        terms = [(mult * c, ts, op, lo) for c, ts, op, lo in blk[field]]
        assembled.extend(terms)
        assembled.extend(mbe.dagger_terms(terms))
    return mbe.merge(assembled)


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------
def term_latex_marked(term, first_term):
    """Term latex; zero terms underlined."""
    s = mbe.term_latex(*term, first_term)
    if is_zero(term):
        return r"\underline{" + s + "}"
    return s


def block_original_lines(terms):
    def key(term):
        _, tensors, op, _ = term
        return (op.rank,
                [str(u) for u in op.upper_index],
                [str(l) for l in op.lower_index],
                [t.latex() for t in tensors])
    ordered = sorted(terms, key=key)
    return [term_latex_marked(t, n == 0) for n, t in enumerate(ordered)]


def write_tex(blocks):
    L = []
    add = L.append
    add(r"\documentclass[11pt]{article}")
    add(r"\usepackage{amsmath,amssymb}")
    add(r"\usepackage[margin=2.5cm]{geometry}")
    add(r"\allowdisplaybreaks")
    add(r"\title{Space restriction to the OBS virtual space}")
    add(r"\date{generated by \texttt{space\_restriction.py}}")
    add(r"\begin{document}")
    add(r"\maketitle")
    add("")
    add(r"In the CF-F12 construction of Yanai the virtual space splits into"
        r" the OBS virtual space and the CABS$+\infty$ remainder.  We restrict"
        r" the virtual space to OBS-vir: every external virtual index is an"
        r" OBS-vir index, and the amplitude $t^{ab}_{ij}$ vanishes when both"
        r" of its indices are OBS-vir.  In the many-body expansions of"
        r" \texttt{many\_body\_equations.tex} (taken from the"
        r" \texttt{expanded\_blocks()} API of \texttt{expand\_commutators.py})"
        r" this means:")
    add(r"\begin{itemize}")
    add(r"\item a virtual index contracted with another tensor is OK -- the"
        r" contraction sum still runs over the full virtual space;")
    add(r"\item a term is zero iff both virtual indices of \emph{one and the"
        r" same} $t^{ab}_{ij}$ amplitude sit on the operator"
        r" $\hat{E}$ -- the free indices of the operator are the external"
        r" (OBS-vir) indices of the equation, so the term carries the"
        r" vanishing amplitude.  Two free virtual indices stemming from"
        r" \emph{different} amplitudes do not kill the term.")
    add(r"\end{itemize}")
    add(r"Each block is shown first as the original expansion with the"
        r" vanishing terms \underline{underlined}, then as the restricted"
        r" expansion gathering the nonzero terms.  Blocks still containing"
        r" operators of rank higher than two are marked; their cumulant"
        r" expansion is done later.")
    add("")

    sections = [("First-order commutators", "first"),
                ("Second order: Fock blocks", "second")]
    for title, want_kind in sections:
        add(r"\section{%s}" % title)
        for blk in blocks:
            if blk["kind"] != want_kind:
                continue
            terms, keep = blk["terms"], blk["keep"]
            n_zero = len(terms) - len(keep)
            add(r"\subsection*{%s\quad(%d terms, %d vanish)}"
                % ("$" + blk["lhs"] + "$", len(terms), n_zero))
            # original, zero terms underlined
            lines = block_original_lines(terms)
            rows = mbe.pack_rows(blk["lhs"], lines)
            add(r"\begin{align}")
            for row in rows[:-1]:
                add(row + r" \\")
            add(rows[-1])
            add(r"\end{align}")
            # restricted
            add(r"restricted:")
            if keep:
                lines = mbe.terms_latex(keep)
                big = [n for n in mbe.ranks_of(keep) if n > 2]
                marker = ""
                if big:
                    names = {3: "three-body", 4: "four-body"}
                    marker = (r"\quad (\text{%s terms})"
                              % " and ".join(names[n] for n in big))
                rows = mbe.pack_rows(blk["lhs"], lines)
                add(r"\begin{align}")
                for row in rows[:-1]:
                    add(row + r" \\")
                add(rows[-1] + marker)
                add(r"\end{align}")
            else:
                add(r"\begin{align}")
                add(blk["lhs"] + r" &= 0")
                add(r"\end{align}")
        add("")

    add(r"\section{The full expression in the OBS-vir restriction}")
    add(r"The commutator stage assembles the blocks into the transcorrelated"
        r" Hamiltonian through $O(\lambda^2)$ as the blocks plus their"
        r" Hermitian conjugates (the anti-Hermitian generator $\hat A ="
        r" \hat T - \hat T^\dagger$), the Fock double-commutator blocks"
        r" carrying the factor $1/2$ (see the Assembly section of"
        r" \texttt{many\_body\_equations.tex}).  Assembling the"
        r" \emph{restricted} blocks the same way and merging over the exact"
        r" identities (conjugate pairs with equal canonical form fold into"
        r" one term, self-conjugate terms keeping coefficient 1 through the"
        r" $1/2$) gives the transcorrelated Hamiltonian in the OBS-vir"
        r" restriction:")
    full = full_expression(blocks)
    lines = mbe.terms_latex(full)
    rows = mbe.pack_rows(r"\bar H - \hat H", lines)
    add(r"\begin{align}")
    for row in rows[:-1]:
        add(row + r" \\")
    add(rows[-1])
    add(r"\end{align}")
    add(r"Keeping only the pure-$T_2$ blocks (the F12 geminal operator"
        r" $\hat A^{F12} = \hat T_2 - \hat T_2^\dagger$;  the geminal"
        r" amplitudes carry the pair-exchange symmetry"
        r" $G^{\alpha\beta}_{ij} = G^{ij}_{\alpha\beta}$ and vanish with"
        r" both indices OBS-vir) this is eq. (S1) of the Supplemental"
        r" Material of Masteran et al., J. Chem. Phys. 158, 057101 (2023),"
        r" the complete transcorrelated Hamiltonian obtained without the"
        r" cumulant decomposition -- verified term by term and numerically"
        r" in \texttt{test/check\_reference\_s1.py}:")
    doubles = full_expression(blocks, doubles_only=True)
    lines = mbe.terms_latex(doubles)
    rows = mbe.pack_rows(r"\bar H^{F12} - \hat H", lines)
    add(r"\begin{align}")
    for row in rows[:-1]:
        add(row + r" \\")
    add(rows[-1])
    add(r"\end{align}")
    add(r"\end{document}")

    TEX_PATH.write_text("\n".join(L) + "\n")
    print(f"wrote {TEX_PATH}")


if __name__ == "__main__":
    # register the running script under its module name, so that the check
    # module imports this very instance and not a second copy (the term
    # machinery relies on Tensor/Operator class identity)
    sys.modules.setdefault("space_restriction", sys.modules["__main__"])
    sys.path.insert(0, str(HERE.parent / "test"))
    from check_reference_s1 import check_full_expression, check_s1

    blocks = build()
    print("space restriction (zero = both virtuals of one and the same t2 "
          "amplitude on the E operator):")
    for n, blk in enumerate(blocks, 1):
        terms, keep, zero = blk["terms"], blk["keep"], blk["zero"]
        print(f"  {n:2d}. {len(terms):3d} terms -> {len(zero):3d} vanish, "
              f"{len(keep):3d} survive   {blk['lhs'][:60]}")
    print()
    check_s1()
    check_full_expression()
    write_tex(blocks)
    compile_tex(TEX_PATH)
