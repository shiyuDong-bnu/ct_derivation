#!/usr/bin/env python3
r"""Generate all commutators needed in CT theory and write them into LaTeX.

The commutators are generated symbolically with sympy's Commutator and kept
in nested-operator form; they are *not* expanded into concrete many-body
equations (that expansion is done elsewhere via the generalized Wick theorem
and cumulant expansion).

The script works in three stages:

1. Elementary operators (have a direct many-body form, convention of
   ct_F_A.tex / ct_V_A.tex / ct_f_a_a.pdf):
       h  = h^q_p E^p_q                     one-body  operator, Hermitian
       v  = (1/2) g^{pq}_{rs} E^{rs}_{pq}   two-body  operator, Hermitian
       f  = f^q_p E^p_q                     one-body (Fock) operator, Hermitian
       t1 = t^a_i E^a_i                     one-body (singles) excitation
       t2 = (1/2) t^{ab}_{ij} E^{ab}_{ij}   two-body (doubles) excitation

2. Composite operators (sums/products of the elementary ones):
       H_b    = h + v                              bare Hamiltonian
       t      = t1 + t2                            excitation operator
       a      = t - t^dagger                       anti-Hermitian CT generator
       \bar H = H_b + [H_b, a] + (1/2)[[f, a], a]  transcorrelated Hamiltonian

3. Expansion of the target operator \bar H into commutators between
   elementary operators:
       first order   [h, tr], [v, tr]                      (4 blocks, + h.c.)
       second order  [[f, tr], ts], [[f, tr]^dagger, ts]   (8 blocks, + h.c.)

The truncation is by perturbation order (Moller-Plesset counting f ~ O(1),
v ~ O(lambda), t ~ O(lambda)), keeping the CT expansion correct through
O(lambda^2): the full first-order term [H_b, a] and the Fock half of the
second-order term, (1/2)[[f, a], a].  The valence double commutator
(1/2)[[v, a], a] is O(lambda^3) and is not generated.

Running the script also checks every reduction identity numerically on
random matrices, writes commutators.tex, and (if latexmk is available)
commutators.pdf.

Public API for the many-body equations:
    elementary_operators()    -- the five elementary operators [h, V, F, T1, T2]
    elementary_definitions()  -- their many-body definitions (sympy exprs)
    commutator_blocks()       -- all commutator blocks between them, as a
                                 dict {"first_order": 4 blocks,
                                       "second_order": 8 blocks}
"""

import shutil
import subprocess
from pathlib import Path

from sympy import Add, Mul, Rational, S, Symbol, latex, sympify
from sympy.core.parameters import evaluate
from sympy.physics.quantum import Commutator, Dagger, Operator

HERE = Path(__file__).resolve().parent
TEX_PATH = HERE / "commutators.tex"


class NCommutator(Commutator):
    """sympy Commutator without canonical reordering.

    sympy's Commutator.eval swaps the arguments ([V, T1] -> -[T1, V]) to get
    a canonical form; for presentation we want the argument order as written.
    Everything else (expansion, printing, subclassing) is sympy's.
    """

    @classmethod
    def eval(cls, a, b):
        if not (a and b):
            return S.Zero
        if a == b:
            return S.Zero
        if a.is_commutative or b.is_commutative:
            return S.Zero
        return None


# ---------------------------------------------------------------------------
# Stage 1: elementary operators
#
# The ones with a direct many-body (second-quantized) definition.
# Everything else is built from these five.
# ---------------------------------------------------------------------------
H = Operator("h")
V = Operator("V")
F = Operator("F")
T1 = Operator("T_1")
T2 = Operator("T_2")

OP_LATEX = {
    H: r"\hat{h}",
    V: r"\hat{V}",
    F: r"\hat{F}",
    T1: r"\hat{T}_{1}",
    T2: r"\hat{T}_{2}",
}

# many-body (second-quantized) definitions of the elementary operators:
# right-hand sides of "op = ...", as sympy expressions (coefficient symbols
# times unitary group generators E), in the convention of ct_F_A.tex /
# ct_V_A.tex / ct_f_a_a.pdf (crossed Fock order, excitation operators,
# general indices over {i} u {a})
OP_DEFINITION = {
    H: Symbol("h^q_p") * Operator("E^p_q"),
    V: Rational(1, 2) * Symbol("g^{pq}_{rs}") * Operator("E^{rs}_{pq}"),
    F: Symbol("f^q_p") * Operator("E^p_q"),
    T1: Symbol("t^a_i") * Operator("E^a_i"),
    T2: Rational(1, 2) * Symbol("t^{ab}_{ij}") * Operator("E^{ab}_{ij}"),
}

# ---------------------------------------------------------------------------
# Stage 2: composite operators
#
# Sums/products/commutators of the elementary ones.  Dagger(...) is kept
# unevaluated so that conjugates print as (...)^dagger instead of being
# rewritten by sympy.
# ---------------------------------------------------------------------------
with evaluate(False):
    HB = H + V               # bare Hamiltonian
    T = T1 + T2              # excitation operator
    A = T - Dagger(T)        # anti-Hermitian CT generator

    CT_first_H = NCommutator(H, A)
    CT_first_V = NCommutator(V, A)
    CT_first_HB = NCommutator(HB, A)
    CT_second_F = Rational(1, 2) * NCommutator(NCommutator(F, A), A)

    # the target operator, in nested composite form
    BAR_H = HB + CT_first_HB + CT_second_F

# ---------------------------------------------------------------------------
# Stage 3: expand \bar H into commutators between elementary operators
#
# First order:  [h, A] = [h, T1] + [h, T2] + h.c., same for V.
# Second order: only the Fock half is O(lambda^2); the valence double
# commutator (1/2)[[V, A], A] is O(lambda^3) and is dropped by the
# perturbation-order truncation.
# ---------------------------------------------------------------------------
with evaluate(False):
    # first-order blocks
    B1 = NCommutator(H, T1)
    B2 = NCommutator(H, T2)
    D1 = NCommutator(V, T1)
    D2 = NCommutator(V, T2)
    first_order_H = [B1, B2]
    first_order_V = [D1, D2]

    # blocks of [F, A], the inner commutator of the second-order term
    C1 = NCommutator(F, T1)
    C2 = NCommutator(F, T2)
    C1D = Dagger(C1)
    C2D = Dagger(C2)

    # second-order blocks
    Ts = [T1, T2]
    second_order_F = [NCommutator(inner, ts)
                      for inner in (C1, C2, C1D, C2D) for ts in Ts]

    # Hermitian block sums: [X, A] = [X, T1] + [X, T2] + h.c.
    B = B1 + B2 + Dagger(B1) + Dagger(B2)    # [h, A]
    D = D1 + D2 + Dagger(D1) + Dagger(D2)    # [V, A]
    C = C1 + C2 + C1D + C2D                  # [F, A]

    # the target operator, expanded into elementary commutators:
    #   \bar H = h + V
    #            + [h, T1] + [h, T2] + [V, T1] + [V, T2] + h.c.
    #            + (1/2)(sum of the 8 nested blocks + h.c.)
    S8 = sum(second_order_F[1:], second_order_F[0])
    BAR_H_EXPANDED = (H + V + B + D
                      + Rational(1, 2) * (S8 + Dagger(S8)))

# composite operators that print as their own symbol
PRESENT = {
    HB: r"\hat{H}_{b}",
    T: r"\hat{T}",
    A: r"\hat{A}",
    B: r"\hat{B}",
    BAR_H: r"\bar{\hat{H}}",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def elementary_operators():
    """Return the list of elementary operators, [h, V, F, T1, T2].

    These are the operators with a direct many-body (second-quantized)
    definition; every other object is composite (a sum, product or
    commutator of these five).
    """
    return [H, V, F, T1, T2]


def elementary_definitions():
    """Return the many-body definitions of the elementary operators.

    Maps each operator in elementary_operators() to the sympy expression on
    the right-hand side of its second-quantized definition, e.g.
    T1 -> t^a_i*E^a_i.
    """
    return dict(OP_DEFINITION)


def commutator_blocks():
    """Return all commutator blocks between elementary operators.

    These are the blocks appearing in the expansion of the target operator
    BAR_H; they are the quantities the many-body equations (generalized
    Wick theorem + cumulant expansion) have to evaluate.  Returns a dict:

        "first_order":  [B1, B2, D1, D2]
            [h, T1], [h, T2], [V, T1], [V, T2] -- enter BAR_H with their
            Hermitian conjugates at O(lambda) + O(lambda^2).
        "second_order": 8 nested blocks
            [[F, Tr], Ts] and [[F, Tr]^dagger, Ts] (r, s = 1, 2) -- enter
            BAR_H as (1/2)(sum + h.c.) at O(lambda^2).
    """
    return {
        "first_order": list(first_order_H + first_order_V),
        "second_order": list(second_order_F),
    }


# ---------------------------------------------------------------------------
# LaTeX printer for the operator expressions
# ---------------------------------------------------------------------------
def is_dagger(expr):
    """True for either sympy Dagger class (physics.quantum and elementary).

    Unevaluated expressions rebuilt by as_coeff_Mul and friends come back as
    sympy.functions.elementary.complexes.Dagger, a different class than the
    physics one, so the check is by name.
    """
    return type(expr).__name__ == "Dagger" or expr.func.__name__ == "Dagger"


def split_mul(expr):
    """Split a (possibly unevaluated) Mul into (numeric coefficient, rest)
    without rebuilding the factors."""
    coeff = S.One
    rest = []
    for arg in expr.args:
        if arg.is_Number:
            coeff *= arg
        else:
            rest.append(arg)
    if not rest:
        return coeff, S.One
    if len(rest) == 1:
        return coeff, rest[0]
    return coeff, Mul(*rest, evaluate=False)


def ct_latex(expr):
    """Print our Commutator/Dagger/Operator expressions as LaTeX."""
    # named composite operators print as their symbol
    for named, tex in PRESENT.items():
        if expr == named:
            return tex
    if isinstance(expr, Commutator):
        return (r"\left[" + ct_latex(expr.args[0]) + r",\, "
                + ct_latex(expr.args[1]) + r"\right]")
    if is_dagger(expr):
        inner = expr.args[0]
        s = ct_latex(inner)
        if isinstance(inner, Add):
            s = r"\left(" + s + r"\right)"
        return s + r"^{\dagger}"
    if isinstance(expr, Operator):
        return OP_LATEX[expr]
    if isinstance(expr, Add):
        parts = []
        for term in expr.args:
            s = ct_latex(term)
            if s.startswith("-"):
                parts.append("- " + s[1:].lstrip())
            else:
                parts.append("+ " + s)
        joined = " ".join(parts)
        return joined[2:] if joined.startswith("+ ") else joined
    if isinstance(expr, Mul):
        coeff, rest = split_mul(expr)
        if rest == 1:
            return latex(coeff)
        body = ct_latex(rest)
        if isinstance(rest, Add):
            body = r"\left(" + body + r"\right)"
        if coeff == -1:
            return "-" + body
        if coeff == 1:
            return body
        return latex(coeff) + r"\, " + body
    return latex(sympify(expr))


# ---------------------------------------------------------------------------
# Numerical check of every reduction identity on random matrices
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# LaTeX document
# ---------------------------------------------------------------------------
def write_tex():
    L = []
    add = L.append
    add(r"\documentclass[11pt]{article}")
    add(r"\usepackage{amsmath,amssymb}")
    add(r"\usepackage[margin=2.5cm]{geometry}")
    add(r"\newcommand{\hc}{+\mathrm{h.c.}}")
    add(r"\title{All commutators needed in CT theory}")
    add(r"\date{generated by \texttt{generate\_commutators.py}}")
    add(r"\begin{document}")
    add(r"\maketitle")
    add("")
    add(r"The commutators of the transcorrelated (CT) expansion,"
        r" generated symbolically with sympy's \texttt{Commutator} and kept in"
        r" nested-operator form.  They are \emph{not} expanded into concrete"
        r" many-body equations; the expansion through the generalized Wick"
        r" theorem and the cumulant expansion is done separately.")
    add("")
    add(r"The truncation is by perturbation order (M\o ller--Plesset"
        r" counting $\hat{F} \sim O(1)$, $\hat{V} \sim O(\lambda)$,"
        r" $\hat{T} \sim O(\lambda)$), keeping the expansion correct through"
        r" $O(\lambda^2)$: the full first-order term $[\hat{H}_b, \hat{A}]$"
        r" ($O(\lambda) + O(\lambda^2)$) and the Fock half of the"
        r" second-order term, $\tfrac{1}{2}[[\hat{F}, \hat{A}], \hat{A}]$"
        r" ($O(\lambda^2)$).  The valence double commutator"
        r" $\tfrac{1}{2}[[\hat{V}, \hat{A}], \hat{A}]$ is $O(\lambda^3)$ and"
        r" is not generated.")
    add("")

    # --- stage 1: elementary operators ------------------------------------
    add(r"\section{Elementary operators}")
    add(r"The elementary operators have a direct many-body form"
        r" (convention of \texttt{ct\_F\_A.tex}, \texttt{ct\_V\_A.tex} and"
        r" \texttt{ct\_f\_a\_a.pdf}: crossed Fock order, excitation"
        r" operators, general indices over $\{i\}\cup\{a\}$):")
    add(r"\begin{align}")
    add(r"\hat{h} &= h^q_p\, \hat{E}^p_q,"
        r" & \hat{V} &= \tfrac{1}{2} g^{pq}_{rs}\, \hat{E}^{rs}_{pq}, \\")
    add(r"\hat{F} &= f^q_p\, \hat{E}^p_q,"
        r" & \hat{T}_1 &= t^a_i\, \hat{E}^a_i, \\")
    add(r"\hat{T}_2 &= \tfrac{1}{2} t^{ab}_{ij}\, \hat{E}^{ab}_{ij},")
    add(r"\end{align}")
    add(r"with $\hat{h}$, $\hat{V}$ and $\hat{F}$ Hermitian.")
    add("")

    # --- stage 2: composite operators --------------------------------------
    add(r"\section{Composite operators}")
    add(r"The composite operators are sums and products of the elementary"
        r" ones:")
    add(r"\begin{align}")
    add(r"\hat{H}_b &= \hat{h} + \hat{V},"
        r" & \hat{T} &= \hat{T}_1 + \hat{T}_2, \\")
    add(r"\hat{A} &= \hat{T} - \hat{T}^\dagger,"
        r" & \bar{\hat{H}} &= \hat{H}_b + \left[\hat{H}_b,\, \hat{A}\right]"
        r" + \tfrac{1}{2}\left[\left[\hat{F},\, \hat{A}\right],\,"
        r" \hat{A}\right],")
    add(r"\end{align}")
    add(r"with $\hat{A}$ anti-Hermitian.  The transcorrelated Hamiltonian"
        r" $\bar{\hat{H}}$ is the target operator.  The blocks of the"
        r" first-order commutators are")
    add(r"\begin{align}")
    add(r"\hat{B}_1 &= " + ct_latex(B1) + r","
        r" & \hat{B}_2 &= " + ct_latex(B2) + r", \\")
    add(r"\hat{D}_1 &= " + ct_latex(D1) + r","
        r" & \hat{D}_2 &= " + ct_latex(D2) + r", \\")
    add(r"\hat{C}_1 &= " + ct_latex(C1) + r","
        r" & \hat{C}_2 &= " + ct_latex(C2) + r",")
    add(r"\end{align}")
    add(r"so that $[\hat{h}, \hat{A}] = \hat{B}_1 + \hat{B}_2"
        r" + \hat{B}_1^\dagger + \hat{B}_2^\dagger$, and similarly for"
        r" $[\hat{V}, \hat{A}]$ and $[\hat{F}, \hat{A}]$.")
    add("")

    # --- first order -------------------------------------------------------
    add(r"\section{First-order commutators}")
    add(r"The first-order term of $\bar{\hat{H}}$ involves the composite"
        r" bare Hamiltonian $\hat{H}_b = \hat{h} + \hat{V}$,")
    add(r"\begin{equation}")
    add(ct_latex(CT_first_HB) + r" = "
        + ct_latex(CT_first_H) + " + " + ct_latex(CT_first_V) + r".")
    add(r"\end{equation}")
    add(r"Since $\hat{A} = \hat{T} - \hat{T}^\dagger$ and"
        r" $[\hat{X}, \hat{T}]^\dagger = -[\hat{X}, \hat{T}^\dagger]$ for"
        r" Hermitian $\hat{X}$, only the two $\hat{T}$ blocks have to be"
        r" computed for each elementary part of $\hat{H}_b$:")
    add(r"\begin{align}")
    add(ct_latex(CT_first_H) + r" &= "
        + ct_latex(B1) + " + " + ct_latex(B2) + r" \hc, \\")
    add(ct_latex(CT_first_V) + r" &= "
        + ct_latex(D1) + " + " + ct_latex(D2) + r" \hc.")
    add(r"\end{align}")
    add("")

    # --- second order ------------------------------------------------------
    add(r"\section{Second-order commutators}")
    add(r"At $O(\lambda^2)$ the second-order term keeps only the Fock half,"
        r" $\tfrac{1}{2}[[\hat{F}, \hat{A}], \hat{A}]$; the valence half"
        r" $\tfrac{1}{2}[[\hat{V}, \hat{A}], \hat{A}] \sim O(\lambda^3)$ is"
        r" dropped.  $\hat{A}$ anti-Hermitian makes"
        r" $[\hat{F}, \hat{A}]$ Hermitian, so that"
        r" $[[\hat{F}, \hat{A}], \hat{T}]^\dagger ="
        r" -[[\hat{F}, \hat{A}], \hat{T}^\dagger]$ and")
    add(r"\begin{equation}")
    add(ct_latex(CT_second_F) + r" = \tfrac{1}{2}"
        + ct_latex(NCommutator(NCommutator(F, A), T)) + r" \hc,")
    add(r"\end{equation}")
    add(r"Expanding"
        r" $[[\hat{F}, \hat{A}], \hat{T}] = [\hat{C}_1, \hat{T}]"
        r" + [\hat{C}_2, \hat{T}]"
        r" + [\hat{C}_1^\dagger, \hat{T}] + [\hat{C}_2^\dagger, \hat{T}]$"
        r" and each $[\,\cdot\,, \hat{T}] = [\,\cdot\,, \hat{T}_1]"
        r" + [\,\cdot\,, \hat{T}_2]$ gives the eight blocks")
    add(r"\begin{align}")
    for i, b in enumerate(second_order_F):
        end = r" \\" if i < len(second_order_F) - 1 else ""
        add(ct_latex(b) + end)
    add(r"\end{align}")
    add(r"The mixed nested commutators are different operators, $"
        + ct_latex(second_order_F[1]) + r" \neq "
        + ct_latex(second_order_F[2]) + r"$, since"
        r" $\hat{T}_1 \hat{T}_2 - \hat{T}_2 \hat{T}_1$ does not vanish;"
        r" the Jacobi shortcut is not available and both mixed blocks"
        r" are kept.")
    add("")

    # --- stage 3: expansion of the target operator --------------------------
    add(r"\section{Expansion of the target operator}")
    add(r"Assembling the composite pieces and expanding every commutator of"
        r" composite operators into commutators between elementary operators,"
        r" the transcorrelated Hamiltonian through $O(\lambda^2)$ is")
    add(r"\begin{align}")
    add(ct_latex(BAR_H) + r" &= " + ct_latex(HB) + " + "
        + ct_latex(CT_first_HB) + " + " + ct_latex(CT_second_F) + r" \\")
    add(r"&= " + ct_latex(H) + " + " + ct_latex(V)
        + " + " + " + ".join(ct_latex(b)
                             for b in first_order_H + first_order_V)
        + r" \hc \nonumber \\")
    s8_terms = " + ".join(ct_latex(b) for b in second_order_F[:4])
    s8d_terms = " + ".join(ct_latex(b) for b in second_order_F[4:])
    add(r"&\quad + \tfrac{1}{2}\left(" + s8_terms + r" \right. \nonumber \\")
    add(r"&\qquad \left. + " + s8d_terms + r" \hc \right). \nonumber")
    add(r"\end{align}")
    add(r"Every remaining object is a commutator between elementary"
        r" operators ($\hat{h}$, $\hat{V}$, $\hat{F}$ with $\hat{T}_1$,"
        r" $\hat{T}_2$); these are the quantities that the generalized Wick"
        r" theorem and the cumulant expansion turn into many-body equations.")
    add("")

    # --- full list ----------------------------------------------------------
    add(r"\section{Complete list}")
    add(r"The commutators that remain to be evaluated (each by one or two"
        r" applications of the $(1,1)$, $(1,2)$, $(2,1)$ or $(2,2)$ rule),"
        r" as returned by \texttt{commutator\_blocks()}:")
    blocks = commutator_blocks()
    all_first = blocks["first_order"]
    all_second = blocks["second_order"]
    add(r"\paragraph{First order (4 blocks $+$ h.c.)}")
    add(r"\begin{align}")
    for i, b in enumerate(all_first):
        end = r" \\" if i < len(all_first) - 1 else ""
        add(ct_latex(b) + end)
    add(r"\end{align}")
    add(r"\paragraph{Second order (8 blocks $+$ h.c.)}")
    add(r"\begin{align}")
    for i, b in enumerate(all_second):
        end = r" \\" if i < len(all_second) - 1 else ""
        add(ct_latex(b) + end)
    add(r"\end{align}")
    add(r"\end{document}")

    TEX_PATH.write_text("\n".join(L) + "\n")
    print(f"wrote {TEX_PATH}")


def compile_tex():
    if shutil.which("latexmk") is None:
        print("latexmk not found, skipping PDF build")
        return
    proc = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", TEX_PATH.name],
        cwd=HERE, capture_output=True, text=True)
    if proc.returncode == 0:
        print(f"built {TEX_PATH.with_suffix('.pdf')}")
    else:
        print("latexmk failed:")
        print("\n".join(proc.stdout.splitlines()[-15:]))
        raise SystemExit(1)


if __name__ == "__main__":
    check_identities()
    write_tex()
    compile_tex()
