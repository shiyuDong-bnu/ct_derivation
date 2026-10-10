#!/usr/bin/env python3
"""Generate all commutators needed in CT theory and write them into LaTeX.

The commutators are generated symbolically with sympy's Commutator and kept
in nested-operator form; they are *not* expanded into concrete many-body
equations (that expansion is done elsewhere via the generalized Wick theorem
and cumulant expansion).

Operators (convention of ct_F_A.tex / ct_V_A.tex / ct_f_a_a.pdf):
    F = f^q_p E^p_q                    one-body (Fock) operator, Hermitian
    V = (1/2) g^{pq}_{rs} E^{rs}_{pq}  two-body (valence) operator, Hermitian
    T = T1 + T2                        excitation part of the generator
    A = T - T^dagger                   anti-Hermitian CT generator
    C = [F, A] = C1 + C2 + C1^dagger + C2^dagger,   Cr = [F, Tr]
    D = [V, A] = D1 + D2 + D1^dagger + D2^dagger,   Dr = [V, Tr]

The truncation is by perturbation order (Moller-Plesset counting F ~ O(1),
V ~ O(lambda), T ~ O(lambda)), keeping the CT expansion correct through
O(lambda^2): the first-order term [H, A] and the Fock half of the
second-order term, (1/2)[[F, A], A].  The valence double commutator
(1/2)[[V, A], A] is O(lambda^3) and is NOT generated.

Commutators generated:
    first order   [F, Tr], [V, Tr]                      (4 blocks, + h.c.)
    second order  [[F, Tr], Ts], [[F, Tr]^dagger, Ts]   (8 blocks, + h.c.)

Running the script also checks every reduction identity numerically on
random matrices, writes commutators.tex, and (if latexmk is available)
commutators.pdf.
"""

import shutil
import subprocess
from pathlib import Path

from sympy import Add, Mul, Rational, S, latex, sympify
from sympy.core.parameters import evaluate
from sympy.physics.quantum import Commutator, Dagger, Operator

HERE = Path(__file__).resolve().parent
TEX_PATH = HERE / "commutators.tex"

# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
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


F = Operator("F")
V = Operator("V")
T1 = Operator("T_1")
T2 = Operator("T_2")

OP_LATEX = {
    F: r"\hat{F}",
    V: r"\hat{V}",
    T1: r"\hat{T}_{1}",
    T2: r"\hat{T}_{2}",
}

# Blocks of the first-order commutators: the elementary objects everything
# else is built from.  Dagger(...) is kept unevaluated so that the conjugate
# blocks print as [F, Tr]^dagger instead of being rewritten by sympy.
with evaluate(False):
    T = T1 + T2
    A = T - Dagger(T)

    C1 = NCommutator(F, T1)
    C2 = NCommutator(F, T2)
    D1 = NCommutator(V, T1)
    D2 = NCommutator(V, T2)

    C1D = Dagger(C1)
    C2D = Dagger(C2)
    D1D = Dagger(D1)
    D2D = Dagger(D2)

    C = C1 + C2 + C1D + C2D
    D = D1 + D2 + D1D + D2D

# composite operators that print as their own symbol
PRESENT = {
    A: r"\hat{A}",
    T: r"\hat{T}",
    C: r"\hat{C}",
    D: r"\hat{D}",
}

# ---------------------------------------------------------------------------
# The commutators needed in CT theory
# ---------------------------------------------------------------------------
with evaluate(False):
    # first order: [F, A] = [F, T1] + [F, T2] + h.c., same for V
    first_order_F = [C1, C2]
    first_order_V = [D1, D2]

    # second order: only the Fock half is O(lambda^2); the valence double
    # commutator (1/2)[[V, A], A] is O(lambda^3) and is dropped by the
    # perturbation-order truncation.
    Ts = [T1, T2]
    second_order_F = [NCommutator(inner, ts)
                      for inner in (C1, C2, C1D, C2D) for ts in Ts]

    CT_second_F = Rational(1, 2) * NCommutator(NCommutator(F, A), A)
    CT_first_F = NCommutator(F, A)
    CT_first_V = NCommutator(V, A)

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

    subs = {F: rand_hermitian(), V: rand_hermitian(),
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

    # [X, A] = [X, T1] + [X, T2] + h.c.  (X Hermitian)
    with evaluate(False):
        for X, blocks, tag in ((F, first_order_F, "F"), (V, first_order_V, "V")):
            lhs = NCommutator(X, A)
            rhs = blocks[0] + blocks[1] + Dagger(blocks[0]) + Dagger(blocks[1])
            checks.append((f"[{tag}, A] = [{tag}, T1] + [{tag}, T2] + h.c.",
                           deviation(lhs, rhs)))

        # (1/2)[[F, A], A] = (1/2)([C, T] + [C, T]^dagger) over the 8 blocks
        for X, blocks, tag in ((F, second_order_F, "F"),):
            lhs = Rational(1, 2) * NCommutator(NCommutator(X, A), A)
            s8 = np.zeros((n, n), dtype=complex)
            for b in blocks:
                s8 = s8 + to_matrix(b)
            rhs_mat = 0.5 * (s8 + s8.conj().T)
            diff = to_matrix(lhs) - rhs_mat
            checks.append((f"(1/2)[[{tag}, A], A] = (1/2)(sum of 8 blocks + h.c.)",
                           float(np.max(np.abs(diff)))))

        # the two mixed blocks are different operators (no Jacobi shortcut)
        diff = to_matrix(NCommutator(C1, T2)) - to_matrix(NCommutator(C2, T1))
        mixed = float(np.max(np.abs(diff)))
        checks.append(("[[F, T1], T2] != [[F, T2], T1] (mixed blocks differ)",
                       mixed))

    print("numerical checks (random 6x6 matrices, F and V Hermitian):")
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
        r" $O(\lambda^2)$: the full first-order term $[\hat{H}, \hat{A}]$"
        r" ($O(\lambda) + O(\lambda^2)$) and the Fock half of the"
        r" second-order term, $\tfrac{1}{2}[[\hat{F}, \hat{A}], \hat{A}]$"
        r" ($O(\lambda^2)$).  The valence double commutator"
        r" $\tfrac{1}{2}[[\hat{V}, \hat{A}], \hat{A}]$ is $O(\lambda^3)$ and"
        r" is not generated.")
    add("")

    # --- definitions -----------------------------------------------------
    add(r"\section{Operators and generator}")
    add(r"Convention of \texttt{ct\_F\_A.tex}, \texttt{ct\_V\_A.tex} and"
        r" \texttt{ct\_f\_a\_a.pdf}: crossed Fock order, excitation operators,"
        r" general indices over $\{i\}\cup\{a\}$:")
    add(r"\begin{align}")
    add(r"\hat{F} &= f^q_p\, \hat{E}^p_q,"
        r" & \hat{V} &= \tfrac{1}{2} g^{pq}_{rs}\, \hat{E}^{rs}_{pq}, \\")
    add(r"\hat{T} &= \hat{T}_1 + \hat{T}_2,"
        r" & \hat{A} &= \hat{T} - \hat{T}^\dagger, \\")
    add(r"\hat{T}_1 &= t^a_i\, \hat{E}^a_i,"
        r" & \hat{T}_2 &= \tfrac{1}{2} t^{a\beta}_{ij}\, \hat{E}^{a\beta}_{ij},")
    add(r"\end{align}")
    add(r"with $\hat{F}$ and $\hat{V}$ Hermitian and $\hat{A}$ anti-Hermitian."
        r" The blocks of the first-order commutators are")
    add(r"\begin{align}")
    add(r"\hat{C}_1 &= " + ct_latex(C1) + r","
        r" & \hat{C}_2 &= " + ct_latex(C2) + r", \\")
    add(r"\hat{D}_1 &= " + ct_latex(D1) + r","
        r" & \hat{D}_2 &= " + ct_latex(D2) + r",")
    add(r"\end{align}")
    add(r"so that $\hat{C} = [\hat{F}, \hat{A}] = \hat{C}_1 + \hat{C}_2"
        r" + \hat{C}_1^\dagger + \hat{C}_2^\dagger$ and"
        r" $\hat{D} = [\hat{V}, \hat{A}] = \hat{D}_1 + \hat{D}_2"
        r" + \hat{D}_1^\dagger + \hat{D}_2^\dagger$.")
    add("")

    # --- first order -----------------------------------------------------
    add(r"\section{First-order commutators}")
    add(r"Since $\hat{A} = \hat{T} - \hat{T}^\dagger$ and"
        r" $[\hat{X}, \hat{T}]^\dagger = -[\hat{X}, \hat{T}^\dagger]$ for"
        r" Hermitian $\hat{X}$, only the two $\hat{T}$ blocks have to be"
        r" computed for each part of the Hamiltonian:")
    add(r"\begin{align}")
    add(ct_latex(CT_first_F) + r" &= "
        + ct_latex(C1) + " + " + ct_latex(C2) + r" \hc, \\")
    add(ct_latex(CT_first_V) + r" &= "
        + ct_latex(D1) + " + " + ct_latex(D2) + r" \hc.")
    add(r"\end{align}")
    add("")

    # --- second order ----------------------------------------------------
    add(r"\section{Second-order commutators}")
    add(r"At $O(\lambda^2)$ the second-order term keeps only the Fock half,"
        r" $\tfrac{1}{2}[[\hat{F}, \hat{A}], \hat{A}]$; the valence half"
        r" $\tfrac{1}{2}[[\hat{V}, \hat{A}], \hat{A}] \sim O(\lambda^3)$ is"
        r" dropped.  $\hat{A}$ anti-Hermitian makes"
        r" $\hat{C} = [\hat{F}, \hat{A}]$ Hermitian, so that"
        r" $[\hat{C}, \hat{T}]^\dagger = -[\hat{C}, \hat{T}^\dagger]$ and")
    add(r"\begin{equation}")
    add(ct_latex(CT_second_F) + r" = \tfrac{1}{2}"
        + ct_latex(NCommutator(C, T)) + r" \hc,")
    add(r"\end{equation}")
    add(r"Expanding"
        r" $[\hat{C}, \hat{T}] = [\hat{C}_1, \hat{T}] + [\hat{C}_2, \hat{T}]"
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

    # --- full list -------------------------------------------------------
    add(r"\section{Complete list}")
    add(r"The commutators that remain to be evaluated (each by one or two"
        r" applications of the $(1,1)$, $(1,2)$, $(2,1)$ or $(2,2)$ rule):")
    add(r"\paragraph{First order (4 blocks $+$ h.c.)}")
    add(r"\begin{align}")
    all_first = first_order_F + first_order_V
    for i, b in enumerate(all_first):
        end = r" \\" if i < len(all_first) - 1 else ""
        add(ct_latex(b) + end)
    add(r"\end{align}")
    add(r"\paragraph{Second order (8 blocks $+$ h.c.)}")
    add(r"\begin{align}")
    for i, b in enumerate(second_order_F):
        end = r" \\" if i < len(second_order_F) - 1 else ""
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
