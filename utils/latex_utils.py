#!/usr/bin/env python3
"""Shared LaTeX helpers of the ct_derivation pipeline.

compile_tex is the pdf build of every stage (latexmk; skipped gracefully
when latexmk is absent, so the numerical checks never depend on it).
idx_latex renders one index symbol of the many-body stages (numbered
dummy letters: p1 -> p_{1}; a bare letter stays as is).
"""

import shutil
import subprocess


def idx_latex(x):
    """LaTeX of one index: p1 -> p_{1}; a bare letter stays as is."""
    name = str(x)
    base = name.rstrip("0123456789")
    num = name[len(base):]
    return "%s_{%s}" % (base, num) if num else base


def compile_tex(tex_path):
    """Build the pdf of a stage's tex file with latexmk, in the tex file's
    own directory.  When latexmk is not installed the build is skipped --
    the checks do not need it."""
    if shutil.which("latexmk") is None:
        print("latexmk not found, skipping PDF build")
        return
    proc = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", tex_path.name],
        cwd=tex_path.parent, capture_output=True, text=True)
    if proc.returncode == 0:
        print(f"built {tex_path.with_suffix('.pdf')}")
    else:
        print("latexmk failed:")
        print("\n".join(proc.stdout.splitlines()[-15:]))
        raise SystemExit(1)
