#!/usr/bin/env python3
"""Parser for the Supplemental-Material tex source of Masteran et al.
(arXiv:2211.09685; the source lives in the repo at
reference/supporting.tex, the compiled pdf at reference/supporting.pdf):
turns the highly regular equation bodies of eq:H (S1), eq:h (S3) and
eq:g (S4) into term lists.

A term is (Rational coeff, [factors], E) where a factor is
  ("t", name, (up atoms), (lo atoms))   tensor h/f/g/G/D (D = density matrix)
  ("d", up_atom, lo_atom)               Kronecker delta
and E is ((up atoms), (lo atoms)) or None (for the coefficient equations
(S3)/(S4) the E is reconstructed from the deltas by the caller).

Atoms: ("i", n) occupied, ("a", n) OBS-vir, ("x", n) CABS,
("al", n) complete virtual, ("p", n) numbered whole-OBS p'_n,
("P",0)/("Q",0)/("R",0)/("S",0) the free primed indices p',q',r',s'.

Intermediates V, V~ (eqs. S6/S7) and X, B (eqs. 24/25 of Yanai-Shiozaki,
JCP 136, 084107 (2012)) are expanded inline with fresh complete-virtual
dummy atoms ("al", 90+...):

  V^{pq}_{ij}   = g^{pq}_{a b} G^{a b}_{ij}
  V~^{ij}_{pq}  = g^{a b}_{pq} G^{ij}_{a b}
  X^{kl}_{ij}   = G^{a b}_{ij} G^{a b}_{kl}
  B^{kl}_{ij}   = G^{a b}_{ij} f^{c}_{a} G^{c b}_{kl}
                + G^{a b}_{ij} f^{b}_{c} G^{a c}_{kl}
"""

import re
from fractions import Fraction
from pathlib import Path

SRC = str(Path(__file__).resolve().parent.parent / "reference"
          / "supporting.tex")

_ATOM = re.compile(r"\\alpha_(\d+)"               # \alpha_N  (first: contains a_N)
                   r"|p\^\{?\\prime\}?_(\d+)"     # {p^\prime}_N or {p^{\prime}_N}
                   r"|\{([iax])_(\d+)\}"          # {i_1} {a_1} {x_1}
                   r"|\{p_(\d+)\}"                # {p_1} (plain whole-space)
                   r"|([pqrs])\^\\prime")         # bare p' q' r' s'


def _atoms(content):
    out = []
    for m in _ATOM.finditer(content):
        if m.group(1):
            out.append(("al", int(m.group(1))))
        elif m.group(2):
            out.append(("p", int(m.group(2))))
        elif m.group(3):
            out.append((m.group(3), int(m.group(4))))
        elif m.group(5):
            out.append(("p", int(m.group(5))))
        else:
            out.append((m.group(6).upper(), 0))
    return out


def _groups(s):
    """Top-level brace groups of s: list of (content, start, end)."""
    out = []
    depth = 0
    for k, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                st = k
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                out.append((s[st + 1:k], st, k + 1))
    return out


def _factor(content):
    """Parse one top-level group of a term into a factor tuple."""
    c = content.strip()
    # factors may be doubly braced: {{\frac{1}{2}}}, {{2}}, {{h^{..}_{..}}}
    # unwrap while the whole content is exactly one brace group
    while c.startswith("{") and c.endswith("}"):
        gs = _groups(c)
        if len(gs) == 1 and gs[0][1] == 0 and gs[0][2] == len(c):
            c = gs[0][0].strip()
        else:
            break
    m = re.fullmatch(r"\\frac\{(\d+)\}\{(\d+)\}", c)
    if m:
        return ("num", Fraction(int(m.group(1)), int(m.group(2))))
    if re.fullmatch(r"\d+", c):
        return ("num", Fraction(int(c)))
    if c.startswith("\\hat{E}"):
        sub = c[len("\\hat{E}"):]
        ups, lows = [], []
        # split ^ {...} and _ {...} groups in order
        pos = 0
        parts = []
        while pos < len(sub):
            if sub[pos] in "^_":
                kind = sub[pos]
                g = _groups(sub[pos + 1:])
                content_g = g[0][0]
                parts.append((kind, content_g))
                pos += 1 + g[0][2]
            else:
                pos += 1
        for kind, cg in parts:
            (ups if kind == "^" else lows).extend(_atoms(cg))
        return ("E", tuple(ups), tuple(lows))
    if c.startswith("\\delta"):
        sub = c[len("\\delta"):]
        pos = 0
        parts = []
        while pos < len(sub):
            if sub[pos] in "^_":
                g = _groups(sub[pos + 1:])
                parts.append((sub[pos], g[0][0]))
                pos += 1 + g[0][2]
            else:
                pos += 1
        up = _atoms(parts[0][1])
        lo = _atoms(parts[1][1])
        assert len(up) == 1 and len(lo) == 1, c
        return ("d", up[0], lo[0])
    # tensor: name then ^{..} / _{..} groups in order
    if c.startswith("\\tilde{V}"):
        name = "Vt"
        rest = c[len("\\tilde{V}"):]
    else:
        name = c[0]
        assert name in "hgfGDVXB", c
        rest = c[1:]
    pos = 0
    ups, lows = [], []
    while pos < len(rest):
        if rest[pos] in "^_":
            g = _groups(rest[pos + 1:])
            (ups if rest[pos] == "^" else lows).extend(_atoms(g[0][0]))
            pos += 1 + g[0][2]
        else:
            pos += 1
    return ("t", name, tuple(ups), tuple(lows))


def _expand_intermediates(factors, counter):
    """Replace V, Vt, X, B factors by their definitions with fresh
    complete-virtual dummies; B splits one term into two."""
    def fresh():
        counter[0] += 1
        return ("al", 90 + counter[0])

    expanded = [[]]
    for f in factors:
        if f[0] != "t" or f[1] not in ("V", "Vt", "X", "B"):
            expanded = [e + [f] for e in expanded]
            continue
        name, up, lo = f[1], f[2], f[3]
        if name == "V":            # V^{pq}_{ij} = g^{pq}_{ab} G^{ab}_{ij}
            a, b = fresh(), fresh()
            new = [[("t", "g", up, (a, b)), ("t", "G", (a, b), lo)]]
        elif name == "Vt":         # V~^{ij}_{pq} = g^{ab}_{pq} G^{ij}_{ab}
            a, b = fresh(), fresh()
            new = [[("t", "g", (a, b), lo), ("t", "G", up, (a, b))]]
        elif name == "X":          # X^{kl}_{ij} = G^{ab}_{ij} G^{ab}_{kl}
            a, b = fresh(), fresh()
            new = [[("t", "G", (a, b), lo), ("t", "G", (a, b), up)]]
        else:                      # B: two pieces
            a, b, c = fresh(), fresh(), fresh()
            a2, b2, c2 = fresh(), fresh(), fresh()
            new = [[("t", "G", (a, b), lo), ("t", "f", (c,), (a,)),
                    ("t", "G", (c, b), up)],
                   [("t", "G", (a2, b2), lo), ("t", "f", (b2,), (c2,)),
                    ("t", "G", (a2, c2), up)]]
        expanded = [e + n for e in expanded for n in new]
    return expanded


def parse_equation(label, expand=True):
    r"""Parse one align body of the SI into a term list.

    Returns [ (Rational coeff, [tensor factors], [delta factors], E or
    None) ].  The body may be wrapped in \hat{S}( ... ) (eq:g): the
    wrapper is stripped (the symmetrizer is redundant at the operator
    level by the diagonal pair symmetry of E)."""
    tex = open(SRC).read()
    start = tex.index("\\label{%s}" % label)
    end = tex.index("\\end{align}", start)
    body = tex[start:end]
    body = body.split("=", 1)[1]           # drop the LHS
    if "\\approx" in body:                 # E3 equations: \overset{..}{\approx}
        body = body.split("\\approx", 1)[1]
    body = body.replace("\\bigg", " ").replace("\\big", " ")
    body = body.replace("\\nonumber", "").replace("\\\\", " ")
    body = body.replace("\\hat{S}", " ")
    # strip one outer pair of parentheses if present (the \hat{S} wrapper)
    if "(" in body:
        i = body.index("(")
        depth = 0
        for k in range(i, len(body)):
            if body[k] == "(":
                depth += 1
            elif body[k] == ")":
                depth -= 1
                if depth == 0:
                    body = body[:i] + " " + body[i + 1:k] + " " + body[k + 1:]
                    break
    # split into signed chunks at top level: + { / - {
    terms = []
    depth = 0
    sign = 1
    buf = ""
    started = False
    for ch in body:
        if depth == 0 and ch in "+-":
            if started:
                terms.append((sign, buf))
                buf = ""
                started = False
            sign = 1 if ch == "+" else -1
            continue
        if depth == 0 and ch in "& \n\t":
            continue
        if depth == 0 and ch == "{":
            started = True
        if not started:
            continue
        buf += ch
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
    if buf.strip():
        terms.append((sign, buf))

    counter = [1000]
    out = []
    for sign, chunk in terms:
        chunk = chunk.strip().rstrip(".,;").strip()
        gs = _groups(chunk)
        if len(gs) == 1 and gs[0][2] == len(chunk):
            chunk = gs[0][0]          # terms are wrapped: {{ f1 f2 ... }}
        coeff = Fraction(sign)
        factors = [_factor(c) for c, _, _ in _groups(chunk)]
        nums = [f for f in factors if f[0] == "num"]
        for nf in nums:
            coeff *= nf[1]
        tensors = [f for f in factors if f[0] == "t"]
        deltas = [f for f in factors if f[0] == "d"]
        es = [f for f in factors if f[0] == "E"]
        if expand:
            pieces = _expand_intermediates(tensors, counter)
        else:
            pieces = [tensors]
        for ts in pieces:
            num = Fraction(1)
            ts2 = []
            for f in ts:
                if f[0] == "num":
                    num *= f[1]
                else:
                    ts2.append(f)
            out.append((coeff * num, ts2, deltas,
                        (es[0][1], es[0][2]) if es else None))
    return out


if __name__ == "__main__":
    for label in ("eq:H", "eq:h", "eq:g"):
        ts = parse_equation(label)
        n_t = sum(len(t[1]) for t in ts)
        print(f"{label}: {len(ts)} terms, {n_t} tensor factors")
