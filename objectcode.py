"""Second half of the edge0 builder: object code — splice, panel rewrites, code patches, gates.

Imported by build.py's main(); kept in a separate module so the data half stays readable.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent


def balance(line: str) -> tuple:
    """Net nesting: plain HTML tags, JSX components (<${...} ... <//>) and template backticks.

    Extra *balanced* markup is allowed (a rewrite may add a <small> pair); an unbalanced tag is
    what makes htm drop the surrounding element, so the nesting depth must not change.
    """
    depth = 0
    for m in re.finditer(r"</?[a-zA-Z][^<>]*>", line):
        tag = m.group(0)
        if tag.startswith("</"):
            depth -= 1
        elif not tag.endswith("/>"):
            depth += 1
    jsx = len(re.findall(r"<\$\{", line)) - line.count("<//>")
    return (depth, jsx, line.count("\u0060") % 2)


def apply_rewrites(text: str, rewrites) -> tuple[str, list[str]]:
    """Whole-line rewrites, anchored on a unique substring, with a structural balance check."""
    report = []
    lines = text.split("\n")
    for anchor, repl in rewrites:
        hits = [i for i, l in enumerate(lines) if anchor in l]
        if len(hits) != 1:
            raise SystemExit(f"anchor {anchor[:60]!r}: {len(hits)} lines (expected 1)")
        i = hits[0]
        a, b = balance(lines[i]), balance(repl)
        if a != b:
            raise SystemExit(f"rewrite {anchor[:50]!r} changes the line structure: net(a)={a} net(b)={b}")
        lines[i] = repl
        report.append(anchor[:56])
    return "\n".join(lines), report


def apply_literals(text: str, subs) -> tuple[str, list[str]]:
    report = []
    for old, new in subs:
        n = text.count(old)
        if n < 1:
            raise SystemExit(f"literal sub {old[:60]!r}: no matches (nothing to rewrite)")
        # copy fixes legitimately recur (a layer range shows up in the rail, the title and the guide);
        # replace every occurrence and report how many were touched.
        text = text.replace(old, new)
        report.append(f"{old[:58]}  x{n}")
    return text, report
