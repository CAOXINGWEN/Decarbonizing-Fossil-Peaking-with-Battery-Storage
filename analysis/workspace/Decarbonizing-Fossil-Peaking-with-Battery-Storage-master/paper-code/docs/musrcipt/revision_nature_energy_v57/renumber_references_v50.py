#!/usr/bin/env python3
"""Renumber manual Nature-style citations by first appearance in the v50 main text."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "musript_new_v50.tex"
SUPP = ROOT / "musript_new_v50_supplementary.tex"
REFS = ROOT / "references_v50.tex"
AUDIT = ROOT / "V50_REFERENCE_RENUMBER_AUDIT.json"

CITE_RE = re.compile(r"\[(\d+(?:--\d+)?(?:,\d+(?:--\d+)?)*)\]")


def expand(spec: str) -> list[int]:
    values: list[int] = []
    for token in spec.split(","):
        if "--" in token:
            start, end = map(int, token.split("--"))
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    return values


def compress(values: list[int]) -> str:
    ordered = sorted(set(values))
    groups: list[str] = []
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1] == ordered[j] + 1:
            j += 1
        run = ordered[i : j + 1]
        if len(run) >= 3:
            groups.append(f"{run[0]}--{run[-1]}")
        else:
            groups.extend(str(value) for value in run)
        i = j + 1
    return ",".join(groups)


def parse_reference_items(text: str) -> tuple[str, list[str], str]:
    before, rest = text.split("\\begin{enumerate}[leftmargin=2.2em,itemsep=0.10em]", 1)
    body, after = rest.split("\\end{enumerate}", 1)
    items = ["\\item " + item.strip() for item in body.split("\\item ")[1:]]
    return before, items, after


main_text = MAIN.read_text(encoding="utf-8")
supp_text = SUPP.read_text(encoding="utf-8")
refs_text = REFS.read_text(encoding="utf-8")
before, items, after = parse_reference_items(refs_text)

appearance: list[int] = []
for match in CITE_RE.finditer(main_text):
    for old_id in expand(match.group(1)):
        if old_id not in appearance:
            appearance.append(old_id)

if any(old_id < 1 or old_id > len(items) for old_id in appearance):
    raise RuntimeError("Citation outside the available reference range")

mapping = {old_id: new_id for new_id, old_id in enumerate(appearance, start=1)}


def rewrite_citations(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        old_values = expand(match.group(1))
        missing = [value for value in old_values if value not in mapping]
        if missing:
            raise RuntimeError(f"Supplementary citation has no main-text reference: {missing}")
        return "[" + compress([mapping[value] for value in old_values]) + "]"

    return CITE_RE.sub(repl, text)


MAIN.write_text(rewrite_citations(main_text), encoding="utf-8")
SUPP.write_text(rewrite_citations(supp_text), encoding="utf-8")

new_items = [items[old_id - 1] for old_id in appearance]
new_refs = (
    before
    + "\\begin{enumerate}[leftmargin=2.2em,itemsep=0.10em]\n"
    + "\n".join(new_items)
    + "\n\\end{enumerate}"
    + after
)
REFS.write_text(new_refs, encoding="utf-8")

audit = {
    "old_reference_count": len(items),
    "new_reference_count": len(new_items),
    "removed_uncited_old_ids": sorted(set(range(1, len(items) + 1)) - set(appearance)),
    "first_appearance_old_ids": appearance,
    "old_to_new": {str(old): new for old, new in mapping.items()},
}
AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
