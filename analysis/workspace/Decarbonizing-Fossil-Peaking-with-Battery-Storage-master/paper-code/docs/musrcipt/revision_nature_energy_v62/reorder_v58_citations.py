#!/usr/bin/env python3
"""Renumber v58 references by first appearance in main text, then supplement."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANUSCRIPTS = [
    ROOT / "musript_new_v58.tex",
    ROOT / "musript_new_v58_supplementary.tex",
]
REFERENCES = ROOT / "references_v58.tex"
MANIFEST = ROOT / "citation_order_v58.json"

CITATION = re.compile(r"(?<!})\[(\d+(?:\s*(?:--|-|,)\s*\d+)*)\]")


def expand_group(group: str) -> list[int]:
    numbers: list[int] = []
    for part in group.split(","):
        part = part.strip()
        match = re.fullmatch(r"(\d+)\s*(?:--|-)\s*(\d+)", part)
        if match:
            start, end = map(int, match.groups())
            numbers.extend(range(start, end + 1))
        else:
            numbers.append(int(part))
    return numbers


def compact_group(numbers: list[int]) -> str:
    ordered = list(dict.fromkeys(numbers))
    parts: list[str] = []
    index = 0
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and ordered[end + 1] == ordered[end] + 1:
            end += 1
        run = ordered[index : end + 1]
        if len(run) >= 3:
            parts.append(f"{run[0]}--{run[-1]}")
        else:
            parts.extend(str(value) for value in run)
        index = end + 1
    return ",".join(parts)


def reference_items(text: str) -> tuple[str, list[str], str]:
    starts = [match.start() for match in re.finditer(r"(?m)^\\item ", text)]
    if not starts:
        raise RuntimeError("No reference items found")
    end = text.index(r"\end{enumerate}", starts[-1])
    items = [
        text[start : starts[index + 1] if index + 1 < len(starts) else end]
        for index, start in enumerate(starts)
    ]
    return text[: starts[0]], items, text[end:]


def main() -> None:
    source_texts = {path: path.read_text(encoding="utf-8") for path in MANUSCRIPTS}
    first_appearance: list[int] = []
    for path in MANUSCRIPTS:
        for match in CITATION.finditer(source_texts[path]):
            for old_number in expand_group(match.group(1)):
                if old_number not in first_appearance:
                    first_appearance.append(old_number)

    ref_text = REFERENCES.read_text(encoding="utf-8")
    header, items, tail = reference_items(ref_text)
    invalid = [number for number in first_appearance if not 1 <= number <= len(items)]
    if invalid:
        raise RuntimeError(f"Citation numbers outside reference list: {invalid}")

    old_to_new = {old: new for new, old in enumerate(first_appearance, start=1)}

    def replace(match: re.Match[str]) -> str:
        remapped = [old_to_new[number] for number in expand_group(match.group(1))]
        return f"[{compact_group(remapped)}]"

    for path, text in source_texts.items():
        path.write_text(CITATION.sub(replace, text), encoding="utf-8")

    reordered = header + "".join(items[number - 1] for number in first_appearance) + tail
    REFERENCES.write_text(reordered, encoding="utf-8")
    MANIFEST.write_text(
        json.dumps(
            {
                "reference_count": len(first_appearance),
                "old_to_new": {str(old): new for old, new in old_to_new.items()},
                "first_appearance_old_numbers": first_appearance,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
