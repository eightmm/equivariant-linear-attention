from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_FILES = tuple(
    sorted((*ROOT.glob("*.md"), *(ROOT / "docs").glob("*.md")))
)
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
FENCE_PATTERN = re.compile(r"^ {0,3}(?P<marker>`{3,}|~{3,})(?P<rest>.*)$")
INLINE_CODE_PATTERN = re.compile(r"`+[^`]*`+")
RESTRICTED_MATH_MACRO_PATTERN = re.compile(r"\\(?:operatorname|rm)\b")


def _outside_fenced_code(text: str) -> tuple[tuple[str, ...], str | None]:
    outside: list[str] = []
    active: tuple[str, int, int] | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = FENCE_PATTERN.match(line)
        if active is not None:
            character, width, opening_line = active
            if match is not None:
                marker = match.group("marker")
                if (
                    marker[0] == character
                    and len(marker) >= width
                    and not match.group("rest").strip()
                ):
                    active = None
            continue
        if match is not None:
            marker = match.group("marker")
            active = (marker[0], len(marker), line_number)
            continue
        outside.append(INLINE_CODE_PATTERN.sub("", line))
    if active is None:
        return tuple(outside), None
    character, width, opening_line = active
    return tuple(outside), f"line {opening_line}: {character * width}"


def test_markdown_uses_github_math_delimiters() -> None:
    offenders: list[str] = []
    for path in MARKDOWN_FILES:
        lines, _ = _outside_fenced_code(path.read_text(encoding="utf-8"))
        prose = "\n".join(lines)
        legacy = any(
            delimiter in prose for delimiter in (r"\[", r"\]", r"\(", r"\)")
        )
        unbalanced_display_math = sum(line.count("$$") for line in lines) % 2
        if legacy or unbalanced_display_math:
            reasons = []
            if legacy:
                reasons.append("legacy delimiter")
            if unbalanced_display_math:
                reasons.append("unbalanced $$")
            offenders.append(
                f"{path.relative_to(ROOT)} ({', '.join(reasons)})"
            )
    assert not offenders, (
        "use GitHub-compatible $...$ or $$...$$ math delimiters in: "
        + ", ".join(offenders)
    )


def test_markdown_avoids_restricted_or_legacy_math_macros() -> None:
    offenders: list[str] = []
    for path in MARKDOWN_FILES:
        lines, _ = _outside_fenced_code(path.read_text(encoding="utf-8"))
        prose = "\n".join(lines)
        found = tuple(
            sorted(set(RESTRICTED_MATH_MACRO_PATTERN.findall(prose)))
        )
        if found:
            offenders.append(
                f"{path.relative_to(ROOT)} ({', '.join(found)})"
            )
    assert not offenders, (
        "use renderer-safe \\text{...} math labels in: "
        + ", ".join(offenders)
    )


def test_markdown_code_fences_are_balanced() -> None:
    offenders: list[str] = []
    for path in MARKDOWN_FILES:
        _, unclosed = _outside_fenced_code(path.read_text(encoding="utf-8"))
        if unclosed is not None:
            offenders.append(f"{path.relative_to(ROOT)} ({unclosed})")
    assert not offenders, "unbalanced Markdown code fences in: " + ", ".join(
        offenders
    )


def test_relative_markdown_links_resolve() -> None:
    missing: list[str] = []
    for path in MARKDOWN_FILES:
        lines, _ = _outside_fenced_code(path.read_text(encoding="utf-8"))
        for raw_target in LINK_PATTERN.findall("\n".join(lines)):
            target = raw_target.strip().strip("<>")
            if target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            relative_target = target.split("#", maxsplit=1)[0]
            if relative_target and not (path.parent / relative_target).exists():
                missing.append(
                    f"{path.relative_to(ROOT)} -> {relative_target}"
                )
    assert not missing, "missing Markdown link targets:\n" + "\n".join(missing)
