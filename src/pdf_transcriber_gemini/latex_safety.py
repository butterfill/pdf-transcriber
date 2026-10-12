"""Conservative, auditable recovery of LaTeX escaping in Gemini JSON responses.

Only known LaTeX commands *inside dollar-delimited mathematics* in the top-level
'markdown' JSON string are repaired. No other JSON fields are ever rewritten.
Validation reports suspicious patterns without altering the Markdown.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

# Deliberately curated, not a generic "backslash plus word" substitution.
# Commands with JSON-valid initial escapes (\\t, \\b, \\f, \\n, \\r) are crucial:
# without pre-decoding repair these can silently become control characters.
MATH_COMMANDS = frozenset("""
alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa
lambda mu nu xi pi varpi rho varrho sigma varsigma tau upsilon phi varphi
chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega
text textrm textbf textit mathrm mathbf mathit mathcal mathbb mathsf
operatorname frac dfrac tfrac sqrt root overline underline underbrace overbrace
hat widehat bar tilde widetilde vec dot ddot acute grave breve check
sum prod int iint iiint oint lim log ln exp sin cos tan min max sup inf
substack binom dbinom tbinom left right middle big Big bigg Bigg
le leq ge geq neq approx sim simeq equiv propto times cdot ast pm mp
partial nabla in notin subset subseteq supset supseteq cup cap
infty forall exists neg land lor to gets rightarrow leftarrow
leftrightarrow Rightarrow Leftarrow Leftrightarrow mapsto
uparrow downarrow Uparrow Downarrow begin end hline
quad qquad qspace space hspace vspace phantom color textcolor
ldots cdots vdots ddots dots bmod pmod mod
""".split()) | frozenset({"%", "_"})

# A conservative subset of macros meaningful after double-backslash; warnings
# do not change output because \\ also has legitimate LaTeX uses (line breaks).
DOUBLE_SLASH_COMMANDS = MATH_COMMANDS - {"_", "%", "begin", "end", "hline"}


@dataclass(frozen=True)
class Repair:
    command: str
    offset: int  # source-file offset, before insertion


@dataclass(frozen=True)
class Warning:
    line: int  # in decoded Markdown
    column: int
    message: str


def _end_string(source: str, quote: int) -> int | None:
    """Find an unescaped JSON quote, tolerating *invalid* backslash escapes."""
    cursor = quote + 1
    while cursor < len(source):
        if source[cursor] == "\\":
            cursor += 2
        elif source[cursor] == '"':
            return cursor
        else:
            cursor += 1
    return None


def _markdown_bounds(source: str) -> tuple[int, int] | None:
    """Locate only the top-level markdown *value*, even if that string is invalid JSON."""
    depth = 0
    index = 0
    while index < len(source):
        char = source[index]
        if char == '"':
            end = _end_string(source, index)
            if end is None:
                return None
            if depth == 1 and source[index:end + 1] == '"markdown"':
                after = end + 1
                while after < len(source) and source[after].isspace():
                    after += 1
                if after < len(source) and source[after] == ":":
                    after += 1
                    while after < len(source) and source[after].isspace():
                        after += 1
                    if after < len(source) and source[after] == '"':
                        value_end = _end_string(source, after)
                        if value_end is not None:
                            return after + 1, value_end
            index = end + 1
            continue
        if char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
        index += 1
    return None


def _math_regions(text: str, *, raw_json: bool = False) -> list[tuple[int, int]]:
    """Recognize $...$ and $$...$$, skipping escaped delimiters and code fences."""
    regions = []
    opening: tuple[int, int] | None = None
    in_fence = False
    in_code = False
    index = 0
    line_start = True
    while index < len(text):
        c = text[index]
        if raw_json and text.startswith(r"\n", index):
            # A JSON newline escape is a Markdown line boundary for code fences.
            line_start = True
            in_code = False
            index += 2
            continue
        if line_start and text.startswith(chr(96) * 3, index):
            in_fence = not in_fence
            line_start = False
            index += 3
            continue
        if c == "\n":
            line_start = True
            in_code = False
            index += 1
            continue
        if not c.isspace() or not line_start:
            line_start = False
        if in_fence:
            index += 1
            continue
        if c == chr(96):
            in_code = not in_code
            index += 1
            continue
        if in_code or c != "$":
            index += 1
            continue
        backslashes = 0
        n = index - 1
        while n >= 0 and text[n] == "\\":
            backslashes += 1
            n -= 1
        # Raw JSON represents one literal backslash with two characters.
        # Thus raw \\$ denotes a Markdown-escaped dollar, not a math delimiter.
        escaped = (backslashes % 2 or (raw_json and (backslashes // 2) % 2))
        if escaped:
            index += 1
            continue
        width = 2 if text.startswith("$$", index) else 1
        if opening is None:
            opening = (index + width, width)
        elif width == opening[1]:
            regions.append((opening[0], index))
            opening = None
        index += width
    return regions


def recover_json_latex(raw: str) -> tuple[str, list[Repair]]:
    """Fix recognized LaTeX escapes in raw markdown JSON; never edit the input file.

    Crucially, this runs even when raw JSON happens to parse, because \\text,
    \\beta, \\frac, \\nu and \\rho can otherwise decode to control characters.
    """
    bounds = _markdown_bounds(raw)
    if bounds is None:
        return raw, []
    start, end = bounds
    body = raw[start:end]
    math = _math_regions(body, raw_json=True)
    if not math:
        return raw, []
    repairs: list[Repair] = []
    fragments: list[str] = []
    cursor = 0
    region_index = 0
    i = 0
    while i < len(body):
        while region_index < len(math) and i >= math[region_index][1]:
            region_index += 1
        in_math = (region_index < len(math) and
                   math[region_index][0] <= i < math[region_index][1])
        if in_math and body[i] == "\\" and i + 1 < len(body) and body[i + 1] != "\\":
            match = re.match(r"\\([A-Za-z]+|[%_])", body[i:])
            if match and match.group(1) in MATH_COMMANDS:
                fragments.extend((body[cursor:i], "\\"))
                repairs.append(Repair("\\" + match.group(1), start + i))
                cursor = i  # retain original slash too: now two slashes in JSON
                i += len(match.group(0))
                continue
        if body[i] == "\\" and i + 1 < len(body):
            i += 2  # already escaped backslash or a JSON escape, leave alone
        else:
            i += 1
    if not repairs:
        return raw, []
    fragments.append(body[cursor:])
    return raw[:start] + "".join(fragments) + raw[end:], repairs


def _json_whitespace_at(body: str, position: int) -> bool:
    """Whether JSON source at position starts a whitespace Markdown character."""
    if position >= len(body):
        return False
    char = body[position]
    if char.isspace():
        return True
    if char != "\\" or position + 1 >= len(body):
        return False
    escaped = body[position + 1]
    if escaped in "nrtf":
        return True
    if escaped == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", body[position + 2:position + 6]):
        return chr(int(body[position + 2:position + 6], 16)).isspace()
    return False


def recover_json_markdown_asterisks(raw: str) -> tuple[str, list[int]]:
    """Recover malformed JSON \\* before whitespace in top-level Markdown only.

    Replace the invalid JSON escape \\* with \\\\* so decoding preserves
    the Markdown-escaped literal asterisk. The raw response is not modified.
    """
    bounds = _markdown_bounds(raw)
    if bounds is None:
        return raw, []
    start, end = bounds
    body = raw[start:end]
    offsets: list[int] = []
    fragments: list[str] = []
    cursor = index = 0
    while index + 1 < len(body):
        if body[index] != "\\":
            index += 1
            continue
        if body[index + 1] == "*" and _json_whitespace_at(body, index + 2):
            fragments.extend((body[cursor:index], "\\"))
            offsets.append(start + index)
            cursor = index
        index += 2  # Skip valid JSON escapes, including already-doubled backslashes.
    if not offsets:
        return raw, []
    fragments.append(body[cursor:])
    return raw[:start] + "".join(fragments) + raw[end:], offsets


def _line_column(text: str, offset: int) -> tuple[int, int]:
    line = text.count("\n", 0, offset) + 1
    prev = text.rfind("\n", 0, offset)
    return line, offset - prev


def latex_warnings(markdown: str) -> list[Warning]:
    """Non-mutating checks: control characters and likely over-escaped math."""
    issues: list[Warning] = []
    regions = _math_regions(markdown)
    for index, char in enumerate(markdown):
        if char in "\b\f\v\r" or (ord(char) < 32 and char not in "\n\t"):
            line, col = _line_column(markdown, index)
            issues.append(Warning(line, col, f"unexpected control character U+{ord(char):04X}"))
    for begin, end in regions:
        snippet = markdown[begin:end]
        # Unbalanced argument braces often mean a clipped or mistyped macro.
        # Braces escaped for display (\\{ and \\}) are not argument delimiters.
        brace_depth = 0
        for index, char in enumerate(snippet):
            if char not in "{}":
                continue
            slash_count = 0
            j = index - 1
            while j >= 0 and snippet[j] == "\\":
                slash_count += 1
                j -= 1
            if slash_count % 2:
                continue
            brace_depth += 1 if char == "{" else -1
            if brace_depth < 0:
                line, col = _line_column(markdown, begin + index)
                issues.append(Warning(line, col, "unmatched closing brace inside math"))
                brace_depth = 0
        if brace_depth:
            line, col = _line_column(markdown, begin)
            issues.append(Warning(line, col, "unclosed argument brace inside math"))
        for match in re.finditer(r"\\\\([A-Za-z]+)", snippet):
            if match.group(1) in DOUBLE_SLASH_COMMANDS:
                line, col = _line_column(markdown, begin + match.start())
                issues.append(Warning(line, col, f"double backslash before \\{match.group(1)} in math; check manually"))
        for match in re.finditer(r"\t", snippet):
            line, col = _line_column(markdown, begin + match.start())
            issues.append(Warning(line, col, "tab inside math; possibly a damaged \\t command"))
        for match in re.finditer(r"\\_", snippet):
            line, col = _line_column(markdown, begin + match.start())
            issues.append(Warning(line, col, "escaped underscore inside math; check manually"))
    return issues


def repair_summary(repairs: list[Repair]) -> str:
    """Compact, deterministic audit summary; caller reports it to CLI/batch UI."""
    counts = Counter(x.command for x in repairs)
    kinds = ", ".join(f"{name} x{count}" for name, count in sorted(counts.items()))
    return f"Recovered {len(repairs)} LaTeX JSON escape(s) in markdown ({kinds}); raw JSON unchanged."
