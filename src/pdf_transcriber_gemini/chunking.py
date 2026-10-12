"""Split long PDFs and join Gemini responses without losing original page references.

Each chunk is a real, temporary PDF, but all final figure extraction is performed
against the complete original PDF. No native PDF text is used for transcription.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from .assembler import FIGURE_MARKER, validate_figures

CHUNK_PAGES = 10
AUTO_CHUNK_THRESHOLD = 30
START_FLAG = "<!-- CHUNK_START_MID_PARAGRAPH -->"
END_FLAG = "<!-- CHUNK_END_MID_PARAGRAPH -->"
PAGE_COMMENT = re.compile(r"<!--\s*pdf page\s+(\d+)(\s*\|\s*article page\s+[^<>]*?)?\s*-->", re.I)
FOOTNOTE_DEF = re.compile(r"(?m)^\[\^([^\]\n]+)\]:")
FOOTNOTE_USE = re.compile(r"\[\^([^\]\n]+)\]")


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int  # original PDF page, 1-based
    end: int  # inclusive

    @property
    def pages(self) -> int:
        return self.end - self.start + 1

    @property
    def name(self) -> str:
        return f"part-{self.index:04d}"


def chunks_for(total: int, size: int = CHUNK_PAGES) -> list[Chunk]:
    if total < 1 or size < 1:
        raise ValueError("PDF page count and chunk size must be positive")
    return [Chunk(i, start, min(start + size - 1, total))
            for i, start in enumerate(range(1, total + 1, size), 1)]


def write_chunk(pdf: Path, chunk: Chunk, destination: Path) -> Path:
    """Make an ASCII-named chunk PDF, avoiding source-filename header issues."""
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{chunk.name}.pdf"
    if path.exists():
        with pymupdf.open(path) as existing:
            if existing.page_count != chunk.pages:
                raise ValueError(f"Cached chunk has wrong length: {path}")
        return path
    staged = destination / f"{chunk.name}.tmp.pdf"
    try:
        with pymupdf.open(pdf) as source:
            with pymupdf.open() as output:
                output.insert_pdf(source, from_page=chunk.start - 1, to_page=chunk.end - 1)
                output.save(staged)
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)
    return path


def chunk_prompt(prompt: str, chunk: Chunk, total: int) -> str:
    """Add only the contextual instructions needed for a PDF excerpt."""
    return prompt.rstrip() + f"""

ADDITIONAL INSTRUCTIONS FOR THIS CHUNK ONLY:
This attached PDF is an excerpt containing ORIGINAL PDF PAGES {chunk.start}–{chunk.end}
(of {total} pages in the source). Transcribe ONLY the material actually visible in
this attached excerpt. Do not invent title, abstract, references or other content
outside it; do not repeat a title or heading not actually present. Preserve any
continuation from earlier pages and any incomplete material at the end.

VERY IMPORTANT: In THIS attached chunk, pdf_page in JSON figures and "pdf page"
in Markdown page comments must refer to the ATTACHED CHUNK, starting at 1.
A separate script will add {chunk.start - 1} to convert them into original page numbers.
Article page numbers, if visible, must remain their original printed numbers.

If the first textual paragraph on this chunk is a continuation from an earlier
source page, start the Markdown with the exact marker {START_FLAG}.
If the final textual paragraph is incomplete and continues on the next chunk,
end the Markdown with the exact marker {END_FLAG}.
These markers are assembly metadata, not part of the article. In all other respects,
follow the original instructions and return exactly the same JSON schema.
"""


def _globalize_pages(text: str, chunk: Chunk) -> str:
    def translate(m: re.Match[str]) -> str:
        number = int(m.group(1))
        if not 1 <= number <= chunk.pages:
            raise ValueError(
                f"Chunk {chunk.index}: page comment {number} is not within "
                f"the attached chunk's {chunk.pages} page(s). Check response.json."
            )
        suffix = m.group(2) or ""
        return f"<!-- pdf page {number + chunk.start - 1}{suffix} -->"
    return PAGE_COMMENT.sub(translate, text)


def _renumber_duplicate_footnotes(parts: list[str]) -> list[str]:
    """Prevent separate chunks defining the same Markdown footnote reference."""
    used: set[str] = set()
    numeric = [int(f) for md in parts for f in FOOTNOTE_DEF.findall(md) if f.isdecimal()]
    next_number = max(numeric, default=0) + 1
    result: list[str] = []
    for md in parts:
        rename: dict[str, str] = {}
        local: set[str] = set()
        for marker in FOOTNOTE_DEF.findall(md):
            if marker in local:
                raise ValueError(f"Duplicate footnote definition within one chunk: {marker}")
            local.add(marker)
            if marker in used:
                while str(next_number) in used or str(next_number) in local:
                    next_number += 1
                rename[marker] = str(next_number)
                used.add(str(next_number))
                next_number += 1
            else:
                used.add(marker)
        # Only rename references in this chunk with a locally conflicting definition.
        result.append(FOOTNOTE_USE.sub(
            lambda m: f"[^{rename.get(m.group(1), m.group(1))}]", md
        ))
    return result


def _join_inline(previous: str, following: str) -> str:
    """Preserve paragraph continuation and move page comments inline."""
    left = previous.rstrip()
    right = following.lstrip()
    # Move leading page comments inside the paragraph rather than using a blank line.
    leading: list[str] = []
    while True:
        match = PAGE_COMMENT.match(right)
        if match is None:
            break
        leading.append(match.group(0))
        right = right[match.end():].lstrip()
    if leading:
        return left + " " + " ".join(leading) + " " + right
    return left + " " + right


def combine_chunks(responses: list[tuple[Chunk, dict]]) -> dict:
    """Create the same schema used by single-PDF assembly, in original coordinates."""
    if not responses:
        raise ValueError("No chunk responses to combine")
    text_parts: list[str] = []
    figures: list[dict] = []
    problems: list[str] = []
    boundaries: list[tuple[bool, bool]] = []
    global_id = 0
    for chunk, data in responses:
        local_figures = validate_figures(data, chunk.pages)
        content = data["markdown"].strip()
        begins = START_FLAG in content
        ends = END_FLAG in content
        content = content.replace(START_FLAG, "").replace(END_FLAG, "").strip()
        content = _globalize_pages(content, chunk)
        mapping = {}
        for fig in local_figures:
            global_id += 1
            mapping[fig["id"]] = f"fig_{global_id}"
            global_fig = copy.deepcopy(fig)
            global_fig["id"] = mapping[fig["id"]]
            global_fig["pdf_page"] += chunk.start - 1
            figures.append(global_fig)
        content = FIGURE_MARKER.sub(
            lambda m: f"<!-- FIGURE:{mapping[m.group(1)]} -->", content
        )
        text_parts.append(content)
        boundaries.append((begins, ends))
        problems.extend(f"Chunk {chunk.index} (pages {chunk.start}–{chunk.end}): {p}"
                        for p in data["problems"])

    text_parts = _renumber_duplicate_footnotes(text_parts)
    merged = text_parts[0]
    for i, next_part in enumerate(text_parts[1:], start=1):
        prev_ends = boundaries[i - 1][1]
        next_begins = boundaries[i][0]
        if prev_ends and next_begins:
            merged = _join_inline(merged, next_part)
        else:
            if prev_ends != next_begins:
                problems.append(
                    f"Review paragraph continuity at chunk boundary {i}/{i+1}: "
                    "only one side indicated a continuing paragraph."
                )
            merged = merged.rstrip() + "\n\n" + next_part.lstrip()
    result = {"markdown": merged.rstrip() + "\n", "figures": figures, "problems": problems}
    # Validate globally after IDs and page references have been adjusted.
    validate_figures(result, max(chunk.end for chunk, _ in responses))
    return result
