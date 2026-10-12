"""Local, extensible citation-to-bibliography matching.

This module has no dependencies on PDF transcription, the model provider, or
Markdown editing. It can be moved to a standalone package when useful elsewhere.
Only unique matches with checkable bibliographic evidence may create links.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Protocol

_YEAR = re.compile(r"(?<![A-Za-z0-9])(?:18|19|20)\d{2}[a-z]?(?![A-Za-z0-9])", re.I)
_COMMENT = re.compile(r"<!--[\s\S]*?-->|\{##[\s\S]*?##\}")
_CORPORATE = re.compile(
    r"\b(?:team|group|committee|consortium|collaboration|institute|society|"
    r"organization|organisation|foundation|association)\b", re.I,
)
_INITIAL = r"[A-Za-z]\.(?:-[A-Za-z]\.)?"
_INITIALS = re.compile(rf"^(?:{_INITIAL}\s*){{1,8}}(?=\S)")
_OTHERS = re.compile(r"\b(?:et\s+al\.?|and\s+colleagues?)\b", re.I)
_NUMERIC = re.compile(r"\[(\d{1,5})\]")


def fold(text: str) -> str:
    """Accent-, punctuation- and case-insensitive name comparison."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold().replace("’", "'")
    return " ".join(re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).split())


def visible(text: str) -> str:
    """Comparison view only; never rewrite original citation or review markup."""
    text = _COMMENT.sub("", text)
    text = text.replace("{==", "").replace("==}", "")
    return " ".join(text.split())


def canonical_author(name: str) -> str | None:
    """Normalize 'A. Author' to 'Author, A.', retaining family particles.

    This canonical form is useful for comparing Gemini (initials-first) and
    Codex (usually surname-first); the raw bibliographic metadata is retained.
    """
    name = " ".join(name.split())
    if not name or fold(name) in {"et al", "and others", "others"}:
        return None
    if _CORPORATE.search(name) and "," not in name:
        return name
    if "," in name:
        surname, given = (part.strip() for part in name.split(",", 1))
        return f"{surname}, {given}" if surname and given else name
    head = _INITIALS.match(name)
    if head:
        surname = name[head.end():].strip()
        initials = name[:head.end()].strip()
        return f"{surname}, {initials}" if surname else name
    return name


def author_family(name: str) -> str:
    canonical = canonical_author(name) or ""
    if _CORPORATE.search(canonical) and "," not in canonical:
        return fold(canonical)
    if "," in canonical:
        return fold(canonical.split(",", 1)[0])
    # An unstructured personal name with multiple non-initial tokens is not
    # safely separable. Such cases are not accepted as a unique match.
    if len(canonical.split()) > 1:
        return ""
    return fold(canonical)


def cited_names(text: str) -> tuple[list[str], bool]:
    """Extract explicit surnames from the portion before a single cited year."""
    text = visible(text).strip(" \t,;:.()[]")
    abbreviated = bool(_OTHERS.search(text))
    text = _OTHERS.sub("", text).strip(" \t,;:.()[]")
    if not text:
        return [], abbreviated
    if _CORPORATE.search(text):
        return [fold(text)], abbreviated
    # The model should identify a citation span rather than its surrounding
    # attribution text; do not guess at authors from an ordinary sentence.
    if re.search(r"\b(?:see|cf|e\.?g|for|review|based|from|in|by)\b", text, re.I):
        return [], abbreviated
    text = re.sub(r"\s+(?:and|&)\s+", ", ", text, flags=re.I)
    results: list[str] = []
    for part in text.split(","):
        part = _INITIALS.sub("", part.strip(" \t,;:.()[]")).strip()
        if not part:
            continue
        if re.search(r"\d", part):
            return [], abbreviated
        results.append(fold(part))
    return results, abbreviated


@dataclass(frozen=True)
class Entry:
    reference_id: str
    year: str | None
    families: tuple[str, ...]
    numeric_label: str | None
    authors_complete: bool = True


@dataclass(frozen=True)
class Match:
    reference_id: str | None
    code: str
    strategy: str
    candidates: tuple[str, ...] = ()


class Matcher(Protocol):
    name: str

    def match(self, text: str, entries: tuple[Entry, ...]) -> Match | None:
        """Return None if this strategy does not apply to the citation."""
        ...


class NumericLabelMatcher:
    name = "numeric_label"

    def match(self, text: str, entries: tuple[Entry, ...]) -> Match | None:
        numeric = _NUMERIC.fullmatch(visible(text).strip())
        if not numeric:
            return None
        target = str(int(numeric.group(1)))
        candidates = tuple(e.reference_id for e in entries if e.numeric_label == target)
        if len(candidates) == 1:
            return Match(candidates[0], "matched", self.name, candidates)
        return Match(None, "ambiguous_numeric_label" if candidates else "unknown_numeric_label",
                     self.name, candidates)


class AuthorYearMatcher:
    name = "author_year"

    def match(self, text: str, entries: tuple[Entry, ...]) -> Match | None:
        cleaned = visible(text)
        years = list(_YEAR.finditer(cleaned))
        if not years:
            return None
        if len(years) != 1:
            return Match(None, "multiple_years", self.name)
        year = years[0].group().casefold()
        authors, abbreviated = cited_names(cleaned[:years[0].start()])
        if not authors:
            return Match(None, "missing_cited_author", self.name)
        same_year = [entry for entry in entries if entry.year == year]
        first_author = [e for e in same_year if e.families and e.families[0] == authors[0]]
        if not first_author:
            # Distinguish a missing entry from a badly structured bibliography.
            if any(e.year == year and not e.families for e in entries):
                return Match(None, "incomplete_bibliography_metadata", self.name)
            return Match(None, "no_author_year_match", self.name)
        incomplete = [e for e in same_year if not e.authors_complete
                      and (not e.families or not e.families[0]
                           or e.families[0] == authors[0])]
        if incomplete:
            return Match(None, "incomplete_bibliography_metadata", self.name,
                         tuple(e.reference_id for e in incomplete))
        candidates = []
        for ref in first_author:
            if len(authors) > len(ref.families) or ref.families[:len(authors)] != tuple(authors):
                continue
            # A written-out multi-author citation normally names all authors;
            # require a complete list unless 'et al.' or 'and colleagues' appears.
            if not abbreviated and len(authors) != len(ref.families):
                continue
            if abbreviated and len(authors) >= len(ref.families):
                continue
            candidates.append(ref.reference_id)
        ids = tuple(candidates)
        if len(ids) == 1:
            return Match(ids[0], "matched", self.name, ids)
        if len(ids) > 1:
            return Match(None, "ambiguous_author_year", self.name, ids)
        return Match(None, "coauthor_mismatch", self.name,
                     tuple(ref.reference_id for ref in first_author))


class UnhandledMatcher:
    name = "unhandled"

    def match(self, text: str, entries: tuple[Entry, ...]) -> Match:
        return Match(None, "unsupported_citation_form", self.name)


class CitationResolver:
    """An ordered matcher pipeline; add a strategy without changing callers.

    New matchers (for context-dependent year-only mentions, ibid., document
    specific conventions, etc.) can be inserted before the fallback strategy.
    """

    def __init__(self, references: list[dict[str, Any]],
                 strategies: tuple[Matcher, ...] | None = None):
        entries: list[Entry] = []
        for ref in references:
            names = ref.get("authors") or []
            if not isinstance(names, list):
                names = []
            # Retain unknown authors in their original positions.
            families = tuple(author_family(person) if isinstance(person, str) else ""
                             for person in names)
            complete = bool(families) and all(families) and ref.get("_authors_complete", True)
            label = ref.get("label")
            number = None
            if isinstance(label, str) and re.fullmatch(r"(?:\d{1,5}|\[\d{1,5}\])", label.strip()):
                number = str(int(label.strip(" []")))
            year = ref.get("year")
            if not isinstance(year, str) or not re.fullmatch(r"(?:18|19|20)\d{2}[a-z]?", year.strip(), re.I):
                year = None
            entries.append(Entry(ref["id"], year.strip().casefold() if year else None, families, number, complete))
        self.entries = tuple(entries)
        self.strategies = strategies or (
            NumericLabelMatcher(), AuthorYearMatcher(), UnhandledMatcher(),
        )

    def resolve(self, text: str) -> Match:
        for strategy in self.strategies:
            decision = strategy.match(text, self.entries)
            if decision is not None:
                return decision
        return Match(None, "unsupported_citation_form", "none")

    def compact_candidates(self, ids: tuple[str, ...]) -> list[dict[str, str]]:
        lookup = {entry.reference_id: entry for entry in self.entries}
        return [
            {"ref": ident,
             "year": entry.year or "",
             "authors": ", ".join(entry.families[:3])}
            for ident in ids[:8] if (entry := lookup.get(ident)) is not None
        ]
