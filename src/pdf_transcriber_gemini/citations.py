from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any

from .citation_matcher import CitationResolver, author_family, canonical_author

ANNOTATION_PROMPT = "REVIEW COMMENT CITATIONS: This Markdown contains RMX review-comment bodies that appear to include citation-like text. RMX highlights use {==highlighted original source text==}{##comment-id##}; the highlighted original words remain eligible for citation detection. Separately, review COMMENT BODIES use {##comment-id: the user's annotation##}, potentially spanning multiple source lines. Those bodies are my personal annotations, not part of the article and must NOT be used to identify the article's references or citation occurrences. Ignore all citations that occur only within review comment bodies. Never propose edits inside a review comment body or across any annotation boundary. The presence of a bibliographic reference inside one of my annotations is not evidence that the author cited it."

COMMENT_BODY_RE = re.compile(r"\{##[^:\n{}]+:\s*.*?##\}", re.DOTALL)
CITATION_LIKE_RE = re.compile(
    r"(?:\b[A-Z][\w'’-]+(?:\s+(?:et\s+al\.?|[A-Z][\w'’-]+))*\s*[,()]?\s*"
    r"(?:18|19|20)\d\d[a-z]?\b|\\cite\w*\s*(?:\[|\{)|\[\d{1,3}(?:\s*[,;–-]\s*\d{1,3})*\])",
    re.IGNORECASE,
)
RESPONSE_FENCE_RE = re.compile(r"\A```(?:json)?\s*\n(.*?)\n```\s*\Z", re.DOTALL | re.IGNORECASE)
REFERENCE_ID_RE = re.compile(r"\Asrc-ref-\d{3,}\Z")


class CitationError(ValueError):
    pass


def read_markdown(path: Path) -> tuple[str, str, str]:
    if not path.is_file() or path.suffix.lower() != ".md":
        raise CitationError(f"Source must be an existing Markdown file: {path}")
    raw = path.read_bytes()
    try:
        source = raw.decode("utf-8")
    except UnicodeError as exc:
        raise CitationError("Markdown must be UTF-8") from exc
    checksum = hashlib.sha256(raw).hexdigest()
    return source, checksum, checksum[:16]


def split_source_lines(source: str) -> list[tuple[int, str]]:
    """Return (character start, content without newline); offsets are into raw text."""
    out: list[tuple[int, str]] = []
    offset = 0
    for line in source.splitlines(keepends=True):
        if line.endswith("\r\n"):
            body = line[:-2]
        elif line.endswith("\n") or line.endswith("\r"):
            body = line[:-1]
        else:
            body = line
        out.append((offset, body))
        offset += len(line)
    return out


def output_paths(markdown: Path) -> dict[str, Path]:
    prefix = markdown.with_suffix("")
    return {
        "request": Path(str(prefix) + ".citation-request.txt"),
        "proposal": Path(str(prefix) + ".citation-proposals.json"),
        "linked": markdown,  # The validated Markdown replaces the source on success.
        "legacy_linked": Path(str(prefix) + ".cited.md"),
        "metadata": Path(str(prefix) + ".citations.json"),
        "failures": Path(str(prefix) + ".citation-match-failures.jsonl"),
    }


def has_generated_citation_markup(source: str) -> bool:
    """Reject re-enriching Markdown that already contains our citation IDs."""
    return bool(
        re.search("<a\\s+id=[\"']src-ref-\\d{3,}[\"']", source)
        or re.search(r"(?m)^\[cite-\d{4,}\]:\s*#src-ref-\d{3,}", source)
    )


def render_request(markdown: Path) -> str:
    """Print-ready AI Studio request; no working files or API calls."""
    source, checksum, source_id = read_markdown(markdown)
    prompt = files("pdf_transcriber_gemini").joinpath("citations_prompt.txt").read_text(encoding="utf-8").strip()
    if any(CITATION_LIKE_RE.search(m.group(0)) for m in COMMENT_BODY_RE.finditer(source)):
        prompt += "\n\n" + ANNOTATION_PROMPT
    lines = split_source_lines(source)
    numbered = "\n".join(f"L{n:06d}|{body}" for n, (_, body) in enumerate(lines, 1))
    return (
        prompt + "\n\n"
        + f"SOURCE_ID: {source_id}\n"
        + f"SOURCE_SHA256: {checksum}\n"
        + "===== BEGIN NUMBERED SOURCE =====\n"
        + numbered + "\n"
        + "===== END NUMBERED SOURCE =====\n"
    )



def normalize_compact_response(obj: dict[str, Any]) -> dict[str, Any]:
    """Expand compact model output to the established validated edit representation.

    New responses contain only LINE|TEXT; old model REF values are ignored.
    All bibliography identities and persistent citation IDs are resolved locally.
    """
    original_references = obj.get("references")
    original_mentions = obj.get("mentions")
    if not isinstance(original_references, list) or not isinstance(original_mentions, list):
        raise CitationError("Compact response must contain references and mentions arrays")
    references: list[dict[str, Any]] = []
    fields = ("label", "authors", "editors", "year", "title", "container_title",
              "volume", "issue", "pages", "publisher", "doi", "isbn", "url")
    for i, ref in enumerate(original_references, 1):
        if not isinstance(ref, dict):
            raise CitationError(f"Reference {i} must be an object")
        line, anchor = ref.get("l"), ref.get("a")
        if type(line) is not int or line < 1 or not isinstance(anchor, str) or not anchor:
            raise CitationError(f"Reference {i} needs positive l and nonempty a")
        normalized = {
            "id": f"src-ref-{i:03d}", "start_line": line, "anchor_text": anchor
        }
        for field in fields:
            value = ref.get(field)
            if value is None or value == "" or value == []:
                continue
            if field in ("authors", "editors"):
                if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
                    raise CitationError(f"Reference {i}: {field} must be an array of strings")
            elif not isinstance(value, str):
                raise CitationError(f"Reference {i}: {field} must be a string")
            if field in ("authors", "editors"):
                # Keep bibliographic author names consistent across providers.
                if field == "authors":
                    normalized["_authors_complete"] = bool(value) and all(author_family(name) for name in value)
                value = [v for name in value if (v := canonical_author(name)) is not None]
            normalized[field] = value
        references.append(normalized)

    mentions: list[dict[str, Any]] = []
    rejected_mentions: list[dict[str, Any]] = []
    for i, raw in enumerate(original_mentions, 1):
        # A single badly transcribed citation must not discard an entire paid
        # Gemini response. Rejected rows are retained in the final JSON with
        # their raw text and a warning; none of their characters are edited.
        error: str | None = None
        if not isinstance(raw, str):
            error = "expected a compact citation string"
        else:
            version = obj.get("schema_version")
            parts = raw.split("|", 1 if version == 4 else (2 if version == 3 else 3))
            expected = 2 if version == 4 else (3 if version == 3 else 4)
            if len(parts) != expected or any(
                re.fullmatch(r"[0-9]+", value) is None for value in parts[:-1]
            ):
                error = ("expected LINE|TEXT" if version == 4 else
                         "expected LINE|REF|TEXT" if version == 3 else
                         "expected LINE|REF|OCC|TEXT")
            else:
                line = int(parts[0])
                target = int(parts[1]) if version != 4 else 0
                citation = parts[-1]
                if line < 1:
                    error = "source line must be positive"
                elif not citation:
                    error = "citation text is empty"
                # The response can contain a newline introduced by Gemini,
                # even when the original citation is on a single source line.
                # Keep it for the whitespace-normalized fallback below.
        if error is not None:
            rejected_mentions.append({"index": i, "raw": raw, "warning": error})
            continue
        mentions.append({
            "line": line,
            "exact_text": citation,
            "reference_id": None,
            "decision": "lookup",
            "model_proposed_index": target if version != 4 else None,
        })
    return {
        "schema_version": obj.get("schema_version"),
        "source_id": obj.get("source_id"),
        "references": references,
        "mentions": mentions,
        "rejected_mentions": rejected_mentions,
        "problems": obj.get("problems", []),
        "_compact": True,
    }


def read_response(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CitationError(f"Gemini JSON not found: {path}")
    raw = path.read_text(encoding="utf-8-sig").strip()
    match = RESPONSE_FENCE_RE.fullmatch(raw)
    if match:
        raw = match.group(1)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CitationError(f"Invalid Gemini JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise CitationError("Gemini response must be a JSON object")
    if obj.get("schema_version") in (2, 3, 4):
        obj = normalize_compact_response(obj)
    elif obj.get("schema_version") != 1:
        raise CitationError("Expected schema_version=4 (legacy versions 1, 2 and 3 supported)")
    if not isinstance(obj.get("source_id"), str):
        raise CitationError("Missing string source_id")
    if not isinstance(obj.get("references"), list) or not isinstance(obj.get("mentions"), list):
        raise CitationError("Expected references and mentions arrays")
    obj.setdefault("problems", [])
    if not isinstance(obj["problems"], list) or not all(isinstance(x, str) for x in obj["problems"]):
        raise CitationError("Expected problems array of strings if provided")
    return obj


def validate_prepared_source(request: Path, checksum: str, source_id: str) -> None:
    if not request.is_file():
        raise CitationError(f"Missing prepared input {request.name}. Run 'prepare' first.")
    text = request.read_text(encoding="utf-8")
    expected_id = f"SOURCE_ID: {source_id}\n"
    expected_hash = f"SOURCE_SHA256: {checksum}\n"
    if expected_id not in text or expected_hash not in text:
        raise CitationError(
            "Markdown changed since 'prepare' (or the request file is wrong). "
            "Regenerate the input and obtain a fresh Gemini result."
        )


def protected_ranges(source: str) -> list[tuple[int, int, str]]:
    """Conservatively avoid edits in annotation bodies, code and existing links."""
    spans: list[tuple[int, int, str]] = []

    def add_matches(pattern: str | re.Pattern[str], reason: str, flags: int = 0) -> None:
        for match in re.finditer(pattern, source, flags) if isinstance(pattern, str) else pattern.finditer(source):
            spans.append((match.start(), match.end(), reason))

    add_matches(COMMENT_BODY_RE, "RMX review comment body")
    add_matches(r"<!--[\s\S]*?-->", "HTML comment")
    add_matches(r"(?m)^[ \t]{0,3}\[(?!\^)[^\]\n]+\]:[^\n]*", "existing Markdown link definition")
    add_matches(r"\[[^\]\n]*(?:\[[^\]\n]*\][^\]\n]*)?\](?:\([^\n)]*\)|\[[^\]\n]*\])", "existing Markdown link")
    add_matches(r"(?i)\\(?:cite|parencite|textcite|autocite)[A-Za-z*]*(?:\[[^\]\n]*\])*\{[^}\n]+\}", "existing keyed citation command")
    add_matches(r"(?i)https?://[^\s<>)]*", "URL")
    add_matches(r"(?i)<(?:https?|mailto):[^>\n]+>", "Markdown autolink")
    add_matches(r"<a\b[^>]*>[\s\S]*?</a>", "HTML anchor", re.IGNORECASE)
    add_matches(r"(?m)(`+)(?!`)[^`\n]*?\1(?!`)", "inline code")

    # Fenced code blocks. Even if unclosed, ban everything from the opener.
    lines = source.splitlines(keepends=True)
    offset = 0
    fence: tuple[str, int, int] | None = None
    for line in lines:
        matched = re.match(r"^[ \t]{0,3}(`{3,}|~{3,})([^\n]*)", line)
        if matched:
            marker = matched.group(1)
            if fence is None:
                fence = (marker[0], len(marker), offset)
            elif marker[0] == fence[0] and len(marker) >= fence[1] and not matched.group(2).strip():
                spans.append((fence[2], offset + len(line), "fenced code block"))
                fence = None
        offset += len(line)
    if fence is not None:
        spans.append((fence[2], len(source), "unclosed fenced code block"))
    return spans


def protected_reason(spans: list[tuple[int, int, str]], start: int, end: int) -> str | None:
    for a, b, reason in spans:
        if start < b and end > a:
            return reason
    return None


def normalized_without_comments(value: str) -> str:
    """Citation comparison ONLY: remove comments and collapse whitespace.

    This is a non-destructive view of the source; never publish this string.
    RMX comment bodies are ignored for locating, but remain protected against
    edits. That is, virtual removal never authorizes wrapping across them.
    """
    value = re.sub(r"<!--[\s\S]*?-->", "", value)
    value = COMMENT_BODY_RE.sub("", value)
    return " ".join(value.split())


def normalized_line_spans(body: str, needle: str, line_start: int) -> list[tuple[int, int]]:
    """Return original source spans matching visible normalized citation text.

    Each normalized character maps back to its position in the UNCHANGED raw
    source line. Thus a match may contain a full inline HTML comment between
    visible characters while retaining the comment verbatim in the edit.
    """
    target = normalized_without_comments(needle)
    if not target:
        return []
    ignored = [*re.finditer(r"<!--[\s\S]*?-->", body), *COMMENT_BODY_RE.finditer(body)]
    ignored.sort(key=lambda x: (x.start(), x.end()))
    # A source position maps to its original location, never to its projected
    # offset. Multiple whitespace characters map to one normalized space.
    raw_chars: list[str] = []
    raw_positions: list[int] = []
    cursor = 0
    for m in ignored:
        if m.start() < cursor:
            continue
        for pos in range(cursor, m.start()):
            raw_chars.append(body[pos])
            raw_positions.append(pos)
        cursor = m.end()
    for pos in range(cursor, len(body)):
        raw_chars.append(body[pos])
        raw_positions.append(pos)
    norm_chars: list[str] = []
    norm_positions: list[int] = []
    for ch, pos in zip(raw_chars, raw_positions):
        if ch.isspace():
            if norm_chars and norm_chars[-1] != " ":
                norm_chars.append(" ")
                norm_positions.append(pos)
        else:
            norm_chars.append(ch)
            norm_positions.append(pos)
    haystack = "".join(norm_chars)
    return [
        (line_start + norm_positions[m.start()],
         line_start + norm_positions[m.end() - 1] + 1)
        for m in re.finditer(re.escape(target), haystack)
    ]


def line_match(lines: list[tuple[int, str]], line_number: Any, needle: Any,
               occurrence: Any = None) -> tuple[int, int]:
    if type(line_number) is not int or not (1 <= line_number <= len(lines)):
        raise CitationError(f"Invalid source line: {line_number!r}")
    if occurrence is not None and (type(occurrence) is not int or occurrence < 1):
        raise CitationError(f"Invalid occurrence_on_line: {occurrence!r}")
    if not isinstance(needle, str) or not needle or "\n" in needle or "\r" in needle:
        raise CitationError("exact_text/anchor_text must be nonempty and on one original source line")
    beginning, body = lines[line_number - 1]
    starts = [match.start() for match in re.finditer(re.escape(needle), body)]
    if occurrence is None:
        if len(starts) != 1:
            raise CitationError(
                f"Text {needle[:55]!r} occurs {len(starts)} times on source line {line_number}; "
                "omit occurrence_on_line only when there is exactly one match"
            )
        occurrence = 1
    if len(starts) < occurrence:
        raise CitationError(
            f"Text {needle[:55]!r} not found at occurrence {occurrence} on source line {line_number}"
        )
    i = beginning + starts[occurrence - 1]
    return i, i + len(needle)


def safe_to_wrap(source: str, start: int, end: int,
                 ranges: list[tuple[int, int, str]], text: str,
                 *, fallback: bool = False) -> str | None:
    # A normalized fallback may include COMPLETE inline HTML comments within
    # its span, but can never start or end inside one, or cross a multiline
    # comment. These comments stay verbatim inside the eventual link label.
    allowed = [
        (a, b) for a, b, reason in ranges
        if fallback and reason == "HTML comment" and start <= a and b <= end
        and "\n" not in source[a:b] and "\r" not in source[a:b]
    ]
    for a, b, reason in ranges:
        if start >= b or end <= a:
            continue
        if (a, b) in allowed:
            continue
        if any(c <= a and b <= d for c, d in allowed):
            # A URL or another protected token *inside* an ignored HTML
            # comment is comment text, not a separate source citation.
            continue
        return f"inside {reason}"
    visible_text = source[start:end]
    if allowed:
        for a, b in sorted(set(allowed), reverse=True):
            visible_text = visible_text[:a-start] + visible_text[b-start:]
    if any(marker in visible_text for marker in ("{==", "==}", "{##", "##}", "<!--", "-->", "```")):
        return "crosses an annotation or Markdown delimiter"
    if "][" in visible_text or "]( " in visible_text or "](" in visible_text:
        return "contains nested link syntax"
    # Never link a model-generated line break across source lines or change
    # any comment; only the original physical source span is wrapped.
    return None


def metadata_reference(ref: dict[str, Any], *, anchor_added: bool,
                       warning: str | None) -> dict[str, Any]:
    # The final JSON contains full structured bibliographic metadata. It does not
    # depend on mutable Markdown offsets or the source's original line numbers.
    fields = ["id", "label", "kind", "authors", "editors", "year", "title", "container_title",
              "volume", "issue", "pages", "publisher", "doi", "isbn", "url"]
    result = {key: ref.get(key) for key in fields}
    for field in ("authors", "editors"):
        if isinstance(result.get(field), list):
            result[field] = [v for name in result[field]
                             if isinstance(name, str)
                             if (v := canonical_author(name)) is not None]
    result["anchor_added"] = anchor_added
    result["warning"] = warning
    return result


def apply(markdown: Path, json_path: Path | None = None, force: bool = False,
          *, check_prepared: bool = False,
          generation: dict[str, str] | None = None) -> dict[str, Any]:
    source, checksum, source_id = read_markdown(markdown)
    paths = output_paths(markdown)
    json_path = json_path or paths["proposal"]
    if check_prepared:
        validate_prepared_source(paths["request"], checksum, source_id)
    if paths["metadata"].exists() and not force:
        raise CitationError(
            f"{paths['metadata'].name} already exists; use --force only after "
            "restoring the unenriched source Markdown."
        )
    if paths["legacy_linked"].exists():
        raise CitationError(
            f"Legacy citation output {paths['legacy_linked'].name} exists. "
            "Move or remove it before in-place enrichment; it will never be deleted automatically."
        )
    if has_generated_citation_markup(source):
        raise CitationError(
            "Markdown already contains generated citation links. Restore its original "
            "unenriched version before rerunning; --force cannot safely re-enrich it."
        )

    model = read_response(json_path)
    if model["source_id"] != source_id:
        raise CitationError(
            f"Model source_id {model['source_id']!r} disagrees with the current source {source_id!r}"
        )
    lines = split_source_lines(source)
    bans = protected_ranges(source)
    edits: list[tuple[int, int, str, str]] = []  # start, end, replacement, original
    rejected_mentions = model.get("rejected_mentions", [])
    warnings: list[str] = [
        f"Mention {item['index']}: {item['warning']}; skipped model row without editing"
        for item in rejected_mentions
    ]
    ref_status: dict[str, bool] = {}
    ref_rows: list[dict[str, Any]] = []
    ref_ids: set[str] = set()

    for n, ref in enumerate(model["references"], 1):
        if not isinstance(ref, dict):
            raise CitationError(f"Reference {n}: expected JSON object")
        rid = ref.get("id")
        if not isinstance(rid, str) or not REFERENCE_ID_RE.fullmatch(rid) or rid in ref_ids:
            raise CitationError(f"Reference {n}: invalid or duplicate id {rid!r}")
        ref_ids.add(rid)
        if rid != f"src-ref-{n:03d}":
            raise CitationError(
                f"Reference IDs must be sequential in source order; expected src-ref-{n:03d}, got {rid}"
            )
        warning = None
        try:
            if f'id="{rid}"' in source or f"id='{rid}'" in source:
                raise CitationError("An anchor with this ID already exists in original Markdown")
            start, stop = line_match(lines, ref.get("start_line"), ref.get("anchor_text"))
            # If the anchor substring repeats on its line, refuse to select an arbitrary occurrence.
            if lines[ref["start_line"] - 1][1].count(ref["anchor_text"]) != 1:
                raise CitationError("anchor_text is not unique on its claimed line")
            reason = protected_reason(bans, start, stop)
            if reason:
                raise CitationError(f"anchor text is inside {reason}")
            edits.append((start, start, f'<a id="{rid}"></a>', ""))
            ref_status[rid] = True
        except (CitationError, TypeError, KeyError) as exc:
            warning = str(exc)
            warnings.append(f"Reference {rid}: {warning}")
            ref_status[rid] = False
        ref_rows.append(metadata_reference(ref, anchor_added=ref_status[rid], warning=warning))

    resolver = CitationResolver(model["references"])
    failures: list[dict[str, Any]] = []
    # Python, not the model, determines citation identities.
    # Python expands that decision to every safe occurrence on that source line.
    # Conflicting proposals (including link vs unresolved) disable the WHOLE group.
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for n, mention in enumerate(model["mentions"], 1):
        if not isinstance(mention, dict):
            raise CitationError(f"Mention {n}: expected JSON object")
        key = (mention.get("line"), mention.get("exact_text"))
        try:
            hash(key)
        except TypeError as exc:
            raise CitationError(f"Mention {n}: invalid location") from exc
        groups.setdefault(key, []).append(mention)

    # Two Gemini rows may spell the same visible citation differently, e.g.
    # with a page comment or inconsistent whitespace. Do not let conflicting
    # reference targets for that SAME normalized citation turn into an
    # arbitrary first-match-wins edit.
    normalized_targets: dict[tuple[int, str], set[str | None]] = {}
    for (line_no, text), proposals in groups.items():
        if isinstance(text, str) and type(line_no) is int:
            key = (line_no, normalized_without_comments(text))
            normalized_targets.setdefault(key, set()).update(
                resolver.resolve(text).reference_id
                for p in proposals
            )

    # Each physical match gets one metadata record and potentially one stable ID.
    # Protected matches are never edited, even when another occurrence links.
    bibliography_start_lines = {
        ref.get("start_line") for ref in model["references"]
        if isinstance(ref, dict) and type(ref.get("start_line")) is int
    }
    mention_rows: list[dict[str, Any]] = []
    candidates: list[tuple[int, int, int, dict[str, Any], str]] = []
    for (line_no, text), proposals in groups.items():
        decision = resolver.resolve(text if isinstance(text, str) else "")
        targets = {decision.reference_id}
        normalized_key = (line_no, normalized_without_comments(text)) if isinstance(text, str) and type(line_no) is int else None
        conflict = len(targets) != 1 or (
            normalized_key is not None and len(normalized_targets.get(normalized_key, set())) > 1
        )
        rid = next(iter(targets)) if not conflict else None
        match_info = {
            "match_reason": decision.code,
            "matcher": decision.strategy,
            "match_candidates": resolver.compact_candidates(decision.candidates),
        }
        if conflict:
            warnings.append(
                f"Line {line_no}, {str(text)[:55]!r}: conflicting reference proposals; "
                "no automatic links"
            )
        if type(line_no) is not int or not (1 <= line_no <= len(lines)) or not isinstance(text, str) or not text:
            warnings.append(f"Line {line_no}: invalid citation location or text")
            mention_rows.append({
                **match_info,
                "id": None, "reference_id": None, "exact_text": text,
                "decision": "unresolved", "original_line": line_no,
                "applied": False, "marker_kind": None, "warning": "invalid location or text"
            })
            continue
        beginning, body = lines[line_no - 1]
        matches = [
            (beginning + match.start(), beginning + match.end())
            for match in re.finditer(re.escape(text), body)
        ]
        fallback = not matches
        if fallback:
            matches = normalized_line_spans(body, text, beginning)
        if not matches:
            warnings.append(f"Line {line_no}: text {text[:55]!r} not found (exact or normalized)")
            mention_rows.append({
                **match_info,
                "id": None, "reference_id": None, "exact_text": text,
                "decision": "unresolved", "original_line": line_no,
                "applied": False, "marker_kind": None,
                "warning": "text not on proposed line (exact or normalized)"
            })
            continue
        for start, stop in matches:
            warning = None
            kind = "link" if rid else "marker"
            if line_no in bibliography_start_lines:
                warning = "proposed citation is on a bibliography-entry line"
            else:
                warning = safe_to_wrap(source, start, stop, bans, text, fallback=fallback)
            if rid and not ref_status.get(rid):
                warning = f"target {rid!r} is missing or has no validated bibliography anchor"
            if conflict:
                # Disagreement about the identity precludes even a persistent
                # marker: the proposals require review before enrichment.
                warning = "conflicting reference proposals on this line"
            row = {
                **match_info,
                "id": None, "reference_id": rid if not warning and rid else None,
                "exact_text": source[start:stop] if fallback else text,
                "proposed_text": text if fallback else None,
                "match_method": "normalized_without_comments" if fallback else "exact",
                "decision": "link" if rid and not warning else "unresolved",
                "original_line": line_no, "applied": False,
                "marker_kind": None, "warning": warning,
            }
            row_idx = len(mention_rows)
            mention_rows.append(row)
            if warning:
                warnings.append(f"Line {line_no}, occurrence {start - beginning}: {warning}")
            else:
                candidates.append((start, stop, row_idx, {"reference_id": rid}, kind))

    # Sort by physical location, including when Gemini did not return rows in
    # source order. Never edit overlapping spans or nest citation links.
    candidates.sort(key=lambda item: (item[0], item[1]))
    chosen: list[tuple[int, int, int, dict[str, Any], str]] = []
    for candidate in candidates:
        start, stop, row_idx, mention, kind = candidate
        if chosen and start < chosen[-1][1]:
            msg = "overlaps another proposed citation; skipped to avoid nested links"
            mention_rows[row_idx]["warning"] = msg
            mention_rows[row_idx]["reference_id"] = None
            mention_rows[row_idx]["decision"] = "unresolved"
            warnings.append(f"Line {mention_rows[row_idx]['original_line']}: {msg}")
            continue
        chosen.append(candidate)

    defs: list[str] = []
    accepted_ids: list[tuple[str, str, str]] = []
    linked_count = 0
    marked_count = 0
    for idx, (start, stop, row_idx, mention, kind) in enumerate(chosen, 1):
        cid = f"cite-{idx:04d}"
        if f"[{cid}]" in source or f"cite-id:{cid}" in source:
            raise CitationError(f"Original source already contains citation ID {cid}")
        original = source[start:stop]
        if kind == "link":
            rid = mention["reference_id"]
            replacement = f"[{original}][{cid}]"
            defs.append(f"[{cid}]: #{rid}")
            linked_count += 1
            mention_rows[row_idx]["decision"] = "linked"
            mention_rows[row_idx]["marker_kind"] = "markdown_link"
        else:
            replacement = f"<!-- cite-id:{cid} -->{original}"
            marked_count += 1
            mention_rows[row_idx]["decision"] = "unresolved"
            mention_rows[row_idx]["marker_kind"] = "html_comment"
        edits.append((start, stop, replacement, original))
        accepted_ids.append((cid, replacement, original))
        mention_rows[row_idx]["id"] = cid
        mention_rows[row_idx]["applied"] = True

    # Edits are applied in descending character order. Once linked, no offset is
    # treated as an enduring identifier: the cite-NNNN string in Markdown is.
    revised = source
    for start, stop, replacement, original in sorted(edits, key=lambda x: (x[0], x[1]), reverse=True):
        if revised[start:stop] != original:
            raise CitationError("Internal edit consistency failure; source was not modified")
        revised = revised[:start] + replacement + revised[stop:]

    newline = "\r\n" if "\r\n" in source and source.count("\r\n") >= source.count("\n") else "\n"
    if defs:
        revised += (newline * (2 if not revised.endswith(("\n", "\r")) else 1))
        revised += newline.join(defs) + newline

    # Strong round-trip invariant: after deleting only our generated markup and
    # definitions, the exact original Unicode source must be recovered.
    restored = revised
    if defs:
        appendix = (newline * (2 if not source.endswith(("\n", "\r")) else 1)) + newline.join(defs) + newline
        if not restored.endswith(appendix):
            raise CitationError("Generated definitions could not be verified")
        restored = restored[:-len(appendix)]
    for cid, inserted, original in accepted_ids:
        if restored.count(inserted) != 1:
            raise CitationError(f"Cannot verify exact recovery for {cid}")
        restored = restored.replace(inserted, original, 1)
    for ref in ref_rows:
        if ref["anchor_added"]:
            a = f'<a id="{ref["id"]}"></a>'
            if restored.count(a) != 1:
                raise CitationError(f"Cannot verify anchor recovery for {ref['id']}")
            restored = restored.replace(a, "", 1)
    if restored != source:
        raise CitationError("Round-trip validation failed: generated Markdown changed original text")

    # Include failures after identity lookup, and every physical occurrence.
    for row in mention_rows:
        if row["decision"] == "linked":
            continue
        line_no = row["original_line"]
        failures.append({
            "source_sha256": checksum, "line": line_no, "id": row["id"],
            "text": row["exact_text"],
            "reason": row["match_reason"] if row["match_reason"] != "matched" else "unsafe_citation_location",
            "matcher": row["matcher"], "candidates": row["match_candidates"],
            "warning": row["warning"],
            "context": lines[line_no - 1][1][:240] if type(line_no) is int and 1 <= line_no <= len(lines) else "",
        })

    metadata = {
        "schema_version": model["schema_version"],
        "source_file": markdown.name,
        "linked_file": paths["linked"].name,
        "source_sha256_at_enrichment": checksum,
        "references": ref_rows,
        "mentions": mention_rows,
        "rejected_mentions": rejected_mentions,
        "problems": model["problems"],
        "warnings": warnings,
        "match_failures_file": paths["failures"].name,
        "summary": {
            "match_failures": len(failures),
            "references_found": len(ref_rows),
            "references_anchored": sum(bool(r["anchor_added"]) for r in ref_rows),
            "mentions_found": len(mention_rows) + len(rejected_mentions),
            "mentions_rejected": len(rejected_mentions),
            "mentions_linked": linked_count,
            "mentions_marked_without_link": marked_count,
            "mentions_with_stable_ids": len(accepted_ids),
            "mentions_without_stable_ids": len(mention_rows) - len(accepted_ids),
        },
    }
    if generation is not None:
        metadata["generation"] = generation
    failures_text = "".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
        for item in failures
    )
    atomic_outputs(
        ((paths["metadata"], json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"),
         (paths["failures"], failures_text),
         (markdown, revised)),
        source_path=markdown, force=force, expected_source_sha256=checksum,
    )
    return metadata


def atomic_outputs(
    outputs: tuple[tuple[Path, str], ...], *,
    source_path: Path, force: bool, expected_source_sha256: str,
) -> None:
    """Publish sidecars and Markdown as one rollback-safe group.

    Stage every file before changing any target. Save backups of every
    pre-existing target, publish sidecars first, and replace the Markdown last.
    If any step fails, restore old files or remove newly created files.
    """
    import shutil

    if not source_path.is_file():
        raise CitationError(f"Source Markdown disappeared: {source_path}")
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != expected_source_sha256:
        raise CitationError("Source Markdown changed before publication")
    if len({p for p, _ in outputs}) != len(outputs):
        raise CitationError("Duplicate output destinations")
    for target, _ in outputs:
        if target != source_path and target.exists() and not force:
            raise CitationError(f"{target.name} already exists; use --force to replace it")

    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    retain_backups = False
    try:
        for target, content in outputs:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="",
                prefix=".citation-stage-", suffix=".tmp",
                dir=target.parent, delete=False,
            ) as handle:
                staged[target] = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if target.exists():
                staged[target].chmod(target.stat().st_mode & 0o7777)

        if hashlib.sha256(source_path.read_bytes()).hexdigest() != expected_source_sha256:
            raise CitationError("Source Markdown changed during staging")
        for target, _ in outputs:
            if target.exists():
                with tempfile.NamedTemporaryFile(
                    prefix=".citation-backup-", dir=target.parent, delete=False
                ) as handle:
                    backups[target] = Path(handle.name)
                shutil.copy2(target, backups[target])

        if hashlib.sha256(source_path.read_bytes()).hexdigest() != expected_source_sha256:
            raise CitationError("Source Markdown changed before replacement")
        for target, _ in outputs:
            # Always replace Markdown last. On failure every previously
            # published sidecar is restored to its exact previous bytes.
            published.append(target)
            staged[target].replace(target)
    except BaseException as exc:
        failures: list[str] = []
        for target in reversed(published):
            try:
                if target in backups:
                    backups[target].replace(target)
                else:
                    target.unlink(missing_ok=True)
            except OSError as error:
                failures.append(f"{target.name}: {error}")
        if failures:
            retain_backups = True
            raise CitationError(
                "Publication rollback incomplete; backup files retained: "
                + "; ".join(failures)
            ) from exc
        raise
    finally:
        for temp in staged.values():
            temp.unlink(missing_ok=True)
        if not retain_backups:
            for temp in backups.values():
                temp.unlink(missing_ok=True)


def atomic_pair(file1: Path, text1: str, file2: Path, text2: str, *,
                force: bool, expected_source_sha256: str) -> None:
    """Retain backward-compatible two-output publication API."""
    atomic_outputs(
        ((file2, text2), (file1, text1)), source_path=file1,
        force=force, expected_source_sha256=expected_source_sha256,
    )
