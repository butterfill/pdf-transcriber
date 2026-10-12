"""Offline regression, including a faithful reconstructed excerpt of the user's PDF transcription."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from pdf_transcriber_gemini import citations

def make_long_source(path: Path) -> None:
    """Long, wide lines reproduce same-line duplications without committing a copyrighted article."""
    lines = ["" for _ in range(440)]
    lines[0] = "# Example"
    lines[18] = (
        "A description (Aksoy et al., 2011), followed much later by another "
        "(Aksoy et al., 2011)."
    )
    lines[20] = "As argued (Pomp et al., 2021), and also (Pomp et al., 2021)."
    lines[275] = "These results (Pomp et al., 2021) confirm (Pomp et al., 2021)."
    lines[289] = "Compare (Bosch et al., 2023) with (Bosch et al., 2023)."
    lines[361] = "Aksoy, E. E., Doe, J. (2011). Learning the semantics of object-action relations."
    lines[381] = "Bosch, T. J., Doe, J. (2023). A tool use article."
    lines[437] = "Pomp, J., Heins, N. (2021). Touching events."
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_proposal(source: Path, references: list[dict], mentions: list[str], version: int = 4) -> dict:
    _, _, sid = citations.read_markdown(source)
    refs = [dict(ref) for ref in references]
    for ref in refs:
        # Source-grounded metadata: former location-only fixtures lacked
        # authors and years, which a deterministic matcher rightly requires.
        sample = re.match(r"^([^,]+),\s*([A-Z][^()]*)\((\d{4}[a-z]?)\)", ref["a"])
        if sample:
            ref.setdefault("authors", [f"{sample[1]}, {sample[2].strip()}"])
            ref.setdefault("year", sample[3])
    if version == 4:
        mentions = [re.sub(r"^(\d+)\|\d+\|", r"\1|", text) for text in mentions]
    return {"schema_version": version, "source_id": sid,
            "references": refs, "mentions": mentions}


def trial(tmp_path: Path, rows: list[str], *, references=None, version=4):
    md = tmp_path / "Paper.md"
    make_long_source(md)
    ref = references or [
        {"l": 362, "a": "Aksoy, E. E.", "authors": ["E. E. Aksoy", "J. Doe"], "year": "2011", "doi": "10.1177/0278364911410459"},
        {"l": 382, "a": "Bosch, T. J.", "authors": ["T. J. Bosch", "J. Doe"], "year": "2023"},
        {"l": 438, "a": "Pomp, J., Heins", "authors": ["J. Pomp", "N. Heins"], "year": "2021"}
    ]
    result = make_proposal(md, ref, rows, version=version)
    response = tmp_path / "model.json"
    response.write_text(json.dumps(result), encoding="utf-8")
    return md, response


def test_real_repeated_citations_all_linked_and_original_unchanged(tmp_path: Path):
    md, response = trial(tmp_path, [
        "19|1|Aksoy et al., 2011",
        "21|3|Pomp et al., 2021",
        "276|3|Pomp et al., 2021",
        "290|2|Bosch et al., 2023",
        "290|3|Sacheli, Candidi, Era, & Aglioti (2015)"
    ])
    original = md.read_bytes()
    result = citations.apply(md, json_path=response)
    linked = citations.output_paths(md)["linked"].read_text()
    assert md.read_bytes() != original
    assert not md.with_suffix(".cited.md").exists()
    assert result["summary"]["mentions_linked"] == 8
    assert len([m for m in result["mentions"] if m["warning"] is None]) == 8
    assert result["summary"]["references_anchored"] == 3
    assert linked.count("[Aksoy et al., 2011][cite-") == 2
    assert linked.count("[Pomp et al., 2021][cite-") >= 4
    assert linked.count("[Bosch et al., 2023][cite-") == 2
    assert len({x["id"] for x in result["mentions"] if x["id"]}) == 8
    assert result["references"][0]["doi"] == "10.1177/0278364911410459"
    assert any("not found" in w for w in result["warnings"])


def test_old_model_indexes_are_ignored_even_when_inconsistent(tmp_path: Path):
    md, response = trial(tmp_path, [
        "19|1|Aksoy et al., 2011",
        "19|2|Aksoy et al., 2011"
    ])
    result = citations.apply(md, json_path=response)
    assert result["summary"]["mentions_linked"] == 2
    assert result["summary"]["match_failures"] == 0


def test_old_v2_occurrence_duplicates_do_not_duplicate_links(tmp_path: Path):
    md, response = trial(tmp_path, [
        "19|1|1|Aksoy et al., 2011",
        "19|1|2|Aksoy et al., 2011",
    ], version=2)
    result = citations.apply(md, json_path=response)
    assert result["summary"]["mentions_linked"] == 2


def test_comment_protection_with_duplicate_text(tmp_path: Path):
    md = tmp_path / "Article.md"
    md.write_text("# A\n\nHere is Smith (2020); <!-- Smith (2020) --> and Smith (2020).\n\n## References\nSmith, A. (2020). Example.\n")
    response = tmp_path / "model.json"
    response.write_text(json.dumps(make_proposal(md,[{"l": 6,"a": "Smith, A. (2020)."}],["3|1|Smith (2020)"])))
    result = citations.apply(md, json_path=response)
    linked = citations.output_paths(md)["linked"].read_text()
    assert result["summary"]["mentions_linked"] == 2
    assert "<!-- Smith (2020) -->" in linked
    assert linked.count("[Smith (2020)][cite-") == 2


def test_ambiguous_overlapping_citations_do_not_nested_link(tmp_path: Path):
    md = tmp_path / "Article.md"
    md.write_text("# A\n\nAs Smith (2020) claims.\n\n## References\nSmith, A. (2020). Example.\n")
    rows = ["3|1|Smith (2020)", "3|1|Smith"]
    response = tmp_path / "response.json"
    response.write_text(json.dumps(make_proposal(md,[{"l": 6,"a": "Smith, A. (2020)."}],rows)))
    result = citations.apply(md,json_path=response)
    assert result["summary"]["mentions_linked"] == 0
    assert result["summary"]["match_failures"] == 2
    assert any("overlaps" in w for w in result["warnings"])


def test_request_prompt_shorter_one_row_per_string(tmp_path: Path):
    md = tmp_path / "Article.md"
    md.write_text("A Smith (2020) and Smith (2020).\n")
    request = citations.render_request(md)
    assert "schema_version=4" in request
    assert "LINE|TEXT" in request
    assert "SOURCE_ID" in request
    assert "OCC:" not in request
    assert "L000001|A Smith (2020)" in request


def test_annotated_prompt_adds_conditional_paragraph(tmp_path: Path):
    md = tmp_path / "Article.md"
    md.write_text("{##note: Smith (2020) seems relevant.##}\n")
    request = citations.render_request(md)
    assert "REVIEW COMMENT CITATIONS" in request


def test_malformed_gemini_row_does_not_discard_valid_citations(tmp_path: Path):
    """An invented newline bridging a PDF page comment must not ruin a paid response."""
    md = tmp_path / "Article.md"
    md.write_text(
        "# Article\n\nAs Smith (2020) notes (Borghi, <!-- pdf page 2 --> 2021), "
        "Smith (2020) agrees.\n\n## References\n"
        "Borghi, A. (2021). Work.\n\nSmith, A. (2020). Work.\n",
        encoding="utf-8",
    )
    response = tmp_path / "response.json"
    response.write_text(json.dumps(make_proposal(
        md,
        [{"l": 6, "a": "Borghi, A. (2021)."}, {"l": 8, "a": "Smith, A. (2020)."}],
        ["3|2|Smith (2020)", "3|1|Borghi, \n 2021"],
    )), encoding="utf-8")
    original = md.read_bytes()
    result = citations.apply(md, json_path=response)
    linked = citations.output_paths(md)["linked"].read_text(encoding="utf-8")
    assert md.read_bytes() != original
    assert not md.with_suffix(".cited.md").exists()
    assert linked.count("[Smith (2020)][cite-") == 2
    assert "Borghi, <!-- pdf page 2 --> 2021" in linked
    assert result["summary"]["mentions_linked"] == 3
    assert result["summary"]["mentions_rejected"] == 0
    assert result["summary"]["mentions_found"] == 3
    assert result["rejected_mentions"] == []
    assert "[Borghi, <!-- pdf page 2 --> 2021][cite-" in linked
    assert any(m.get("match_method") == "normalized_without_comments" for m in result["mentions"])
    assert result["references"][0]["anchor_added"] is True
    assert result["references"][1]["anchor_added"] is True


@pytest.mark.parametrize("bad", [
    "3|1|",
    "not a proposal",
    "0|1|Smith (2020)",
    "3|1|",
])
def test_malformed_compact_rows_are_skipped(tmp_path: Path, bad: str):
    md = tmp_path / "Article.md"
    md.write_text("# Article\n\nAs Smith (2020) noted.\n\n## References\nSmith, A. (2020). Work.\n")
    response = tmp_path / "response.json"
    response.write_text(json.dumps(make_proposal(
        md, [{"l": 6, "a": "Smith, A. (2020)."}],
        [bad, "3|1|Smith (2020)"],
    )))
    result = citations.apply(md, json_path=response)
    assert result["summary"]["mentions_linked"] == 1
    assert result["summary"]["mentions_rejected"] == 1
    assert len(result["rejected_mentions"]) == 1
    assert "Mention 1:" in result["warnings"][0]

def test_fallback_page_break_comment_and_newline_model_text(tmp_path: Path):
    """The saved Borghi proposal is linkable while preserving page metadata."""
    md = tmp_path / "Article.md"
    md.write_text(
        "# Article\n\nAs (Borghi, \t<!-- pdf page 2 | article page 1785 -->  2021) notes.\n"
        "\n## References\nBorghi, A. (2021). Example.\n", encoding="utf-8"
    )
    response = tmp_path / "response.json"
    response.write_text(json.dumps(make_proposal(
        md, [{"l": 6, "a": "Borghi, A. (2021)."}], ["3|1|Borghi, \n 2021"]
    )))
    original = md.read_bytes()
    result = citations.apply(md, json_path=response)
    linked = citations.output_paths(md)["linked"].read_text()
    assert md.read_bytes() != original
    assert not md.with_suffix(".cited.md").exists()
    assert "[Borghi, \t<!-- pdf page 2 | article page 1785 -->  2021][cite-0001]" in linked
    assert result["summary"]["mentions_linked"] == 1
    assert result["summary"]["mentions_rejected"] == 0
    assert result["mentions"][0]["match_method"] == "normalized_without_comments"
    assert result["mentions"][0]["proposed_text"] == "Borghi, \n 2021"
    assert result["mentions"][0]["exact_text"] == "Borghi, \t<!-- pdf page 2 | article page 1785 -->  2021"


def test_fallback_whitespace_only_and_repeated_normalized_citations(tmp_path: Path):
    md = tmp_path / "Article.md"
    md.write_text(
        "# Article\n\nAs Smith   (2020) says, Smith  (2020) agrees.\n\n"
        "## References\nSmith, A. (2020). Example.\n"
    )
    response = tmp_path / "response.json"
    response.write_text(json.dumps(make_proposal(
        md, [{"l": 6, "a": "Smith, A. (2020)."}], ["3|1|Smith (2020)"]
    )))
    result = citations.apply(md, json_path=response)
    text = citations.output_paths(md)["linked"].read_text()
    assert result["summary"]["mentions_linked"] == 2
    assert "[Smith   (2020)][cite-0001]" in text
    assert "[Smith  (2020)][cite-0002]" in text
    assert len({m["reference_id"] for m in result["mentions"]}) == 1


def test_rmx_review_comments_can_be_ignored_for_match_but_never_crossed(tmp_path: Path):
    md = tmp_path / "Article.md"
    md.write_text(
        "# Article\n\nAs Smith {##c-1: A note on Smith (2020).##} (2020) claims.\n"
        "\n## References\nSmith, A. (2020). Example.\n"
    )
    response = tmp_path / "response.json"
    response.write_text(json.dumps(make_proposal(
        md, [{"l": 6, "a": "Smith, A. (2020)."}], ["3|1|Smith (2020)"]
    )))
    result = citations.apply(md, json_path=response)
    linked = citations.output_paths(md)["linked"].read_text()
    assert "Smith {##c-1: A note on Smith (2020).##} (2020)" in linked
    assert result["summary"]["mentions_linked"] == 0
    assert any("review comment body" in w for w in result["warnings"])


def test_fallback_never_matches_from_inside_an_html_comment(tmp_path: Path):
    md = tmp_path / "Article.md"
    md.write_text(
        "# Article\n\n<!-- Smith (2020) --> Nothing here.\n\n"
        "## References\nSmith, A. (2020). Example.\n"
    )
    response = tmp_path / "response.json"
    response.write_text(json.dumps(make_proposal(
        md, [{"l": 6, "a": "Smith, A. (2020)."}], ["3|1|Smith  (2020)"]
    )))
    result = citations.apply(md, json_path=response)
    assert result["summary"]["mentions_linked"] == 0


def test_normalized_equivalent_mentions_with_ambiguous_authors_abstain(tmp_path: Path):
    md = tmp_path / "Article.md"
    md.write_text(
        "# Article\n\nAs Smith   (2020) notes.\n\n## References\n"
        "Smith, A. (2020). First.\nSmith, B. (2020). Second.\n"
    )
    response = tmp_path / "response.json"
    response.write_text(json.dumps(make_proposal(md,
        [{"l": 6, "a": "Smith, A. (2020)."}, {"l": 7, "a": "Smith, B. (2020)."}],
        ["3|1|Smith (2020)", "3|2|Smith   (2020)"]
    )))
    result = citations.apply(md, json_path=response)
    assert result["summary"]["mentions_linked"] == 0
    assert result["summary"]["match_failures"] >= 1


def test_fallback_does_not_cross_real_source_lines(tmp_path: Path):
    md = tmp_path / "Article.md"
    md.write_text(
        "# Article\n\nAs Smith, <!-- page break\n2020 --> (2020) notes.\n\n"
        "## References\nSmith, A. (2020). Example.\n"
    )
    response = tmp_path / "response.json"
    response.write_text(json.dumps(make_proposal(
        md, [{"l": 7, "a": "Smith, A. (2020)."}], ["3|1|Smith, (2020)"]
    )))
    result = citations.apply(md, json_path=response)
    assert result["summary"]["mentions_linked"] == 0
    assert any("not found" in w for w in result["warnings"])


def test_fallback_ignores_urls_inside_page_comments(tmp_path: Path):
    md = tmp_path / "Article.md"
    md.write_text(
        "# Article\n\nAs (Smith, <!-- page link https://example.org/x --> 2020) argues.\n"
        "\n## References\nSmith, A. (2020). Example.\n"
    )
    response = tmp_path / "response.json"
    response.write_text(json.dumps(make_proposal(md,
        [{"l": 6, "a": "Smith, A. (2020)."}], ["3|1|Smith, 2020"]
    )))
    result = citations.apply(md, json_path=response)
    assert result["summary"]["mentions_linked"] == 1
    out = citations.output_paths(md)["linked"].read_text()
    assert "[Smith, <!-- page link https://example.org/x --> 2020][cite-0001]" in out
