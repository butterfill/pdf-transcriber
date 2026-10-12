"""In-place citation publishing and rollback regression tests (no model calls)."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from pdf_transcriber_gemini import citations
from pdf_transcriber_gemini import citation_service as service


def source_and_proposals(root: Path) -> tuple[Path, Path]:
    md = root / "Article.md"
    md.write_text(
        "# Article\n\nSmith (2020) noted; Smith (2020) agreed.\n\n"
        "## References\nSmith, A. (2020). Example.\n", encoding="utf-8",
    )
    response = root / "response.json"
    response.write_text(json.dumps({
        "schema_version": 4,
        "source_id": citations.read_markdown(md)[2],
        "references": [{"l": 6, "a": "Smith, A. (2020).", "authors": ["Smith, A."], "year": "2020"}],
        "mentions": ["3|Smith (2020)"],
    }), encoding="utf-8")
    return md, response


def test_apply_replaces_md_and_json_without_creating_duplicate(tmp_path: Path):
    md, response = source_and_proposals(tmp_path)
    before = md.read_bytes()
    result = citations.apply(md, json_path=response)
    assert md.read_bytes() != before
    assert md.read_text().count("[Smith (2020)][cite-") == 2
    assert result["source_file"] == "Article.md"
    assert result["linked_file"] == "Article.md"
    assert (tmp_path / "Article.citations.json").exists()
    assert not (tmp_path / "Article.cited.md").exists()
    assert not list(tmp_path.glob(".citation-*"))


def test_already_linked_source_refuses_repeat_even_with_force(tmp_path: Path):
    md, response = source_and_proposals(tmp_path)
    citations.apply(md, json_path=response)
    enriched = md.read_bytes()
    with pytest.raises(citations.CitationError, match="already contains generated citation links"):
        citations.apply(md, json_path=response, force=True)
    assert md.read_bytes() == enriched


def test_sidecar_already_exists_is_protected_without_force(tmp_path: Path):
    md, response = source_and_proposals(tmp_path)
    before = md.read_bytes()
    sidecar = tmp_path / "Article.citations.json"
    sidecar.write_text("untouched")
    with pytest.raises(citations.CitationError, match="already exists"):
        citations.apply(md, json_path=response)
    assert md.read_bytes() == before
    assert sidecar.read_text() == "untouched"


def test_force_replaces_only_sidecar_when_source_is_unlinked(tmp_path: Path):
    md, response = source_and_proposals(tmp_path)
    (tmp_path / "Article.citations.json").write_text("old")
    result = citations.apply(md, json_path=response, force=True)
    assert result["summary"]["mentions_linked"] == 2
    assert (tmp_path / "Article.citations.json").read_text() != "old"


def test_source_commit_failure_rolls_back_new_sidecar(tmp_path: Path, monkeypatch):
    md, response = source_and_proposals(tmp_path)
    before = md.read_bytes()
    original_replace = Path.replace

    def broken_replace(self, target):
        if Path(target) == md and self.name.startswith(".citation-stage-"):
            raise OSError("simulated failure installing Markdown")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", broken_replace)
    with pytest.raises(OSError, match="simulated failure"):
        citations.apply(md, json_path=response)
    assert md.read_bytes() == before
    assert not (tmp_path / "Article.citations.json").exists()
    assert not list(tmp_path.glob(".citation-*"))


def test_source_commit_failure_restores_old_sidecar(tmp_path: Path, monkeypatch):
    md, response = source_and_proposals(tmp_path)
    before = md.read_bytes()
    sidecar = tmp_path / "Article.citations.json"
    sidecar.write_text("old metadata")
    original_replace = Path.replace

    def broken_replace(self, target):
        if Path(target) == md and self.name.startswith(".citation-stage-"):
            raise OSError("simulated failure installing Markdown")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", broken_replace)
    with pytest.raises(OSError, match="simulated failure"):
        citations.apply(md, json_path=response, force=True)
    assert md.read_bytes() == before
    assert sidecar.read_text() == "old metadata"


def test_legacy_cited_md_refused_without_deleting_user_file(tmp_path: Path):
    md, response = source_and_proposals(tmp_path)
    old = md.read_bytes()
    legacy = tmp_path / "Article.cited.md"
    legacy.write_text("older generated output")
    with pytest.raises(citations.CitationError, match="Legacy citation output"):
        citations.apply(md, json_path=response, force=True)
    assert md.read_bytes() == old
    assert legacy.read_text() == "older generated output"


def test_service_resume_cached_response_updates_markdown(tmp_path: Path, monkeypatch):
    md, response = source_and_proposals(tmp_path)
    _, digest, sid = citations.read_markdown(md)
    workspace = tmp_path / ".pdf-citations-resume-test"
    workspace.mkdir()
    (workspace / "source.json").write_text(json.dumps({
        "source": str(md.resolve()), "sha256": digest, "source_id": sid,
    }))
    (workspace / "response.json").write_bytes(response.read_bytes())
    monkeypatch.setattr(service, "_gemini_citations", lambda *a, **kw: pytest.fail("paid twice"))
    assert service.find_citations(md, resume=workspace) == 0
    assert md.read_text().count("[Smith (2020)][cite-") == 2
    assert not (tmp_path / "Article.cited.md").exists()
    assert not workspace.exists()


def test_service_refuses_legacy_duplicate_before_model_call(tmp_path: Path, monkeypatch):
    md, response = source_and_proposals(tmp_path)
    (tmp_path / "Article.cited.md").write_text("legacy output")
    monkeypatch.setattr(service, "_gemini_citations", lambda *a, **kw: pytest.fail("paid"))
    with pytest.raises(FileExistsError, match="Legacy linked"):
        service.find_citations(md)


def test_unresolved_identity_writes_compact_corpus_without_bibtex_keys(tmp_path: Path):
    md = tmp_path / "Article.md"
    md.write_text(
        "# Article\n\nSmith (2020) and Smith et al., 2020 and 2021.\n\n"
        "## References\nSmith, A. (2020). First.\nSmith, B. (2020). Second.\n"
    )
    response = tmp_path / "response.json"
    response.write_text(json.dumps({
        "schema_version": 4, "source_id": citations.read_markdown(md)[2],
        "references": [
            {"l": 6, "a": "Smith, A. (2020).", "authors": ["Smith, A."], "year": "2020"},
            {"l": 7, "a": "Smith, B. (2020).", "authors": ["Smith, B."], "year": "2020"},
        ],
        "mentions": ["3|Smith (2020)", "3|Smith et al., 2020", "3|2021"],
    }))
    result = citations.apply(md, json_path=response)
    corpus = tmp_path / "Article.citation-match-failures.jsonl"
    assert corpus.exists()
    rows = [json.loads(line) for line in corpus.read_text().splitlines()]
    assert result["summary"]["match_failures"] == len(rows)
    assert {r["reason"] for r in rows} >= {"ambiguous_author_year", "missing_cited_author"}
    assert all(r["source_sha256"] == result["source_sha256_at_enrichment"] for r in rows)
    assert all(r["matcher"] and isinstance(r["candidates"], list) for r in rows)
    assert all("citekey" not in r and "bibtex_key" not in r for r in rows)
    assert not any("Smith:2020" in corpus.read_text() for _ in [0])
    assert not (tmp_path / "Article.cited.md").exists()


def test_failure_corpus_rolls_back_on_markdown_install_error(tmp_path: Path, monkeypatch):
    md, response = source_and_proposals(tmp_path)
    old = md.read_bytes()
    original_replace = Path.replace

    def fail_source(self, target):
        if Path(target) == md and self.name.startswith(".citation-stage-"):
            raise OSError("source install blocked")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_source)
    with pytest.raises(OSError, match="source install blocked"):
        citations.apply(md, json_path=response)
    assert md.read_bytes() == old
    assert not (tmp_path / "Article.citations.json").exists()
    assert not (tmp_path / "Article.citation-match-failures.jsonl").exists()
    assert not list(tmp_path.glob(".citation-stage-*"))
    assert not list(tmp_path.glob(".citation-backup-*"))


def test_failure_corpus_is_protected_before_enrichment(tmp_path: Path):
    md, response = source_and_proposals(tmp_path)
    log = tmp_path / "Article.citation-match-failures.jsonl"
    log.write_text("preserve me\n")
    previous = md.read_bytes()
    with pytest.raises(citations.CitationError, match="already exists"):
        citations.apply(md, json_path=response)
    assert log.read_text() == "preserve me\n"
    assert md.read_bytes() == previous
    assert not (tmp_path / "Article.citations.json").exists()


def test_legacy_ref_numbers_never_determine_link_target(tmp_path: Path):
    md, response = source_and_proposals(tmp_path)
    old = json.loads(response.read_text())
    old["schema_version"] = 3
    old["mentions"] = ["3|0|Smith (2020)"]
    response.write_text(json.dumps(old))
    result = citations.apply(md, json_path=response)
    assert result["summary"]["mentions_linked"] == 2
    assert all(m["reference_id"] == "src-ref-001" for m in result["mentions"])


@pytest.mark.parametrize("authors", [["Smith, A.", "et al."], ["Alice Smith", "Brown, B."]])
def test_compact_normalization_preserves_incomplete_author_evidence(tmp_path, authors):
    md, response = source_and_proposals(tmp_path)
    obj = json.loads(response.read_text())
    obj["references"][0]["authors"] = authors
    response.write_text(json.dumps(obj))
    result = citations.apply(md, json_path=response)
    assert result["summary"]["mentions_linked"] == 0
    assert result["summary"]["match_failures"] == 2


def test_failure_log_records_each_occurrence_and_missing_anchor(tmp_path):
    md, response = source_and_proposals(tmp_path)
    obj = json.loads(response.read_text())
    obj["references"][0]["a"] = "nonexistent bibliography snippet"
    response.write_text(json.dumps(obj))
    result = citations.apply(md, json_path=response)
    rows = [json.loads(line) for line in citations.output_paths(md)["failures"].read_text().splitlines()]
    assert result["summary"]["match_failures"] == len(rows) == 2
    assert all(row["reason"] == "unsafe_citation_location" for row in rows)
    assert all("no validated bibliography anchor" in row["warning"] for row in rows)


def test_service_refuses_failure_log_before_any_model_preflight(tmp_path, monkeypatch):
    md, _ = source_and_proposals(tmp_path)
    log = citations.output_paths(md)["failures"]
    log.write_text("existing log")
    monkeypatch.setattr(service, "api_key", lambda: pytest.fail("model preflight reached"))
    monkeypatch.setattr(service, "_gemini_citations", lambda *a, **kw: pytest.fail("paid"))
    with pytest.raises(FileExistsError, match="failure log"):
        service.find_citations(md)
    assert log.read_text() == "existing log"
    assert not list(tmp_path.glob(".pdf-citations-*"))


@pytest.mark.parametrize("failed_target", ["failures", "linked"])
def test_three_file_transaction_restores_all_existing_outputs(tmp_path, monkeypatch, failed_target):
    md, response = source_and_proposals(tmp_path)
    paths = citations.output_paths(md)
    paths["metadata"].write_bytes(b"original metadata\r\n")
    paths["failures"].write_bytes(b"original failures\r\n")
    original = {paths[k]: paths[k].read_bytes() for k in ("linked", "metadata", "failures")}
    replace = Path.replace

    def fail_install(self, target):
        if Path(target) == paths[failed_target] and self.name.startswith(".citation-stage-"):
            raise OSError("injected installation failure")
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_install)
    with pytest.raises(OSError, match="injected installation failure"):
        citations.apply(md, json_path=response, force=True)
    assert all(path.read_bytes() == content for path, content in original.items())
    assert not list(tmp_path.glob(".citation-*"))
