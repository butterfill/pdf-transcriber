"""API-free service tests: output placement, resume, and developer workflows."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pdf_transcriber_gemini import citation_service as service
from pdf_transcriber_gemini import citations


def setup_case(root: Path) -> Path:
    md = root / "Essay.md"
    md.write_text("# Essay\n\nAs Smith (2020) puts it, Smith (2020) agrees.\n\n## References\nSmith, A. (2020). *Some title*.\n")
    return md


def model(md: Path) -> dict:
    return {"schema_version": 4, "source_id": citations.read_markdown(md)[2],
            "references": [{"l": 6,"a": "Smith, A. (2020).", "authors":["Smith, A."], "year":"2020", "title":"Some title"}],
            "mentions": ["3|Smith (2020)"]}


def test_find_citations_saves_only_generated_outputs_and_cleans_workspace(tmp_path: Path, monkeypatch):
    md = setup_case(tmp_path)
    old = md.read_bytes()
    monkeypatch.setenv("GEMINI_API_KEY", "fake")

    def mock_gemini(request: str, workspace: Path, *, standard: bool):
        assert "L000003|As Smith (2020)" in request
        assert "LINE|TEXT" in request
        assert standard is False
        response = workspace / "response.json"
        response.write_text(json.dumps(model(md)))
        return response

    monkeypatch.setattr(service, "_gemini_citations", mock_gemini)
    assert service.find_citations(md) == 0
    assert md.read_bytes() != old
    assert md.read_text().count("[Smith (2020)][cite-") == 2
    assert not md.with_name("Essay.cited.md").exists()
    assert (tmp_path / "Essay.citations.json").exists()
    assert not list(tmp_path.glob(".pdf-citations-*"))


def test_resume_reuses_cached_api_response(tmp_path: Path, monkeypatch):
    md = setup_case(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")

    def incomplete(request, workspace, *, standard):
        (workspace / "response.json").write_text("not-json")
        return workspace / "response.json"

    monkeypatch.setattr(service, "_gemini_citations", incomplete)
    with pytest.raises(ValueError):
        service.find_citations(md)
    recovery, = tmp_path.glob(".pdf-citations-*")
    (recovery / "response.json").write_text(json.dumps(model(md)))
    monkeypatch.setattr(service, "_gemini_citations", lambda *a,**k: pytest.fail("repaid for cached response"))
    assert service.find_citations(md, resume=recovery) == 0
    assert not recovery.exists()


def test_developer_prompts_stdout_and_manual_apply(tmp_path: Path, capsys):
    md = setup_case(tmp_path)
    assert service.dev_main(["prompt", "find-citations", str(md)]) == 0
    out = capsys.readouterr().out
    assert "SOURCE_ID" in out
    assert "L000003|As Smith" in out
    proposals = tmp_path / "Essay.citation-proposals.json"
    proposals.write_text(json.dumps(model(md)))
    assert service.dev_main(["apply", "find-citations", str(md)]) == 0
    assert not (tmp_path / "Essay.cited.md").exists()
    assert md.read_text().count("[Smith (2020)][cite-") == 2


def test_output_exists_fails_before_model_call(tmp_path: Path, monkeypatch):
    md = setup_case(tmp_path)
    md.with_name("Essay.cited.md").write_text("Already exists")
    monkeypatch.setattr(service, "_gemini_citations", lambda *a,**k: pytest.fail("should not pay"))
    with pytest.raises(FileExistsError):
        service.find_citations(md)


def test_resume_saved_response_with_invalid_mention_without_model_call(tmp_path: Path, monkeypatch):
    """A parse failure must be recoverable without API credits or credentials."""
    md = setup_case(tmp_path)
    _, sha, sid = citations.read_markdown(md)
    workspace = tmp_path / ".pdf-citations-recovery"
    workspace.mkdir()
    (workspace / "source.json").write_text(json.dumps({
        "source": str(md.resolve()), "sha256": sha, "source_id": sid,
    }))
    response = model(md)
    response["mentions"].append("3|1|Smith (\n2020)")
    (workspace / "response.json").write_text(json.dumps(response))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(
        service, "_gemini_citations", lambda *args, **kwargs: pytest.fail("charged twice"),
    )
    assert service.find_citations(md, resume=workspace) == 0
    result = json.loads((tmp_path / "Essay.citations.json").read_text())
    assert result["summary"]["mentions_linked"] == 2
    assert result["summary"]["mentions_rejected"] == 0
    assert any("not found (exact or normalized)" in w for w in result["warnings"])
    assert not workspace.exists()
