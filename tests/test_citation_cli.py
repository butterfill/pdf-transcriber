"""Argument propagation tests for public citation commands (no model calls)."""
from __future__ import annotations

from pathlib import Path

import pytest

from pdf_transcriber_gemini import cli
from pdf_transcriber_gemini import citation_service


def test_go_codex_model_overrides_citations_only(tmp_path: Path, monkeypatch):
    pdf = tmp_path / "Article.pdf"
    pdf.write_bytes(b"not a real PDF; conversion is stubbed")
    calls = []

    def fake_transcribe(path, *, standard, resume=None, chunk=None, on_event=None):
        calls.append(("transcription", standard, chunk))
        path.with_suffix(".md").write_text("# Example\n", encoding="utf-8")
        return 0

    def fake_find(path, *, standard=False, codex=False, model=None, **kwargs):
        calls.append(("citations", standard, codex, model))
        return 0

    monkeypatch.setattr(cli, "go", fake_transcribe)
    monkeypatch.setattr(citation_service, "find_citations", fake_find)
    monkeypatch.setattr(citation_service, "require_codex", lambda: "/fake/codex")
    assert cli.main(["go", "--standard", "--codex", "--model", "gpt-6.1-sol", str(pdf)]) == 0
    assert calls == [
        ("transcription", True, None),
        ("citations", False, True, "gpt-6.1-sol"),
    ]


def test_find_citations_codex_dispatch(tmp_path: Path, monkeypatch):
    md = tmp_path / "Article.md"
    md.write_text("# A\n", encoding="utf-8")
    seen = []
    def fake_find(path, **kwargs):
        seen.append((path, kwargs))
        return 0
    monkeypatch.setattr(citation_service, "find_citations", fake_find)
    assert cli.main(["find-citations", str(md), "--codex", "--model", "gpt-6-luna"]) == 0
    assert seen[0][1]["codex"] is True
    assert seen[0][1]["model"] == "gpt-6-luna"
    assert cli.main(["find-citations", str(md)]) == 0
    assert seen[1][1]["codex"] is False
    assert seen[1][1]["model"] is None


def test_go_missing_codex_preflight_before_pdf_conversion(tmp_path: Path, monkeypatch, capsys):
    pdf = tmp_path / "Article.pdf"
    pdf.write_bytes(b"fake")
    monkeypatch.setattr(citation_service, "require_codex", lambda: (_ for _ in ()).throw(
        RuntimeError("Codex CLI is not installed")
    ))
    monkeypatch.setattr(cli, "go", lambda *a, **kw: pytest.fail("PDF request was charged"))
    assert cli.main(["go", str(pdf), "--codex"]) == 1
    assert "Codex CLI is not installed" in capsys.readouterr().err
    assert not pdf.with_suffix(".md").exists()


def test_transcribe_does_not_accept_citation_backend_options(tmp_path: Path):
    pdf = tmp_path / "Article.pdf"
    pdf.touch()
    with pytest.raises(SystemExit):
        cli.main(["transcribe", str(pdf), "--codex"])
    with pytest.raises(SystemExit):
        cli.main(["transcribe", str(pdf), "--model", "gpt-6-luna"])
