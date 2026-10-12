"""Offline tests for the public manual AI Studio workflow."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

from pdf_transcriber_gemini import aistudio, cli


def make_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "Karlin et al. - 2021 - Auditory Feedback - karlin2021_auditory.pdf"
    with pymupdf.open() as document:
        page = document.new_page(width=300, height=400)
        page.draw_rect(pymupdf.Rect(60, 100, 210, 230), fill=(0.2, 0.5, 0.9))
        document.save(path)
    return path


def response_json(tmp_path: Path, *, figures: bool = True) -> Path:
    result = {"markdown": "# Transcription\n\nText.\n", "figures": [], "problems": ["Check small print"]}
    if figures:
        result["markdown"] += "\n<!-- FIGURE:fig_1 -->\n"
        result["figures"].append({
            "id": "fig_1", "pdf_page": 1, "box_2d": [250, 200, 575, 700],
            "description": "Figure",
        })
    path = tmp_path / "response.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


@pytest.mark.parametrize("command", ["aistudio", "transcribe"])
def test_direct_import_creates_markdown_and_figures(tmp_path, monkeypatch, command):
    pdf = make_pdf(tmp_path)
    response = response_json(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(cli, "transcribe", lambda *a, **kw: pytest.fail("API called"))
    args = ["aistudio", str(pdf), str(response)] if command == "aistudio" else [
        "transcribe", str(pdf), "--from-json", str(response)
    ]
    assert cli.main(args) == 0
    assert "![Figure]" in pdf.with_suffix(".md").read_text(encoding="utf-8")
    assert (tmp_path / "karlin2021_auditory" / "fig_1.png").is_file()
    assert response.is_file()
    assert list(tmp_path.glob(".pdf-transcriber-import-*")) == []


def test_existing_markdown_is_never_overwritten(tmp_path):
    pdf = make_pdf(tmp_path)
    response = response_json(tmp_path)
    pdf.with_suffix(".md").write_text("Keep me", encoding="utf-8")
    assert cli.main(["aistudio", str(pdf), str(response)]) == 1
    assert pdf.with_suffix(".md").read_text(encoding="utf-8") == "Keep me"


def test_invalid_figure_preserves_original_json_and_no_outputs(tmp_path):
    pdf = make_pdf(tmp_path)
    response = response_json(tmp_path)
    original = json.loads(response.read_text(encoding="utf-8"))
    original["figures"][0]["pdf_page"] = 99
    response.write_text(json.dumps(original), encoding="utf-8")
    assert cli.main(["transcribe", str(pdf), "--from-json", str(response)]) == 1
    assert not pdf.with_suffix(".md").exists()
    assert not (tmp_path / "karlin2021_auditory").exists()
    assert json.loads(response.read_text(encoding="utf-8")) == original


def test_no_figures_does_not_create_directory(tmp_path):
    pdf = make_pdf(tmp_path)
    response = response_json(tmp_path, figures=False)
    assert cli.main(["aistudio", str(pdf), str(response)]) == 0
    assert pdf.with_suffix(".md").exists()
    assert not (tmp_path / "karlin2021_auditory").exists()


def test_prompt_only_no_browser_or_clipboard(tmp_path, monkeypatch, capsys):
    pdf = make_pdf(tmp_path)
    monkeypatch.setattr(aistudio.webbrowser, "open", lambda *a: pytest.fail("Opened browser"))
    monkeypatch.setattr(aistudio, "_clipboard_write", lambda *a: pytest.fail("Clipboard used"))
    assert cli.main(["aistudio", str(pdf), "--prompt-only"]) == 0
    assert "TRANSCRIPTION REQUIREMENT" in capsys.readouterr().out


def test_interactive_clipboard_handoff_reveals_pdf_on_mac(tmp_path, monkeypatch):
    pdf = make_pdf(tmp_path)
    response = response_json(tmp_path)
    clipboard = response.read_text(encoding="utf-8")
    monkeypatch.setattr(aistudio.sys, "platform", "darwin")
    monkeypatch.setattr(aistudio.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    calls = []
    monkeypatch.setattr(aistudio, "_clipboard_write", lambda value: calls.append("write") or True)
    monkeypatch.setattr(aistudio, "_clipboard_read", lambda: clipboard)
    monkeypatch.setattr(aistudio.webbrowser, "open", lambda url: calls.append("browser") or True)
    monkeypatch.setattr(aistudio.subprocess, "run",
                        lambda args, **kw: calls.append(args) or None)
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert cli.main(["aistudio", str(pdf)]) == 0
    assert ["open", "-R", str(pdf)] in calls
    assert "browser" in calls
    assert pdf.with_suffix(".md").is_file()


def test_file_path_interactive_import(tmp_path, monkeypatch):
    pdf = make_pdf(tmp_path)
    response = response_json(tmp_path)
    monkeypatch.setattr(aistudio.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(aistudio, "_clipboard_write", lambda _: True)
    monkeypatch.setattr(aistudio, "_mac_reveal_pdf", lambda _: None)
    monkeypatch.setattr(aistudio.webbrowser, "open", lambda _: True)
    monkeypatch.setattr("builtins.input", lambda _: repr(str(response)).replace('"', "'"))
    assert cli.main(["aistudio", str(pdf)]) == 0
    assert pdf.with_suffix(".md").is_file()


def interactive_terminal(monkeypatch):
    monkeypatch.setattr(aistudio.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(aistudio, "_clipboard_write", lambda _: True)
    monkeypatch.setattr(aistudio, "_mac_reveal_pdf", lambda _: None)
    monkeypatch.setattr(aistudio.webbrowser, "open", lambda _: True)


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, EOFError])
def test_interactive_cancel_has_no_traceback_or_outputs(tmp_path, monkeypatch, capsys, interrupt):
    pdf = make_pdf(tmp_path)
    interactive_terminal(monkeypatch)

    def cancel(_):
        raise interrupt

    monkeypatch.setattr("builtins.input", cancel)
    assert cli.main(["aistudio", str(pdf)]) == 130
    assert "Cancelled" in capsys.readouterr().err
    assert not pdf.with_suffix(".md").exists()
    assert not list(tmp_path.glob(".pdf-transcriber-*"))


def test_interactive_unquoted_path_with_spaces(tmp_path, monkeypatch):
    pdf = make_pdf(tmp_path)
    response = response_json(tmp_path).rename(tmp_path / "AI Studio response.json")
    interactive_terminal(monkeypatch)
    answers = iter([str(response)])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert cli.main(["aistudio", str(pdf)]) == 0
    assert pdf.with_suffix(".md").is_file()


def test_interactive_retries_invalid_clipboard_then_imports_fenced_json(tmp_path, monkeypatch):
    pdf = make_pdf(tmp_path)
    response = response_json(tmp_path)
    interactive_terminal(monkeypatch)
    values = iter(["invalid JSON", "```json\n" + response.read_text() + "\n```"])
    monkeypatch.setattr(aistudio, "_clipboard_read", lambda: next(values))
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert cli.main(["aistudio", str(pdf)]) == 0
    assert pdf.with_suffix(".md").is_file()
    assert not list(tmp_path.glob(".pdf-transcriber-*"))


def test_render_failure_cleans_scratch_and_leaves_no_outputs(tmp_path, monkeypatch):
    from pdf_transcriber_gemini import assembler

    pdf = make_pdf(tmp_path)
    response = response_json(tmp_path)
    original = response.read_bytes()

    def fail_render(*args):
        raise RuntimeError("Cannot render figure")

    monkeypatch.setattr(assembler, "render_figure", fail_render)
    assert cli.main(["aistudio", str(pdf), str(response)]) == 1
    assert response.read_bytes() == original
    assert not pdf.with_suffix(".md").exists()
    assert not (tmp_path / "karlin2021_auditory").exists()
    assert not list(tmp_path.glob(".pdf-transcriber-*"))
