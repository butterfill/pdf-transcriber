"""Terminal presentation, review/back flow and event streaming (offline)."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pymupdf
import pytest
from rich.console import Console

from pdf_transcriber_gemini import cli, discovery
from pdf_transcriber_gemini.presentation import (
    BatchDisplay, paper_identity, review_selection,
)
from pdf_transcriber_gemini.reporting import capture, emit


PAPER_1 = (
    "Frank et al. - 2016 - Perceptual-Cognitive Changes During Motor Learning "
    "The Influence of Mental and Physical Practice on frank2016_perceptualcognitive.pdf"
)
PAPER_2 = (
    "Schack and Mechsner - 2006 - Representation of motor skills in human "
    "long-term memory schack2006_representation.pdf"
)


def make_pdf(path: Path) -> None:
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.save(path)


def test_review_shows_untruncated_keys_and_never_wraps_at_narrow_width() -> None:
    output = io.StringIO()
    console = Console(file=output, width=82, color_system=None)
    review_selection(console, [Path(PAPER_1), Path(PAPER_2)], standard=False)
    text = output.getvalue()
    assert "REVIEW SELECTION" in text
    assert "2 PDFs · Gemini Flex" in text
    assert "frank2016_perceptualcognitive" in text
    assert "schack2006_representation" in text
    assert all(len(line) <= 82 for line in text.splitlines())
    assert str(Path(PAPER_1)) not in text  # full filename available with v, not default


def test_review_full_paths_optional_and_back_restores_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / PAPER_1
    make_pdf(pdf)
    answers = iter(["v", "b"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    # An interactive rich Console that can be inspected without a real TTY.
    result = discovery.confirm_paid_conversion([pdf], standard=False, dry_run=False)
    assert result == "back"


def test_back_to_selection_then_explicit_yes_preserves_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    pdf = tmp_path / PAPER_1
    make_pdf(pdf)
    chosen = [[pdf], [pdf]]
    seen_initial = []

    def select(items, *, root, initial_selected=None):
        seen_initial.append(initial_selected)
        return chosen.pop(0)

    approvals = iter(["back", True])
    monkeypatch.setattr(cli, "interactive_selection", select)
    monkeypatch.setattr(cli, "confirm_paid_conversion", lambda *a, **k: next(approvals))
    calls = []

    def fake_go(source, *, standard, chunk, on_event):
        calls.append(source)
        on_event("workspace", f"Working folder: {source.parent}/.pdf-transcriber-xyz")
        on_event("model_start", "Transcribing via gemini-3.8-flash (flex; attempt 1/3)...")
        on_event("saved_markdown", f"Wrote: {source.with_suffix('.md')}")
        return 0

    monkeypatch.setattr(cli, "go", fake_go)
    assert cli.convert_missing(papers_dir=tmp_path) == 0
    assert seen_initial == [None, {pdf}]
    assert calls == [pdf]
    text = capsys.readouterr().out
    assert "Markdown saved" in text
    assert "TRANSCRIPTION COMPLETE" in text
    assert "Frank et al." in text


def test_preview_rejects_payment_if_not_explicit_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda *_: "")
    assert discovery.confirm_paid_conversion([Path(PAPER_1)], standard=True, dry_run=False) is False
    assert discovery.confirm_paid_conversion([Path(PAPER_1)], standard=True, dry_run=True) is False


def test_dashboard_states_and_workspace_compact() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, width=78, color_system=None)
    display = BatchDisplay([Path(PAPER_1), Path(PAPER_2)], standard=False,
                           root=Path('/papers'), console=console)
    display.begin(0)
    display.event("workspace", "Working folder: /papers/.pdf-transcriber-abcdef")
    display.event("chunk_plan", "Transcribing 32 pages in 4 chunks of up to 10 pages.")
    display.event("chunk_start", "Chunk 2/4 (original pages 11–20)")
    display.event("chunk_done", "Finished chunk 2/4")
    display.event("model_start", "Transcribing via gemini-3.8-flash (flex; attempt 1/3)...")
    console.print(display.render())
    text = buffer.getvalue()
    assert "● active" in text and "● pending" in text
    assert "1/4 chunks" in text
    assert "Gemini transcription" in text
    assert ".pdf-transcriber-abcdef" in text
    assert "/papers/.pdf-transcriber-abcdef" not in text
    assert all(len(line) <= 78 for line in text.splitlines())
    display.finish(error=RuntimeError("Simulated failure"))
    display.begin(1)
    display.finish()
    final = io.StringIO()
    display.console = Console(file=final, width=78, color_system=None)
    display.summary()
    result = final.getvalue()
    assert "FAILED" in result and "DONE" in result
    assert "TRANSCRIPTION SUMMARY" in result


def test_failed_batch_keeps_recovery_command_and_converts_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    first = tmp_path / "a.pdf"
    second = tmp_path / "b.pdf"
    make_pdf(first)
    make_pdf(second)
    monkeypatch.setattr(cli, "interactive_selection", lambda items, *, root: [first, second])
    monkeypatch.setattr(cli, "confirm_paid_conversion", lambda *a, **k: True)
    calls = []

    def fake_go(source, *, standard, chunk, on_event):
        calls.append(source)
        if source == first:
            on_event("recovery", f"RECOVERY: kept {tmp_path}/.pdf-transcriber-a; retry with:\n"
                                  f"  pdf-transcriber go '{source}' --resume '{tmp_path}/.pdf-transcriber-a'")
            raise RuntimeError("Gemini temporarily unavailable")
        on_event("saved_markdown", f"Wrote: {source.with_suffix('.md')}")
        return 0

    monkeypatch.setattr(cli, "go", fake_go)
    assert cli.convert_missing(papers_dir=tmp_path, parallel=1) == 1
    assert calls == [first, second]
    text = capsys.readouterr().out
    assert "Gemini temporarily unavailable" in text
    assert "--resume" in text
    assert ".pdf-transcriber-a" in text.replace("\n", "")
    assert "1 failed" in text and "1 done" in text


def test_reporting_capture_is_exception_safe() -> None:
    events = []
    with pytest.raises(ZeroDivisionError):
        with capture(lambda event, message: events.append((event, message))):
            emit("model_start", "testing")
            1 / 0
    assert events == [("model_start", "testing")]
    # Outside the context the sink was reset; no leaked dashboard.
    assert paper_identity(Path(PAPER_1))[1] == "frank2016_perceptualcognitive"


def test_run_stop_keeps_pending_after_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    make_pdf(a)
    make_pdf(b)
    monkeypatch.setattr(cli, "interactive_selection", lambda items, *, root: [a, b])
    monkeypatch.setattr(cli, "confirm_paid_conversion", lambda *args, **kw: True)
    calls = []

    def interrupted(source, *, standard, chunk, on_event):
        calls.append(source)
        on_event("recovery", "RECOVERY: kept /papers/.pdf-transcriber-abc; retry with:\n  pdf-transcriber go ...")
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "go", interrupted)
    assert cli.convert_missing(papers_dir=tmp_path, parallel=1) == 1
    assert calls == [a]
    text = capsys.readouterr().out
    assert "1 not started" in text
    assert "PENDING" in text


def test_non_tty_go_retains_plain_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    pdf = tmp_path / "plain.pdf"
    make_pdf(pdf)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")

    def fake(file, workspace, *, standard, prompt, schema):
        emit("uploading", f"Uploading PDF: {file.name}")
        path = workspace / "response.json"
        path.write_text(json.dumps({"markdown": "# Test", "figures": [], "problems": []}))
        return path

    monkeypatch.setattr(cli, "transcribe", fake)
    monkeypatch.setattr(cli, "delete_uploaded", lambda *_: None)
    assert cli.main(["transcribe", str(pdf)]) == 0
    text = capsys.readouterr().out
    assert "Working folder:" in text
    assert "Uploading PDF: plain.pdf" in text
    assert "Wrote:" in text
