"""Offline tests for bounded parallel convert-missing scheduling and progress routing."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from pdf_transcriber_gemini import cli
from pdf_transcriber_gemini.presentation import BatchDisplay


def selected_pdfs(root: Path, count: int) -> list[Path]:
    result = []
    for i in range(count):
        path = root / f"article-{i:02}.pdf"
        path.write_bytes(b"%PDF-stub")
        result.append(path)
    return result


def choose_all(monkeypatch: pytest.MonkeyPatch, paths: list[Path]) -> None:
    monkeypatch.setattr(cli, "interactive_selection", lambda items, *, root: list(paths))
    monkeypatch.setattr(cli, "confirm_paid_conversion", lambda *args, **kwargs: True)


def capture_display(monkeypatch: pytest.MonkeyPatch) -> list[BatchDisplay]:
    created = []
    original = cli.BatchDisplay

    class RecordingDisplay(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(cli, "BatchDisplay", RecordingDisplay)
    return created


def test_parallel_bounded_and_worker_events_cannot_mix(tmp_path: Path, monkeypatch):
    paths = selected_pdfs(tmp_path, 7)
    choose_all(monkeypatch, paths)
    displays = capture_display(monkeypatch)
    lock = threading.Lock()
    barrier = threading.Barrier(3, timeout=5)
    active = 0
    peak = 0
    started = []

    def fake_go(pdf, *, standard, chunk, on_event):
        nonlocal active, peak
        assert standard is True
        assert chunk is False
        with lock:
            active += 1
            peak = max(peak, active)
            started.append(pdf)
            position = len(started)
        if position <= 3:
            barrier.wait()
        on_event("workspace", f"Working folder: {pdf.parent}/.pdf-transcriber-{pdf.stem}")
        on_event("saved_markdown", f"Wrote: {pdf.with_suffix('.md')}")
        time.sleep(0.03)
        with lock:
            active -= 1
        return 0

    monkeypatch.setattr(cli, "go", fake_go)
    assert cli.convert_missing(
        papers_dir=tmp_path, parallel=3, standard=True, chunk=False
    ) == 0
    assert peak == 3
    assert len(started) == 7 and set(started) == set(paths)
    assert len(displays) == 1
    display = displays[0]
    assert all(s.status == "done" for s in display.states)
    assert all(s.workdir == tmp_path / f".pdf-transcriber-{s.pdf.stem}"
               for s in display.states)
    assert all(s.markdown == s.pdf.with_suffix(".md") for s in display.states)


def test_parallel_failure_does_not_block_queued_jobs_or_lose_recovery(
    tmp_path: Path, monkeypatch,
):
    paths = selected_pdfs(tmp_path, 5)
    choose_all(monkeypatch, paths)
    displays = capture_display(monkeypatch)
    attempted = []
    lock = threading.Lock()

    def fake_go(pdf, *, standard, chunk, on_event):
        with lock:
            attempted.append(pdf)
        on_event("workspace", f"Working folder: {tmp_path}/.pdf-transcriber-{pdf.stem}")
        if pdf == paths[2]:
            on_event(
                "recovery",
                f"RECOVERY: kept {tmp_path}/.pdf-transcriber-{pdf.stem}; retry with:\n"
                f"  pdf-transcriber transcribe '{pdf}' --resume "
                f"'{tmp_path}/.pdf-transcriber-{pdf.stem}'",
            )
            raise RuntimeError("Simulated temporary API problem")
        on_event("saved_markdown", f"Wrote: {pdf.with_suffix('.md')}")
        return 0

    monkeypatch.setattr(cli, "go", fake_go)
    assert cli.convert_missing(papers_dir=tmp_path, parallel=2) == 1
    assert set(attempted) == set(paths)
    state = displays[0].states
    assert sum(s.status == "done" for s in state) == 4
    assert state[2].status == "failed"
    assert "--resume" in state[2].recovery
    assert paths[2].stem in state[2].recovery
    assert state[2].workdir == tmp_path / f".pdf-transcriber-{paths[2].stem}"


def test_parallel_keyboard_interrupt_preserves_not_started(
    tmp_path: Path, monkeypatch,
):
    paths = selected_pdfs(tmp_path, 4)
    choose_all(monkeypatch, paths)
    displays = capture_display(monkeypatch)
    attempted = []

    def fake_go(pdf, *, standard, chunk, on_event):
        attempted.append(pdf)
        if pdf == paths[0]:
            on_event("recovery", f"RECOVERY: kept {tmp_path}/.pdf-transcriber-a")
            raise KeyboardInterrupt
        time.sleep(0.05)
        return 0

    monkeypatch.setattr(cli, "go", fake_go)
    assert cli.convert_missing(papers_dir=tmp_path, parallel=2) == 1
    assert set(attempted) == set(paths[:2])
    states = displays[0].states
    assert states[0].status == "failed"
    assert states[1].status == "done"
    assert [s.status for s in states[2:]] == ["pending", "pending"]


@pytest.mark.parametrize("bad", ["0", "-1", "17", "not-a-number"])
def test_invalid_parallel_count_is_rejected_before_selection(bad, monkeypatch):
    monkeypatch.setattr(
        cli, "interactive_selection",
        lambda *args, **kwargs: pytest.fail("Cannot select or pay with invalid concurrency"),
    )
    with pytest.raises(SystemExit):
        cli.main(["convert-missing", "--parallel", bad])


def test_parallel_cli_flags_and_default(monkeypatch):
    calls = []

    def fake_convert_missing(**options):
        calls.append(options)
        return 0

    monkeypatch.setattr(cli, "convert_missing", fake_convert_missing)
    assert cli.main(["convert-missing"]) == 0
    assert cli.main(["convert-missing", "--parallel", "4"]) == 0
    assert cli.main(["--convert-missing", "-j", "1"]) == 0
    assert [call["parallel"] for call in calls] == [3, 4, 1]


def test_dry_run_never_launches_threads_or_gemini(tmp_path: Path, monkeypatch):
    chosen = selected_pdfs(tmp_path, 4)
    choose_all(monkeypatch, chosen)
    monkeypatch.setattr(
        cli, "go", lambda *args, **kwargs: pytest.fail("Dry-run cannot spawn PDF workers")
    )
    assert cli.convert_missing(papers_dir=tmp_path, dry_run=True, parallel=4) == 0


def test_parallel_display_routes_and_refocuses_active_job(tmp_path: Path):
    import io

    from rich.console import Console

    output = io.StringIO()
    display = BatchDisplay(
        [tmp_path / "a.pdf", tmp_path / "b.pdf"],
        standard=False, root=tmp_path,
        console=Console(file=output, width=82, color_system=None),
    )
    display.begin(0)
    display.begin(1)
    display.event_for(0, "uploading", "Uploading PDF: a.pdf")
    display.event_for(1, "chunk_start", "Chunk 1/3 (original pages 1–10)")
    assert display.states[0].phase == "Uploading PDF"
    assert "Chunk 1/3" in display.states[1].phase
    display.console.print(display.render())
    assert "2 active" in output.getvalue()
    display.finish_for(1)
    assert display.active == 0
    assert display.states[0].status == "running"


@pytest.mark.parametrize("publication_failure", [False, True])
def test_real_pdf_processing_stays_on_main_thread(tmp_path, monkeypatch, publication_failure):
    """Exercise preflight, chunk creation and figure publication with real PDFs."""
    import json

    import pymupdf

    paths = [tmp_path / f"paper-{i}.pdf" for i in range(2)]
    for path in paths:
        with pymupdf.open() as document:
            document.new_page()
            document.new_page()
            document.save(path)
    choose_all(monkeypatch, paths)
    displays = capture_display(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test")
    original_open = pymupdf.open
    calls = []

    def checked_open(*args, **kwargs):
        assert threading.current_thread() is threading.main_thread()
        calls.append(args)
        return original_open(*args, **kwargs)

    monkeypatch.setattr(pymupdf, "open", checked_open)
    barrier = threading.Barrier(2, timeout=5)

    def fake_transcribe(pdf, workspace, **kwargs):
        assert threading.current_thread() is not threading.main_thread()
        barrier.wait()
        result = workspace / "response.json"
        result.write_text(json.dumps({
            "markdown": "Offline transcription\n<!-- FIGURE:fig1 -->",
            "figures": [{"id": "fig1", "pdf_page": 1,
                         "box_2d": [100, 100, 500, 500], "description": "Test figure"}],
            "problems": [],
        }))
        return result

    monkeypatch.setattr(cli, "transcribe", fake_transcribe)
    monkeypatch.setattr(cli, "delete_uploaded", lambda workspace: None)
    original_publish = cli.publish

    def checked_publish(pdf, data, workspace):
        assert threading.current_thread() is threading.main_thread()
        if publication_failure and pdf == paths[1]:
            raise OSError("Simulated disk failure during publication")
        return original_publish(pdf, data, workspace)

    monkeypatch.setattr(cli, "publish", checked_publish)
    assert cli.convert_missing(papers_dir=tmp_path, parallel=2, chunk=True) == int(publication_failure)
    assert len(calls) >= (7 if publication_failure else 8)
    for path in paths[:1] if publication_failure else paths:
        assert "Offline transcription" in path.with_suffix(".md").read_text()
        assert (path.with_suffix("") / "fig_1.png").is_file()
    if publication_failure:
        state = displays[0].states[1]
        assert state.status == "failed"
        assert "Simulated disk failure" in state.error
        assert "transcribe" in state.recovery and "--resume" in state.recovery
        assert (state.workdir / "chunks" / "part-0001" / "response.json").is_file()
        assert (state.workdir / "error.txt").is_file()
        assert not paths[1].with_suffix(".md").exists()
    else:
        assert all(state.status == "done" for state in displays[0].states)
        assert not list(tmp_path.glob(".pdf-transcriber-*"))


@pytest.mark.parametrize("interrupt_at", ["wait", "event", "begin"])
def test_main_thread_sigint_drains_workers_and_restores_handler(
    tmp_path, monkeypatch, interrupt_at,
):
    import concurrent.futures
    import signal

    paths = selected_pdfs(tmp_path, 4)
    choose_all(monkeypatch, paths)
    displays = capture_display(monkeypatch)
    attempted = []
    original_handler = signal.getsignal(signal.SIGINT)
    barrier = threading.Barrier(2, timeout=5)

    def fake_go(pdf, **kwargs):
        attempted.append(pdf)
        barrier.wait()
        kwargs["on_event"]("saved_markdown", f"Wrote: {pdf.with_suffix('.md')}")
        return 0

    monkeypatch.setattr(cli, "go", fake_go)
    if interrupt_at == "wait":
        original = concurrent.futures.wait

        def interrupted_wait(*args, **kwargs):
            signal.raise_signal(signal.SIGINT)
            signal.raise_signal(signal.SIGINT)
            return original(*args, **kwargs)

        monkeypatch.setattr(concurrent.futures, "wait", interrupted_wait)
    elif interrupt_at == "event":
        original = BatchDisplay.event_for

        def interrupted_event(self, *args):
            signal.raise_signal(signal.SIGINT)
            return original(self, *args)

        monkeypatch.setattr(BatchDisplay, "event_for", interrupted_event)
    else:
        original = BatchDisplay.begin

        def interrupted_begin(self, index):
            if index == 1:
                signal.raise_signal(signal.SIGINT)
            return original(self, index)

        monkeypatch.setattr(BatchDisplay, "begin", interrupted_begin)
    assert cli.convert_missing(papers_dir=tmp_path, parallel=2) == 1
    assert set(attempted) == set(paths[:2])
    assert [state.status for state in displays[0].states] == ["done", "done", "pending", "pending"]
    assert signal.getsignal(signal.SIGINT) == original_handler
