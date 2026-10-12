"""Calm, width-aware terminal presentation for a paid batch conversion.

The live dashboard is transient: keep the selection review and final summary,
not a wall of network-status messages. Failures retain complete recovery info.
The Gemini and PDF processing modules know nothing about Rich.
"""
from __future__ import annotations

import re
import shutil
import sys
import time
from contextlib import contextmanager
from typing import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

AMBER = "#EAA33B"
GREEN = "#53C995"
RED = "#F17870"
BLUE = "#83C5DE"
MUTED = "grey62"


def _clean(value: str) -> str:
    """Display text, not terminal control characters, from untrusted filenames."""
    return "".join(" " if ord(c) < 32 or ord(c) == 127 else c for c in value)


def paper_identity(pdf: Path) -> tuple[str, str]:
    """Produce a readable short article label and the bibliographic key, if known."""
    stem = _clean(pdf.stem)
    pieces = stem.split(" - ", 2)
    if len(pieces) != 3:
        return stem, ""
    author, year, remainder = pieces
    tail = remainder.rsplit(" ", 1)
    key = tail[-1] if len(tail) == 2 and re.search(r"\d{4}", tail[-1]) else ""
    title = tail[0] if key else remainder
    label = f"{author} · {year}  {title}" if year else f"{author}  {title}"
    return label, key


def _file_row(pdf: Path) -> Text:
    return Text(_clean(str(pdf)), style="white", no_wrap=False)


def _review_table(pdfs: Sequence[Path], *, full: bool) -> Table:
    table = Table(box=box.SIMPLE, expand=True, show_edge=False,
                  header_style=f"bold {MUTED}", padding=(0, 1), show_lines=False)
    table.add_column("", width=4, justify="right", no_wrap=True)
    table.add_column("PAPER", ratio=5, overflow="ellipsis", no_wrap=not full)
    if not full:
        longest_key = max((len(paper_identity(pdf)[1]) for pdf in pdfs), default=0)
        table.add_column("KEY", width=max(12, min(longest_key, 38)), style=MUTED,
                         no_wrap=True, overflow="ellipsis")
    for i, pdf in enumerate(pdfs, 1):
        label, key = paper_identity(pdf)
        number = Text(f"{i:02d}", style=AMBER)
        if full:
            table.add_row(number, _file_row(pdf))
        else:
            paper = Text(label, style="white", no_wrap=True, overflow="ellipsis")
            table.add_row(number, paper, Text(key, style=MUTED))
    return table


def review_selection(console: Console, pdfs: Sequence[Path], *, standard: bool, full: bool = False,
                     dry_run: bool = False) -> None:
    tier = "Standard" if standard else "Flex"
    title = Text("  REVIEW SELECTION", style="bold white")
    title.append(f"    {len(pdfs)} PDFs · Gemini {tier}", style=MUTED)
    console.print()
    console.print(title)
    console.print(_review_table(pdfs, full=full))
    if dry_run:
        console.print(Text("  DRY RUN · No API requests will be made", style=AMBER))


@dataclass
class RunState:
    pdf: Path
    status: str = "pending"  # pending, running, done, failed
    phase: str = "Queued"
    workdir: Path | None = None
    markdown: Path | None = None
    figures: Path | None = None
    chunk_total: int = 0
    chunk_done: int = 0
    chunk_current: int = 0
    seconds: float = 0
    started: float = 0
    error: str = ""
    recovery: str = ""
    warning: list[str] | None = None


class BatchDisplay:
    """Transient running view, persistent compact results, with no API coupling."""

    def __init__(self, pdfs: Sequence[Path], *, standard: bool, root: Path,
                 console: Console | None = None) -> None:
        self.console = console or Console()
        self.root = root
        self.standard = standard
        self.states = [RunState(pdf=pdf, warning=[]) for pdf in pdfs]
        self.active = -1
        self.started = time.monotonic()
        self.live: Live | None = None

    def __enter__(self) -> BatchDisplay:
        if self.console.is_terminal:
            self.live = Live(console=self.console, get_renderable=self.render,
                             refresh_per_second=8, transient=True, vertical_overflow="ellipsis")
            self.live.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.live is not None:
            self.live.stop()

    def _update(self) -> None:
        if self.live is not None:
            self.live.refresh()

    def begin(self, index: int) -> None:
        self.active = index
        state = self.states[index]
        state.status = "running"
        state.phase = "Preparing PDF"
        state.started = time.monotonic()
        self._update()

    def event_for(self, index: int, event: str, message: str) -> None:
        """Apply an individual worker's progress on the UI/main thread."""
        self.active = index
        self.event(event, message)

    def finish_for(self, index: int, *, error: Exception | None = None) -> None:
        """Complete one conversion without losing focus on remaining workers."""
        self.active = index
        self.finish(error=error)
        for other in range(len(self.states) - 1, -1, -1):
            if self.states[other].status == "running":
                self.active = other
                self._update()
                break

    def event(self, event: str, message: str) -> None:
        if self.active < 0:
            return
        state = self.states[self.active]
        if event == "workspace":
            state.workdir = Path(message.removeprefix("Working folder: "))
        elif event in {"uploading", "upload_reuse", "upload_retry", "uploaded"}:
            state.phase = {
                "uploading": "Uploading PDF",
                "upload_reuse": "Reusing previous upload",
                "upload_retry": "Retrying upload",
                "uploaded": "Upload complete",
            }[event]
        elif event == "chunk_plan":
            match = re.search(r"in (\d+) chunks", message)
            if match:
                state.chunk_total = int(match.group(1))
        elif event == "chunk_start":
            match = re.search(r"Chunk (\d+)/(\d+) \(original pages ([^)]*)\)", message)
            if match:
                state.chunk_current = int(match.group(1))
                state.chunk_total = int(match.group(2))
                state.phase = f"Chunk {match.group(1)}/{match.group(2)} · pages {match.group(3)}"
        elif event == "chunk_done":
            state.chunk_done += 1
            state.phase = f"Chunk {state.chunk_done}/{state.chunk_total} complete"
        elif event == "model_start":
            match = re.search(r"attempt (\d+)/(\d+)", message)
            suffix = f" · attempt {match.group(1)}/{match.group(2)}" if match else ""
            prefix = f"Chunk {state.chunk_current}/{state.chunk_total} · " if state.chunk_total else ""
            state.phase = f"{prefix}Gemini transcription{suffix}"
        elif event == "model_done":
            state.phase = "Gemini response saved"
        elif event == "retry":
            state.phase = "Waiting to retry Gemini request"
        elif event in {"chunk_cached", "cached"}:
            state.phase = "Reusing saved Gemini response"
        elif event == "assembling":
            state.phase = "Extracting figures and writing Markdown"
        elif event == "saved_markdown":
            state.markdown = Path(message.removeprefix("Wrote: "))
            state.phase = "Markdown written"
        elif event == "saved_figures":
            state.figures = Path(message.removeprefix("Figures: ").rstrip("/"))
        elif event == "no_figures":
            state.phase = "No substantive figures"
        elif event == "recovery":
            state.recovery = message
        elif event in {"warning", "note"}:
            assert state.warning is not None
            state.warning.append(message)
        self._update()

    def finish(self, *, error: Exception | None = None) -> None:
        state = self.states[self.active]
        state.seconds = time.monotonic() - state.started
        if error is None:
            state.status = "done"
            state.phase = "Done"
        else:
            state.status = "failed"
            state.phase = "Failed"
            state.error = _clean(str(error))
        self._update()

    def _status(self, state: RunState) -> Text:
        if state.status == "done":
            return Text("✓ done", style=f"bold {GREEN}")
        if state.status == "failed":
            return Text("✕ failed", style=f"bold {RED}")
        if state.status == "running":
            return Text("● active", style=f"bold {BLUE}")
        return Text("● pending", style=AMBER)

    def render(self) -> Group:
        n = len(self.states)
        done = sum(s.status == "done" for s in self.states)
        failed = sum(s.status == "failed" for s in self.states)
        running = sum(s.status == "running" for s in self.states)
        mode = "STANDARD" if self.standard else "FLEX"
        head = Text("PDF TRANSCRIBER", style="bold white")
        head.append(f"   ·   Gemini 3.8 Flash / {mode}", style=MUTED)
        head.append(f"   ·   {done} done", style=GREEN)
        if running > 1:
            head.append(f"   ·   {running} active", style=BLUE)
        if failed:
            head.append(f"   ·   {failed} failed", style=RED)
        width = max(14, min(32, self.console.width - 35))
        good_units = round(width * done / n) if n else 0
        bad_units = round(width * failed / n) if n else 0
        track = Text("  ")
        track.append("━" * good_units, style=GREEN)
        track.append("━" * bad_units, style=RED)
        track.append("─" * (width - good_units - bad_units), style=AMBER)
        track.append(f"   {done + failed}/{n} processed", style=MUTED)
        elapsed = int(time.monotonic() - self.started)
        track.append(f"   {elapsed // 60}:{elapsed % 60:02d}", style=MUTED)
        table = Table(box=None, expand=True, padding=(0, 1), show_header=False,
                      show_edge=False, show_lines=False)
        table.add_column("", width=3, no_wrap=True)
        table.add_column("PDF", ratio=4, no_wrap=True, overflow="ellipsis")
        table.add_column("STATUS", width=12, no_wrap=True, justify="right")
        try:
            height = shutil.get_terminal_size((100, 25)).lines
        except OSError:
            height = 25
        max_rows = min(n, max(3, min(9, height - 14)))
        start = min(max(0, self.active - 2), max(0, n - max_rows)) if self.active >= 0 else 0
        end = min(n, start + max_rows)
        if start:
            table.add_row("", Text(f"··· {start} earlier", style=MUTED), "")
        for i in range(start, end):
            item = self.states[i]
            label, _ = paper_identity(item.pdf)
            selected = i == self.active
            label_style = "bold white" if selected else ("white" if item.status == "done" else MUTED)
            table.add_row(
                Text(f"{i + 1:02d}", style=BLUE if selected else MUTED),
                Text(label, style=label_style, overflow="ellipsis", no_wrap=True),
                self._status(item),
            )
        if end < n:
            table.add_row("", Text(f"··· {n - end} more", style=MUTED), "")
        current = self.states[self.active] if self.active >= 0 else None
        detail: list = []
        if current and current.status == "running":
            stage = Text("   ")
            stage.append_text(Text("CURRENT", style=f"bold {BLUE}"))
            stage.append(f"   {self.active + 1}/{n}  ", style=MUTED)
            stage.append(_clean(current.pdf.name), style="white")
            stage.no_wrap = True
            stage.overflow = "ellipsis"
            detail.append(stage)
            detail.append(Spinner("dots", text=Text("  " + current.phase, style=BLUE), style=BLUE))
            if current.chunk_total > 1:
                completed = current.chunk_done
                width = 24
                filled = round(width * completed / current.chunk_total)
                bar = Text("  ")
                bar.append("━" * filled, style=GREEN)
                bar.append("─" * (width - filled), style="grey37")
                bar.append(f"  {completed}/{current.chunk_total} chunks", style=MUTED)
                detail.append(bar)
            if current.workdir:
                work = Text("  Workspace  ", style=MUTED)
                work.append(current.workdir.name, style="white")
                work.append("   ·   same folder as PDF", style=MUTED)
                detail.append(work)
        return Group(
            head, track, table,
            Panel(Group(*detail), border_style="grey30", padding=(0, 1), title="In progress", title_align="left")
            if detail else Text("", style=MUTED),
        )

    def summary(self, *, interrupted: bool = False) -> None:
        successes = sum(s.status == "done" for s in self.states)
        failures = sum(s.status == "failed" for s in self.states)
        remaining = sum(s.status == "pending" for s in self.states)
        elapsed = time.monotonic() - self.started
        title = Text("  TRANSCRIPTION COMPLETE", style=f"bold {GREEN}")
        if failures or interrupted:
            title = Text("  TRANSCRIPTION SUMMARY", style="bold white")
        title.append(f"    {successes} done", style=GREEN)
        if failures:
            title.append(f"  ·  {failures} failed", style=RED)
        if remaining:
            title.append(f"  ·  {remaining} not started", style=AMBER)
        title.append(f"  ·  {int(elapsed // 60)}m {int(elapsed % 60):02d}s", style=MUTED)
        self.console.print()
        self.console.print(title)
        output = Table(box=box.SIMPLE, expand=True, show_edge=False, padding=(0, 1),
                       show_header=False)
        output.add_column("", width=10, no_wrap=True)
        output.add_column("PAPER", ratio=4, no_wrap=True, overflow="ellipsis")
        output.add_column("OUTPUT", ratio=2, no_wrap=True, overflow="ellipsis")
        for state in self.states:
            name, _key = paper_identity(state.pdf)
            if state.status == "done":
                label = Text("✓  DONE", style=f"bold {GREEN}")
                dest = Text("Markdown saved", style=GREEN)
                if state.figures:
                    dest.append("  + figures", style=MUTED)
                dest.append(f"  ·  {int(state.seconds)}s", style=MUTED)
            elif state.status == "failed":
                label = Text("✕  FAILED", style=f"bold {RED}")
                dest = Text(state.error, style=RED, no_wrap=True, overflow="ellipsis")
            else:
                label = Text("●  PENDING", style=AMBER)
                dest = Text("not sent", style=MUTED)
            output.add_row(label, Text(name, overflow="ellipsis", no_wrap=True), dest)
        self.console.print(output)
        for state in self.states:
            for warning in state.warning or []:
                self.console.print(Text("  ! " + warning, style=AMBER))
            if state.status == "failed":
                self.console.print(Text("  Failed: " + state.pdf.name, style=RED))
                self.console.print(Text("  Reason: " + state.error, style=RED))
                if state.recovery:
                    # Complete recovery command must not be truncated.
                    self.console.print(Text("  " + state.recovery.replace("\n", "\n  "), style=MUTED))
                else:
                    self.console.print(Text("  " + state.error, style=RED))
        if not failures and not interrupted:
            self.console.print(Text("  Markdown and figures saved alongside the source PDFs.", style=MUTED))


@contextmanager
def quiet_terminal_input() -> Iterator[None]:
    """Avoid echoing stray arrow-key sequences during long API requests.

    Keep ISIG enabled so Ctrl-C still interrupts. Only for an interactive TTY,
    and always restore the original mode, including after an exception.
    """
    try:
        import termios
        if not sys.stdin.isatty():
            yield
            return
        fd = sys.stdin.fileno()
        original = termios.tcgetattr(fd)
    except (ImportError, OSError, ValueError):
        yield
        return
    changed = original.copy()
    changed[3] &= ~termios.ECHO
    try:
        termios.tcsetattr(fd, termios.TCSANOW, changed)
        yield
    finally:
        try:
            termios.tcflush(fd, termios.TCIFLUSH)
        finally:
            termios.tcsetattr(fd, termios.TCSANOW, original)
