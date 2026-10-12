"""Discover missing PDFs and select paid conversions in a keyboard-driven terminal UI."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_PAPERS_DIR = Path.home() / "syncthing" / "papers"


@dataclass(frozen=True)
class MissingPdf:
    path: Path
    modified: float
    created: float
    creation_available: bool


def discover_missing(papers_dir: Path, *, limit: int | None = None) -> tuple[list[MissingPdf], int]:
    """Mirror pdfmd: recursive *.pdf, same-basename .md, newest mtime first.

    Exclude Syncthing conflict copies and our own crash-recovery chunk PDFs.
    Creation time is stat.birthtime on macOS; use mtime as a stated fallback.
    """
    if not papers_dir.is_dir():
        raise ValueError(f"Papers directory does not exist: {papers_dir}")
    missing: list[MissingPdf] = []
    conflicts = 0
    for pdf in papers_dir.rglob("*.pdf"):
        if any(part.startswith(".pdf-transcriber-") for part in pdf.relative_to(papers_dir).parts[:-1]):
            continue
        if ".sync-conflict-" in pdf.name.casefold():
            conflicts += 1
            continue
        try:
            if pdf.with_suffix(".md").is_file():
                continue
            stat = pdf.stat()
            if not pdf.is_file():
                continue
            birth = getattr(stat, "st_birthtime", None)
            missing.append(MissingPdf(pdf, stat.st_mtime, birth if birth is not None else stat.st_mtime,
                                      birth is not None))
        except OSError:
            continue
    missing.sort(key=lambda p: p.modified, reverse=True)
    if limit is not None:
        if limit < 0:
            raise ValueError("-N must be nonnegative")
        missing = missing[:limit]
    return missing, conflicts


def apply_shortcut(items: list[MissingPdf], selected: set[Path], shortcut: str,
                   *, now: datetime | None = None) -> set[Path]:
    """a=all, n=none, r=today, 1..9=within the last N×24 hours."""
    if shortcut == "a":
        return {item.path for item in items}
    if shortcut == "n":
        return set()
    if shortcut == "r":
        today = (now or datetime.now().astimezone()).date()
        return {item.path for item in items if datetime.fromtimestamp(item.created).date() == today}
    if shortcut in "123456789" and len(shortcut) == 1:
        anchor = now or datetime.now().astimezone()
        cutoff = anchor.timestamp() - int(shortcut) * 24 * 60 * 60
        return {item.path for item in items if cutoff <= item.created <= anchor.timestamp()}
    raise ValueError(f"Unknown selection shortcut: {shortcut}")


def interactive_selection(
    items: list[MissingPdf], *, root: Path = DEFAULT_PAPERS_DIR,
    initial_selected: set[Path] | None = None,
) -> list[Path] | None:
    """Show all candidates, with zero preselected. Return None on cancellation.

    The TUI needs a real terminal: never select everything or run the API as a
    fallback when stdin is piped or the user cannot inspect the selection.
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise RuntimeError("Interactive selection requires a terminal (TTY); no PDFs converted")

    from .presentation import paper_identity
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.styles import Style

    selected: set[Path] = set(initial_selected or set())
    cursor = 0
    offset = 0
    status = ("Selection restored · Space to toggle · Enter review · q cancel"
              if selected else "None selected · ↑↓ move · Enter review · q cancel")
    keys = KeyBindings()

    def visible_rows() -> int:
        try:
            return max(3, os.get_terminal_size().lines - 7)
        except OSError:
            return 15

    def scroll() -> None:
        nonlocal offset
        height = visible_rows()
        if cursor < offset:
            offset = cursor
        if cursor >= offset + height:
            offset = cursor - height + 1

    def header():
        return [("class:title", f" PDF TRANSCRIBER  ·  {len(items)} missing PDF(s)\n"),
                ("class:count", f" {len(selected)} selected   ·   a all   n none   r today   1–9 days   Space toggle\n")]

    def rows():
        view = items[offset:offset + visible_rows()]
        fragments = []
        for index, item in enumerate(view, start=offset):
            chosen = item.path in selected
            active = index == cursor
            filename, _key = paper_identity(item.path)
            try:
                parent = item.path.parent.relative_to(root)
                if parent != Path("."):
                    filename = f"{parent}/  {filename}"
            except ValueError:
                pass
            try:
                columns = os.get_terminal_size().columns
            except OSError:
                columns = 100
            max_width = max(26, columns - 28)
            if len(filename) > max_width:
                filename = filename[:max_width - 1] + "…"
            date_label = datetime.fromtimestamp(item.created).strftime("%Y-%m-%d")
            line = f"{'❯' if active else ' '} [{'x' if chosen else ' '}] {date_label}  {filename}\n"
            fragments.append(("class:cursor" if active else ("class:chosen" if chosen else ""), line))
        return fragments

    def footer():
        return [("class:status", f" {status}\n"),
                ("class:muted", f" {offset+1}–{min(len(items), offset+visible_rows())} of {len(items)}  ·  Nothing sent until you confirm\n")]

    app = Application(
        layout=Layout(HSplit([
            Window(FormattedTextControl(header), height=2),
            Window(FormattedTextControl(rows), height=Dimension(weight=1)),
            Window(FormattedTextControl(footer), height=2),
        ])),
        key_bindings=keys,
        style=Style.from_dict({"title": "bold #eaa33b", "count": "bold #53c995",
                               "cursor": "reverse", "chosen": "bold #53c995",
                               "status": "#b8bec5", "muted": "#959aa1"}),
        full_screen=True,
        mouse_support=False,
    )

    @keys.add("up")
    @keys.add("k")
    def _up(event):
        nonlocal cursor
        cursor = max(0, cursor - 1)
        scroll()

    @keys.add("down")
    @keys.add("j")
    def _down(event):
        nonlocal cursor
        cursor = min(len(items) - 1, cursor + 1)
        scroll()

    @keys.add("space")
    def _toggle(event):
        path = items[cursor].path
        if path in selected:
            selected.remove(path)
        else:
            selected.add(path)

    @keys.add("a")
    @keys.add("n")
    @keys.add("r")
    @keys.add("1")
    @keys.add("2")
    @keys.add("3")
    @keys.add("4")
    @keys.add("5")
    @keys.add("6")
    @keys.add("7")
    @keys.add("8")
    @keys.add("9")
    def _preset(event):
        nonlocal selected, status
        key = event.key_sequence[-1].key
        selected = apply_shortcut(items, selected, key)
        label = {"a":"all", "n":"none", "r":"created today"}.get(key, f"last {key}×24 hours")
        status = f"Selected {len(selected)} ({label}) · Space to toggle individuals · Enter to review"

    @keys.add("enter")
    def _enter(event):
        event.app.exit(result=[item.path for item in items if item.path in selected])

    @keys.add("escape")
    @keys.add("c-c")
    @keys.add("q")
    def _cancel(event):
        event.app.exit(result=None)

    return app.run()


def confirm_paid_conversion(pdfs: list[Path], *, standard: bool, dry_run: bool) -> bool | str:
    """Review compact titles; yes starts payment, b returns to the selection UI.

    v reveals full paths on request. An empty answer always cancels.
    """
    if not pdfs:
        return False
    from rich.console import Console
    from rich.text import Text
    from .presentation import AMBER, MUTED, review_selection

    console = Console()
    full = False
    while True:
        review_selection(console, pdfs, standard=standard, full=full, dry_run=dry_run)
        if dry_run:
            return False
        console.print(Text("  This will send the selected PDFs to Gemini (billable API requests).",
                           style=MUTED))
        console.print(Text("  yes  start    b  back to selection    v  full filenames    Enter  cancel",
                           style=AMBER))
        try:
            reply = input("  Your choice › ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return False
        if reply == "yes":
            return True
        if reply == "b":
            return "back"
        if reply == "v":
            full = not full
            continue
        if not reply or reply in {"n", "no", "q", "cancel"}:
            return False
        console.print(Text("  Type yes to proceed; no API request has been made.", style=AMBER))
