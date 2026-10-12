"""pdf-transcriber: direct PDF-to-Markdown transcription with Gemini."""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import sys
import tempfile
import traceback
from collections.abc import Callable
from contextvars import ContextVar
from importlib.resources import files
from pathlib import Path

import pymupdf

from .api import api_key, delete_uploaded, transcribe
from .assembler import figure_folder_name, load_gemini_json, publish, validate_figures
from .chunking import (
    AUTO_CHUNK_THRESHOLD,
    CHUNK_PAGES,
    chunk_prompt,
    chunks_for,
    combine_chunks,
    write_chunk,
)
from .discovery import (
    DEFAULT_PAPERS_DIR,
    confirm_paid_conversion,
    discover_missing,
    interactive_selection,
)
from .presentation import BatchDisplay, quiet_terminal_input
from .reporting import capture, emit

# PyMuPDF must run on one thread. Parallel API workers dispatch all local PDF
# operations to the scheduler's main thread; ordinary conversions call directly.
_pdf_dispatch: ContextVar[Callable | None] = ContextVar("pdf_dispatch", default=None)


def _pdf_operation(operation: Callable, *args, **kwargs):
    dispatch = _pdf_dispatch.get()
    if dispatch is None:
        return operation(*args, **kwargs)
    return dispatch(operation, args, kwargs)


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _preflight(path: Path) -> tuple[Path, int]:
    pdf = path.expanduser().resolve(strict=True)
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        raise ValueError("Provide an existing PDF file")
    if pdf.with_suffix(".md").exists():
        raise FileExistsError(f"{pdf.with_suffix('.md')} already exists; refusing to overwrite")
    figure_dir = pdf.parent / figure_folder_name(pdf.stem)
    if figure_dir.exists():
        raise FileExistsError(f"Figure directory already exists: {figure_dir}")
    with pymupdf.open(pdf) as document:
        if document.needs_pass:
            raise ValueError("PDF is password-protected")
        count = document.page_count
    if count < 1:
        raise ValueError("PDF contains no pages")
    return pdf, count


def _single_response(pdf: Path, workspace: Path, *, standard: bool, prompt: str, schema: dict) -> dict:
    response_path = workspace / "response.json"
    if not response_path.exists():
        response_path = transcribe(pdf, workspace, standard=standard, prompt=prompt, schema=schema)
    else:
        emit("cached", "Reusing saved response.json (no new API call).")
    return load_gemini_json(response_path)


def _chunked_response(
    pdf: Path, workspace: Path, *, pages: int, standard: bool, prompt: str, schema: dict
) -> dict:
    specs = chunks_for(pages)
    emit("chunk_plan", f"Transcribing {pages} pages in {len(specs)} chunks of up to {CHUNK_PAGES} pages.")
    chunk_results = []
    for spec in specs:
        chunk_workspace = workspace / "chunks" / spec.name
        chunk_workspace.mkdir(parents=True, exist_ok=True)
        response_path = chunk_workspace / "response.json"
        emit("chunk_start", f"Chunk {spec.index}/{len(specs)} (original pages {spec.start}–{spec.end})")
        if response_path.is_file():
            emit("chunk_cached", "Reusing completed chunk response.json (no new API call).")
        else:
            chunk_pdf = _pdf_operation(write_chunk, pdf, spec, chunk_workspace)
            response_path = transcribe(
                chunk_pdf, chunk_workspace, standard=standard,
                prompt=chunk_prompt(prompt, spec, pages), schema=schema,
            )
        data = load_gemini_json(response_path)
        # Catch model errors now, before paying for subsequent chunks.
        validate_figures(data, spec.pages)
        chunk_results.append((spec, data))
        emit("chunk_done", f"Finished chunk {spec.index}/{len(specs)}")
    combined = combine_chunks(chunk_results)
    (workspace / "combined.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return combined


def _go_impl(
    pdf_path: Path, *, standard: bool = False, resume: Path | None = None,
    chunk: bool | None = None,
) -> int:
    pdf, pages = _pdf_operation(_preflight, pdf_path)
    requested_chunked = (pages > AUTO_CHUNK_THRESHOLD) if chunk is None else chunk
    if resume is None:
        api_key()  # avoid creating temporary folders for a missing credential
        workspace = Path(tempfile.mkdtemp(prefix=".pdf-transcriber-", dir=pdf.parent))
        meta = {
            "pdf": str(pdf), "sha256": _digest(pdf),
            "model": "gemini-3.8-flash", "service_tier": "standard" if standard else "flex",
            "chunked": requested_chunked, "chunk_pages": CHUNK_PAGES,
        }
        (workspace / "source.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    else:
        workspace = resume.expanduser().resolve(strict=True)
        if not workspace.is_dir():
            raise ValueError("--resume must point to a working directory")
        meta = json.loads((workspace / "source.json").read_text(encoding="utf-8"))
        if meta.get("pdf") != str(pdf) or meta.get("sha256") != _digest(pdf):
            raise ValueError("Recovery folder belongs to another PDF, or PDF contents have changed")
        if workspace.parent != pdf.parent or not workspace.name.startswith(".pdf-transcriber-"):
            raise ValueError("Recovery folder must be a pdf-transcriber working folder beside the PDF")
        was_chunked = bool(meta.get("chunked", False))  # old recovery folders were unchunked
        if chunk is not None and chunk != was_chunked:
            raise ValueError("--chunk / --no-chunk conflicts with this recovery folder's original mode")
        if meta.get("chunk_pages", CHUNK_PAGES) != CHUNK_PAGES:
            raise ValueError("Recovery folder uses an unsupported chunk size")
    use_chunks = bool(meta.get("chunked", False))
    emit("workspace", f"Working folder: {workspace}")
    if pages > 40 and not use_chunks:
        emit("warning", f"WARNING: {pages} unchunked pages may exceed the model's output token limit.")
    try:
        prompt = files("pdf_transcriber_gemini").joinpath("prompt.txt").read_text(encoding="utf-8")
        schema = json.loads(
            files("pdf_transcriber_gemini").joinpath("schema.json").read_text(encoding="utf-8")
        )
        if use_chunks:
            data = _chunked_response(
                pdf, workspace, pages=pages, standard=standard, prompt=prompt, schema=schema
            )
        else:
            data = _single_response(pdf, workspace, standard=standard, prompt=prompt, schema=schema)
        emit("assembling", "Extracting figures and assembling Markdown")
        md, figdir = _pdf_operation(publish, pdf, data, workspace)
        emit("saved_markdown", f"Wrote: {md}")
        if figdir:
            emit("saved_figures", f"Figures: {figdir}/")
        else:
            emit("no_figures", "No substantive figures")
        for note in data["problems"]:
            emit("note", f"MODEL NOTE: {note}")
        if use_chunks:
            for spec in chunks_for(pages):
                delete_uploaded(workspace / "chunks" / spec.name)
        else:
            delete_uploaded(workspace)
        shutil.rmtree(workspace)
        emit("success", f"Completed: {pdf.name}")
        return 0
    except BaseException:
        (workspace / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        emit("recovery",
             f"RECOVERY: kept {workspace}; retry with:\n"
             f"  pdf-transcriber transcribe {shlex.quote(str(pdf))} --resume {shlex.quote(str(workspace))}")
        raise


def go(
    pdf_path: Path, *, standard: bool = False, resume: Path | None = None,
    chunk: bool | None = None,
    on_event: Callable[[str, str], None] | None = None,
) -> int:
    """Run one conversion, optionally reporting structured events to a live batch view."""
    with capture(on_event):
        return _go_impl(pdf_path, standard=standard, resume=resume, chunk=chunk)


def _parallel_batch(
    selected: list[Path], display: BatchDisplay, *, workers: int,
    standard: bool, chunk: bool | None,
) -> bool:
    """Run independent PDFs concurrently, keeping all Rich updates on the main thread.

    At most workers conversions are submitted at any time. The worker threads
    never touch the UI: context-local go() events are delivered via a queue.
    Ctrl-C stops submitting new PDFs, but already-started uploads/requests are
    allowed to finish to preserve safe publication and recovery semantics.
    """
    import signal
    from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
    from queue import Empty, SimpleQueue

    messages: SimpleQueue[tuple[int, str, str]] = SimpleQueue()
    pdf_operations: SimpleQueue = SimpleQueue()
    inflight: dict[Future[int], int] = {}
    next_index = 0
    interrupted = False

    def dispatch(operation, args, kwargs):
        result = Future()
        pdf_operations.put((operation, args, kwargs, result))
        return result.result()

    def worker(index):
        token = _pdf_dispatch.set(dispatch)
        try:
            return go(
                selected[index], standard=standard, chunk=chunk,
                on_event=lambda event, message: messages.put((index, event, message)),
            )
        finally:
            _pdf_dispatch.reset(token)

    def stop_submission(_signum=None, _frame=None):
        nonlocal interrupted
        interrupted = True

    def drain_pdf_operations():
        while True:
            try:
                operation, args, kwargs, result = pdf_operations.get_nowait()
            except Empty:
                return
            try:
                value = operation(*args, **kwargs)
            except BaseException as exc:
                result.set_exception(exc)
            else:
                result.set_result(value)

    def drain_events() -> None:
        while True:
            try:
                index, event, message = messages.get_nowait()
            except Empty:
                return
            display.event_for(index, event, message)

    def fill(executor: ThreadPoolExecutor) -> None:
        nonlocal next_index
        while not interrupted and len(inflight) < workers and next_index < len(selected):
            index = next_index
            future = executor.submit(worker, index)
            inflight[future] = index
            next_index += 1
            display.begin(index)

    # A handler sets a flag instead of raising between submit() and bookkeeping,
    # or halfway through PDF publication/UI updates. Restore it even on failure.
    previous_handler = signal.signal(signal.SIGINT, stop_submission)
    announced = False
    try:
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="pdf-transcriber"
        ) as executor:
            fill(executor)
            while inflight:
                if interrupted and not announced:
                    announced = True
                    display.console.print(
                        "  Interrupted: no new PDFs will start; already-running "
                        "conversions will finish safely."
                    )
                drain_pdf_operations()
                try:
                    finished, _ = wait(
                        tuple(inflight), timeout=0.05, return_when=FIRST_COMPLETED
                    )
                except KeyboardInterrupt:
                    stop_submission()
                    continue
                drain_events()
                for future in sorted(finished, key=lambda item: inflight[item]):
                    index = inflight.pop(future)
                    try:
                        result = future.result()
                    except KeyboardInterrupt:
                        stop_submission()
                        display.finish_for(index, error=RuntimeError("Interrupted by user"))
                    except Exception as exc:
                        display.finish_for(index, error=exc)
                    else:
                        if result not in (None, 0):
                            display.finish_for(
                                index, error=RuntimeError(f"Conversion exited with status {result}")
                            )
                        else:
                            display.finish_for(index)
                fill(executor)
            drain_events()
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    return interrupted


def convert_missing(
    *, papers_dir: Path = DEFAULT_PAPERS_DIR, limit: int | None = None,
    chunk: bool | None = None, standard: bool = False, dry_run: bool = False,
    parallel: int = 3,
) -> int:
    from rich.console import Console
    from rich.text import Text

    from .presentation import AMBER, MUTED

    if not isinstance(parallel, int) or isinstance(parallel, bool) or not 1 <= parallel <= 16:
        raise ValueError("--parallel must be an integer between 1 and 16")
    root = papers_dir.expanduser().resolve()
    missing, conflicts = discover_missing(root, limit=limit)
    console = Console()
    if not missing:
        console.print(Text(f"No PDFs without Markdown found in {root}.", style=MUTED))
        return 0
    label = Text(f"  {len(missing)} PDFs need transcription", style="bold white")
    if conflicts:
        label.append(f"   ·   {conflicts} Syncthing conflicts ignored", style=MUTED)
    console.print(label)
    console.print(Text(f"  Scanning {root}", style=MUTED, overflow="ellipsis", no_wrap=True))
    if any(not item.creation_available for item in missing):
        console.print(Text("  Filesystem has no creation dates for some PDFs; date keys use mtime.",
                           style=AMBER))

    # No API access until both the selector and explicit paid confirmation succeed.
    previous: set[Path] | None = None
    while True:
        if previous is None:
            chosen = interactive_selection(missing, root=root)
        else:
            chosen = interactive_selection(missing, root=root, initial_selected=previous)
        if chosen is None:
            console.print(Text("  Cancelled · No PDFs converted.", style=MUTED))
            return 0
        if not chosen:
            console.print(Text("  Nothing selected · No PDFs converted.", style=MUTED))
            return 0
        console.print(Text(f"  Concurrency: up to {min(parallel, len(chosen))} PDF(s) at once", style=MUTED))
        decision = confirm_paid_conversion(chosen, standard=standard, dry_run=dry_run)
        if decision == "back":
            previous = set(chosen)
            continue
        if dry_run:
            return 0
        if decision is not True:
            console.print(Text("  Not confirmed · No PDFs converted.", style=MUTED))
            return 0
        break

    display = BatchDisplay(chosen, standard=standard, root=root, console=console)
    interrupted = False
    with quiet_terminal_input(), display:
        if parallel == 1 or len(chosen) == 1:
            # Keep the original single-worker semantics, including prompt
            # Ctrl-C interruption and exactly one event sink.
            for index, pdf in enumerate(chosen):
                display.begin(index)
                try:
                    go(pdf, standard=standard, chunk=chunk, on_event=display.event)
                except KeyboardInterrupt:
                    display.finish(error=RuntimeError("Interrupted by user"))
                    interrupted = True
                    break
                except Exception as exc:
                    display.finish(error=exc)
                else:
                    display.finish()
        else:
            interrupted = _parallel_batch(
                chosen, display, workers=min(parallel, len(chosen)),
                standard=standard, chunk=chunk,
            )
    display.summary(interrupted=interrupted)
    return 1 if interrupted or any(state.status == "failed" for state in display.states) else 0


def _parallel_count(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--parallel must be an integer between 1 and 16") from exc
    if not 1 <= result <= 16:
        raise argparse.ArgumentTypeError("--parallel must be between 1 and 16")
    return result


def _add_chunk_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--chunk", dest="chunk", action="store_const", const=True, default=None,
        help=f"Force {CHUNK_PAGES}-page chunks, even for short PDFs",
    )
    group.add_argument(
        "--no-chunk", dest="chunk", action="store_const", const=False,
        help=f"Never chunk (default auto-chunks PDFs with >{AUTO_CHUNK_THRESHOLD} pages)",
    )



def _add_citation_backend_flags(parser: argparse.ArgumentParser) -> None:
    """Provider/model selection belongs to citation analysis, not PDF transcription."""
    parser.add_argument(
        "--codex", action="store_true",
        help="Use installed Codex CLI for citations instead of Gemini",
    )
    parser.add_argument(
        "--model", metavar="MODEL",
        help="Citation model override (Gemini default: gemini-3.8-flash; "
             "Codex default: gpt-6-luna)",
    )


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    # Hidden developer commands live outside the public argparse subparser tree.
    if args_list and args_list[0] == "_dev":
        try:
            from .citation_service import dev_main
            return dev_main(args_list[1:])
        except (ValueError, OSError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    # Convenience alias for users accustomed to flags instead of subcommands.
    if args_list and args_list[0] == "--convert-missing":
        args_list[0] = "convert-missing"
    parser = argparse.ArgumentParser(prog="pdf-transcriber", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("go", help="Transcribe PDF and enrich citations")
    command.add_argument("pdf", type=Path, help="Path to source PDF")
    command.add_argument("--standard", action="store_true", help="Use Standard instead of default Flex")
    command.add_argument("--resume", type=Path, metavar="DIR", help="Recover an interrupted run")
    _add_chunk_flags(command)
    _add_citation_backend_flags(command)

    studio = sub.add_parser("aistudio", help="Manual AI Studio transcription and JSON import (no API)")
    studio.add_argument("pdf", type=Path, help="Original PDF")
    studio.add_argument("json", nargs="?", type=Path, help="Previously saved AI Studio JSON")
    studio.add_argument("--prompt-only", action="store_true", help="Print transcription prompt and exit")

    transcription = sub.add_parser("transcribe", help="Transcribe a PDF without citation enrichment")
    transcription.add_argument("pdf", type=Path, help="Path to source PDF")
    transcription.add_argument("--from-json", type=Path, metavar="FILE",
                               help="Import an AI Studio response locally; no API call")
    transcription.add_argument("--standard", action="store_true", help="Use Standard instead of default Flex")
    transcription.add_argument("--resume", type=Path, metavar="DIR", help="Recover interrupted PDF conversion")
    _add_chunk_flags(transcription)

    citation = sub.add_parser("find-citations", help="Find and link references in existing Markdown")
    citation.add_argument("source", type=Path, help="Path to original .md or corresponding .pdf")
    citation.add_argument("--standard", action="store_true", help="Use Standard instead of default Flex")
    citation.add_argument("--resume", type=Path, metavar="DIR", help="Recover interrupted citation analysis")
    citation.add_argument("--force", action="store_true", help="Replace generated citation outputs")
    _add_citation_backend_flags(citation)

    missing = sub.add_parser(
        "convert-missing", help="Interactively choose PDFs missing Markdown (none selected by default)"
    )
    missing.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS_DIR,
                         help="Root directory to scan (default: ~/syncthing/papers)")
    missing.add_argument("-N", type=int, default=None, help="Show only N newest missing PDFs")
    missing.add_argument("--standard", action="store_true", help="Use Standard instead of default Flex")
    missing.add_argument("--dry-run", action="store_true", help="Preview selection without any API calls")
    missing.add_argument(
        "-j", "--parallel", type=_parallel_count, default=3, metavar="N",
        help="Run up to N PDFs concurrently (default: 3 for Gemini Tier 1; 1-16)",
    )
    _add_chunk_flags(missing)
    parsed = parser.parse_args(args_list)
    try:
        if parsed.command == "aistudio":
            from .aistudio import aistudio
            if parsed.prompt_only and parsed.json is not None:
                parser.error("aistudio --prompt-only cannot be combined with a JSON file")
            try:
                return aistudio(parsed.pdf, parsed.json, prompt_only=parsed.prompt_only)
            except (KeyboardInterrupt, EOFError):
                print("Cancelled AI Studio transcription.", file=sys.stderr)
                return 130
        if parsed.command == "transcribe" and parsed.from_json is not None:
            from .aistudio import import_json, _report
            if parsed.resume is not None or parsed.standard or parsed.chunk is not None:
                parser.error("transcribe --from-json cannot be used with --resume, --standard or chunk flags")
            _report(*import_json(parsed.pdf, parsed.from_json))
            return 0
        if parsed.command == "find-citations":
            from .citation_service import find_citations
            return find_citations(
                parsed.source, standard=parsed.standard,
                resume=parsed.resume, force=parsed.force,
                codex=parsed.codex, model=parsed.model,
            )
        if parsed.command in ("go", "transcribe"):
            source = parsed.pdf.expanduser().resolve()
            if source.suffix.lower() != ".pdf" or not source.is_file():
                raise ValueError(f"Expected an existing PDF: {source}")
            markdown = source.with_suffix(".md")
            combined = parsed.command == "go"
            if combined:
                from .citation_service import find_citations
                from .citations import output_paths
                existing = output_paths(markdown)
                if existing["legacy_linked"].exists():
                    raise FileExistsError(
                        f"Legacy .cited.md file exists: {existing['legacy_linked']}. "
                        "Move or remove it before using the in-place citation workflow."
                    )
                if existing["metadata"].exists():
                    raise FileExistsError(
                        "Citation metadata already exists. Restore the original unlinked "
                        "Markdown before regenerating citations."
                    )
                # Fail before a potentially paid PDF transcription if Codex
                # is not installed. No Codex execution or login check yet.
                if parsed.codex:
                    from .citation_service import require_codex
                    require_codex()
                if parsed.resume is not None and markdown.is_file():
                    raise ValueError(
                        "Markdown already exists; use find-citations --resume for citation recovery."
                    )
            # Reuse finished transcription when only citation linking failed.
            if parsed.command == "transcribe" or not markdown.is_file():
                if sys.stdout.isatty() and sys.stdin.isatty():
                    from rich.console import Console
                    display = BatchDisplay([source], standard=parsed.standard,
                                           root=source.parent, console=Console())
                    failed = False
                    interrupted = False
                    with quiet_terminal_input(), display:
                        display.begin(0)
                        try:
                            go(source, standard=parsed.standard, resume=parsed.resume,
                               chunk=parsed.chunk, on_event=display.event)
                        except KeyboardInterrupt:
                            display.finish(error=RuntimeError("Interrupted by user"))
                            failed = interrupted = True
                        except Exception as exc:
                            display.finish(error=exc)
                            failed = True
                        else:
                            display.finish()
                    display.summary(interrupted=interrupted)
                    if failed:
                        return 1
                else:
                    status = go(source, standard=parsed.standard, resume=parsed.resume,
                                chunk=parsed.chunk)
                    if status:
                        return status
            else:
                emit("cached", f"Reusing existing Markdown: {markdown.name}")
            if combined:
                return find_citations(
                    markdown,
                    standard=parsed.standard if not parsed.codex else False,
                    codex=parsed.codex, model=parsed.model,
                )
            return 0
        if parsed.command == "convert-missing":
            return convert_missing(
                papers_dir=parsed.papers_dir, limit=parsed.N, standard=parsed.standard,
                chunk=parsed.chunk, dry_run=parsed.dry_run,
                parallel=parsed.parallel,
            )
    except (ValueError, OSError, RuntimeError, TimeoutError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
