"""Manual Google AI Studio transcription: prompt handoff and local JSON assembly."""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import webbrowser
from importlib.resources import files
from pathlib import Path

from .assembler import figure_folder_name, load_gemini_json, publish, validate_figures

AI_STUDIO_URL = "https://aistudio.google.com/prompts/new_chat"


def _pdf(path: Path) -> Path:
    pdf = path.expanduser().resolve(strict=True)
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        raise ValueError("Provide an existing PDF")
    return pdf


def _outputs_available(pdf: Path) -> None:
    if pdf.with_suffix(".md").exists():
        raise FileExistsError(f"{pdf.with_suffix('.md')} already exists; refusing to overwrite")
    figure_dir = pdf.parent / figure_folder_name(pdf.stem)
    if figure_dir.exists():
        raise FileExistsError(f"Figure directory already exists: {figure_dir}")


def transcription_prompt() -> str:
    return files("pdf_transcriber_gemini").joinpath("prompt.txt").read_text(encoding="utf-8")


def import_json(pdf_path: Path, response_path: Path) -> tuple[Path, Path | None, list[str]]:
    """Publish a saved AI Studio response locally; never contact the Gemini API."""
    import pymupdf

    pdf = _pdf(pdf_path)
    response = response_path.expanduser().resolve(strict=True)
    _outputs_available(pdf)
    data = load_gemini_json(response)
    with pymupdf.open(pdf) as document:
        if document.needs_pass or document.page_count < 1:
            raise ValueError("PDF is password-protected or empty")
        validate_figures(data, document.page_count)
    with tempfile.TemporaryDirectory(prefix=".pdf-transcriber-import-", dir=pdf.parent) as temp:
        markdown, figures = publish(pdf, data, Path(temp))
    return markdown, figures, data["problems"]


def _clipboard_write(value: str) -> bool:
    if sys.platform == "darwin" and shutil.which("pbcopy"):
        command = ["pbcopy"]
    elif shutil.which("wl-copy"):
        command = ["wl-copy"]
    elif shutil.which("xclip"):
        command = ["xclip", "-selection", "clipboard"]
    else:
        return False
    try:
        subprocess.run(command, input=value, text=True, check=True, timeout=10)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _clipboard_read() -> str | None:
    if sys.platform == "darwin" and shutil.which("pbpaste"):
        command = ["pbpaste"]
    elif shutil.which("wl-paste"):
        command = ["wl-paste", "--no-newline"]
    elif shutil.which("xclip"):
        command = ["xclip", "-selection", "clipboard", "-o"]
    else:
        return None
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
        return result.stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _mac_reveal_pdf(pdf: Path) -> None:
    if sys.platform == "darwin":
        try:
            subprocess.run(["open", "-R", str(pdf)], check=True, timeout=10)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"WARNING: Finder could not reveal the PDF: {exc}", file=sys.stderr)


def _report(markdown: Path, figures: Path | None, problems: list[str]) -> None:
    print(f"Wrote: {markdown}")
    print(f"Figures: {figures}/" if figures else "No substantive figures")
    for problem in problems:
        print(f"MODEL NOTE: {problem}")


def aistudio(pdf_path: Path, response: Path | None = None, *, prompt_only: bool = False) -> int:
    pdf = _pdf(pdf_path)
    if prompt_only:
        print(transcription_prompt(), end="")
        return 0

    _outputs_available(pdf)
    if response is not None:
        _report(*import_json(pdf, response))
        return 0

    if not sys.stdin.isatty():
        raise ValueError(
            "Interactive AI Studio mode needs a terminal. "
            "Use 'pdf-transcriber aistudio PDF RESPONSE.json' or '--prompt-only'."
        )

    prompt = transcription_prompt()
    if _clipboard_write(prompt):
        print("Transcription prompt copied to clipboard.")
    else:
        prompt_path = pdf.with_suffix(".aistudio-prompt.txt")
        if prompt_path.exists():
            raise FileExistsError(f"Prompt file exists: {prompt_path}; refusing to overwrite")
        prompt_path.write_text(prompt, encoding="utf-8")
        print(f"Clipboard unavailable. Prompt saved to: {prompt_path}")

    _mac_reveal_pdf(pdf)
    print(f"PDF: {pdf}")
    try:
        opened = webbrowser.open(AI_STUDIO_URL)
        if opened:
            print("Opened Google AI Studio in browser.")
        else:
            print(f"Open AI Studio: {AI_STUDIO_URL}")
    except (OSError, webbrowser.Error) as exc:
        print(f"Open AI Studio manually: {AI_STUDIO_URL} ({exc})")
    print("Upload the PDF, paste the prompt, run it, then copy the entire JSON response.")
    print("Press Enter to import clipboard JSON, or enter a JSON file path (drag-and-drop works).")
    while True:
        raw = input("JSON file (Enter = clipboard, Ctrl-C to cancel): ").strip()
        if raw:
            # Accept literal paths as well as shell-quoted drag-and-drop paths.
            candidate = Path(raw).expanduser()
            if not candidate.is_file():
                try:
                    parts = shlex.split(raw)
                except ValueError as exc:
                    print(f"Invalid path: {exc}")
                    continue
                if len(parts) != 1:
                    print("Please provide one JSON path.")
                    continue
                candidate = Path(parts[0]).expanduser()
            if not candidate.is_file():
                print(f"File not found: {candidate}")
                continue
            response_path = candidate
            cleanup = False
        else:
            clipboard = _clipboard_read()
            if not clipboard or not clipboard.strip():
                print("Clipboard is empty or unavailable. Save the response as a JSON file and enter its path.")
                continue
            # The shared assembler accepts raw or fenced JSON and validates the schema.
            handle, name = tempfile.mkstemp(prefix=".pdf-transcriber-clipboard-", suffix=".json", dir=pdf.parent)
            with os.fdopen(handle, "w", encoding="utf-8") as out:
                out.write(clipboard)
            response_path = Path(name)
            cleanup = True
        try:
            _report(*import_json(pdf, response_path))
            return 0
        except (ValueError, OSError) as exc:
            print(f"Could not import response: {exc}", file=sys.stderr)
            print("Correct the JSON and try again; no output was overwritten.")
        finally:
            if cleanup:
                response_path.unlink(missing_ok=True)
