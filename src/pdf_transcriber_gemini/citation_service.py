"""API orchestration and hidden developer tools for citation enrichment."""
from __future__ import annotations

import argparse
import json
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from importlib.resources import files
from pathlib import Path

from .api import MODEL, _retryable, api_key
from .citations import (
    CitationError,
    apply,
    has_generated_citation_markup,
    output_paths,
    read_markdown,
    render_request,
)


def resolve_markdown(path: Path) -> Path:
    """A PDF filename denotes its already-transcribed same-stem Markdown file."""
    p = path.expanduser().resolve()
    if p.suffix.lower() == ".pdf":
        p = p.with_suffix(".md")
    if p.suffix.lower() != ".md" or p.name.endswith(".cited.md"):
        raise CitationError("Provide an original .md transcription (or its .pdf name), not .cited.md")
    if not p.is_file():
        raise CitationError(f"Markdown transcription not found: {p}")
    return p


CODEX_DEFAULT_MODEL = "gpt-6-luna"


def require_codex() -> str:
    """Resolve Codex CLI without spawning a model or requiring a Gemini API key."""
    executable = shutil.which("codex")
    if not executable:
        raise RuntimeError(
            "Codex CLI is not installed or is not on PATH. Install it with "
            "`npm install -g @openai/codex` or `brew install --cask codex`, "
            "then run `codex login`."
        )
    return executable


def _gemini_citations(
    request: str, workspace: Path, *, standard: bool, model: str = MODEL,
) -> Path:
    from google import genai

    tier = "standard" if standard else "flex"
    schema = json.loads(
        files("pdf_transcriber_gemini").joinpath("citations_schema.json").read_text(encoding="utf-8")
    )
    client = genai.Client(api_key=api_key(), vertexai=False)
    try:
        config = dict(
            model=model,
            input=[{"type": "text", "text": request}],
            response_format=[{"type": "text", "mime_type": "application/json", "schema": schema}],
            service_tier=tier,
            store=False,
            timeout=1800 if tier == "flex" else 900, # seconds
        )
        for attempt in range(3):
            try:
                print(f"Finding citations using {model} ({tier}, attempt {attempt+1}/3)...", flush=True)
                answer = client.interactions.create(**config)
                break
            except Exception as exc:
                if attempt >= 2 or not _retryable(exc):
                    raise
                delay = min(60, 5 * (2 ** attempt)) + random.uniform(0, 1)
                print(f"Temporary Gemini failure; retrying in {delay:.0f}s.", flush=True)
                time.sleep(delay)
        status = str(getattr(answer, "status", "completed")).lower()
        raw = getattr(answer, "output_text", None)
        (workspace / "interaction.json").write_text(
            json.dumps({
                "model": model,
                "requested_service_tier": tier,
                "interaction_id": getattr(answer, "id", None),
                "status": status,
            }, indent=2), encoding="utf-8"
        )
        if not isinstance(raw, str) or not raw.strip():
            raise RuntimeError("Gemini returned no JSON; see interaction.json")
        if not status.endswith("completed"):
            (workspace / "partial-response.txt").write_text(raw, encoding="utf-8")
            raise RuntimeError(f"Gemini citation response incomplete ({status}); partial text saved")
        result = workspace / "response.json"
        result.write_text(raw, encoding="utf-8")
        return result
    finally:
        client.close()


def _codex_citations(request: str, workspace: Path, *, model: str) -> Path:
    """Run the installed Codex CLI non-interactively in a read-only sandbox.

    The compact prompt intentionally allows missing bibliography fields. Codex's
    strict --output-schema mode would force nullable values for every optional
    field, adding tokens; instead validate the final JSON with our existing
    local citation parser, the same one used for Gemini.
    """
    from .citations import read_response

    executable = require_codex()
    # Unlike Gemini's Interactions API, Codex is not given a response_format
    # schema by this wrapper. Include the existing compact schema in the prompt
    # and validate the returned JSON locally rather than forcing every optional
    # bibliographic field to appear through strict structured output.
    schema = files("pdf_transcriber_gemini").joinpath("citations_schema.json").read_text(
        encoding="utf-8"
    )
    codex_request = (
        "OUTPUT JSON SHAPE FOR THIS NON-INTERACTIVE RUN (omit unavailable metadata):\n"
        + schema + "\n\n" + request
    )
    final = workspace / "codex-final-message.txt"
    command = [
        executable, "exec", "--ephemeral", "--skip-git-repo-check",
        "--sandbox", "read-only", "--model", model,
        "--output-last-message", str(final), "-",
    ]
    print(f"Finding citations using Codex CLI ({model})...", flush=True)
    try:
        completed = subprocess.run(
            command, input=codex_request, cwd=workspace, text=True, encoding="utf-8",
            capture_output=True, timeout=1800, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            "Codex citation analysis timed out after 30 minutes; inspect the recovery folder"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Could not launch Codex CLI: {exc}") from exc

    (workspace / "codex-stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (workspace / "codex-stderr.txt").write_text(completed.stderr, encoding="utf-8")
    (workspace / "interaction.json").write_text(
        json.dumps({"provider": "codex", "model": model, "returncode": completed.returncode,
                    "command": ["codex", *command[1:-1], "-"]}, indent=2), encoding="utf-8"
    )
    if completed.returncode != 0:
        tail = completed.stderr.strip().splitlines()[-1:] or completed.stdout.strip().splitlines()[-1:]
        detail = (" " + tail[0][:250]) if tail else ""
        raise RuntimeError(
            f"Codex CLI exited with status {completed.returncode}.{detail} "
            "See codex-stderr.txt in the recovery folder. "
            "Ensure `codex login` has been completed and the model is available."
        )
    if not final.is_file() or not final.read_text(encoding="utf-8").strip():
        raise RuntimeError("Codex returned no final message; see codex-stdout.txt and codex-stderr.txt")
    try:
        read_response(final)  # validate the response before caching it as successful
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Codex did not produce valid citation JSON ({exc}); "
            "inspect codex-final-message.txt in the recovery folder"
        ) from exc
    response = workspace / "response.json"
    response.write_text(final.read_text(encoding="utf-8"), encoding="utf-8")
    return response


def find_citations(
    source: Path, *, standard: bool = False, resume: Path | None = None, force: bool = False,
    codex: bool = False, model: str | None = None,
) -> int:
    markdown = resolve_markdown(source)
    if model is not None and not model.strip():
        raise ValueError("--model must be a nonempty model ID")
    if codex and standard:
        raise ValueError("--standard applies only to Gemini; omit it with --codex")
    source_text, checksum, sid = read_markdown(markdown)
    outputs = output_paths(markdown)
    if outputs["legacy_linked"].exists():
        raise FileExistsError(
            f"Legacy linked file {outputs['legacy_linked'].name} exists. "
            "Move or remove it before running in-place enrichment; it is not deleted automatically."
        )
    if has_generated_citation_markup(source_text):
        raise CitationError(
            "Markdown already contains generated citation links. Restore the unlinked "
            "source version before rerunning; --force does not remove citation IDs."
        )
    if not force and outputs["failures"].exists():
        raise FileExistsError(
            f"Citation failure log {outputs['failures'].name} already exists; "
            "use --force only with the original unlinked Markdown."
        )
    if not force and outputs["metadata"].exists():
        raise FileExistsError(
            f"Citation metadata already exists for {markdown.name}; "
            "use --force to replace metadata only with the original unlinked Markdown."
        )
    if resume is not None:
        workspace = resume.expanduser().resolve(strict=True)
        if not workspace.is_dir() or workspace.parent != markdown.parent or not workspace.name.startswith(".pdf-citations-"):
            raise CitationError("--resume must be a saved .pdf-citations-* folder beside the Markdown")
        metadata = json.loads((workspace / "source.json").read_text(encoding="utf-8"))
        if metadata.get("source") != str(markdown) or metadata.get("sha256") != checksum:
            raise CitationError("Recovery folder doesn't match this exact Markdown source")
        # Older recovery folders lack provider/model; they are Gemini runs.
        provider = metadata.get("provider", "gemini")
        chosen_model = metadata.get("model") or (CODEX_DEFAULT_MODEL if provider == "codex" else MODEL)
        if codex and provider != "codex":
            raise CitationError("Recovery folder belongs to Gemini, not Codex")
        if model is not None and model != chosen_model:
            raise CitationError(f"Recovery folder uses model {chosen_model!r}, not {model!r}")
    else:
        provider = "codex" if codex else "gemini"
        chosen_model = model or (CODEX_DEFAULT_MODEL if codex else MODEL)
        if codex:
            require_codex()  # fail before making a workspace; no Gemini credentials required
        else:
            api_key()  # existing Gemini credential preflight
        workspace = Path(tempfile.mkdtemp(prefix=".pdf-citations-", dir=markdown.parent))
        (workspace / "source.json").write_text(
            json.dumps({"source": str(markdown), "sha256": checksum,
                        "source_id": sid, "provider": provider, "model": chosen_model,
                        "requested_tier": ("standard" if standard else "flex") if provider == "gemini" else None}, indent=2),
            encoding="utf-8",
        )
    print(f"Citation working folder: {workspace}", flush=True)
    try:
        response = workspace / "response.json"
        if not response.is_file():
            request = render_request(markdown)
            (workspace / "request.txt").write_text(request, encoding="utf-8")
            if provider == "codex":
                response = _codex_citations(request, workspace, model=chosen_model)
            elif chosen_model == MODEL:
                response = _gemini_citations(request, workspace, standard=standard)
            else:
                response = _gemini_citations(
                    request, workspace, standard=standard, model=chosen_model
                )
        else:
            print("Reusing saved citation response (no model call).", flush=True)
        # Verify source again after the possibly long API request, before any edit.
        if read_markdown(markdown)[1] != checksum:
            raise CitationError("Source changed during model request; no files published")
        result = apply(
            markdown, json_path=response, force=force,
            generation={"provider": provider, "model": chosen_model},
        )
        print(f"Wrote {outputs['linked'].name} and {outputs['metadata'].name}")
        summary = result["summary"]
        print(f"Linked {summary['mentions_linked']} citations; "
              f"marked {summary['mentions_marked_without_link']} unresolved; "
              f"anchored {summary['references_anchored']}/{summary['references_found']} references")
        for warning in result["warnings"]:
            print(f"WARNING: {warning}", file=sys.stderr)
        shutil.rmtree(workspace)
        return 0
    except BaseException:
        (workspace / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("RECOVERY: citation work kept at " + str(workspace), file=sys.stderr)
        print("To resume: pdf-transcriber find-citations " + shlex.quote(str(markdown))
              + " --resume " + shlex.quote(str(workspace)), file=sys.stderr)
        raise


def dev_main(argv: list[str]) -> int:
    """Hidden development namespace; omitted entirely from top-level --help."""
    parser = argparse.ArgumentParser(prog="pdf-transcriber _dev")
    action = parser.add_subparsers(dest="action", required=True)
    prompt = action.add_parser("prompt", help="Print a complete prompt to standard output")
    prompt.add_argument("task", choices=["transcribe", "find-citations"])
    prompt.add_argument("source", type=Path)
    apply_cmd = action.add_parser("apply", help="Apply previously saved model proposals")
    apply_cmd.add_argument("task", choices=["find-citations"])
    apply_cmd.add_argument("source", type=Path)
    apply_cmd.add_argument("--json", type=Path, help="Model response JSON; default: <stem>.citation-proposals.json")
    apply_cmd.add_argument("--force", action="store_true", help="Overwrite existing .citations.json (Markdown is updated in place)")
    args = parser.parse_args(argv)
    if args.action == "prompt":
        if args.task == "transcribe":
            if args.source.suffix.lower() != ".pdf" or not args.source.expanduser().is_file():
                raise ValueError("Transcription prompt requires an existing PDF")
            print(files("pdf_transcriber_gemini").joinpath("prompt.txt").read_text(encoding="utf-8"), end="")
        else:
            print(render_request(resolve_markdown(args.source)), end="")
        return 0
    markdown = resolve_markdown(args.source)
    result = apply(markdown, json_path=args.json, force=args.force)
    print(f"Wrote {output_paths(markdown)['linked']} and {output_paths(markdown)['metadata']}")
    print(json.dumps(result["summary"], indent=2))
    for w in result["warnings"]:
        print(f"WARNING: {w}", file=sys.stderr)
    return 0
