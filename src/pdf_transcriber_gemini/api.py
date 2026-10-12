"""Gemini Developer API upload + structured transcription (Interactions API, SDK 2.x)."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

from .reporting import emit

MODEL = "gemini-3.8-flash"  # stable, September 2026; do not use a -latest alias


def api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) before running `go`.")
    return key


def file_state(file: object) -> str:
    state = getattr(file, "state", None)
    return str(getattr(state, "value", state) or "ACTIVE").split(".")[-1].upper()


def _retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    try:
        return int(code) in (408, 429, 500, 502, 503, 504)
    except (TypeError, ValueError):
        return False


def _uploaded_file(client: object, pdf: Path, workspace: Path) -> object:
    """Reuse a still-active uploaded PDF on resume; otherwise upload once."""
    record = workspace / "uploaded.json"
    if record.exists():
        info = json.loads(record.read_text(encoding="utf-8"))
        name = info.get("name")
        if name:
            try:
                uploaded = client.files.get(name=name)
                if file_state(uploaded) != "FAILED" and getattr(uploaded, "uri", None):
                    emit("upload_reuse", f"Reusing uploaded PDF: {name}")
                    return uploaded
            except Exception:
                emit("upload_retry", "Previous upload unavailable; uploading the PDF again.")

    emit("uploading", f"Uploading PDF: {pdf.name}")
    # Passing a pathname makes google-genai send its basename in the
    # X-Goog-Upload-File-Name HTTP header, which fails for Unicode characters
    # such as en dashes. A binary stream avoids the header and is supported by
    # the SDK's resumable Files API; set MIME type explicitly for streams.
    with pdf.open("rb") as stream:
        uploaded = client.files.upload(file=stream, config={"mime_type": "application/pdf"})
    if not getattr(uploaded, "name", None):
        raise RuntimeError("File API did not return a file name")
    record.write_text(
        json.dumps({"name": uploaded.name, "uri": uploaded.uri, "mime_type": uploaded.mime_type}, indent=2),
        encoding="utf-8",
    )
    emit("uploaded", "PDF uploaded to Gemini Files API")
    return uploaded


def _wait_active(client: object, uploaded: object) -> object:
    until = time.monotonic() + 300
    while file_state(uploaded) == "PROCESSING":
        if time.monotonic() >= until:
            raise TimeoutError("Uploaded PDF was not ready within 5 minutes")
        time.sleep(3)
        uploaded = client.files.get(name=uploaded.name)
    if file_state(uploaded) == "FAILED":
        raise RuntimeError("Gemini File API could not process the uploaded PDF")
    if file_state(uploaded) != "ACTIVE":
        raise RuntimeError(f"Unexpected Gemini File API state: {file_state(uploaded)}")
    if not getattr(uploaded, "uri", None):
        raise RuntimeError("Gemini File API did not return a usable URI")
    return uploaded


def transcribe(pdf: Path, workspace: Path, *, standard: bool, prompt: str, schema: dict) -> Path:
    """Save the raw JSON response before any processing, for failure recovery."""
    from google import genai  # load lazily; local assembly/tests need no API access

    tier = "standard" if standard else "flex"
    client = genai.Client(api_key=api_key(), vertexai=False)
    try:
        uploaded = _wait_active(client, _uploaded_file(client, pdf, workspace))
        # New Interactions API response_format (not legacy GenerateContentConfig).
        request = dict(
            model=MODEL,
            input=[
                {"type": "text", "text": prompt},
                {"type": "document", "uri": uploaded.uri, "mime_type": "application/pdf"},
            ],
            response_format=[{"type": "text", "mime_type": "application/json", "schema": schema}],
            service_tier=tier,
            store=False,
            timeout=1800 if tier == "flex" else 900,  # seconds; Flex requests can sit in the queue
        )
        for attempt in range(3):
            try:
                emit("model_start", f"Transcribing via {MODEL} ({tier}; attempt {attempt + 1}/3)...")
                interaction = client.interactions.create(**request)
                break
            except Exception as exc:
                if attempt == 2 or not _retryable(exc):
                    raise
                delay = min(60.0, 5 * 2 ** attempt) + random.uniform(0, 1)
                emit("retry", f"Temporary API failure ({type(exc).__name__}); retrying in {delay:.0f}s.")
                time.sleep(delay)
        result = workspace / "response.json"
        raw = getattr(interaction, "output_text", None)
        metadata = {
            "model": MODEL,
            "requested_service_tier": tier,
            "interaction_id": getattr(interaction, "id", None),
            "status": str(getattr(interaction, "status", "unknown")),
        }
        (workspace / "interaction.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if not isinstance(raw, str) or not raw.strip():
            raise RuntimeError("Gemini returned no text; see interaction.json")
        status = str(getattr(interaction, "status", "completed")).lower()
        if not status.endswith("completed"):
            (workspace / "partial-response.txt").write_text(raw, encoding="utf-8")
            raise RuntimeError(
                f"Gemini did not complete the response (status={status}); "
                "partial-response.txt saved for inspection"
            )
        result.write_text(raw, encoding="utf-8")
        emit("model_done", "Gemini transcription saved")
        return result
    finally:
        client.close()


def delete_uploaded(workspace: Path) -> None:
    """Best-effort remote cleanup on success only; failed runs keep 48h upload."""
    record = workspace / "uploaded.json"
    if not record.exists():
        return
    name = json.loads(record.read_text(encoding="utf-8")).get("name")
    if not name:
        return
    try:
        from google import genai
        client = genai.Client(api_key=api_key(), vertexai=False)
        try:
            client.files.delete(name=name)
        finally:
            client.close()
    except Exception as exc:
        emit("warning", f"Could not remove Gemini upload ({type(exc).__name__}); Google expires it after 48 hours.")
