"""Offline regression tests; no credentials, network requests, or PDF text transcription."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

from pdf_transcriber_gemini import api, cli
from pdf_transcriber_gemini.assembler import validate_figures


def make_pdf(path: Path, *, rotate: bool = False) -> None:
    pdf = pymupdf.open()
    page = pdf.new_page(width=300, height=400)
    page.draw_rect(pymupdf.Rect(60, 100, 210, 230), color=(0, 0, 0), fill=(0.2, 0.5, 0.9))
    if rotate:
        page.set_rotation(90)
    pdf.save(path)
    pdf.close()


def model_result(with_figure: bool = True) -> dict:
    data = {"markdown": "# Essay\n\nBody.\n", "figures": [], "problems": []}
    if with_figure:
        data["markdown"] += "\n<!-- FIGURE:fig_1 -->\n\nCaption.\n"
        data["figures"].append({
            "id": "fig_1", "pdf_page": 1, "box_2d": [250, 200, 575, 700],
            "description": "A blue rectangle",
        })
    return data


def test_figure_marker_validation() -> None:
    data = model_result()
    data["markdown"] = "# Essay\n"
    with pytest.raises(ValueError, match="placeholder mismatch"):
        validate_figures(data, 1)


def test_go_writes_files_and_cleans_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "Article title - smith2024_example.pdf"
    make_pdf(pdf)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    seen: dict = {}

    def fake_transcribe(file, work, *, standard, prompt, schema):
        seen.update({"standard": standard, "prompt": prompt, "schema": schema})
        response = work / "response.json"
        response.write_text(json.dumps(model_result()), encoding="utf-8")
        return response

    monkeypatch.setattr(cli, "transcribe", fake_transcribe)
    monkeypatch.setattr(cli, "delete_uploaded", lambda _: None)
    assert cli.main(["transcribe", str(pdf)]) == 0
    md = pdf.with_suffix(".md")
    assert md.exists()
    assert "![A blue rectangle]" in md.read_text(encoding="utf-8")
    assert (tmp_path / "smith2024_example" / "fig_1.png").is_file()
    assert list(tmp_path.glob(".pdf-transcriber-*")) == []
    assert seen["standard"] is False
    assert "TRANSCRIPTION REQUIREMENT" in seen["prompt"]
    assert seen["schema"]["required"] == ["markdown", "figures", "problems"]


def test_go_standard_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "a.pdf"
    make_pdf(pdf)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    tiers: list[bool] = []

    def fake_transcribe(file, work, *, standard, prompt, schema):
        tiers.append(standard)
        response = work / "response.json"
        response.write_text(json.dumps(model_result(False)))
        return response

    monkeypatch.setattr(cli, "transcribe", fake_transcribe)
    monkeypatch.setattr(cli, "delete_uploaded", lambda _: None)
    assert cli.main(["transcribe", "--standard", str(pdf)]) == 0
    assert tiers == [True]
    assert pdf.with_suffix(".md").exists()
    assert not (tmp_path / "a").exists()


def test_failure_preserves_response_and_resume_without_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "a - key2026.pdf"
    make_pdf(pdf)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")

    def wrong(file, work, *, standard, prompt, schema):
        response = work / "response.json"
        response.write_text("not JSON")
        return response

    monkeypatch.setattr(cli, "transcribe", wrong)
    monkeypatch.setattr(cli, "delete_uploaded", lambda _: None)
    assert cli.main(["transcribe", str(pdf)]) == 1
    works = list(tmp_path.glob(".pdf-transcriber-*"))
    assert len(works) == 1 and (works[0] / "response.json").exists()
    assert not pdf.with_suffix(".md").exists()
    (works[0] / "response.json").write_text(json.dumps(model_result(False)))
    monkeypatch.setattr(cli, "transcribe", lambda *a, **k: pytest.fail("API called on resume"))
    assert cli.main(["transcribe", str(pdf), "--resume", str(works[0])]) == 0
    assert pdf.with_suffix(".md").exists()
    assert not works[0].exists()


def test_do_not_pay_when_markdown_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "a.pdf"
    make_pdf(pdf)
    pdf.with_suffix(".md").write_text("Existing")
    monkeypatch.setattr(cli, "transcribe", lambda *a, **k: pytest.fail("API was called"))
    assert cli.main(["transcribe", str(pdf)]) == 1
    assert pdf.with_suffix(".md").read_text() == "Existing"


def test_api_uses_file_uri_schema_and_flex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "document.pdf"
    make_pdf(pdf)
    monkeypatch.setenv("GEMINI_API_KEY", "not-real")
    calls = []
    upload = SimpleNamespace(name="files/example", uri="https://generativelanguage.googleapis.com/file/example", mime_type="application/pdf", state="PROCESSING")
    active = SimpleNamespace(**{**vars(upload), "state": "ACTIVE"})

    class FakeFiles:
        def upload(self, **kwargs):
            calls.append(("upload", kwargs))
            return upload

        def get(self, **kwargs):
            calls.append(("get", kwargs))
            return active

    class FakeInteractions:
        def create(self, **kwargs):
            calls.append(("create", kwargs))
            return SimpleNamespace(output_text=json.dumps(model_result(False)), status="completed", id="test-id")

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(("client", kwargs))
            self.files = FakeFiles()
            self.interactions = FakeInteractions()

        def close(self):
            calls.append(("close", None))

    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=SimpleNamespace(Client=FakeClient)))
    monkeypatch.setattr(api.time, "sleep", lambda seconds: None)
    schema = {"type": "object", "properties": {}}
    result_path = api.transcribe(pdf, tmp_path, standard=False, prompt="Prompt", schema=schema)
    assert json.loads(result_path.read_text())["markdown"].startswith("# Essay")
    request = [v for k, v in calls if k == "create"][0]
    assert request["model"] == "gemini-3.8-flash"
    assert request["service_tier"] == "flex"
    assert request["response_format"] == [{"type": "text", "mime_type": "application/json", "schema": schema}]
    assert request["input"][1]["type"] == "document"
    assert request["input"][1]["uri"] == upload.uri
    assert "data" not in request["input"][1]
    assert (tmp_path / "uploaded.json").exists()
    assert (tmp_path / "interaction.json").exists()


def test_unicode_pdf_filename_uploads_as_binary_stream(tmp_path: Path) -> None:
    """The SDK must not receive a path (it would use a non-ASCII HTTP header)."""
    pdf = tmp_path / "Pomp – Action Segmentation in the Brain–Object–Action pomp2024_action.pdf"
    make_pdf(pdf)
    seen_stream = None

    class FakeFiles:
        def upload(self, *, file, config):
            nonlocal seen_stream
            seen_stream = file
            assert isinstance(file, io.IOBase)
            assert file.read(5) == b"%PDF-"
            assert config == {"mime_type": "application/pdf"}
            return SimpleNamespace(
                name="files/unicode-pdf", uri="file://uploaded", mime_type="application/pdf"
            )

    uploaded = api._uploaded_file(SimpleNamespace(files=FakeFiles()), pdf, tmp_path)
    assert uploaded.name == "files/unicode-pdf"
    assert seen_stream is not None and seen_stream.closed
    assert json.loads((tmp_path / "uploaded.json").read_text(encoding="utf-8"))["name"] == uploaded.name


def test_resumed_pdf_cannot_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "a.pdf"
    make_pdf(pdf)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    works = tmp_path / ".pdf-transcriber-abc"
    works.mkdir()
    (works / "source.json").write_text(json.dumps({"pdf": str(pdf), "sha256": "wrong"}))
    assert cli.main(["transcribe", str(pdf), "--resume", str(works)]) == 1
    assert works.exists()
