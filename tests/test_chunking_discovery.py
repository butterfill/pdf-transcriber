"""Chunk joining, resumability, time shortcuts and safe bulk selection (offline)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pymupdf
import pytest

from pdf_transcriber_gemini import cli, discovery
from pdf_transcriber_gemini.chunking import Chunk, chunks_for, combine_chunks, write_chunk
from pdf_transcriber_gemini.discovery import MissingPdf, apply_shortcut, discover_missing


def make_pdf(path: Path, pages: int = 1) -> None:
    with pymupdf.open() as doc:
        for i in range(pages):
            p = doc.new_page(width=200, height=280)
            p.draw_rect(pymupdf.Rect(20 + (i % 5), 40, 160, 180), fill=(0.3, 0.5, 0.9))
        doc.save(path)


def result(markdown: str, figures: list | None = None) -> dict:
    return {"markdown": markdown, "figures": figures or [], "problems": []}


def fig(fid: str, page: int = 1) -> dict:
    return {"id": fid, "pdf_page": page, "box_2d": [100, 100, 750, 820],
            "description": "A blue rectangle"}


def test_chunk_specs_and_real_pdf_splits(tmp_path: Path) -> None:
    assert [(c.start, c.end) for c in chunks_for(31)] == [(1, 10), (11, 20), (21, 30), (31, 31)]
    source = tmp_path / "Source–name.pdf"
    make_pdf(source, 11)
    part = write_chunk(source, Chunk(2, 11, 11), tmp_path / "workspace")
    assert part.name == "part-0002.pdf"
    with pymupdf.open(part) as doc:
        assert doc.page_count == 1
    assert part == write_chunk(source, Chunk(2, 11, 11), tmp_path / "workspace")


def test_join_offsets_figure_ids_footnotes_and_continuation() -> None:
    results = [
        (Chunk(1, 1, 10), result(
            "# Paper\n\nFirst paragraph.\n\n[^1]: one\n\n"
            "<!-- FIGURE:fig_1 -->\n\nThe compound word" + "\n" + "<!-- CHUNK_END_MID_PARAGRAPH -->",
            [fig("fig_1", 10)])),
        (Chunk(2, 11, 20), result(
            "<!-- CHUNK_START_MID_PARAGRAPH -->\n\n"
            "<!-- pdf page 1 | article page 200 -->\n\ncontinued here."
            "\n\n[^1]: duplicate number, different footnote\n\n"
            "<!-- FIGURE:fig_1 -->",
            [fig("fig_1", 2)])),
    ]
    final = combine_chunks(results)
    assert "<!-- pdf page 11 | article page 200 --> continued here." in final["markdown"]
    assert "The compound word <!-- pdf page 11" in final["markdown"]
    assert "<!-- FIGURE:fig_1 -->" in final["markdown"]
    assert "<!-- FIGURE:fig_2 -->" in final["markdown"]
    assert [f["pdf_page"] for f in final["figures"]] == [10, 12]
    assert "[^1]: one" in final["markdown"]
    assert "[^2]: duplicate number" in final["markdown"]
    assert final["problems"] == []


def test_chunk_page_comment_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="not within"):
        combine_chunks([(Chunk(3, 21, 30), result("<!-- pdf page 21 --> body"))])


def test_auto_chunk_threshold_and_explicit_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    seen: list[int] = []
    monkeypatch.setattr(cli, "delete_uploaded", lambda path: None)

    def fake_transcribe(file: Path, work: Path, *, standard: bool, prompt: str, schema: dict):
        with pymupdf.open(file) as doc:
            seen.append(doc.page_count)
        (work / "response.json").write_text(json.dumps(result("# Part\n\nBody.")), encoding="utf-8")
        return work / "response.json"

    monkeypatch.setattr(cli, "transcribe", fake_transcribe)
    p30 = tmp_path / "thirty.pdf"
    make_pdf(p30, 30)
    assert cli.main(["transcribe", str(p30)]) == 0
    assert seen == [30]
    seen.clear()
    p31 = tmp_path / "thirty-one.pdf"
    make_pdf(p31, 31)
    assert cli.main(["transcribe", str(p31)]) == 0
    assert seen == [10, 10, 10, 1]
    seen.clear()
    pshort = tmp_path / "short.pdf"
    make_pdf(pshort, 12)
    assert cli.main(["transcribe", str(pshort), "--chunk"]) == 0
    assert seen == [10, 2]
    seen.clear()
    pnolong = tmp_path / "no-chunk.pdf"
    make_pdf(pnolong, 31)
    assert cli.main(["transcribe", str(pnolong), "--no-chunk"]) == 0
    assert seen == [31]
    assert not list(tmp_path.glob(".pdf-transcriber-*"))


def test_failed_chunk_resume_only_retries_missing_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "a.pdf"
    make_pdf(pdf, 21)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setattr(cli, "delete_uploaded", lambda _: None)
    seen = []

    def initially_fails(file, work, *, standard, prompt, schema):
        seen.append(file.name)
        if file.name == "part-0002.pdf":
            raise RuntimeError("Simulated API failure")
        result_path = work / "response.json"
        result_path.write_text(json.dumps(result(f"Page chunk {file.name}.")))
        return result_path

    monkeypatch.setattr(cli, "transcribe", initially_fails)
    assert cli.main(["transcribe", str(pdf), "--chunk"]) == 1
    works = list(tmp_path.glob(".pdf-transcriber-*"))
    assert len(works) == 1
    assert (works[0] / "chunks" / "part-0001" / "response.json").exists()
    assert not (works[0] / "chunks" / "part-0002" / "response.json").exists()
    seen.clear()

    def success(file, work, *, standard, prompt, schema):
        seen.append(file.name)
        result_path = work / "response.json"
        result_path.write_text(json.dumps(result(f"Page chunk {file.name}.")))
        return result_path

    monkeypatch.setattr(cli, "transcribe", success)
    assert cli.main(["transcribe", str(pdf), "--resume", str(works[0])]) == 0
    assert seen == ["part-0002.pdf", "part-0003.pdf"]
    md = pdf.with_suffix(".md").read_text()
    assert "Page chunk part-0001.pdf." in md
    assert "Page chunk part-0003.pdf." in md
    assert not works[0].exists()


def test_chunked_final_images_crop_original_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "Something - author2026_short.pdf"
    make_pdf(pdf, 11)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setattr(cli, "delete_uploaded", lambda _: None)

    def fake_transcribe(file, work, *, standard, prompt, schema):
        number = int(file.stem[-4:])
        data = result(f"<!-- pdf page 1 -->\n\n<!-- FIGURE:fig_1 -->",
                      [fig("fig_1", 1)])
        response_path = work / "response.json"
        response_path.write_text(json.dumps(data))
        return response_path

    monkeypatch.setattr(cli, "transcribe", fake_transcribe)
    assert cli.main(["transcribe", str(pdf), "--chunk"]) == 0
    md = pdf.with_suffix(".md").read_text()
    assert "<!-- pdf page 1 -->" in md
    assert "<!-- pdf page 11 -->" in md
    assert "<!-- Figure fig_2 | pdf page 11" in md
    assert (tmp_path / "author2026_short" / "fig_1.png").is_file()
    assert (tmp_path / "author2026_short" / "fig_2.png").is_file()


def test_discovery_matches_pdfmd_and_ignores_workspaces_and_conflicts(tmp_path: Path) -> None:
    old = tmp_path / "old.pdf"
    fresh = tmp_path / "sub" / "fresh.pdf"
    converted = tmp_path / "withmd.pdf"
    conflict = tmp_path / "ABC.sync-conflict-2026.pdf"
    temp_pdf = tmp_path / ".pdf-transcriber-abc" / "chunks" / "part-0001.pdf"
    for p in (old, fresh, converted, conflict, temp_pdf):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"%PDF-1.4")
    converted.with_suffix(".md").write_text("done")
    os.utime(old, (1_500_000_000, 1_500_000_000))
    missing, conflicts = discover_missing(tmp_path)
    assert [p.path for p in missing] == [fresh, old]
    assert conflicts == 1
    assert [p.path for p in discover_missing(tmp_path, limit=1)[0]] == [fresh]


def test_selection_date_shortcuts_and_default_none() -> None:
    now = datetime(2026, 10, 10, 18, 0).astimezone()
    def day_path(name, dt):
        return MissingPdf(Path(name), dt.timestamp(), dt.timestamp(), True)
    items = [day_path("today.pdf", now - timedelta(hours=1)),
             day_path("yesterday.pdf", now - timedelta(hours=25)),
             day_path("old.pdf", now - timedelta(days=4))]
    assert apply_shortcut(items, set(), "n") == set()
    assert apply_shortcut(items, set(), "r", now=now) == {Path("today.pdf")}
    assert apply_shortcut(items, set(), "1", now=now) == {Path("today.pdf")}
    assert apply_shortcut(items, set(), "2", now=now) == {Path("today.pdf"), Path("yesterday.pdf")}
    assert apply_shortcut(items, set(), "a") == {item.path for item in items}


def test_noninteractive_bulk_selection_is_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "a.pdf"
    make_pdf(pdf)
    monkeypatch.setattr(cli, "interactive_selection", lambda items, *, root: [])
    monkeypatch.setattr(cli, "go", lambda *a, **k: pytest.fail("Must not make API call"))
    assert cli.main(["--convert-missing", "--papers-dir", str(tmp_path)]) == 0
    monkeypatch.setattr(cli, "interactive_selection", lambda items, *, root: [pdf])
    monkeypatch.setattr(cli, "confirm_paid_conversion", lambda *a, **k: False)
    assert cli.main(["convert-missing", "--papers-dir", str(tmp_path)]) == 0


def test_explicit_flags_not_allowed_to_override_resume_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "x.pdf"
    make_pdf(pdf)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    workspace = tmp_path / ".pdf-transcriber-temp"
    workspace.mkdir()
    (workspace / "source.json").write_text(json.dumps({
        "pdf": str(pdf.resolve()), "sha256": cli._digest(pdf), "chunked": True
    }))
    assert cli.main(["transcribe", str(pdf), "--resume", str(workspace), "--no-chunk"]) == 1
