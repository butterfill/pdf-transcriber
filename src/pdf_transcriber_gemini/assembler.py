"""Validate transcription JSON and materialize Markdown/figures from PDF geometry.

PDF text extraction is never used as a transcription source; bounding boxes are used
only to avoid losing vector/text overlays on embedded raster images.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from urllib.parse import quote

import pymupdf

from .latex_safety import (
    latex_warnings,
    recover_json_latex,
    recover_json_markdown_asterisks,
    repair_summary,
)
from .reporting import emit

FIGURE_MARKER = re.compile(r"<!--\s*FIGURE:([A-Za-z0-9_-]+)\s*-->")
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


def figure_folder_name(stem: str) -> str:
    """Original convention: use final ' - ' (or '__') filename segment."""
    return re.split(r"\s+-\s+|__", stem)[-1]


def fail(message: str) -> None:
    raise ValueError(message)


def load_gemini_json(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8-sig").strip()
    # Permit copying an AI Studio fenced JSON code block verbatim.
    match = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", raw, re.DOTALL | re.I)
    if match:
        raw = match.group(1)
    repaired, repairs = recover_json_latex(raw)
    repaired, asterisk_repairs = recover_json_markdown_asterisks(repaired)
    try:
        data = json.loads(repaired)
    except json.JSONDecodeError as exc:
        fail(f"{path.name}: invalid JSON ({exc}). Save only the model's JSON response.")
    if not isinstance(data, dict):
        fail("Top-level JSON must be an object.")
    if not isinstance(data.get("markdown"), str) or not data["markdown"].strip():
        fail("JSON needs a non-empty string field 'markdown'.")
    if not isinstance(data.get("figures"), list):
        fail("JSON needs an array field 'figures' (use [] if no figures).")
    if not isinstance(data.get("problems"), list) or not all(
        isinstance(x, str) for x in data["problems"]
    ):
        fail("JSON needs an array of strings called 'problems' (use [] if none).")
    if repairs:
        emit("warning", f"WARNING: {path.name}: {repair_summary(repairs)}")
    if asterisk_repairs:
        emit("warning", f"WARNING: {path.name}: Recovered {len(asterisk_repairs)} "
             "Markdown asterisk JSON escape(s) before whitespace; raw JSON unchanged.")
    warnings = latex_warnings(data["markdown"])
    for warning in warnings[:20]:
        emit("warning", f"WARNING: {path.name}: LaTeX review at Markdown "
             f"{warning.line}:{warning.column}: {warning.message}")
    if len(warnings) > 20:
        emit("warning", f"WARNING: {path.name}: {len(warnings) - 20} further "
             "LaTeX warnings suppressed")
    return data


def validate_figures(data: dict, page_count: int) -> list[dict]:
    seen = set()
    figures: list[dict] = []
    for index, figure in enumerate(data["figures"], start=1):
        if not isinstance(figure, dict):
            fail(f"Figure {index} is not an object.")
        figure_id = figure.get("id")
        if not isinstance(figure_id, str) or not ID_PATTERN.fullmatch(figure_id):
            fail(f"Figure {index}: 'id' must contain only letters, numbers, _ or -.")
        if figure_id in seen:
            fail(f"Duplicate figure ID: {figure_id}")
        seen.add(figure_id)
        page_no = figure.get("pdf_page")
        if type(page_no) is not int or not 1 <= page_no <= page_count:
            fail(f"{figure_id}: pdf_page must be in 1..{page_count} (1-based).")
        coords = figure.get("box_2d")
        if not isinstance(coords, list) or len(coords) != 4 or any(
            isinstance(n, bool) or not isinstance(n, (float, int)) or
            not math.isfinite(n) or not 0 <= n <= 1000
            for n in coords
        ):
            fail(f"{figure_id}: box_2d must have four numbers in 0..1000.")
        y0, x0, y1, x1 = coords
        if not (x0 < x1 and y0 < y1):
            fail(f"{figure_id}: box_2d must be [ymin, xmin, ymax, xmax].")
        description = figure.get("description")
        if not isinstance(description, str) or not description.strip():
            fail(f"{figure_id}: description must be a non-empty string.")
        figures.append(figure)

    # A missing marker must not silently produce a transcription with missing figures.
    found = FIGURE_MARKER.findall(data["markdown"])
    if len(found) != len(set(found)):
        fail("A FIGURE marker appears more than once in markdown.")
    missing = sorted(seen - set(found))
    unknown = sorted(set(found) - seen)
    if missing or unknown:
        fail(f"Figure placeholder mismatch. Missing markers: {missing}; unknown IDs: {unknown}.")
    return figures


def figure_clips(page: pymupdf.Page, box_2d: list[float]) -> tuple[pymupdf.Rect, pymupdf.Rect]:
    """Return (displayed-page clip for rendering, unrotated box for image matching)."""
    y0, x0, y1, x1 = box_2d
    view = page.rect  # PDF page as displayed, including its rotation
    display_clip = pymupdf.Rect(
        view.x0 + view.width * x0 / 1000,
        view.y0 + view.height * y0 / 1000,
        view.x0 + view.width * x1 / 1000,
        view.y0 + view.height * y1 / 1000,
    )
    # PyMuPDF's get_pixmap clip is in displayed-page coordinates for rotated pages.
    display_clip = pymupdf.Rect(
        display_clip.x0 - 1.5, display_clip.y0 - 1.5,
        display_clip.x1 + 1.5, display_clip.y1 + 1.5,
    ) & view
    if display_clip.is_empty or display_clip.width < 2 or display_clip.height < 2:
        fail("Figure box falls outside the displayed PDF page.")
    unrotated_clip = (
        display_clip * page.derotation_matrix if page.rotation else display_clip
    )
    return display_clip, unrotated_clip


def native_raster(doc: pymupdf.Document, page: pymupdf.Page, clip: pymupdf.Rect):
    """Use an embedded JPEG/PNG only when it closely matches the whole figure.

    In all other situations (vector/composite/cropped/rotated figures), render the
    PDF page region, preserving overlays and visible appearance.
    """
    if page.rotation:
        return None
    candidates = []
    for image in page.get_image_info(xrefs=True):
        xref = image.get("xref", 0)
        if not xref:
            continue
        box = pymupdf.Rect(image["bbox"])
        # Strict match: model bbox must closely surround this ONE embedded image.
        tolerance = max(3.5, 0.015 * max(box.width, box.height))
        if max(abs(a - b) for a, b in zip(tuple(box), tuple(clip))) > tolerance:
            continue
        matrix = image.get("transform")
        if matrix is not None:
            a, b, c, d, _e, _f = matrix
            if not (a > 0 and d > 0 and abs(b) < 0.001 and abs(c) < 0.001):
                continue
        candidates.append(xref)
    if len(candidates) != 1:
        return None
    xref = candidates[0]
    # If PDF text or vector artwork is superimposed on the raster, extracting
    # the embedded file alone would lose it. Read bounding boxes only, never
    # use extracted text as a transcription source.
    for block in page.get_text("blocks"):
        if len(block) > 6 and block[6] == 0 and pymupdf.Rect(block[:4]).intersects(clip):
            return None
    if any(pymupdf.Rect(drawing["rect"]).intersects(clip) for drawing in page.get_drawings()):
        return None
    # Transparency may be carried in a separate mask, so render such images.
    masks = {im[0]: im[1] for im in page.get_images(full=True)}
    if masks.get(xref, 0):
        return None
    extracted = doc.extract_image(xref)
    if not extracted:
        return None
    ext = extracted.get("ext", "").lower()
    if ext not in {"png", "jpeg", "jpg"}:
        return None
    return ("jpg" if ext == "jpeg" else ext), extracted["image"]


def render_figure(doc: pymupdf.Document, figure: dict) -> tuple[str, bytes]:
    page = doc[figure["pdf_page"] - 1]
    display_clip, unrotated_clip = figure_clips(page, figure["box_2d"])
    original = native_raster(doc, page, unrotated_clip)
    if original is not None:
        return original
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(4, 4),
        colorspace=pymupdf.csRGB,
        clip=display_clip,
        alpha=False,
    )
    return "png", pix.tobytes("png")


def safe_description(description: str) -> str:
    """Make figure descriptions safe for HTML comments and Markdown alt text."""
    return " ".join(description.split()).replace("-->", "->")


def assembled_markdown(data: dict, figures: list[dict], fig_dir_name: str, filename_map: dict) -> str:
    replacements = {}
    for fig in figures:
        fid = fig["id"]
        description = safe_description(fig["description"])
        alt = description.replace("[", r"\[").replace("]", r"\]")
        # Percent-encode spaces and parentheses, if they occur in a folder name.
        relative = quote(f"{fig_dir_name}/{filename_map[fid]}", safe="/._-~")
        replacements[fid] = (
            f"<!-- Figure {fid} | pdf page {fig['pdf_page']} | {description} -->\n"
            f"![{alt}]({relative})"
        )

    def replace(match: re.Match[str]) -> str:
        return replacements[match.group(1)]

    md = FIGURE_MARKER.sub(replace, data["markdown"])
    return md.rstrip() + "\n"



def publish(pdf: Path, data: dict, workspace: Path) -> tuple[Path, Path | None]:
    """Stage then publish outputs beside PDF; leave workspace intact on exceptions.

    Never overwrite existing results or figure directories. This prevents surprise
    data loss and avoids sending paid model requests when outputs already exist.
    """
    markdown_path = pdf.with_suffix(".md")
    folder_name = figure_folder_name(pdf.stem)
    if folder_name in {"", ".", ".."} or "/" in folder_name or "\\" in folder_name:
        fail("Could not infer a safe figure directory name")
    figure_dir = pdf.parent / folder_name
    with pymupdf.open(pdf) as document:
        if document.needs_pass:
            fail("PDF is password-protected")
        figures = validate_figures(data, document.page_count)
        if markdown_path.exists():
            fail(f"{markdown_path.name} exists; refusing to overwrite")
        if figures and figure_dir.exists():
            fail(f"{figure_dir.name}/ exists; refusing to overwrite")
        staged_figures = workspace / "figures"
        staged_figures.mkdir(exist_ok=True)
        filenames = {}
        for fig in figures:
            ext, binary = render_figure(document, fig)
            name = f"{fig['id']}.{ext}"
            (staged_figures / name).write_bytes(binary)
            filenames[fig["id"]] = name
        staged_md = workspace / "transcription.md"
        staged_md.write_text(
            assembled_markdown(data, figures, folder_name, filenames), encoding="utf-8"
        )

    # Publish on the same filesystem, with a rollback if the second rename fails.
    moved_figures = False
    try:
        if figures:
            staged_figures.rename(figure_dir)
            moved_figures = True
        staged_md.rename(markdown_path)
    except Exception:
        if moved_figures and figure_dir.exists():
            figure_dir.rename(staged_figures)
        raise
    return markdown_path, figure_dir if figures else None
