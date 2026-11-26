"""Self-contained HTML export with a reading view and a presentation mode."""

from __future__ import annotations

import re
from base64 import b64encode
from html import escape
from importlib.resources import files
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from webinar_transcriber import __version__
from webinar_transcriber.export.formatting import (
    FORMATTED_PARAGRAPH_RE,
    format_count,
    format_duration,
    format_timecode,
    resolve_section_image,
    section_timecode,
)
from webinar_transcriber.text_utils import split_paragraph_blocks

if TYPE_CHECKING:
    from webinar_transcriber.models import ReportDocument, ReportSection

# Frames are re-encoded so a long webinar stays a shareable single file
MAX_IMAGE_EDGE_PX = 1920
IMAGE_QUALITY = 82
SPEAKER_COLOR_COUNT = 8

_TEMPLATE_RESOURCE = files("webinar_transcriber.assets").joinpath("report_template.html")
_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")
_INLINE_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def write_html_report(report: ReportDocument, output_path: Path) -> Path:
    """Write the report as one HTML file with inline styles, script, and images.

    Returns:
        Path: The written HTML artifact path.
    """
    sections = report.sections
    has_contents = len(sections) > 1
    values = {
        "lang": escape(report.detected_language or "und"),
        "title": escape(report.title),
        "layout_class": "has-toc" if has_contents else "no-toc",
        "toc": _render_toc(sections) if has_contents else "",
        "meta": _render_meta(report),
        "overview": _render_overview(report),
        "sections": "\n".join(
            _render_section(index, section, image_base_dir=output_path.parent)
            for index, section in enumerate(sections, start=1)
        ),
        "version": escape(__version__),
    }
    template = _TEMPLATE_RESOURCE.read_text(encoding="utf-8")
    # One pass over the template, so placeholder-like text inside report values stays literal
    document = _PLACEHOLDER_RE.sub(lambda match: values[match[1]], template)
    output_path.write_text(document, encoding="utf-8")
    return output_path


def _render_meta(report: ReportDocument) -> str:
    chips = [report.media_type.value.capitalize()]
    if report.sections:
        chips.append(format_duration(report.sections[-1].end_sec))
        chips.append(format_count(len(report.sections), "section"))
    if report.detected_language:
        chips.append(f"Language: {report.detected_language}")
    chips.append(Path(report.source_file).name)
    return "".join(f"<li>{escape(chip)}</li>" for chip in chips)


def _render_overview(report: ReportDocument) -> str:
    cards = []
    if report.summary:
        items = "".join(f"<li>{_inline(item)}</li>" for item in report.summary)
        cards.append(
            '<section class="card" id="summary" data-slide="Summary">'
            f'<h2>Summary</h2><ul class="summary-list">{items}</ul></section>'
        )
    if report.action_items:
        items = "".join(f"<li>{_inline(item)}</li>" for item in report.action_items)
        cards.append(
            '<section class="card" id="action-items" data-slide="Action items">'
            f'<h2>Action items</h2><ul class="action-list">{items}</ul></section>'
        )
    return f'<div class="overview">{"".join(cards)}</div>' if cards else ""


def _render_toc(sections: list[ReportSection]) -> str:
    links = "".join(
        f'<li><a href="#section-{index}"><span class="toc-time">'
        f"{format_timecode(section.start_sec)}</span>"
        f'<span class="toc-title">{escape(section.title)}</span></a></li>'
        for index, section in enumerate(sections, start=1)
    )
    return f'<nav class="toc" aria-label="Contents"><h2>Contents</h2><ol>{links}</ol></nav>'


def _render_section(index: int, section: ReportSection, *, image_base_dir: Path) -> str:
    timecode = section_timecode(section.start_sec, section.end_sec)
    parts = [
        f'<section class="section" id="section-{index}">',
        '<header class="section-head">',
        f'<span class="section-index">{index:02d}</span>',
        f"<div><h2>{escape(section.title)}</h2>",
        f'<a class="timecode" href="#section-{index}">{timecode}</a></div>',
        "</header>",
    ]
    if image_path := resolve_section_image(section.image_path, image_base_dir):
        parts.append(_render_image(image_path, alt=section.title))
    if section.tldr and (tldr := section.tldr.strip()):
        lines = [
            line.strip()
            for block in split_paragraph_blocks(tldr) or [tldr]
            for line in block.splitlines()
            if line.strip()
        ]
        parts.append(
            '<div class="notes"><p class="label">AI-generated notes</p>'
            f'<div class="notes-body">{_render_blocks(lines)}</div></div>'
        )
    if blocks := split_paragraph_blocks(section.transcript_text):
        parts.append(
            '<div class="transcript"><p class="label">Transcript</p>'
            f'<div class="transcript-body">{_render_blocks(blocks)}</div></div>'
        )
    parts.append("</section>")
    return "\n".join(parts)


def _render_image(image_path: Path, *, alt: str) -> str:
    with Image.open(image_path) as image:
        image.thumbnail((MAX_IMAGE_EDGE_PX, MAX_IMAGE_EDGE_PX))
        width, height = image.size
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="WEBP", quality=IMAGE_QUALITY)
    data = b64encode(buffer.getvalue()).decode("ascii")
    return (
        f'<figure class="frame"><img src="data:image/webp;base64,{data}" '
        f'width="{width}" height="{height}" alt="{escape(alt)}" loading="lazy"></figure>'
    )


def _render_blocks(blocks: list[str]) -> str:
    html: list[str] = []
    open_list: str | None = None
    for block in blocks:
        match = FORMATTED_PARAGRAPH_RE.match(block)
        list_tag, item = None, None
        if match and (item := match["bullet"] or match["number"]):
            list_tag = "ul" if match["bullet"] else "ol"
        if open_list != list_tag:
            if open_list:
                html.append(f"</{open_list}>")
            if list_tag:
                html.append(f"<{list_tag}>")
            open_list = list_tag
        if item:
            html.append(f"<li>{_inline(item)}</li>")
        elif match and (speaker := match["speaker"]):
            color = int(speaker[1:]) % SPEAKER_COLOR_COUNT
            html.append(
                f'<p class="turn"><span class="speaker spk-{color}">{speaker}</span> '
                f"{_inline(match['speaker_body'].strip())}</p>"
            )
        else:
            html.append(f"<p>{_inline(block)}</p>")
    if open_list:
        html.append(f"</{open_list}>")
    return "".join(html)


def _inline(text: str) -> str:
    return _INLINE_BOLD_RE.sub(r"<strong>\1</strong>", escape(text)).replace("\n", "<br>")
