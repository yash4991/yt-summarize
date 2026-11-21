"""Optional exports: PDF study guide and Notion page."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .storage import atomic_write_bytes

log = logging.getLogger(__name__)

PDF_CSS = """
body { font-family: Helvetica, Arial, sans-serif; font-size: 11pt; line-height: 1.45; color: #1f2328; }
h1 { font-size: 18pt; color: #0b3d91; border-bottom: 2px solid #0b3d91; padding-bottom: 4px; }
h2 { font-size: 14pt; color: #0b3d91; margin-top: 16px; }
h3 { font-size: 12pt; color: #333; margin-top: 12px; }
a { color: #0b5cd6; text-decoration: none; }
li { margin-bottom: 3px; }
hr { border: 0; border-top: 1px solid #ccc; }
"""


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #


def export_pdf(markdown_text: str, path: Path, title: str) -> Path:
    from markdown_pdf import MarkdownPdf, Section

    pdf = MarkdownPdf(toc_level=2, optimize=True)
    # Each "---" separated part starts on a new page.
    for part in re.split(r"\n\s*---\s*\n", markdown_text):
        if part.strip():
            pdf.add_section(Section(part, toc=True), user_css=PDF_CSS)
    pdf.meta["title"] = title
    pdf.meta["creator"] = "Interview Prep Agent"
    tmp = path.with_name(f".tmp-{path.name}")
    pdf.save(str(tmp))
    atomic_write_bytes(path, tmp.read_bytes())
    tmp.unlink(missing_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Notion
# --------------------------------------------------------------------------- #

_INLINE = re.compile(r"\*\*(?P<bold>.+?)\*\*|\[(?P<text>[^\]]+)\]\((?P<url>[^)]+)\)")
_MAX_TEXT = 1900  # Notion limit is 2000 characters per text object.


def _rich_text(line: str) -> list[dict]:
    """Convert **bold** and [text](url) to Notion rich text."""
    out: list[dict] = []

    def add(content: str, bold: bool = False, url: str | None = None) -> None:
        for i in range(0, len(content), _MAX_TEXT):
            piece = content[i : i + _MAX_TEXT]
            if not piece:
                continue
            item = {"type": "text", "text": {"content": piece}}
            if url:
                item["text"]["link"] = {"url": url}
            if bold:
                item["annotations"] = {"bold": True}
            out.append(item)

    pos = 0
    for m in _INLINE.finditer(line):
        add(line[pos : m.start()])
        if m.group("bold"):
            add(m.group("bold"), bold=True)
        else:
            add(m.group("text"), url=m.group("url"))
        pos = m.end()
    add(line[pos:])
    return out


def markdown_to_blocks(md: str) -> list[dict]:
    blocks: list[dict] = []
    for raw in md.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.strip() == "---":
            blocks.append({"object": "block", "type": "divider", "divider": {}})
            continue
        m = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m:
            kind = f"heading_{len(m.group(1))}"
            blocks.append({"object": "block", "type": kind, kind: {"rich_text": _rich_text(m.group(2))}})
            continue
        m = re.match(r"^\s*(?:[-*]|\d+\.)\s+(.*)$", line)
        if m:
            kind = "numbered_list_item" if re.match(r"^\s*\d+\.", line) else "bulleted_list_item"
            blocks.append({"object": "block", "type": kind, kind: {"rich_text": _rich_text(m.group(1))}})
            continue
        blocks.append({"object": "block", "type": "paragraph", "paragraph": {"rich_text": _rich_text(line)}})
    return blocks


def export_notion(markdown_text: str, title: str, token: str, parent_page_id: str) -> str:
    """Create a Notion page under ``parent_page_id`` and return its URL."""
    from notion_client import Client

    client = Client(auth=token)
    blocks = markdown_to_blocks(markdown_text)
    # The page title is already the Notion page title; skip the first H1.
    if blocks and blocks[0]["type"] == "heading_1":
        blocks = blocks[1:]
    page = client.pages.create(
        parent={"page_id": parent_page_id},
        properties={"title": {"title": [{"type": "text", "text": {"content": title[:_MAX_TEXT]}}]}},
        children=blocks[:100],
    )
    # Notion accepts at most 100 blocks per request.
    for i in range(100, len(blocks), 100):
        client.blocks.children.append(block_id=page["id"], children=blocks[i : i + 100])
    return page.get("url", "")
