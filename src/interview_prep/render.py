"""Turn the structured study pack into readable documents.

The same content is rendered two ways:

* ``txt``: plain text that reads well in Notepad (the core deliverable).
* ``md``:  Markdown, used for .md files, the PDF and Notion.

Timestamps become clickable links that open the video at that second.
"""

from __future__ import annotations

import textwrap
from typing import Literal

from .models import KeyPoint, StudyPack, Transcript, VideoMeta
from .youtube import fmt_ts, timestamped_text

Fmt = Literal["txt", "md"]


def ts_url(meta: VideoMeta, seconds: int) -> str:
    return f"https://youtu.be/{meta.video_id}?t={int(seconds)}"


def valid_ts(meta: VideoMeta, seconds: int | None) -> int | None:
    """Drop timestamps the model got wrong (negative or past the end of the video)."""
    if seconds is None or seconds < 0:
        return None
    if meta.duration_seconds and seconds > meta.duration_seconds + 5:
        return None
    return int(seconds)


class Doc:
    """A tiny document builder that writes either plain text or Markdown."""

    def __init__(self, fmt: Fmt, meta: VideoMeta) -> None:
        self.fmt, self.meta, self.lines = fmt, meta, []

    def title(self, text: str) -> None:
        if self.fmt == "md":
            self.lines += [f"# {text}", ""]
        else:
            self.lines += [text.upper(), "=" * min(len(text), 78), ""]

    def heading(self, text: str, level: int = 2) -> None:
        if self.fmt == "md":
            self.lines += [f"{'#' * level} {text}", ""]
        else:
            ch = "-" if level == 2 else "~"
            self.lines += [text.upper() if level == 2 else text, ch * min(len(text), 78), ""]

    def para(self, text: str) -> None:
        for p in [p.strip() for p in text.replace("\r", "").split("\n\n") if p.strip()]:
            p = " ".join(p.split())
            self.lines += ([p] if self.fmt == "md" else textwrap.wrap(p, 88)) + [""]

    def ts(self, seconds: int | None) -> str:
        s = valid_ts(self.meta, seconds)
        if s is None:
            return ""
        if self.fmt == "md":
            return f" ([{fmt_ts(s)}]({ts_url(self.meta, s)}))"
        return f"  [{fmt_ts(s)}]"

    def bullets(self, items: list[str | KeyPoint], numbered: bool = False) -> None:
        if not items:
            return
        for n, item in enumerate(items, 1):
            if isinstance(item, KeyPoint):
                text = item.point.strip() + self.ts(item.timestamp_seconds)
            else:
                text = str(item).strip()
            marker = f"{n}." if numbered else ("-" if self.fmt == "md" else "•")
            if self.fmt == "md":
                self.lines.append(f"{marker} {text}")
            else:
                wrapped = textwrap.wrap(text, 84) or [""]
                indent = " " * (len(marker) + 1)
                self.lines.append(f"  {marker} {wrapped[0]}")
                self.lines += [f"  {indent}{w}" for w in wrapped[1:]]
        self.lines.append("")

    def label(self, name: str, value: str) -> None:
        self.lines += [f"**{name}:** {value}" if self.fmt == "md" else f"{name}: {value}", ""]

    def rule(self) -> None:
        self.lines += ["---" if self.fmt == "md" else "-" * 40, ""]

    def header_block(self, kind: str) -> None:
        m = self.meta
        self.title(f"{m.title} - {kind}")
        dur = f" ({fmt_ts(m.duration_seconds)})" if m.duration_seconds else ""
        if self.fmt == "md":
            self.lines += [f"**Video:** [{m.title}]({m.watch_url}) - {m.channel}{dur}", ""]
        else:
            self.lines += [f"Video:   {m.title} - {m.channel}{dur}", f"Link:    {m.watch_url}", ""]

    def text(self) -> str:
        return "\n".join(self.lines).rstrip() + "\n"


def timestamp_note(fmt: Fmt) -> str:
    if fmt == "md":
        return "_Timestamps link to the exact moment in the video._"
    return "Timestamps like [03:15] show where in the video each point is explained."


def render_short(pack: StudyPack, meta: VideoMeta, fmt: Fmt = "txt") -> str:
    s = pack.short
    d = Doc(fmt, meta)
    d.header_block("Short Summary")
    d.label("Topic", s.topic)
    d.heading("What it is")
    d.para(s.what_it_is)
    d.heading("Key points")
    d.bullets(s.key_points)
    d.heading("Say this in the interview")
    d.para(f"“{s.interview_one_liner.strip().strip(chr(34))}”")
    d.para(timestamp_note(fmt))
    return d.text()


def render_detailed(pack: StudyPack, meta: VideoMeta, fmt: Fmt = "txt") -> str:
    x = pack.detailed
    d = Doc(fmt, meta)
    d.header_block("Detailed Summary")
    d.label("Topic", x.topic)
    d.heading("Overview")
    d.para(x.overview)
    for i, sec in enumerate(x.sections, 1):
        d.heading(f"{i}. {sec.heading}")
        d.para(sec.explanation)
        d.bullets(sec.bullets)
    for name, items in [
        ("Examples", x.examples),
        ("Trade-offs", x.trade_offs),
        ("Common mistakes", x.common_mistakes),
        ("Beyond the video (often asked in interviews)", x.beyond_the_video),
    ]:
        if items:
            d.heading(name)
            d.bullets(items)
    d.heading("Quick recap")
    d.bullets(x.quick_recap)
    d.para(timestamp_note(fmt))
    return d.text()


def render_qa(pack: StudyPack, meta: VideoMeta, fmt: Fmt = "txt") -> str:
    d = Doc(fmt, meta)
    d.header_block("Interview Q&A")
    order = {"easy": 0, "medium": 1, "hard": 2}
    items = sorted(pack.qa.questions, key=lambda q: order.get(q.difficulty, 1))
    for n, q in enumerate(items, 1):
        d.heading(f"Q{n}. {q.question}  [{q.difficulty.upper()}]", level=3)
        d.para(q.model_answer)
        if q.follow_up:
            d.label("Likely follow-up", q.follow_up)
        s = valid_ts(meta, q.timestamp_seconds)
        if s is not None:
            url = ts_url(meta, s)
            d.label("Covered in video at", f"[{fmt_ts(s)}]({url})" if fmt == "md" else f"{fmt_ts(s)}  ({url})")
    return d.text()


def render_review(pack: StudyPack, meta: VideoMeta, fmt: Fmt = "md") -> str:
    r = pack.review
    d = Doc(fmt, meta)
    d.header_block("Quality Review")
    d.bullets([
        f"Overall: {r.overall}/10",
        f"Faithfulness to the video: {r.faithfulness_score}/10",
        f"Coverage: {r.coverage_score}/10",
        f"Clarity: {r.clarity_score}/10",
        f"Revisions made after review: {pack.revisions}",
    ])
    d.para(r.verdict)
    if r.issues:
        d.heading("Remaining notes")
        d.bullets([f"[{i.severity}] {i.location}: {i.problem} -> {i.fix}" for i in r.issues])
    return d.text()


def render_transcript(transcript: Transcript, meta: VideoMeta) -> str:
    """The English transcript the agents worked from, one timestamped passage per paragraph."""
    lines = [f"{meta.title} - Transcript (English)".upper(), "=" * 60, "",
             f"Video:   {meta.title} - {meta.channel}", f"Link:    {meta.watch_url}",
             f"Source:  {transcript.describe()}", ""]
    for block in timestamped_text(transcript.segments).splitlines():
        lines += textwrap.wrap(block, 100, subsequent_indent="        ") + [""]
    return "\n".join(lines).rstrip() + "\n"


def render_study_guide(pack: StudyPack, meta: VideoMeta) -> str:
    """All three documents in one Markdown file (used for the PDF and Notion)."""
    parts = [render_short(pack, meta, "md"), render_detailed(pack, meta, "md"), render_qa(pack, meta, "md")]
    return "\n\n---\n\n".join(parts)


DOCUMENTS = {
    "Short Summary": render_short,
    "Detailed Summary": render_detailed,
    "Interview Q&A": render_qa,
}
