"""Batch processing: many URLs and/or whole playlists.

A playlist becomes one folder containing a numbered sub-folder per video and a
combined study guide that links them all together:

    output/
      System Design Course/
        Combined Study Guide.txt / .md / .pdf
        01 - Caching Explained/
          Caching Explained - Short Summary.txt
          ...
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import render
from .config import Settings
from .exporters import export_pdf
from .flow import Progress, process_video
from .models import VideoResult
from .storage import atomic_write_text, safe_name
from .youtube import YouTubeError, is_playlist_only, list_playlist, parse_playlist_id, parse_video_id

log = logging.getLogger(__name__)


@dataclass
class Job:
    url: str
    group: str | None = None  # playlist / batch folder name
    index: int | None = None  # position within the group (for "01 - " prefixes)


@dataclass
class Failure:
    url: str
    error: str


@dataclass
class BatchReport:
    results: list[VideoResult] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)
    guides: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(r.stats.total_tokens for r in self.results if not r.stats.from_cache)


def expand(urls: list[str], group: str | None = None, playlist_limit: int | None = None,
           expand_playlists: bool = False, progress: Progress = lambda m: None) -> list[Job]:
    """Turn user input into a flat list of jobs.

    ``/playlist?list=...`` URLs are always expanded. A ``watch?v=...&list=...`` URL is
    treated as that one video unless ``expand_playlists`` is set.
    """
    jobs: list[Job] = []
    for url in [u.strip() for u in urls if u.strip()]:
        if parse_playlist_id(url) and (is_playlist_only(url) or expand_playlists):
            title, ids = list_playlist(url, limit=playlist_limit)
            progress(f"Playlist “{title}”: {len(ids)} videos.")
            jobs += [Job(f"https://www.youtube.com/watch?v={vid}", group=title, index=i)
                     for i, vid in enumerate(ids, 1)]
        else:
            parse_video_id(url)  # validate early, with a clear error
            jobs.append(Job(url, group=group))
    if group:
        # Number manually grouped videos in the order they were given.
        grouped = [j for j in jobs if j.group == group]
        for i, j in enumerate(grouped, 1):
            j.index = j.index or i
    return jobs


def run_batch(urls: list[str], settings: Settings | None = None, *, force: bool = False,
              group: str | None = None, playlist_limit: int | None = None,
              expand_playlists: bool = False, progress: Progress | None = None,
              crew_runner: Callable | None = None) -> BatchReport:
    settings = settings or Settings()
    say = progress or (lambda m: None)
    report = BatchReport()
    try:
        jobs = expand(urls, group=group, playlist_limit=playlist_limit,
                      expand_playlists=expand_playlists, progress=say)
    except YouTubeError as e:
        report.failures.append(Failure(", ".join(urls), str(e)))
        return report

    by_group: dict[str, list[VideoResult]] = {}
    for n, job in enumerate(jobs, 1):
        say(f"[{n}/{len(jobs)}] {job.url}")
        parent = settings.output_dir / safe_name(job.group) if job.group else None
        width = max(2, len(str(len(jobs))))
        prefix = f"{job.index:0{width}d} - " if job.group and job.index else ""
        try:
            result = process_video(job.url, settings, force=force, progress=say, output_parent=parent,
                                   folder_prefix=prefix, crew_runner=crew_runner)
        except Exception as e:  # one bad video must not stop the whole playlist
            log.exception("Failed: %s", job.url)
            report.failures.append(Failure(job.url, str(e)))
            say(f"Skipped {job.url}: {e}")
            continue
        report.results.append(result)
        if job.group:
            by_group.setdefault(job.group, []).append(result)

    for name, results in by_group.items():
        if results:
            report.guides.append(str(write_combined_guide(settings, name, results)))
            say(f"Combined study guide written for “{name}”.")
    return report


def write_combined_guide(settings: Settings, name: str, results: list[VideoResult]) -> Path:
    """One document with the short summary of every video, in order, for revision."""
    folder = settings.output_dir / safe_name(name)
    md, txt = [f"# {name} - Combined Study Guide", ""], [f"{name.upper()} - COMBINED STUDY GUIDE", "=" * 60, ""]
    md += ["| # | Video | Topic | Quality |", "|---|---|---|---|"]
    for i, r in enumerate(results, 1):
        md.append(f"| {i} | [{r.meta.title}]({r.meta.watch_url}) | {r.pack.short.topic} | {r.pack.review.overall}/10 |")
        txt.append(f"{i:>2}. {r.meta.title}  ({r.pack.short.topic})")
    md.append("")
    txt.append("")
    for i, r in enumerate(results, 1):
        md += ["---", "", render.render_short(r.pack, r.meta, "md").replace("# ", f"# {i}. ", 1), ""]
        txt += ["", render.render_short(r.pack, r.meta, "txt")]
    atomic_write_text(folder / "Combined Study Guide.txt", "\n".join(txt).rstrip() + "\n")
    md_path = folder / "Combined Study Guide.md"
    md_text = "\n".join(md).rstrip() + "\n"
    if "md" in settings.exports:
        atomic_write_text(md_path, md_text)
    if "pdf" in settings.exports:
        try:
            export_pdf(md_text, folder / "Combined Study Guide.pdf", f"{name} - Combined Study Guide")
        except Exception:
            log.exception("Combined PDF export failed")
    return folder / "Combined Study Guide.txt"
