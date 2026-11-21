"""Command line interface.

    interview-prep <url> [<url> ...]          one or more videos
    interview-prep <playlist-url>              a whole playlist
    interview-prep ui                          launch the web UI
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import Settings, missing_api_key


def setup_console() -> None:
    # Windows consoles default to a legacy code page; crewai prints emoji.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def setup_logging(log_dir: Path, verbose: bool) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_dir / "interview_prep.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    if verbose:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        root.addHandler(console)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="interview-prep",
        description="Turn YouTube videos into interview-ready study notes (short + detailed summary + Q&A).",
    )
    p.add_argument("urls", nargs="*", help="YouTube video or playlist URLs, or 'ui' to launch the web app.")
    p.add_argument("-f", "--file", type=Path, help="Text file with one URL per line.")
    p.add_argument("--force", action="store_true", help="Ignore the cache and regenerate everything.")
    p.add_argument("--group", help="Put all given videos in one folder with a combined study guide.")
    p.add_argument("--whole-playlist", action="store_true",
                   help="For watch?v=...&list=... URLs, process the whole playlist instead of one video.")
    p.add_argument("--limit", type=int, help="Only the first N videos of each playlist.")
    p.add_argument("--export", help="Comma-separated extra formats: md,pdf,notion (default from EXPORTS or md,pdf).")
    p.add_argument("--model", help="LLM, e.g. anthropic/claude-sonnet-5-5 (default from LLM_MODEL).")
    p.add_argument("--out", type=Path, help="Output folder (default: output).")
    p.add_argument("-v", "--verbose", action="store_true", help="Show agent reasoning and logs.")
    return p


def main(argv: list[str] | None = None) -> int:
    setup_console()
    args = build_parser().parse_args(argv)

    if args.urls[:1] == ["ui"]:
        app = Path(__file__).with_name("app.py")
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])

    urls = list(args.urls)
    if args.file:
        urls += [ln.strip() for ln in args.file.read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.lstrip().startswith("#")]
    if not urls:
        build_parser().print_help()
        return 2

    settings = Settings().with_overrides(
        model=args.model,
        output_dir=args.out,
        exports=tuple(x.strip() for x in args.export.split(",")) if args.export else None,
        verbose=args.verbose or None,
    )
    setup_logging(settings.cache_dir / "logs", args.verbose)
    key = missing_api_key(settings.model)
    if key:
        print(f"Warning: {key} is not set (needed for {settings.model}). Only cached videos will work.\n"
              f"Add it to a .env file in this folder - see .env.example.")

    from .pipeline import run_batch

    report = run_batch(urls, settings, force=args.force, group=args.group, playlist_limit=args.limit,
                       expand_playlists=args.whole_playlist, progress=lambda m: print(f"  {m}", flush=True))

    print()
    for r in report.results:
        cost = "cached, 0 tokens" if r.stats.from_cache else f"{r.stats.total_tokens:,} tokens, {r.stats.seconds}s"
        print(f"OK    {r.meta.title}")
        print(f"      quality {r.pack.review.overall}/10 | {cost}")
        print(f"      {r.folder}")
        if r.notion_url:
            print(f"      Notion: {r.notion_url}")
    for g in report.guides:
        print(f"GUIDE {g}")
    for f in report.failures:
        print(f"FAIL  {f.url}\n      {f.error}")
    return 1 if report.failures and not report.results else 0


if __name__ == "__main__":
    sys.exit(main())
