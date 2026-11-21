"""Everything that talks to YouTube. No LLM calls in here.

* Video IDs / playlist IDs are parsed from any common URL shape.
* Metadata and playlists come from ``yt-dlp`` (no API key needed).
* Captions come from ``youtube-transcript-api``. English captions written by a
  human are preferred over auto-generated ones.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import parse_qs, urlparse

import yt_dlp
from youtube_transcript_api import (
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from .models import Segment, Transcript, VideoMeta

log = logging.getLogger(__name__)

ENGLISH_CODES = ["en", "en-US", "en-GB", "en-IN", "en-CA", "en-AU"]
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


class YouTubeError(RuntimeError):
    """A problem the user can understand and act on (bad URL, no captions, ...)."""


# --------------------------------------------------------------------------- #
# URL parsing
# --------------------------------------------------------------------------- #


def parse_video_id(url_or_id: str) -> str:
    """Return the 11-character video id from a URL (or a bare id)."""
    s = url_or_id.strip()
    if _ID_RE.match(s):
        return s
    u = urlparse(s if "://" in s else f"https://{s}")
    host = (u.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    candidate = ""
    if host == "youtu.be":
        candidate = u.path.lstrip("/").split("/")[0]
    elif host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        if u.path == "/watch":
            candidate = parse_qs(u.query).get("v", [""])[0]
        else:
            parts = [p for p in u.path.split("/") if p]
            if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live", "v", "e"}:
                candidate = parts[1]
    if _ID_RE.match(candidate):
        return candidate
    raise YouTubeError(f"Could not find a YouTube video id in: {url_or_id!r}")


def parse_playlist_id(url: str) -> str | None:
    """Return the playlist id if the URL points at a playlist, else None."""
    u = urlparse(url.strip() if "://" in url else f"https://{url.strip()}")
    host = (u.hostname or "").lower()
    if "youtube.com" not in host and "youtu.be" not in host:
        return None
    pid = parse_qs(u.query).get("list", [None])[0]
    # "RD..." lists are auto-generated mixes, which are endless and not courses.
    if pid and not pid.startswith("RD"):
        return pid
    return None


def is_playlist_only(url: str) -> bool:
    """True for /playlist?list=... URLs (no specific video)."""
    if not parse_playlist_id(url):
        return False
    try:
        parse_video_id(url)
        return False
    except YouTubeError:
        return True


# --------------------------------------------------------------------------- #
# Metadata & playlists (yt-dlp)
# --------------------------------------------------------------------------- #

class _YdlLogger:
    """Send yt-dlp's console output to our log file; errors reach the user via YouTubeError."""

    def debug(self, msg: str) -> None:
        log.debug(msg)

    info = debug

    def warning(self, msg: str) -> None:
        log.warning(msg)

    def error(self, msg: str) -> None:
        log.warning(msg)


_YDL_OPTS = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True, "logger": _YdlLogger()}


def fetch_metadata(video_id: str) -> VideoMeta:
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with yt_dlp.YoutubeDL(_YDL_OPTS) as ydl:
            info = ydl.extract_info(url, download=False, process=False)
    except yt_dlp.utils.DownloadError as e:
        raise YouTubeError(f"Could not read video {video_id}: {e}") from e
    return VideoMeta(
        video_id=video_id,
        title=(info.get("title") or video_id).strip(),
        channel=info.get("channel") or info.get("uploader") or "",
        duration_seconds=int(info.get("duration") or 0),
        url=url,
        upload_date=info.get("upload_date") or "",
        description=(info.get("description") or "")[:2000],
    )


def list_playlist(url: str, limit: int | None = None) -> tuple[str, list[str]]:
    """Return (playlist title, [video ids]) without downloading anything."""
    pid = parse_playlist_id(url)
    if not pid:
        raise YouTubeError(f"Not a playlist URL: {url!r}")
    opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist", "skip_download": True,
            "logger": _YdlLogger()}
    if limit:
        opts["playlistend"] = limit
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/playlist?list={pid}", download=False)
    except yt_dlp.utils.DownloadError as e:
        raise YouTubeError(f"Could not read playlist {pid}: {e}") from e
    ids = [e["id"] for e in info.get("entries") or [] if e and _ID_RE.match(e.get("id") or "")]
    if not ids:
        raise YouTubeError(f"Playlist {pid} has no accessible videos.")
    return (info.get("title") or pid).strip(), ids


# --------------------------------------------------------------------------- #
# Transcripts
# --------------------------------------------------------------------------- #


class SubtitlesNotFound(YouTubeError):
    """The video has no usable subtitles at all (they may be disabled)."""


def _lang(code: str) -> str:
    return code.lower().split("-")[0]


def choose_track(tracks: list, source_languages: tuple[str, ...] = ("hi",)):
    """Pick the best subtitle track.

    Order: English, then each of ``source_languages`` (e.g. Hindi), then any other
    language. Within each, human-written subtitles beat auto-generated ones.
    Returns None if there are no tracks.
    """
    wanted = ["en", *[_lang(c) for c in source_languages if _lang(c) != "en"]]

    def rank(t) -> tuple[int, int]:
        lang = _lang(t.language_code)
        group = wanted.index(lang) if lang in wanted else len(wanted)
        return (group, 1 if t.is_generated else 0)

    return min(tracks, key=rank, default=None)


def fetch_transcript(video_id: str, api: YouTubeTranscriptApi | None = None,
                     source_languages: tuple[str, ...] = ("hi",)) -> Transcript:
    """Fetch the best available subtitles, in whatever language they are.

    Non-English subtitles are returned as-is; translating them is a separate step.
    Raises SubtitlesNotFound if the video has no subtitles at all.
    """
    api = api or YouTubeTranscriptApi()
    try:
        listing = api.list(video_id)
        chosen = choose_track(list(listing), source_languages)
        if chosen is None:
            raise SubtitlesNotFound(f"Video {video_id} has no subtitles in any language.")
        fetched = chosen.fetch()
    except NoTranscriptFound as e:
        raise SubtitlesNotFound(f"Video {video_id} has no subtitles in any language.") from e
    except TranscriptsDisabled as e:
        raise SubtitlesNotFound(f"Subtitles are disabled for video {video_id}.") from e
    except VideoUnavailable as e:
        raise YouTubeError(f"Video {video_id} is unavailable (private, deleted or region-locked).") from e
    except (RequestBlocked, IpBlocked) as e:
        raise YouTubeError(
            "YouTube is blocking requests from this network (common on cloud servers). "
            "Run locally or configure a proxy."
        ) from e

    segments = [
        Segment(start=s.start, duration=s.duration, text=_clean(s.text))
        for s in fetched.snippets
        if _clean(s.text)
    ]
    if not segments:
        raise SubtitlesNotFound(f"The subtitles for {video_id} are empty.")
    log.info("Transcript %s: %d lines (%s, generated=%s)", video_id, len(segments),
             chosen.language_code, chosen.is_generated)
    return Transcript(
        video_id=video_id,
        language_code=chosen.language_code,
        is_generated=chosen.is_generated,
        segments=segments,
    )


_NOISE_RE = re.compile(r"\[(?:music|applause|laughter|silence|inaudible)\]", re.IGNORECASE)


def _clean(text: str) -> str:
    text = _NOISE_RE.sub(" ", text.replace("\n", " "))
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# Formatting for the LLM
# --------------------------------------------------------------------------- #


def fmt_ts(seconds: float | int) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def timestamped_text(segments: list[Segment], block_seconds: int = 30) -> str:
    """Group caption lines into ~30s paragraphs, each starting with [mm:ss].

    The markers let the agents cite where in the video each idea appears,
    which we turn into clickable links.
    """
    blocks: list[str] = []
    current: list[str] = []
    block_start = segments[0].start if segments else 0.0
    for seg in segments:
        if current and seg.start - block_start >= block_seconds:
            blocks.append(f"[{fmt_ts(block_start)}] {' '.join(current)}")
            current, block_start = [], seg.start
        current.append(seg.text)
    if current:
        blocks.append(f"[{fmt_ts(block_start)}] {' '.join(current)}")
    return "\n".join(blocks)


_MARKER_RE = re.compile(r"^\s*\[(?:(\d+):)?(\d{1,2}):(\d{2})\]\s*(.*)$")


def parse_ts(marker: str) -> int | None:
    """'[1:02:05]' or '[03:15]' -> seconds."""
    m = _MARKER_RE.match(marker if marker.strip().startswith("[") else f"[{marker}]")
    if not m:
        return None
    h, mnt, sec = int(m.group(1) or 0), int(m.group(2)), int(m.group(3))
    return h * 3600 + mnt * 60 + sec


def parse_marked_lines(text: str) -> list[Segment]:
    """Parse '[mm:ss] text' lines (e.g. LLM output) back into segments.
    Lines without a marker are appended to the previous segment."""
    segments: list[Segment] = []
    for line in text.splitlines():
        m = _MARKER_RE.match(line)
        if m:
            start = int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            body = m.group(4).strip()
            if body:
                segments.append(Segment(start=start, text=body))
        elif line.strip() and segments:
            segments[-1].text += " " + line.strip()
    return segments


def split_blocks(text: str, max_chars: int) -> list[str]:
    """Split timestamped text into chunks of at most ``max_chars`` on block boundaries."""
    chunks, current, size = [], [], 0
    for line in text.splitlines():
        if current and size + len(line) + 1 > max_chars:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks
