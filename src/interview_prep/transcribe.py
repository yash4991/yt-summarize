"""Fallback for videos with no subtitles: Gemini listens to the video and writes
a clean English transcript with timestamps.

Gemini can read a public YouTube URL directly, so nothing is downloaded and no
ffmpeg is needed. Long videos are processed in windows (15 minutes by default)
using clip offsets. Frames are sampled very sparsely (we only need the speech),
which keeps the token cost to roughly 40 tokens per second of video.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from .models import Segment, Transcript, VideoMeta
from .retry import with_retry
from .translate import CLEANUP_RULES
from .youtube import fmt_ts, parse_marked_lines

log = logging.getLogger(__name__)

TRANSCRIBE_PROMPT = """You are an expert technical transcriber and translator preparing the transcript of \
a YouTube video for job-interview study notes.

Video title: {title}
Channel: {channel}
Video description (topic context): {description}

You are given the part of the video from {start} to {end}. Listen to the speech (it may be in Hindi, \
English or a mix of both) and write it down.

{rules}
6. Start a new line roughly every 20-30 seconds of speech. Begin each line with the time it starts, \
measured from the START OF THE FULL VIDEO, as [mm:ss] or [h:mm:ss]. Output nothing else.
"""


class TranscriptionError(RuntimeError):
    pass


def gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def _gemini_call(model: str, url: str, start: int | None, end: int | None, prompt: str) -> tuple[str, int]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=gemini_api_key())
    video = types.Part(
        file_data=types.FileData(file_uri=url),
        video_metadata=types.VideoMetadata(
            start_offset=f"{start}s" if start is not None else None,
            end_offset=f"{end}s" if end is not None else None,
            fps=0.1,  # one frame every 10 seconds: we only need the audio
        ),
    )
    resp = client.models.generate_content(
        model=model,
        contents=types.Content(role="user", parts=[video, types.Part(text=prompt)]),
        config=types.GenerateContentConfig(
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
            temperature=0.1,
            max_output_tokens=32000,
        ),
    )
    usage = resp.usage_metadata
    tokens = (usage.total_token_count or 0) if usage else 0
    return resp.text or "", tokens


def fix_offsets(segments: list[Segment], start: int, end: int) -> list[Segment]:
    """Models sometimes give times relative to the clip instead of the full video.
    If every timestamp fits inside the clip length but not the clip's real range, shift it."""
    if not segments or start == 0:
        return segments
    length = end - start
    times = [s.start for s in segments]
    in_real_range = all(start - 5 <= t <= end + 5 for t in times)
    looks_relative = max(times) <= length + 5 and min(times) < start - 5
    if not in_real_range and looks_relative:
        return [s.model_copy(update={"start": s.start + start}) for s in segments]
    return segments


def transcribe_video(
    meta: VideoMeta,
    model: str = "gemini-2.5-flash",
    window_seconds: int = 900,
    progress: Callable[[str], None] = lambda m: None,
    call: Callable[..., tuple[str, int]] = _gemini_call,
) -> tuple[Transcript, int]:
    """Return (English transcript, tokens used)."""
    model = model.split("/", 1)[1] if model.startswith(("gemini/", "google/")) else model
    if call is _gemini_call and not gemini_api_key():
        raise TranscriptionError(
            "This video has no subtitles. Transcribing it needs a Gemini key: set GEMINI_API_KEY in .env."
        )
    duration = meta.duration_seconds
    windows = ([(s, min(s + window_seconds, duration)) for s in range(0, duration, window_seconds)]
               if duration else [(None, None)])

    segments: list[Segment] = []
    tokens = 0
    for i, (start, end) in enumerate(windows, 1):
        label = f"{fmt_ts(start)}-{fmt_ts(end)}" if start is not None else "full video"
        progress(f"No subtitles: Gemini is transcribing and translating the audio "
                 f"({label}, part {i}/{len(windows)})...")
        prompt = TRANSCRIBE_PROMPT.format(
            title=meta.title, channel=meta.channel or "unknown", description=meta.description.strip() or "(none)",
            start=fmt_ts(start or 0), end=fmt_ts(end) if end else "the end", rules=CLEANUP_RULES,
        )
        try:
            text, used = with_retry(lambda s=start, e=end, p=prompt: call(model, meta.watch_url, s, e, p),
                                    f"transcription part {i}", progress)
        except Exception as e:
            raise TranscriptionError(f"Gemini could not transcribe this video: {e}") from e
        tokens += used
        part = parse_marked_lines(text)
        if start is not None:
            part = fix_offsets(part, start, end)
        segments += part

    segments = [s for s in sorted(segments, key=lambda s: s.start) if s.text.strip().lower() != "[unclear]"]
    if not segments:
        raise TranscriptionError("Gemini returned no transcript for this video (it may have no speech).")
    return Transcript(video_id=meta.video_id, language_code="en", is_generated=True, segments=segments,
                      source="ai_transcription"), tokens
