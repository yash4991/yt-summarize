"""On-disk cache so the same video is never downloaded or summarised twice.

* ``transcripts/<video_id>.json``  metadata + captions (independent of the model)
* ``packs/<video_id>__<model>__<prompt_version>.json``  the generated study pack

Changing the model or bumping ``PROMPT_VERSION`` automatically misses the cache.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel, ValidationError

from .models import RunStats, StudyPack, Transcript, VideoMeta
from .storage import atomic_write_text

log = logging.getLogger(__name__)


class TranscriptEntry(BaseModel):
    meta: VideoMeta
    transcript: Transcript | None  # None = the video has no subtitles


class PackEntry(BaseModel):
    pack: StudyPack
    stats: RunStats


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")


class Cache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _transcript_path(self, video_id: str) -> Path:
        return self.root / "transcripts" / f"{video_id}.json"

    def _pack_path(self, video_id: str, model: str, prompt_version: str) -> Path:
        return self.root / "packs" / f"{video_id}__{_slug(model)}__{_slug(prompt_version)}.json"

    def _load(self, path: Path, model: type[BaseModel]):
        if not path.exists():
            return None
        try:
            return model.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, ValidationError) as e:
            # A corrupt cache entry is not fatal: drop it and regenerate.
            log.warning("Ignoring unreadable cache file %s (%s)", path, e)
            path.unlink(missing_ok=True)
            return None

    def get_transcript(self, video_id: str) -> TranscriptEntry | None:
        return self._load(self._transcript_path(video_id), TranscriptEntry)

    def put_transcript(self, meta: VideoMeta, transcript: Transcript | None) -> None:
        entry = TranscriptEntry(meta=meta, transcript=transcript)
        atomic_write_text(self._transcript_path(meta.video_id), entry.model_dump_json())

    def _english_path(self, video_id: str, model: str, prompt_version: str) -> Path:
        return self.root / "english" / f"{video_id}__{_slug(model)}__{_slug(prompt_version)}.json"

    def get_english(self, video_id: str, model: str, prompt_version: str) -> Transcript | None:
        """Translated / AI-transcribed English text (the expensive part for non-English videos)."""
        return self._load(self._english_path(video_id, model, prompt_version), Transcript)

    def put_english(self, video_id: str, model: str, prompt_version: str, transcript: Transcript) -> None:
        atomic_write_text(self._english_path(video_id, model, prompt_version), transcript.model_dump_json())

    def get_pack(self, video_id: str, model: str, prompt_version: str) -> PackEntry | None:
        return self._load(self._pack_path(video_id, model, prompt_version), PackEntry)

    def put_pack(self, video_id: str, pack: StudyPack, stats: RunStats) -> None:
        entry = PackEntry(pack=pack, stats=stats)
        path = self._pack_path(video_id, stats.model, stats.prompt_version)
        atomic_write_text(path, entry.model_dump_json(indent=2))

    def stats(self) -> dict[str, int]:
        return {
            "transcripts": len(list((self.root / "transcripts").glob("*.json"))),
            "packs": len(list((self.root / "packs").glob("*.json"))),
            "english": len(list((self.root / "english").glob("*.json"))),
        }


def dump_json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
