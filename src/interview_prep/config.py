"""Runtime settings, read from environment variables (and a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Bump this whenever prompts or output schemas change, so old cached results
# are not reused for the new prompts.
PROMPT_VERSION = "2026.11.2"


DEFAULT_MODEL = "gemini/gemini-2.5-flash"

# Provider prefix -> accepted API-key variables (any one of them is enough).
_KEY_VARS = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "claude": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}


def missing_api_key(model: str) -> str | None:
    """Name of the API-key variable this model needs but which is not set, else None."""
    names = _KEY_VARS.get(model.split("/")[0].lower(), ())
    if not names or any(os.getenv(n) for n in names):
        return None
    return " or ".join(names)


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", DEFAULT_MODEL))
    output_dir: Path = field(default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "output")))
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("CACHE_DIR", ".cache")))
    # Exports in addition to the .txt files: any of "md", "pdf", "notion".
    exports: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            x.strip() for x in os.getenv("EXPORTS", "md,pdf").split(",") if x.strip()
        )
    )
    # Quality gate: if the reviewer scores faithfulness below this, or finds a
    # major issue, the coach rewrites the pack once with the reviewer's notes.
    min_faithfulness: int = field(default_factory=lambda: int(os.getenv("MIN_FAITHFULNESS", "8")))
    max_revisions: int = field(default_factory=lambda: int(os.getenv("MAX_REVISIONS", "1")))
    # Transcripts longer than this (characters) are condensed chunk by chunk first.
    max_transcript_chars: int = field(default_factory=lambda: int(os.getenv("MAX_TRANSCRIPT_CHARS", "240000")))
    chunk_chars: int = field(default_factory=lambda: int(os.getenv("CHUNK_CHARS", "60000")))
    # Non-English subtitles: languages to prefer after English (e.g. "hi" = Hindi).
    # They are translated to English and cleaned up by the LLM.
    subtitle_languages: tuple[str, ...] = field(
        default_factory=lambda: tuple(x.strip() for x in os.getenv("SUBTITLE_LANGUAGES", "hi").split(",") if x.strip())
    )
    translate_chunk_chars: int = field(default_factory=lambda: int(os.getenv("TRANSLATE_CHUNK_CHARS", "15000")))
    # No subtitles at all: let Gemini listen to the video and transcribe + translate it.
    transcribe_fallback: bool = field(default_factory=lambda: _bool("TRANSCRIBE_FALLBACK", True))
    transcribe_model: str = field(default_factory=lambda: os.getenv("TRANSCRIBE_MODEL", "gemini-2.5-flash"))
    transcribe_window_seconds: int = field(default_factory=lambda: int(os.getenv("TRANSCRIBE_WINDOW_SECONDS", "900")))
    notion_token: str | None = field(default_factory=lambda: os.getenv("NOTION_TOKEN") or None)
    notion_parent_page_id: str | None = field(default_factory=lambda: os.getenv("NOTION_PARENT_PAGE_ID") or None)
    verbose: bool = field(default_factory=lambda: _bool("CREW_VERBOSE", False))

    def with_overrides(self, **kw) -> Settings:
        data = {k: v for k, v in self.__dict__.items()}
        data.update({k: v for k, v in kw.items() if v is not None})
        data["output_dir"] = Path(data["output_dir"])
        data["cache_dir"] = Path(data["cache_dir"])
        return Settings(**data)
