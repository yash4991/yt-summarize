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


def is_local(model: str) -> bool:
    """'local' or 'local/<model-id>' means the local OpenAI-compatible server (LM Studio)."""
    return model == "local" or model.startswith("local/")


@dataclass(frozen=True)
class Limits:
    max_transcript_chars: int
    chunk_chars: int
    translate_chunk_chars: int
    max_output_tokens: int
    parallel: int


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

    # --- Local model (LM Studio or any OpenAI-compatible server) ------------
    # Use it with LLM_MODEL=local (or local/<model-id>), or as FALLBACK_MODEL.
    local_base_url: str = field(default_factory=lambda: os.getenv("LOCAL_BASE_URL", "http://localhost:1234/v1"))
    local_model: str = field(default_factory=lambda: os.getenv("LOCAL_MODEL", ""))  # empty = first loaded model
    local_api_key: str = field(default_factory=lambda: os.getenv("LOCAL_API_KEY", "lm-studio"))
    # Must match (or be below) the context length the model is loaded with in LM Studio.
    local_context_tokens: int = field(default_factory=lambda: int(os.getenv("LOCAL_CONTEXT_TOKENS", "16384")))
    # "none" turns off long hidden "thinking", which is far too slow on a laptop GPU.
    local_reasoning_effort: str = field(default_factory=lambda: os.getenv("LOCAL_REASONING_EFFORT", "none"))
    local_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("LOCAL_TIMEOUT_SECONDS", "1800")))
    # Model to switch to when the main model's daily quota runs out (e.g. "local"). Empty = no fallback.
    fallback_model: str = field(default_factory=lambda: os.getenv("FALLBACK_MODEL", ""))

    def limits(self, model: str | None = None) -> Limits:
        """Size limits for a model. Local models have a much smaller context window."""
        model = model or self.model
        if not is_local(model):
            return Limits(self.max_transcript_chars, self.chunk_chars, self.translate_chunk_chars, 16000, 4)
        ctx = self.local_context_tokens
        # Each crew task sees the transcript plus earlier tasks' output, and must leave room to answer.
        return Limits(
            max_transcript_chars=int(ctx * 0.3 * 3.5),
            chunk_chars=int(ctx * 0.35 * 3.5),
            translate_chunk_chars=int(ctx * 0.3 * 2.5),  # Devanagari packs fewer chars per token
            max_output_tokens=min(16000, ctx // 2),
            parallel=1,  # one GPU: parallel requests only queue up
        )

    def with_overrides(self, **kw) -> Settings:
        data = {k: v for k, v in self.__dict__.items()}
        data.update({k: v for k, v in kw.items() if v is not None})
        data["output_dir"] = Path(data["output_dir"])
        data["cache_dir"] = Path(data["cache_dir"])
        return Settings(**data)
