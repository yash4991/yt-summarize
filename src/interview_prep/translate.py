"""Turn non-English (typically auto-generated Hindi) subtitles into clean English.

Auto-generated Hindi subtitles for tech videos are usually Hinglish written in
Devanagari, with no punctuation and many mis-heard words ("रम" for RAM, "रेप्स"
for replicas). YouTube's own machine translation keeps those mistakes. Here
an LLM translates and repairs the text using the video title and description
as topic context, keeping the [mm:ss] markers so timestamps keep working.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from .config import Settings
from .llm import make_llm
from .models import Transcript, VideoMeta, language_name
from .retry import with_retry
from .youtube import parse_marked_lines, split_blocks, timestamped_text

log = logging.getLogger(__name__)

CLEANUP_RULES = """\
Rewrite it as clear, correct English:
1. Translate everything into natural English. Write technical terms in their standard English form \
(e.g. load balancer, EC2, Kafka, horizontal scaling, time complexity).
2. Fix mis-recognised words to what the speaker most plausibly meant, judging from the topic and the \
surrounding sentences. Speech recognition often garbles technical terms and English words said in an \
Indian accent.
3. If the speaker corrects themselves, keep only the corrected meaning.
4. Drop filler ("okay", "right", "so basically", "guys"), greetings and repetition, but do NOT summarise: \
keep every technical point, example, number and step.
5. If a passage is truly unintelligible, write [unclear] instead of guessing."""

TRANSLATE_PROMPT = """You are an expert technical translator and editor. You are preparing the transcript \
of a YouTube video so it can be turned into job-interview study notes.

Video title: {title}
Channel: {channel}
Video description (topic context): {description}

Below is part {part} of {parts} of the video's {kind} {language} subtitles. They come from speech \
recognition, so expect English words written in {language} script, missing punctuation and \
mis-heard words.

{rules}
6. Output exactly one line for each input line, starting with the same [mm:ss] marker as that input \
line. Output nothing else: no headings, notes or explanations.

SUBTITLES:
{chunk}
"""


class TranslationError(RuntimeError):
    pass


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Some local models put their hidden reasoning in <think> tags inside the answer."""
    return _THINK_RE.sub("", text or "")


def make_llm_call(settings: Settings) -> Callable[[str, str], tuple[str, int]]:
    def call(model: str, prompt: str) -> tuple[str, int]:
        llm = make_llm(settings, model, temperature=0.1)
        text = llm.call([{"role": "user", "content": prompt}])
        usage = getattr(llm, "_token_usage", None) or {}
        return strip_reasoning(str(text or "")), int(usage.get("total_tokens", 0))

    return call


def translate_transcript(
    transcript: Transcript,
    meta: VideoMeta,
    model: str,
    chunk_chars: int = 15000,
    progress: Callable[[str], None] = lambda m: None,
    llm_call: Callable[[str, str], tuple[str, int]] | None = None,
    settings: Settings | None = None,
    parallel: int = 4,
) -> tuple[Transcript, int]:
    """Return (English transcript, tokens used)."""
    llm_call = llm_call or make_llm_call(settings or Settings())
    source_text = timestamped_text(transcript.segments)
    chunks = split_blocks(source_text, chunk_chars)
    language = language_name(transcript.language_code)
    kind = "auto-generated" if transcript.is_generated else "human-written"
    progress(f"Translating {language} subtitles to English and cleaning them up "
             f"({len(chunks)} part{'s' if len(chunks) != 1 else ''})...")

    def work(i: int, chunk: str) -> tuple[str, int]:
        prompt = TRANSLATE_PROMPT.format(
            title=meta.title, channel=meta.channel or "unknown", description=meta.description.strip() or "(none)",
            part=i, parts=len(chunks), kind=kind, language=language, rules=CLEANUP_RULES, chunk=chunk,
        )
        expected = len(chunk.splitlines())
        tokens = 0
        for attempt in (1, 2):
            text, used = with_retry(lambda: llm_call(model, prompt), f"translation part {i}", progress)
            tokens += used
            got = len(parse_marked_lines(text))
            # Lines may legitimately merge or drop (pure filler), but losing most of the
            # markers means the model ignored the format: retry once.
            if got >= max(1, expected // 2):
                return text, tokens
            log.warning("Translation part %d returned %d/%d marked lines (attempt %d)", i, got, expected, attempt)
        if got:
            return text, tokens
        raise TranslationError(f"Could not translate part {i} of the subtitles into the expected format.")

    with ThreadPoolExecutor(max_workers=max(1, min(parallel, len(chunks)))) as pool:
        results = list(pool.map(lambda a: work(*a), enumerate(chunks, 1)))

    segments = parse_marked_lines("\n".join(text for text, _ in results))
    segments = [s for s in segments if s.text.strip().lower() != "[unclear]"]
    if not segments:
        raise TranslationError("Translation produced no text.")
    english = Transcript(
        video_id=transcript.video_id,
        language_code="en",
        is_generated=transcript.is_generated,
        segments=segments,
        source="subtitles",
        translated_from=transcript.language_code,
    )
    return english, sum(t for _, t in results)
