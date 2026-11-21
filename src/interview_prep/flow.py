"""The per-video CrewAI Flow.

    load_video ──► prepare_transcript ──► generate_pack ──► save_outputs
   (YouTube +       (English text:          (Coach + Reviewer   (txt / md / pdf /
    cache)           translate Hindi subs,   crew, revise once)   notion, metadata)
                     or Gemini transcribes
                     if no subtitles;
                     condense if long)

The LLM is used to get clean English text (only when the subtitles are not
English), to condense very long transcripts and to write the study pack.
Everything else is plain, deterministic Python.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel, PrivateAttr

from . import render
from .cache import Cache, dump_json
from .config import PROMPT_VERSION, Settings
from .exporters import export_notion, export_pdf
from .models import RunStats, StudyPack, Transcript, VideoMeta, VideoResult
from .retry import with_retry
from .storage import atomic_write_text, safe_name, video_folder
from .transcribe import transcribe_video
from .translate import translate_transcript
from .youtube import (
    SubtitlesNotFound,
    fetch_metadata,
    fetch_transcript,
    fmt_ts,
    parse_video_id,
    split_blocks,
    timestamped_text,
)

log = logging.getLogger(__name__)

Progress = Callable[[str], None]


class PrepState(BaseModel):
    url: str = ""
    force: bool = False
    output_parent: str = ""
    folder_prefix: str = ""
    meta: VideoMeta | None = None
    transcript: Transcript | None = None  # original subtitles (None if the video has none)
    english: Transcript | None = None  # what the agents read
    transcript_note: str = ""
    prep_tokens: int = 0
    llm_text: str = ""
    condensed: bool = False
    pack: StudyPack | None = None
    stats: RunStats | None = None
    result: VideoResult | None = None


class GenerationError(RuntimeError):
    pass


def run_crew(settings: Settings, meta: VideoMeta, llm_text: str, progress: Progress,
             transcript_note: str = "") -> tuple[StudyPack, RunStats]:
    """Run the Coach + Reviewer crew, with one revision round if the review is poor."""
    from .crew import SummaryCrew

    stats = RunStats(model=settings.model, prompt_version=PROMPT_VERSION)
    notes = ""
    pack: StudyPack | None = None
    for attempt in range(settings.max_revisions + 1):
        progress("Coach is writing the summaries and Q&A..." if attempt == 0
                 else "Coach is revising based on reviewer feedback...")
        inputs = {
            "title": meta.title,
            "channel": meta.channel or "unknown channel",
            "duration": fmt_ts(meta.duration_seconds) if meta.duration_seconds else "unknown",
            "transcript": llm_text,
            "revision_notes": notes,
            "transcript_note": transcript_note,
        }
        out = with_retry(
            lambda inputs=inputs: SummaryCrew(model=settings.model, verbose=settings.verbose)
            .crew().kickoff(inputs=inputs),
            "the summaries", progress)
        _add_usage(stats, out.token_usage)
        results = [t.pydantic for t in out.tasks_output]
        if len(results) != 4 or any(r is None for r in results):
            raise GenerationError("The model did not return all four structured outputs.")
        short, detailed, qa, review = results
        pack = StudyPack(short=short, detailed=detailed, qa=qa, review=review, revisions=attempt)
        progress(f"Reviewer scored it {review.overall}/10 (faithfulness {review.faithfulness_score}/10).")
        good = review.faithfulness_score >= settings.min_faithfulness and not review.has_major_issues
        if good or attempt == settings.max_revisions:
            break
        notes = (
            "IMPORTANT - a reviewer checked your previous draft and found these problems. "
            "Write a new version that fixes all of them:\n"
            + "\n".join(f"- [{i.severity}] {i.location}: {i.problem} Fix: {i.fix}" for i in review.issues)
        )
    assert pack is not None
    return pack, stats


def _add_usage(stats: RunStats, usage) -> None:
    if usage is None:
        return
    stats.total_tokens += usage.total_tokens or 0
    stats.prompt_tokens += usage.prompt_tokens or 0
    stats.completion_tokens += usage.completion_tokens or 0
    stats.llm_requests += usage.successful_requests or 0


class InterviewPrepFlow(Flow[PrepState]):
    # Note: function defaults on PrivateAttr get bound as methods, so these
    # default to None and are resolved in configure().
    _settings: Settings = PrivateAttr(default_factory=Settings)
    _progress: Progress | None = PrivateAttr(default=None)
    _crew_runner: Callable | None = PrivateAttr(default=None)  # swappable for tests

    def configure(self, settings: Settings, progress: Progress | None = None,
                  crew_runner: Callable | None = None) -> InterviewPrepFlow:
        self._settings = settings
        self._progress = progress or (lambda msg: None)
        self._crew_runner = crew_runner or run_crew
        return self

    @property
    def cache(self) -> Cache:
        return Cache(self._settings.cache_dir)

    # ------------------------------------------------------------------ #

    @start()
    def load_video(self) -> None:
        s = self.state
        video_id = parse_video_id(s.url)
        cached = None if s.force else self.cache.get_transcript(video_id)
        if cached:
            self._progress("Using cached video details and subtitles.")
            s.meta, s.transcript = cached.meta, cached.transcript
            return
        self._progress("Reading video details...")
        s.meta = fetch_metadata(video_id)
        self._progress(f"Downloading subtitles for “{s.meta.title}”...")
        try:
            s.transcript = fetch_transcript(video_id, source_languages=self._settings.subtitle_languages)
        except SubtitlesNotFound as e:
            self._progress(str(e))
            s.transcript = None
        self.cache.put_transcript(s.meta, s.transcript)

    @listen(load_video)
    def prepare_transcript(self) -> None:
        s, cfg = self.state, self._settings
        s.english = self._english_transcript()
        if s.english.translated_from or s.english.source == "ai_transcription":
            s.transcript_note = (
                f"NOTE: this transcript was produced from {s.english.describe()}. It was cleaned up, but a few "
                "words may still be wrong. Rely on the technical meaning that fits the topic, not on odd wording."
            )
        text = timestamped_text(s.english.segments)
        if len(text) > cfg.max_transcript_chars:
            from .crew import condense_chunk

            chunks = split_blocks(text, cfg.chunk_chars)
            notes = []
            for i, chunk in enumerate(chunks, 1):
                self._progress(f"Long video: condensing part {i}/{len(chunks)}...")
                notes.append(condense_chunk(cfg.model, chunk, i, len(chunks)))
            text, s.condensed = "\n".join(notes), True
        s.llm_text = text

    def _english_transcript(self) -> Transcript:
        """English subtitles as-is; otherwise translate them, or transcribe the audio if there are none."""
        s, cfg = self.state, self._settings
        if s.transcript is not None and s.transcript.is_english:
            return s.transcript
        if s.transcript is None and not cfg.transcribe_fallback:
            raise SubtitlesNotFound(
                f"“{s.meta.title}” has no subtitles. Enable TRANSCRIBE_FALLBACK to let Gemini transcribe it."
            )
        # Translations / transcriptions are cached per model, so they are only paid for once.
        model = cfg.model if s.transcript is not None else cfg.transcribe_model
        cached = None if s.force else self.cache.get_english(s.meta.video_id, model, PROMPT_VERSION)
        if cached:
            self._progress("Using cached English transcript.")
            return cached
        if s.transcript is not None:
            english, s.prep_tokens = translate_transcript(
                s.transcript, s.meta, cfg.model, chunk_chars=cfg.translate_chunk_chars, progress=self._progress)
        else:
            english, s.prep_tokens = transcribe_video(
                s.meta, cfg.transcribe_model, cfg.transcribe_window_seconds, progress=self._progress)
        self._progress(f"English transcript ready ({len(english.segments)} passages).")
        self.cache.put_english(s.meta.video_id, model, PROMPT_VERSION, english)
        return english

    @listen(prepare_transcript)
    def generate_pack(self) -> None:
        s, cfg = self.state, self._settings
        cached = None if s.force else self.cache.get_pack(s.meta.video_id, cfg.model, PROMPT_VERSION)
        if cached:
            self._progress("Using cached study pack (no LLM cost).")
            s.pack = cached.pack
            s.stats = cached.stats.model_copy(update={"from_cache": True, "seconds": 0.0})
            return
        t0 = time.perf_counter()
        s.pack, stats = self._crew_runner(cfg, s.meta, s.llm_text, self._progress,
                                          transcript_note=s.transcript_note)
        stats.seconds = round(time.perf_counter() - t0, 1)
        # The cache key must match the lookup above, whatever the runner reported.
        stats.model, stats.prompt_version = cfg.model, PROMPT_VERSION
        stats.transcript_prep_tokens = s.prep_tokens
        s.stats = stats
        self.cache.put_pack(s.meta.video_id, s.pack, stats)

    @listen(generate_pack)
    def save_outputs(self) -> VideoResult:
        s, cfg = self.state, self._settings
        meta, pack = s.meta, s.pack
        parent = Path(s.output_parent) if s.output_parent else cfg.output_dir
        folder = video_folder(parent, meta.title, meta.video_id, prefix=s.folder_prefix)
        name = safe_name(meta.title, max_len=60)
        self._progress(f"Saving files to {folder}...")

        files: list[Path] = []
        for kind, fn in render.DOCUMENTS.items():
            p = folder / f"{name} - {kind}.txt"
            atomic_write_text(p, fn(pack, meta, "txt"))
            files.append(p)
            if "md" in cfg.exports:
                p = folder / f"{name} - {kind}.md"
                atomic_write_text(p, fn(pack, meta, "md"))
                files.append(p)

        guide_md = render.render_study_guide(pack, meta)
        if "pdf" in cfg.exports:
            try:
                files.append(export_pdf(guide_md, folder / f"{name} - Study Guide.pdf", meta.title))
            except Exception as e:  # a PDF problem should never lose the text files
                log.exception("PDF export failed")
                self._progress(f"PDF export failed: {e}")

        notion_url = None
        if "notion" in cfg.exports:
            if cfg.notion_token and cfg.notion_parent_page_id:
                try:
                    self._progress("Publishing to Notion...")
                    notion_url = export_notion(guide_md, meta.title, cfg.notion_token, cfg.notion_parent_page_id)
                except Exception as e:
                    log.exception("Notion export failed")
                    self._progress(f"Notion export failed: {e}")
            else:
                self._progress("Notion export skipped: set NOTION_TOKEN and NOTION_PARENT_PAGE_ID.")

        transcript_path = folder / f"{name} - Transcript (English).txt"
        atomic_write_text(transcript_path, render.render_transcript(s.english, meta))
        files.append(transcript_path)

        review_path = folder / "Quality Review.md"
        atomic_write_text(review_path, render.render_review(pack, meta, "md"))
        files.append(review_path)

        result = VideoResult(meta=meta, folder=str(folder), files=[str(f) for f in files],
                             pack=pack, stats=s.stats, notion_url=notion_url)
        atomic_write_text(folder / "metadata.json", dump_json({
            "video": meta.model_dump(),
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "transcript": {
                "origin": s.english.describe(),
                "source": s.english.source,
                "original_language": s.english.translated_from or s.english.language_code
                if s.english.source == "subtitles" else None,
                "auto_generated": s.english.is_generated,
                "translated": bool(s.english.translated_from or s.english.source == "ai_transcription"),
                "condensed_for_length": s.condensed,
            },
            "run": s.stats.model_dump(),
            "quality": {**pack.review.model_dump(), "overall": pack.review.overall, "revisions": pack.revisions},
            "files": [Path(f).name for f in result.files],
            "notion_url": notion_url,
        }))
        s.result = result
        return result


def process_video(url: str, settings: Settings | None = None, *, force: bool = False,
                  progress: Progress | None = None, output_parent: Path | None = None,
                  folder_prefix: str = "", crew_runner: Callable | None = None) -> VideoResult:
    settings = settings or Settings()
    # Our own progress messages replace crewai's flow panels unless verbose is on.
    flow = InterviewPrepFlow(suppress_flow_events=not settings.verbose).configure(settings, progress, crew_runner)
    flow.kickoff(inputs={
        "url": url,
        "force": force,
        "output_parent": str(output_parent) if output_parent else "",
        "folder_prefix": folder_prefix,
    })
    if flow.state.result is None:
        raise GenerationError(f"Processing failed for {url}")
    return flow.state.result
