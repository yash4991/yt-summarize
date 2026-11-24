"""Hindi / no-subtitle support: track choice, translation, AI transcription, and the Flow wiring."""

from types import SimpleNamespace

import pytest

from interview_prep import flow as flow_mod
from interview_prep.flow import process_video
from interview_prep.models import Segment, Transcript, VideoMeta
from interview_prep.transcribe import TranscriptionError, fix_offsets, transcribe_video
from interview_prep.translate import TranslationError, translate_transcript
from interview_prep.youtube import SubtitlesNotFound, choose_track, parse_marked_lines, split_blocks, timestamped_text


def track(code, generated):
    return SimpleNamespace(language_code=code, is_generated=generated)


# --------------------------------------------------------------------------- #
# Choosing a subtitle track
# --------------------------------------------------------------------------- #


def test_english_beats_hindi_and_manual_beats_auto():
    tracks = [track("hi", False), track("en", True), track("en", False)]
    assert choose_track(tracks) is tracks[2]
    assert choose_track([track("hi", False), track("en", True)]).language_code == "en"


def test_hindi_used_when_no_english():
    tracks = [track("ta", False), track("hi", True)]
    assert choose_track(tracks, ("hi",)).language_code == "hi"
    # Preference order follows the setting.
    assert choose_track([track("hi", True), track("mr", True)], ("mr", "hi")).language_code == "mr"


def test_any_language_as_last_resort_and_none_when_empty():
    assert choose_track([track("ta", True)], ("hi",)).language_code == "ta"
    assert choose_track([], ("hi",)) is None


def test_parse_marked_lines():
    segs = parse_marked_lines("[00:05] Hello there\ncontinued\n[1:02:03] Later\n[00:10]\nnoise")
    assert [(s.start, s.text) for s in segs] == [(5, "Hello there continued"), (3723, "Later noise")]


# --------------------------------------------------------------------------- #
# Translation
# --------------------------------------------------------------------------- #


@pytest.fixture
def hindi(meta) -> Transcript:
    segs = [Segment(start=i * 31, text=f"हिंदी वाक्य {i} रम रेप्स") for i in range(12)]
    return Transcript(video_id=meta.video_id, language_code="hi", is_generated=True, segments=segs)


def echo_translator(model, prompt):
    """Fake LLM: 'translates' each marked line, keeping its marker."""
    body = prompt.split("SUBTITLES:\n", 1)[1]
    markers = [ln.split("]", 1)[0] for ln in body.splitlines() if ln.startswith("[")]
    lines = [f"{m}] English text about RAM and replicas" for m in markers]
    return "\n".join(lines), 100


def test_translate_keeps_timestamps_and_marks_origin(hindi, meta):
    en, tokens = translate_transcript(hindi, meta, "gemini/x", chunk_chars=200, llm_call=echo_translator)
    assert en.language_code == "en" and en.translated_from == "hi" and en.is_english
    assert [s.start for s in en.segments] == [s.start for s in hindi.segments]
    assert all("RAM" in s.text for s in en.segments)
    parts = len(split_blocks(timestamped_text(hindi.segments), 200))
    assert parts > 1 and tokens == 100 * parts  # each part billed once
    assert "auto-generated Hindi subtitles, translated to English by AI" == en.describe()


def test_translate_prompt_has_topic_context(hindi, meta):
    seen = []
    meta = meta.model_copy(update={"description": "Learn Redis caching for interviews"})
    translate_transcript(hindi, meta, "m", llm_call=lambda m, p: seen.append(p) or echo_translator(m, p))
    assert "Learn Redis caching" in seen[0] and meta.title in seen[0] and "Hindi" in seen[0]


def test_translate_retries_bad_format_then_fails(hindi, meta):
    calls = []

    def bad(model, prompt):
        calls.append(1)
        return "Sure! Here is the translation without any markers.", 5

    with pytest.raises(TranslationError):
        translate_transcript(hindi, meta, "m", chunk_chars=100_000, llm_call=bad)
    assert len(calls) == 2  # one retry


# --------------------------------------------------------------------------- #
# AI transcription (no subtitles)
# --------------------------------------------------------------------------- #


def test_fix_offsets_shifts_relative_times():
    rel = [Segment(start=5, text="a"), Segment(start=300, text="b")]
    assert [s.start for s in fix_offsets(rel, 900, 1800)] == [905, 1200]
    absolute = [Segment(start=905, text="a"), Segment(start=1500, text="b")]
    assert [s.start for s in fix_offsets(absolute, 900, 1800)] == [905, 1500]


def test_transcribe_in_windows(meta):
    meta = meta.model_copy(update={"duration_seconds": 2000})
    windows = []

    def fake_call(model, url, start, end, prompt):
        windows.append((model, start, end))
        assert url == meta.watch_url and "START OF THE FULL VIDEO" in prompt
        return f"[{start // 60:02d}:10] English part at {start}\n[{start // 60:02d}:40] [unclear]", 50

    t, tokens = transcribe_video(meta, "gemini/gemini-2.5-flash", window_seconds=900, call=fake_call)
    assert windows == [("gemini-2.5-flash", 0, 900), ("gemini-2.5-flash", 900, 1800), ("gemini-2.5-flash", 1800, 2000)]
    assert [s.start for s in t.segments] == [10, 910, 1810]  # [unclear] dropped
    assert t.source == "ai_transcription" and tokens == 150


def test_transcribe_needs_gemini_key(meta, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(TranscriptionError, match="GEMINI_API_KEY"):
        transcribe_video(meta)


# --------------------------------------------------------------------------- #
# Flow wiring
# --------------------------------------------------------------------------- #


@pytest.fixture
def youtube(monkeypatch, meta):
    """Fake YouTube whose subtitles the test controls."""
    state = {"transcript": None}
    monkeypatch.setattr(flow_mod, "fetch_metadata", lambda vid: meta)

    def fake_fetch(vid, **kw):
        if state["transcript"] is None:
            raise SubtitlesNotFound("none")
        return state["transcript"]

    monkeypatch.setattr(flow_mod, "fetch_transcript", fake_fetch)
    return state


def test_hindi_video_is_translated_once_and_noted(youtube, hindi, settings, fake_runner, monkeypatch):
    youtube["transcript"] = hindi
    calls = []

    def fake_translate(t, meta, model, chunk_chars, progress, **kw):
        calls.append(model)
        segs = [Segment(start=s.start, text="Clean English about replicas") for s in t.segments]
        return Transcript(video_id=t.video_id, language_code="en", is_generated=True, segments=segs,
                          translated_from="hi"), 777

    monkeypatch.setattr(flow_mod, "translate_transcript", fake_translate)
    r = process_video("abcdefghijk", settings, crew_runner=fake_runner)
    assert calls == [settings.model]
    assert "Clean English about replicas" in fake_runner.calls[0]
    assert "translated to English by AI" in fake_runner.notes[0]
    assert r.stats.transcript_prep_tokens == 777
    transcript_file = [f for f in r.files if f.endswith("Transcript (English).txt")]
    assert transcript_file and "Clean English" in open(transcript_file[0], encoding="utf-8").read()

    process_video("abcdefghijk", settings, crew_runner=fake_runner)
    assert calls == [settings.model]  # cached: not translated again


def test_english_video_skips_translation(youtube, transcript, settings, fake_runner, monkeypatch):
    youtube["transcript"] = transcript
    monkeypatch.setattr(flow_mod, "translate_transcript", lambda *a, **k: pytest.fail("should not translate"))
    process_video("abcdefghijk", settings, crew_runner=fake_runner)
    assert fake_runner.notes == [""]


def test_no_subtitles_uses_ai_transcription(youtube, settings, fake_runner, monkeypatch):
    def fake_transcribe(meta, model, window, progress):
        segs = [Segment(start=0, text="Transcribed English")]
        return Transcript(video_id=meta.video_id, language_code="en", is_generated=True, segments=segs,
                          source="ai_transcription"), 50

    monkeypatch.setattr(flow_mod, "transcribe_video", fake_transcribe)
    r = process_video("abcdefghijk", settings, crew_runner=fake_runner)
    assert "Transcribed English" in fake_runner.calls[0]
    assert "transcribed from the video's audio" in fake_runner.notes[0]
    assert r.stats.transcript_prep_tokens == 50


def test_no_subtitles_and_transcription_disabled(youtube, settings, fake_runner):
    settings.transcribe_fallback = False
    with pytest.raises(SubtitlesNotFound, match="TRANSCRIBE_FALLBACK"):
        process_video("abcdefghijk", settings, crew_runner=fake_runner)


def test_describe_variants():
    base = dict(video_id="x", segments=[])
    assert Transcript(language_code="en", is_generated=False, **base).describe() == "human-written English subtitles"
    assert "audio" in Transcript(language_code="en", is_generated=True, source="ai_transcription", **base).describe()
    assert VideoMeta(video_id="abcdefghijk", title="t").description == ""


# --------------------------------------------------------------------------- #
# Retries on temporary provider errors
# --------------------------------------------------------------------------- #


def test_retry_on_overload_then_success():
    from interview_prep.retry import with_retry

    attempts, waits = [], []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")
        return "ok"

    assert with_retry(flaky, "x", delays=(1, 2, 3), sleep=waits.append) == "ok"
    assert waits == [1, 2]


def test_no_retry_on_real_errors():
    from interview_prep.retry import with_retry

    waits = []
    with pytest.raises(ValueError):
        with_retry(lambda: (_ for _ in ()).throw(ValueError("invalid API key")), "x", sleep=waits.append)
    assert waits == []


def test_daily_quota_fails_fast_with_clear_message():
    from interview_prep.retry import QuotaExceeded, with_retry

    msg = ("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please "
           "check your plan and billing details. * Quota exceeded for metric: generate_content_free_tier_requests, "
           "limit: 20, model: gemini-2.5-flash, description: Generate content requests per day limit exceeded on "
           "Free Tier.\nPlease retry in 10h52m38.268503698s.'}}")
    waits = []
    with pytest.raises(QuotaExceeded) as info:
        with_retry(lambda: (_ for _ in ()).throw(RuntimeError(msg)), "x", sleep=waits.append)
    assert waits == []
    text = str(info.value)
    assert "20 requests/day for gemini-2.5-flash" in text and "10h 52m" in text and "{" not in text


def test_per_minute_limit_is_retried():
    from interview_prep.retry import is_transient, quota_error

    e = RuntimeError("429 RESOURCE_EXHAUSTED. You exceeded your current quota, please check your plan and billing "
                     "details. Quota exceeded for metric: generate_content_requests_per_minute. Please retry in 31.2s.")
    assert quota_error(e) is None and is_transient(e)
