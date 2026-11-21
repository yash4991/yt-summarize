"""End-to-end tests of the Flow with YouTube and the LLM replaced by fakes."""

import json

import pytest

from interview_prep import flow as flow_mod
from interview_prep import pipeline
from interview_prep.cache import Cache
from interview_prep.flow import process_video, run_crew
from interview_prep.models import VideoMeta

from .conftest import make_pack


@pytest.fixture
def offline(monkeypatch, meta, transcript):
    calls = {"meta": 0, "transcript": 0}

    def fake_meta(vid):
        calls["meta"] += 1
        return meta.model_copy(update={"video_id": vid})

    def fake_transcript(vid, **kw):
        calls["transcript"] += 1
        return transcript.model_copy(update={"video_id": vid})

    monkeypatch.setattr(flow_mod, "fetch_metadata", fake_meta)
    monkeypatch.setattr(flow_mod, "fetch_transcript", fake_transcript)
    return calls


def test_process_video_writes_files(offline, settings, fake_runner):
    r = process_video("https://youtu.be/abcdefghijk", settings, crew_runner=fake_runner)
    names = sorted(p.split("\\")[-1].split("/")[-1] for p in r.files)
    assert "Caching Explained Part 1 - Short Summary.txt" in names
    assert "Caching Explained Part 1 - Detailed Summary.txt" in names
    assert "Caching Explained Part 1 - Interview Q&A.txt" in names
    assert "Caching Explained Part 1 - Short Summary.md" in names
    md = json.loads((settings.output_dir / "Caching Explained Part 1" / "metadata.json").read_text(encoding="utf-8"))
    assert md["run"]["total_tokens"] == 1234
    assert md["quality"]["overall"] == r.pack.review.overall
    # The LLM saw timestamped text.
    assert fake_runner.calls[0].startswith("[00:00] ")


def test_cache_skips_youtube_and_llm(offline, settings, fake_runner):
    process_video("abcdefghijk", settings, crew_runner=fake_runner)
    r2 = process_video("abcdefghijk", settings, crew_runner=fake_runner)
    assert offline == {"meta": 1, "transcript": 1}
    assert len(fake_runner.calls) == 1
    assert r2.stats.from_cache
    process_video("abcdefghijk", settings, crew_runner=fake_runner, force=True)
    assert len(fake_runner.calls) == 2 and offline["transcript"] == 2


def test_corrupt_cache_is_ignored(offline, settings, fake_runner):
    process_video("abcdefghijk", settings, crew_runner=fake_runner)
    for f in (settings.cache_dir / "transcripts").glob("*.json"):
        f.write_bytes(b"\x00" * 100)
    assert Cache(settings.cache_dir).get_transcript("abcdefghijk") is None


def test_long_transcript_is_condensed(offline, settings, fake_runner, monkeypatch):
    import interview_prep.crew as crew_pkg

    seen = []
    monkeypatch.setattr(crew_pkg, "condense_chunk", lambda model, chunk, i, n: seen.append(i) or f"[00:00] notes {i}")
    settings.max_transcript_chars, settings.chunk_chars = 500, 300
    process_video("abcdefghijk", settings, crew_runner=fake_runner)
    assert len(seen) > 1
    assert fake_runner.calls[0].startswith("[00:00] notes 1")


def test_playlist_batch_with_failure(offline, settings, fake_runner, monkeypatch):
    ids = ["aaaaaaaaaaa", "bbbbbbbbbbb"]
    monkeypatch.setattr(pipeline, "list_playlist", lambda url, limit=None: ("My Course", ids))
    real = flow_mod.fetch_transcript

    def flaky(vid, **kw):
        if vid == "bbbbbbbbbbb":
            raise RuntimeError("no captions")
        return real(vid, **kw)

    monkeypatch.setattr(flow_mod, "fetch_transcript", flaky)
    rep = pipeline.run_batch(["https://www.youtube.com/playlist?list=PLx"], settings, crew_runner=fake_runner)
    assert len(rep.results) == 1 and len(rep.failures) == 1
    course = settings.output_dir / "My Course"
    assert (course / "Combined Study Guide.txt").exists()
    assert (course / "Combined Study Guide.md").exists()
    assert (course / "01 - Caching Explained Part 1").is_dir()


class _FakeOutput:
    def __init__(self, pack):
        parts = (pack.short, pack.detailed, pack.qa, pack.review)
        self.tasks_output = [type("T", (), {"pydantic": x})() for x in parts]
        self.token_usage = None


def test_revision_loop(monkeypatch, settings):
    """A poor first review triggers exactly one rewrite with the reviewer's notes."""
    import interview_prep.crew as crew_pkg

    outputs = [_FakeOutput(make_pack(faithfulness=5, major=True)), _FakeOutput(make_pack(faithfulness=9))]
    inputs_seen = []

    class FakeCrew:
        def __init__(self, model, verbose=False):
            pass

        def crew(self):
            return self

        def kickoff(self, inputs):
            inputs_seen.append(inputs)
            return outputs.pop(0)

    monkeypatch.setattr(crew_pkg, "SummaryCrew", FakeCrew)
    meta = VideoMeta(video_id="abcdefghijk", title="T")
    pack, _ = run_crew(settings, meta, "[00:00] text", lambda m: None)
    assert pack.revisions == 1 and pack.review.faithfulness_score == 9
    assert inputs_seen[0]["revision_notes"] == ""
    assert "reviewer" in inputs_seen[1]["revision_notes"]
