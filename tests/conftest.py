from __future__ import annotations

import pytest

from interview_prep.config import Settings
from interview_prep.models import (
    DetailedSummary,
    InterviewQA,
    KeyPoint,
    QAItem,
    ReviewIssue,
    ReviewReport,
    RunStats,
    Section,
    Segment,
    ShortSummary,
    StudyPack,
    Transcript,
    VideoMeta,
)


@pytest.fixture
def meta() -> VideoMeta:
    return VideoMeta(video_id="abcdefghijk", title="Caching: Explained? <Part 1>", channel="Tech Channel",
                     duration_seconds=600, url="https://www.youtube.com/watch?v=abcdefghijk")


@pytest.fixture
def transcript() -> Transcript:
    segs = [Segment(start=i * 7.5, duration=7.0, text=f"sentence number {i} about caching") for i in range(40)]
    return Transcript(video_id="abcdefghijk", language_code="en", is_generated=False, segments=segs)


def make_pack(faithfulness: int = 9, major: bool = False) -> StudyPack:
    return StudyPack(
        short=ShortSummary(
            topic="Caching",
            what_it_is="Caching stores results so repeated reads are fast.",
            key_points=[KeyPoint(point="Cache hits avoid the database.", timestamp_seconds=65),
                        KeyPoint(point="Bad timestamp is dropped.", timestamp_seconds=99999),
                        KeyPoint(point="No timestamp at all.")],
            interview_one_liner="A cache trades memory and staleness for speed.",
        ),
        detailed=DetailedSummary(
            topic="Caching",
            overview="Para one.\n\nPara two.",
            sections=[Section(heading="Eviction", explanation="LRU evicts the least recently used item.",
                              bullets=[KeyPoint(point="LRU is common.", timestamp_seconds=120)])],
            examples=["CDN caching images"],
            trade_offs=["Staleness vs speed"],
            common_mistakes=["Forgetting invalidation"],
            beyond_the_video=["Cache stampede protection"],
            quick_recap=["Caches make reads fast"],
        ),
        qa=InterviewQA(questions=[
            QAItem(question="What is a hard cache problem?", difficulty="hard", model_answer="Invalidation.",
                   follow_up="How would you solve it?", timestamp_seconds=300),
            QAItem(question="What is a cache?", difficulty="easy", model_answer="Fast storage.", follow_up=""),
        ]),
        review=ReviewReport(
            faithfulness_score=faithfulness, coverage_score=8, clarity_score=9,
            issues=[ReviewIssue(severity="major" if major else "minor", location="short", problem="p", fix="f")],
            verdict="Solid.",
        ),
    )


@pytest.fixture
def pack() -> StudyPack:
    return make_pack()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(model="anthropic/test-model", output_dir=tmp_path / "out", cache_dir=tmp_path / "cache",
                    exports=("md",), notion_token=None, notion_parent_page_id=None)


@pytest.fixture
def fake_runner():
    """Stands in for the LLM crew. Records how often it was called."""
    calls = []

    def runner(settings, meta, llm_text, progress):
        calls.append(llm_text)
        return make_pack(), RunStats(model=settings.model, prompt_version="test", total_tokens=1234)

    runner.calls = calls
    return runner
