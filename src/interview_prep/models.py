"""Data models shared across the pipeline.

Two groups live here:

* Plain data about a video (metadata, transcript segments).
* The structured outputs the LLM agents must return. Using Pydantic models
  (via ``output_pydantic``) means the formatting of the final files is done by
  our own renderer, so every summary has the same, predictable layout.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Video data
# --------------------------------------------------------------------------- #


class VideoMeta(BaseModel):
    video_id: str
    title: str
    channel: str = ""
    duration_seconds: int = 0
    url: str = ""
    upload_date: str = ""

    @property
    def watch_url(self) -> str:
        return self.url or f"https://www.youtube.com/watch?v={self.video_id}"


class Segment(BaseModel):
    """One caption line from YouTube."""

    start: float
    duration: float = 0.0
    text: str


class Transcript(BaseModel):
    video_id: str
    language_code: str
    is_generated: bool
    segments: list[Segment]

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments)


# --------------------------------------------------------------------------- #
# LLM outputs
# --------------------------------------------------------------------------- #


class KeyPoint(BaseModel):
    point: str = Field(description="One clear, self-contained idea in 1-2 sentences.")
    timestamp_seconds: int | None = Field(
        default=None,
        description="Second in the video where this idea is explained, taken from the "
        "[mm:ss] markers in the transcript. Null if it is not tied to one moment.",
    )


class ShortSummary(BaseModel):
    topic: str = Field(description="The interview topic this video covers, 2-6 words.")
    what_it_is: str = Field(description="A 2-3 sentence plain-English explanation.")
    key_points: list[KeyPoint] = Field(description="5-7 most important takeaways.")
    interview_one_liner: str = Field(
        description="One crisp sentence a candidate could say in an interview."
    )


class Section(BaseModel):
    heading: str
    explanation: str = Field(description="1-3 short paragraphs. Separate paragraphs with a blank line.")
    bullets: list[KeyPoint] = Field(default_factory=list)


class DetailedSummary(BaseModel):
    topic: str
    overview: str = Field(description="A 1-2 paragraph overview of the topic.")
    sections: list[Section] = Field(description="3-8 sections that build up the topic step by step.")
    examples: list[str] = Field(default_factory=list, description="Concrete examples or analogies.")
    trade_offs: list[str] = Field(default_factory=list, description="Pros/cons, when to use what.")
    common_mistakes: list[str] = Field(default_factory=list, description="Mistakes candidates make.")
    beyond_the_video: list[str] = Field(
        default_factory=list,
        description="Important related points interviewers ask about that the video does NOT cover. "
        "Each one must be clearly correct, general knowledge.",
    )
    quick_recap: list[str] = Field(description="5-8 bullet recap for last-minute revision.")


class QAItem(BaseModel):
    question: str
    difficulty: Literal["easy", "medium", "hard"]
    model_answer: str = Field(description="A strong answer in 3-6 sentences, as a candidate would say it.")
    follow_up: str = Field(default="", description="A likely follow-up question from the interviewer.")
    timestamp_seconds: int | None = Field(
        default=None, description="Where in the video the answer is covered, if anywhere."
    )


class InterviewQA(BaseModel):
    questions: list[QAItem] = Field(description="10-15 questions, mixed difficulty, easy first.")


class ReviewIssue(BaseModel):
    severity: Literal["minor", "major"]
    location: str = Field(description="Which document and section, e.g. 'detailed > Caching'.")
    problem: str
    fix: str


class ReviewReport(BaseModel):
    faithfulness_score: int = Field(
        ge=1, le=10, description="10 = every claim is supported by the transcript or is clearly correct."
    )
    coverage_score: int = Field(ge=1, le=10, description="10 = all important ideas of the video are covered.")
    clarity_score: int = Field(ge=1, le=10, description="10 = easy to digest for an interview candidate.")
    issues: list[ReviewIssue] = Field(default_factory=list)
    verdict: str = Field(description="One or two sentences summarising the review.")

    @property
    def overall(self) -> float:
        return round((self.faithfulness_score * 2 + self.coverage_score + self.clarity_score) / 4, 1)

    @property
    def has_major_issues(self) -> bool:
        return any(i.severity == "major" for i in self.issues)


class StudyPack(BaseModel):
    """Everything generated for one video."""

    short: ShortSummary
    detailed: DetailedSummary
    qa: InterviewQA
    review: ReviewReport
    revisions: int = 0


class RunStats(BaseModel):
    model: str
    prompt_version: str
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_requests: int = 0
    seconds: float = 0.0
    from_cache: bool = False


class VideoResult(BaseModel):
    meta: VideoMeta
    folder: str
    files: list[str]
    pack: StudyPack
    stats: RunStats
    notion_url: str | None = None
