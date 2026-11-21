"""The CrewAI crew: an Interview Coach writes the study pack, a Reviewer checks it."""

from __future__ import annotations

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from ..models import DetailedSummary, InterviewQA, ReviewReport, ShortSummary


@CrewBase
class SummaryCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self, model: str, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

    def _llm(self, temperature: float) -> LLM:
        return LLM(model=self.model, temperature=temperature, max_tokens=8000)

    @agent
    def interview_coach(self) -> Agent:
        return Agent(config=self.agents_config["interview_coach"], llm=self._llm(0.3), verbose=self.verbose)

    @agent
    def quality_reviewer(self) -> Agent:
        return Agent(config=self.agents_config["quality_reviewer"], llm=self._llm(0.0), verbose=self.verbose)

    @task
    def short_summary(self) -> Task:
        return Task(config=self.tasks_config["short_summary"], output_pydantic=ShortSummary)

    @task
    def detailed_summary(self) -> Task:
        return Task(config=self.tasks_config["detailed_summary"], output_pydantic=DetailedSummary)

    @task
    def interview_qa(self) -> Task:
        return Task(
            config=self.tasks_config["interview_qa"],
            output_pydantic=InterviewQA,
            context=[self.short_summary(), self.detailed_summary()],
        )

    @task
    def review(self) -> Task:
        return Task(
            config=self.tasks_config["review"],
            output_pydantic=ReviewReport,
            context=[self.short_summary(), self.detailed_summary(), self.interview_qa()],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, process=Process.sequential, verbose=self.verbose)


CONDENSE_PROMPT = """You are condensing part {part} of {parts} of a long video transcript so it can be \
summarised for job-interview preparation.

Rewrite this part as dense, factual notes. Keep every technical idea, definition, example, number \
and trade-off. Drop filler, jokes and repetition. KEEP the [mm:ss] timestamp marker at the start \
of each note line, copied from the transcript line the note came from.

TRANSCRIPT PART:
{chunk}
"""


def condense_chunk(model: str, chunk: str, part: int, parts: int) -> str:
    llm = LLM(model=model, temperature=0.0, max_tokens=8000)
    return llm.call([{"role": "user", "content": CONDENSE_PROMPT.format(part=part, parts=parts, chunk=chunk)}])
