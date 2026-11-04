"""
The shapes an LLM must reply in, shared by every provider (Gemini, Claude, ...).

Each provider asks its model for "structured output" matching these Pydantic
classes, so the reply always parses - we never scrape JSON out of prose.

Any planner class (GeminiPlanner, ClaudePlanner, or one you add) must offer:
    name                                  e.g. "Gemini (gemini-3.8-flash)"
    usage                                 a Usage object it keeps updated
    interpret_request(text)            -> GoalsExtraction
    propose_plans(brief)               -> (PlanProposals, conversation)
    repair_plans(conversation, feedback) -> (PlanProposals, conversation)
    explain(facts)                     -> str
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


class LLMError(RuntimeError):
    """Any problem talking to an LLM, with a message a student can act on."""


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class GoalsExtraction(BaseModel):
    target_semesters: int | None = Field(description="Semesters the student wants to finish in")
    max_credits_per_semester: int | None = Field(description="Max credits per semester, if stated")
    stated_credits_completed: int | None = Field(description="Credits the student says they have done")
    interests: list[str] = Field(description="Lowercase interest tags, e.g. ai, web, security")
    avoid_courses: list[str] = Field(description="Course codes the student wants to avoid")
    must_take_courses: list[str] = Field(description="Course codes the student insists on taking")
    summary: str = Field(description="One sentence restating the request")


class ProposedSemester(BaseModel):
    term: str = Field(description='e.g. "Spring 2027"')
    courses: list[str] = Field(description="Course codes, e.g. CSE420")
    milestones: list[str] = Field(description='e.g. "Thesis: Pre-thesis 1 (P1)"')


class ProposedPlan(BaseModel):
    name: str
    semesters: list[ProposedSemester]
    rationale: str


class PlanProposals(BaseModel):
    feasible: bool = Field(description="Can the goal be met under every rule?")
    feasibility_note: str
    plans: list[ProposedPlan]
