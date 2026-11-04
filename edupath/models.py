"""
Plain data classes used everywhere in EduPath.

Nothing in this file does any work - it only describes the *shape* of the data:
courses, requirement groups, a student's record, their goals, a semester plan,
and the violations the validator can report. Start reading here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# --------------------------------------------------------------------------- #
# Curriculum
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Course:
    code: str                                   # "CSE221"
    title: str                                  # "Algorithms"
    credits: float                              # 3 (CSE400 is 4)
    category: str                               # "Program Core", "GenEd", ...
    hard_prereqs: tuple[str, ...] = ()          # must be PASSED in an earlier semester
    soft_prereqs: tuple[str, ...] = ()          # should be done earlier OR taken in the same semester
    min_credits_completed: int = 0              # e.g. CSE400 needs 75 credits first
    offered: tuple[str, ...] = ()               # terms it runs in; empty = every term
    tags: tuple[str, ...] = ()                  # interest tags: "ai", "web", "security", ...
    aliases: tuple[str, ...] = ()               # equivalent codes, e.g. CSE161 -> CSE110
    verified: bool = True                       # False = prerequisite data was assumed, not published

    def is_offered_in(self, term: str) -> bool:
        """`term` is like "Fall 2027"; we only care about the season."""
        return not self.offered or term.split()[0] in self.offered


@dataclass
class RequirementGroup:
    """One block of the degree outline, e.g. "Program Core" or "GenEd Stream 3".

    kind == "all":    every course in `courses` is required.
    kind == "choose": any `count` courses from `courses`; optionally at least
                      `min_from_count` of them must come from `min_from`.
    """

    id: str
    name: str
    kind: Literal["all", "choose"]
    courses: list[str]
    count: int = 0
    min_from: list[str] = field(default_factory=list)
    min_from_count: int = 0


@dataclass
class ThesisPolicy:
    course: str
    stages: list[str]
    min_credits_to_start: int


@dataclass
class Policy:
    total_credits_required: int
    terms: list[str]
    min_credits_per_semester: int
    max_credits_per_semester: int
    thesis: ThesisPolicy


# --------------------------------------------------------------------------- #
# Student
# --------------------------------------------------------------------------- #

# How far along the 3-semester thesis track the student is.
ThesisStage = Literal["not_started", "p1_done", "p2_done"]

# Semesters of thesis work still needed *including* the CSE400 defense semester.
THESIS_SEMESTERS_LEFT: dict[str, int] = {"not_started": 3, "p1_done": 2, "p2_done": 1}


@dataclass
class StudentRecord:
    """What the student has already done. Entered by hand (see data/students/)."""

    name: str
    student_id: str
    program: str
    completed: list[str]                        # passed course codes
    thesis_stage: ThesisStage = "not_started"
    interests: list[str] = field(default_factory=list)


@dataclass
class PlanningGoals:
    """What the student wants. Filled from their natural-language request
    (by Claude, or by the offline parser) plus anything set in the UI."""

    target_semesters: int
    start_term: str                             # "Spring 2027"
    max_credits_per_semester: int
    min_credits_per_semester: int
    interests: list[str] = field(default_factory=list)
    avoid_courses: list[str] = field(default_factory=list)
    must_take_courses: list[str] = field(default_factory=list)
    stated_credits_completed: int | None = None  # what the student *said*; cross-checked


# --------------------------------------------------------------------------- #
# Plans and validation
# --------------------------------------------------------------------------- #


@dataclass
class Semester:
    term: str                                   # "Spring 2027"
    courses: list[str]
    milestones: list[str] = field(default_factory=list)  # e.g. "Start Pre-thesis 1"


@dataclass
class Plan:
    name: str
    semesters: list[Semester]
    source: Literal["solver", "llm"]
    rationale: str = ""

    def all_courses(self) -> list[str]:
        return [c for s in self.semesters for c in s.courses]


Severity = Literal["error", "warning"]


@dataclass
class Violation:
    code: str                                   # "PREREQ_VIOLATION", "CREDIT_OVERLOAD", ...
    severity: Severity
    message: str
    term: str | None = None
    course: str | None = None


@dataclass
class ValidationReport:
    violations: list[Violation]
    semester_credits: list[float]
    remaining_after_plan: list[str]             # requirement-group names still unmet

    @property
    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "error"]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return not self.errors
