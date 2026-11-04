"""
The EduPath AGENT: the pipeline that answers a student's request.

    request text ──► interpret (LLM or offline parser) ──► PlanningGoals
                                                                  │
    student record ──► degree audit ──► feasibility check ◄───────┘
                                              │
                    ┌─────────────────────────┼──────────────────────────┐
             mode = "solver"           mode = "llm_verified"       mode = "llm_only"
         deterministic solver       LLM proposes ─► validator     LLM proposes
         builds the plans           ─► feedback ─► LLM repairs    (validator only
                                    (up to N rounds; solver plans    reports, never
                                     as a last-resort fallback)      corrects)
                    └─────────────────────────┼──────────────────────────┘
                                              ▼
                       every plan validated ─► explanation (LLM or template)

The three modes are the three arms of the research comparison.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

from .curriculum import Curriculum
from .schemas import LLMError, PlanProposals
from .models import Plan, PlanningGoals, Semester, StudentRecord, ValidationReport
from .prompts import build_planning_context, format_violations
from .request_parser import parse_request
from .requirements import DegreeAudit, audit
from .solver import Feasibility, feasibility, generate_plans, select_courses
from .validator import validate_plan

MODES = {
    "solver": "LLM understands the request, constraint solver builds the plans (recommended)",
    "llm_verified": "LLM builds the plans, validator checks them and sends errors back for repair",
    "llm_only": "LLM builds the plans with no verification (research baseline)",
}


@dataclass
class PlanOutcome:
    plan: Plan
    report: ValidationReport
    origin: str                              # "solver", "llm", "llm_repaired", "solver_fallback"


@dataclass
class AgentResult:
    mode: str
    goals: PlanningGoals
    audit: DegreeAudit
    feasibility: Feasibility
    outcomes: list[PlanOutcome]              # what the student sees
    first_attempt: list[PlanOutcome] = field(default_factory=list)   # LLM's plans before any repair
    explanation: str = ""
    trace: list[str] = field(default_factory=list)                    # step-by-step log
    notes: list[str] = field(default_factory=list)                    # warnings for the student
    repair_rounds: int = 0
    llm_claimed_feasible: bool | None = None
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


class EduPathAgent:
    def __init__(self, curriculum: Curriculum, planner=None,
                 max_repair_rounds: int = 2, solver_fallback: bool = True):
        self.cur = curriculum
        self.planner = planner               # GeminiPlanner / ClaudePlanner; None = offline
        self.max_repair_rounds = max_repair_rounds
        self.solver_fallback = solver_fallback

    # ------------------------------------------------------------------ #
    # Step 1: understand the request
    # ------------------------------------------------------------------ #

    def interpret(self, request_text: str, defaults: PlanningGoals,
                  trace: list[str]) -> PlanningGoals:
        if not request_text.strip():
            trace.append("No request text - using the settings from the form.")
            return defaults
        if self.planner:
            extracted = self.planner.interpret_request(request_text)
            trace.append(f"{self.planner.name} read the request: {extracted.summary}")
        else:
            extracted = parse_request(request_text)
            trace.append("Read the request with the offline parser (no API key).")

        goals = replace(defaults)
        if extracted.target_semesters:
            goals.target_semesters = extracted.target_semesters
        if extracted.max_credits_per_semester:
            goals.max_credits_per_semester = extracted.max_credits_per_semester
        goals.stated_credits_completed = extracted.stated_credits_completed
        goals.interests = sorted(set(defaults.interests) | {i.lower() for i in extracted.interests})
        goals.avoid_courses = sorted({self.cur.normalize(c) for c in defaults.avoid_courses + extracted.avoid_courses})
        goals.must_take_courses = sorted({self.cur.normalize(c)
                                          for c in defaults.must_take_courses + extracted.must_take_courses})
        trace.append(f"Goals: {goals.target_semesters} semester(s) from {goals.start_term}, "
                     f"max {goals.max_credits_per_semester} credits/semester, "
                     f"interests={goals.interests or '-'}, avoid={goals.avoid_courses or '-'}")
        return goals

    # ------------------------------------------------------------------ #
    # Step 2..: plan
    # ------------------------------------------------------------------ #

    def run(self, student: StudentRecord, request_text: str, defaults: PlanningGoals,
            mode: str = "solver", explain: bool = True) -> AgentResult:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        if mode != "solver" and not self.planner:
            raise LLMError(f"Mode '{mode}' needs an LLM. Put GEMINI_API_KEY (free) in .env or use mode 'solver'.")

        start = time.time()
        usage_before = (self.planner.usage.calls, self.planner.usage.input_tokens,
                        self.planner.usage.output_tokens) if self.planner else (0, 0, 0)
        trace: list[str] = []
        goals = self.interpret(request_text, defaults, trace)
        deg = audit(self.cur, student.completed)
        trace.append(f"Degree audit: {deg.credits_completed:g}/{deg.credits_required} credits, "
                     f"{len(deg.unmet)} requirement group(s) unmet.")
        feas = feasibility(self.cur, student, goals, select_courses(self.cur, student, goals))
        trace.append(feas.message)

        result = AgentResult(mode, goals, deg, feas, [], trace=trace, notes=self._notes(student, goals, deg))

        if mode == "solver":
            self._run_solver(student, goals, result)
        else:
            self._run_llm(student, goals, deg, result, verify=(mode == "llm_verified"))

        note = self._unverified_note(result.outcomes)
        if note:
            result.notes.append(note)
        if explain:
            result.explanation = self._explain(result)
        if self.planner:
            u = self.planner.usage
            result.llm_calls = u.calls - usage_before[0]
            result.input_tokens = u.input_tokens - usage_before[1]
            result.output_tokens = u.output_tokens - usage_before[2]
        result.seconds = round(time.time() - start, 2)
        return result

    def _run_solver(self, student, goals, result: AgentResult) -> None:
        solved = generate_plans(self.cur, student, goals)
        result.outcomes = [PlanOutcome(p, r, "solver") for p, r in solved.plans]
        result.trace.append(f"Constraint solver produced {len(result.outcomes)} valid plan(s).")
        if not result.outcomes and not result.feasibility.feasible:
            # Show the earliest possible graduation instead of nothing.
            longer = replace(goals, target_semesters=result.feasibility.min_semesters)
            alt = generate_plans(self.cur, student, longer)
            for p, _ in alt.plans:
                p.name += f" ({len(p.semesters)} semesters - earliest possible)"
            # Validate against the *new* goal so the student sees the plan is otherwise sound.
            result.outcomes = [PlanOutcome(p, validate_plan(self.cur, student, longer, p), "solver")
                               for p, _ in alt.plans]
            result.trace.append(f"Goal impossible; showing {len(result.outcomes)} plan(s) for the "
                                f"earliest possible graduation ({result.feasibility.min_semesters} semesters).")

    def _run_llm(self, student, goals, deg, result: AgentResult, verify: bool) -> None:
        brief = build_planning_context(self.cur, student, goals, deg)
        proposals, messages = self.planner.propose_plans(brief)
        result.llm_claimed_feasible = proposals.feasible
        outcomes = self._check(student, goals, proposals, "llm")
        result.first_attempt = outcomes
        result.trace.append(f"The LLM proposed {len(outcomes)} plan(s); "
                            f"{sum(o.report.is_valid for o in outcomes)} passed the validator.")

        if verify:
            while any(not o.report.is_valid for o in outcomes) and result.repair_rounds < self.max_repair_rounds:
                result.repair_rounds += 1
                feedback = format_violations([(o.plan.name, o.report) for o in outcomes])
                proposals, messages = self.planner.repair_plans(messages, feedback)
                result.llm_claimed_feasible = proposals.feasible
                outcomes = self._check(student, goals, proposals, "llm_repaired")
                result.trace.append(f"Repair round {result.repair_rounds}: "
                                    f"{sum(o.report.is_valid for o in outcomes)}/{len(outcomes)} plan(s) valid.")
            valid = [o for o in outcomes if o.report.is_valid]
            if valid:
                outcomes = valid
            elif self.solver_fallback:
                self._run_solver(student, goals, result)
                for o in result.outcomes:
                    o.origin = "solver_fallback"
                result.trace.append("The LLM could not produce a valid plan; using the solver's plans instead.")
                return
        result.outcomes = outcomes

    def _check(self, student, goals, proposals: PlanProposals, origin: str) -> list[PlanOutcome]:
        out = []
        for p in proposals.plans:
            plan = Plan(p.name,
                        [Semester(s.term, [self.cur.normalize(c) for c in s.courses], list(s.milestones))
                         for s in p.semesters],
                        "llm", p.rationale)
            out.append(PlanOutcome(plan, validate_plan(self.cur, student, goals, plan), origin))
        return out

    # ------------------------------------------------------------------ #
    # Notes and explanations
    # ------------------------------------------------------------------ #

    def _notes(self, student: StudentRecord, goals: PlanningGoals, deg: DegreeAudit) -> list[str]:
        notes = []
        if goals.stated_credits_completed is not None and goals.stated_credits_completed != deg.credits_completed:
            notes.append(f"You said {goals.stated_credits_completed} credits, but your record adds up to "
                         f"{deg.credits_completed:g}. Plans use your record - please check it.")
        required = {c for g in deg.groups if g.kind == "all" for c in g.missing}
        cannot_avoid = [c for c in goals.avoid_courses if c in required]
        if cannot_avoid:
            notes.append(f"{', '.join(cannot_avoid)} is required for the degree, so it can't be avoided.")
        if deg.unrecognized:
            notes.append(f"Not in the curriculum file (ignored): {', '.join(deg.unrecognized)}.")
        if goals.max_credits_per_semester > self.cur.policy.max_credits_per_semester:
            notes.append(f"{goals.max_credits_per_semester} credits/semester is above the normal "
                         f"{self.cur.policy.max_credits_per_semester}; it needs advisor/chairperson approval.")
        return notes

    def _unverified_note(self, outcomes: list[PlanOutcome]) -> str | None:
        planned = {c for o in outcomes for c in o.plan.all_courses()}
        unverified = sorted(c for c in planned if c in self.cur.courses and not self.cur.courses[c].verified)
        if not unverified:
            return None
        return ("Prerequisites/titles for these planned courses were assumed, not published - confirm "
                f"with your advisor: {', '.join(unverified)}.")

    def _explain(self, result: AgentResult) -> str:
        if self.planner:
            try:
                return self.planner.explain(self._facts(result))
            except LLMError as e:
                result.trace.append(f"LLM explanation failed ({e}); using the template.")
        return template_explanation(result)

    def _facts(self, result: AgentResult) -> str:
        lines = [f"Student goal: graduate in {result.goals.target_semesters} semester(s) starting "
                 f"{result.goals.start_term}; interests: {', '.join(result.goals.interests) or 'none'}.",
                 f"Feasibility: {result.feasibility.message}",
                 *[f"- {r}" for r in result.feasibility.reasons],
                 f"Notes: {' '.join(result.notes) or 'none'}", ""]
        for o in result.outcomes:
            lines.append(f"Plan '{o.plan.name}' (source: {o.origin}; valid: {o.report.is_valid}): {o.plan.rationale}")
            for s, cr in zip(o.plan.semesters, o.report.semester_credits):
                titles = ", ".join(f"{c} {self.cur.courses[c].title}" if c in self.cur.courses else c
                                   for c in s.courses)
                lines.append(f"  {s.term} ({cr:g} cr): {titles} {'; '.join(s.milestones)}")
            for v in o.report.violations:
                lines.append(f"  [{v.severity}] {v.message}")
        return "\n".join(lines)


def template_explanation(result: AgentResult) -> str:
    """Plain explanation without an LLM. (The feasibility headline and notes are
    shown separately by the app and CLI, so they are not repeated here.)"""
    f = result.feasibility
    parts = ["What sets the minimum number of semesters:", *[f"- {r}" for r in f.reasons], ""]
    if result.outcomes:
        parts.append(f"I found {len(result.outcomes)} plan(s):")
        for o in result.outcomes:
            credits = " / ".join(f"{c:g}" for c in o.report.semester_credits)
            parts.append(f"- **{o.plan.name}** ({credits} credits): {o.plan.rationale}")
    else:
        parts.append("No valid plan could be built. Try more semesters or a higher credit limit.")
    parts += ["", "Always confirm your final plan with your BRAC University advisor."]
    return "\n".join(parts)
