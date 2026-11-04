"""
Every piece of text we send to Claude lives here, so you can tune prompts
without touching program logic.

build_planning_context() turns the curriculum + student into a compact text
brief. Both LLM modes (unverified and verified) get exactly the same brief, so
the only difference between them in the evaluation is the validator feedback.
"""

from __future__ import annotations

from .curriculum import Curriculum, term_sequence
from .models import THESIS_SEMESTERS_LEFT, PlanningGoals, StudentRecord, ValidationReport
from .requirements import DegreeAudit

INTERPRET_SYSTEM = """\
You read a university student's planning request and extract their goals as structured data.
Only fill a field when the student actually said it; otherwise use null (or an empty list).
Course codes look like CSE422 or MAT216 - write them without spaces.
Interests should be short lowercase tags such as: ai, data, web, security, networks, systems, \
hardware, robotics, graphics."""

PLANNER_SYSTEM = """\
You are EduPath, an academic planning assistant for BRAC University students.
You build semester-by-semester course plans that a real student could register for.
A plan is only useful if it obeys every rule in the brief: prerequisites, credit limits, the thesis \
timeline, and the full degree requirements. Use only course codes that appear in the brief, and \
never schedule a course the student has already completed.
When the student's goal is impossible under the rules, say so (feasible = false), explain the \
binding constraint in feasibility_note, and give the closest plan that does obey the rules, \
using more semesters if that is what it takes."""

EXPLAIN_SYSTEM = """\
You are EduPath, explaining course plans to a BRAC University student.
The plans you are given were checked by a deterministic rule validator; trust its results and do \
not invent courses, prerequisites or rules. Write for a student: short paragraphs and bullet \
points, plain language, no JSON. Cover: whether their goal is achievable and why, how the plans \
differ and who each suits, any prerequisite chains or risks to watch, and next steps (confirm \
with their advisor, especially anything marked as unverified data)."""


def build_planning_context(curriculum: Curriculum, student: StudentRecord, goals: PlanningGoals,
                           deg: DegreeAudit) -> str:
    p = curriculum.policy
    terms = term_sequence(goals.start_term, goals.target_semesters, p.terms)
    thesis_left = THESIS_SEMESTERS_LEFT[student.thesis_stage]

    lines: list[str] = []
    lines += [
        f"# Program: {curriculum.program} - {p.total_credits_required} credits total",
        "",
        "## Rules (all are hard constraints unless marked soft)",
        "- HP = hard prerequisite: must be passed in an EARLIER semester (not the same one).",
        "- SP = soft prerequisite: take it earlier or in the same semester (soft).",
        f"- Credits per semester: at most {goals.max_credits_per_semester}; "
        f"at least {goals.min_credits_per_semester} except the final semester (soft).",
        "- Each course at most once. Never repeat a completed course. Only use codes from this brief.",
        f"- {p.thesis.course} (4 credits) is the last of three thesis semesters: "
        f"{' -> '.join(p.thesis.stages)}. Pre-thesis 1 can start only after "
        f"{p.thesis.min_credits_to_start} completed credits. "
        f"This student has {thesis_left} thesis semester(s) left including the {p.thesis.course} "
        f"semester, so {p.thesis.course} can go no earlier than planned semester {thesis_left}.",
        f"- {p.thesis.course} also needs {curriculum.courses[p.thesis.course].min_credits_completed} "
        "completed credits before the semester it is taken.",
        f"- Use at most {goals.target_semesters} semester(s), in this order: {', '.join(terms)}.",
        "- After the plan, every requirement group below must be complete.",
        "",
        "## Course catalog (code | title | credits | HP | SP | tags)",
    ]
    for c in curriculum.courses.values():
        if c.category == "GenEd Option":         # listed under their requirement group instead
            continue
        lines.append(f"{c.code} | {c.title} | {c.credits:g} | {', '.join(c.hard_prereqs) or '-'} | "
                     f"{', '.join(c.soft_prereqs) or '-'} | {', '.join(c.tags) or '-'}")
    lines.append("GenEd option courses (3 credits each, no prerequisites): listed under their groups below.")

    completed = {curriculum.normalize(c) for c in student.completed}
    lines += ["", "## Requirement groups and what is still missing"]
    for g in deg.groups:
        if g.done:
            lines.append(f"- {g.name}: DONE")
        elif g.kind == "all":
            lines.append(f"- {g.name}: still need {', '.join(g.missing)}")
        else:
            options = [c for c in g.options if c not in completed]
            strict = [c for c in g.min_from if c not in completed]
            extra = f" (at least {g.min_from_needed} from: {', '.join(strict)})" if g.min_from_needed else ""
            lines.append(f"- {g.name}: pick {g.picks_needed} more from: {', '.join(options)}{extra}")

    lines += [
        "",
        "## Student",
        f"- Completed ({len(student.completed)} courses, {deg.credits_completed:g} credits): "
        f"{', '.join(sorted(curriculum.normalize(c) for c in student.completed))}",
        f"- Thesis stage: {student.thesis_stage.replace('_', ' ')}",
        f"- Interests: {', '.join(goals.interests + student.interests) or 'none given'}",
        f"- Wants to avoid: {', '.join(goals.avoid_courses) or 'nothing'}",
        f"- Must take: {', '.join(goals.must_take_courses) or 'nothing extra'}",
        "",
        "## Task",
        f"Propose up to 3 genuinely different plans (e.g. balanced load, front-loaded, different "
        f"electives) that finish the degree within {goals.target_semesters} semester(s). Prefer "
        "electives that match the student's interests. Put thesis stages in each semester's milestones.",
    ]
    return "\n".join(lines)


def format_violations(named_reports: list[tuple[str, ValidationReport]]) -> str:
    """Feedback message for the repair loop: exactly what the validator found."""
    out = ["A rule validator checked your plans and found problems. Fix every error and return "
           "the full set of plans again. Keep plans that had no errors unless you can improve them."]
    for name, report in named_reports:
        if report.is_valid:
            out.append(f"\nPlan '{name}': no errors.")
            continue
        out.append(f"\nPlan '{name}':")
        out += [f"- [{v.code}] {v.message}" for v in report.errors]
    return "\n".join(out)
