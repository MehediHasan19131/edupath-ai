"""
The deterministic PLANNER (constraint solver).

Two steps:
  1. select_courses()  - decide WHAT to take: every missing required course, plus
                         picks for "choose N" groups (electives ranked by interests).
  2. schedule()        - decide WHEN: fill semesters one by one, always placing the
                         courses on the longest remaining prerequisite chain first
                         ("critical path first" list scheduling), never breaking a rule.

generate_plans() runs these with a few strategies to give the student real
alternatives, then double-checks every plan with the validator.

feasibility() explains, before planning, the minimum number of semesters the
student needs and why - this is what lets the agent say "3 semesters is not
possible, here is the earliest you can graduate".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .curriculum import Curriculum, term_sequence
from .models import THESIS_SEMESTERS_LEFT, Plan, PlanningGoals, Semester, StudentRecord, ValidationReport
from .requirements import DegreeAudit, audit
from .validator import validate_plan

# --------------------------------------------------------------------------- #
# 1. What to take
# --------------------------------------------------------------------------- #


def select_courses(curriculum: Curriculum, student: StudentRecord, goals: PlanningGoals,
                   alternative: int = 0) -> list[str]:
    """Return the course codes the student still has to take.

    `alternative` > 0 picks the next-best electives instead of the best ones,
    so we can offer a plan with a different flavour.
    """
    completed = {curriculum.normalize(c) for c in student.completed}
    deg = audit(curriculum, sorted(completed))
    avoid = {curriculum.normalize(c) for c in goals.avoid_courses}
    interests = {i.lower() for i in goals.interests + student.interests}

    chosen: list[str] = []
    for g in deg.groups:
        if g.kind == "all":
            chosen += [c for c in g.missing if c not in chosen]

    for code in goals.must_take_courses:
        code = curriculum.normalize(code)
        if code in curriculum.courses and code not in completed and code not in chosen:
            chosen.append(code)

    # Each planned course may fill only one group, exactly like in the audit.
    assigned = {c for g in deg.groups if g.kind == "all" for c in g.missing}
    for g in deg.groups:
        if g.kind != "choose" or g.picks_needed == 0:
            continue
        already_counted = [c for c in chosen if c in g.options and c not in assigned][: g.picks_needed]
        assigned |= set(already_counted)
        needed = g.picks_needed - len(already_counted)
        from_min_needed = max(0, g.min_from_needed - sum(c in g.min_from for c in already_counted))

        def reachable(code: str) -> bool:
            """Can we actually take it? All its prerequisites are passed or already planned."""
            return all(p in completed or p in chosen for p in curriculum.all_prereqs(code))

        def score(code: str):
            # Lower sorts first: matches interests, then "preferred" options (e.g. CSE electives).
            c = curriculum.courses[code]
            return (-len(interests & set(c.tags)), code not in g.min_from, code)

        pool = sorted((c for c in g.options
                       if c not in completed and c not in chosen and c not in avoid and reachable(c)),
                      key=score)
        picks: list[str] = []
        strict = [c for c in pool if c in g.min_from]
        picks += _rotate(strict, alternative)[:min(from_min_needed, needed)]
        rest = [c for c in pool if c not in picks]
        picks += _rotate(rest, alternative)[: max(0, needed - len(picks))]
        chosen += picks
        assigned |= set(picks)

    return chosen


def _rotate(items: list[str], k: int) -> list[str]:
    if not items:
        return items
    k %= len(items)
    return items[k:] + items[:k]


# --------------------------------------------------------------------------- #
# 2. When to take it
# --------------------------------------------------------------------------- #


def schedule(curriculum: Curriculum, student: StudentRecord, goals: PlanningGoals,
             to_take: list[str], strategy: str) -> tuple[list[Semester], list[str]]:
    """Place `to_take` into semesters. Returns (semesters, courses_left_over).

    strategy:
      "balanced"     - spread credits evenly over the target semesters
      "front_loaded" - take as much as allowed early; lighter final semester
    """
    policy = curriculum.policy
    thesis = policy.thesis
    completed = {curriculum.normalize(c) for c in student.completed}
    done = set(completed)
    credits_done = curriculum.credits(completed)
    remaining = list(to_take)
    chain = curriculum.chain_lengths(set(to_take))
    unlocks = {c: sum(c in curriculum.courses[o].hard_prereqs + curriculum.courses[o].soft_prereqs
                      for o in to_take) for c in to_take}
    thesis_left = THESIS_SEMESTERS_LEFT[student.thesis_stage]
    terms = term_sequence(goals.start_term, goals.target_semesters, policy.terms)
    credits_at_start: list[float] = []
    semesters: list[Semester] = []

    for i, term in enumerate(terms):
        if not remaining:
            break
        credits_at_start.append(credits_done)
        semesters_left = len(terms) - i
        left_credits = curriculum.credits(remaining)
        max_c = goals.max_credits_per_semester

        if strategy == "front_loaded":
            target = max_c
        else:
            target = min(max_c, math.ceil(left_credits / semesters_left))
        # Never leave more than the future semesters can hold.
        must_take_now = left_credits - (semesters_left - 1) * max_c

        def eligible(code: str) -> bool:
            c = curriculum.courses[code]
            if not all(p in done for p in c.hard_prereqs):
                return False
            if not c.is_offered_in(term) or credits_done < c.min_credits_completed:
                return False
            if code == thesis.course:
                # Defend in the final semester: the last term, or earlier if everything left fits now.
                is_final = semesters_left == 1 or left_credits <= max_c
                return is_final and _thesis_ok(i, thesis_left, credits_at_start, thesis.min_credits_to_start)
            return True

        # Critical-path first, then courses that unlock others, then alphabetical.
        candidates = sorted((c for c in remaining if eligible(c)),
                            key=lambda c: (-chain[c], c != thesis.course, -unlocks[c], c))
        picked: list[str] = []
        load = 0.0
        for code in candidates:
            cr = curriculum.courses[code].credits
            if load + cr > max_c:
                continue
            critical = chain[code] >= semesters_left
            # Soft prerequisite still to come? Wait for it, unless there is no time to wait.
            soft_pending = any(p in remaining and p not in picked
                               for p in curriculum.courses[code].soft_prereqs)
            if soft_pending and not critical:
                continue
            if load + cr <= target or critical or load < must_take_now:
                picked.append(code)
                load += cr

        semesters.append(Semester(term, picked))
        for code in picked:
            remaining.remove(code)
        done |= set(picked)
        credits_done += load

    add_thesis_milestones(semesters, thesis_left, thesis.course)
    return semesters, remaining


def _thesis_ok(i: int, thesis_left: int, credits_at_start: list[float], min_to_start: int) -> bool:
    if i < thesis_left - 1:
        return False
    if thesis_left == 3:
        return credits_at_start[i - 2] >= min_to_start
    return True


def add_thesis_milestones(semesters: list[Semester], thesis_left: int, thesis_code: str) -> None:
    """Label the P1 / P2 / CSE400 semesters, counting back from where CSE400 landed."""
    idx = next((i for i, s in enumerate(semesters) if thesis_code in s.courses), None)
    if idx is None:
        return
    stages = ["Pre-thesis 1 (P1)", "Pre-thesis 2 (P2)", "Final thesis defense (CSE400)"][3 - thesis_left:]
    for k, stage in enumerate(stages):
        j = idx - (len(stages) - 1) + k
        if 0 <= j < len(semesters):
            semesters[j].milestones.append(f"Thesis: {stage}")


# --------------------------------------------------------------------------- #
# 3. Feasibility analysis
# --------------------------------------------------------------------------- #


@dataclass
class Feasibility:
    credits_completed: float
    credits_to_take: float
    min_semesters: int
    reasons: list[str] = field(default_factory=list)   # what drives the minimum
    feasible: bool = True
    message: str = ""


def feasibility(curriculum: Curriculum, student: StudentRecord, goals: PlanningGoals,
                to_take: list[str]) -> Feasibility:
    completed = {curriculum.normalize(c) for c in student.completed}
    done_credits = curriculum.credits(completed)
    need = curriculum.credits(to_take)
    max_c = goals.max_credits_per_semester

    by_credits = math.ceil(need / max_c) if need else 0
    chain = curriculum.chain_lengths(set(to_take))
    by_chain = max(chain.values(), default=0)
    longest = max(chain, key=chain.get) if chain else None

    by_thesis = 0
    thesis = curriculum.policy.thesis
    if thesis.course in to_take:
        left = THESIS_SEMESTERS_LEFT[student.thesis_stage]
        wait = 0
        if left == 3 and done_credits < thesis.min_credits_to_start:
            wait = math.ceil((thesis.min_credits_to_start - done_credits) / max_c)
        by_thesis = wait + left

    lower = max(by_credits, by_chain, by_thesis)
    reasons = [
        f"{need:g} credits left at {max_c} credits/semester -> at least {by_credits} semester(s)",
        f"Longest prerequisite chain starts at {longest} -> at least {by_chain} semester(s)" if longest else "",
        f"Thesis track ({student.thesis_stage.replace('_', ' ')}) -> at least {by_thesis} semester(s)"
        if by_thesis else "",
    ]
    reasons = [r for r in reasons if r]
    feasible = lower <= goals.target_semesters
    msg = (f"Graduating in {goals.target_semesters} semester(s) looks possible (minimum is {lower})."
           if feasible else
           f"Graduating in {goals.target_semesters} semester(s) is NOT possible under these rules. "
           f"The earliest is {lower} semester(s).")
    return Feasibility(done_credits, need, lower, reasons, feasible, msg)


# --------------------------------------------------------------------------- #
# 4. Put it together
# --------------------------------------------------------------------------- #


@dataclass
class SolverResult:
    plans: list[tuple[Plan, ValidationReport]]
    feasibility: Feasibility
    audit: DegreeAudit


STRATEGIES = [
    # (plan name, scheduling strategy, elective alternative index)
    ("Balanced load", "balanced", 0),
    ("Front-loaded (lighter final semester)", "front_loaded", 0),
    ("Balanced, alternative electives", "balanced", 1),
]


def generate_plans(curriculum: Curriculum, student: StudentRecord, goals: PlanningGoals) -> SolverResult:
    deg = audit(curriculum, student.completed)
    base = select_courses(curriculum, student, goals)
    feas = feasibility(curriculum, student, goals, base)

    plans: list[tuple[Plan, ValidationReport]] = []
    seen: set[tuple] = set()
    for name, strategy, alt in STRATEGIES:
        to_take = select_courses(curriculum, student, goals, alternative=alt)
        semesters, left = schedule(curriculum, student, goals, to_take, strategy)
        if left:
            continue                         # couldn't fit everything in the target
        key = tuple(tuple(sorted(s.courses)) for s in semesters)
        if key in seen:
            continue
        seen.add(key)
        plan = Plan(name, semesters, "solver", _rationale(strategy, alt))
        report = validate_plan(curriculum, student, goals, plan)
        if report.is_valid:                  # belt and braces: the solver must never ship an invalid plan
            plans.append((plan, report))

    if not plans and feas.feasible:
        feas.feasible = False
        feas.message = ("No valid plan found within the target, even though simple bounds allow it. "
                        "Course offerings or the thesis timeline probably block it - try one more semester.")
    return SolverResult(plans, feas, deg)


def _rationale(strategy: str, alt: int) -> str:
    text = {
        "balanced": "Spreads credits evenly so no semester is overloaded.",
        "front_loaded": "Takes the maximum load early, leaving a lighter final semester for the thesis defense.",
    }[strategy]
    if alt:
        text += " Uses different elective choices than the first plan."
    return text + " Courses on the longest prerequisite chain are scheduled first."
