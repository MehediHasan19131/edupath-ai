"""
The constraint VALIDATOR - the heart of the research idea.

It takes ANY plan (written by the solver, by Claude, or by hand) and checks it
against hard rules. It never guesses and never calls an LLM, so its verdict is
reproducible. Every problem becomes a `Violation` with a code, so the
evaluation can count them.

Rule codes (errors make a plan invalid, warnings do not):

  error    UNKNOWN_COURSE         course code is not in the curriculum (hallucinated?)
  error    ALREADY_COMPLETED      student already passed it
  error    DUPLICATE_COURSE       same course planned twice
  error    PREREQ_VIOLATION       a hard prerequisite isn't passed in an EARLIER semester
  error    CREDIT_OVERLOAD        semester above the max credit limit
  error    NOT_OFFERED            course not offered in that term
  error    MIN_CREDITS_NOT_MET    e.g. CSE400 before 75 credits
  error    THESIS_TIMELINE        CSE400 too early for the P1 -> P2 -> CSE400 sequence
  error    SEMESTER_LIMIT         plan uses more semesters than the student allowed
  error    REQUIREMENTS_UNMET     degree still incomplete after the plan
  warning  SOFT_PREREQ            soft prerequisite neither done earlier nor taken alongside
  warning  CREDIT_UNDERLOAD       below the minimum load (final semester is exempt)
  warning  AVOIDED_COURSE         student asked to avoid this course
  warning  NOT_COUNTED            course does not count toward any remaining requirement
"""

from __future__ import annotations

from .curriculum import Curriculum, term_sequence
from .models import (THESIS_SEMESTERS_LEFT, Plan, PlanningGoals, StudentRecord, ValidationReport,
                     Violation)
from .requirements import audit


def validate_plan(curriculum: Curriculum, student: StudentRecord, goals: PlanningGoals,
                  plan: Plan) -> ValidationReport:
    v: list[Violation] = []
    policy = curriculum.policy

    completed = {curriculum.normalize(c) for c in student.completed}
    done_before = set(completed)            # grows semester by semester
    credits_before = curriculum.credits(completed)
    seen_in_plan: set[str] = set()
    semester_credits: list[float] = []

    # Terms are assigned by position, so a wrongly-labelled term from an LLM can't confuse us.
    terms = term_sequence(goals.start_term, max(len(plan.semesters), 1), policy.terms)

    if len(plan.semesters) > goals.target_semesters:
        v.append(Violation("SEMESTER_LIMIT", "error",
                           f"Plan uses {len(plan.semesters)} semesters but the goal is "
                           f"{goals.target_semesters}."))

    thesis_code = policy.thesis.course
    thesis_left = THESIS_SEMESTERS_LEFT[student.thesis_stage]
    credits_at_start: list[float] = []      # credits completed before each semester

    for i, sem in enumerate(plan.semesters):
        term = terms[i]
        credits_at_start.append(credits_before)
        codes = [curriculum.normalize(c) for c in sem.courses]
        this_sem = set(codes)
        load = 0.0

        for code in codes:
            course = curriculum.courses.get(code)
            if course is None:
                v.append(Violation("UNKNOWN_COURSE", "error",
                                   f"{code} is not a course in the {curriculum.program} curriculum.",
                                   term, code))
                continue
            load += course.credits

            if code in completed:
                v.append(Violation("ALREADY_COMPLETED", "error",
                                   f"{code} is already completed.", term, code))
            if code in seen_in_plan:
                v.append(Violation("DUPLICATE_COURSE", "error",
                                   f"{code} appears more than once in the plan.", term, code))
            seen_in_plan.add(code)

            missing_hard = [p for p in course.hard_prereqs if p not in done_before]
            if missing_hard:
                same_sem = [p for p in missing_hard if p in this_sem]
                detail = (f" ({', '.join(same_sem)} is in the same semester - hard prerequisites "
                          f"must be passed first)") if same_sem else ""
                v.append(Violation("PREREQ_VIOLATION", "error",
                                   f"{code} needs {', '.join(missing_hard)} passed before {term}{detail}.",
                                   term, code))

            missing_soft = [p for p in course.soft_prereqs
                            if p not in done_before and p not in this_sem]
            if missing_soft:
                v.append(Violation("SOFT_PREREQ", "warning",
                                   f"{code} has soft prerequisite {', '.join(missing_soft)}; "
                                   f"take it earlier or in the same semester.", term, code))

            if not course.is_offered_in(term):
                v.append(Violation("NOT_OFFERED", "error",
                                   f"{code} is not offered in {term.split()[0]}.", term, code))

            if credits_before < course.min_credits_completed:
                v.append(Violation("MIN_CREDITS_NOT_MET", "error",
                                   f"{code} needs {course.min_credits_completed} completed credits; "
                                   f"you will have {credits_before:g} by {term}.", term, code))

            if code in goals.avoid_courses:
                v.append(Violation("AVOIDED_COURSE", "warning",
                                   f"You asked to avoid {code}.", term, code))

            if code == thesis_code:
                v.extend(_check_thesis(i, term, thesis_left, credits_at_start,
                                       policy.thesis.min_credits_to_start, completed, curriculum))

        semester_credits.append(load)
        is_last = i == len(plan.semesters) - 1
        if load > goals.max_credits_per_semester:
            v.append(Violation("CREDIT_OVERLOAD", "error",
                               f"{term} has {load:g} credits (max {goals.max_credits_per_semester}).", term))
        elif 0 < load < goals.min_credits_per_semester and not is_last:
            v.append(Violation("CREDIT_UNDERLOAD", "warning",
                               f"{term} has only {load:g} credits (min {goals.min_credits_per_semester}).",
                               term))

        # Only courses that exist count as "passed" for later semesters.
        passed_now = {c for c in codes if c in curriculum.courses}
        done_before |= passed_now
        credits_before += curriculum.credits(passed_now - completed)

    # ---- after the whole plan: is the degree complete? -------------------- #
    final = audit(curriculum, sorted(done_before))
    for code in final.not_counted:
        if code in seen_in_plan:
            v.append(Violation("NOT_COUNTED", "warning",
                               f"{code} doesn't count toward any remaining requirement.", course=code))
    remaining = [g.name for g in final.unmet]
    if remaining:
        details = []
        for g in final.unmet:
            details.append(f"{g.name}: missing {', '.join(g.missing)}" if g.missing
                           else f"{g.name}: {g.picks_needed} more course(s) needed")
        v.append(Violation("REQUIREMENTS_UNMET", "error",
                           "Degree not complete after this plan - " + "; ".join(details)))

    return ValidationReport(v, semester_credits, remaining)


def _check_thesis(i: int, term: str, thesis_left: int, credits_at_start: list[float],
                  min_to_start: int, completed: set[str], curriculum: Curriculum) -> list[Violation]:
    """CSE400 is the LAST of three thesis semesters (P1, P2, CSE400).

    If the student has not started P1, CSE400 can be no earlier than the 3rd
    planned semester, and P1 (two semesters before) needs 75+ credits.
    """
    out: list[Violation] = []
    earliest = thesis_left - 1                  # 0-based index of the earliest CSE400 semester
    if i < earliest:
        out.append(Violation(
            "THESIS_TIMELINE", "error",
            f"CSE400 in {term} is too early: you still need {thesis_left - 1} pre-thesis "
            f"semester(s) first, so CSE400 can be in planned semester {earliest + 1} at the earliest.",
            term, "CSE400"))
    elif thesis_left == 3:
        p1_index = i - 2
        credits_at_p1 = credits_at_start[p1_index]
        if credits_at_p1 < min_to_start:
            out.append(Violation(
                "THESIS_TIMELINE", "error",
                f"Pre-thesis 1 would start in planned semester {p1_index + 1} with only "
                f"{credits_at_p1:g} credits (needs {min_to_start}).", term, "CSE400"))
    return out
