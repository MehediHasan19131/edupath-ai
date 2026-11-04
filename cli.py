"""
Command-line EduPath.

    python cli.py                                   # sample student, offline solver
    python cli.py --request "I want to graduate in 3 semesters and I like AI"
    python cli.py --student data/students/cse_60_credits.json --semesters 6
    python cli.py --mode llm_verified               # needs an LLM in .env (e.g. EDUPATH_PROVIDER=ollama)
"""

from __future__ import annotations

import argparse

from edupath.agent import MODES, EduPathAgent
from edupath.curriculum import load_curriculum
from edupath.llm import make_planner
from edupath.models import PlanningGoals
from edupath.students import load_student


def main() -> None:
    ap = argparse.ArgumentParser(description="EduPath AI - academic planning agent")
    ap.add_argument("--student", default="data/students/cse_96_credits.json")
    ap.add_argument("--request", default="I am a CSE student. I have completed 96 credits and want to "
                                         "graduate in three semesters. Which courses should I take?")
    ap.add_argument("--mode", choices=list(MODES), default="solver")
    ap.add_argument("--semesters", type=int, default=3, help="default if the request doesn't say")
    ap.add_argument("--start", default="Spring 2027", help='first planned term, e.g. "Spring 2027"')
    ap.add_argument("--max-credits", type=int, default=None)
    ap.add_argument("--offline", action="store_true", help="never call an LLM, even if a key is set")
    args = ap.parse_args()

    cur = load_curriculum()
    student = load_student(args.student)
    planner = make_planner("offline" if args.offline else None)
    agent = EduPathAgent(cur, planner)

    defaults = PlanningGoals(
        target_semesters=args.semesters,
        start_term=args.start,
        max_credits_per_semester=args.max_credits or cur.policy.max_credits_per_semester,
        min_credits_per_semester=cur.policy.min_credits_per_semester,
        interests=student.interests,
    )
    result = agent.run(student, args.request, defaults, mode=args.mode)

    print(f"\n=== EduPath AI ({MODES[args.mode]}) ===")
    print(f"Student: {student.name} ({student.student_id})   LLM: {planner.name if planner else 'off (offline)'}\n")
    for step in result.trace:
        print(f"  > {step}")
    print(f"\n{result.feasibility.message}")
    for note in result.notes:
        print(f"  NOTE: {note}")

    for o in result.outcomes:
        status = "VALID" if o.report.is_valid else f"INVALID ({len(o.report.errors)} errors)"
        print(f"\n--- {o.plan.name}  [{status}, from {o.origin}] ---")
        for sem, cr in zip(o.plan.semesters, o.report.semester_credits):
            print(f"  {sem.term:<12} {cr:>4g} cr  " + ", ".join(sem.courses))
            for m in sem.milestones:
                print(f"  {'':<12}          * {m}")
        for v in o.report.violations:
            print(f"  [{v.severity}] {v.message}")

    print("\n=== Explanation ===\n" + result.explanation)
    if result.llm_calls:
        print(f"\n({result.llm_calls} LLM calls, {result.input_tokens} in / {result.output_tokens} out tokens, "
              f"{result.seconds}s)")


if __name__ == "__main__":
    main()
