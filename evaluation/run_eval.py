"""
Run the research comparison.

    python -m evaluation.run_eval                                   # solver only (free, offline)
    python -m evaluation.run_eval --modes solver,llm_only,llm_verified --repeats 3
    python -m evaluation.run_eval --modes llm_only --scenarios A96-3sem,B60-6sem

LLM modes call the LLM in .env (local Ollama is free and unlimited but slow;
Gemini free tier has daily limits; Claude is paid). The script prints how many
scenario runs it will make and asks before starting (skip with --yes).
Compare models by changing EDUPATH_OLLAMA_MODEL between runs.

Output: evaluation/results/<timestamp>/runs.jsonl, summary.json, summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from edupath.agent import MODES, EduPathAgent  # noqa: E402
from edupath.curriculum import load_curriculum  # noqa: E402
from edupath.llm import make_planner  # noqa: E402
from edupath.schemas import LLMError  # noqa: E402
from edupath.models import PlanningGoals  # noqa: E402
from edupath.solver import feasibility, select_courses  # noqa: E402
from edupath.students import STUDENTS_DIR, load_student  # noqa: E402
from evaluation.metrics import plan_metrics, summarize  # noqa: E402

SCENARIOS = Path(__file__).resolve().parent / "scenarios.json"
RESULTS = Path(__file__).resolve().parent / "results"


def run_one(agent: EduPathAgent, cur, sc: dict, mode: str) -> dict:
    student = load_student(STUDENTS_DIR / sc["student"])
    goals = PlanningGoals(
        target_semesters=sc["semesters"],
        start_term=sc.get("start", "Spring 2027"),
        max_credits_per_semester=sc["max_credits"],
        min_credits_per_semester=cur.policy.min_credits_per_semester,
        interests=sc.get("interests", []),
        avoid_courses=sc.get("avoid", []),
    )
    # Ground truth from the deterministic feasibility analysis.
    truth = feasibility(cur, student, goals, select_courses(cur, student, goals)).feasible
    row = {"scenario": sc["id"], "mode": mode, "truly_feasible": truth,
           "expected_feasible": sc.get("expected_feasible")}
    try:
        # Empty request text: goals are given explicitly, so we measure PLANNING, not parsing.
        res = agent.run(student, "", goals, mode=mode, explain=False)
    except LLMError as e:
        return {**row, "error": str(e)}

    said_feasible = res.feasibility.feasible if mode == "solver" else res.llm_claimed_feasible
    final = plan_metrics(res.outcomes if truth else [])
    first = plan_metrics(res.first_attempt) if res.first_attempt and truth else None
    row.update({
        **{k: v for k, v in final.items()},
        "first_attempt_valid_rate": first["plan_valid_rate"] if first else
        (final["plan_valid_rate"] if mode == "solver" else None),
        "first_attempt_errors": first["error_counts"] if first else None,
        "said_feasible": said_feasible,
        "feasibility_correct": said_feasible == truth,
        "false_feasible": bool(said_feasible) and not truth,
        "repair_rounds": res.repair_rounds,
        "llm_calls": res.llm_calls,
        "tokens": res.input_tokens + res.output_tokens,
        "seconds": res.seconds,
        "origins": sorted({o.origin for o in res.outcomes}),
    })
    return row


def write_summary(out_dir: Path, rows: list[dict]) -> str:
    summary = summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    cols = ["runs", "plan_valid_rate", "first_attempt_valid_rate", "prereq_violation_rate", "credit_ok_rate",
            "requirements_met_rate", "hallucinated_courses", "feasibility_accuracy", "false_feasible_rate",
            "avg_repair_rounds", "avg_llm_calls", "avg_tokens", "avg_seconds"]
    lines = ["| metric | " + " | ".join(summary) + " |", "|---|" + "---|" * len(summary)]
    for c in cols:
        lines.append(f"| {c} | " + " | ".join("-" if summary[m][c] is None else str(summary[m][c])
                                             for m in summary) + " |")
    md = "\n".join(lines)
    (out_dir / "summary.md").write_text(
        "# EduPath evaluation\n\nPlan-quality metrics use feasible scenarios only; "
        "feasibility metrics use all scenarios.\n\n" + md + "\n")
    return md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="solver", help=f"comma list from {list(MODES)}")
    ap.add_argument("--scenarios", default="", help="comma list of scenario ids (default: all)")
    ap.add_argument("--repeats", type=int, default=1, help="repeat LLM runs (they are non-deterministic)")
    ap.add_argument("--repair-rounds", type=int, default=2)
    ap.add_argument("--effort", default="high", help="LLM thinking effort for planning: low|medium|high")
    ap.add_argument("--yes", action="store_true", help="don't ask before spending API credits")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]
    scenarios = json.loads(SCENARIOS.read_text())["scenarios"]
    if args.scenarios:
        wanted = set(args.scenarios.split(","))
        scenarios = [s for s in scenarios if s["id"] in wanted]

    cur = load_curriculum()
    llm_modes = [m for m in modes if m != "solver"]
    planner = None
    if llm_modes:
        planner = make_planner(plan_effort=args.effort)
        if planner is None:
            sys.exit("LLM modes need an LLM: put EDUPATH_PROVIDER=ollama (free, local) in .env.")
        n = len(scenarios) * len(llm_modes) * args.repeats
        print(f"Using {planner.name}. This will make {n} LLM scenario runs "
              f"(each 1-{args.repair_rounds + 1} LLM calls).")
        if not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
            sys.exit("Cancelled.")

    # The research arm measures the LLM itself, so no silent solver fallback here.
    agent = EduPathAgent(cur, planner, max_repair_rounds=args.repair_rounds, solver_fallback=False)

    out_dir = RESULTS / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with (out_dir / "runs.jsonl").open("w") as f:
        for sc in scenarios:
            for mode in modes:
                for rep in range(args.repeats if mode != "solver" else 1):
                    row = run_one(agent, cur, sc, mode) | {"repeat": rep}
                    rows.append(row)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    validity = (f"valid={row.get('plan_valid_rate')}" if row["truly_feasible"]
                                else "infeasible goal")
                    status = row.get("error") or f"{validity}  feasibility_ok={row['feasibility_correct']}"
                    print(f"{sc['id']:<18} {mode:<13} rep{rep}  {status}")
                    if row.get("expected_feasible") is not None and row["expected_feasible"] != row["truly_feasible"]:
                        print(f"  ! scenario says expected_feasible={row['expected_feasible']} "
                              f"but the solver says {row['truly_feasible']}")

    print("\n" + write_summary(out_dir, rows))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
