"""
Run the research comparison.

    python -m evaluation.run_eval                                   # solver only (free, offline)
    python -m evaluation.run_eval --modes solver,llm_only,llm_verified --repeats 3
    python -m evaluation.run_eval --modes llm_only --scenarios A96-3sem,B60-6sem
    python -m evaluation.run_eval --modes solver,llm_only,llm_verified --provider groq --model openai/gpt-oss-120b
    python -m evaluation.run_eval --resume evaluation/results/<folder>     # finish / retry a run

LLM modes call the LLM in .env, or the one given with --provider / --model:
  * ollama (local, free, unlimited, slow: ~15 hours for all 12 scenarios with qwen3:8b)
  * groq   (cloud, free, fast, but ~200K tokens/day per model - the full set needs ~2 days;
            when the daily limit is hit, the remaining runs fail fast - continue the next
            day with --resume)
The script prints how many scenario runs it will make and asks before starting
(skip with --yes). On a Mac it keeps the computer awake until it finishes.

--resume keeps the finished runs in that folder, re-runs only the missing or failed
ones (e.g. timeouts, rate limits) and rewrites the summary. Use the same --modes,
--scenarios, --repeats and model as the original run.

Failed runs count as failures in the summary (see evaluation/metrics.py).

Output: evaluation/results/<timestamp>-<model>/runs.jsonl, summary.json, summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
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
    start = time.time()
    try:
        # Empty request text: goals are given explicitly, so we measure PLANNING, not parsing.
        res = agent.run(student, "", goals, mode=mode, explain=False)
    except LLMError as e:
        return {**row, "error": str(e), "seconds_until_error": round(time.time() - start, 1)}

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


def write_summary(out_dir: Path, rows: list[dict], model: str) -> str:
    summary = summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps({"model": model, "modes": summary}, indent=2))
    cols = ["runs", "failed_runs", "plan_valid_rate", "plan_valid_rate_completed", "first_attempt_valid_rate",
            "prereq_violation_rate", "credit_ok_rate", "requirements_met_rate", "hallucinated_courses",
            "feasibility_accuracy", "false_feasible_rate", "avg_repair_rounds", "avg_llm_calls", "avg_tokens",
            "avg_seconds"]
    lines = ["| metric | " + " | ".join(summary) + " |", "|---|" + "---|" * len(summary)]
    for c in cols:
        lines.append(f"| {c} | " + " | ".join("-" if summary[m][c] is None else str(summary[m][c])
                                             for m in summary) + " |")
    md = "\n".join(lines)
    (out_dir / "summary.md").write_text(
        f"# EduPath evaluation\n\nLLM: {model}\n\n"
        "Plan-quality metrics use feasible scenarios only; feasibility metrics use all scenarios. "
        "Failed runs (timeouts, API errors) count as 0 for plan_valid_rate, first_attempt_valid_rate, "
        "requirements_met_rate and feasibility_accuracy; `plan_valid_rate_completed` and the rows below "
        "it use completed runs only.\n\n" + md + "\n")
    return md


def keep_mac_awake() -> None:
    """Stop macOS sleeping while this script runs (sleep stalls the local model)."""
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="solver", help=f"comma list from {list(MODES)}")
    ap.add_argument("--scenarios", default="", help="comma list of scenario ids (default: all)")
    ap.add_argument("--repeats", type=int, default=1, help="repeat LLM runs (they are non-deterministic)")
    ap.add_argument("--repair-rounds", type=int, default=2)
    ap.add_argument("--effort", default="high", help="LLM thinking effort for planning: low|medium|high")
    ap.add_argument("--provider", default=None, help="ollama | groq | gemini | claude (default: from .env)")
    ap.add_argument("--model", default=None, help="model name, e.g. qwen3:8b or openai/gpt-oss-120b")
    ap.add_argument("--resume", default=None, help="results folder to finish (re-runs missing/failed runs)")
    ap.add_argument("--yes", action="store_true", help="don't ask before starting")
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
        planner = make_planner(args.provider, plan_effort=args.effort, model=args.model)
        if planner is None:
            sys.exit("LLM modes need an LLM: put EDUPATH_PROVIDER=ollama (free, local) in .env, "
                     "or pass --provider.")
        if hasattr(planner, "max_wait"):
            planner.max_wait = 90            # batch job: OK to wait out per-minute limits
    model = planner.name if planner else "none (solver only)"

    # Which runs are already done (when resuming)?
    out_dir = Path(args.resume) if args.resume else \
        RESULTS / (time.strftime("%Y%m%d-%H%M%S") + "-" + re.sub(r"[^A-Za-z0-9.]+", "-", model.split(",")[0]).strip("-"))
    old: dict[tuple, dict] = {}
    if args.resume:
        for line in (out_dir / "runs.jsonl").read_text().splitlines():
            r = json.loads(line)
            if not r.get("error"):
                old[(r["scenario"], r["mode"], r.get("repeat", 0))] = r
    todo = [(sc, mode, rep) for sc in scenarios for mode in modes
            for rep in range(args.repeats if mode != "solver" else 1)
            if (sc["id"], mode, rep) not in old]

    n_llm = sum(1 for _, mode, _ in todo if mode != "solver")
    if llm_modes:
        print(f"Using {model}. {len(old)} run(s) already done; this will make {n_llm} LLM scenario runs "
              f"(each 1-{args.repair_rounds + 1} LLM calls).")
        if not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
            sys.exit("Cancelled.")
    keep_mac_awake()

    # The research arm measures the LLM itself, so no silent solver fallback here.
    agent = EduPathAgent(cur, planner, max_repair_rounds=args.repair_rounds, solver_fallback=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = list(old.values())
    with (out_dir / "runs.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        for sc, mode, rep in todo:
            row = run_one(agent, cur, sc, mode) | {"repeat": rep, "model": model}
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            validity = (f"valid={row.get('plan_valid_rate')}" if row["truly_feasible"]
                        else "infeasible goal")
            status = ("FAILED: " + row["error"]) if row.get("error") else \
                f"{validity}  feasibility_ok={row['feasibility_correct']}"
            print(f"{sc['id']:<18} {mode:<13} rep{rep}  {status}")
            if row.get("expected_feasible") is not None and row["expected_feasible"] != row["truly_feasible"]:
                print(f"  ! scenario says expected_feasible={row['expected_feasible']} "
                      f"but the solver says {row['truly_feasible']}")

    print("\n" + write_summary(out_dir, rows, model))
    failed = sum(1 for r in rows if r.get("error"))
    print(f"\nSaved to {out_dir}")
    if failed:
        print(f"{failed} run(s) failed. To retry just those:  python -m evaluation.run_eval "
              f"--modes {args.modes} --resume {out_dir}" + (f" --provider {args.provider}" if args.provider else "")
              + (f" --model {args.model}" if args.model else "") + (" --scenarios " + args.scenarios if args.scenarios else "")
              + (f" --repeats {args.repeats}" if args.repeats != 1 else ""))


if __name__ == "__main__":
    main()
