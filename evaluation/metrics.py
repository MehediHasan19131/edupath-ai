"""
Metrics for "Constraint-Grounded Planning for Reliable LLM-Based Academic Advisors".

All metrics are computed by the deterministic validator, never by an LLM, so
every mode is judged by the same referee.

Per run (one scenario x one mode x one repetition):
  plan_valid_rate        share of returned plans with zero errors          (higher = better)
  prereq_violation_rate  PREREQ_VIOLATION errors / course placements        (lower  = better)
  credit_ok_rate         semesters within the credit limit / all semesters  (higher = better)
  requirements_met_rate  plans that complete the degree / plans             (higher = better)
  hallucinated_courses   UNKNOWN_COURSE errors                              (lower  = better)
  feasibility_correct    did the mode correctly say "possible" / "impossible"?
  false_feasible         said "possible" when it is not (the dangerous error)
"""

from __future__ import annotations

from statistics import mean


def plan_metrics(outcomes) -> dict:
    """Metrics over a list of PlanOutcome."""
    if not outcomes:
        return {"n_plans": 0, "plan_valid_rate": 0.0, "prereq_violation_rate": None,
                "credit_ok_rate": None, "requirements_met_rate": 0.0, "hallucinated_courses": 0,
                "error_counts": {}}
    placements = sum(len(o.plan.all_courses()) for o in outcomes)
    semesters = sum(len(o.plan.semesters) for o in outcomes)
    errors = [v for o in outcomes for v in o.report.errors]
    counts: dict[str, int] = {}
    for v in errors:
        counts[v.code] = counts.get(v.code, 0) + 1
    overloaded_semesters = len({(id(o), v.term) for o in outcomes for v in o.report.errors
                                if v.code == "CREDIT_OVERLOAD"})
    return {
        "n_plans": len(outcomes),
        "plan_valid_rate": mean(o.report.is_valid for o in outcomes),
        "prereq_violation_rate": counts.get("PREREQ_VIOLATION", 0) / placements if placements else 0.0,
        "credit_ok_rate": 1 - overloaded_semesters / semesters if semesters else 1.0,
        "requirements_met_rate": mean(not any(v.code == "REQUIREMENTS_UNMET" for v in o.report.errors)
                                      for o in outcomes),
        "hallucinated_courses": counts.get("UNKNOWN_COURSE", 0),
        "error_counts": counts,
    }


def summarize(rows: list[dict]) -> dict[str, dict]:
    """Average the per-run rows for each mode."""
    out: dict[str, dict] = {}
    for mode in sorted({r["mode"] for r in rows}):
        rs = [r for r in rows if r["mode"] == mode and not r.get("error")]
        feas = [r for r in rs if r["truly_feasible"]]          # plan quality only where a plan exists
        infeas = [r for r in rs if not r["truly_feasible"]]

        def avg(key, subset):
            vals = [r[key] for r in subset if r.get(key) is not None]
            return round(mean(vals), 3) if vals else None

        out[mode] = {
            "runs": len(rs),
            "failed_runs": sum(1 for r in rows if r["mode"] == mode and r.get("error")),
            "plan_valid_rate": avg("plan_valid_rate", feas),
            "first_attempt_valid_rate": avg("first_attempt_valid_rate", feas),
            "prereq_violation_rate": avg("prereq_violation_rate", feas),
            "credit_ok_rate": avg("credit_ok_rate", feas),
            "requirements_met_rate": avg("requirements_met_rate", feas),
            "hallucinated_courses": sum(r["hallucinated_courses"] for r in feas),
            "feasibility_accuracy": avg("feasibility_correct", rs),
            "false_feasible_rate": avg("false_feasible", infeas),
            "avg_repair_rounds": avg("repair_rounds", rs),
            "avg_llm_calls": avg("llm_calls", rs),
            "avg_tokens": avg("tokens", rs),
            "avg_seconds": avg("seconds", rs),
        }
    return out
