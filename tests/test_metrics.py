"""Failed runs (timeouts, API errors) must count against a mode, not silently vanish."""

from evaluation.metrics import summarize


def test_failed_runs_count_as_failures():
    rows = [
        {"scenario": "A", "mode": "llm_verified", "truly_feasible": True, "plan_valid_rate": 1.0,
         "first_attempt_valid_rate": 0.0, "requirements_met_rate": 1.0, "feasibility_correct": True,
         "false_feasible": False, "hallucinated_courses": 0, "prereq_violation_rate": 0.0,
         "credit_ok_rate": 1.0, "repair_rounds": 2, "llm_calls": 3, "tokens": 100, "seconds": 10},
        {"scenario": "B", "mode": "llm_verified", "truly_feasible": True, "error": "timeout"},
    ]
    s = summarize(rows)["llm_verified"]
    assert s["runs"] == 2 and s["failed_runs"] == 1
    assert s["plan_valid_rate"] == 0.5               # the timeout counts as 0
    assert s["plan_valid_rate_completed"] == 1.0     # completed runs only
    assert s["feasibility_accuracy"] == 0.5
    assert s["avg_seconds"] == 10                    # cost averages ignore the failed run
