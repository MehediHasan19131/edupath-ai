import json
from pathlib import Path

import pytest

from edupath.models import PlanningGoals
from edupath.solver import generate_plans
from edupath.students import STUDENTS_DIR, load_student

SCENARIOS = json.loads((Path(__file__).resolve().parent.parent / "evaluation" / "scenarios.json").read_text())


@pytest.mark.parametrize("sc", SCENARIOS["scenarios"], ids=lambda s: s["id"])
def test_every_scenario(cur, sc):
    student = load_student(STUDENTS_DIR / sc["student"])
    goals = PlanningGoals(sc["semesters"], "Spring 2027", sc["max_credits"], 9, interests=sc.get("interests", []))
    result = generate_plans(cur, student, goals)

    assert result.feasibility.feasible == sc["expected_feasible"]
    if sc["expected_feasible"]:
        assert result.plans, "a feasible goal must produce at least one plan"
        for plan, report in result.plans:
            assert report.is_valid, report.errors
            assert len(plan.semesters) <= sc["semesters"]
    else:
        assert result.plans == []
        assert result.feasibility.min_semesters > sc["semesters"]


def test_example_from_project_brief(cur, student_a, goals3):
    """96 credits, graduate in 3 semesters: possible, with the thesis spread over all three."""
    result = generate_plans(cur, student_a, goals3)
    plan, _ = result.plans[0]
    assert "CSE400" in plan.semesters[-1].courses
    assert plan.semesters[0].milestones == ["Thesis: Pre-thesis 1 (P1)"]


def test_interests_steer_electives(cur, student_a, goals3):
    goals3.interests = ["security"]
    student_a.interests = []
    plan, _ = generate_plans(cur, student_a, goals3).plans[0]
    assert "CSE447" in plan.all_courses()
