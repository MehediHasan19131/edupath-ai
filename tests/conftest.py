import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edupath.curriculum import load_curriculum  # noqa: E402
from edupath.models import PlanningGoals  # noqa: E402
from edupath.students import STUDENTS_DIR, load_student  # noqa: E402


@pytest.fixture(scope="session")
def cur():
    return load_curriculum()


@pytest.fixture
def student_a():
    """96 credits, thesis not started - the 'graduate in three semesters' example."""
    return load_student(STUDENTS_DIR / "cse_96_credits.json")


@pytest.fixture
def goals3():
    return PlanningGoals(target_semesters=3, start_term="Spring 2027",
                         max_credits_per_semester=15, min_credits_per_semester=9)
