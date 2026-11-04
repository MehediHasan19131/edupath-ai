"""Load and save student records (data/students/*.json)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import StudentRecord

STUDENTS_DIR = Path(__file__).resolve().parent.parent / "data" / "students"


def load_student(path: str | Path) -> StudentRecord:
    raw = json.loads(Path(path).read_text())
    return StudentRecord(
        name=raw.get("name", "Student"),
        student_id=raw.get("student_id", ""),
        program=raw.get("program", "CSE"),
        completed=raw["completed"],
        thesis_stage=raw.get("thesis_stage", "not_started"),
        interests=raw.get("interests", []),
    )


def save_student(student: StudentRecord, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(student), indent=2))


def list_sample_students() -> list[Path]:
    return sorted(STUDENTS_DIR.glob("*.json"))
