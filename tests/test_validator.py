from edupath.models import Plan, Semester, StudentRecord
from edupath.validator import validate_plan

GOOD = [
    ["CSE360", "ANT101", "BUS334", "CSE350"],
    ["CSE420", "CSE421", "CSE427", "CSE428", "CSE460"],
    ["CSE400", "CSE461", "CSE470", "CSE471"],
]


def make(semesters):
    terms = ["Spring 2027", "Summer 2027", "Fall 2027", "Spring 2028"]
    return Plan("test", [Semester(t, list(cs)) for t, cs in zip(terms, semesters)], "llm")


def codes(report):
    return {v.code for v in report.violations}


def test_good_plan_is_valid(cur, student_a, goals3):
    r = validate_plan(cur, student_a, goals3, make(GOOD))
    assert r.is_valid, r.errors
    assert r.semester_credits == [12, 15, 13]


def test_hard_prereq_in_same_semester_is_an_error(cur, student_a, goals3):
    # CSE461 needs CSE360 passed BEFORE, not alongside.
    bad = [["CSE360", "CSE461", "ANT101", "BUS334"], GOOD[1], ["CSE400", "CSE350", "CSE470", "CSE471"]]
    r = validate_plan(cur, student_a, goals3, make(bad))
    assert any(v.code == "PREREQ_VIOLATION" and v.course == "CSE461" for v in r.errors)


def test_overload_unknown_and_repeat(cur, student_a, goals3):
    bad = [GOOD[0] + ["CSE420", "CSE999"], GOOD[1] + ["CSE221"], GOOD[2]]
    c = codes(validate_plan(cur, student_a, goals3, make(bad)))
    assert {"CREDIT_OVERLOAD", "UNKNOWN_COURSE", "ALREADY_COMPLETED", "DUPLICATE_COURSE"} <= c


def test_thesis_too_early(cur, student_a, goals3):
    bad = [GOOD[0] + ["CSE400"], GOOD[1], [c for c in GOOD[2] if c != "CSE400"]]
    r = validate_plan(cur, student_a, goals3, make(bad))
    assert any(v.code == "THESIS_TIMELINE" for v in r.errors)


def test_thesis_needs_75_credits_at_pre_thesis_1(cur, goals3):
    s = StudentRecord("x", "1", "CSE", ["CSE110"], "not_started")    # 3 credits only
    plan = make([["CSE111"], ["CSE220"], ["CSE400"]])
    r = validate_plan(cur, s, goals3, plan)
    assert any(v.code == "MIN_CREDITS_NOT_MET" for v in r.errors)
    assert any(v.code == "THESIS_TIMELINE" for v in r.errors)


def test_incomplete_degree_is_reported(cur, student_a, goals3):
    short = [GOOD[0], GOOD[1], ["CSE400", "CSE461", "CSE470"]]    # CSE471 missing
    r = validate_plan(cur, student_a, goals3, make(short))
    unmet = [v for v in r.errors if v.code == "REQUIREMENTS_UNMET"]
    assert unmet and "CSE471" in unmet[0].message


def test_too_many_semesters(cur, student_a, goals3):
    r = validate_plan(cur, student_a, goals3, make(GOOD + [["CSE391"]]))
    assert "SEMESTER_LIMIT" in codes(r)


def test_soft_prereq_is_only_a_warning(cur, goals3):
    # CSE341 has soft prerequisite CSE340.
    s = StudentRecord("x", "1", "CSE", [], "not_started")
    r = validate_plan(cur, s, goals3, make([["CSE341"]]))
    soft = [v for v in r.violations if v.code == "SOFT_PREREQ"]
    assert soft and soft[0].severity == "warning"
