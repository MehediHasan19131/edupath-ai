from edupath.requirements import audit
from edupath.students import STUDENTS_DIR, load_student


def test_curriculum_adds_up_to_136(cur):
    total = sum(cur.credits(g.courses) if g.kind == "all" else 3 * g.count for g in cur.groups)
    assert total == cur.policy.total_credits_required == 136


def test_audit_96_credit_student(cur, student_a):
    a = audit(cur, student_a.completed)
    assert a.credits_completed == 96
    core = next(g for g in a.groups if g.id == "program_core")
    assert core.missing == ["CSE350", "CSE360", "CSE420", "CSE421", "CSE460", "CSE461", "CSE470", "CSE471"]
    # PSY101 fills Stream 4, the extra ECO101 spills into GenEd electives.
    assert next(g for g in a.groups if g.id == "gened_stream4").counted == ["PSY101"]
    assert next(g for g in a.groups if g.id == "gened_elective").counted == ["ECO101"]


def test_aliases_count_as_the_real_course(cur):
    d = load_student(STUDENTS_DIR / "cse_108_credits_p1_done.json")   # has CSE161 and DEV101
    a = audit(cur, d.completed)
    assert "CSE110" not in next(g for g in a.groups if g.id == "program_core").missing
    assert "EMB101" not in next(g for g in a.groups if g.id == "gened_required").missing


def test_program_elective_needs_one_cse_course(cur):
    # Two GenEd options alone must NOT complete "Program electives" (min 1 CSE elective).
    gened_full = ["HUM101", "PSY101", "CST201", "HUM102", "PSY102"]   # fills streams 3/4/5 + 2 GenEd electives
    a = audit(cur, gened_full + ["ANT101", "ANT342"])
    pe = next(g for g in a.groups if g.id == "program_elective")
    assert not pe.done and pe.picks_needed == 1 and pe.min_from_needed == 1
