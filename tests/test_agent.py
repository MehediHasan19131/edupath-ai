"""Tests the agent's LLM modes with a scripted fake instead of a real LLM API."""

import pytest

from edupath.agent import EduPathAgent
from edupath.schemas import GoalsExtraction, LLMError, PlanProposals, ProposedPlan, ProposedSemester, Usage
from edupath.request_parser import parse_request

GOOD = [["CSE360", "ANT101", "BUS334", "CSE350"],
        ["CSE420", "CSE421", "CSE427", "CSE428", "CSE460"],
        ["CSE400", "CSE461", "CSE470", "CSE471"]]
# Classic LLM mistake: CSE461 together with its prerequisite CSE360.
BAD = [["CSE360", "CSE461", "ANT101", "BUS334"],
       ["CSE420", "CSE421", "CSE427", "CSE428", "CSE460"],
       ["CSE400", "CSE350", "CSE470", "CSE471"]]


def proposal(semesters, feasible=True):
    terms = ["Spring 2027", "Summer 2027", "Fall 2027"]
    return PlanProposals(feasible=feasible, feasibility_note="", plans=[ProposedPlan(
        name="LLM plan", rationale="",
        semesters=[ProposedSemester(term=t, courses=c, milestones=[]) for t, c in zip(terms, semesters)])])


class FakePlanner:
    """Replays a fixed list of proposals: first for propose_plans, then one per repair."""

    def __init__(self, *proposals):
        self.queue = list(proposals)
        self.usage = Usage()
        self.feedback = []

    def interpret_request(self, text):
        return parse_request(text)

    def propose_plans(self, brief):
        self.usage.calls += 1
        return self.queue.pop(0), [brief]

    def repair_plans(self, messages, feedback):
        self.usage.calls += 1
        self.feedback.append(feedback)
        return self.queue.pop(0), messages + [feedback]

    def explain(self, facts):
        raise LLMError("offline")          # forces the template explanation


def test_llm_only_keeps_invalid_plan(cur, student_a, goals3):
    agent = EduPathAgent(cur, FakePlanner(proposal(BAD)))
    r = agent.run(student_a, "", goals3, mode="llm_only")
    assert r.repair_rounds == 0
    assert not r.outcomes[0].report.is_valid


def test_llm_verified_repairs_with_validator_feedback(cur, student_a, goals3):
    fake = FakePlanner(proposal(BAD), proposal(GOOD))
    r = EduPathAgent(cur, fake).run(student_a, "", goals3, mode="llm_verified")
    assert r.repair_rounds == 1
    assert "PREREQ_VIOLATION" in fake.feedback[0] and "CSE461" in fake.feedback[0]
    assert r.outcomes[0].report.is_valid and r.outcomes[0].origin == "llm_repaired"
    assert not r.first_attempt[0].report.is_valid


def test_llm_verified_falls_back_to_solver(cur, student_a, goals3):
    fake = FakePlanner(proposal(BAD), proposal(BAD), proposal(BAD))
    r = EduPathAgent(cur, fake, max_repair_rounds=2).run(student_a, "", goals3, mode="llm_verified")
    assert r.repair_rounds == 2
    assert r.outcomes and all(o.origin == "solver_fallback" and o.report.is_valid for o in r.outcomes)


def test_llm_modes_need_a_planner(cur, student_a, goals3):
    with pytest.raises(LLMError):
        EduPathAgent(cur, None).run(student_a, "", goals3, mode="llm_only")


def test_request_overrides_defaults_and_flags_mismatch(cur, student_a, goals3):
    r = EduPathAgent(cur).run(student_a, "I have completed 95 credits and want to finish in 4 semesters, "
                                         "max 12 credits per semester", goals3)
    assert r.goals.target_semesters == 4 and r.goals.max_credits_per_semester == 12
    assert any("95" in n for n in r.notes)              # stated 95 vs record 96


def test_offline_parser():
    g = parse_request("Graduate in three semesters, avoid CSE 460, I like machine learning and web dev")
    assert g.target_semesters == 3
    assert g.avoid_courses == ["CSE460"]
    assert {"ai", "web"} <= set(g.interests)
    assert parse_request("hello").target_semesters is None
