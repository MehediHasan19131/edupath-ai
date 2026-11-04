"""Checks the Groq planner against a fake Groq server - no key, no network, no quota used."""

import json

import httpx
import pytest

import edupath.llm_groq as llm_groq
from edupath.agent import EduPathAgent
from edupath.llm_groq import GroqPlanner, strict_schema
from edupath.schemas import LLMError, PlanProposals

PLAN_JSON = {"feasible": True, "feasibility_note": "ok", "plans": [
    {"name": "P", "rationale": "r", "semesters": [{"term": "Spring 2027", "courses": ["CSE360"], "milestones": []}]}]}


def fake_planner(replies, statuses=None):
    sent = []
    statuses = list(statuses or [])

    def handler(request):
        sent.append(json.loads(request.content))
        code = statuses.pop(0) if statuses else 200
        if code != 200:
            return httpx.Response(code, headers={"retry-after": "0"}, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": json.dumps(replies.pop(0))},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}})

    client = httpx.Client(base_url="https://groq.test", transport=httpx.MockTransport(handler))
    return GroqPlanner(client=client), sent


def test_propose_and_repair_keep_the_conversation():
    planner, sent = fake_planner([PLAN_JSON, PLAN_JSON])
    proposals, convo = planner.propose_plans("brief")
    assert proposals.plans[0].semesters[0].courses == ["CSE360"]
    planner.repair_plans(convo, "fix PREREQ_VIOLATION")
    assert [m["role"] for m in sent[1]["messages"]] == ["system", "user", "assistant", "user"]
    fmt = sent[0]["response_format"]["json_schema"]
    assert fmt["strict"] is True and sent[0]["reasoning_effort"] == "high"
    assert planner.usage.calls == 2


def test_strict_schema_closes_every_object():
    s = strict_schema(PlanProposals)
    objects = [s] + [d for d in s["$defs"].values()]
    assert all(o["additionalProperties"] is False for o in objects)


def test_retries_when_rate_limited(monkeypatch):
    monkeypatch.setattr(llm_groq.time, "sleep", lambda s: None)
    planner, sent = fake_planner([PLAN_JSON], statuses=[429, 200])
    proposals, _ = planner.propose_plans("brief")
    assert proposals.feasible and len(sent) == 2


def test_app_keeps_working_when_the_free_limit_is_used_up(cur, monkeypatch, student_a, goals3):
    monkeypatch.setattr(llm_groq.time, "sleep", lambda s: None)
    planner, _ = fake_planner([], statuses=[429] * 20)
    result = EduPathAgent(cur, planner).run(student_a, "I want to graduate in 3 semesters", goals3)
    assert result.outcomes and all(o.report.is_valid for o in result.outcomes)
    assert any("offline parser" in t for t in result.trace)


def test_missing_key_is_explained(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMError, match="console.groq.com"):
        GroqPlanner()
