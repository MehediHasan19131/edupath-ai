"""Checks the Ollama planner against a fake Ollama server - no model download needed."""

import json

import httpx
import pytest

from edupath.llm_ollama import OllamaPlanner
from edupath.schemas import LLMError

PLAN_JSON = {"feasible": True, "feasibility_note": "ok", "plans": [
    {"name": "P", "rationale": "r", "semesters": [{"term": "Spring 2027", "courses": ["CSE360"], "milestones": []}]}]}


def fake_planner(replies, status=200):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        if status != 200:
            return httpx.Response(status, json={"error": "model 'qwen3:8b' not found"})
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": json.dumps(replies.pop(0)), "thinking": "hmm"},
            "done_reason": "stop", "prompt_eval_count": 100, "eval_count": 20})

    client = httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    return OllamaPlanner(client=client), sent


def test_propose_and_repair_keep_the_conversation():
    planner, sent = fake_planner([PLAN_JSON, PLAN_JSON])
    proposals, convo = planner.propose_plans("brief")
    assert proposals.plans[0].semesters[0].courses == ["CSE360"]
    planner.repair_plans(convo, "fix PREREQ_VIOLATION")
    assert [m["role"] for m in sent[1]["messages"]] == ["system", "user", "assistant", "user"]
    assert sent[0]["format"]["title"] == "PlanProposals" and sent[0]["think"] is True
    assert planner.usage.calls == 2 and planner.usage.input_tokens == 200


def test_missing_model_gives_a_helpful_error():
    planner, _ = fake_planner([], status=404)
    with pytest.raises(LLMError, match="ollama pull"):
        planner.propose_plans("brief")
