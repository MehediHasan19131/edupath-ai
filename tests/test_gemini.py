"""Checks the Gemini planner against a fake HTTP server - no key, no network, no quota used."""

import json

import httpx2
from google import genai
from google.genai import types

import edupath.llm_gemini as llm_gemini
from edupath.llm_gemini import GeminiPlanner

PLAN_JSON = {"feasible": True, "feasibility_note": "ok", "plans": [
    {"name": "P", "rationale": "r", "semesters": [{"term": "Spring 2027", "courses": ["CSE360"], "milestones": []}]}]}


def fake_planner(replies, statuses=None):
    sent = []
    statuses = list(statuses or [])

    def handler(request):
        sent.append(json.loads(request.content))
        if statuses:
            code = statuses.pop(0)
            if code != 200:
                return httpx2.Response(code, json={"error": {"code": code, "message": "slow down", "status": "X"}})
        text = json.dumps(replies.pop(0))
        return httpx2.Response(200, json={
            "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20}})

    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    client = genai.Client(api_key="test", http_options=types.HttpOptions(httpx_client=http))
    return GeminiPlanner(client=client), sent


def test_propose_and_repair_keep_the_conversation(monkeypatch):
    monkeypatch.setattr(llm_gemini, "MIN_SECONDS_BETWEEN_CALLS", 0)
    planner, sent = fake_planner([PLAN_JSON, PLAN_JSON])
    proposals, convo = planner.propose_plans("brief")
    assert proposals.plans[0].semesters[0].courses == ["CSE360"]
    planner.repair_plans(convo, "fix PREREQ_VIOLATION")
    assert [c["role"] for c in sent[1]["contents"]] == ["user", "model", "user"]
    assert sent[0]["generationConfig"]["responseMimeType"] == "application/json"
    assert planner.usage.calls == 2


def test_retries_when_rate_limited(monkeypatch):
    monkeypatch.setattr(llm_gemini, "MIN_SECONDS_BETWEEN_CALLS", 0)
    monkeypatch.setattr(llm_gemini.time, "sleep", lambda s: None)
    planner, sent = fake_planner([PLAN_JSON], statuses=[429, 200])
    proposals, _ = planner.propose_plans("brief")
    assert proposals.feasible and len(sent) == 2
