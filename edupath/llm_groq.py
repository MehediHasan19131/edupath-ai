"""
Groq provider (FREE cloud API - good for the hosted web app).

Setup:
    1. Free key at https://console.groq.com/keys (sign in with Google or GitHub)
    2. In .env (or Streamlit Cloud -> Settings -> Secrets):
           GROQ_API_KEY=...
           EDUPATH_PROVIDER=groq

Same four jobs as llm_ollama.py, but the model runs on Groq's servers, so it is
fast and needs nothing on your computer. We use "strict" structured output:
Groq constrains the reply to our Pydantic class's JSON schema.

Free-tier limits (per model, shared by everyone using your key) are roughly
30 requests/minute, 8K tokens/minute and 200K tokens/day. Reading a request
plus writing an explanation uses about 2-3K tokens; the LLM planning modes use
10-25K per run. When Groq says "too many requests" we wait and retry, and the
agent falls back to the offline parser / template explanation if it still fails.
Requests (course lists, no names) are processed by Groq.
"""

from __future__ import annotations

import copy
import json
import os
import time

import httpx
from pydantic import BaseModel, ValidationError

from .prompts import EXPLAIN_SYSTEM, INTERPRET_SYSTEM, PLANNER_SYSTEM
from .schemas import GoalsExtraction, LLMError, PlanProposals, Usage

DEFAULT_GROQ_MODEL = os.getenv("EDUPATH_GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_URL = "https://api.groq.com/openai/v1"

RETRIES = 3              # on 429 / 5xx
MAX_WAIT_SECONDS = 30    # never make a web visitor wait longer than this per retry (run_eval raises it)
TIMEOUT_SECONDS = 120
# Groq counts input + max_completion_tokens against the tokens-per-minute limit, so a
# request asking for more than the limit is refused outright. Keep every request under it.
TOKENS_PER_MINUTE = int(os.getenv("EDUPATH_GROQ_TPM", "8000"))


class GroqPlanner:
    def __init__(self, model: str = DEFAULT_GROQ_MODEL, client: httpx.Client | None = None,
                 plan_effort: str = "high"):
        self.model = model
        self.name = f"Groq ({model}, free tier)"
        if client is None:
            key = os.getenv("GROQ_API_KEY")
            if not key:
                raise LLMError("GROQ_API_KEY is missing. Get a free key at console.groq.com/keys.")
            client = httpx.Client(base_url=GROQ_URL, timeout=TIMEOUT_SECONDS,
                                  headers={"Authorization": f"Bearer {key}"})
        self.client = client
        self.plan_effort = plan_effort           # low | medium | high (gpt-oss reasoning effort)
        self.usage = Usage()
        self.max_wait = MAX_WAIT_SECONDS

    # ---- the four jobs ----------------------------------------------------- #

    def interpret_request(self, request_text: str) -> GoalsExtraction:
        reply = self._call(INTERPRET_SYSTEM, [_user(request_text)], GoalsExtraction, "low", max_tokens=2000)
        return _parse(GoalsExtraction, reply)

    def propose_plans(self, brief: str) -> tuple[PlanProposals, list]:
        """Returns the proposals AND the conversation so far (needed for repairs)."""
        messages = [_user(brief)]
        reply = self._call(PLANNER_SYSTEM, messages, PlanProposals, self.plan_effort, max_tokens=8000)
        return _parse(PlanProposals, reply), messages + [_assistant(reply)]

    def repair_plans(self, messages: list, feedback: str) -> tuple[PlanProposals, list]:
        """Continue the same conversation with the validator's findings."""
        messages = messages + [_user(feedback)]
        reply = self._call(PLANNER_SYSTEM, messages, PlanProposals, self.plan_effort, max_tokens=8000)
        return _parse(PlanProposals, reply), messages + [_assistant(reply)]

    def explain(self, facts: str) -> str:
        return self._call(EXPLAIN_SYSTEM, [_user(facts)], None, "low", max_tokens=3000).strip()

    # ---- plumbing ---------------------------------------------------------- #

    def _call(self, system: str, messages: list, schema: type[BaseModel] | None, effort: str,
              max_tokens: int) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
        }
        if self.model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = effort
            body["include_reasoning"] = False     # we only need the answer
        if schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "strict": True, "schema": strict_schema(schema)}}
        # Rough token estimate (~3.5 characters per token) of what we send.
        sent_tokens = len(json.dumps(body)) // 3.5
        body["max_completion_tokens"] = int(max(1000, min(max_tokens, TOKENS_PER_MINUTE - sent_tokens - 300)))

        for attempt in range(RETRIES + 1):
            try:
                r = self.client.post("/chat/completions", json=body)
            except httpx.HTTPError as e:
                raise LLMError(f"Could not reach Groq: {e}") from e
            retryable = r.status_code == 429 or r.status_code >= 500
            wait = _retry_after(r, attempt)
            if retryable and attempt < RETRIES and wait <= self.max_wait:
                time.sleep(wait)
                continue
            break                                 # success, or a long wait = daily limit used up
        if r.status_code != 200:
            raise LLMError(_friendly(r))

        data = r.json()
        self.usage.calls += 1
        u = data.get("usage") or {}
        self.usage.input_tokens += u.get("prompt_tokens") or 0
        self.usage.output_tokens += u.get("completion_tokens") or 0
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise LLMError(f"Groq ran out of output tokens ({body['max_completion_tokens']}) before finishing - "
                           "try a lower effort (less reasoning).")
        return choice["message"].get("content") or ""


def strict_schema(schema: type[BaseModel]) -> dict:
    """Groq's strict mode wants additionalProperties: false on every object."""
    s = copy.deepcopy(schema.model_json_schema())

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(s)
    return s


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _parse(schema: type[BaseModel], content: str):
    try:
        return schema.model_validate_json(content)
    except ValidationError as e:
        raise LLMError(f"Groq's reply didn't match the expected shape: {e.errors()[0]['msg']}") from e


def _retry_after(r: httpx.Response, attempt: int) -> float:
    try:
        return float(r.headers.get("retry-after", ""))
    except ValueError:
        return 5 * 2 ** attempt


def _friendly(r: httpx.Response) -> str:
    try:
        message = r.json().get("error", {}).get("message", r.text)
    except ValueError:
        message = r.text
    if r.status_code == 401:
        return "Groq API key missing or invalid. Check GROQ_API_KEY (free at console.groq.com/keys)."
    if r.status_code == 429:
        return ("Groq free-tier limit reached. Try again in a minute (or tomorrow if the daily limit is used up). "
                f"Groq said: {message}")
    if r.status_code == 413:
        return f"Request too large for Groq's free tier: {message}"
    if r.status_code == 404:
        return f"Groq model not found ({message}). Set EDUPATH_GROQ_MODEL to a current model."
    return f"Groq error {r.status_code}: {message}"
