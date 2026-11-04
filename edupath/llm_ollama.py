"""
Ollama provider (FREE, runs on your own computer - no key, no internet, no limits).

Setup, once:
    1. Install Ollama from https://ollama.com (or `brew install ollama`)
    2. ollama pull qwen3:8b            (about 5 GB; needs ~8 GB of free RAM)
    3. put EDUPATH_PROVIDER=ollama in .env

Same four jobs as llm_gemini.py. We call Ollama's /api/chat with `format` set
to the JSON schema of our Pydantic class, so the model can only produce JSON in
that shape. Student records never leave your machine.

Speed: an 8B model on a laptop writes roughly 20-40 words a second, and with
"thinking" on it reasons before answering, so a planning call can take a few
minutes. Use plan_effort="low" (thinking off) for quicker, weaker plans.
"""

from __future__ import annotations

import json
import os

import httpx
from pydantic import BaseModel, ValidationError

from .prompts import EXPLAIN_SYSTEM, INTERPRET_SYSTEM, PLANNER_SYSTEM
from .schemas import GoalsExtraction, LLMError, PlanProposals, Usage

DEFAULT_OLLAMA_MODEL = os.getenv("EDUPATH_OLLAMA_MODEL", "qwen3:8b")
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

CONTEXT_TOKENS = 16384       # room for the ~2k-token brief plus repair rounds
MAX_OUTPUT_TOKENS = 12000    # thinking + answer
TIMEOUT_SECONDS = 900        # local models are slow; don't give up too early


class OllamaPlanner:
    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, client: httpx.Client | None = None,
                 plan_effort: str = "high"):
        self.model = model
        self.name = f"Ollama ({model}, local)"
        self.client = client or httpx.Client(base_url=OLLAMA_URL, timeout=TIMEOUT_SECONDS)
        self.plan_effort = plan_effort           # low = thinking off; medium/high = thinking on
        self.usage = Usage()

    # ---- the four jobs ----------------------------------------------------- #

    def interpret_request(self, request_text: str) -> GoalsExtraction:
        reply = self._call(INTERPRET_SYSTEM, [_user(request_text)], GoalsExtraction, effort="low")
        return _parse(GoalsExtraction, reply)

    def propose_plans(self, brief: str) -> tuple[PlanProposals, list]:
        """Returns the proposals AND the conversation so far (needed for repairs)."""
        messages = [_user(brief)]
        reply = self._call(PLANNER_SYSTEM, messages, PlanProposals, self.plan_effort)
        return _parse(PlanProposals, reply), messages + [_assistant(reply)]

    def repair_plans(self, messages: list, feedback: str) -> tuple[PlanProposals, list]:
        """Continue the same conversation with the validator's findings."""
        messages = messages + [_user(feedback)]
        reply = self._call(PLANNER_SYSTEM, messages, PlanProposals, self.plan_effort)
        return _parse(PlanProposals, reply), messages + [_assistant(reply)]

    def explain(self, facts: str) -> str:
        reply = self._call(EXPLAIN_SYSTEM, [_user(facts)], schema=None, effort="low")
        return reply["content"].strip()

    # ---- plumbing ---------------------------------------------------------- #

    def _call(self, system: str, messages: list, schema: type[BaseModel] | None, effort: str) -> dict:
        if schema:
            # The format grammar forces the shape; the model still needs to see the
            # field descriptions to know what goes where.
            system += ("\n\nReply with JSON only, matching this schema:\n"
                       + json.dumps(schema.model_json_schema()))
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": False,
            "think": effort != "low",
            "options": {"num_ctx": CONTEXT_TOKENS, "num_predict": MAX_OUTPUT_TOKENS, "temperature": 0.6},
        }
        if schema:
            body["format"] = schema.model_json_schema()

        try:
            r = self.client.post("/api/chat", json=body)
        except httpx.ConnectError as e:
            raise LLMError("Ollama isn't running. Open the Ollama app (or run `ollama serve`), "
                           "then try again.") from e
        except httpx.TimeoutException as e:
            raise LLMError(f"Ollama took longer than {TIMEOUT_SECONDS}s. Try plan effort 'low' "
                           "or a smaller model.") from e
        if r.status_code != 200:
            raise LLMError(_friendly(r, self.model))

        data = r.json()
        self.usage.calls += 1
        self.usage.input_tokens += data.get("prompt_eval_count") or 0
        self.usage.output_tokens += data.get("eval_count") or 0
        reply = data.get("message") or {}
        if schema and data.get("done_reason") == "length":
            raise LLMError(f"Ollama ran out of output tokens ({MAX_OUTPUT_TOKENS}) before finishing the JSON.")
        return {"content": reply.get("content") or "", "thinking": reply.get("thinking") or ""}


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(reply: dict) -> dict:
    # Only the answer goes back into the conversation; Qwen3 is meant to drop old thinking.
    return {"role": "assistant", "content": reply["content"]}


def _parse(schema: type[BaseModel], reply: dict):
    try:
        return schema.model_validate_json(reply["content"])
    except ValidationError as e:
        raise LLMError(f"Ollama's reply didn't match the expected shape: {e.errors()[0]['msg']}") from e


def _friendly(r: httpx.Response, model: str) -> str:
    try:
        message = r.json().get("error", r.text)
    except ValueError:
        message = r.text
    if r.status_code == 404 and "not found" in message:
        return f"Model '{model}' isn't downloaded yet. Run:  ollama pull {model}"
    return f"Ollama error {r.status_code}: {message}"
