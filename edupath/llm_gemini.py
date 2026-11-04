"""
Gemini provider (FREE tier - needs a free key from https://aistudio.google.com/apikey,
saved as GEMINI_API_KEY in .env).

Same four jobs as llm_claude.py. Gemini's structured output takes our Pydantic
class as `response_schema` and hands back a parsed object in `response.parsed`.

Free-tier notes:
  * Requests per minute/day are limited. We wait a few seconds between calls
    and retry automatically when Google says "too many requests" (HTTP 429).
  * Google may use free-tier prompts to improve its products - use sample or
    anonymised student records, not real personal data.
"""

from __future__ import annotations

import os
import time

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from .prompts import EXPLAIN_SYSTEM, INTERPRET_SYSTEM, PLANNER_SYSTEM
from .schemas import GoalsExtraction, LLMError, PlanProposals, Usage

DEFAULT_GEMINI_MODEL = os.getenv("EDUPATH_GEMINI_MODEL", "gemini-3.8-flash")

# Seconds to wait between calls, to stay under the free-tier per-minute limit.
MIN_SECONDS_BETWEEN_CALLS = float(os.getenv("EDUPATH_GEMINI_MIN_INTERVAL", "4"))
RETRIES = 4                                  # on 429 / 5xx, waiting 15s, 30s, 60s, 120s

THINKING = {"low": types.ThinkingLevel.LOW, "medium": types.ThinkingLevel.MEDIUM,
            "high": types.ThinkingLevel.HIGH}


class GeminiPlanner:
    def __init__(self, model: str = DEFAULT_GEMINI_MODEL, client: genai.Client | None = None,
                 plan_effort: str = "high"):
        self.model = model
        self.name = f"Gemini ({model}, free tier)"
        self.client = client or genai.Client()   # reads GEMINI_API_KEY (or GOOGLE_API_KEY)
        self.plan_effort = plan_effort           # low | medium | high
        self.usage = Usage()
        self._last_call = 0.0

    # ---- the four jobs ----------------------------------------------------- #

    def interpret_request(self, request_text: str) -> GoalsExtraction:
        response = self._call(INTERPRET_SYSTEM, [_user(request_text)], GoalsExtraction, effort="low")
        return response.parsed

    def propose_plans(self, brief: str) -> tuple[PlanProposals, list]:
        """Returns the proposals AND the conversation so far (needed for repairs)."""
        contents = [_user(brief)]
        response = self._call(PLANNER_SYSTEM, contents, PlanProposals, self.plan_effort)
        return response.parsed, contents + [response.candidates[0].content]

    def repair_plans(self, contents: list, feedback: str) -> tuple[PlanProposals, list]:
        """Continue the same conversation with the validator's findings."""
        contents = contents + [_user(feedback)]
        response = self._call(PLANNER_SYSTEM, contents, PlanProposals, self.plan_effort)
        return response.parsed, contents + [response.candidates[0].content]

    def explain(self, facts: str) -> str:
        response = self._call(EXPLAIN_SYSTEM, [_user(facts)], schema=None, effort="medium")
        return (response.text or "").strip()

    # ---- plumbing ---------------------------------------------------------- #

    def _call(self, system: str, contents: list, schema: type[BaseModel] | None, effort: str):
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=16000,
            thinking_config=types.ThinkingConfig(thinking_level=THINKING.get(effort, types.ThinkingLevel.HIGH)),
        )
        if schema:
            config.response_mime_type = "application/json"
            config.response_schema = schema

        for attempt in range(RETRIES + 1):
            self._wait_turn()
            try:
                response = self.client.models.generate_content(model=self.model, contents=contents, config=config)
                break
            except errors.APIError as e:
                retryable = e.code == 429 or (e.code or 0) >= 500
                if retryable and attempt < RETRIES:
                    time.sleep(15 * 2 ** attempt)
                    continue
                raise LLMError(_friendly(e)) from e
            except Exception as e:                # network problems etc.
                raise LLMError(f"Could not reach Gemini: {e}") from e

        self._track(response)
        if schema and response.parsed is None:
            raise LLMError(f"Gemini returned no structured output ({_why_empty(response)}).")
        return response

    def _wait_turn(self) -> None:
        wait = MIN_SECONDS_BETWEEN_CALLS - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def _track(self, response) -> None:
        u = response.usage_metadata
        self.usage.calls += 1
        if u:
            self.usage.input_tokens += u.prompt_token_count or 0
            self.usage.output_tokens += (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)


def _user(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


def _friendly(e: errors.APIError) -> str:
    if e.code in (400, 401, 403) and "API key" in (e.message or ""):
        return "Gemini API key missing or invalid. Put GEMINI_API_KEY in .env (free from aistudio.google.com/apikey)."
    if e.code == 429:
        return "Gemini free-tier limit reached (too many requests). Wait a while, or try again tomorrow."
    if e.code == 404:
        return f"Gemini model not found ({e.message}). Set EDUPATH_GEMINI_MODEL in .env to a current model."
    return f"Gemini API error {e.code}: {e.message}"


def _why_empty(response) -> str:
    if response.prompt_feedback and response.prompt_feedback.block_reason:
        return f"blocked: {response.prompt_feedback.block_reason}"
    if response.candidates:
        return f"finish_reason={response.candidates[0].finish_reason}"
    return "no candidates"
