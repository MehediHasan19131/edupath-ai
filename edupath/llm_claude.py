"""
Claude provider (paid - needs ANTHROPIC_API_KEY).

Four jobs, each one API call:
  interpret_request() - free text  -> GoalsExtraction   (structured output)
  propose_plans()     - brief      -> PlanProposals     (structured output)
  repair_plans()      - feedback   -> PlanProposals     (same conversation, structured output)
  explain()           - results    -> friendly text
"""

from __future__ import annotations

import os

import anthropic
from pydantic import BaseModel

from .prompts import EXPLAIN_SYSTEM, INTERPRET_SYSTEM, PLANNER_SYSTEM
from .schemas import GoalsExtraction, LLMError, PlanProposals, Usage

DEFAULT_CLAUDE_MODEL = os.getenv("EDUPATH_CLAUDE_MODEL", "claude-opus-5-5")

# If Claude's safety classifiers decline a request, the API retries it on a
# suitable fallback model inside the same call instead of failing.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudePlanner:
    def __init__(self, model: str = DEFAULT_CLAUDE_MODEL, client: anthropic.Anthropic | None = None,
                 plan_effort: str = "high"):
        self.model = model
        self.name = f"Claude ({model})"
        self.client = client or anthropic.Anthropic()
        self.plan_effort = plan_effort      # low | medium | high | xhigh | max
        self.usage = Usage()

    # ---- the four jobs ----------------------------------------------------- #

    def interpret_request(self, request_text: str) -> GoalsExtraction:
        parsed, _ = self._parse(INTERPRET_SYSTEM, [{"role": "user", "content": request_text}],
                                GoalsExtraction, effort="low")
        return parsed

    def propose_plans(self, brief: str) -> tuple[PlanProposals, list]:
        """Returns the proposals AND the conversation so far (needed for repairs)."""
        messages: list = [{"role": "user", "content": brief}]
        parsed, response = self._parse(PLANNER_SYSTEM, messages, PlanProposals, self.plan_effort)
        messages.append({"role": "assistant", "content": response.content})
        return parsed, messages

    def repair_plans(self, messages: list, feedback: str) -> tuple[PlanProposals, list]:
        """Continue the same conversation with the validator's findings."""
        messages = messages + [{"role": "user", "content": feedback}]
        parsed, response = self._parse(PLANNER_SYSTEM, messages, PlanProposals, self.plan_effort)
        messages.append({"role": "assistant", "content": response.content})
        return parsed, messages

    def explain(self, facts: str) -> str:
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=EXPLAIN_SYSTEM,
                messages=[{"role": "user", "content": facts}],
                output_config={"effort": "medium"},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.APIError as e:
            raise LLMError(f"Claude API error: {e}") from e
        self._track(response)
        self._check_refusal(response)
        return "".join(b.text for b in response.content if b.type == "text").strip()

    # ---- plumbing ---------------------------------------------------------- #

    def _parse(self, system: str, messages: list, schema: type[BaseModel], effort: str):
        try:
            response = self.client.beta.messages.parse(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=messages,
                output_format=schema,
                output_config={"effort": effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as e:
            raise LLMError("Claude API key missing or invalid. Set ANTHROPIC_API_KEY.") from e
        except anthropic.RateLimitError as e:
            raise LLMError("Claude API rate limit reached - wait a minute and retry.") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Could not reach the Claude API - check your internet connection.") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Claude API error {e.status_code}: {e.message}") from e

        self._track(response)
        self._check_refusal(response)
        if response.parsed_output is None:
            raise LLMError(f"Claude returned no structured output (stop_reason={response.stop_reason}).")
        return response.parsed_output, response

    def _track(self, response) -> None:
        self.usage.calls += 1
        self.usage.input_tokens += response.usage.input_tokens
        self.usage.output_tokens += response.usage.output_tokens

    @staticmethod
    def _check_refusal(response) -> None:
        if response.stop_reason == "refusal":
            why = response.stop_details.explanation if response.stop_details else "no details"
            raise LLMError(f"Claude declined this request ({why}).")
