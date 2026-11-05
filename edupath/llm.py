"""
Picks which LLM EduPath talks to.

    make_planner()            -> OllamaPlanner, GroqPlanner, GeminiPlanner, ClaudePlanner, or None (offline)

Choice order:
  1. EDUPATH_PROVIDER in .env ("ollama", "groq", "gemini", "claude" or "offline"), if set.
     Ollama (free, local) is only used when chosen here - it needs no key to detect.
  2. otherwise whichever key exists: GROQ_API_KEY (free), GEMINI_API_KEY (free), ANTHROPIC_API_KEY
  3. no key at all -> None, and EduPath runs offline (rule-based parser + solver)

To add another provider, write a class with the methods listed in schemas.py
and add it here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def make_planner(provider: str | None = None, plan_effort: str = "high", model: str | None = None):
    """model overrides the provider's default model (e.g. "openai/gpt-oss-120b")."""
    load_dotenv(ENV_FILE)
    kwargs = {"plan_effort": plan_effort, **({"model": model} if model else {})}
    provider = (provider or os.getenv("EDUPATH_PROVIDER") or "").strip().lower()
    if not provider:
        if os.getenv("GROQ_API_KEY"):
            provider = "groq"
        elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            provider = "gemini"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "claude"
        else:
            provider = "offline"

    if provider == "ollama":
        from .llm_ollama import OllamaPlanner
        return OllamaPlanner(**kwargs)
    if provider == "groq":
        from .llm_groq import GroqPlanner
        return GroqPlanner(**kwargs)
    if provider == "gemini":
        from .llm_gemini import GeminiPlanner          # imported here so Claude-only users
        return GeminiPlanner(**kwargs)               # don't need google-genai, and vice versa
    if provider == "claude":
        from .llm_claude import ClaudePlanner
        return ClaudePlanner(**kwargs)
    if provider == "offline":
        return None
    raise ValueError(f"Unknown EDUPATH_PROVIDER '{provider}' (use ollama, groq, gemini, claude or offline).")
