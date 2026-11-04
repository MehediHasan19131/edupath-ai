"""
A tiny rule-based reader for planning requests, used when no LLM API key is
set. It understands phrases like "3 semesters", "three semesters",
"95 credits", "max 12 credits per semester", "avoid CSE460", "interested in AI".

An LLM does this far better (see interpret_request in llm_gemini.py); this keeps the app
usable offline and is a handy baseline.
"""

from __future__ import annotations

import re

from .schemas import GoalsExtraction

NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}

INTEREST_KEYWORDS = {
    "ai": ["ai", "artificial intelligence", "machine learning", "ml", "deep learning", "nlp", "llm"],
    "data": ["data", "database", "analytics"],
    "web": ["web", "full stack", "frontend", "backend", "software engineering"],
    "security": ["security", "cyber", "cryptography"],
    "networks": ["network"],
    "robotics": ["robot"],
    "graphics": ["graphics", "vision", "image"],
    "hardware": ["hardware", "embedded", "vlsi", "electronics"],
}

COURSE = r"([A-Za-z]{3})\s?-?(\d{3})"


def _num(token: str) -> int:
    return int(token) if token.isdigit() else NUMBER_WORDS[token]


def parse_request(text: str) -> GoalsExtraction:
    t = text.lower()
    num = r"(\d+|" + "|".join(NUMBER_WORDS) + r")"

    sem = re.search(num + r"\s+(?:more\s+)?(?:semesters?|terms?)", t)
    max_cr = re.search(r"(?:max(?:imum)?|at most|no more than|up to)\s+(\d+)\s+credits?", t) or \
        re.search(r"(\d+)\s+credits?\s+(?:per|a|each)\s+(?:semester|term)", t)
    done = re.search(r"(?:completed|finished|done|have)\s+(?:about\s+)?(\d+)\s+credits?", t)

    avoid = []
    for m in re.finditer(r"(?:avoid|skip|don't want|do not want|not take)\s+((?:" + COURSE + r"[\s,and]*)+)", text,
                         flags=re.I):
        avoid += [a + b for a, b in re.findall(COURSE, m.group(1))]
    must = []
    for m in re.finditer(r"(?:must take|want to take|definitely take)\s+((?:" + COURSE + r"[\s,and]*)+)", text,
                         flags=re.I):
        must += [a + b for a, b in re.findall(COURSE, m.group(1))]

    interests = [tag for tag, words in INTEREST_KEYWORDS.items()
                 if any(re.search(r"\b" + re.escape(w) + r"\b", t) for w in words)]

    return GoalsExtraction(
        target_semesters=_num(sem.group(1)) if sem else None,
        max_credits_per_semester=int(max_cr.group(1)) if max_cr else None,
        stated_credits_completed=int(done.group(1)) if done else None,
        interests=interests,
        avoid_courses=[c.upper() for c in avoid],
        must_take_courses=[c.upper() for c in must],
        summary=text.strip()[:200],
    )
