"""
Loads the curriculum JSON and answers questions about it:
  - "what is CSE221?"            -> curriculum.get("CSE221")
  - "CSE161 is the same as?"     -> curriculum.normalize("CSE161")  == "CSE110"
  - "which term comes next?"     -> next_term("Fall 2026")          == "Spring 2027"

To change courses, credits or prerequisites, edit data/bracu_cse_curriculum.json,
not this file.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .models import Course, Policy, RequirementGroup, ThesisPolicy

DEFAULT_CURRICULUM = Path(__file__).resolve().parent.parent / "data" / "bracu_cse_curriculum.json"


class Curriculum:
    def __init__(self, courses: dict[str, Course], groups: list[RequirementGroup], policy: Policy,
                 program: str, sources: dict[str, str]):
        self.courses = courses
        self.groups = groups
        self.policy = policy
        self.program = program
        self.sources = sources
        # alias -> canonical code, e.g. "CSE161" -> "CSE110"
        self._aliases = {alias: c.code for c in courses.values() for alias in c.aliases}

    # ---- lookups ---------------------------------------------------------- #

    def normalize(self, code: str) -> str:
        """Clean up a course code typed by a human or an LLM: ' cse 221 ' -> 'CSE221'."""
        code = code.upper().replace(" ", "").replace("-", "").strip()
        return self._aliases.get(code, code)

    def get(self, code: str) -> Course | None:
        return self.courses.get(self.normalize(code))

    def credits(self, codes) -> float:
        return sum(self.courses[c].credits for c in codes if c in self.courses)

    def all_prereqs(self, code: str) -> set[str]:
        """Every hard prerequisite, transitively (CSE420 -> CSE321, CSE221, CSE220, ...)."""
        seen: set[str] = set()
        stack = list(self.courses[code].hard_prereqs) if code in self.courses else []
        while stack:
            p = stack.pop()
            if p not in seen:
                seen.add(p)
                if p in self.courses:
                    stack.extend(self.courses[p].hard_prereqs)
        return seen

    def chain_lengths(self, codes: set[str]) -> dict[str, int]:
        """For each course in `codes`: the length of the longest hard-prerequisite
        chain that *starts* at it, counting only courses inside `codes`.

        If CSE341 -> CSE360 -> CSE461 are all still to do, CSE341 gets 3, which
        means it must be taken at least 3 semesters before graduation.
        This is the "critical path" idea from project scheduling.
        """
        dependents: dict[str, list[str]] = {c: [] for c in codes}
        for c in codes:
            for p in self.courses[c].hard_prereqs:
                if p in dependents:
                    dependents[p].append(c)

        @lru_cache(maxsize=None)
        def length(c: str) -> int:
            return 1 + max((length(d) for d in dependents[c]), default=0)

        return {c: length(c) for c in codes}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_curriculum(path: str | Path = DEFAULT_CURRICULUM) -> Curriculum:
    raw = json.loads(Path(path).read_text())

    courses: dict[str, Course] = {}
    for c in raw["courses"]:
        courses[c["code"]] = Course(
            code=c["code"],
            title=c["title"],
            credits=c["credits"],
            category=c["category"],
            hard_prereqs=tuple(c.get("hard_prereqs", [])),
            soft_prereqs=tuple(c.get("soft_prereqs", [])),
            min_credits_completed=c.get("min_credits_completed", 0),
            offered=tuple(c.get("offered", [])),
            tags=tuple(c.get("tags", [])),
            aliases=tuple(c.get("aliases", [])),
            verified=c.get("verified", True),
        )

    # Option pools (GenEd stream lists) become simple 3-credit courses.
    pools = {k: v for k, v in raw.get("option_pools", {}).items() if not k.startswith("_")}
    for pool_name, codes in pools.items():
        stream = pool_name.replace("gened_", "").replace("stream", "Stream ")
        for code in codes:
            courses.setdefault(code, Course(code=code, title=f"GenEd {stream} option",
                                            credits=3, category="GenEd Option"))

    def expand(items: list[str]) -> list[str]:
        """'@gened_stream3' in a requirement means 'every code in that pool'."""
        out: list[str] = []
        for item in items:
            for code in (pools[item[1:]] if item.startswith("@") else [item]):
                if code not in out:
                    out.append(code)
        return out

    groups = [
        RequirementGroup(
            id=g["id"], name=g["name"], kind=g["kind"],
            courses=expand(g["courses"]),
            count=g.get("count", 0),
            min_from=expand(g.get("min_from", [])),
            min_from_count=g.get("min_from_count", 0),
        )
        for g in raw["requirements"]
    ]

    p = raw["policy"]
    policy = Policy(
        total_credits_required=p["total_credits_required"],
        terms=p["terms"],
        min_credits_per_semester=p["min_credits_per_semester"],
        max_credits_per_semester=p["max_credits_per_semester"],
        thesis=ThesisPolicy(**{k: p["thesis"][k] for k in ("course", "stages", "min_credits_to_start")}),
    )

    return Curriculum(courses, groups, policy, raw["program"], raw.get("sources", {}))


# --------------------------------------------------------------------------- #
# Term helpers  (BRACU runs Spring, Summer, Fall)
# --------------------------------------------------------------------------- #


def next_term(term: str, terms: list[str]) -> str:
    season, year = term.split()
    i = terms.index(season)
    if i + 1 < len(terms):
        return f"{terms[i + 1]} {year}"
    return f"{terms[0]} {int(year) + 1}"


def term_sequence(start: str, n: int, terms: list[str]) -> list[str]:
    out = [start]
    while len(out) < n:
        out.append(next_term(out[-1], terms))
    return out[:n]
