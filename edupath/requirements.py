"""
Degree audit: given the courses a student has passed, which requirement groups
are done and what is still missing?

Each passed course counts toward at most ONE group. Groups are filled in the
order they appear in the curriculum JSON, so a second Arts course first fills
"Stream 3", and the extra one spills into "GenEd electives".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .curriculum import Curriculum


@dataclass
class GroupStatus:
    id: str
    name: str
    kind: str
    counted: list[str]                      # passed courses counted here
    missing: list[str]                      # kind == "all": courses still required
    picks_needed: int                       # kind == "choose": how many more to pick
    min_from_needed: int                    # of those picks, how many must come from `min_from`
    options: list[str] = field(default_factory=list)   # kind == "choose": eligible options
    min_from: list[str] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return not self.missing and self.picks_needed == 0


@dataclass
class DegreeAudit:
    groups: list[GroupStatus]
    credits_completed: float
    credits_required: int
    unrecognized: list[str]                 # codes not in the curriculum at all
    not_counted: list[str]                  # known courses that fill no group (extra)

    @property
    def is_complete(self) -> bool:
        return all(g.done for g in self.groups)

    @property
    def unmet(self) -> list[GroupStatus]:
        return [g for g in self.groups if not g.done]

    @property
    def credits_remaining(self) -> float:
        return max(0.0, self.credits_required - self.credits_completed)


def audit(curriculum: Curriculum, completed: list[str]) -> DegreeAudit:
    passed = []
    for code in completed:
        c = curriculum.normalize(code)
        if c not in passed:
            passed.append(c)

    unrecognized = [c for c in passed if c not in curriculum.courses]
    unused = [c for c in passed if c in curriculum.courses]
    statuses: list[GroupStatus] = []

    for g in curriculum.groups:
        if g.kind == "all":
            counted = [c for c in g.courses if c in unused]
            missing = [c for c in g.courses if c not in unused]
            for c in counted:
                unused.remove(c)
            statuses.append(GroupStatus(g.id, g.name, g.kind, counted, missing, 0, 0))
        else:
            # Fill the "at least N from min_from" part first, then anything else.
            from_min = [c for c in unused if c in g.min_from][: g.min_from_count]
            still_reserved = g.min_from_count - len(from_min)   # slots only a min_from course may fill
            others = [c for c in unused if c in g.courses and c not in from_min]
            counted = from_min + others[: max(0, g.count - len(from_min) - still_reserved)]
            for c in counted:
                unused.remove(c)
            statuses.append(GroupStatus(
                g.id, g.name, g.kind, counted, [],
                picks_needed=g.count - len(counted),
                min_from_needed=max(0, g.min_from_count - len(from_min)),
                options=list(g.courses), min_from=list(g.min_from),
            ))

    return DegreeAudit(
        groups=statuses,
        credits_completed=curriculum.credits(passed),
        credits_required=curriculum.policy.total_credits_required,
        unrecognized=unrecognized,
        not_counted=unused,
    )
