"""
EduPath AI - web app.

    streamlit run app.py

Works without any LLM (offline parser + constraint solver). Set
EDUPATH_PROVIDER=ollama in .env to use a free local model (see .env.example).
"""

from __future__ import annotations

import streamlit as st

from edupath.agent import MODES, AgentResult, EduPathAgent
from edupath.curriculum import load_curriculum
from edupath.llm import make_planner
from edupath.models import PlanningGoals, StudentRecord
from edupath.requirements import audit
from edupath.schemas import LLMError
from edupath.students import list_sample_students, load_student

st.set_page_config(page_title="EduPath AI", page_icon="🎓", layout="wide")


@st.cache_resource
def get_curriculum():
    return load_curriculum()


cur = get_curriculum()
_planner = make_planner()                  # Ollama / Gemini / Claude from .env, or None (offline)
planner_name = _planner.name if _planner else None
has_key = planner_name is not None

# --------------------------------------------------------------------------- #
# Sidebar: who is the student, what are the settings
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("Student record")
    samples = {p.stem: p for p in list_sample_students()}
    choice = st.selectbox("Load a sample record", list(samples), index=list(samples).index("cse_96_credits")
                          if "cse_96_credits" in samples else 0)
    base = load_student(samples[choice])

    selectable = sorted(c for c in cur.courses)
    completed = st.multiselect("Completed courses (edit freely)", selectable,
                               default=[cur.normalize(c) for c in base.completed if cur.normalize(c) in cur.courses],
                               key=f"completed_{choice}")
    stages = ["not_started", "p1_done", "p2_done"]
    thesis_stage = st.selectbox("Thesis stage", stages, index=stages.index(base.thesis_stage),
                                format_func=lambda s: {"not_started": "Not started",
                                                       "p1_done": "Pre-thesis 1 done",
                                                       "p2_done": "Pre-thesis 2 done"}[s])
    student = StudentRecord(base.name, base.student_id, "CSE", completed, thesis_stage, base.interests)

    st.header("Planning settings")
    st.caption("Used unless your request says otherwise.")
    season = st.selectbox("First planned term", cur.policy.terms, index=0)
    year = st.number_input("Year", 2024, 2035, 2027)
    semesters = st.slider("Semesters to graduate in", 1, 12, 3)
    max_cr = st.slider("Max credits per semester", 6, 21, cur.policy.max_credits_per_semester)
    tags = sorted({t for c in cur.courses.values() for t in c.tags})
    interests = st.multiselect("Interests", tags, default=[t for t in base.interests if t in tags])

    st.header("Planner")
    mode = st.radio("Mode", list(MODES), format_func=lambda m: {
        "solver": "LLM + constraint solver (recommended)",
        "llm_verified": "LLM plans, validator repairs",
        "llm_only": "LLM only (unverified baseline)"}[m],
        disabled=not has_key, help="The two LLM modes exist for the research comparison.")
    if has_key:
        st.caption(f"AI model: {planner_name}")
    else:
        st.info("No LLM configured - running offline (rule-based parser + solver). "
                "Put EDUPATH_PROVIDER=ollama in .env to turn on the free local AI.")
        mode = "solver"

# --------------------------------------------------------------------------- #
# Main area
# --------------------------------------------------------------------------- #

st.title("🎓 EduPath AI")
st.caption(f"Academic planning agent for {cur.program}, BRAC University · "
           "LLM reasoning grounded by a deterministic constraint solver")

deg = audit(cur, student.completed)
c1, c2, c3 = st.columns(3)
c1.metric("Credits completed", f"{deg.credits_completed:g} / {deg.credits_required}")
c2.metric("Credits remaining", f"{deg.credits_remaining:g}")
c3.metric("Requirement groups done", f"{sum(g.done for g in deg.groups)} / {len(deg.groups)}")

request = st.text_area(
    "Ask EduPath",
    f"I am a CSE student. I have completed {deg.credits_completed:g} credits and want to graduate in "
    f"{semesters} semesters. Which courses should I take?",
    height=90,
)

if st.button("Plan my path", type="primary"):
    defaults = PlanningGoals(semesters, f"{season} {year}", max_cr, cur.policy.min_credits_per_semester,
                             interests=interests)
    agent = EduPathAgent(cur, make_planner())
    with st.spinner("Planning..." if mode == "solver" else f"{planner_name} is planning (this can take a minute)..."):
        try:
            st.session_state.result = agent.run(student, request, defaults, mode=mode)
        except LLMError as e:
            st.error(str(e))

result: AgentResult | None = st.session_state.get("result")

tab_plans, tab_audit, tab_graph, tab_trace = st.tabs(["Plans", "Degree audit", "Prerequisite map", "How it decided"])

with tab_plans:
    if not result:
        st.write("Press **Plan my path** to get started.")
    else:
        (st.success if result.feasibility.feasible else st.warning)(result.feasibility.message)
        for n in result.notes:
            st.info(n)
        st.markdown(result.explanation)

        for o in result.outcomes:
            badge = "✅ valid" if o.report.is_valid else f"❌ {len(o.report.errors)} error(s)"
            with st.expander(f"{o.plan.name} · {badge} · source: {o.origin}", expanded=True):
                if o.plan.rationale:
                    st.caption(o.plan.rationale)
                cols = st.columns(len(o.plan.semesters))
                for col, sem, cr in zip(cols, o.plan.semesters, o.report.semester_credits):
                    with col:
                        st.markdown(f"**{sem.term}** · {cr:g} cr")
                        for code in sem.courses:
                            c = cur.courses.get(code)
                            st.write(f"`{code}` {c.title if c else '⚠️ unknown course'}")
                        for m in sem.milestones:
                            st.caption(f"📌 {m}")
                for v in o.report.violations:
                    (st.error if v.severity == "error" else st.warning)(f"[{v.code}] {v.message}")

with tab_audit:
    for g in deg.groups:
        need = len(g.missing) if g.kind == "all" else g.picks_needed
        total = len(g.counted) + need
        st.markdown(f"**{'✅' if g.done else '⬜'} {g.name}** — {len(g.counted)}/{total}")
        st.progress(len(g.counted) / total if total else 1.0)
        if g.missing:
            st.caption("Still needed: " + ", ".join(g.missing))
        elif g.picks_needed:
            st.caption(f"Pick {g.picks_needed} more" +
                       (f" (at least {g.min_from_needed} from {', '.join(g.min_from)})" if g.min_from_needed else ""))

with tab_graph:
    st.caption("Remaining courses and their hard prerequisites (solid) / soft prerequisites (dashed). "
               "Grey = already completed.")
    remaining = {c for g in deg.groups if g.kind == "all" for c in g.missing}
    if result:
        remaining |= {c for o in result.outcomes for c in o.plan.all_courses() if c in cur.courses}
    done = {cur.normalize(c) for c in student.completed}
    dot = ["digraph G {", "rankdir=LR;", 'node [shape=box, style="rounded,filled", fillcolor="#e8f0fe"];']
    nodes = set(remaining)
    for c in remaining:
        course = cur.courses[c]
        for p in course.hard_prereqs:
            dot.append(f'"{p}" -> "{c}";')
            nodes.add(p)
        for p in course.soft_prereqs:
            dot.append(f'"{p}" -> "{c}" [style=dashed];')
            nodes.add(p)
    for n in nodes:
        if n in done:
            dot.append(f'"{n}" [fillcolor="#dddddd", fontcolor="#555555"];')
    dot.append("}")
    st.graphviz_chart("\n".join(dot))

with tab_trace:
    if result:
        st.write(f"Mode: **{MODES[result.mode]}**")
        for step in result.trace:
            st.write("• " + step)
        g = result.goals
        st.json({"target_semesters": g.target_semesters, "start_term": g.start_term,
                 "max_credits_per_semester": g.max_credits_per_semester, "interests": g.interests,
                 "avoid_courses": g.avoid_courses, "must_take_courses": g.must_take_courses,
                 "stated_credits_completed": g.stated_credits_completed})
        if result.first_attempt:
            st.subheader("The LLM's first attempt (before repairs)")
            for o in result.first_attempt:
                st.write(f"**{o.plan.name}** — {'valid' if o.report.is_valid else f'{len(o.report.errors)} error(s)'}")
                for v in o.report.errors:
                    st.caption(f"[{v.code}] {v.message}")
        if result.llm_calls:
            st.caption(f"{result.llm_calls} LLM calls · {result.input_tokens + result.output_tokens} tokens · "
                       f"{result.seconds}s")
