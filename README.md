# EduPath AI — Autonomous Academic Planning Agent

A course-planning agent for **BRAC University CSE** students. You ask in plain English —
*"I've completed 96 credits and want to graduate in three semesters. What should I take?"* —
and EduPath reads your record, audits your degree, checks every prerequisite, and proposes
several semester-by-semester plans, explaining anything that blocks your goal.

The research idea behind it: **an LLM is good at understanding people, but bad at
guaranteeing rules. A deterministic constraint solver is the opposite.** EduPath combines
them and lets you measure the difference.

---

## Quick start

```bash
cd edupath-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Web app** (works offline; no API key needed):

```bash
streamlit run app.py
```

**Command line:**

```bash
python cli.py
```

```bash
python cli.py --request "I have 96 credits, finish in 2 semesters, avoid CSE460, I like security"
```

**Tests:**

```bash
python -m pytest -q
```

**Turn on the AI — free, on your own computer** (natural-language understanding, LLM plans,
written explanations). Uses [Ollama](https://ollama.com) with **Qwen3 8B** (Apache 2.0 licence):

1. Install Ollama from https://ollama.com (or `brew install ollama`) and open it.
2. Download the model (about 5 GB, needs ~8 GB of free RAM):
   ```bash
   ollama pull qwen3:8b
   ```
3. Create your settings file (it already contains `EDUPATH_PROVIDER=ollama`):
   ```bash
   cp .env.example .env
   ```

That's it — the app, CLI and evaluation pick it up automatically. No key, no usage limits,
and student records never leave your machine. It is slower than a cloud model: a planning
call with thinking on takes a minute or more. Without an LLM, EduPath uses a small rule-based
parser and template explanations instead.

Other models: set `EDUPATH_OLLAMA_MODEL` in `.env` (e.g. `llama3.1:8b`, or `qwen3:4b` for speed)
after `ollama pull`-ing it. Gemini (free tier, not available for every Google project) and
Claude (paid) are also supported; see `.env.example`.

**Host it for free** on [Streamlit Community Cloud](https://share.streamlit.io): sign in with
GitHub → *Create app* → pick this repo, branch `main`, file `app.py` → *Deploy*. Pushing to
`main` redeploys it.

Free servers can't run a model themselves, so the hosted app calls **Groq's free API**
(`openai/gpt-oss-120b`) to read requests and write explanations, while the constraint solver
builds the plans. Get a free key at https://console.groq.com/keys, then in the app's
*Settings → Secrets* paste:

```toml
GROQ_API_KEY = "your-key"
EDUPATH_PROVIDER = "groq"
EDUPATH_SOLVER_ONLY = "1"
```

`EDUPATH_SOLVER_ONLY` hides the LLM planning modes (10–25K tokens per run) so visitors can't
use up the free daily quota (~200K tokens/day, shared by everyone). If the quota does run
out, the app keeps working with the offline parser and template explanations. Without any
secrets, the hosted app simply runs offline.

---

## How it works

```
 "I want to graduate in 3 semesters,         data/students/*.json
  I like AI"                                 (your completed courses)
        │                                              │
        ▼                                              ▼
 ┌──────────────┐                            ┌──────────────────┐
 │  INTERPRET   │  Gemini/Claude (or offline │   DEGREE AUDIT   │  which requirement
 │  llm_*.py    │  parser) → PlanningGoals   │  requirements.py │  groups are missing?
 └──────┬───────┘                            └────────┬─────────┘
        └──────────────────┬──────────────────────────┘
                           ▼
                 ┌────────────────────┐
                 │ FEASIBILITY CHECK  │  minimum semesters from credits,
                 │ solver.py          │  prerequisite chains, thesis timeline
                 └─────────┬──────────┘
                           ▼
   ┌───────────────────────┼─────────────────────────────┐
   │ mode "solver"         │ mode "llm_verified"         │ mode "llm_only"
   │ constraint solver     │ LLM proposes ──┐            │ LLM proposes
   │ builds 3 plans        │   ▲            ▼            │ (no checking —
   │ (recommended)         │   └── fix ◄── VALIDATOR     │  research baseline)
   └───────────────────────┼─────────────────────────────┘
                           ▼
                ┌──────────────────────┐
                │ VALIDATOR (always)   │  every plan is checked by the same
                │ validator.py         │  deterministic rules before you see it
                └──────────┬───────────┘
                           ▼
                 EXPLANATION (LLM or template)
```

**Two different "checkers"** — this is the key idea, so it is worth being precise:

* The **validator** (`validator.py`) *judges* a finished plan. It never builds anything.
  It is the referee for both the app and the research evaluation.
* The **solver** (`solver.py`) *builds* plans that obey the rules by construction
  (critical-path-first scheduling), then still runs them through the validator.

---

## Codebase tour — where to edit what

| File | What it does | Edit it when you want to… |
|---|---|---|
| `data/bracu_cse_curriculum.json` | **All curriculum facts**: courses, credits, prerequisites, requirement groups, credit limits, thesis rules, sources | fix a prerequisite, add an elective, change the credit limit, add course offerings |
| `data/students/*.json` | Hand-entered student records | add yourself or classmates (anonymised) |
| `edupath/models.py` | Data shapes only (`Course`, `Plan`, `Violation`, …). **Read this first.** | add a new field (e.g. CGPA) |
| `edupath/curriculum.py` | Loads the JSON; aliases (CSE161 = CSE110); prerequisite chains; term sequence | support a new JSON field |
| `edupath/requirements.py` | Degree audit: which groups are done / missing | change how courses count toward groups |
| `edupath/validator.py` | The rule checker; every rule has a code like `PREREQ_VIOLATION` | **add a new rule** (e.g. time-slot clashes, CGPA-based credit limits) |
| `edupath/solver.py` | Chooses courses + schedules them; feasibility analysis; plan strategies | add a planning strategy, change elective ranking |
| `edupath/prompts.py` | Every prompt sent to the LLM (shared by all providers) | tune how the LLM is instructed |
| `edupath/schemas.py` | The reply shapes every LLM must follow (Pydantic) | add a field you want the LLM to return |
| `edupath/llm.py` | Picks the provider from `.env` (`EDUPATH_PROVIDER`, else Groq → Gemini → Claude → offline) | add a new provider |
| `edupath/llm_ollama.py` | Local model calls via Ollama (free, default Qwen3 8B) | change local model, context size, timeouts |
| `edupath/llm_groq.py` | Groq cloud calls (free tier, used by the hosted app), retries on rate limits | change Groq model or token limits |
| `edupath/llm_gemini.py` | Gemini calls (free tier), rate-limit waits and retries | change Gemini model or thinking level |
| `edupath/llm_claude.py` | Claude calls (paid, optional) | change Claude model or effort |
| `edupath/request_parser.py` | Offline regex parser for requests | teach it new phrasings |
| `edupath/agent.py` | The pipeline that ties it all together; the three modes | change the agent's flow or repair loop |
| `app.py` | Streamlit web UI | change the interface |
| `cli.py` | Command-line interface | — |
| `evaluation/` | Research harness: scenarios, metrics, runner | add scenarios / metrics |
| `tests/` | 41 tests, including a fake LLM for the repair loop and fake Gemini / Ollama / Groq servers | add a test for every rule you add |

### Example: add a new rule

Say BRAC only allows 12 credits in Summer. In `validator.py`, inside the per-semester loop:

```python
if term.startswith("Summer") and load > 12:
    v.append(Violation("SUMMER_OVERLOAD", "error", f"{term} has {load:g} credits (Summer max 12).", term))
```

Add a test in `tests/test_validator.py`. Because the solver double-checks its plans with the
validator and the LLM receives the validator's messages during repair, both automatically
respect the new rule (the solver will drop plans that break it — teach `schedule()` the rule
too if you want it to plan around it).

---

## The data — where it comes from and how much to trust it

| What | Source | Status |
|---|---|---|
| CSE course list, **HP / SP prerequisites** | [cse.bracu.ac.bd/course/list](https://cse.bracu.ac.bd/course/list) | official |
| Degree structure: GenEd 39 + School Core 12 + Program Core 75 + Electives 6 + CSE400 4 = **136** | "Degree/Course Outline – CSE" linked from [cse.bracu.ac.bd/advising](https://cse.bracu.ac.bd/advising) | official |
| Course equivalents (CSE161/EEE103 → CSE110, EEE283 → CSE260, DEV101 → EMB101) | same outline | official |
| Thesis = P1 → P2 → CSE400 over three semesters, start after 75 credits | [cse.bracu.ac.bd/advising](https://cse.bracu.ac.bd/advising) | official |
| Prerequisites of MAT120/215/216, PHY112, ENG102 | not on those pages | **assumed** (`"verified": false`) |
| CSE elective list (CSE391, 427, 428, 431, 447) and their prerequisites | the department's elective sheet was not publicly readable | **assumed** (`"verified": false`) |
| Credit limits 9–15 per semester | — | **editable default** — confirm with your advisor |
| Course offerings per term | not published | not modelled (every course assumed offered every term; add `"offered": ["Spring","Fall"]` to a course to restrict it) |

EduPath tells the student when a plan uses an unverified course. **Before you use it for
real advising or publish results, check the `"verified": false` entries with your
department** and correct them in the JSON — no code changes needed.

The GenEd stream course lists (HUM101, PSY101, CST201, …) come from the outline, which
gives codes but not titles; they show as "GenEd Stream N option".

---

## Research: *Constraint-Grounded Planning for Reliable LLM-Based Academic Advisors*

**Research question.** Can combining LLM reasoning with a deterministic constraint solver
eliminate invalid academic plans?

**Three arms** (the agent's three modes, all judged by the same validator):

| Arm | Who plans | Who checks |
|---|---|---|
| `llm_only` | LLM (Qwen3 via Ollama, Gemini or Claude) | nobody (validator only *measures*) |
| `llm_verified` | LLM | validator → errors sent back → LLM repairs (≤ N rounds) |
| `solver` | constraint solver (LLM only interprets + explains) | validator |

All LLM arms receive the **same brief** (`prompts.build_planning_context`), so the only
difference between `llm_only` and `llm_verified` is the verification feedback. In the
evaluation, `llm_verified` runs **without** the solver fallback so it measures the LLM itself.

**Metrics** (`evaluation/metrics.py`):

* **Plan validity rate** — plans with zero rule errors
* **Prerequisite violation rate** — `PREREQ_VIOLATION` errors per course placement
* **Credit constraint satisfaction** — semesters within the credit limit
* **Requirement completion** — plans that actually finish the degree
* **Hallucinated courses** — course codes that don't exist
* **Feasibility accuracy** / **false-feasible rate** — does it correctly say "impossible"?
  (Telling a student a plan works when it doesn't is the most harmful failure.)
* **First-attempt vs. after-repair validity**, repair rounds, LLM calls, tokens, latency
* **User satisfaction** — not automatable; see `evaluation/user_study.md`

**Run it:**

```bash
python -m evaluation.run_eval
```

```bash
python -m evaluation.run_eval --modes solver,llm_only,llm_verified --repeats 3
```

A local model is slow, so start small:

```bash
python -m evaluation.run_eval --modes llm_only,llm_verified --scenarios A96-3sem,A96-2sem,D108-2sem
```

The first command needs no key (solver only). The second calls the LLM in `.env` — it prints how many runs
it will make and asks before starting. Results go to `evaluation/results/<timestamp>/`
(`runs.jsonl` per run, `summary.md` table). Scenarios live in `evaluation/scenarios.json`:
12 cases over 4 students, half of them deliberately impossible.

**Ideas to strengthen the paper**

* Add more real (anonymised) student records — more scenarios, more statistical power.
* Ablations: remove the degree audit from the brief; vary `--effort`; vary repair rounds;
  compare models via `EDUPATH_OLLAMA_MODEL` (e.g. `qwen3:8b` vs `llama3.1:8b` vs `qwen3:4b`) —
  "does the validator help regardless of the model?" is a strong result.
* Error taxonomy: `runs.jsonl` stores `first_attempt_errors` by rule code — chart which
  rules LLMs break most.
* Measure the interpretation step separately: give `run_eval` the natural-language request
  instead of explicit goals and compare extracted vs. intended goals.

---

## Limitations

* Credit limits, course offerings and section/time clashes are simplified (no timetable data).
* The thesis is modelled as the standard three-semester track; internship-based CSE400 or
  repeated pre-thesis semesters aren't modelled.
* Grades, retakes and CGPA-based rules aren't modelled yet (`StudentRecord` is the place to add them).
* Always confirm a plan with your BRAC University advisor.
