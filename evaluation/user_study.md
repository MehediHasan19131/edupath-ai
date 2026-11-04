# User satisfaction study (template)

Automatic metrics can't tell you whether students find a plan *useful*. Run a small study:

1. Recruit 10–20 CSE students (with consent; anonymise records).
2. Enter each student's completed courses into `data/students/` (or the app sidebar).
3. Show each student plans from two modes in random order, unlabeled (e.g. `llm_only` vs `solver`).
4. After each plan, ask (1 = strongly disagree … 5 = strongly agree):

| # | Statement |
|---|---|
| Q1 | I could register for this plan as written. |
| Q2 | The plan matches my goal (graduation time, workload). |
| Q3 | The plan reflects my interests. |
| Q4 | The explanation helped me understand prerequisite conflicts. |
| Q5 | I would trust this plan without checking it myself. |
| Q6 | Overall, I am satisfied with this plan. |

5. Optional: ask an academic advisor to rate each plan's correctness (blind to the mode).
6. Compare modes with a paired test (Wilcoxon signed-rank) per question.

Record answers in a CSV: `student, mode, Q1, Q2, Q3, Q4, Q5, Q6, comments`.
