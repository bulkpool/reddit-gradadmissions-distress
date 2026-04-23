# Agent Rules

Read this file before making any changes to the repository.

---

## Step 1 — Always Read LLM_CONTEXT.md First

Before touching any notebook, data file, or script, read `docs/LLM_CONTEXT.md`. It contains:
- Full data schemas (column names, dtypes, row counts)
- Notebook cell IDs for targeted edits
- Key function signatures
- Known gotchas and field name inconsistencies
- Current results and open issues

Do not re-explore the codebase if the answer is already in `LLM_CONTEXT.md`.

---

## Step 2 — Environment

Always use the project venv. Never use system Python.

```bash
~/venvs/jupyter/bin/python3
~/venvs/jupyter/bin/pip
~/venvs/jupyter/bin/jupyter
```

---

## Step 3 — Making Changes

### Notebooks
- Use the `NotebookEdit` tool with the `cell_id` from `LLM_CONTEXT.md` to target specific cells.
- Do not re-read the entire notebook if you only need to change one cell.
- After changing a notebook, update the relevant cell_id entry in `LLM_CONTEXT.md` if the cell content changed significantly.

### Data files
- Parquet files are outputs — regenerate by running the upstream notebook, don't edit them directly.
- Do not commit raw JSONL files (`r_gradadmissions_*.jsonl`) — they are too large and excluded from git.

### Code patterns
- `mh_score` = `mean(sigmoid(clf.decision_function(texts)))` — never use `predict_proba`.
- Exposure = user commented on an anchor thread, matched via `link_id` (raw) or `post_id` (cleaned).
- PSM caliper = 0.05; features = `['pre_mh_score', 'log1p_n_posts_pre']` ± `community_breadth_log`.

---

## Step 4 — After Making Changes, Update LLM_CONTEXT.md

This is mandatory. After every session that changes:

| What changed | What to update in LLM_CONTEXT.md |
|-------------|----------------------------------|
| Data schema (new column, renamed column) | The relevant schema table |
| New figure added | Figure → Notebook Mapping table |
| Function signature changed | Key Functions section |
| New notebook cell added | Cell IDs table for that notebook |
| Results changed | Current Status & Open Issues section |
| New file added or removed | Repo Layout + What Is/Isn't in Git |
| Bug fixed or gotcha discovered | Known Gotchas section |

Update the **Last updated** date at the top of `LLM_CONTEXT.md`.

---

## Step 5 — Committing

- Stage specific files by name — never `git add -A` or `git add .`.
- Always include `docs/LLM_CONTEXT.md` in commits that change code or data.
- Commit message format: `<action>: <what changed>` (e.g. `Fix NB08 paths + update LLM_CONTEXT`).
- Pull remote changes before committing: `git fetch && git status`.
- Co-author line: `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`

---

## Quick Reference — Common Pitfalls

| Pitfall | Correct behaviour |
|---------|------------------|
| Using system Python | Use `~/venvs/jupyter/bin/python3` |
| Reading `posts_clean.jsonl` directly | It's not in git — run NB01 first or note it's missing |
| Assuming `mh_score` column exists in parquets | It's `mean_mh_score` in `post_level_scores.parquet` |
| Using `created_utc` on cleaned files | Cleaned files use `created_dt` (ISO string) |
| Anchor threshold = 0.45 | Actual code uses 0.5 (OR across dimensions) |
| `causalimpact` datetime index | Use integer positions — see NB08 cell `b0000021`/`b0000022` |
| `community_breadth_log` always in PSM | Only if `breadth_coverage >= 0.95` (almost never true) |
