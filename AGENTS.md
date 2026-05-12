# AGENTS.md — Rules for AI Agents Working in This Repo

Read this before touching anything. `docs/LLM_CONTEXT.md` is the authoritative codebase snapshot; read it next.

---

## Environment

- **Always use** `~/venvs/jupyter/bin/python3` and `~/venvs/jupyter/bin/jupyter`. Never use system Python.
- Install packages with `~/venvs/jupyter/bin/pip install <pkg>`.
- Working directory is the repo root. `Path('..').resolve()` inside a notebook kernel resolves to the repo root because notebooks live in `notebooks/` and are executed with CWD = `notebooks/`.

---

## Running Notebooks

**Use `run_pipeline.py` to execute notebooks — do not run them directly.**

```bash
# Full pipeline for all subreddits
~/venvs/jupyter/bin/python3 run_pipeline.py

# Specific subreddit(s)
~/venvs/jupyter/bin/python3 run_pipeline.py --subreddits gradadmissions

# Resume from a step (two-digit prefix, e.g. "04")
~/venvs/jupyter/bin/python3 run_pipeline.py --start-from 04

# Skip comparison notebook
~/venvs/jupyter/bin/python3 run_pipeline.py --skip-comparison
```

`run_pipeline.py` injects the correct `SUBREDDIT` value into each notebook's config cell via regex and runs it via `nbconvert`. **Source notebooks are never modified.** Temp output notebooks are cleaned up automatically.

To run a single notebook manually (e.g. for debugging):
```bash
~/venvs/jupyter/bin/jupyter nbconvert --to notebook --execute \
  --output /tmp/out.ipynb \
  --ExecutePreprocessor.timeout=7200 \
  notebooks/06_did_analysis.ipynb
```

---

## Editing Notebooks

- Use `NotebookEdit` with `cell_id` to target specific cells. Cell IDs are listed in `docs/LLM_CONTEXT.md`.
- Always read the cell first before editing.
- **Do not change the `SUBREDDIT` variable directly in notebook source** — `run_pipeline.py` injects it at runtime. If you need to add a new subreddit, update `run_pipeline.py` (the `PRE_CLEANED` set if it has pre-cleaned data, and the `--subreddits` default).
- Config cells (the ones with `SUBREDDIT = '...'  # change to`) must keep the `# change to` comment — `inject_subreddit()` in `run_pipeline.py` uses this as a sentinel.

---

## Data Layout

All pipeline outputs go to `data/processed/{SUBREDDIT}/`. The three supported subreddits are:

| Subreddit | Raw data location | NB01 needed? |
|-----------|------------------|--------------|
| `gradadmissions` | `r_gradadmissions_posts.jsonl` (repo root) + `data/raw/r_gradadmissions_2022_posts.jsonl` | Yes |
| `mscs` | `data/raw/r_MSCS_posts.jsonl` + `data/raw/r_MSCS_2022_posts.jsonl` | Yes |
| `mba` | `data/mba/posts_clean.jsonl.gz` (pre-cleaned, already includes 2022) | **No** — decompress `.gz` to `data/processed/mba/` |

All three subreddits cover **Aug 2022–Jul 2025 (three admission cycles)**. Cycle numbering is chronological: `cycle=1` → 2022 cycle, `cycle=2` → 2023 cycle, `cycle=3` → 2024 cycle. NB01 concatenates the 2022 and main raw files per subreddit. Downstream notebooks derive the cycle list from the data — do not hardcode `[1, 2]` or `[1, 2, 3]`; iterate over `sorted(panel['cycle'].unique())` instead.

`data/processed/mba/posts_clean.jsonl` and `comments_clean.jsonl` are **not in git** — decompress from `data/mba/*.jsonl.gz` before running the MBA pipeline from NB03.

---

## Critical Schema Rules

- **Cleaned JSONL fields**: `id`, `author`, `created_dt` (ISO string), `clean_text`, `post_id` (comments), `score`
- **Raw JSONL fields**: `id`, `author`, `created_utc` (Unix int), `selftext`/`body`, `link_id` (`"t3_<post_id>"`)
- NB03, NB04, and NB08 all read from **cleaned** files. Do not reintroduce raw field names (`created_utc`, `body`, `link_id`).
- `panel_scores.parquet` has `pre_mh_score`/`post_mh_score` columns. `post_level_scores.parquet` has `mean_mh_score` (not `mh_score`). NB06 renames to `mh_score` at runtime in long format.
- `panel_scores.parquet` contains `exposure_intensity` (float, BART top-neg score) but NOT `exposure_prob` — NB06 reads `exposure_prob` separately from `exposure_labels.parquet`. GPS weights are nearly degenerate (mean ≈ 1.0) in current data.

---

## Known Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `KeyError: 'record_id'` | Using raw field name instead of cleaned `id` | Use `r['id']` not `r['record_id']` |
| `NameError: COMMENTS_RAW` | Old variable name | Use `COMMENTS_CLEAN` |
| `FileNotFoundError` in NB06 | `os.chdir()` was present, making `Path('..').resolve()` go above project root | Do not add `os.chdir()` to NB06 |
| NB06 kernel dies instantly (7s) | OOM — Linux OOM killer fires when swap is full | Close browser tabs, retry |
| `ArrowInvalid: Could not convert 'pooled'` | Mixed int/str in `cycle` column when saving parquet | `df['cycle'] = df['cycle'].astype(str)` before `to_parquet()` |
| NB05 timeout after 2h | Reddit API breadth fetch for large subreddit | Re-run — checkpoint resumes automatically from `breadth_checkpoint.jsonl` |
| SUBREDDIT injection silently fails | Config cell uses double-space `SUBREDDIT  =` | `inject_subreddit()` uses `r"SUBREDDIT\s*=\s*'"` regex — verify it matches |

---

## What NOT to Do

- Do not modify `SUBREDDIT = '...'` in notebook source files directly.
- Do not use `os.chdir()` in any notebook — it breaks `Path('..').resolve()`.
- Do not add raw-field references (`created_utc`, `body`, `link_id`) to NB03/NB04/NB08.
- Do not commit `posts_clean.jsonl`, `comments_clean.jsonl`, or `pipeline_logs/` — they are in `.gitignore`.
- Do not commit `_run_*.ipynb` temp notebooks — also gitignored.
- Do not run notebooks with system Python — only `~/venvs/jupyter/bin/`.
- Do not create new intermediate output files outside `data/processed/{SUBREDDIT}/`.
- Do not hardcode cycle lists (`for cycle in [1, 2]` etc.) in any notebook — derive from the loaded data.

---

## Git Hygiene

- Commit parquets and figures when the pipeline produces new results.
- Update `docs/LLM_CONTEXT.md` after any schema change, new notebook, or new subreddit.
- Typical commit scope: `data/processed/`, `figures/`, `notebooks/`, `docs/LLM_CONTEXT.md`.
- Do not force-push. Do not amend published commits.

---

## Verification Checklist (after pipeline changes)

1. `~/venvs/jupyter/bin/python3 -c "import ast; ast.parse(open('run_pipeline.py').read())"` — syntax check
2. Simulate injection: check each notebook has `SUBREDDIT\s*=\s*'` + `# change to` in its config cell
3. Spot-check output parquet row counts against `docs/LLM_CONTEXT.md` schema section
4. For NB07: verify `comparison_summary.parquet` has rows for all 3 subreddits, `cycle` column is string dtype
