# Quickstart — Reproducing the Project

This guide walks you through setting up the environment and running the pipeline for one or all three subreddits.

For a per-notebook description, see [Pipeline](pipeline.md). For the result tables, see [Results](results.md).

---

## Prerequisites

### Python Environment

The project uses a Jupyter venv at `~/venvs/jupyter/`. Always use the venv binaries — never system Python.

```bash
~/venvs/jupyter/bin/pip install \
    scikit-learn statsmodels joblib \
    pandas numpy matplotlib requests \
    causalimpact==0.2.6 \
    torch transformers
```

Requirements:
- `causalimpact 0.2.6` is incompatible with pandas 2.x — NB08 patches `pandas.core.dtypes.common.is_datetime_or_timedelta_dtype` at runtime.
- `torch` + `transformers` are needed by NB03 for `facebook/bart-large-mnli` zero-shot inference.

### Raw Data

| Subreddit | Files | Location | In git? |
|-----------|-------|----------|---------|
| r/GradAdmissions | `r_gradadmissions_posts.jsonl`, `r_gradadmissions_comments.jsonl` (Aug 2023 onwards) | repo root | ❌ |
| r/GradAdmissions | `r_gradadmissions_2022_posts.jsonl`, `r_gradadmissions_2022_comments.jsonl` (Aug 2022 cycle) | `data/raw/` | ❌ |
| r/MSCS | `r_MSCS_posts.jsonl`, `r_MSCS_comments.jsonl`, `r_MSCS_2022_posts.jsonl`, `r_MSCS_2022_comments.jsonl` | `data/raw/` | ❌ |
| r/MBA | `posts_clean.jsonl.gz`, `comments_clean.jsonl.gz` (pre-cleaned, includes 2022) | `data/mba/` | ✅ |

For r/MBA, decompress the gzipped files into `data/processed/mba/` once before running anything past NB01:

```bash
mkdir -p data/processed/mba
gunzip -c data/mba/posts_clean.jsonl.gz    > data/processed/mba/posts_clean.jsonl
gunzip -c data/mba/comments_clean.jsonl.gz > data/processed/mba/comments_clean.jsonl
```

---

## What's Already in the Repo

The repo ships with all pipeline outputs and figures pre-computed — you can read [Results](results.md) without running anything.

| Pre-computed | Skips |
|--------------|-------|
| `models/clf_*.joblib` | SVM classifier training (Arctic Shift API + ~15 min training) |
| `data/processed/{sub}/anchor_posts.parquet` | NB03 keyword + BART NLI pass |
| `data/processed/{sub}/exposure_labels.parquet` | NB03 exposure classification |
| `data/processed/{sub}/panel_scores.parquet` | NB04 panel scoring |
| `data/processed/{sub}/panel_scores_alt.parquet` | NB08 re-scored alt panel |
| `data/processed/{sub}/post_level_scores.parquet` | NB04 post-level dataset |
| `data/processed/{sub}/dose_exposure.parquet` | NB04 dose-response dataset |
| `data/processed/{sub}/user_community_breadth.parquet` | NB05 Arctic Shift breadth (~3 h API run per subreddit) |
| `data/processed/{sub}/breadth_checkpoint.jsonl` | NB05 resumable checkpoint |
| `data/processed/{sub}/did_summary.csv` | NB06 result tables |
| `data/processed/{sub}/parallel_trends_test_v2.csv` | NB06 §4a formal pre-trend test |
| `data/processed/comparison_summary.parquet` | NB07 cross-subreddit summary |
| `data/processed/combined_pooled_did.csv` | NB07 §9 headline pooled DiD |
| `figures/*.png` | All output figures |

---

## Running the Pipeline

Use `run_pipeline.py` — it injects the correct `SUBREDDIT` value into each notebook's config cell at runtime and executes via `nbconvert`. Source notebooks are never modified.

```bash
# Full pipeline for all three subreddits + comparison + anchor characterisation
~/venvs/jupyter/bin/python3 run_pipeline.py

# Single subreddit
~/venvs/jupyter/bin/python3 run_pipeline.py --subreddits gradadmissions
~/venvs/jupyter/bin/python3 run_pipeline.py --subreddits mscs
~/venvs/jupyter/bin/python3 run_pipeline.py --subreddits mba

# Resume from a step (skip notebooks whose 2-digit prefix < NN)
~/venvs/jupyter/bin/python3 run_pipeline.py --start-from 04

# Skip NB07 cross-subreddit comparison
~/venvs/jupyter/bin/python3 run_pipeline.py --skip-comparison

# Skip NB09 anchor characterisation
~/venvs/jupyter/bin/python3 run_pipeline.py --skip-validation
```

Each subreddit runs `NB01 → NB03 → NB04 → NB05 → NB06 → NB08` in sequence, with NB01 skipped for MBA (`PRE_CLEANED = {'mba'}`). After all three subreddits complete, NB07 and NB09 run once across all of them.

Logs land in `pipeline_logs/{timestamp}_{slug}.log`.

---

## Step-by-Step (per subreddit)

| Step | Notebook | Approx runtime | Notes |
|------|----------|---------------|-------|
| 1 | `01_clean_corpus.ipynb` | ~5 min | Reads raw JSONL, normalises text, derives `post_id` from `link_id`. Skipped for MBA. |
| 2 | `02_train_classifiers.ipynb` | ~15 min | One-off — do not re-run. Models are in git. |
| 3 | `03_exposure_labels.ipynb` | ~5–30 min | BART NLI inference dominates; faster with GPU. |
| 4 | `04_panel_scores.ipynb` | ~10 min | Re-scores all panel-user text. |
| 5 | `05_collect_community_breadth.ipynb` | ~1–3 h | Arctic Shift API; checkpointed. Skip if `user_community_breadth.parquet` exists. |
| 6 | `06_did_analysis.ipynb` | ~2 min | Main per-subreddit results. |
| 7 | `08_alt_analysis.ipynb` | ~15 min | Re-scores raw text + runs CausalImpact BSTS. |

After all three subreddits:

| Step | Notebook | Approx runtime | Notes |
|------|----------|---------------|-------|
| 8 | `07_comparison_analysis.ipynb` | ~3 min | Cross-subreddit comparison + combined pooled DiD (§9). |
| 9 | `09_nli_anchor_validation.ipynb` | <1 min | Reads existing BART fields; no inference. |

---

## Running a Single Notebook Manually

`run_pipeline.py` is the supported path. If you really need to invoke a notebook by hand (e.g. for debugging), you must inject the `SUBREDDIT` value yourself or edit the config cell:

```bash
~/venvs/jupyter/bin/jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=7200 \
  --output /tmp/out.ipynb \
  notebooks/06_did_analysis.ipynb
```

The config cell sentinel is the `# change to` comment on the `SUBREDDIT = '...'` line — `run_pipeline.py`'s `inject_subreddit()` uses this to find the right line. Don't remove the sentinel.

---

## File Reference

| File | Tracked in git | How to regenerate |
|------|---------------|---|
| `r_gradadmissions_posts.jsonl` (repo root) | ❌ | Obtain from data source |
| `r_gradadmissions_comments.jsonl` (repo root) | ❌ | Obtain from data source |
| `data/raw/r_gradadmissions_2022_*.jsonl` | ❌ | Obtain from data source |
| `data/raw/r_MSCS_*.jsonl` | ❌ | Obtain from data source |
| `data/mba/*.jsonl.gz` | ✅ | — |
| `data/processed/{sub}/posts_clean.jsonl` | ❌ | Run NB01 (or decompress MBA `.gz`) |
| `data/processed/{sub}/comments_clean.jsonl` | ❌ | Run NB01 (or decompress MBA `.gz`) |
| `data/processed/{sub}/*.parquet` | ✅ | Run the relevant notebook |
| `data/processed/{sub}/did_summary.csv` | ✅ | Run NB06 |
| `data/processed/{sub}/parallel_trends_test_v2.csv` | ✅ | Run NB06 §4a |
| `data/processed/combined_pooled_did.csv` | ✅ | Run NB07 (requires all 3 subs' NB06 outputs) |
| `data/processed/comparison_summary.parquet` | ✅ | Run NB07 |
| `models/clf_*.joblib` | ✅ | Run NB02 (don't, unless you must) |
| `figures/*.png` | ✅ | Run the notebook that produces them |
