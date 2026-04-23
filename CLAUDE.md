# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Always use the project venv — never system Python:
```bash
~/venvs/jupyter/bin/python3        # Python interpreter
~/venvs/jupyter/bin/jupyter        # Jupyter / nbconvert
~/venvs/jupyter/bin/pip            # package installs
```

Execute a single notebook manually:
```bash
~/venvs/jupyter/bin/jupyter nbconvert --to notebook --execute notebooks/<nb>.ipynb \
  --ExecutePreprocessor.timeout=7200 --output /tmp/out.ipynb
```

## Running the Pipeline

```bash
# Full pipeline (all three subreddits)
~/venvs/jupyter/bin/python3 run_pipeline.py

# One subreddit only
~/venvs/jupyter/bin/python3 run_pipeline.py --subreddits gradadmissions

# Resume from a step (skip notebooks with prefix < NN)
~/venvs/jupyter/bin/python3 run_pipeline.py --start-from 05

# Skip the comparison notebook (NB07)
~/venvs/jupyter/bin/python3 run_pipeline.py --skip-comparison
```

`run_pipeline.py` injects `SUBREDDIT` into each notebook's config cell and runs it via nbconvert. Source notebooks are never modified. Outputs land in `data/processed/{subreddit}/`. Logs go to `pipeline_logs/`.

**NB02 is a one-off classifier training step — do not re-run it** (it makes API calls and will time out).

## Architecture

This is a causal inference study on r/GradAdmissions, r/MSCS, and r/MBA covering three admission cycles (Aug 2022–Jul 2025). The research question: does exposure to high-distress "anchor posts" during Sep–Nov increase a user's mental health distress in the following Dec–May decision season?

### Pipeline Flow

```
NB01 (clean corpus)
  → posts_clean.jsonl, comments_clean.jsonl

NB03 (exposure labels)
  → anchor_posts.parquet        (high-distress posts: keyword + SVM threshold)
  → exposure_labels.parquet     (per-user per-cycle exposed/unexposed flag)

NB04 (panel scores)
  → panel_scores.parquet        (user-level pre/post MH scores, Aug pre-period)
  → post_level_scores.parquet   (every scored post for matched-panel users)
  → dose_exposure.parquet       (anchor comment counts per user)

NB05 (community breadth, hits Arctic Shift API)
  → user_community_breadth.parquet

NB06 (MAIN RESULTS: PSM + DiD)
  → fig_att_coef_{SUBREDDIT}.png
  → fig_parallel_trends_{SUBREDDIT}.png
  → did_summary.csv

NB07 (cross-subreddit comparison, runs after all three subreddits)
  → comparison_summary.parquet
  → fig_att_comparison.png, fig_mhscore_distributions.png, fig_anchor_comparison.png

NB08 (alt analysis: Sep–Nov pre-period + CausalImpact)
  → panel_scores_alt.parquet
  → fig_parallel_trends_alt_{SUBREDDIT}.png
  → fig_causal_impact_cycle{1,2,3}_{SUBREDDIT}.png
```

Diagnostic notebooks (not in the main pipeline run):
- **NB04a**: differential attrition, pre-period quality, anchor comment timing
- **NB05a**: pipeline funnel waterfall (labeled → panel)
- **NB06a**: PSM+DiD sensitivity to pre-period length restrictions

### SUBREDDIT Parameterization

Every pipeline notebook has a config cell with `SUBREDDIT = 'gradadmissions'  # change to`. `run_pipeline.py` injects the correct value at runtime. All output paths are derived as `data/processed/{SUBREDDIT}/`. Supported values: `'gradadmissions'`, `'mscs'`, `'mba'`.

MBA is special: `PRE_CLEANED = {'mba'}` in `run_pipeline.py` skips NB01 for it (its cleaned JSONLs are decompressed from `data/mba/*.jsonl.gz`).

### Scoring

Models (`models/clf_anxiety/depression/stress.joblib`) are LinearSVC. Scores are computed as `sigmoid(clf.decision_function(texts))` — **not** `predict_proba`. Mean of the three dimensions = `mh_score`.

### PSM + DiD (NB06)

1. Per cycle, merge panel with breadth; include `community_breadth_log` in PSM features if coverage ≥ 95%.
2. `psm_match()`: StandardScaler → LogisticRegression propensity score → 1-NN matching with caliper 0.05.
3. `run_did()`: OLS with HC3 errors: `mh_score ~ period + exposed + period×exposed + log1p_n_posts`.
4. ATT = coefficient on `period×exposed`.

### Column Name Gotchas

- `panel_scores.parquet`: columns are `pre_mh_score`, `post_mh_score`
- `post_level_scores.parquet`: column is `mean_mh_score` (not `mh_score`)
- NB06 reshapes to `mh_score` at runtime for the DiD regression
- `dose_exposure.parquet`: `cycle` is **float64**, not int — cast with `.astype(int)` if needed
- `panel_scores_alt.parquet` (NB08): already includes `community_breadth` and `community_breadth_log`

### CausalImpact Compatibility Patch (NB08)

`causalimpact 0.2.6` is incompatible with pandas 2.x without a runtime patch (applied in NB08):
```python
import pandas.core.dtypes.common as _pdc
if not hasattr(_pdc, 'is_datetime_or_timedelta_dtype'):
    _pdc.is_datetime_or_timedelta_dtype = (
        lambda x: pd.api.types.is_datetime64_any_dtype(x) or
                  pd.api.types.is_timedelta64_dtype(x)
    )
```
Pass integer index positions to `CausalImpact()`, not datetime objects.

## What Is and Isn't in Git

**In git**: all `*.parquet`, `models/clf_*.joblib`, `figures/*.png`, all notebooks, all docs, `data/mba/*.jsonl.gz`.

**Not in git** (regenerable or too large):
- `r_gradadmissions_posts/comments.jsonl` (repo root — raw GA 2023+2024)
- `data/raw/r_gradadmissions_2022_*.jsonl`, `data/raw/r_MSCS_*.jsonl` — raw cycle files
- `data/processed/*/posts_clean.jsonl` and `comments_clean.jsonl` — run NB01 to generate; for MBA decompress from `data/mba/*.jsonl.gz`
- `user_community_breadth.parquet` (deleted 2026-04-22 — re-run NB05 before NB06)

## Key Docs

- `docs/LLM_CONTEXT.md` — authoritative dense context including data schemas, cell IDs for NotebookEdit, known gotchas, and current status. **Update this after every session that changes code, schemas, or results.**
- `docs/flow.md` — pipeline flow diagram
- `docs/methodology.md` — study design details
