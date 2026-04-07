# Quickstart — Reproducing the Project

This guide walks you through setting up the environment and running the full pipeline from scratch.

---

## Prerequisites

### Python Environment

All notebooks use a Jupyter Python environment. The project uses a venv at `~/venvs/jupyter/`. Install required packages:

```bash
~/venvs/jupyter/bin/pip install \
    vaderSentiment scikit-learn statsmodels joblib \
    pandas numpy matplotlib requests causalimpact
```

> `causalimpact 0.2.6` has a pandas 2.x incompatibility; NB08 patches it at runtime automatically.

### Raw Data

Raw Reddit data lives at the **repository root** (not in `data/`). Place these files there:

```
r_gradadmissions_posts.jsonl          # Raw posts
r_gradadmissions_comments.jsonl       # Raw comments
r_gradadmissions_posts.cleaned.jsonl  # Pre-cleaned posts (used by old NB07 baseline)
r_gradadmissions_comments.cleaned.jsonl
```

---

## What's Already in the Repo

Several expensive steps are pre-computed and included:

| Already included | Skips |
|-----------------|-------|
| `models/clf_anxiety.joblib` etc. | SVM classifier training (~15 min API + training) |
| `data/processed_v2/panel_scores_v2.parquet` | Pre/post scoring pass |
| `data/processed_v2/panel_scores_alt.parquet` | NB08 Sep–Nov panel |
| `data/processed_v2/post_level_scores_v2.parquet` | Post-level DiD dataset |
| `data/processed_v2/dose_exposure_v2.parquet` | Dose-response dataset |
| `data/processed_v2/exposure_labels_v2.parquet` | Exposure classification |
| `data/processed_v2/anchor_posts_v2.parquet` | Anchor post list |
| `data/processed_v2/user_community_breadth_v2.parquet` | Breadth (~3 hr API run) |
| `data/processed_v2/breadth_checkpoint_v2.jsonl` | API checkpoint (resumable) |
| `figures/*.png` | All output figures |

---

## Execution Order

```
01 → 02 → 03 → 04 → 05 → 06    (main pipeline)
                          └→ 08  (alternative analysis)
00  (EDA, standalone)
07  (optional VADER baseline)
```

| Step | Notebook | Runtime | Notes |
|------|----------|---------|-------|
| 1 | `01_clean_corpus.ipynb` | ~5 min | Reads raw JSONL from repo root |
| 2 | `02_train_classifiers.ipynb` | ~15 min | Skip if `models/clf_*.joblib` exist |
| 3 | `03_exposure_labels_v2.ipynb` | ~3 min | — |
| 4 | `04_panel_scores.ipynb` | ~10 min | — |
| 5 | `05_collect_community_breadth.ipynb` | ~3 hr | Skip — already done, see above |
| 6 | `06_did_analysis_v2.ipynb` | ~2 min | Main results |
| 7 | `08_alt_analysis.ipynb` | ~15 min | Reads raw JSONL; needs causalimpact |

---

## Step-by-step

### Step 1 — Clean the Corpus (`01_clean_corpus.ipynb`)

Applies text normalization, deduplication, bot filtering, and adds `post_id` mapping to comments (derived from `link_id`). All downstream notebooks depend on this.

**Reads**: `r_gradadmissions_posts.jsonl`, `r_gradadmissions_comments.jsonl` (repo root)

**Writes**: `data/processed_v2/posts_clean.jsonl`, `data/processed_v2/comments_clean.jsonl`

---

### Step 2 — Train SVM Classifiers (`02_train_classifiers.ipynb`)

Trains three binary SVMs (anxiety, depression, stress) via Arctic Shift API pulls from mental health subreddits.

> **Shortcut**: Skip this step — `models/clf_*.joblib` are already in the repo.

**Writes**: `models/clf_anxiety.joblib`, `clf_depression.joblib`, `clf_stress.joblib`

---

### Step 3 — Exposure Labels (`03_exposure_labels_v2.ipynb`)

Identifies anchor posts and classifies all panel users as exposed (commented on anchor thread via `link_id`) or unexposed.

> **Shortcut**: Skip — `data/processed_v2/exposure_labels_v2.parquet` and `anchor_posts_v2.parquet` are included.

---

### Step 4 — Panel Scoring (`04_panel_scores.ipynb`)

Scores every panel user's August (pre) and Dec–May (post) text. Also produces post-level scores and dose-exposure data.

> **Shortcut**: Skip — all three output parquets are included.

---

### Step 5 — Community Breadth (`05_collect_community_breadth.ipynb`) — *skip*

> **Skip this step** — `data/processed_v2/user_community_breadth_v2.parquet` is included.

Makes ~21k Arctic Shift API requests. Checkpoint in `data/processed_v2/breadth_checkpoint_v2.jsonl` — safe to resume if re-running.

---

### Step 6 — Main DiD Analysis (`06_did_analysis_v2.ipynb`)

Runs PSM, user-level DiD (RQ1 + RQ2), post-level DiD, and dose-response analysis.

**Reads**: `data/processed_v2/panel_scores_v2.parquet`, `user_community_breadth_v2.parquet`

**Writes**: figures to `figures/`

**Runtime**: ~2 min

---

### Step 7 — Alternative Analysis (`08_alt_analysis.ipynb`)

Redefines the pre-period to Sep–Nov (4× more matched pairs) and runs Causal Impact.

**Reads**: raw JSONL from repo root, `exposure_labels_v2.parquet`, `anchor_posts_v2.parquet`, `user_community_breadth_v2.parquet`, `models/clf_*.joblib`

**Writes**: `data/processed_v2/panel_scores_alt.parquet`, `figures/fig_causal_impact_*.png`, `figures/fig_parallel_trends_alt.png`

**Runtime**: ~15 min (rescores all raw text)

---

## File Reference

| File | In Repo | How to Get It |
|------|---------|---------------|
| `r_gradadmissions_posts.jsonl` | No | Obtain from data source |
| `r_gradadmissions_comments.jsonl` | No | Obtain from data source |
| `data/processed_v2/posts_clean.jsonl` | No | Run NB01 |
| `data/processed_v2/comments_clean.jsonl` | No | Run NB01 |
| `data/processed_v2/panel_scores_v2.parquet` | Yes | — |
| `data/processed_v2/panel_scores_alt.parquet` | Yes | — |
| `data/processed_v2/post_level_scores_v2.parquet` | Yes | — |
| `data/processed_v2/dose_exposure_v2.parquet` | Yes | — |
| `data/processed_v2/exposure_labels_v2.parquet` | Yes | — |
| `data/processed_v2/anchor_posts_v2.parquet` | Yes | — |
| `data/processed_v2/user_community_breadth_v2.parquet` | Yes | — |
| `data/processed_v2/breadth_checkpoint_v2.jsonl` | Yes | — |
| `models/clf_anxiety.joblib` | Yes | — |
| `models/clf_depression.joblib` | Yes | — |
| `models/clf_stress.joblib` | Yes | — |
| `figures/fig_*.png` | Yes | — |
