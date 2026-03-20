# Quickstart — Reproducing the Project

This guide walks you through setting up the environment and running the full pipeline from scratch.

---

## Prerequisites

### Python Environment

All notebooks use a Jupyter Python environment. Install the required packages once:

```bash
pip install vaderSentiment scikit-learn statsmodels joblib pandas numpy matplotlib requests
```

### Raw Data

The raw Reddit data is **not included** in this repository (files are 125 MB and 275 MB). Place the following files in `data/raw/` before running:

```
data/raw/
├── r_gradadmissions_posts.cleaned.jsonl      # 88,441 posts
└── r_gradadmissions_comments.cleaned.jsonl   # 380,722 comments
```

---

## What's Already in the Repo

You don't need to re-run everything from scratch. Several expensive steps are pre-computed and included:

| Already included | Skips |
|-----------------|-------|
| `data/processed/user_community_breadth.parquet` | ~3 hours of Arctic Shift API calls (notebook 04) |
| `data/processed/training_data_raw.parquet` | ~15 min of API calls for training data (notebook 05) |
| `models/clf_*.joblib` | SVM classifier training (notebook 05) |
| `data/processed/user_weekly_scores_v2.parquet` | Weekly score aggregation (notebook 05) |

The two files you **must** generate are:
- `data/processed/scored_corpus.parquet` — run notebook 01 (~2 min)
- `data/processed/scored_corpus_v2.parquet` — run notebook 05 (scoring only, ~5 min)

---

## Execution Order

Run notebooks in this order. Each one reads outputs from the previous.

```
01_score_corpus.ipynb              (~2 min)
02_anchor_events.ipynb             (~1 min)
03_did_analysis.ipynb              (~2 min, optional — VADER baseline only)
04_collect_community_breadth.ipynb (SKIP — already done, see above)
05_train_classifiers.ipynb         (~5 min if skipping API pull)
06_did_analysis_v2.ipynb           (~5 min) ← main results
```

See [Pipeline](pipeline.md) for what each notebook does in detail.

---

## Step-by-step

### Step 1 — Score the Corpus (`01_score_corpus.ipynb`)

Applies VADER sentiment analysis to all 467,525 posts and comments.

**Reads**: `data/raw/r_gradadmissions_posts.cleaned.jsonl`, `data/raw/r_gradadmissions_comments.cleaned.jsonl`

**Writes**: `data/processed/scored_corpus.parquet` *(required — not in repo)*

**Runtime**: ~2 minutes

---

### Step 2 — Find Anchor Events (`02_anchor_events.ipynb`)

Identifies the 7,075 high-distress "anchor posts" and classifies all users as exposed or unexposed for each event week.

**Reads**: `data/processed/scored_corpus.parquet`

**Writes**: `data/processed/anchor_posts.parquet`, `data/processed/user_weekly_scores.parquet`, `data/processed/exposure_labels.parquet`

**Runtime**: ~1 minute

---

### Step 3 — VADER DiD Baseline (`03_did_analysis.ipynb`) — *optional*

First-pass DiD using VADER scores. Useful as a comparison to the SVM results but not required.

**Reads**: `data/processed/scored_corpus.parquet`, `data/processed/user_weekly_scores.parquet`, `data/processed/exposure_labels.parquet`, `data/processed/anchor_posts.parquet`, `data/processed/user_community_breadth.parquet`

**Writes**: figures only

**Runtime**: ~2 minutes

---

### Step 4 — Community Breadth (`04_collect_community_breadth.ipynb`) — *already done*

> **Skip this step** — `data/processed/user_community_breadth.parquet` is already in the repo.

This notebook queries the [Arctic Shift API](https://arctic-shift.photon-reddit.com) for each user's cross-subreddit activity. It took ~3 hours to run (25,316 API requests). The checkpoint file (`data/processed/breadth_checkpoint.jsonl`) is also included — if you do want to re-run, it will resume from where it left off.

---

### Step 5 — Train Classifiers & Score Corpus (`05_train_classifiers.ipynb`)

Trains three SVM mental-health classifiers (anxiety, depression, stress) and scores the full corpus.

> **Shortcut**: The trained models (`models/clf_*.joblib`) and training data (`data/processed/training_data_raw.parquet`) are already in the repo. The only thing you need to run is the **scoring section** (cells 4–6) to produce `scored_corpus_v2.parquet`.

**Reads**: `data/processed/scored_corpus.parquet`, `data/processed/training_data_raw.parquet`

**Writes**: `data/processed/scored_corpus_v2.parquet` *(required — not in repo)*, `data/processed/user_weekly_scores_v2.parquet`

**Runtime**: ~5 minutes (scoring only)

---

### Step 6 — Main DiD Analysis (`06_did_analysis_v2.ipynb`)

Runs the full causal analysis: propensity score matching, DiD regression, RQ1, RQ2, cross-cycle replication, and event study plot.

**Reads**: `data/processed/user_weekly_scores_v2.parquet`, `data/processed/exposure_labels.parquet`, `data/processed/user_community_breadth.parquet`

**Writes**: `figures/fig_event_study_v2.png`

**Runtime**: ~5 minutes

---

## File Reference

| File | In Repo | How to Get It |
|------|---------|---------------|
| `data/raw/*.cleaned.jsonl` | No | Obtain from data source |
| `data/processed/scored_corpus.parquet` (92 MB) | No | Run notebook 01 |
| `data/processed/scored_corpus_v2.parquet` (108 MB) | No | Run notebook 05 |
| `data/processed/anchor_posts.parquet` | Yes | — |
| `data/processed/user_weekly_scores.parquet` | Yes | — |
| `data/processed/user_weekly_scores_v2.parquet` | Yes | — |
| `data/processed/exposure_labels.parquet` | Yes | — |
| `data/processed/user_community_breadth.parquet` | Yes | — |
| `data/processed/training_data_raw.parquet` | Yes | — |
| `data/processed/breadth_checkpoint.jsonl` | Yes | — |
| `models/clf_anxiety.joblib` | Yes | — |
| `models/clf_depression.joblib` | Yes | — |
| `models/clf_stress.joblib` | Yes | — |
| `figures/fig_*.png` | Yes | — |
