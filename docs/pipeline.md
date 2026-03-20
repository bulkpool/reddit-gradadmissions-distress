# Pipeline Details

This document describes what each notebook does, the data it produces, and key design decisions.

For running instructions, see [Quickstart](quickstart.md).

---

## Data Overview

**Source**: r/GradAdmissions Reddit posts and comments, Aug 2023 – Jul 2025.

| File | Records | Description |
|------|---------|-------------|
| `r_gradadmissions_posts.cleaned.jsonl` | 88,441 | Posts with title, body, outcome label |
| `r_gradadmissions_comments.cleaned.jsonl` | 380,722 | Comments |

**Key fields**: `author`, `created_date`, `clean_text`, `outcome` (Accepted / Rejected / Waitlisted / Unknown), `degree_type`, `score`, `num_comments`

---

## Notebook 01 — VADER Scoring

**File**: `notebooks/01_score_corpus.ipynb`

Loads both JSONL files, removes bots and deleted accounts, and applies [VADER](https://github.com/cjhutto/vaderSentiment) sentiment scoring to every post and comment.

**What VADER adds**:

| Column | Description |
|--------|-------------|
| `vader_compound` | Overall sentiment ∈ [−1, 1]. Positive = good, negative = bad. |
| `vader_neg` | Fraction of text tokens that are negative-valence |
| `distress_score` | Same as `vader_neg` — used as a simple distress proxy |
| `is_negative` | True when `vader_compound < −0.05` |
| `week` | ISO 8601 week label (e.g. `2024-W03`) |

**Output**: `data/processed/scored_corpus.parquet` — 467,525 rows, 90,539 unique users

**Key stats**:
- Community overall skews positive (mean compound = 0.32)
- Rejected posts: mean compound −0.09 vs Accepted +0.47 — VADER captures the broad signal

---

## Notebook 02 — Anchor Event Identification

**File**: `notebooks/02_anchor_events.ipynb`

Identifies **anchor posts** — the "treatment events" in the causal design. An anchor post is a high-distress self-disclosure about a negative admissions experience.

**Anchor post criteria** (post must meet at least one):
1. Contains a negative keyword (reject, decline, waitlist, funding lost, anxiety, depressed, stress, falling apart, can't cope...) **AND** is above the distress threshold (`vader_compound < −0.05` or `vader_neg > 0.10`)
2. Self-labeled outcome is Rejected or Waitlisted **AND** above the distress threshold

**Result**: 7,075 anchor posts from 5,471 unique authors

**Exposure classification**:
- **Exposed**: user authored an anchor post that week (self-disclosure)
- **Unexposed**: user was active the same week but didn't author any anchor post

> **Why authorship?** The dataset doesn't include `link_id` on comments, so we can't directly identify who *read* a specific post. Authorship is a conservative but clean proxy — the author is unambiguously the most directly exposed person.

**Outputs**:
| File | Description |
|------|-------------|
| `data/processed/anchor_posts.parquet` | 7,075 anchor posts |
| `data/processed/user_weekly_scores.parquet` | Mean distress per (user, week) — 210,133 rows |
| `data/processed/exposure_labels.parquet` | Exposed/unexposed label per (user, event_week) |

---

## Notebook 03 — VADER DiD Baseline (optional)

**File**: `notebooks/03_did_analysis.ipynb`

Runs the full DiD pipeline using `mean_distress` (VADER neg) as the outcome. Produces baseline results for comparison with the SVM approach.

**Result**: DiD = +0.014, p < 0.0001 on VADER neg. The effect is detectable but VADER conflates topic language with genuine distress — words like "rejected" and "stressed" appear in advisory posts too. The SVM approach (notebook 06) is more sensitive.

---

## Notebook 04 — Community Breadth Collection

**File**: `notebooks/04_collect_community_breadth.ipynb`

Queries the [Arctic Shift API](https://arctic-shift.photon-reddit.com) for each panel user's activity across all of Reddit during the study window.

**Community breadth** = number of distinct subreddits a user posted or commented in (excluding r/GradAdmissions and their own profile sub).

**API endpoint**: `GET /api/users/interactions/subreddits?author={user}&after=2023-08-01&before=2025-07-31`

**Scale**: 25,316 users queried, ~3 hours at 2.5 req/sec with dynamic rate limiting.

**Fault tolerance**: Progress is saved to `data/processed/breadth_checkpoint.jsonl` every 500 users. Interrupting and re-running the notebook resumes from the last checkpoint.

**Stats** (25,305 users):
- Mean: 23.1 subreddits, Median: 10
- 7.5% of users post exclusively in r/GradAdmissions (breadth = 0)

**Output**: `data/processed/user_community_breadth.parquet`

> This step is already complete — the output is included in the repository.

---

## Notebook 05 — SVM Classifier Training

**File**: `notebooks/05_train_classifiers.ipynb`

Trains three binary SVM classifiers following the methodology of Low et al. (2020), then applies them to the full corpus to produce a more sensitive distress measure.

**Why SVM over VADER?** VADER fires on any negative word regardless of context. The SVMs are trained on *actual mental health posts* from r/anxiety, r/depression, and r/stress — they learn the style and framing of distressed writing, not just keyword presence.

**Training data** (Jan 2022 – Jul 2023, before the study window):

| Classifier | Positive class | Negative (control) |
|------------|---------------|-------------------|
| anxiety | r/anxiety — 2,000 posts | r/personalfinance, r/learnprogramming, r/todayilearned, r/careerguidance — 2,000 each |
| depression | r/depression — 2,000 posts | same |
| stress | r/stress — 2,000 posts | same |

**Features**: TF-IDF on unigrams + bigrams, max 50k features, validated via 5-fold cross-validation.

**Composite score**: `mh_score = mean(anx_score, dep_score, str_score)` — continuous ∈ (0, 1).

**Validation** — mh_score correctly orders outcomes by distress:
| Outcome | Mean mh_score |
|---------|--------------|
| Rejected | 0.451 |
| Waitlisted | 0.446 |
| Accepted | 0.401 |

**Outputs**:
| File | Description |
|------|-------------|
| `models/clf_anxiety.joblib` | Trained pipeline (TF-IDF + LinearSVC) |
| `models/clf_depression.joblib` | — |
| `models/clf_stress.joblib` | — |
| `data/processed/scored_corpus_v2.parquet` | Full corpus + mh_score columns |
| `data/processed/user_weekly_scores_v2.parquet` | Weekly `mean_mh_score` per user |

---

## Notebook 06 — Main DiD Analysis (SVM)

**File**: `notebooks/06_did_analysis_v2.ipynb`

The main results notebook. Runs the full causal pipeline using `mean_mh_score` as the outcome.

**Steps**:

1. **Build panel**: for each (user, event_week) in exposure_labels, collect `mean_mh_score` at offsets −2, −1, +1, +2 weeks → 200,646 observations
2. **Propensity score matching**: match exposed to unexposed users on pre-event mh_score + activity level (caliper = 0.05) → 2,843 matched pairs
3. **DiD regression**: `mean_mh_score ~ post + exposed + post×exposed + log(n_posts)` with HC3 robust standard errors
4. **RQ2**: add `post×exposed × community_breadth_log` interaction
5. **Cross-cycle replication**: run separately on 2023–24 and 2024–25
6. **Event study plot**: visualize parallel pre-trends and post-event divergence

See [Results](results.md) for the output.
