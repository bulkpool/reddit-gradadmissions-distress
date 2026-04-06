# Pipeline Details

This document describes what each notebook does, the data it produces, and key design decisions.

For running instructions, see [Quickstart](quickstart.md).

---

## Data Overview

**Source**: r/GradAdmissions Reddit posts and comments, Aug 2023 – Jul 2025.

| File | Size | Description |
|------|------|-------------|
| `data/raw/r_gradadmissions_posts.jsonl` | 286 MB | Raw post dumps from Pushshift/Arctic Shift |
| `data/raw/r_gradadmissions_comments.jsonl` | 847 MB | Raw comment dumps |

---

## Execution Order

```
01 → 02 → 03 → 04 → 05 → 06
```

| Notebook | File | Depends on |
|----------|------|------------|
| 01 | `01_clean_corpus.ipynb` | Raw data only |
| 02 | `02_train_classifiers.ipynb` | Arctic Shift API (training data) |
| 03 | `03_exposure_labels_v2.ipynb` | 01, 02 |
| 04 | `04_panel_scores.ipynb` | 01, 02, 03 |
| 05 | `05_collect_community_breadth.ipynb` | 04 |
| 06 | `06_did_analysis_v2.ipynb` | 04, 05 |
| 07 | `07_did_analysis_vader_baseline.ipynb` | Optional — uses old pipeline outputs |

Notebooks 01 and 02 have no dependency on each other and can run in parallel.

---

## Notebook 01 — Corpus Cleaning

**File**: `notebooks/01_clean_corpus.ipynb`

Reads the raw JSONL dumps and produces clean, canonical JSONL files used by all downstream notebooks. No scoring happens here.

**Cleaning steps:**

| Step | What it does |
|------|-------------|
| Date/author validation | Parses `created_utc` (Unix timestamp → ISO datetime), drops null/deleted/removed authors and bodies |
| Dedup | Deduplicates on `id`, keeps first occurrence |
| Bot filtering | Drops `AutoModerator` and any author matching `*bot` pattern |
| Text normalization | Lowercase → strip URLs → strip non-alpha → collapse whitespace → `clean_text` field |
| Comment→post mapping | Derives `post_id = link_id.removeprefix("t3_")` on comments — links each comment to its parent post thread |

The comment→post mapping was missing from the old pipeline. Without it, exposure identification had to rely on same-week activity as a proxy; now comments are directly linked to the anchor threads they came from.

**Outputs**:

| File | Description |
|------|-------------|
| `data/processed_v2/posts_clean.jsonl` | `id, author, created_dt, clean_text, score, num_comments` |
| `data/processed_v2/comments_clean.jsonl` | `id, author, created_dt, post_id, clean_text, score` |

---

## Notebook 02 — SVM Classifier Training

**File**: `notebooks/02_train_classifiers.ipynb`

Trains three binary SVM classifiers following Low et al. (2020), using posts from mental health subreddits as the positive class and general-topic subreddits as the negative class.

**Why SVM over VADER?** VADER fires on any negative word regardless of context — "rejected" scores negative whether someone is venting distress or asking about acceptance rates. The SVMs learn the *style* of distressed writing from actual mental health communities.

**Training data** (Jan 2022 – Jul 2023, before the study window to prevent leakage):

| Classifier | Positive class | Negative (control) |
|------------|---------------|-------------------|
| anxiety | r/anxiety — 2,000 posts | r/personalfinance, r/learnprogramming, r/todayilearned, r/careerguidance |
| depression | r/depression — 2,000 posts | same |
| stress | r/stress — 2,000 posts | same |

**Features**: TF-IDF on unigrams + bigrams, max 50k features. Validated via 5-fold CV (F1: 0.88–0.94).

**Composite score**: `mean_mh_score = mean(anx_score, dep_score, str_score)` — continuous ∈ (0, 1), higher = more distress language.

**Outputs**:

| File | Description |
|------|-------------|
| `models/clf_anxiety.joblib` | Fitted Pipeline (TF-IDF + LinearSVC) |
| `models/clf_depression.joblib` | — |
| `models/clf_stress.joblib` | — |

---

## Notebook 03 — Exposure Labels (v2)

**File**: `notebooks/03_exposure_labels_v2.ipynb`

Identifies anchor posts and classifies panel users as exposed or unexposed using correct thread-level linking via `post_id` / `link_id`.

**Anchor post definition** (post must meet all three):
1. Falls within the anchor period: Sep 1–Nov 30 of a cycle year
2. Matches negative keyword list (rejection, re-applicant, anxiety, stress, depression...)
3. `mean_mh_score > 0.45` from the three SVM classifiers (notebook 02)

**Exposure classification**:
- **Exposed**: user commented on an anchor post thread (identified via `link_id`); anchor post authors are excluded
- **Unexposed**: active in r/GradAdmissions during Aug 1–May 31 of that cycle but never commented on an anchor thread

**Cycle windows**:

| | Cycle 1 | Cycle 2 |
|-|---------|---------|
| Anchor period | Sep 1–Nov 30, 2023 | Sep 1–Nov 30, 2024 |
| Active window | Aug 1, 2023–May 31, 2024 | Aug 1, 2024–May 31, 2025 |

**Outputs**:

| File | Description |
|------|-------------|
| `data/processed_v2/anchor_posts_v2.parquet` | Anchor posts with SVM scores |
| `data/processed_v2/exposure_labels_v2.parquet` | `author, exposed (bool), cycle` |

---

## Notebook 04 — Panel Scoring (Pre / Post Windows)

**File**: `notebooks/04_panel_scores.ipynb`

Scores each panel user's text in the **pre-baseline** (August) and **post-outcome** (December–May) windows using the SVM classifiers, then merges with exposure labels to build the analysis panel.

**Scoring windows**:

| | Cycle 1 | Cycle 2 |
|-|---------|---------|
| Pre baseline | Aug 1–31, 2023 | Aug 1–31, 2024 |
| Post outcome | Dec 1, 2023–May 31, 2024 | Dec 1, 2024–May 31, 2025 |

**Steps**:
1. Load panel users from `exposure_labels_v2.parquet`
2. Filter clean JSONL to Aug windows → score with 3 SVMs → aggregate `pre_mh_score`, `pre_n_posts`
3. Filter clean JSONL to Dec–May windows → score → aggregate `post_mh_score`, `post_n_posts`
4. Inner join pre + post + exposure labels; drop users missing either window

**Output**: `data/processed_v2/panel_scores_v2.parquet`

| Column | Description |
|--------|-------------|
| `author` | Reddit username |
| `cycle` | 1 or 2 |
| `exposed` | bool |
| `pre_mh_score` | Mean SVM score across Aug posts/comments |
| `pre_n_posts` | Count of Aug posts/comments |
| `post_mh_score` | Mean SVM score across Dec–May posts/comments |
| `post_n_posts` | Count of Dec–May posts/comments |

---

## Notebook 05 — Community Breadth Collection

**File**: `notebooks/05_collect_community_breadth.ipynb`

Queries the [Arctic Shift API](https://arctic-shift.photon-reddit.com) for each v2 panel user's activity across all of Reddit during the study window.

**Community breadth** = number of distinct subreddits a user posted or commented in (excluding r/GradAdmissions and their own profile sub).

**Strategy**: Reuses the existing `data/processed/user_community_breadth.parquet` (25,305 users from the old pipeline) as a cache. Only fetches Arctic Shift for users in the new v2 panel not already covered.

**API endpoint**: `GET /api/users/interactions/subreddits?author={user}&after=2023-08-01&before=2025-07-31`

**Fault tolerance**: Progress is saved to `data/processed_v2/breadth_checkpoint_v2.jsonl` every 500 users. Safe to interrupt and resume.

**Output**: `data/processed_v2/user_community_breadth_v2.parquet`

| Column | Description |
|--------|-------------|
| `author` | Reddit username |
| `community_breadth` | # distinct subreddits (int) |
| `community_breadth_log` | `log1p(breadth)` — used in regression |
| `subreddits_json` | JSON list of subreddit names |
| `status` | `ok` / `timeout` / `http_4xx` |

---

## Notebook 06 — PSM + DiD Analysis

**File**: `notebooks/06_did_analysis_v2.ipynb`

The main results notebook. Runs propensity-score matching and difference-in-differences regression using `mean_mh_score` as the outcome.

**Steps**:

1. **PSM per cycle**: 1:1 nearest-neighbor matching on `pre_mh_score` + `pre_n_posts` (+ `community_breadth_log` if coverage ≥ 95%), caliper = 0.05. Balance assessed via standardised mean differences (SMD < 0.1).

2. **Reshape to long**: one row per user × period (0 = pre, 1 = post).

3. **RQ1 — Main DiD**:
   ```
   mh_score ~ period + exposed + period×exposed + log1p(n_posts)
   ```
   OLS with HC3 robust standard errors. Run for Cycle 1, Cycle 2, and pooled (+ cycle FE).

4. **RQ2 — Community breadth moderation**:
   ```
   mh_score ~ period + exposed + community_breadth_log
            + period×exposed + period×breadth + exposed×breadth
            + period×exposed×breadth + log1p(n_posts)
   ```
   Three-way interaction coefficient = moderating effect of breadth.

5. **Parallel trends check**: plot pre/post mean mh_score by exposure for the matched sample.

6. **Figures**: ATT coefficient plot, parallel trends plot, regression table → `figures/`.

See [Results](results.md) for the output.

---

## Notebook 07 — VADER DiD Baseline (optional)

**File**: `notebooks/07_did_analysis_vader_baseline.ipynb`

Runs the DiD pipeline using VADER `distress_score` as the outcome instead of SVM `mh_score`. Useful as a robustness check — if the SVM and VADER results point in the same direction, the finding is not an artefact of the measurement choice.

> Note: This notebook uses old pipeline outputs from `data/processed/` and the ±2-week event-study design. It is kept for comparison purposes only. The main results are in notebook 06.
