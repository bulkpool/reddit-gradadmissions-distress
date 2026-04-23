# Pipeline Details

This document describes what each notebook does, the data it produces, and key design decisions.

For running instructions, see [Quickstart](quickstart.md).

---

## Data Overview

**Source**: r/GradAdmissions Reddit posts and comments, Aug 2023 – Jul 2025.

Raw files live at the **repository root** (not in `data/raw/`):

| File | Description |
|------|-------------|
| `r_gradadmissions_posts.jsonl` | Raw post dumps |
| `r_gradadmissions_comments.jsonl` | Raw comment dumps |
| `r_gradadmissions_posts.cleaned.jsonl` | Pre-cleaned posts (original pipeline) |
| `r_gradadmissions_comments.cleaned.jsonl` | Pre-cleaned comments (original pipeline) |

All intermediate outputs go to `data/processed/`.

---

## Execution Order

```
01 → 02 → 03 → 04 → 05 → 06
                          └→ 08 (alternative analysis, independent of 06)
00 (EDA only, no downstream deps)
07 (optional VADER baseline, uses old processed/ outputs)
```

| Notebook | File | Depends on |
|----------|------|------------|
| 00 | `00_exploratory_topic_sentiment.ipynb` | Raw data (EDA only) |
| 01 | `01_clean_corpus.ipynb` | Raw JSONL files |
| 02 | `02_train_classifiers.ipynb` | Arctic Shift API (training data) |
| 03 | `03_exposure_labels.ipynb` | 01, 02 |
| 04 | `04_panel_scores.ipynb` | 01, 02, 03 |
| 05 | `05_collect_community_breadth.ipynb` | 04 |
| 06 | `06_did_analysis.ipynb` | 04, 05 |
| 07 | `07_did_analysis_vader_baseline.ipynb` | Old `data/processed/` outputs |
| 08 | `08_alt_analysis.ipynb` | Raw JSONL, 03, 02 models |

Notebooks 01 and 02 have no dependency on each other and can run in parallel.

---

## Notebook 00 — Exploratory EDA

**File**: `notebooks/00_exploratory_topic_sentiment.ipynb`

Exploratory analysis only. Produces visualizations of post volume, VADER sentiment distribution, and topic patterns. No outputs consumed by downstream notebooks.

---

## Notebook 01 — Corpus Cleaning

**File**: `notebooks/01_clean_corpus.ipynb`

Reads the raw JSONL dumps and produces clean, canonical JSONL files used by all downstream notebooks (03, 04).

**Cleaning steps**:

| Step | What it does |
|------|-------------|
| Date/author validation | Parses `created_utc` → ISO datetime, drops null/deleted/removed/bot authors |
| Dedup | Deduplicates on `id`, keeps first occurrence |
| Text normalization | Lowercase → strip URLs → strip non-alpha → collapse whitespace → `clean_text` field |
| Comment→post mapping | Derives `post_id = link_id.removeprefix("t3_")` on comments — links each comment to its parent thread |

The `post_id` field on comments is critical: it enables thread-level exposure identification in NB03.

**Outputs** → `data/processed/`:

| File | Schema |
|------|--------|
| `posts_clean.jsonl` | `id, author, created_dt, clean_text, score, num_comments` |
| `comments_clean.jsonl` | `id, author, created_dt, post_id, clean_text, score` |

---

## Notebook 02 — SVM Classifier Training

**File**: `notebooks/02_train_classifiers.ipynb`

Trains three binary SVM classifiers following Low et al. (2020), using posts from mental health subreddits as the positive class.

**Why SVM over VADER?** VADER fires on any negative word regardless of context — "rejected, what are my chances?" scores as distressed. The SVMs learn the *style* of distressed writing (self-referential framing, help-seeking language) from actual mental health communities.

**Training data** (Jan 2022 – Jul 2023, before study window to prevent leakage):

| Classifier | Positive class | Negative (control) |
|------------|---------------|-------------------|
| anxiety | r/anxiety — 2,000 posts | r/personalfinance, r/learnprogramming, r/todayilearned, r/careerguidance |
| depression | r/depression — 2,000 posts | same |
| stress | r/stress — 2,000 posts | same |

**Features**: TF-IDF unigrams + bigrams, max 50k features. 5-fold CV F1: 0.88–0.94.

**Composite score**: `mh_score = mean(sigmoid(anx_df), sigmoid(dep_df), sigmoid(str_df))` ∈ (0, 1).

**Outputs** → `models/`:

| File | Description |
|------|-------------|
| `clf_anxiety.joblib` | Fitted Pipeline (TF-IDF + LinearSVC) |
| `clf_depression.joblib` | — |
| `clf_stress.joblib` | — |

---

## Notebook 03 — Exposure Labels (v2)

**File**: `notebooks/03_exposure_labels.ipynb`

Identifies anchor posts and classifies panel users as exposed or unexposed using thread-level linking via `post_id` / `link_id`.

**Anchor post definition** (post must meet all three):
1. Falls within the anchor period: Sep 1–Nov 30 of the cycle year
2. Matches negative keyword list (rejection, re-applicant, anxiety, stress, depression…)
3. `mh_score > 0.45` from the three SVM classifiers

**Exposure classification**:
- **Exposed**: user commented on an anchor post thread (matched via `link_id`); anchor post authors are excluded
- **Unexposed**: active in r/GradAdmissions during Aug 1–May 31 of that cycle but never commented on an anchor thread

**Cycle windows**:

| | Cycle 1 | Cycle 2 |
|-|---------|---------|
| Anchor period | Sep 1–Nov 30, 2023 | Sep 1–Nov 30, 2024 |
| Active window | Aug 1, 2023–May 31, 2024 | Aug 1, 2024–May 31, 2025 |

**Outputs** → `data/processed/`:

| File | Description |
|------|-------------|
| `anchor_posts.parquet` | Anchor posts with SVM scores; columns include `id, cycle` |
| `exposure_labels.parquet` | `author, exposed (bool), cycle` |

---

## Notebook 04 — Panel Scoring

**File**: `notebooks/04_panel_scores.ipynb`

Scores each panel user's text in the **pre-baseline** (August) and **post-outcome** (December–May) windows, then builds the analysis panel. Also produces post-level scores for the post-level DiD in NB06.

**Scoring windows**:

| | Cycle 1 | Cycle 2 |
|-|---------|---------|
| Pre baseline | Aug 1–31, 2023 | Aug 1–31, 2024 |
| Post outcome | Dec 1, 2023–May 31, 2024 | Dec 1, 2024–May 31, 2025 |

**Outputs** → `data/processed/`:

| File | Schema | Description |
|------|--------|-------------|
| `panel_scores.parquet` | `author, cycle, exposed, pre_mh_score, pre_n_posts, post_mh_score, post_n_posts` | User-level pre/post scores |
| `post_level_scores.parquet` | `author, cycle, exposed, period, mh_score, anx, dep, str_, dt` | One row per post/comment for post-level DiD |
| `dose_exposure.parquet` | `author, cycle, n_anchor_comments` | Anchor comment count per user (for dose-response) |

---

## Notebook 05 — Community Breadth Collection

**File**: `notebooks/05_collect_community_breadth.ipynb`

Queries the Arctic Shift API for each v2 panel user's cross-subreddit activity.

**Community breadth** = number of distinct subreddits a user posted/commented in (excluding r/GradAdmissions and their own profile sub).

**Strategy**: Reuses any existing cache from the old pipeline. Only fetches Arctic Shift for users in the v2 panel not already covered.

**Fault tolerance**: Progress saved to `data/processed/breadth_checkpoint.jsonl` every 500 users.

**Output** → `data/processed/user_community_breadth.parquet`:

| Column | Description |
|--------|-------------|
| `author` | Reddit username |
| `community_breadth` | # distinct subreddits |
| `community_breadth_log` | `log1p(breadth)` — used in regression |

---

## Notebook 06 — Main Analysis (PSM + DiD)

**File**: `notebooks/06_did_analysis.ipynb`

The primary results notebook. Runs PSM, user-level DiD, post-level DiD, and dose-response analysis.

**Steps**:

1. **PSM per cycle**: 1:1 nearest-neighbor matching on propensity score from logistic regression on `pre_mh_score + log1p_n_posts_pre` (+ `community_breadth_log` if ≥ 95% coverage). Caliper = 0.05. SMD balance checks printed per feature.

2. **User-level DiD** (long format, one row per user × period):
   ```
   mh_score ~ period + exposed + period×exposed + log1p_posts
   ```
   OLS with HC3 robust SEs. Run per cycle and pooled (+ cycle FE).

3. **Post-level DiD** (22,355 observations): same specification but one row per individual post, with and without user fixed effects. Cluster SEs on author. Also decomposed per dimension (anxiety, depression, stress).

4. **Dose-response analysis**: uses `dose_exposure.parquet` to test whether effect scales with number of anchor comments seen.

5. **RQ2 — Community breadth moderation**: three-way interaction `period × exposed × breadth_log`.

**Key results** (current pipeline):

| Specification | DiD estimate | p-value |
|--------------|-------------|---------|
| Cycle 1 user-level | +0.0069 | 0.476 n.s. |
| Cycle 2 user-level | +0.0102 | 0.226 n.s. |
| Pooled + cycle FE | +0.0080 | 0.207 n.s. |
| Post-level pooled (no FE) | −0.0008 | 0.858 n.s. |
| Post-level pooled (+user FE) | +0.0038 | 0.343 n.s. |

Effects are directionally positive but not statistically significant. PSM yields ~155 matched pairs per cycle (low power due to August pre-period coverage of ~6.5%).

---

## Notebook 07 — VADER DiD Baseline (optional)

**File**: `notebooks/07_did_analysis_vader_baseline.ipynb`

Runs the DiD pipeline using VADER `distress_score` as the outcome. Uses older `data/processed/` outputs. Kept for comparison purposes; not part of the main pipeline.

---

## Notebook 08 — Alternative Analysis: Sep–Nov Pre-Period + Causal Impact

**File**: `notebooks/08_alt_analysis.ipynb`

Addresses the power problem in NB06 (only 6.5% user coverage with August pre-period) using two complementary approaches.

**Inputs**: `r_gradadmissions_posts.jsonl`, `r_gradadmissions_comments.jsonl` (raw, at repo root), plus `exposure_labels.parquet`, `anchor_posts.parquet`, `user_community_breadth.parquet`, and the three SVM models.

**Approach 1 — Redefined pre-period**:
- Pre = each user's Sep–Nov activity *before* their first anchor comment (exposed users) or full Sep–Nov (unexposed)
- Post = Dec 1–May 31 (same as NB06)
- Coverage: 36.5% of panel users (vs. 6.5%), ~668 matched pairs/cycle

**Approach 2 — Causal Impact** (Bayesian structural time series):
- Aggregates exposed vs. unexposed weekly mean mh_score into a wide time series
- Uses unexposed group as synthetic control via Google's `causalimpact` library
- Pre-period: Sep 1–Nov 30; post-period: Dec 1–May 31
- Does not require individual pre+post coverage

**Key results**:

| Method | Cycle | Estimate | p-value |
|--------|-------|---------|---------|
| DiD | 1 | +0.0058 | 0.365 n.s. |
| DiD | 2 | +0.0090 | 0.104 n.s. |
| DiD | Pooled | +0.0076 | 0.067 (borderline) |
| CausalImpact | 1 | +2.1% relative | — |
| CausalImpact | 2 | +2.7% relative | — |

Both methods show directionally consistent positive effects.

**Note**: `causalimpact 0.2.6` is incompatible with pandas 2.x; the notebook patches `pandas.core.dtypes.common.is_datetime_or_timedelta_dtype` at runtime and uses integer index positions for pre/post period arguments.

**Output** → `data/processed/panel_scores_alt.parquet`
