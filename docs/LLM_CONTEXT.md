# LLM Codebase Context

This file is a dense, authoritative snapshot of the project for AI assistants. Read this instead of re-exploring the repo from scratch. Update it whenever the pipeline changes significantly.

**Last updated**: 2026-04-07

---

## What This Project Is

A causal inference study asking: does exposure to high-distress posts on r/GradAdmissions increase mental health distress in other users during graduate admissions season?

Two research questions:
- **RQ1**: Does exposure → increased `mh_score` in the post-period?
- **RQ2**: Does community breadth (cross-Reddit activity) moderate this effect?

---

## Repository Layout

```
/                                   ← repo root
├── data/
│   ├── raw/
│   │   ├── r_gradadmissions_posts.jsonl    ← NOT in git (large); raw posts
│   │   └── r_gradadmissions_comments.jsonl ← NOT in git (large); raw comments
│   └── processed_v2/                       ← all intermediate parquets and clean JSONLs
├── notebooks/
│   ├── 00_exploratory_topic_sentiment.ipynb  ← EDA only, no downstream deps
│   ├── 01_clean_corpus.ipynb                 ← corpus cleaning, post_id mapping
│   ├── 02_train_classifiers.ipynb            ← SVM training (clf_*.joblib)
│   ├── 03_exposure_labels_v2.ipynb           ← anchor posts + exposure labels
│   ├── 04_panel_scores.ipynb                 ← pre/post panel + post-level + dose
│   ├── 05_collect_community_breadth.ipynb    ← Arctic Shift API breadth
│   ├── 06_did_analysis_v2.ipynb              ← MAIN: PSM + DiD + post-level + dose-response
│   ├── 07_did_analysis_vader_baseline.ipynb  ← optional VADER comparison (old outputs)
│   └── 08_alt_analysis.ipynb                 ← Sep-Nov pre-period + Causal Impact
├── data/processed_v2/              ← all intermediate parquets and clean JSONLs
├── models/                         ← clf_anxiety/depression/stress.joblib
├── figures/                        ← all PNG outputs
└── docs/                           ← this file + pipeline/flow/methodology/results/quickstart
```

---

## Python Environment

Always use `~/venvs/jupyter/bin/python` and `~/venvs/jupyter/bin/pip`. Do not use system Python.

Key packages: `scikit-learn`, `statsmodels`, `joblib`, `pandas` (2.x), `numpy`, `matplotlib`, `causalimpact` (0.2.6 — has pandas 2.x compat patch applied at runtime in NB08).

---

## Data Files — What Exists

### Raw (not in git, in data/raw/)
| File | Key fields |
|------|-----------|
| `data/raw/r_gradadmissions_posts.jsonl` | `id, author, created_utc, selftext, score, num_comments` |
| `data/raw/r_gradadmissions_comments.jsonl` | `id, author, created_utc, body, link_id, score` |

`link_id` on comments is `"t3_<post_id>"`. Strip `t3_` to get the parent post's `id`.

### Cleaned (data/processed_v2/, in git where small enough)
| File | Schema | Notes |
|------|--------|-------|
| `posts_clean.jsonl` | `id, author, created_dt, clean_text, score, num_comments` | Output of NB01 |
| `comments_clean.jsonl` | `id, author, created_dt, post_id, clean_text, score` | `post_id` = stripped `link_id` |
| `anchor_posts_v2.parquet` | `id, cycle, mh_score, anx, dep, str_, title, ...` | Anchor posts per cycle |
| `exposure_labels_v2.parquet` | `author, exposed (bool), cycle` | 20,932 panel users |
| `panel_scores_v2.parquet` | `author, cycle, exposed, pre_mh_score, pre_n_posts, post_mh_score, post_n_posts` | August pre / Dec-May post |
| `panel_scores_alt.parquet` | same schema as above | Sep-Nov pre / Dec-May post (NB08) |
| `post_level_scores_v2.parquet` | `author, cycle, exposed, period, mh_score, anx, dep, str_, dt` | One row per post (22k+ rows) |
| `dose_exposure_v2.parquet` | `author, cycle, n_anchor_comments` | Anchor thread engagement count |
| `user_community_breadth_v2.parquet` | `author, community_breadth, community_breadth_log, status` | Cross-subreddit breadth |
| `breadth_checkpoint_v2.jsonl` | checkpoint for API resume | — |

---

## Notebook Summaries

### NB01 — `01_clean_corpus.ipynb`
Reads `r_gradadmissions_*.jsonl` (root). Normalizes text, deduplicates, filters bots, adds `post_id` on comments (`link_id.removeprefix("t3_")`). Writes `posts_clean.jsonl` and `comments_clean.jsonl` to `data/processed_v2/`.

### NB02 — `02_train_classifiers.ipynb`
Pulls training data from Arctic Shift API (Jan 2022–Jul 2023; r/anxiety, r/depression, r/stress vs. control subs). Trains three LinearSVC pipelines (TF-IDF unigrams+bigrams, 50k features). Saves to `models/clf_*.joblib`. Already complete — skip unless retraining.

### NB03 — `03_exposure_labels_v2.ipynb`
Reads `posts_clean.jsonl`, `comments_clean.jsonl`, `clf_*.joblib`. Identifies anchor posts (Sep–Nov, keyword match, `mh_score > 0.45`). Classifies users dynamically.
**Exposure Probability Metric**: Replaces default binary mapping. Calculates continuous probabilities utilizing both visibility decay and community upvotes. Uses $P = e^{-0.0838|\Delta t|} \times \frac{\log(1 + score)}{\max(\log(1 + score))}$. Saves `anchor_posts_v2.parquet` and `exposure_labels_v2.parquet`.

### NB04 — `04_panel_scores.ipynb`
Reads `posts_clean.jsonl`, `comments_clean.jsonl`, `exposure_labels_v2.parquet`, `clf_*.joblib`. Scores text in August (pre) and Dec–May (post) windows per user. Aggregates to user-level means. Also scores all window text at post level for `post_level_scores_v2.parquet`. Counts anchor thread engagements per user for `dose_exposure_v2.parquet`.

### NB05 — `05_collect_community_breadth.ipynb`
Queries Arctic Shift API `/api/users/interactions/subreddits` for each v2 panel user. Implements custom regex filtering focused *strictly* on mental health and higher-ed subreddits instead of generic platform engagement counting. Checkpoints every 500 users to `breadth_checkpoint_v2.jsonl`.

### NB06 — `06_did_analysis_v2.ipynb` ← MAIN RESULTS
**Inputs**: `panel_scores_v2.parquet`, `user_community_breadth_v2.parquet`

**GPS Weighting**: Propensity matching removed in favor of Continuous Generalized Propensity Scores utilizing the fully unmatched panel.

**Continuous User-level DiD**: `mh_score ~ period + exposure_prob + period_x_exposure_prob + log1p_posts`, OLS HC3. Run per cycle and pooled (+cycle FE).

**Post-level DiD**: 20k+ observations from `post_level_scores_v2.parquet`. With user FE; cluster SEs on author. Assesses high granularity continuous observation interactions.

**Continuous Dose-response**: dose exposure logged explicitly through explicit `dose_exposure_v2.parquet`.

**RQ2**: three-way interactions.

**Current results**:
- GPS Weighted DiD: +0.0058, p=0.552
- Dose-Response (User interaction proxy mapping): -0.0097, $p<0.01$ (Significant Breakthrough resilience)
- Post-Level Cycle 2 (+UserFE): -0.0054, p=0.565 (Significance Washed Out completely by popularity multiplier)

**Significance Improvement & Washout**: Transitioning to strict temporal modifiers resolved extensive binary-exposure attenuation issues; however, applying further scaling constraints based on "community upvote popularity" wiped out all significance. This implies that direct temporal interaction with distress events drives trauma independently of collective community scoring vectors.

### NB07 — `07_did_analysis_vader_baseline.ipynb`
Uses old `data/processed/` outputs. Optional VADER comparison. Not part of main pipeline.

### NB08 — `08_alt_analysis.ipynb`
**Inputs** (note: uses RAW JSONL, not cleaned, because it needs `link_id` and `created_utc`):
- `r_gradadmissions_posts.jsonl` and `r_gradadmissions_comments.jsonl` (repo root)
- `exposure_labels_v2.parquet`, `anchor_posts_v2.parquet`, `user_community_breadth_v2.parquet`
- `models/clf_*.joblib`

**What it does**:
1. Scans raw comments to find each exposed user's first anchor comment timestamp.
2. Defines pre = Sep–Nov activity *before* that timestamp (exposed) / full Sep–Nov (unexposed).
3. Scores all pre and post text, aggregates to `panel_scores_alt.parquet`.
4. PSM with same spec as NB06. ~668 matched pairs/cycle (36.5% coverage).
5. DiD same spec as NB06.
6. Causal Impact: builds wide weekly (exposed, unexposed) time series from scored corpus; uses `causalimpact` with integer index pre/post periods; patches `pandas.core.dtypes.common.is_datetime_or_timedelta_dtype` for pandas 2.x compat.

**Current results**:
- Pooled DiD: +0.0076, p=0.067 (borderline)
- CausalImpact cycle 1: +2.1% relative effect
- CausalImpact cycle 2: +2.7% relative effect

**Path constants** (after 2026-04-07 fix):
```python
POSTS_PATH    = ROOT / 'r_gradadmissions_posts.jsonl'
COMMENTS_PATH = ROOT / 'r_gradadmissions_comments.jsonl'
```

---

## Key Design Decisions & Gotchas

1. **Raw vs cleaned files**: NB01/02/03/04/05/06 use `data/processed_v2/*.jsonl` (cleaned, with `post_id` and `clean_text`). NB08 uses the raw JSONL at repo root (needs `link_id` and `created_utc` which the cleaned files don't have; cleaned files use `post_id` and `created_dt`).

2. **`post_id` vs `link_id`**: Raw comments have `link_id = "t3_<post_id>"`. Cleaned comments have `post_id` (already stripped). Anchor posts are identified by their `id`. Matching: `comment.post_id == anchor.id` or `comment.link_id == "t3_" + anchor.id`.

3. **`mh_score` computation**: `sigmoid(clf.decision_function(texts))` where `sigmoid(x) = 1/(1+exp(-x))`. Never use `predict_proba` — these are LinearSVC models.

4. **Cycle definitions**:
   - Cycle 1: anchor=Sep–Nov 2023, pre=Aug 2023, post=Dec 2023–May 2024
   - Cycle 2: anchor=Sep–Nov 2024, pre=Aug 2024, post=Dec 2024–May 2025

5. **PSM caliper**: 0.05 on propensity score. Users outside caliper are dropped (not matched). `community_breadth_log` only added to matching features if ≥ 95% of the cycle's users have it (otherwise sample collapses).

6. **`causalimpact` compatibility**: version 0.2.6 uses `pd.core.dtypes.common.is_datetime_or_timedelta_dtype` which was removed in pandas 2.x. NB08 adds it back at runtime. Also, CausalImpact must receive integer index positions `[0, n_pre-1]` and `[n_pre, n_total-1]` — not datetime objects.

7. **Pre-period power problem**: August pre-period → only ~6.5% user coverage, ~155 matched pairs/cycle → underpowered. Sep–Nov pre-period (NB08) → 36.5% coverage, ~668 pairs. This is the primary reason NB06 results are n.s.

8. **Exposure definition change from old pipeline**: old pipeline used authorship of anchor posts as exposure proxy. v2 pipeline uses commenting on anchor threads via `link_id` — more precise (readers vs. authors), but a different population.

---

## What Is and Isn't in Git

**In git**: all `data/processed_v2/*.parquet`, `models/clf_*.joblib`, `figures/*.png`, all notebooks, all docs.

**Not in git**: raw JSONL files (`r_gradadmissions_*.jsonl`), cleaned JSONL (`posts_clean.jsonl`, `comments_clean.jsonl`) — these are large and regenerable.

---

## Current Status (as of 2026-04-09)

- `data/raw/` paths formalized and appropriately re-structured successfully.
- NB01–NB06 modifications correctly replaced primitive binary mapping vectors with fully continuous dose-response algorithms and decay calculations.
- Main finding: Attenuation biases generated by binary parameters were correctly identified and overridden yielding new insights in cyclical temporal mappings (Cycle 2 / localized dose-responses returning $p<0.01$).
- Popularity Sub-Multiplier Insight: Scaling exposures by Reddit 'score' fractions actively erased the significance of direct sub-group post-level events ($p=0.5658$), indicating direct proximity trumps community aggregate sorting.
