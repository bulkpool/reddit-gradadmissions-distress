# Pipeline Details

This document describes what each notebook does, the data it produces, and key design decisions.

For running instructions, see [Quickstart](quickstart.md). For results, see [Results](results.md).

---

## Data Overview

**Source**: Three Reddit communities — r/GradAdmissions, r/MSCS, r/MBA — covering Aug 2022 – Jul 2025 (three admission cycles).

Raw inputs (none of them are tracked in git):

| Subreddit | Files | Location |
|-----------|-------|----------|
| r/GradAdmissions | `r_gradadmissions_posts.jsonl`, `r_gradadmissions_comments.jsonl` (Aug 2023 onwards) + `data/raw/r_gradadmissions_2022_{posts,comments}.jsonl` (Aug 2022 cycle) | repo root + `data/raw/` |
| r/MSCS | `data/raw/r_MSCS_{posts,comments}.jsonl` + `data/raw/r_MSCS_2022_{posts,comments}.jsonl` | `data/raw/` |
| r/MBA | `data/mba/posts_clean.jsonl.gz`, `data/mba/comments_clean.jsonl.gz` (pre-cleaned, includes 2022) | `data/mba/` (in git) |

All intermediate outputs go to `data/processed/{SUBREDDIT}/`. NB01 is skipped for MBA — its cleaned `.jsonl.gz` files are decompressed into the processed dir.

---

## Execution Order

```
For each SUBREDDIT in {gradadmissions, mscs, mba}:
    NB01 → NB03 → NB04 → NB05 → NB06 → NB08
After all three subreddits complete:
    NB07 (cross-subreddit comparison + combined pooled DiD)
    NB09 (anchor BART characterisation)

NB00, NB04a, NB05a, NB06a are diagnostic — not part of run_pipeline.py.
NB02 is one-off and should not be re-run (Arctic Shift API training data).
```

Orchestrated by `run_pipeline.py`, which injects the correct `SUBREDDIT` value into each notebook's config cell at runtime and executes via `nbconvert`. Source notebooks are never modified.

| Notebook | File | Depends on |
|----------|------|------------|
| 00 | `00_exploratory_topic_sentiment.ipynb` | Raw data (EDA only) |
| 01 | `01_clean_corpus.ipynb` | Raw JSONL files |
| 02 | `02_train_classifiers.ipynb` | Arctic Shift API (training corpus) |
| 03 | `03_exposure_labels.ipynb` | 01 outputs, 02 models, `facebook/bart-large-mnli` |
| 04 | `04_panel_scores.ipynb` | 01, 02, 03 |
| 04a | `04a_exposure_checks.ipynb` | 03, 04 (diagnostic) |
| 05 | `05_collect_community_breadth.ipynb` | 04, Arctic Shift API |
| 05a | `05a_pipeline_funnel.ipynb` | 03, 04 (diagnostic) |
| 06 | `06_did_analysis.ipynb` | 04, 05 |
| 06a | `06a_stratified_pre_exposure.ipynb` | 04, 05 (diagnostic) |
| 07 | `07_comparison_analysis.ipynb` | NB06 outputs for all 3 subreddits |
| 08 | `08_alt_analysis.ipynb` | Raw JSONL, 03 outputs, 02 models, 05 |
| 09 | `09_nli_anchor_validation.ipynb` | NB03 outputs for all 3 subreddits |

---

## Notebook 00 — Exploratory EDA

**File**: `notebooks/00_exploratory_topic_sentiment.ipynb`

Exploratory analysis only. Topic modelling (LDA) + VADER sentiment on a single month of r/GradAdmissions data. No outputs consumed downstream.

---

## Notebook 01 — Corpus Cleaning

**File**: `notebooks/01_clean_corpus.ipynb`

Concatenates the per-cycle raw JSONLs for the chosen subreddit, normalises text, and produces clean canonical files for downstream notebooks.

**Cleaning steps**:

| Step | What it does |
|------|--------------|
| Date / author validation | Parses `created_utc` → ISO datetime; drops null / deleted / removed / bot authors |
| Dedup | Deduplicates on `id`, keeps first occurrence |
| Text normalisation | Lowercase → strip URLs → strip non-alpha → collapse whitespace → `clean_text` |
| Comment → post mapping | Derives `post_id = link_id.removeprefix("t3_")` on comments, linking each comment to its parent thread |

The `post_id` field is what enables **thread-level** exposure identification in NB03.

**Outputs** → `data/processed/{SUBREDDIT}/`:

| File | Schema |
|------|--------|
| `posts_clean.jsonl` | `id, author, created_dt, clean_text, score, num_comments` |
| `comments_clean.jsonl` | `id, author, created_dt, post_id, clean_text, score` |

> NB01 is skipped for MBA (`PRE_CLEANED = {'mba'}` in `run_pipeline.py`) — decompress `data/mba/*.jsonl.gz` into `data/processed/mba/` once before running the MBA pipeline from NB03.

---

## Notebook 02 — SVM Classifier Training

**File**: `notebooks/02_train_classifiers.ipynb`

Trains three binary LinearSVCs following Low et al. (2020). **One-off — do not re-run**: it pulls training data via the Arctic Shift API and can time out.

**Training data** (Jan 2022 – Jul 2023, before the study window to prevent leakage):

| Classifier | Positive class | Negative (control) |
|------------|---------------|--------------------|
| anxiety | r/anxiety — 2,000 posts | r/personalfinance, r/learnprogramming, r/todayilearned, r/careerguidance |
| depression | r/depression — 2,000 posts | same |
| stress | r/stress — 2,000 posts | same |

**Features**: TF-IDF unigrams + bigrams, max 50k features. 5-fold CV F1: 0.88–0.94 per classifier.

**Composite score** (used in NB03 for anchor characterisation only, and in NB04 / NB08 for the outcome variable):

```
mh_score = mean(sigmoid(anx_df), sigmoid(dep_df), sigmoid(str_df))  ∈ (0, 1)
```

> **Note**: scoring uses `decision_function` + sigmoid, not `predict_proba` — the models are `LinearSVC` and don't expose calibrated probabilities.

**Outputs** → `models/`:

| File | Description |
|------|-------------|
| `clf_anxiety.joblib` | Fitted `Pipeline(TfidfVectorizer + LinearSVC)` |
| `clf_depression.joblib` | — |
| `clf_stress.joblib` | — |

---

## Notebook 03 — Exposure Labels

**File**: `notebooks/03_exposure_labels.ipynb`

Identifies anchor posts and classifies panel users as exposed or unexposed using thread-level linking via `post_id` / `link_id`.

**Anchor post definition** (post must meet both criteria):

1. Falls within the anchor period: Sep 1 – Nov 30 of the cycle year.
2. Matches at least one regex from the negative-admissions keyword lexicon (`reject(ed|ion)`, `decline`, `waitlist`, `no funding`, `mental health`, `anxi(ous|ety)`, `depress(ed|ion)`, `stress(ful)?`, `falling apart`, `gave up`, `imposter`, etc.) **and** zero-shot NLI via `facebook/bart-large-mnli` classifies the post as one of `negative admissions outcome`, `rejection or funding loss`, or `giving up on graduate school` — i.e. not the `general admissions discussion` control label (`bart_is_negative == True`).

> Anchor selection used to threshold on `mh_score > 0.45`. Replaced with the keyword + BART NLI rule on 2026-04-23 (commit `6b55593`) to break the circularity between SVM-defined treatment and SVM-measured outcome.

**Exposure classification**:

- **Exposed**: user commented on an anchor post thread (matched via `link_id`). Anchor post *authors* are excluded.
- **Unexposed**: active in the subreddit across the full Aug 1 – May 31 cycle window but never commented on any anchor thread.

**Cycle windows** (chronological):

| | Cycle 1 | Cycle 2 | Cycle 3 |
|-|---------|---------|---------|
| Anchor period | Sep – Nov 2022 | Sep – Nov 2023 | Sep – Nov 2024 |
| Active window | Aug 2022 – May 2023 | Aug 2023 – May 2024 | Aug 2024 – May 2025 |

**Outputs** → `data/processed/{SUBREDDIT}/`:

| File | Description |
|------|-------------|
| `anchor_posts.parquet` | Anchor posts with SVM scores, BART top-label and top-neg score, and `bart_is_negative` flag |
| `exposure_labels.parquet` | `(author, cycle, exposed, exposure_intensity, exposure_prob)` per user-cycle. `exposure_intensity` = max BART top-neg score across anchor threads commented on; `exposure_prob` = popularity-weighted exposure score in [0, 1] |

---

## Notebook 04 — Panel Scoring

**File**: `notebooks/04_panel_scores.ipynb`

Scores each panel user's text in two windows, aggregates to user-level pre / post panel rows, and emits a post-level dataset for the NB06 post-level DiD plus a dose-exposure count.

**Scoring windows** (`PRE_PERIOD_STRATEGY = 'anchor_window_before_first_comment'`):

| | Cycle 1 | Cycle 2 | Cycle 3 |
|-|---------|---------|---------|
| Pre baseline | Sep–Nov 2022 *before* each user's first anchor comment | Sep–Nov 2023 *before* first anchor | Sep–Nov 2024 *before* first anchor |
| Post outcome | Dec 2022 – May 2023 | Dec 2023 – May 2024 | Dec 2024 – May 2025 |

> The pre-period was August-only in an earlier draft. NB04 was switched to Sep–Nov on 2026-04-13 (commit `17db224`), raising panel coverage from ~6.5 % to ~36 % and matched-pair counts by roughly 4×.

**Outputs** → `data/processed/{SUBREDDIT}/`:

| File | Schema | Description |
|------|--------|-------------|
| `panel_scores.parquet` | `author, cycle, exposed, exposure_intensity, pre_{mh,anx,dep,str}_score, pre_n_posts, post_{mh,anx,dep,str}_score, post_n_posts` | User-level pre / post means + dimension scores |
| `post_level_scores.parquet` | `author, cycle, window, created_dt, anx_score, dep_score, str_score, mean_mh_score` | One row per post / comment for the post-level DiD |
| `dose_exposure.parquet` | `author, cycle, n_anchor_comments, log1p_n_anchor` | Per-user count of anchor-thread comments (for dose-response) |

> `panel_scores.parquet` exposes columns named `pre_mh_score` / `post_mh_score`; `post_level_scores.parquet` exposes `mean_mh_score` (not `mh_score`). NB06 reshapes to a single `mh_score` column at runtime.

---

## Notebook 05 — Community Breadth Collection

**File**: `notebooks/05_collect_community_breadth.ipynb`

Queries the Arctic Shift API for each panel user's cross-subreddit activity and computes **community breadth** = number of distinct subreddits a user posted / commented in across the study window, excluding the focal admissions subreddit and the user's own profile sub.

**Fault tolerance**: checkpointed to `data/processed/{SUBREDDIT}/breadth_checkpoint.jsonl` every 500 users.

**Coverage**: 100 %. Failed fetches and deleted accounts are imputed to `community_breadth = 0` so that PSM can use `community_breadth_log` without dropping rows (commit `b9830d4`, 2026-04-22).

**Output** → `data/processed/{SUBREDDIT}/user_community_breadth.parquet`:

| Column | Description |
|--------|-------------|
| `author` | Reddit username |
| `community_breadth` | Number of distinct subreddits |
| `community_breadth_log` | `log1p(breadth)` — used in regression |
| `subreddits_json` | JSON list of subreddit names |
| `status` | `"ok"`, `"http_404"`, etc. |

---

## Notebook 06 — Main Analysis (PSM + DiD)

**File**: `notebooks/06_did_analysis.ipynb`

The primary per-subreddit results notebook. Runs PSM, user-level DiD across four treatment encodings, per-dimension DiD, RQ2 breadth moderation, post-level DiD, dose-response, and a battery of robustness checks.

**Steps**:

1. **PSM per cycle**: 1 : 1 nearest-neighbour matching on the propensity score from a logistic regression on `pre_mh_score + log1p(n_posts_pre)` (+ `community_breadth_log` if ≥ 95 % coverage — currently always). Features standardised before LR. Caliper = 0.05. SMD balance check printed per feature.

2. **Formal parallel-trends pre-test (§4a)**: in the pre-period only, regress `mh_score ~ week_number × exposed + log1p(n_posts) + C(cycle_str)` over matched panel users. Saves `parallel_trends_test_v2.csv` and `fig_parallel_trends_pretest_{SUBREDDIT}.png`. The `week × exposed` interaction is n.s. across every subreddit, every cycle, and pooled — assumption holds.

3. **User-level DiD** (long format, one row per user × period). OLS with HC3 robust SEs:

   ```
   mh_score ~ period + exposed + period × exposed + log1p(n_posts) + C(cycle_str)
   ```

   Run per cycle and pooled. `C(cycle_str)` is added automatically when more than one cycle is present (fix in commit `3730e47`, 2026-04-25). Specifications: **Binary DiD**, **Intensity DiD** (`exposure_intensity`), **Exposure-Prob DiD** (`exposure_prob`), **GPS-WLS** (Hirano–Imbens weighting). All four written to `did_summary.csv`.

4. **Per-dimension DiD**: same OLS spec on `anx_score`, `dep_score`, `str_score` separately.

5. **RQ2 — Breadth moderation**: three-way interaction `period × exposed × community_breadth_log`.

6. **Post-level DiD** (~22 K observations on matched-author posts only): one row per individual post; run with and without user fixed effects; cluster SEs on author.

7. **Dose-response**: coefficient on `log1p(n_anchor_comments)` interacted with `period`.

8. **Robustness checks (§11)**: placebo test, tight Dec–Jan post-window, BART confidence sensitivity at 0.4 / 0.6.

**Headline results (Binary DiD, pooled, cycle FE, 2026-04-26)**:

| Subreddit | ATT | 95 % CI | p |
|-----------|-----|---------|---|
| r/GradAdmissions | +0.0041 | [−0.0027, +0.0110] | 0.235 n.s. |
| r/MSCS | −0.0073 | [−0.0199, +0.0053] | 0.256 n.s. |
| r/MBA | +0.0065 | [+0.0011, +0.0119] | **0.019\*** |

Sources: `data/processed/{sub}/did_summary.csv`. See [Results](results.md) for the full per-cycle / per-spec / per-dimension tables.

**Output files** → `data/processed/{SUBREDDIT}/`:

- `did_summary.csv` — every specification × cycle row
- `parallel_trends_test_v2.csv` — formal pre-trend test results

**Figures** → `figures/`: `fig_att_coef_{sub}.png`, `fig_parallel_trends_{sub}.png`, `fig_parallel_trends_pretest_{sub}.png`.

---

## Notebook 07 — Cross-Subreddit Comparison

**File**: `notebooks/07_comparison_analysis.ipynb`

Runs after NB06 has completed for all three subreddits. Builds the cross-community comparison and computes the **combined pooled DiD** (§9) — the headline result that pools across all three subreddits and all three cycles.

**Combined pooled DiD (§9)** stacks each subreddit's matched long-format dataframe and runs:

```
mh_score ~ period + exposed + period × exposed + log1p(n_posts)
         + C(subreddit) + C(cycle_str)
```

**Headline result** (HC3 SEs):

| ATT | 95 % CI | p | n matched users | n obs |
|-----|---------|---|-----------------|-------|
| **+0.0045** | [+0.0007, +0.0083] | **0.021\*** | 6,012 | 12,416 |

**Other NB07 sections**:

- §6 — pairwise Z-test across subreddit-spec combinations (9 tests; all n.s., min p = 0.25 — no subreddit is significantly different from another).
- §7 — anchor-post characteristics + pre/post `mh_score` distributions across subreddits.
- §8 — `comparison_summary.parquet`: panel size, exposed count, matched pairs, anchor counts, and per-subreddit pooled ATT/CI for every (subreddit, cycle) row.

**Outputs** → `data/processed/`:

| File | Description |
|------|-------------|
| `combined_pooled_did.csv` | The §9 headline pooled ATT row |
| `comparison_summary.parquet` | Per-subreddit per-cycle metrics |

**Figures** → `figures/`: `fig_att_comparison.png` (forest plot, 3 subs side by side), `fig_mhscore_distributions.png`, `fig_anchor_comparison.png`.

---

## Notebook 08 — Alternative Analysis: Re-Scored Panel + Causal Impact

**File**: `notebooks/08_alt_analysis.ipynb`

Independent implementation of the panel + DiD. Re-scores raw text inline (rather than reading NB04's aggregates) and adds Bayesian Causal Impact (BSTS) as a second identification strategy.

**Inputs**: `data/processed/{SUBREDDIT}/{posts,comments}_clean.jsonl`, `exposure_labels.parquet`, `anchor_posts.parquet`, `user_community_breadth.parquet`, plus the three SVM models.

**Approach 1 — Re-scored DiD**:

- Pre = each user's Sep–Nov activity *before* their first anchor comment (exposed users) or full Sep–Nov (unexposed).
- Post = Dec 1 – May 31 (same as NB06).
- PSM uses LR + 1-NN ball-tree (no scaler; differs from NB06's StandardScaler + euclidean NN), 1:1, caliper 0.05.

**Approach 2 — Causal Impact** (Bayesian structural time series):

- Aggregates exposed vs. unexposed weekly mean `mh_score` into a wide time series.
- Uses unexposed group as a synthetic control via the `causalimpact` library.
- Pre-period: Sep 1 – Nov 30. Post-period: Dec 1 – May 31. One figure per cycle present in the panel.

**`causalimpact 0.2.6` is incompatible with pandas 2.x**; the notebook patches `pandas.core.dtypes.common.is_datetime_or_timedelta_dtype` at runtime and passes integer index positions (not datetime objects) to `CausalImpact()`.

**Output** → `data/processed/{SUBREDDIT}/panel_scores_alt.parquet` (alt panel with re-scored Sep–Nov pre-period + always-100 % community breadth coverage).

**Figures** → `figures/`: `fig_parallel_trends_alt_{sub}.png`, `fig_causal_impact_cycle{1,2,3}_{sub}.png`.

---

## Notebook 09 — Anchor BART Characterisation

**File**: `notebooks/09_nli_anchor_validation.ipynb`

Reads BART fields already present in each subreddit's `anchor_posts.parquet` (no inference is run; NB03 already produced them). Summarises label distributions, BART top-neg score statistics, per-cycle counts, and shows the BART score distribution plus SVM-vs-BART scatter.

**Figures** → `figures/fig_nli_validation_scores.png`.

---

## Diagnostic Notebooks (not in `run_pipeline.py`)

| NB | What it answers |
|----|-----------------|
| **04a** `04a_exposure_checks.ipynb` | Differential attrition between exposed and unexposed, pre-period span distribution, anchor-comment volume by month, baseline comparison of panel users vs. dropped users. Figures: `fig_preperiod_span.png`, `fig_anchor_comment_timing.png`. |
| **05a** `05a_pipeline_funnel.ipynb` | Waterfall: labelled → has activity → has pre-period → has post-period → in panel. Diagnoses where exposed users drop out (~30 % no activity at all, ~25 % missing pre, ~10 % missing post). Figures: `fig_pipeline_funnel.png`, `fig_funnel_exposed_vs_unexposed.png`. |
| **06a** `06a_stratified_pre_exposure.ipynb` | PSM + DiD sensitivity to pre-period span restrictions (full / ≥ 7d / ≥ 14d). Figure: `fig_sensitivity_pre_period.png`. |
