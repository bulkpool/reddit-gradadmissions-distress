# LLM Codebase Context

**Purpose**: Dense, authoritative snapshot for AI assistants. Read this file before touching anything else in the repo. After making any changes, update the relevant sections here so future sessions start fresh.

**Last updated**: 2026-04-23 (MBA NB06 OOM crash fixed; formal parallel trends test added to NB06 §4a; combined pooled DiD across all 3 subreddits added to NB07 §9; parallel_trends_test_v2.csv + fig_parallel_trends_pretest_{SUBREDDIT}.png + combined_pooled_did.csv now generated)

---

## Project in One Paragraph

Causal inference study on **r/GradAdmissions, r/MSCS, and r/MBA**. All three subreddits now cover **Aug 2022–Jul 2025 (three admission cycles)**. RQ1: does exposure to high-distress anchor posts increase users' mental health distress scores in the following Dec–May decision season? RQ2: does cross-Reddit community breadth moderate this effect? Method: SVM-based distress scoring → anchor post identification → propensity-score matched DiD. The pipeline is parameterized by `SUBREDDIT` (set at top of each notebook). Orchestrated via `run_pipeline.py`.

**Cycle numbering (chronological):** `cycle=1` → 2022 cycle (anchor Sep–Nov 2022, post Dec 2022–May 2023). `cycle=2` → 2023 cycle. `cycle=3` → 2024 cycle. NB06/NB07/NB08 now derive the cycle list dynamically from each panel (no hardcoded `[1, 2]`).

**Headline results (3-subreddit combined pooled DiD, NB07 §9, 2026-04-23)**:
- **Combined pooled DiD (subreddit + cycle FE) = +0.0045, 95% CI [+0.0007, +0.0083], p = 0.021*** — significant at p < 0.05 across 6,012 matched users, 12,416 obs. This is the primary cross-community result.
- Per subreddit (cycle FE): r/GradAdmissions +0.0038 p=0.255 n.s., r/MSCS +0.0065 p=0.270 n.s., r/MBA +0.0055 p=0.032*
- Per cycle (subreddit FE): Cycle 1 +0.0042 p=0.255 n.s., Cycle 2 +0.0047 p=0.141 n.s., Cycle 3 +0.0044 p=0.171 n.s.
- Per dimension (all pooled): Anxiety +0.0045 p=0.023*, Depression +0.0045 p=0.041*, Stress +0.0045 p=0.030*
- **Formal parallel trends test (NB06 §4a)**: week_number × exposed interaction is n.s. across all subreddits, cycles, and pooled spec — parallel trends assumption confirmed.
- Per-subreddit NB06 pooled results: GA +0.0037 p=0.27 (n.s.), MSCS +0.0065 p=0.27 (n.s.), MBA +0.0066 p=0.017* (significant)

---

## Environment

- Python venv: `~/venvs/jupyter/bin/` — always use this, never system Python
- Run notebooks: `~/venvs/jupyter/bin/jupyter nbconvert --to notebook --execute <nb>`
- Key packages: `scikit-learn`, `statsmodels`, `joblib`, `pandas` (2.x), `numpy`, `matplotlib`, `causalimpact==0.2.6`, `torch`, `transformers` (needed by NB03 for BART inference)

---

## Repo Layout

```
/                                        ← repo root
├── r_gradadmissions_posts.jsonl         ← NOT in git — raw posts (GA, 2023 + 2024 cycles)
├── r_gradadmissions_comments.jsonl      ← NOT in git — raw comments (GA, 2023 + 2024 cycles)
├── run_pipeline.py                      ← orchestrator: injects SUBREDDIT, runs NB01→NB08+NB07
├── notebooks/
│   ├── 00_exploratory_topic_sentiment.ipynb  ← EDA only; SUBREDDIT config at top
│   ├── 01_clean_corpus.ipynb             ← SUBREDDIT config at top
│   ├── 02_train_classifiers.ipynb        ← one-off setup, avoid running to prevent API timeouts
│   ├── 03_exposure_labels.ipynb       ← SUBREDDIT config at top
│   ├── 04_panel_scores.ipynb             ← SUBREDDIT config at top
│   ├── 04a_exposure_checks.ipynb         ← diagnostic: attrition, pre-period quality, anchor timing; SUBREDDIT config at top
│   ├── 05_collect_community_breadth.ipynb ← SUBREDDIT config at top
│   ├── 05a_pipeline_funnel.ipynb         ← diagnostic: waterfall funnel; SUBREDDIT config at top
│   ├── 06_did_analysis.ipynb          ← MAIN RESULTS; SUBREDDIT config at top
│   ├── 06a_stratified_pre_exposure.ipynb ← diagnostic: PSM+DiD sensitivity; SUBREDDIT config at top
│   ├── 07_comparison_analysis.ipynb      ← 3-subreddit comparison (NB07)
│   ├── 08_alt_analysis.ipynb             ← Sep–Nov pre-period + CausalImpact; SUBREDDIT config
│   └── 09_nli_anchor_validation.ipynb   ← Anchor post characterisation: reads BART fields from anchor_posts.parquet; no inference (all 3 subreddits; no SUBREDDIT param)
├── data/
│   ├── raw/
│   │   ├── r_gradadmissions_2022_posts.jsonl    ← NOT in git — GA 2022 cycle posts (Aug'22–Jul'23)
│   │   ├── r_gradadmissions_2022_comments.jsonl ← NOT in git — GA 2022 cycle comments
│   │   ├── r_MSCS_posts.jsonl            ← NOT in git — MSCS posts (2023 + 2024 cycles)
│   │   ├── r_MSCS_comments.jsonl         ← NOT in git — MSCS comments
│   │   ├── r_MSCS_2022_posts.jsonl       ← NOT in git — MSCS 2022 cycle posts
│   │   └── r_MSCS_2022_comments.jsonl    ← NOT in git — MSCS 2022 cycle comments
│   ├── mba/
│   │   ├── posts_clean.jsonl.gz          ← in git — pre-cleaned MBA posts (gzipped, 15MB)
│   │   └── comments_clean.jsonl.gz       ← in git — pre-cleaned MBA comments (gzipped, 65MB)
│   └── processed/
│       ├── gradadmissions/               ← GradAdmissions pipeline outputs
│       │   ├── anchor_posts.parquet
│       │   ├── exposure_labels.parquet
│       │   ├── panel_scores.parquet
│       │   ├── post_level_scores.parquet
│       │   ├── dose_exposure.parquet
│       │   ├── panel_scores_alt.parquet
│       │   ├── user_community_breadth.parquet
│       │   ├── did_summary.csv           ← NB06 output
│       │   └── parallel_trends_test_v2.csv ← NB06 §4a output
│       ├── mscs/                         ← MSCS pipeline outputs (same file set)
│       ├── mba/                          ← MBA pipeline outputs (same file set)
│       │   ├── posts_clean.jsonl         ← NOT in git — decompressed from data/mba/
│       │   └── comments_clean.jsonl      ← NOT in git — decompressed from data/mba/
│       ├── comparison_summary.parquet    ← NB07 output (all key metrics, 3 subreddits)
│       ├── combined_pooled_did.csv       ← NB07 §9 output (headline pooled ATT across all 3 subreddits)
│       └── nli_anchor_validation.parquet ← OBSOLETE — delete if present; BART fields now in anchor_posts.parquet
├── models/                               ← clf_anxiety/depression/stress.joblib
├── figures/                              ← all PNG outputs
└── docs/
    ├── LLM_CONTEXT.md                    ← THIS FILE — update after every change
    ├── pipeline.md, flow.md, methodology.md, results.md, quickstart.md
```

---

## Collaborators

| Name | GitHub | Role |
|------|--------|------|
| Ayush Prasad | bulkpool | Main pipeline (NB04, NB06) |
| Sabelle Huang | sabelle / vedaduddu14 | NB08, docs |

---

## Data File Schemas (complete)

### Raw JSONL (repo root, NOT in git)

**`r_gradadmissions_posts.jsonl`** — key fields used by pipeline:
- `id` (str), `author` (str), `created_utc` (int, unix timestamp), `selftext` (str), `score` (int), `num_comments` (int)

**`r_gradadmissions_comments.jsonl`** — key fields:
- `id` (str), `author` (str), `created_utc` (int), `body` (str), `link_id` (str, format `"t3_<post_id>"`), `score` (int)

> Strip `t3_` from `link_id` to get the parent post's `id`. This is how exposure is identified.

### Cleaned JSONL (`data/processed/`, NOT in git — run NB01 to generate)

**`posts_clean.jsonl`**: `id, author, created_dt (ISO str), clean_text, score, num_comments`

**`comments_clean.jsonl`**: `id, author, created_dt (ISO str), post_id (= link_id minus "t3_"), clean_text, score`

> Note: cleaned files use `created_dt` (string) and `clean_text`, NOT `created_utc` and `selftext`/`body`. Do not confuse with raw fields.

### Parquet Files (`data/processed/gradadmissions/` or `data/processed/mscs/`)

> All parquet outputs are now in subreddit-specific subdirectories. `SUBREDDIT` at top of each notebook controls which dir is used.
> Row counts below are from the earlier 2-cycle run — re-run the pipeline to refresh with the 2022 cycle included.

**`exposure_labels.parquet`** — GA: 28,758 rows (exposed=3,999) | MSCS: 4,706 (exposed=496) | MBA: 32,283 (exposed=5,587) [3-cycle, BART-selected anchors]
| Column | Type | Notes |
|--------|------|-------|
| `author` | str | Reddit username |
| `exposed` | bool | — |
| `cycle` | int64 | 1, 2, or 3 (1 = 2022, 2 = 2023, 3 = 2024) |
| `exposure_intensity` | float64 | max BART top-neg score across anchor threads commented on (0–1); 0 for unexposed |
| `exposure_prob` | float64 | popularity-weighted exposure score [0,1] |

**`anchor_posts.parquet`** — GA: 2,010 | MSCS: 144 | MBA: 885 [3-cycle, BART-selected; keyword filter + bart_is_negative==True]
| Column | Type | Notes |
|--------|------|-------|
| `id` | str | Reddit post ID |
| `author` | str | — |
| `created_dt` | datetime64[ns, UTC] | — |
| `cycle` | float64 | 1.0, 2.0, or 3.0 |
| `clean_text` | str | — |
| `anx_score` | float64 | SVM score (characterisation only — not used for selection) |
| `dep_score` | float64 | — |
| `str_score` | float64 | — |
| `mean_mh_score` | float64 | — |
| `bart_top_label` | str | Highest-scoring BART label (one of 4 candidates) |
| `bart_top_score` | float64 | Entailment score for `bart_top_label` |
| `bart_top_neg_label` | str | Highest-scoring label among the 3 negative candidates |
| `bart_top_neg_score` | float64 | Entailment score for `bart_top_neg_label`; used as `exposure_intensity` |
| `bart_is_negative` | bool | Always True (selection criterion) |
| `score` | int64 | Reddit upvotes |
| `num_comments` | int64 | — |

**`panel_scores.parquet`** — GA: 10,161 rows (9,801 users, exposed=1,236) | MSCS: 2,112 (2,049, exposed=242) | MBA: 10,338 (9,229, exposed=1,891) [3-cycle, BART-selected]
| Column | Type | Notes |
|--------|------|-------|
| `author` | str | — |
| `cycle` | int64 | 1, 2, or 3 |
| `exposed` | bool | True=322 / False=1,101 |
| `exposure_intensity` | float64 | max BART top-neg score across anchor threads commented on (0–1); 0 for unexposed |
| `pre_mh_score` | float64 | [0.103, 0.661] — Aug mean |
| `pre_anx_score` | float64 | [0.103, 0.672] |
| `pre_dep_score` | float64 | [0.088, 0.648] |
| `pre_str_score` | float64 | [0.117, 0.719] |
| `pre_n_posts` | int64 | [1, 482] |
| `post_mh_score` | float64 | [0.154, 0.655] — Dec–May mean |
| `post_anx_score` | float64 | [0.174, 0.660] |
| `post_dep_score` | float64 | [0.138, 0.688] |
| `post_str_score` | float64 | [0.150, 0.657] |
| `post_n_posts` | int64 | [1, 1453] |

> Only 1,368 of 20,932 panel users are active in August (~6.5%) — the primary power bottleneck.

**`panel_scores_alt.parquet`** — 7,868 rows, 7,644 unique authors (Sep–Nov pre-period, NB08)
| Column | Type | Notes |
|--------|------|-------|
| `author` | str | — |
| `exposed` | bool | True=758 / False=7,110 |
| `cycle` | int64 | 1, 2, or 3 |
| `pre_mh_score` | float64 | [0.138, 0.750] |
| `pre_n_posts` | int64 | [1, 87] |
| `post_mh_score` | float64 | [0.147, 0.778] |
| `post_n_posts` | int64 | [1, 1556] |
| `community_breadth` | float64 | [0, 100], no nulls (100% coverage) |
| `community_breadth_log` | float64 | log1p(breadth), no nulls |

**`post_level_scores.parquet`** — 147,569 rows (all matched-panel users, pre+post) [2-cycle]
| Column | Type | Notes |
|--------|------|-------|
| `author` | str | 10,693 unique |
| `cycle` | int64 | 1, 2, or 3 |
| `window` | str | `"pre"` or `"post"` (NOT period 0/1) |
| `created_dt` | datetime64[ns, UTC] | — |
| `anx_score` | float64 | [0.065, 0.988] |
| `dep_score` | float64 | [0.056, 0.866] |
| `str_score` | float64 | [0.056, 0.940] |
| `mean_mh_score` | float64 | [0.070, 0.830] — note: col is `mean_mh_score` not `mh_score` |

> NB06 post-level DiD filters this to matched authors only → 22,355 observations.

**`dose_exposure.parquet`** — 2,300 rows [2-cycle]
| Column | Type | Notes |
|--------|------|-------|
| `author` | str | 2,257 unique |
| `cycle` | float64 | 1.0, 2.0, or 3.0 |
| `n_anchor_comments` | int64 | [1, 49] |
| `log1p_n_anchor` | float64 | [0.693, 3.912] |

**`user_community_breadth.parquet`** — 100% coverage of panel users (e.g., 7,770 rows for GA)
| Column | Type | Notes |
|--------|------|-------|
| `author` | str | — |
| `community_breadth` | float64 | [0, 100], missing/errors imputed as 0 |
| `community_breadth_log` | float64 | log1p(breadth) |
| `subreddits_json` | str | JSON list of subreddit names |
| `status` | str | `"ok"`, `"http_404"`, etc. |

> Coverage is now 100%. Missing accounts or errors default to `community_breadth = 0` so that PSM in NB06 can utilize `community_breadth_log` without dropping rows.

> `nli_anchor_validation.parquet` is now obsolete — BART fields are embedded directly in each subreddit's `anchor_posts.parquet`. Delete it if present.

---

## Anchor Post Identification (NB03)

**Selection method (as of 2026-04-22):** keyword filter + BART NLI (`facebook/bart-large-mnli`).
SVM scores are still computed and saved in `anchor_posts.parquet` for characterisation, but are **not** used for selection. This breaks the circularity between the treatment definition and the SVM-based outcome measure in NB04/NB06.

**Selection logic**: post must match keyword AND `bart_is_negative == True` (BART top label is not `"general admissions discussion"`).

**BART candidate labels:**
```python
CANDIDATE_LABELS = [
    'negative admissions outcome',
    'rejection or funding loss',
    'giving up on graduate school',
    'general admissions discussion',   # ← negative class
]
```

**`exposure_intensity`** in `exposure_labels.parquet` = max `bart_top_neg_score` (float 0–1) across anchor threads the user commented on. Previously was `n_dims` (int 0–3, number of SVM dimensions above p67). NB06 RQ1b intensity DiD uses this float score.

---

## Key Functions

### `score_texts` / `score_texts_all` (NB04, NB08)
```python
def score_texts_all(texts):  # NB04 name
    anx  = sigmoid(clf_anx.decision_function(texts))
    dep  = sigmoid(clf_dep.decision_function(texts))
    str_ = sigmoid(clf_str.decision_function(texts))
    mean = np.stack([anx, dep, str_], axis=1).mean(axis=1)
    return anx, dep, str_, mean  # all np.arrays of shape (n,)

def sigmoid(x): return 1 / (1 + np.exp(-x))
```
> Uses `decision_function` + sigmoid, NOT `predict_proba`. Models are LinearSVC.

### `psm_match` — NB06 (StandardScaler + euclidean NN on propensity score)
```python
def psm_match(df, features, caliper):
    clean = df.dropna(subset=features).copy()
    X = StandardScaler().fit_transform(clean[features])
    lr = LogisticRegression(max_iter=1000, random_state=42).fit(X, clean['exposed'].astype(int))
    clean['pscore'] = lr.predict_proba(X)[:, 1]
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(unexp_df[['pscore']])
    # returns: matched_authors (set), n_pairs (int), clean (DataFrame)
```

### `psm_match` — NB08 (no scaler, ball_tree NN, returns DataFrame not set)
```python
def psm_match(df, caliper=0.05):
    # features = ['pre_mh_score', 'log1p_pre_n_posts'] + optionally 'community_breadth_log'
    lr = LogisticRegression(max_iter=1000).fit(X, y)
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(control[['pscore']])
    # returns: matched DataFrame (both treated + control rows)
```

### PSM Feature Selection Logic (NB06)
```python
CALIPER = 0.05
MATCH_FEATURES = ['pre_mh_score', 'log1p_n_posts_pre']
breadth_coverage = panel['community_breadth'].notna().mean()
if breadth_coverage >= 0.95:
    MATCH_FEATURES.append('community_breadth_log')
# else: match on pre_mh_score + log1p_n_posts_pre only
```

### `run_did` — NB06
```python
def run_did(df, label):
    # df must have: mh_score, period (0/1), exposed (int), period_x_exposed, log1p_n_posts
    m = smf.ols('mh_score ~ period + exposed + period_x_exposed + log1p_n_posts',
                data=df.dropna(subset=['mh_score'])).fit(cov_type='HC3')
    # returns: fitted OLS model
```

### `assign_window` — NB04 (cycle/window from datetime)
```python
def assign_window(dt):
    for cycle, w in CYCLES.items():
        if w['pre_start'] <= dt <= w['pre_end']:   return cycle, 'pre'
        if w['post_start'] <= dt <= w['post_end']:  return cycle, 'post'
    return None  # outside all windows
```

### `assign_window` — NB08 (respects first anchor comment cutoff)
```python
def assign_window(author, dt, cycle):
    if w['post_start'] <= dt <= w['post_end']:  return 'post'
    if w['anchor_start'] <= dt <= w['anchor_end']:
        cutoff = first_anchor_comment.get(author)  # None if unexposed
        if cutoff is None or dt < cutoff:  return 'pre'
    return None
```

---

## Figure → Notebook Mapping

| Figure | Produced by | Description |
|--------|-------------|-------------|
| `fig_att_coef_{SUBREDDIT}.png` | NB06 | ATT coefficient plot (user-level DiD per cycle + pooled) |
| `fig_parallel_trends_{SUBREDDIT}.png` | NB06 | Pre/post mean mh_score — August pre-period matched panel |
| `fig_parallel_trends_alt_{SUBREDDIT}.png` | NB08 | Pre/post mean mh_score — Sep–Nov pre-period matched panel |
| `fig_causal_impact_cycle{1,2,3}_{SUBREDDIT}.png` | NB08 | CausalImpact BSTS output — one per cycle present in the panel |
| `fig_nli_validation_scores.png` | NB09 | Histogram of BART top-negative-label scores + SVM vs BART scatter (all anchor posts) |
| `fig_nli_kappa_confusion.png` | NB09 | Confusion matrix for Cohen's Kappa (requires posts_clean.jsonl) |
| `fig_parallel_trends_pretest_{SUBREDDIT}.png` | NB06 §4a | Pre-period weekly mh_score trend: exposed vs unexposed (formal parallel trends test; one per subreddit) |
| `fig_att_comparison.png` | NB07 | Forest plot: ATT coefficients for all 3 subreddits side by side |
| `fig_mhscore_distributions.png` | NB07 | Pre/post mh_score boxplots: all 3 subreddits |
| `fig_anchor_comparison.png` | NB07 | Anchor post characteristics: all 3 subreddits |
| `fig_preperiod_span.png` | NB04a | Distribution of pre-period span (days) for exposed panel users + median by month |
| `fig_anchor_comment_timing.png` | NB04a | Anchor comment volume by month + first comment per user (exposure moment) |
| `fig_pipeline_funnel.png` | NB05a | Waterfall: labeled → has activity → has pre → has post → in panel (exposed users) |
| `fig_funnel_exposed_vs_unexposed.png` | NB05a | Side-by-side bar: dropout category rates for exposed vs unexposed |
| `fig_sensitivity_pre_period.png` | NB06a | ATT + 95% CI across 3 pre-period length restrictions (full, ≥7d, ≥14d) |
| `fig_event_study_v2.png` | old pipeline | Event study with SVM mh_score (±2 weeks) |
| `fig_event_study_clean.png` | old pipeline | Event study with 95% CI (clean version) |
| `fig_event_study.png` | old pipeline | Event study with VADER distress |
| `fig_did_estimates.png` | old pipeline | VADER vs SVM DiD comparison + cross-cycle |
| `fig_significance.png` | old pipeline | mh_score by outcome label + seasonal pattern |
| `fig_anchor_eda.png` | old pipeline | Anchor post breakdown by label/VADER |
| `fig_anchor_posts_per_week.png` | old pipeline | Anchor post volume per week |
| `fig_community_breadth_dist.png` | old pipeline | Breadth distribution (raw + log) |
| `fig_breadth_by_exposure.png` | old pipeline | Breadth: exposed vs unexposed |
| `fig_classifier_scores.png` | old pipeline | Score distributions for 3 classifiers |
| `fig_score_correlation.png` | old pipeline | mh_score vs VADER correlation |
| `fig_weekly_sentiment.png` | old pipeline | Weekly VADER sentiment + post volume |
| `fig_monthly_distress.png` | old pipeline | Monthly mean distress + volume |

---

## Cell IDs for NotebookEdit

Use these with the `NotebookEdit` tool (`cell_id` parameter) to target specific cells without reading the whole notebook.

### NB01 `01_clean_corpus.ipynb`
| cell_id | Content |
|---------|---------|
| `b2c3d4e5` | imports + path setup |
| `d4e5f6a7` | regex constants (_URL_RE, _NONALPHA) |
| `f6a7b8c9` | read raw posts loop |
| `c1d2e3f4` | write clean JSONLs to disk |

### NB02 `02_train_classifiers.ipynb`
| cell_id | Content |
|---------|---------|
| `b2c3d4e5` | imports + path setup |
| `d4e5f6g7` | MH_SUBREDDITS list + API pull |
| `h8i9j0k1` | train/eval loop per classifier |

### NB03 `03_exposure_labels.ipynb`
| cell_id | Content |
|---------|---------|
| `ad0e35c3` | SUBREDDIT config + path setup |
| `d2a34181` | imports |
| `5daa73aa` | load posts |
| `7765a7b6` | cycle assignment |
| `52c84e0e` | load SVM classifiers |
| `f88617fb` | load comments (uses `post_id`, `created_dt`) |

### NB04 `04_panel_scores.ipynb`
| cell_id | Content |
|---------|---------|
| `2819b08e` | SUBREDDIT config + path setup |
| `b1000002` | imports |
| `d1000004` | load exposure labels |
| `f1000006` | first anchor comment scan (uses cleaned JSONL) |
| `g1000007` | load classifiers |
| `i1000009` | `process_file` over cleaned JSONL (`clean_text`, `created_dt`) |
| `ff080f5a` | save post_level_scores.parquet |
| `71c81c1c` | dose-response: count anchor comments |
| `m1000013` | aggregate to user-level pre/post |
| `r1000018` | save panel_scores.parquet |

### NB04a `04a_exposure_checks.ipynb`
| cell_id | Content |
|---------|---------|
| `a1b2c3d4` | markdown header |
| `b2c3d4e5` | imports + DATA_DIR (`processed/gradadmissions`) |
| `d4e5f6a7` | differential attrition check (in_panel rate by exposed/unexposed) |
| `e5f6a7b8` | attrition by cycle |
| `a7b8c9d0` | pre-period span computation (pre_first, pre_last, pre_span_days) |
| `b8c9d0e1` | span threshold breakdown (0, 3, 7, 14, 30 days) |
| `c9d0e1f2` | figure: span distribution + median span by month |
| `e1f2a3b4` | anchor comment timing scan (raw JSONL, first anchor comment per user) |
| `f2a3b4c5` | figure: anchor comments by month + first comment per user |
| `b4c5d6e7` | panel vs dropped user baseline comparison |
| `d6e7f8a9` | diagnostic summary printout |

### NB05a `05a_pipeline_funnel.ipynb`
| cell_id | Content |
|---------|---------|
| `a0000001` | markdown header |
| `b0000002` | imports + DATA_DIR (`processed/gradadmissions`) |
| `d0000004` | build has_pre/has_post flags + classify each user into fate group |
| `f0000006` | exposed-user funnel table (in_panel / post_only / pre_only / no_activity) |
| `h0000008` | unexposed-user funnel table |
| `j0000010` | waterfall bar chart (labeled → has activity → has pre → has post → panel) |
| `l0000012` | side-by-side exposed vs unexposed dropout rates + differential attrition flag |
| `n0000014` | per-cycle breakdown table |
| `p0000016` | activity volume for dropped users (pre_n, post_n by fate) |
| `r0000018` | root-cause summary printout |

### NB06a `06a_stratified_pre_exposure.ipynb`
| cell_id | Content |
|---------|---------|
| `a1b2c3d4` | markdown header |
| `b2c3d4e5` | imports + DATA_DIR (`processed/gradadmissions`) |
| `d4e5f6a7` | pre-period span computation + exposed user span distribution |
| `e5f6a7b8` | median span by month of first pre-activity |
| `a7b8c9d0` | `run_psm_did()` helper (PSM + DiD, returns ATT/CI/SMD/n_matched) |
| `b8c9d0e1` | `run_dose_response()` helper |
| `d0e1f2a3` | run all three sample versions (full, ≥7d, ≥14d) |
| `f2a3b4c5` | summary table (ATT, p, CI, SMD, dose coef across all versions) |
| `b4c5d6e7` | forest plot: ATT + 95% CI across three pre-period restrictions |

### NB05 `05_collect_community_breadth.ipynb`
| cell_id | Content |
|---------|---------|
| `b2000002` | imports + path setup |
| `d2000004` | load v2 panel users |
| `f2000006` | load checkpoint + old cache |
| `h2000008` | fetch_breadth() API function |
| `i2000009` | main API fetch loop |
| `n2000014` | save user_community_breadth.parquet |

### NB06 `06_did_analysis.ipynb`
| cell_id | Content |
|---------|---------|
| `b3000002` | imports + path setup |
| `d3000004` | load panel + merge breadth |
| `f3000006` | CALIPER, MATCH_FEATURES, psm_match() def |
| `h3000008` | PSM per cycle → long_rows |
| `j3000010` | pre-period balance check + visual parallel trends |
| `62ccb4a5` | §4a markdown header (formal parallel trends test) |
| `7d7c7285` | §4a formal pre-trend test: loads post_level_scores.parquet, regresses mh_score ~ week_number × exposed (pre-period only, matched users); saves parallel_trends_test_v2.csv + fig_parallel_trends_pretest_{SUBREDDIT}.png |
| `l3000012` | run_did() def + RQ1 user-level DiD |
| `f28af885` | run_did_dim() def + per-dimension DiD |
| `n3000014` | RQ2 breadth moderation |
| `p3000016` | ATT coefficient plot |
| `50f27f84` | post-level DiD setup |
| `a881695e` | run_post_level_did() + results |
| `5d371355` | dose-response analysis |
| `1b89be1d` | robustness checks header (markdown) |
| `e27bc930` | robustness checks: (a) placebo, (b) tight post-window Dec–Jan, (c) BART confidence threshold sensitivity at 0.4 / 0.6 |
| `5413ee45` | summary comparison table header (markdown) |
| `e4c0bcfe` | summary comparison table + save did_summary.csv |

### NB07 `07_comparison_analysis.ipynb`
| cell_id | Content |
|---------|---------|
| `c7000002` | imports + paths — loads all 3 subreddit subdirs |
| `c7000004` | panel and anchor post summary per subreddit per cycle |
| `c7000006` | anchor post characteristics comparison (boxplots, 3 subreddits) |
| `c7000008` | PSM + DiD helper functions (psm_match, build_long_df, run_did) |
| `c7000010` | run PSM + DiD for all subreddits |
| `c7000012` | ATT coefficient comparison forest plot (3 subreddits) |
| `c7000014` | pre/post mh_score distribution boxplots (3 rows × 2 cols) |
| `c7000016` | Pairwise Z-test across all subreddit combinations × specs (9 tests) |
| `c7000018` | summary table + save comparison_summary.parquet |
| `0909343c` | §9 markdown header (combined pooled DiD across all 3 subreddits) |
| `9730fcc7` | §9 combined pooled DiD: stacks all_long_dfs from §4, runs OLS with subreddit + cycle FE; saves combined_pooled_did.csv |

### NB08 `08_alt_analysis.ipynb`
| cell_id | Content |
|---------|---------|
| `b0000002` | imports + path setup (POSTS_PATH, COMMENTS_PATH) |
| `b0000004` | load exposure_labels, anchor_posts, breadth |
| `b0000005` | scan raw comments for first_anchor_comment timestamps |
| `b0000007` | load SVM classifiers |
| `b0000009` | assign_window() + process_file() + scan raw JSONL |
| `b0000010` | build corpus DataFrame + score all texts |
| `b0000012` | aggregate to panel_scores_alt |
| `b0000013` | save panel_scores_alt.parquet |
| `b0000015` | psm_match() + run PSM per cycle |
| `b0000017` | parallel trends plot |
| `b0000019` | run_did() + DiD per cycle + pooled |
| `b0000021` | CausalImpact setup + pandas patch |
| `b0000022` | CausalImpact run per cycle + plots |
| `b0000024` | coverage comparison summary |

### NB09 `09_nli_anchor_validation.ipynb`
NB09 was rewritten (2026-04-22) — it no longer runs BART inference. It reads BART fields already present in `anchor_posts.parquet` from NB03 and produces summary stats and figures.
| cell_id | Content |
|---------|---------|
| `a9000001` | markdown header |
| `b9000002` | config: `SUBREDDITS = ['gradadmissions', 'mscs', 'mba']` |
| `c9000003` | imports + ROOT / DATA / FIG_DIR path setup |
| `e9000005` | load anchor_posts.parquet from all 3 subreddits → `anchors` concat |
| `h9000008` | (1) summary stats: per-subreddit counts, label distribution, cycle breakdown, score stats |
| `i9000009` | (2) BART score distribution histogram + SVM mh_score vs BART top-neg scatter (coloured by label) |
| `j9000010` | (3) high-confidence examples by BART label (top 5 per label) |
| `k9000011` | (4) per-cycle anchor counts + BART score by subreddit × label + SVM score summary |

---

## Key Variable Names at End of Each Notebook

| Notebook | Key variables in memory at end |
|----------|-------------------------------|
| NB03 | `anchor_posts` (df), `exposure_df` (df), `anchor_ids_by_cycle` (dict) |
| NB04 | `panel` (df, user-level), `corpus` (df, post-level), `dose_df` (df) |
| NB05 | `combined` (df, breadth data) |
| NB06 | `panel` (df, merged with breadth), `matched_by_cycle` (dict cycle→set of authors), `long_df` (long format for DiD), `post_level` (df, post-level data for matched users) |
| NB08 | `first_anchor_comment` (dict author→datetime), `corpus` (df, scored raw text), `panel` (df, alt panel), `matched_by_cycle` (dict cycle→DataFrame), `all_matched` (df), `weekly` (df, weekly aggregated for CausalImpact) |

---

## Known Gotchas

0. **SUBREDDIT config variable**: NB01, NB03, NB04, NB05, NB06, NB08 all have `SUBREDDIT = 'gradadmissions'` at the top of their first code cell. Supported values: `'gradadmissions'`, `'mscs'`, `'mba'`. All output paths automatically route to `data/processed/{SUBREDDIT}/`. Use `run_pipeline.py` to inject this automatically — do not edit notebooks manually.

1. **Raw vs cleaned field names**:
   - Raw: `created_utc` (int), `selftext` / `body`, `link_id` = `"t3_<id>"`
   - Cleaned: `created_dt` (ISO str), `clean_text`, `post_id` (stripped)
   - NB03, NB04, NB08 all use cleaned files (`posts_clean.jsonl` / `comments_clean.jsonl` from `DATA`)

2. **`mh_score` column naming inconsistency**:
   - `panel_scores.parquet` → `pre_mh_score`, `post_mh_score`
   - `post_level_scores.parquet` → column is `mean_mh_score` (not `mh_score`)
   - NB06 reshapes to `mh_score` in long format at runtime

3. **Anchor selection is BART-based, not SVM-threshold-based**: NB03 uses keyword filter + `bart_is_negative==True`. SVM scores (`anx_score` etc.) are still in `anchor_posts.parquet` for characterisation only.

4. **NB06 PSM uses StandardScaler before logistic regression; NB08 does not.** They are not directly comparable implementations.

5. **`causalimpact 0.2.6` + pandas 2.x**: NB08 patches at runtime:
   ```python
   import pandas.core.dtypes.common as _pdc
   if not hasattr(_pdc, 'is_datetime_or_timedelta_dtype'):
       _pdc.is_datetime_or_timedelta_dtype = (
           lambda x: pd.api.types.is_datetime64_any_dtype(x) or
                     pd.api.types.is_timedelta64_dtype(x)
       )
   ```
   Also: pass integer index positions `[0, n_pre-1]` / `[n_pre, n_total-1]` to `CausalImpact()`, not datetime objects.

6. **`posts_clean.jsonl` and `comments_clean.jsonl` are NOT in git** — they must be generated by NB01. Any attempt to `Read` them will fail.

7. **`user_community_breadth.parquet` now has 100% coverage** of panel users. Failed API fetches or deleted accounts are assigned `community_breadth = 0`. `community_breadth_log` is always included in PSM.

8. **`panel_scores_alt.parquet` also has 100% coverage** for community breadth.

9. **`dose_exposure.parquet` `cycle` column is float64**, not int — it was created with a float merge key. Use `.astype(int)` if needed.

10. **NB06 post-level DiD uses 22,355 rows** — filtered from the full 147,569-row `post_level_scores.parquet` down to matched authors only.

11. **`panel_scores.parquet` does NOT include `exposure_prob`** — NB06 reads it from `exposure_labels.parquet` via a separate merge. `exposure_intensity` (float 0–1, BART top-neg score) IS present in `panel_scores.parquet`.

12. **NB06 MBA kernel (OOM risk)** — MBA NB06 previously crashed with `DeadKernelError` under RAM pressure (~1Gi available). Fixed 2026-04-23 by running standalone. If it recurs, close browser tabs and re-run with `run_pipeline.py --subreddits mba --start-from 06`.

---

## What Is and Isn't in Git

**In git**: all `*.parquet`, `models/clf_*.joblib`, `figures/*.png`, all notebooks, all docs, `data/mba/*.jsonl.gz`.

**NOT in git** (too large or regenerable):
- `r_gradadmissions_posts.jsonl` / `r_gradadmissions_comments.jsonl` (raw, repo root)
- `data/raw/r_MSCS_posts.jsonl` / `data/raw/r_MSCS_comments.jsonl` (raw, data/raw/)
- `data/processed/gradadmissions/posts_clean.jsonl` / `comments_clean.jsonl` (run NB01)
- `data/processed/mscs/posts_clean.jsonl` / `comments_clean.jsonl` (run NB01 with SUBREDDIT='mscs')
- `data/processed/mba/posts_clean.jsonl` / `comments_clean.jsonl` (decompress from `data/mba/*.gz`)

---

## Current Status & Open Issues

- **Full BART pipeline + all results complete (2026-04-23)**: NB03 BART anchor selection ran for all 3 subreddits (GA=2,010, MSCS=144, MBA=885 anchors). NB04/NB05/NB06/NB08 completed for all subreddits. MBA NB06 OOM crash fixed. All `did_summary.csv`, `parallel_trends_test_v2.csv`, and figures are current.
- **Headline finding — combined pooled DiD (NB07 §9)**: ATT = **+0.0045**, 95% CI [+0.0007, +0.0083], p = **0.021*** across 6,012 matched users, 12,416 obs × 3 cycles × 3 subreddits. This is the primary cross-community result (subreddit + cycle FE). See `data/processed/combined_pooled_did.csv`.
- **Per-subreddit NB06 results (pooled, cycle FE)**: GA +0.0037 p=0.27 (n.s.), MSCS +0.0065 p=0.27 (n.s.), MBA +0.0066 p=0.017*. Individual subreddits are underpowered; significance achieved in pooled analysis.
- **Formal parallel trends test (NB06 §4a, 2026-04-23)**: All subreddits, all cycles, and pooled spec confirm n.s. interaction of week_number × exposed in the pre-period (p > 0.05 throughout). See `parallel_trends_test_v2.csv` and `fig_parallel_trends_pretest_{SUBREDDIT}.png`. Parallel trends assumption holds.
- **NB09 (anchor characterisation, 2026-04-23)**: Reads BART fields from `anchor_posts.parquet` directly (no inference). Ran as final pipeline step.
- **NB04a/05a/06a (diagnostic notebooks, gradadmissions only, 2026-04-22)**: NB04a checks differential attrition, pre-period quality, and anchor timing. NB05a produces a pipeline funnel showing 967/2,871 exposed users enter the panel (33.7%); dominant dropout is no activity at all (863, 30%), then missing pre-period baseline (737, 25.7%), then missing post-period (304, 10.6%). NB06a tests PSM+DiD sensitivity to pre-period length restrictions (full, ≥7d, ≥14d).
- **`run_pipeline.py`**: use this to run the pipeline. `PRE_CLEANED = {'mba'}` skips NB01 for MBA.
- **NB06 GPS weighting is nearly degenerate**: stabilized weights have mean ≈ 1.0 (max ≈ 2.8 for MBA). GPS-WLS results are in `did_summary.csv` but are not the primary spec. Does not affect primary DiD.
- **Pairwise Z-test in NB07** (cell `c7000016`): all 3 pairs (GA–MSCS, GA–MBA, MSCS–MBA) × 3 specs are not significant (min p = 0.25). No subreddit is statistically distinguishable from another — heterogeneity is likely due to power, not true effect differences.
- **Breadth moderation (RQ2)**: All subreddits n.s. (MBA RQ2 pooled: +0.0004, p=0.85). `community_breadth_log` included in PSM for all subreddits (100% coverage).
- **Data time range**: all three subreddits cover Aug 2022–Jul 2025 (three admission cycles). Raw 2022-cycle JSONLs live under `data/raw/r_{gradadmissions,MSCS}_2022_{posts,comments}.jsonl`; NB01 concatenates them. MBA's raw already included 2022.
- **3-cycle refactor (2026-04-21)**: NB03/NB04 CYCLES dict, NB05 `AFTER_DATE=2022-08-01`, NB06/NB07/NB08 loops all derive cycles dynamically from data (no hardcoded `[1, 2]`).

## Running the Pipeline

### Single command (both subreddits)
```bash
~/venvs/jupyter/bin/python3 run_pipeline.py
```
Runs NB01 → NB03 → NB04 → NB05 → NB06 → NB08 for each subreddit in sequence, then NB07 comparison.

### Options
```bash
# One subreddit only
~/venvs/jupyter/bin/python3 run_pipeline.py --subreddits gradadmissions

# Resume from a specific step (skips earlier notebooks)
~/venvs/jupyter/bin/python3 run_pipeline.py --start-from 04

# Skip the comparison notebook (NB07)
~/venvs/jupyter/bin/python3 run_pipeline.py --skip-comparison

# Skip the anchor characterisation notebook (NB09)
~/venvs/jupyter/bin/python3 run_pipeline.py --skip-validation
```

### How it works
`run_pipeline.py` injects the correct `SUBREDDIT` value into each notebook's config cell, executes via `nbconvert`, and writes output to a temp directory (source notebooks are never modified). All data outputs land in `data/processed/{subreddit}/`.

---

## How to Update This File

After any session where you change code, data schemas, results, or add notebooks/figures:
1. Update the relevant section(s) above in place.
2. Update the **Last updated** date at the top.
3. Commit with: `git add docs/LLM_CONTEXT.md && git commit -m "Update LLM_CONTEXT: <what changed>"`
