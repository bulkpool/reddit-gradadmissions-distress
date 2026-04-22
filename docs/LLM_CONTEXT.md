# LLM Codebase Context

**Purpose**: Dense, authoritative snapshot for AI assistants. Read this file before touching anything else in the repo. After making any changes, update the relevant sections here so future sessions start fresh.

**Last updated**: 2026-04-21 (3-cycle update — 2022 admission cycle added for all three subreddits)

---

## Project in One Paragraph

Causal inference study on **r/GradAdmissions, r/MSCS, and r/MBA**. All three subreddits now cover **Aug 2022–Jul 2025 (three admission cycles)**. RQ1: does exposure to high-distress anchor posts increase users' mental health distress scores in the following Dec–May decision season? RQ2: does cross-Reddit community breadth moderate this effect? Method: SVM-based distress scoring → anchor post identification → propensity-score matched DiD. The pipeline is parameterized by `SUBREDDIT` (set at top of each notebook). Orchestrated via `run_pipeline.py`.

**Cycle numbering (chronological):** `cycle=1` → 2022 cycle (anchor Sep–Nov 2022, post Dec 2022–May 2023). `cycle=2` → 2023 cycle. `cycle=3` → 2024 cycle. NB06/NB07/NB08 now derive the cycle list dynamically from each panel (no hardcoded `[1, 2]`).

**Headline results (NB07, 2026-04-21 — from the earlier 2-cycle run; re-run pipeline to refresh with 3 cycles)**:
- **r/MBA pooled DiD = +0.0081, 95% CI [+0.0024, +0.0138], p = 0.0054** (significant at p < 0.01). Cycle 1 also significant in the 2-cycle run: +0.0088, p = 0.0205.
- **r/GradAdmissions pooled DiD = +0.0043, CI [-0.0030, +0.0115], p = 0.25** (not significant).
- **r/MSCS pooled DiD = +0.0062, CI [-0.0063, +0.0188], p = 0.33** (not significant). Cycle 2 marginal: +0.0133, p = 0.096.
- All three subreddits show positive point estimates; only r/MBA crosses statistical significance. r/MBA has the largest matched panel (1,425 pairs pooled vs 928 for GA, 191 for MSCS).

---

## Environment

- Python venv: `~/venvs/jupyter/bin/` — always use this, never system Python
- Run notebooks: `~/venvs/jupyter/bin/jupyter nbconvert --to notebook --execute <nb>`
- Key packages: `scikit-learn`, `statsmodels`, `joblib`, `pandas` (2.x), `numpy`, `matplotlib`, `causalimpact==0.2.6`

---

## Repo Layout

```
/                                        ← repo root
├── r_gradadmissions_posts.jsonl         ← NOT in git — raw posts (GA, 2023 + 2024 cycles)
├── r_gradadmissions_comments.jsonl      ← NOT in git — raw comments (GA, 2023 + 2024 cycles)
├── run_pipeline.py                      ← orchestrator: injects SUBREDDIT, runs NB01→NB08+NB07
├── notebooks/
│   ├── 00_exploratory_topic_sentiment.ipynb  (EDA only)
│   ├── 01_clean_corpus.ipynb             ← SUBREDDIT config at top
│   ├── 02_train_classifiers.ipynb
│   ├── 03_exposure_labels_v2.ipynb       ← SUBREDDIT config at top
│   ├── 04_panel_scores.ipynb             ← SUBREDDIT config at top
│   ├── 05_collect_community_breadth.ipynb ← SUBREDDIT config at top
│   ├── 06_did_analysis_v2.ipynb          ← MAIN RESULTS; SUBREDDIT config at top
│   ├── 07_comparison_analysis.ipynb      ← 3-subreddit comparison (NB07)
│   └── 08_alt_analysis.ipynb             ← Sep–Nov pre-period + CausalImpact; SUBREDDIT config
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
│   └── processed_v2/
│       ├── gradadmissions/               ← GradAdmissions pipeline outputs
│       │   ├── anchor_posts_v2.parquet
│       │   ├── exposure_labels_v2.parquet
│       │   ├── panel_scores_v2.parquet
│       │   ├── post_level_scores_v2.parquet
│       │   ├── dose_exposure_v2.parquet
│       │   ├── panel_scores_alt.parquet
│       │   └── user_community_breadth_v2.parquet
│       ├── mscs/                         ← MSCS pipeline outputs (same parquet set)
│       ├── mba/                          ← MBA pipeline outputs (same parquet set)
│       │   ├── posts_clean.jsonl         ← NOT in git — decompressed from data/mba/
│       │   └── comments_clean.jsonl      ← NOT in git — decompressed from data/mba/
│       └── comparison_summary.parquet    ← NB07 output (all key metrics, 3 subreddits)
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

### Cleaned JSONL (`data/processed_v2/`, NOT in git — run NB01 to generate)

**`posts_clean.jsonl`**: `id, author, created_dt (ISO str), clean_text, score, num_comments`

**`comments_clean.jsonl`**: `id, author, created_dt (ISO str), post_id (= link_id minus "t3_"), clean_text, score`

> Note: cleaned files use `created_dt` (string) and `clean_text`, NOT `created_utc` and `selftext`/`body`. Do not confuse with raw fields.

### Parquet Files (`data/processed_v2/gradadmissions/` or `data/processed_v2/mscs/`)

> All parquet outputs are now in subreddit-specific subdirectories. `SUBREDDIT` at top of each notebook controls which dir is used.
> Row counts below are from the earlier 2-cycle run — re-run the pipeline to refresh with the 2022 cycle included.

**`exposure_labels_v2.parquet`** — GA: 22,269 rows (exposed=2,784) | MSCS: 4,299 (exposed=374) | MBA: 25,139 (exposed=3,762) [2-cycle; row counts will grow with 2022 cycle]
| Column | Type | Notes |
|--------|------|-------|
| `author` | str | Reddit username |
| `exposed` | bool | — |
| `cycle` | int64 | 1, 2, or 3 (1 = 2022, 2 = 2023, 3 = 2024) |

**`anchor_posts_v2.parquet`** — GA: 997 rows | MSCS: 100 | MBA: 471 [2-cycle]
| Column | Type | Notes |
|--------|------|-------|
| `id` | str | Reddit post ID |
| `author` | str | — |
| `created_dt` | datetime64[ns, UTC] | — |
| `cycle` | float64 | 1.0, 2.0, or 3.0 |
| `clean_text` | str | — |
| `anx_score` | float64 | — |
| `dep_score` | float64 | — |
| `str_score` | float64 | — |
| `mean_mh_score` | float64 | — |
| `score` | int64 | Reddit upvotes |
| `num_comments` | int64 | — |

**`panel_scores_v2.parquet`** — GA: 7,770 rows (7,578 users) | MSCS: 1,951 (1,894) | MBA: 8,261 (7,619) [2-cycle]
| Column | Type | Notes |
|--------|------|-------|
| `author` | str | — |
| `cycle` | int64 | 1, 2, or 3 |
| `exposed` | bool | True=322 / False=1,101 |
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

**`post_level_scores_v2.parquet`** — 147,569 rows (all matched-panel users, pre+post) [2-cycle]
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

**`dose_exposure_v2.parquet`** — 2,300 rows [2-cycle]
| Column | Type | Notes |
|--------|------|-------|
| `author` | str | 2,257 unique |
| `cycle` | float64 | 1.0, 2.0, or 3.0 |
| `n_anchor_comments` | int64 | [1, 49] |
| `log1p_n_anchor` | float64 | [0.693, 3.912] |

**`user_community_breadth_v2.parquet`** — 100% coverage of panel users (e.g., 7,770 rows for GA)
| Column | Type | Notes |
|--------|------|-------|
| `author` | str | — |
| `community_breadth` | float64 | [0, 100], missing/errors imputed as 0 |
| `community_breadth_log` | float64 | log1p(breadth) |
| `subreddits_json` | str | JSON list of subreddit names |
| `status` | str | `"ok"`, `"http_404"`, etc. |

> Coverage is now 100%. Missing accounts or errors default to `community_breadth = 0` so that PSM in NB06 can utilize `community_breadth_log` without dropping rows.

---

## Anchor Post Identification (NB03)

**Threshold**: `MH_THRESHOLD = 0.5` (per-dimension, OR logic — post qualifies if ANY one dimension > 0.5)

**Keyword list** (regex, case-insensitive OR):
```python
NEGATIVE_KEYWORDS = [
    r'\breject(?:ed|ion)\b',        r'\bdeclin(?:ed|ing)\b',
    r'\bwaitlist(?:ed)?\b',         r'\bfunding\s+(?:lost|cut|removed|denied|gap|issue)\b',
    r'\bno\s+funding\b',            r'\bstipend\b',
    r'\bwithdrew?\s+(?:offer|admission)\b',  r'\bacceptance\s+rate\b',
    r'\bno\s+(?:offer|response|interview)\b', r'\bsilence\s+from\b',
    r'\bnot\s+(?:accepted|admitted|selected)\b', r'\bgave\s+up\b',
    r'\bmental\s+health\b',         r'\banxi(?:ous|ety)\b',
    r'\bdepress(?:ed|ing|ion)\b',   r'\bstress(?:ed|ful)?\b',
    r'\boverwhelm(?:ed|ing)\b',     r'\bscared\b',
    r'\bworr(?:ied|ying)\b',        r'\bfalling\s+apart\b',
    r'\bbreaking\s+down\b',         r'\bcan(?:\'t|not)\s+(?:take|handle|cope)\b',
    r'\bno\s+chance\b',             r'\bnot\s+good\s+enough\b',
    r'\bgave\s+up\b',               r'\bregret\b',
    r'\bfailed\b',                  r'\bimposter\b',
]
```

**Filter logic**: post must match keyword AND (`anx_score > 0.5` OR `dep_score > 0.5` OR `str_score > 0.5`).

> Docs and old comments say threshold=0.45 — **the actual code uses 0.5**. The parquet shows `mean_mh_score` as low as 0.39, which is possible when only one dimension fires above 0.5.

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
| `fig_att_comparison.png` | NB07 | Forest plot: ATT coefficients for all 3 subreddits side by side |
| `fig_mhscore_distributions.png` | NB07 | Pre/post mh_score boxplots: all 3 subreddits |
| `fig_anchor_comparison.png` | NB07 | Anchor post characteristics: all 3 subreddits |
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

### NB03 `03_exposure_labels_v2.ipynb`
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
| `ff080f5a` | save post_level_scores_v2.parquet |
| `71c81c1c` | dose-response: count anchor comments |
| `m1000013` | aggregate to user-level pre/post |
| `r1000018` | save panel_scores_v2.parquet |

### NB05 `05_collect_community_breadth.ipynb`
| cell_id | Content |
|---------|---------|
| `b2000002` | imports + path setup |
| `d2000004` | load v2 panel users |
| `f2000006` | load checkpoint + old cache |
| `h2000008` | fetch_breadth() API function |
| `i2000009` | main API fetch loop |
| `n2000014` | save user_community_breadth_v2.parquet |

### NB06 `06_did_analysis_v2.ipynb`
| cell_id | Content |
|---------|---------|
| `b3000002` | imports + path setup |
| `d3000004` | load panel + merge breadth |
| `f3000006` | CALIPER, MATCH_FEATURES, psm_match() def |
| `h3000008` | PSM per cycle → long_rows |
| `j3000010` | pre-period balance check |
| `l3000012` | run_did() def + RQ1 user-level DiD |
| `f28af885` | run_did_dim() def + per-dimension DiD |
| `n3000014` | RQ2 breadth moderation |
| `p3000016` | ATT coefficient plot |
| `50f27f84` | post-level DiD setup |
| `a881695e` | run_post_level_did() + results |
| `5d371355` | dose-response analysis |
| `e27bc930` | placebo test |

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

0. **SUBREDDIT config variable**: NB01, NB03, NB04, NB05, NB06, NB08 all have `SUBREDDIT = 'gradadmissions'` at the top of their first code cell. Supported values: `'gradadmissions'`, `'mscs'`, `'mba'`. All output paths automatically route to `data/processed_v2/{SUBREDDIT}/`. Use `run_pipeline.py` to inject this automatically — do not edit notebooks manually.

1. **Raw vs cleaned field names**:
   - Raw: `created_utc` (int), `selftext` / `body`, `link_id` = `"t3_<id>"`
   - Cleaned: `created_dt` (ISO str), `clean_text`, `post_id` (stripped)
   - NB03, NB04, NB08 all use cleaned files (`posts_clean.jsonl` / `comments_clean.jsonl` from `DATA_V2`)

2. **`mh_score` column naming inconsistency**:
   - `panel_scores_v2.parquet` → `pre_mh_score`, `post_mh_score`
   - `post_level_scores_v2.parquet` → column is `mean_mh_score` (not `mh_score`)
   - NB06 reshapes to `mh_score` in long format at runtime

3. **Anchor threshold is 0.5 per dimension (OR), not 0.45 on mean**: some docs say 0.45 — the code is authoritative at 0.5.

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

7. **`user_community_breadth_v2.parquet` now has 100% coverage** of panel users. Failed API fetches or deleted accounts are assigned `community_breadth = 0`. `community_breadth_log` is always included in PSM.

8. **`panel_scores_alt.parquet` also has 100% coverage** for community breadth.

9. **`dose_exposure_v2.parquet` `cycle` column is float64**, not int — it was created with a float merge key. Use `.astype(int)` if needed.

10. **NB06 post-level DiD uses 22,355 rows** — filtered from the full 147,569-row `post_level_scores_v2.parquet` down to matched authors only.

11. **`panel_scores_v2.parquet` includes `exposure_intensity` (int, 0–3) but NOT `exposure_prob`**. NB06 reads `exposure_prob` via `row.get('exposure_prob', 0.0)`, which silently returns 0.0. GPS weighting is therefore degenerate (no-op) for all subreddits in the current data.

12. **NB06 kernel dies (OOM) on low-memory systems** — Linux OOM killer fires when swap is nearly full (seen with Brave browser open). Close browser tabs before running NB06. The Python logic itself is fine; the kernel process launch fails due to swap exhaustion.

---

## What Is and Isn't in Git

**In git**: all `*.parquet`, `models/clf_*.joblib`, `figures/*.png`, all notebooks, all docs, `data/mba/*.jsonl.gz`.

**NOT in git** (too large or regenerable):
- `r_gradadmissions_posts.jsonl` / `r_gradadmissions_comments.jsonl` (raw, repo root)
- `data/raw/r_MSCS_posts.jsonl` / `data/raw/r_MSCS_comments.jsonl` (raw, data/raw/)
- `data/processed_v2/gradadmissions/posts_clean.jsonl` / `comments_clean.jsonl` (run NB01)
- `data/processed_v2/mscs/posts_clean.jsonl` / `comments_clean.jsonl` (run NB01 with SUBREDDIT='mscs')
- `data/processed_v2/mba/posts_clean.jsonl` / `comments_clean.jsonl` (decompress from `data/mba/*.gz`)

---

## Current Status & Open Issues

- **NB01–NB08 (r/GradAdmissions, r/MSCS, r/MBA)**: all complete. All three subreddits have full parquet outputs.
- **NB07 (comparison)**: last run 2026-04-21 15:31. Produces `fig_att_comparison.png`, `fig_mhscore_distributions.png`, `fig_anchor_comparison.png`, `comparison_summary.parquet`.
- **`run_pipeline.py`**: use this to run the pipeline. `PRE_CLEANED = {'mba'}` skips NB01 for MBA.
- **Primary finding (r/MBA)**: pooled DiD = +0.0081, p = 0.0054 ** — statistically significant. Driven by Cycle 1 (+0.0088, p = 0.02) more than Cycle 2 (+0.0071, p = 0.12). This is the first subreddit in the study with a significant result.
- **r/GradAdmissions and r/MSCS remain null** in the primary NB06 analysis (pooled p = 0.25 and 0.33 respectively). Positive point estimates, underpowered.
- **Why MBA hits significance**: largest matched panel (1,425 pairs pooled vs 928 GA, 191 MSCS). Effect size is actually similar to GA (+0.008 vs +0.004), but SE is tighter.
- **NB06 GPS weighting is a no-op for all subreddits** — `exposure_prob` column missing from panel, `.get()` falls back to 0.0. This is a latent bug in the pipeline that does not affect the primary DiD results but makes the GPS-weighted robustness check degenerate.
- **Pairwise Z-test in NB07** (cell `c7000016`): compares all 3 pairs (GA–MSCS, GA–MBA, MSCS–MBA) × 3 specs (Cycle 1, Cycle 2, Pooled). All pairwise differences are **not significant** (min p = 0.25 for MSCS vs MBA Cycle 1). MBA's pooled ATT (+0.0081) is not statistically distinguishable from GA's (+0.0043, pairwise p = 0.42) despite MBA being internally significant — MBA just has enough N to detect a non-zero effect, while GA does not.
- **Breadth moderation (RQ2)**: With the NB05 fix, `community_breadth_log` is now reliably included in PSM and used as a moderator. Pipeline needs to be re-run to observe updated RQ2 results.
- **Data time range**: all three subreddits now cover Aug 2022–Jul 2025 (three admission cycles). Raw 2022-cycle JSONLs live under `data/raw/r_{gradadmissions,MSCS}_2022_{posts,comments}.jsonl`; NB01 concatenates them with the main file. MBA's raw already included 2022.
- **3-cycle refactor (2026-04-21)**: NB03/NB04 CYCLES dict, NB05 `AFTER_DATE=2022-08-01`, NB06 ATT/parallel-trends/post-level/dose/placebo loops, NB07 spec_keys, and NB08 CYCLES + CausalImpact loop all updated to derive cycles dynamically from data instead of hardcoding `[1, 2]`. Re-run the full pipeline to populate parquets with 2022-cycle rows before reading the results docs.

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

# Skip the comparison notebook
~/venvs/jupyter/bin/python3 run_pipeline.py --skip-comparison
```

### How it works
`run_pipeline.py` injects the correct `SUBREDDIT` value into each notebook's config cell, executes via `nbconvert`, and writes output to a temp directory (source notebooks are never modified). All data outputs land in `data/processed_v2/{subreddit}/`.

---

## How to Update This File

After any session where you change code, data schemas, results, or add notebooks/figures:
1. Update the relevant section(s) above in place.
2. Update the **Last updated** date at the top.
3. Commit with: `git add docs/LLM_CONTEXT.md && git commit -m "Update LLM_CONTEXT: <what changed>"`
