# LLM Codebase Context

**Purpose**: Dense, authoritative snapshot for AI assistants. Read this file before touching anything else in the repo. After making any changes, update the relevant sections here so future sessions start fresh.

**Last updated**: 2026-04-07

---

## Project in One Paragraph

Causal inference study on r/GradAdmissions (Aug 2023–Jul 2025). RQ1: does exposure to high-distress anchor posts increase users' mental health distress scores in the following Dec–May decision season? RQ2: does cross-Reddit community breadth moderate this effect? Method: SVM-based distress scoring → anchor post identification → propensity-score matched DiD. Current results: small positive effects (~+0.007–0.010) that are directionally consistent but not statistically significant at p < 0.05 in the primary analysis (NB06). Alternative analysis (NB08) with Sep–Nov pre-period yields borderline pooled DiD p = 0.067 and CausalImpact +2.1–2.7%.

---

## Environment

- Python venv: `~/venvs/jupyter/bin/` — always use this, never system Python
- Run notebooks: `~/venvs/jupyter/bin/jupyter nbconvert --to notebook --execute <nb>`
- Key packages: `scikit-learn`, `statsmodels`, `joblib`, `pandas` (2.x), `numpy`, `matplotlib`, `causalimpact==0.2.6`

---

## Repo Layout

```
/                                        ← repo root
├── r_gradadmissions_posts.jsonl         ← NOT in git — raw posts
├── r_gradadmissions_comments.jsonl      ← NOT in git — raw comments
├── r_gradadmissions_posts.cleaned.jsonl ← NOT in git — pre-cleaned (old pipeline)
├── r_gradadmissions_comments.cleaned.jsonl ← NOT in git
├── notebooks/
│   ├── 00_exploratory_topic_sentiment.ipynb  (EDA only)
│   ├── 01_clean_corpus.ipynb
│   ├── 02_train_classifiers.ipynb
│   ├── 03_exposure_labels_v2.ipynb
│   ├── 04_panel_scores.ipynb
│   ├── 05_collect_community_breadth.ipynb
│   ├── 06_did_analysis_v2.ipynb          ← MAIN RESULTS
│   ├── 07_did_analysis_vader_baseline.ipynb  (optional, uses old data/processed/)
│   └── 08_alt_analysis.ipynb             ← Sep–Nov pre-period + CausalImpact
├── data/processed_v2/                    ← all intermediate outputs (in git where small)
├── models/                               ← clf_anxiety/depression/stress.joblib
├── figures/                              ← all PNG outputs
└── docs/
    ├── LLM_CONTEXT.md                    ← THIS FILE — update after every change
    ├── AGENTS.md                         ← agent rules — read before making changes
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

### Parquet Files (`data/processed_v2/`)

**`exposure_labels_v2.parquet`** — 21,730 rows, 20,932 unique authors
| Column | Type | Notes |
|--------|------|-------|
| `author` | str | Reddit username |
| `exposed` | bool | True=2,033 / False=19,697 |
| `cycle` | int64 | 1 or 2 |

**`anchor_posts_v2.parquet`** — 597 rows
| Column | Type | Notes |
|--------|------|-------|
| `id` | str | Reddit post ID |
| `author` | str | 548 unique authors |
| `created_dt` | datetime64[ns, UTC] | — |
| `cycle` | float64 | 1.0 or 2.0 |
| `clean_text` | str | — |
| `anx_score` | float64 | [0.325, 0.756] |
| `dep_score` | float64 | [0.273, 0.763] |
| `str_score` | float64 | [0.387, 0.765] |
| `mean_mh_score` | float64 | [0.392, 0.748] |
| `score` | int64 | Reddit upvotes [0, 755] |
| `num_comments` | int64 | [0, 155] |

**`panel_scores_v2.parquet`** — 1,423 rows, 1,368 unique authors (August pre-period)
| Column | Type | Notes |
|--------|------|-------|
| `author` | str | — |
| `cycle` | int64 | 1 or 2 |
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
| `cycle` | int64 | 1 or 2 |
| `pre_mh_score` | float64 | [0.138, 0.750] |
| `pre_n_posts` | int64 | [1, 87] |
| `post_mh_score` | float64 | [0.147, 0.778] |
| `post_n_posts` | int64 | [1, 1556] |
| `community_breadth` | float64 | [0, 100], **6,528 nulls** |
| `community_breadth_log` | float64 | log1p(breadth), **6,528 nulls** |

**`post_level_scores_v2.parquet`** — 147,569 rows (all matched-panel users, pre+post)
| Column | Type | Notes |
|--------|------|-------|
| `author` | str | 10,693 unique |
| `cycle` | int64 | 1 or 2 |
| `window` | str | `"pre"` or `"post"` (NOT period 0/1) |
| `created_dt` | datetime64[ns, UTC] | — |
| `anx_score` | float64 | [0.065, 0.988] |
| `dep_score` | float64 | [0.056, 0.866] |
| `str_score` | float64 | [0.056, 0.940] |
| `mean_mh_score` | float64 | [0.070, 0.830] — note: col is `mean_mh_score` not `mh_score` |

> NB06 post-level DiD filters this to matched authors only → 22,355 observations.

**`dose_exposure_v2.parquet`** — 2,300 rows
| Column | Type | Notes |
|--------|------|-------|
| `author` | str | 2,257 unique |
| `cycle` | float64 | 1.0 or 2.0 |
| `n_anchor_comments` | int64 | [1, 49] |
| `log1p_n_anchor` | float64 | [0.693, 3.912] |

**`user_community_breadth_v2.parquet`** — 1,689 rows
| Column | Type | Notes |
|--------|------|-------|
| `author` | str | — |
| `community_breadth` | float64 | [0, 100] |
| `community_breadth_log` | float64 | log1p(breadth), [0, 4.615] |
| `subreddits_json` | str | JSON list of subreddit names |
| `status` | str | always `"ok"` in current data |

> Only 1,689 users have breadth data — less than the 1,368-user panel. NB06 merges on left join so unmatched users get null breadth.

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
| `fig_att_coef_v2.png` | NB06 | ATT coefficient plot (user-level DiD per cycle + pooled) |
| `fig_parallel_trends_v2.png` | NB06 | Pre/post mean mh_score — August pre-period matched panel |
| `fig_parallel_trends_alt.png` | NB08 | Pre/post mean mh_score — Sep–Nov pre-period matched panel |
| `fig_causal_impact_cycle1.png` | NB08 | CausalImpact BSTS output — Cycle 1 |
| `fig_causal_impact_cycle2.png` | NB08 | CausalImpact BSTS output — Cycle 2 |
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
| `a1000002` | imports + path setup |
| `a1000004` | load posts + parse dates |
| `a1000007` | load SVM classifiers |
| `a1000009` | NEGATIVE_KEYWORDS + anchor filtering |
| `a1000010` | save anchor_posts_v2.parquet |
| `a1000012` | load comments |
| `a1000014` | classify exposed users via link_id |
| `a1000018` | build + save exposure_labels_v2.parquet |

### NB04 `04_panel_scores.ipynb`
| cell_id | Content |
|---------|---------|
| `b1000002` | imports + path setup |
| `d1000004` | load exposure labels |
| `g1000007` | build corpus (pre+post windows) |
| `i1000009` | load classifiers |
| `k1000011` | score all texts |
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

1. **Raw vs cleaned field names**:
   - Raw: `created_utc` (int), `selftext` / `body`, `link_id` = `"t3_<id>"`
   - Cleaned: `created_dt` (ISO str), `clean_text`, `post_id` (stripped)
   - NB08 uses raw files; NB03/04 use cleaned files

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

7. **`user_community_breadth_v2.parquet` has only 1,689 rows** — far fewer than the 20,932 panel users. NB06 does a left join, so most users have null breadth (hence the `breadth_coverage < 0.95` branch almost always triggers, excluding `community_breadth_log` from PSM).

8. **`panel_scores_alt.parquet` community breadth has 6,528 nulls** (out of 7,868 rows) for the same reason.

9. **`dose_exposure_v2.parquet` `cycle` column is float64**, not int — it was created with a float merge key. Use `.astype(int)` if needed.

10. **NB06 post-level DiD uses 22,355 rows** — filtered from the full 147,569-row `post_level_scores_v2.parquet` down to matched authors only.

---

## What Is and Isn't in Git

**In git**: all `*.parquet`, `models/clf_*.joblib`, `figures/*.png`, all notebooks, all docs.

**NOT in git** (too large or regenerable):
- `r_gradadmissions_posts.jsonl` / `r_gradadmissions_comments.jsonl` (raw)
- `r_gradadmissions_*.cleaned.jsonl` (pre-cleaned, old pipeline)
- `data/processed_v2/posts_clean.jsonl` / `comments_clean.jsonl` (run NB01)

---

## Current Status & Open Issues

- **NB01–NB06**: complete and producing outputs.
- **NB08**: fixed 2026-04-07 (wrong POSTS_PATH/COMMENTS_PATH). Now uses `r_gradadmissions_*.jsonl` at repo root. `causalimpact` installed.
- **Primary power problem**: NB06 has ~155 matched pairs/cycle (August pre-period). NB08 has ~668. Consider making NB08 the primary analysis.
- **Breadth moderation (RQ2)**: inconclusive due to low breadth coverage (1,689/20,932 users). Results vary by specification.
- **Null results in NB06**: all p > 0.20. Likely underpowered, not absence of effect.

---

## How to Update This File

After any session where you change code, data schemas, results, or add notebooks/figures:
1. Update the relevant section(s) above in place.
2. Update the **Last updated** date at the top.
3. Commit with: `git add docs/LLM_CONTEXT.md && git commit -m "Update LLM_CONTEXT: <what changed>"`
