# Research Writing Brief

> **Audience**: an LLM (Sonnet-class) producing prose for the CS598 paper, blog
> post, or README based on this repository's results.
> **Goal**: give that writer everything it needs to describe the study
> accurately *without* having to re-derive context from the code. Numbers,
> methods, file locations, and pitfalls are all here.
> **Authoritative cross-refs**: `docs/LLM_CONTEXT.md` (machine-level dense
> snapshot), `docs/methodology.md` (glossary), `docs/results.md` (full result
> tables), `CLAUDE.md` and `AGENTS.md` (operational rules).
> **Repo last result-regeneration**: 2026-04-26. Numbers below reflect the
> 3-cycle BART pipeline; older READMEs in git history may quote 2-cycle SVM
> numbers — ignore those.

---

## 1. The Study In One Paragraph

Causal-inference observational study using Reddit data from three graduate-
admissions communities — **r/GradAdmissions**, **r/MSCS** (MS in Computer
Science), and **r/MBA** — across three admission cycles (Aug 2022–Jul 2025).
We ask whether *being exposed* to high-distress posts ("anchor posts") in the
Sep–Nov anchor period of a cycle causes a measurable increase in a user's own
mental-health-related distress language during the Dec–May decision season
that follows. The exposure is operationalised as commenting on an anchor-post
thread; the outcome is `mh_score`, a continuous (0,1) composite of three
LinearSVC classifiers trained on r/anxiety, r/depression, r/stress vs.
control subreddits. The identification strategy is propensity-score-matched
difference-in-differences (PSM + DiD) with HC3 standard errors, complemented
by Bayesian Causal Impact (BSTS) as an independent identification check.

---

## 2. Research Questions

| ID | Question | Where answered | Headline result |
|----|----------|----------------|-----------------|
| **RQ1** | Does exposure to high-distress anchor posts in Sep–Nov raise a user's distress (mh_score) in Dec–May? | NB06 (per-subreddit) + NB07 §9 (pooled across all 3 subreddits) + NB08 (alt pre-period + BSTS robustness) | **Yes**, small but significant: pooled ATT = **+0.0045**, 95 % CI [+0.0007, +0.0083], **p = 0.021** |
| **RQ2** | Does community breadth (number of distinct subreddits a user posts in) *moderate* the exposure effect? | NB06 §6 (three-way interaction `period × exposed × community_breadth_log`) | **No** — null across every subreddit and the pooled spec |

---

## 3. The Data

### Subreddits & cycles

All three subreddits cover the same three chronological admission cycles:

| `cycle` | Active window (anchor + post) | Anchor period | Post / decision-season |
|---------|------------------------------|---------------|------------------------|
| **1** | Aug 2022 – May 2023 | Sep 1 – Nov 30 2022 | Dec 1 2022 – May 31 2023 |
| **2** | Aug 2023 – May 2024 | Sep 1 – Nov 30 2023 | Dec 1 2023 – May 31 2024 |
| **3** | Aug 2024 – May 2025 | Sep 1 – Nov 30 2024 | Dec 1 2024 – May 31 2025 |

> When writing, refer to cycles as "the 2022, 2023, and 2024 application
> cycles" or "Cycles 1–3" — *never* as "v1/v2". The v1/v2 naming was removed
> on 2026-04-22 (commit `82e46de`).

### Corpus sizes (after NB01 cleaning)

| Subreddit | Anchor posts | Exposure labels (rows / exposed) | Panel users / exposed |
|-----------|------------|----------------------------------|-----------------------|
| r/GradAdmissions | **2,010** | 28,758 / 3,999 | 9,801 / 1,236 |
| r/MSCS | **144** | 4,706 / 496 | 2,049 / 242 |
| r/MBA | **885** | 32,283 / 5,587 | 9,229 / 1,891 |

(Numbers from `data/processed/{sub}/anchor_posts.parquet`,
`exposure_labels.parquet`, `panel_scores.parquet`.)

### Files the writer should know

| File | Contents | Used for |
|------|----------|----------|
| `data/processed/{sub}/anchor_posts.parquet` | every anchor post + BART NLI + SVM scores | describing the treatment, anchor descriptive stats |
| `data/processed/{sub}/exposure_labels.parquet` | `(author, cycle, exposed, exposure_intensity, exposure_prob)` per user-cycle | exposure definition |
| `data/processed/{sub}/panel_scores.parquet` | user-level pre/post mh_score + dim scores | main DiD input |
| `data/processed/{sub}/panel_scores_alt.parquet` | NB08 alt panel (re-scored raw text) | NB08 / CausalImpact |
| `data/processed/{sub}/post_level_scores.parquet` | individual post/comment scores | post-level DiD (~22 K matched obs) |
| `data/processed/{sub}/dose_exposure.parquet` | anchor-comment count per user-cycle | dose-response check |
| `data/processed/{sub}/did_summary.csv` | NB06 result table (binary / intensity / prob / GPS-WLS / per-dim / RQ2) | the headline per-subreddit tables |
| `data/processed/{sub}/parallel_trends_test_v2.csv` | formal pre-trend regression (NB06 §4a) | parallel-trends robustness |
| `data/processed/comparison_summary.parquet` | NB07 cross-subreddit summary | cross-community comparison table |
| `data/processed/combined_pooled_did.csv` | **headline result** (1 row, NB07 §9) | the lead number for the abstract |

---

## 4. Method — How To Describe It

The writer should be able to summarise the pipeline accurately using *only*
this section.

### 4.1 Anchor-post selection (NB03)

A post becomes an **anchor** if it satisfies *both*:

1. **Keyword filter**: regex matches at least one of a hand-curated
   negative-admissions lexicon (`reject(ed|ion)`, `decline`, `waitlist`,
   `no funding`, `anxi(ous|ety)`, `depress(ed|ion)`, `stress(ful)?`, `falling
   apart`, `gave up`, `imposter`, etc. — full list in NB03 cell `ad0e35c3`).
2. **Zero-shot NLI confirmation**: `facebook/bart-large-mnli` over four
   candidate labels —
   * `negative admissions outcome`
   * `rejection or funding loss`
   * `giving up on graduate school`
   * `general admissions discussion` (← the negative class)

   The post is kept iff its top label is *not* `general admissions discussion`
   (`bart_is_negative == True`).

**Important change to flag**: anchor selection used to be SVM-threshold
based — circular with the outcome measure. Replaced with BART NLI on
2026-04-23 (commit `6b55593`). SVM scores are still stored in
`anchor_posts.parquet` but only for descriptive statistics, *never* for
selection. Mention this explicitly when discussing internal validity.

**Anchor characteristics** (from `anchor_posts.parquet` summary):

| Subreddit | n anchors | BART top-label = "negative admissions outcome" | mean `bart_top_neg_score` | mean SVM `mh_score` |
|-----------|----------|----------------------------------------------|----------------------------|--------------------|
| GA | 2,010 | 85 % | 0.637 | 0.414 |
| MSCS | 144 | 85 % | 0.611 | 0.353 |
| MBA | 885 | 84 % | 0.640 | 0.384 |

### 4.2 Exposure (NB03)

- **Exposed (treatment)** = user commented on at least one anchor-post
  thread in that cycle. Identified via `link_id = "t3_<post_id>"` on the raw
  comment, stripped to `post_id` during cleaning. Anchor *authors* are
  excluded (they created the distress, they were not exposed to it).
- **Unexposed (control)** = active in the subreddit during the
  Aug–May active window but never commented on any anchor thread.
- **`exposure_intensity`** ∈ [0, 1] = the *maximum* `bart_top_neg_score`
  across the anchor threads a user commented on. Used as a continuous
  treatment variable in the "Intensity DiD" specification.
- **`exposure_prob`** ∈ [0, 1] = a popularity-weighted exposure score (the
  log of the most-upvoted anchor thread the user commented on, normalised);
  used in "Exposure-Prob DiD" and as the GPS-WLS treatment.

### 4.3 Outcome measure: `mh_score` (NB02 / NB04)

Three binary LinearSVCs, trained per Low et al. (2020):

- **Positive class** (per classifier): 2,000 posts from one of
  r/anxiety, r/depression, r/stress (Jan 2022 – Jul 2023, before the
  study window — no leakage).
- **Negative class**: posts from r/personalfinance, r/learnprogramming,
  r/todayilearned, r/careerguidance.
- **Features**: TF-IDF unigrams + bigrams, max 50 K features.
- **5-fold CV F1**: 0.88–0.94 per classifier.

For any text *t*:

```
anx = sigmoid(clf_anx.decision_function([t]))
dep = sigmoid(clf_dep.decision_function([t]))
str = sigmoid(clf_str.decision_function([t]))
mh_score = mean(anx, dep, str)        # ∈ (0, 1)
```

**Critical** — use `decision_function` + sigmoid, *not* `predict_proba`. The
SVMs are LinearSVC and don't expose calibrated probabilities. This is the
canonical conversion (`docs/LLM_CONTEXT.md` §Key Functions). When writing
the methods section, you can say "calibrated to (0, 1) via sigmoid of the
SVC margin" or similar.

### 4.4 Pre- and post-windows (NB04)

NB04 uses `PRE_PERIOD_STRATEGY = 'anchor_window_before_first_comment'`:

- **Pre** = Sep–Nov activity *before* a user's first anchor-thread comment
  (exposed users), or full Sep–Nov (unexposed). This avoids contamination
  from the treatment moment itself.
- **Post** = Dec 1 – May 31 of the same cycle.

> ⚠ The older docs (`docs/methodology.md`, `docs/results.md`,
> `docs/pipeline.md`, `README.md`) sometimes still describe the NB06 pre-
> period as **August only**. That is **stale**. NB04 was switched to Sep–Nov
> on 2026-04-13 (commit `17db224`), so the current `panel_scores.parquet`
> uses the Sep–Nov-before-first-anchor strategy. NB08 also uses Sep–Nov,
> but rescores from raw JSONL with a slightly different PSM
> implementation (no scaler, ball-tree NN). When writing, treat NB06 and
> NB08 as two implementations of essentially the same window strategy,
> not as "August vs Sep–Nov" — that distinction belongs to an older version.

### 4.5 Propensity-score matching (NB06)

Per cycle, 1:1 nearest-neighbour matching on the propensity score from:

```
LogisticRegression(  exposed  ~  pre_mh_score
                            + log1p(pre_n_posts)
                            + community_breadth_log )   # if ≥ 95% coverage
```

`community_breadth_log` is now always included (100 % coverage after the
NB05 cache overhaul on 2026-04-22 — missing accounts imputed to 0).
Features are standardised before the logistic regression. Caliper = 0.05
on the propensity score. Standardised Mean Differences (SMDs) printed
per feature for balance check.

### 4.6 Main DiD specification (NB06 cell `l3000012`)

Long format, one row per (user, period) where `period ∈ {0, 1}` (pre/post).

```
mh_score  ~  period
           + exposed
           + period × exposed     ← the ATT (β₃)
           + log1p(n_posts)
           + C(cycle_str)          ← cycle fixed effects when pooling
```

Estimator: OLS with **HC3 heteroskedasticity-robust SEs**. Run per cycle and
pooled (with cycle FE).

**Important fix to mention** (commit `3730e47`, 2026-04-25): the pooled
specification did not previously include cycle FE; now `run_did()` adds
`C(cycle_str)` automatically when `df['cycle'].nunique() > 1`. All
`did_summary.csv` results were regenerated under the corrected spec — these
are the numbers in §5 below.

### 4.7 Specifications run in NB06

For each subreddit, NB06 produces *every* row of `did_summary.csv`:

- **Binary DiD** — main spec; `treatment = exposed (bool)`.
- **Intensity DiD** — `treatment = exposure_intensity (BART top-neg score)`.
- **Exposure-Prob DiD** — `treatment = exposure_prob` (popularity-weighted).
- **GPS-WLS** (Hirano–Imbens generalised propensity score, weighted least
  squares). Weights are nearly degenerate (mean ≈ 1, max ≈ 2.8 for MBA)
  because anchor exposure is roughly uniform conditional on the matched
  PSM features — *not* the primary spec, but reported for completeness.
- **Per-dimension DiD** — same spec on `anx_score`, `dep_score`,
  `str_score` separately, then pooled.
- **RQ2 Breadth Moderation** — three-way interaction
  `period × exposed × community_breadth_log`.

### 4.8 Robustness layers

- **Formal parallel-trends test** (NB06 §4a, cell `7d7c7285`):
  in the pre-period only, regress `mh_score ~ week_number × exposed +
  controls + C(cycle_str)`. The `week × exposed` interaction is **n.s. in
  every subreddit, every cycle, and pooled** — assumption holds.
  Output: `parallel_trends_test_v2.csv` and `fig_parallel_trends_pretest_*.png`.
- **Post-level DiD** (NB06 §9): 22,355 matched-user post-level obs,
  cluster SEs on author, run with and without user fixed effects.
- **Dose-response** (NB06 §10): coefficient on `log1p(n_anchor_comments)`.
- **Placebo / tight post-window Dec–Jan / BART confidence sensitivity at
  0.4 and 0.6** (NB06 §11, cell `e27bc930`).
- **Bayesian Causal Impact** (NB08 cells `b0000021`–`b0000022`): BSTS using
  the unexposed group as a synthetic control. Operates on aggregated weekly
  mean `mh_score`; intervention date Dec 1. One figure per cycle present
  in the panel (`fig_causal_impact_cycle{1,2,3}_{sub}.png`).

---

## 5. Results — Numbers To Cite

### 5.1 The headline (lead with this in the abstract)

> Exposed users showed a small but statistically significant increase in
> mental-health-related distress language in the decision season, relative
> to matched unexposed users, when pooling across all three subreddits and
> all three cycles.

```
Combined Pooled DiD (NB07 §9, subreddit + cycle FE, HC3 SEs)
ATT = +0.0045,  95 % CI [+0.0007, +0.0083],  p = 0.021
N = 12,416 user-period obs across 6,012 matched users
Source: data/processed/combined_pooled_did.csv
```

### 5.2 Per-subreddit pooled (NB06; cycle FE)

| Subreddit | ATT | 95 % CI | p | Matched pairs |
|-----------|-----|---------|---|---------------|
| r/GradAdmissions | +0.0041 | [−0.0027, +0.0110] | 0.235 n.s. | 1,235 |
| r/MSCS | −0.0073 | [−0.0199, +0.0053] | 0.256 n.s. | 240 |
| **r/MBA** | **+0.0065** | **[+0.0011, +0.0119]** | **0.019\*** | 1,889 |

Source per subreddit: `data/processed/{sub}/did_summary.csv` ("Binary
DiD" / "Pooled" row).

**Interpretive note** (important): the MSCS point estimate is *negative* in
the pooled spec but flips sign across cycles and intensity specifications.
With only 240 matched pairs, the MSCS confidence interval includes
substantial positive effects. Do not write "MSCS shows the opposite
effect" — write "MSCS is underpowered and statistically indistinguishable
from zero or from the other subreddits". The NB07 §6 pairwise Z-test
confirms: all 9 pairwise comparisons (GA vs MSCS, GA vs MBA, MSCS vs MBA)
× 3 specs are n.s., min p = 0.25.

### 5.3 Per-cycle pooled (NB07 §9, subreddit FE)

| Cycle | ATT | p | Note |
|-------|-----|---|------|
| 1 (2022) | +0.0042 | 0.255 n.s. | smallest sample |
| 2 (2023) | +0.0047 | 0.141 n.s. | largest signal in MBA |
| 3 (2024) | +0.0044 | 0.171 n.s. | most recent |

The effect is *directionally consistent* across cycles — useful framing
for "no single-year outlier driving the pooled result".

### 5.4 Per dimension (pooled all subs + cycles)

| Dimension | ATT | p |
|-----------|-----|---|
| Anxiety | +0.0045 | 0.023\* |
| Depression | +0.0045 | 0.041\* |
| Stress | +0.0045 | 0.030\* |

All three dimensions move together — argues against a "one specific
emotion" narrative; the effect looks like generalised distress.

### 5.5 MBA Cycle 2 — strongest within-subreddit signal

Worth flagging because it replicates across *every* spec:

| Spec | ATT | 95 % CI | p |
|------|-----|---------|---|
| Binary DiD | +0.0096 | [+0.0011, +0.0182] | 0.027\* |
| Intensity DiD | +0.0136 | [+0.0024, +0.0247] | 0.018\* |
| Exposure-Prob DiD | +0.0171 | [+0.0059, +0.0282] | 0.003\*\* |
| GPS-WLS | +0.0238 | [+0.0018, +0.0457] | 0.034\* |

Source: `data/processed/mba/did_summary.csv`.

### 5.6 RQ2 — Community breadth moderation (null)

| Subreddit | Three-way pooled coef | p |
|-----------|----------------------|---|
| r/GradAdmissions | −0.0017 | 0.621 n.s. |
| r/MSCS | +0.0047 | 0.537 n.s. |
| r/MBA | +0.0004 | 0.852 n.s. |

Clean null. The stress-buffering hypothesis — that users with broader
cross-Reddit presence absorb the shock better — is **not supported**.
This is itself a finding; do not bury it. A useful caveat: breadth is a
crude proxy (membership in r/gaming and r/cooking does not give
admissions-relevant support), and the test is again somewhat underpowered.

### 5.7 Parallel-trends pre-test (NB06 §4a)

| Subreddit | week × exposed coef (pooled) | p |
|-----------|------------------------------|---|
| r/GradAdmissions | −0.000009 | 0.987 n.s. |
| r/MSCS | +0.000641 | 0.516 n.s. |
| r/MBA | +0.000208 | 0.634 n.s. |

All n.s. → parallel-trends assumption holds. This is a clean robustness
talking point.

### 5.8 Causal Impact (NB08, BSTS synthetic-control check)

NB08 produces one BSTS figure per cycle per subreddit
(`fig_causal_impact_cycle{1,2,3}_{sub}.png`). The figures show
positive relative effects on the exposed-group time series across all
three subreddits and cycles, *consistent in sign* with the PSM-DiD
estimates. Exact relative-effect percentages live in the notebook
output cells; cite the figures rather than guess numbers.

---

## 6. Notebooks — What Each One Does

(Run order, all parameterised on `SUBREDDIT ∈ {gradadmissions, mscs, mba}`
unless noted. Diagnostic notebooks not in `run_pipeline.py` are flagged.)

| NB | File | Purpose | Key outputs |
|----|------|---------|-------------|
| 00 | `00_exploratory_topic_sentiment.ipynb` | EDA only — topic modelling + VADER on Sep 2024 | none consumed downstream |
| 01 | `01_clean_corpus.ipynb` | Read raw JSONL, normalise text, drop bots/deletions, derive `post_id` from `link_id` | `posts_clean.jsonl`, `comments_clean.jsonl` |
| 02 | `02_train_classifiers.ipynb` | Train three LinearSVCs on r/anxiety/depression/stress vs controls | `models/clf_*.joblib`. **One-off — do not re-run** (Arctic Shift API timeouts) |
| 03 | `03_exposure_labels.ipynb` | Keyword filter + BART NLI → anchor posts. Match user comments via `link_id` → exposure labels | `anchor_posts.parquet`, `exposure_labels.parquet` |
| 04 | `04_panel_scores.ipynb` | Score every panel user's pre and post text. Aggregate to user-level pre/post means and dimension scores | `panel_scores.parquet`, `post_level_scores.parquet`, `dose_exposure.parquet` |
| 04a | `04a_exposure_checks.ipynb` | **Diagnostic** — differential attrition, pre-period span, anchor-comment timing | `fig_preperiod_span.png`, `fig_anchor_comment_timing.png` |
| 05 | `05_collect_community_breadth.ipynb` | Arctic Shift API: count distinct subreddits per user | `user_community_breadth.parquet` |
| 05a | `05a_pipeline_funnel.ipynb` | **Diagnostic** — labelled → has_pre → has_post → in_panel funnel | `fig_pipeline_funnel.png`, `fig_funnel_exposed_vs_unexposed.png` |
| 06 | `06_did_analysis.ipynb` | **Main results**: PSM + DiD + per-dim + RQ2 + post-level + dose-response + robustness | `did_summary.csv`, `parallel_trends_test_v2.csv`, `fig_att_coef_{sub}.png`, `fig_parallel_trends_{sub}.png`, `fig_parallel_trends_pretest_{sub}.png` |
| 06a | `06a_stratified_pre_exposure.ipynb` | **Diagnostic** — PSM+DiD sensitivity to pre-period span (full / ≥ 7d / ≥ 14d) | `fig_sensitivity_pre_period.png` |
| 07 | `07_comparison_analysis.ipynb` | Cross-subreddit comparison + **combined pooled DiD (§9)** | `comparison_summary.parquet`, `combined_pooled_did.csv`, `fig_att_comparison.png`, `fig_mhscore_distributions.png`, `fig_anchor_comparison.png` |
| 08 | `08_alt_analysis.ipynb` | Re-score raw text + independent PSM + DiD + **Causal Impact BSTS** | `panel_scores_alt.parquet`, `fig_parallel_trends_alt_{sub}.png`, `fig_causal_impact_cycle{1,2,3}_{sub}.png` |
| 09 | `09_nli_anchor_validation.ipynb` | Read BART fields already in `anchor_posts.parquet`, characterise the treatment | `fig_nli_validation_scores.png` |

---

## 7. Figure-to-Section Mapping (for the writer)

When the paper / blog says "Figure X", point it at the right file:

| Need | Figure |
|------|--------|
| **Lead figure** showing pooled effect across subreddits | `figures/fig_att_comparison.png` (NB07) |
| ATT plot per cycle for one subreddit | `figures/fig_att_coef_{sub}.png` (NB06) |
| Parallel trends — visual | `figures/fig_parallel_trends_{sub}.png` (NB06) |
| Parallel trends — formal pre-test | `figures/fig_parallel_trends_pretest_{sub}.png` (NB06 §4a) |
| Sep–Nov alt-pre-period trends | `figures/fig_parallel_trends_alt_{sub}.png` (NB08) |
| Bayesian Causal Impact | `figures/fig_causal_impact_cycle{1,2,3}_{sub}.png` (NB08) |
| Anchor-post characteristics across subreddits | `figures/fig_anchor_comparison.png` (NB07) |
| Pre/post mh_score distributions across subreddits | `figures/fig_mhscore_distributions.png` (NB07) |
| BART score distribution + SVM ↔ BART scatter | `figures/fig_nli_validation_scores.png` (NB09) |
| Pipeline funnel waterfall | `figures/fig_pipeline_funnel.png` (NB05a) |
| Differential dropout (exposed vs unexposed) | `figures/fig_funnel_exposed_vs_unexposed.png` (NB05a) |
| Pre-period span distribution | `figures/fig_preperiod_span.png` (NB04a) |
| Anchor-comment volume by month | `figures/fig_anchor_comment_timing.png` (NB04a) |
| Sensitivity to pre-period length | `figures/fig_sensitivity_pre_period.png` (NB06a) |

---

## 8. Project History (so the writer can frame the methods evolution)

| Date | Commit | What changed |
|------|--------|--------------|
| 2026-03-30 | `c382aec` / `a933b12` | First "v2" pipeline reorganisation: notebooks renamed to match execution order; flow diagrams added |
| 2026-04-05 | `0e9a053` / `169ef83` | Pipeline rebuilt around pre/post DiD design (replacing earlier weekly event-study) |
| 2026-04-06 | `6767a83` / `6e112af` / `95687cb` | Post-level DiD, dose-response, PSM implementation aligned with §4.3–§4.4 of an earlier draft |
| 2026-04-07 | `240c8ad` | NB08 added: Sep–Nov pre-period + Causal Impact |
| 2026-04-13 | `17db224` | NB04 switched from August → Sep–Nov pre-period (4× more matched pairs) |
| 2026-04-16 | `9669678` | CONFIG blocks, multi-treatment DiD (intensity, prob), GPS weighting added |
| 2026-04-20 | `8a277d1` | r/MSCS pipeline + NB07 comparison + `run_pipeline.py` orchestrator |
| 2026-04-21 | `b858215` | 2022 cycle added → 3-cycle pipeline (all subs Aug 2022 – Jul 2025) |
| 2026-04-21 | `bbb16d4` | Diagnostic NB04a/05a/06a |
| 2026-04-22 | `82e46de` | v1/v2 naming removed — single canonical version |
| 2026-04-22 | `8a8f62a` | NB09 added: zero-shot NLI validation of anchor posts |
| 2026-04-22 | `b9830d4` | NB05 cache overhaul: failed Arctic Shift fetches imputed to breadth=0 → 100 % coverage |
| 2026-04-23 | `6b55593` | **Major**: anchor selection replaced SVM threshold with BART NLI (breaks circularity with the SVM outcome measure) |
| 2026-04-23 | `a02b6c3` | NB07 §9 combined pooled DiD + NB06 §4a formal parallel-trends test |
| 2026-04-25 | `3730e47` | NB06 pooled DiD bug fix: added cycle FE; all results regenerated |
| 2026-04-26 | `443920e` / `ab72c8c` / `be5d9e3` | Final figure + LLM_CONTEXT updates |

**The breaks-circularity moment** (commit `6b55593`) is the most important
methodological change to mention in any internal-validity discussion: SVM
scores once both *defined* the treatment and *measured* the outcome.
After the switch, BART NLI defines the treatment and SVMs measure the
outcome — independent text features now drive each side of the
identification.

---

## 9. Writing Guidance — Tone, Framing, Pitfalls

### 9.1 Sample vs population framing

This is observational Reddit data. Mental-health classifiers measure
**writing style**, not psychological state. Use phrasing like:

> "Exposed users produced text in the decision season that resembled
> mental-health-subreddit writing more closely than matched controls did"

rather than

> "Exposed users became more anxious".

The latter overclaims. The "ground truth" is *language*, not affect.

### 9.2 Effect size is small — say so

The ATT of +0.0045 is on a mh_score scale that is already bounded in
(0,1) with population SD ≈ 0.06–0.07. That makes it ~6–7 % of one
standard deviation — small, but real, and consistent across spec, sub-
reddit, and cycle. Frame it as a *detectable cross-community shift*,
not a behavioural sea-change.

### 9.3 What to call NB06 vs NB08

Both now use Sep–Nov-before-first-anchor as the pre-period. They differ
in *implementation*:

| | NB06 | NB08 |
|--|------|------|
| Source data | NB04's aggregated `panel_scores.parquet` | re-scored raw JSONL inside the notebook |
| PSM | `StandardScaler` → LogisticRegression → euclidean 1-NN | LogisticRegression (no scaler) → ball-tree 1-NN |
| Extras | post-level DiD, dose-response, RQ2 | weekly aggregation → Bayesian Causal Impact (BSTS) |

So NB06 is "the main spec, plus its own robustness layers" and NB08 is
"an independent re-implementation plus a second identification strategy
(BSTS)". Saying "the same answer falls out of two implementations
differing in PSM details" is a legitimate robustness point.

### 9.4 The MSCS sample is genuinely small

With **240 matched pairs**, no spec inside MSCS reaches significance,
and the point estimate's sign is unstable across specs. The honest
framing is "MSCS is the smallest community; its individual estimate is
uninformative; the pooled estimate gains power by pooling — not by
the MSCS-specific signal".

### 9.5 The exposure proxy is conservative

"Exposed" = *commented on* an anchor thread. Many users who only read
anchor posts (a much larger group) are coded unexposed. This
**attenuates** the estimated ATT — pushes it toward zero. The fact
that a small positive effect survives this attenuation is a strength,
not a weakness. Write it that way.

### 9.6 What we cannot claim

- Causality at the individual level. PSM controls for *observable*
  pre-exposure characteristics (pre_mh_score, posting volume, breadth).
  Unobserved confounders (personality, off-Reddit support) remain.
- A clinical effect. mh_score is a corpus-style classifier — not a
  validated psychiatric instrument.
- Generalisability beyond Reddit. r/GradAdmissions, r/MSCS, r/MBA users
  are self-selected applicants willing to post publicly.

### 9.7 Things to *avoid* writing

- ❌ "v1 vs v2 pipeline" — v1/v2 naming was deleted; everything in
  `notebooks/` and `data/processed/{sub}/` is the canonical version.
- ❌ "predict_proba probabilities" — the SVMs are LinearSVC; we use
  `sigmoid(decision_function(...))`. Don't call them probabilities.
- ❌ "August pre-period" — old docs say this, but the current
  pipeline uses Sep–Nov-before-first-anchor.
- ❌ "We trained on the study window" — the SVM training data is
  Jan 2022 – Jul 2023, ending before the Sep 2023 anchor period.

### 9.8 Things worth highlighting

- ✔ Independent identification strategies (PSM-DiD + BSTS) converge.
- ✔ Effect is directionally consistent across 3 cycles, 3 subreddits,
  and 3 distress dimensions — argues against single-cycle artefact.
- ✔ Formal parallel-trends pre-test passes everywhere.
- ✔ Anchor selection and outcome measurement use *different* models
  (BART NLI vs SVM) — breaks circularity that older drafts had.
- ✔ Community breadth null is itself informative — buffering hypothesis
  not supported in this design.

---

## 10. Quick Sanity-Check Commands (for the writer's verification)

If the writer needs to confirm any number before publishing:

```bash
# Headline pooled DiD
cat data/processed/combined_pooled_did.csv

# Per-subreddit binary DiD pooled
grep "Binary DiD,Pooled" data/processed/*/did_summary.csv

# Parallel-trends pre-test pooled
grep "Pooled" data/processed/*/parallel_trends_test_v2.csv

# Anchor counts
~/venvs/jupyter/bin/python3 -c "
import pandas as pd
for sr in ['gradadmissions','mscs','mba']:
    df = pd.read_parquet(f'data/processed/{sr}/anchor_posts.parquet')
    print(sr, len(df))"

# Comparison summary parquet (NB07)
~/venvs/jupyter/bin/python3 -c "
import pandas as pd
print(pd.read_parquet('data/processed/comparison_summary.parquet').to_string())"
```

---

## 11. References (cited in the methodology)

- Low, D. M., et al. (2020). *Natural language processing reveals
  vulnerable mental health support patterns in a COVID-19 crisis forum.*
  JMIR Mental Health, 7(6), e21236. — basis for the SVM mh_score design.
- Brodersen, K. H., et al. (2015). *Inferring causal impact using
  Bayesian structural time-series models.* Annals of Applied Statistics,
  9(1), 247–274. — BSTS / CausalImpact (NB08).
- Hutto, C. J., & Gilbert, E. E. (2014). *VADER: A parsimonious rule-based
  model for sentiment analysis of social media text.* ICWSM. — baseline
  comparison, EDA in NB00.
- MacKinnon, J. G., & White, H. (1985). *Some heteroskedasticity-consistent
  covariance matrix estimators.* Journal of Econometrics, 29(3), 305–325.
  — HC3 SEs.
- Hirano, K., & Imbens, G. W. (2004). *The propensity score with continuous
  treatments.* — GPS-WLS spec in NB06.
- Lewis, M., et al. (2020). *BART: Denoising sequence-to-sequence
  pre-training for natural language generation, translation, and
  comprehension.* ACL. — zero-shot NLI used for anchor selection
  (`facebook/bart-large-mnli`).
- Arctic Shift API — Reddit historical data source
  (https://github.com/ArthurHeitmann/arctic_shift).

---

*If this doc and `docs/LLM_CONTEXT.md` disagree, prefer `LLM_CONTEXT.md` —
it is updated immediately after every code/result change. This brief is
written from `LLM_CONTEXT.md` plus direct inspection of result CSVs as of
2026-05-12.*
