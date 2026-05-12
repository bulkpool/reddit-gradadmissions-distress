# Methodology

This document covers the key design concepts, terminology, and limitations of the study.

---

## Core Concepts

### Anchor Post

A Reddit post identified as a **high-distress negative disclosure** related to graduate admissions. These are the treatment events.

A post qualifies as an anchor if it meets both criteria:
1. Falls within the anchor period: Sep 1–Nov 30 of the cycle year
2. Matches a negative-admissions keyword (reject, decline, waitlist, anxiety, depressed, stress, falling apart, gave up, imposter, …) **and** zero-shot NLI via `facebook/bart-large-mnli` classifies it as one of three negative labels (`negative admissions outcome` / `rejection or funding loss` / `giving up on graduate school`) rather than `general admissions discussion` (`bart_is_negative == True`).

> Anchor selection used to threshold on `mh_score > 0.45` from the SVM classifiers. That was replaced with the keyword + BART NLI rule on 2026-04-23 to break the circularity between the SVM-defined treatment and the SVM-measured outcome — see commit `6b55593`. SVM scores are still stored in `anchor_posts.parquet` for descriptive use but are no longer used for selection.

**Result**: 2,010 anchors in r/GradAdmissions, 144 in r/MSCS, 885 in r/MBA across all three cycles; see `data/processed/{sub}/anchor_posts.parquet`.

---

### Exposure

Whether a user was **treated** in a given cycle.

- **Exposed**: user commented on an anchor post thread, identified via `link_id` in the raw JSONL matching the anchor post's `id`. Anchor post *authors* are excluded.
- **Unexposed**: active in the same subreddit during the full Aug 1–May 31 window of that cycle but never commented on any anchor thread.

Exposure is identified at thread level using `link_id` (raw comments field: `link_id = "t3_<post_id>"`). The cleaned files carry this as `post_id` after stripping the `t3_` prefix.

---

### Difference-in-Differences (DiD)

A causal inference method that removes the influence of time trends by comparing *changes* across two groups.

**Logic**: If exposed and unexposed users are similar before the event, any divergence afterward can be attributed to the exposure.

**Regression**:
```
mh_score = β₀ + β₁·period + β₂·exposed + β₃·(period × exposed) + β₄·log(n_posts) + ε
```

`β₃` is the DiD estimate — the effect of exposure, net of time trends. Positive = exposure increases distress.

**Why not just compare before/after?** Distress naturally rises during peak admissions season. DiD removes this confound by using the unexposed group as a counterfactual experiencing the same time trends without the exposure.

---

### Parallel Trends Assumption

DiD's key assumption: absent treatment, the exposed and unexposed groups would have followed the same distress trajectory.

Verified with the parallel trends plots (`fig_parallel_trends_{SUBREDDIT}.png` for NB06, `fig_parallel_trends_alt_{SUBREDDIT}.png` for NB08) and a formal pre-trend regression (`fig_parallel_trends_pretest_{SUBREDDIT}.png`) — the week_number × exposed interaction is n.s. across all subreddits and cycles.

---

### Propensity Score Matching (PSM)

Because exposed users (those who engaged with distressed content) may differ from unexposed users in baseline characteristics, a direct comparison risks confounding. PSM corrects for this.

**Process** (1:1 nearest-neighbor, per cycle):
1. Fit logistic regression predicting `exposed` from `pre_mh_score + log1p_n_posts_pre` (+ `community_breadth_log` if ≥ 95% coverage)
2. Compute propensity score (probability of exposure given pre-treatment characteristics)
3. Match each exposed user to the nearest unexposed user (caliper = 0.05 on propensity score)
4. Check balance via Standardised Mean Differences (SMD < 0.1 target)
5. Only matched pairs enter the DiD regression

**NB06 result** (Sep–Nov pre-period): per-subreddit matched pairs are 1,235 (GA), 240 (MSCS), and 1,889 (MBA) pooled across all three cycles. See `did_summary.csv` per subreddit.

**NB08 result**: independent re-implementation on re-scored raw text; PSM uses the same Sep–Nov-before-first-anchor strategy with a slightly different matcher (no scaler, ball-tree NN). Used as a robustness check rather than as a "more powerful" alternative — NB04 already adopted the Sep–Nov pre-period (commit `17db224`, 2026-04-13).

---

### mh_score

The primary outcome variable. Continuous ∈ (0, 1), measuring how closely a user's writing resembles posts from mental health subreddits.

```
mh_score = mean(sigmoid(anx_df), sigmoid(dep_df), sigmoid(str_df))
```

`anx_df`, `dep_df`, `str_df` are `decision_function` outputs from three LinearSVC models passed through a sigmoid. Aggregated to mean per (user, cycle, window) at the panel level.

**Validity check**: mh_score correctly ranks posts by self-reported outcome across the corpus — Rejected scores higher than Accepted.

---

### Community Breadth

Number of distinct subreddits a user posted or commented in during the study window (Aug 2022–Jul 2025), excluding the focal admissions subreddit and the user's own profile sub. Log-transformed (`log1p(breadth)`) in regression models. Failed Arctic Shift fetches and deleted accounts are imputed to `breadth = 0`, giving 100 % coverage across panel users.

**Theoretical role (RQ2)**: proxy for social network diversity. The stress-buffering hypothesis predicts higher breadth → smaller distress response. Results in the current pipeline are inconclusive due to sample size constraints.

---

### Pre-Period Definition

Both NB06 and NB08 use the same window definition (since `17db224`, 2026-04-13): **Pre = Sep–Nov activity *before* each user's first anchor comment** (or full Sep–Nov for unexposed users). This filters baseline activity to before treatment onset and provides ~36 % panel coverage — a substantial improvement over the August-only design used in earlier drafts. NB06 reads aggregated user-level scores from `panel_scores.parquet`; NB08 re-scores raw text inline (so the two pipelines share a definition but not an implementation).

---

### Causal Impact (Bayesian Structural Time Series)

Implemented in NB08 as an independent identification strategy. Aggregates exposed and unexposed users into weekly mean mh_score time series. Fits a BSTS model using the unexposed group as a synthetic control for the exposed group's counterfactual trajectory.

Does not require individual pre+post coverage — all users with any weekly activity contribute. Pre-period = Sep–Nov; intervention = Dec 1; post-period = Dec–May.

**Implementation note**: `causalimpact 0.2.6` is incompatible with pandas 2.x. NB08 patches `pandas.core.dtypes.common.is_datetime_or_timedelta_dtype` at runtime and passes integer index positions (not datetime objects) as pre/post period arguments to `CausalImpact()`.

---

### Post-Level DiD

Implemented in NB06 as a complement to user-level DiD. Uses individual posts/comments as observations rather than user-period means, giving 22,355 observations vs. ~562 in the user-level analysis.

Run with and without user fixed effects; SEs clustered on author. Outcomes decomposed per dimension (anxiety, depression, stress).

---

### VADER vs SVM Classifiers

| | VADER | SVM (mh_score) |
|--|-------|----------------|
| Type | Rule-based lexicon | Trained classifier |
| Training | Hand-crafted word list | ~14,000 Reddit posts from mental health subs |
| Fires on | Any negative word | Language *patterns* of distressed writing |
| Problem | "Don't stress about it" scores negative | — |
| Sensitivity | Lower for domain-specific distress | Higher |

---

### HC3 Robust Standard Errors

Standard OLS assumes homoskedasticity. Social media data violates this — users vary widely in posting frequency. HC3 corrects for non-constant variance, giving reliable p-values without distributional assumptions.

---

## Limitations

1. **Per-subreddit power**: Individual subreddits — especially r/MSCS with 240 matched pairs pooled across three cycles — are underpowered to detect the small effect on their own. Significance is achieved in the cross-community pooled spec (NB07 §9) and in r/MBA alone (p = 0.032).

2. **Effect size is small**: pooled ATT of +0.0045 corresponds to ~6–7 % of one population SD of `mh_score`. Directionally consistent across all 3 subreddits, all 3 cycles, and all 3 distress dimensions (anx / dep / stress) — argues against single-cycle or single-emotion artefact.

3. **Exposure proxy**: "Exposed" means *commented on* an anchor thread, not merely *read* it. Many truly exposed users (who read but didn't comment) are classified as unexposed, attenuating the estimated effect.

4. **PSM on observables only**: Matching controls for pre-period distress, posting activity, and (where ≥ 95 % coverage) log-community-breadth, but unobserved differences (personality, resilience, offline support networks) remain as potential confounders.

5. **Text-based distress proxy**: `mh_score` measures how someone writes, not their psychological state directly. Users may be distressed without writing distressed posts, or write distressed-sounding posts strategically.

6. **Community breadth as social support proxy**: counting distinct subreddits is crude — belonging to r/gaming and r/cooking does not provide admissions-relevant social support. The RQ2 null may reflect proxy invalidity rather than absent moderation.

7. **Reddit-specific generalizability**: r/GradAdmissions, r/MSCS, and r/MBA are each self-selected communities of applicants willing to post publicly.

---

## References

- Low, D.M., et al. (2020). Natural language processing reveals vulnerable mental health support patterns in a COVID-19 crisis forum. *JMIR Mental Health*, 7(6), e21236.
- Brodersen, K.H., et al. (2015). Inferring causal impact using Bayesian structural time-series models. *Annals of Applied Statistics*, 9(1), 247–274.
- Hutto, C.J. & Gilbert, E.E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *ICWSM*.
- MacKinnon, J.G. & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators. *Journal of Econometrics*, 29(3), 305–325.
