# Results

Full results from the 3-cycle BART-pipeline (Aug 2022–Jul 2025). For methodology details, see [Methodology](methodology.md).

---

## RQ1 — Does Exposure Increase Distress?

**Question**: Do users who commented on a high-distress anchor post show a larger increase in mental health distress (Dec–May) compared to matched unexposed users?

### Combined Pooled DiD — Primary Result (NB07 §9)

Stacks all three subreddits and all three cycles; subreddit + cycle fixed effects; HC3 SEs.

| ATT | 95% CI | p | n matched users | n obs |
|-----|--------|---|---|---|
| **+0.0045** | [+0.0007, +0.0083] | **0.021*** | 6,012 | 12,416 |

This is the headline cross-community estimate. See `data/processed/combined_pooled_did.csv`.

### Per-Subreddit Pooled DiD (NB06 + NB07, cycle FE)

| Subreddit | ATT | 95% CI | p | Matched pairs |
|-----------|-----|--------|---|---|
| r/GradAdmissions | +0.0038 | [−0.0028, +0.0104] | 0.255 n.s. | 1,235 |
| r/MSCS | +0.0065 | [−0.0051, +0.0181] | 0.270 n.s. | 240 |
| r/MBA | **+0.0055** | [+0.0005, +0.0106] | **0.032*** | 1,889 |

Individual subreddits are underpowered; significance is achieved in MBA alone and in the combined pooled spec.

### MBA Cycle 2 — Multi-Spec Replication

MBA Cycle 2 is the strongest within-subreddit signal, replicating across every specification:

| Spec | ATT | 95% CI | p |
|------|-----|--------|---|
| Binary DiD | +0.0096 | [+0.0011, +0.0182] | 0.027* |
| Intensity DiD | +0.0136 | [+0.0024, +0.0247] | 0.018* |
| Exposure-Prob DiD | +0.0171 | [+0.0059, +0.0282] | 0.003** |
| GPS-WLS | +0.0238 | [+0.0018, +0.0457] | 0.034* |

### Per-Dimension Results (Pooled, r/MBA)

| Dimension | ATT | p |
|-----------|-----|---|
| Anxiety | +0.0073 | 0.007** |
| Depression | +0.0068 | 0.029* |
| Stress | +0.0057 | 0.055 (borderline) |

### NB08 — Alternative Pre-Period (Sep–Nov, all subreddits)

Pre-period redefined to Sep–Nov activity before first anchor comment. Higher coverage than the August panel.

CausalImpact (Bayesian Structural Time Series) shows consistent positive relative effects across all subreddits and cycles.

---

## RQ2 — Does Community Breadth Moderate the Effect?

**Question**: Do users with broader cross-Reddit presence show a different distress response?

Three-way interaction `period × exposed × community_breadth_log` in NB06. Clean null across all subreddits:

| Subreddit | ATT (breadth mod, pooled) | p |
|-----------|---------------------------|---|
| r/GradAdmissions | −0.0017 | 0.629 n.s. |
| r/MSCS | +0.0047 | 0.537 n.s. |
| r/MBA | +0.0004 | 0.852 n.s. |

Community breadth does not moderate the distress response to anchor exposure.

---

## Parallel Trends Assumption

Formally tested via week_number × exposed regression in the pre-period (NB06 §4a). All subreddits, all cycles, and the pooled spec are n.s. — the assumption holds.

| Subreddit | Pooled coef | p |
|-----------|-------------|---|
| r/GradAdmissions | +0.000684 | 0.204 n.s. |
| r/MSCS | −0.000450 | 0.636 n.s. |
| r/MBA | +0.000297 | 0.502 n.s. |

See `fig_parallel_trends_pretest_{SUBREDDIT}.png` and `parallel_trends_test_v2.csv` per subreddit.

---

## Key Figures

| Figure | Description |
|--------|-------------|
| `fig_att_coef_{SUBREDDIT}.png` | ATT coefficient plot per cycle + pooled (NB06) |
| `fig_parallel_trends_{SUBREDDIT}.png` | Pre/post means by exposure — August pre-period matched panel (NB06) |
| `fig_parallel_trends_pretest_{SUBREDDIT}.png` | Formal parallel trends pre-test: weekly mh_score trend (NB06 §4a) |
| `fig_parallel_trends_alt_{SUBREDDIT}.png` | Pre/post means — Sep–Nov pre-period panel (NB08) |
| `fig_causal_impact_cycle{1,2,3}_{SUBREDDIT}.png` | CausalImpact BSTS output per cycle (NB08) |
| `fig_att_comparison.png` | Forest plot: ATT for all 3 subreddits side by side (NB07) |
| `fig_mhscore_distributions.png` | Pre/post mh_score boxplots: all 3 subreddits (NB07) |
| `fig_anchor_comparison.png` | Anchor post characteristics across subreddits (NB07) |
| `fig_nli_validation_scores.png` | BART score distribution + SVM vs BART scatter (NB09) |

---

## Summary

Exposure to high-distress anchor posts is associated with a small but statistically significant increase in distress scores when pooling across all three communities and cycles (ATT = +0.0045, p = 0.021). The effect is directionally consistent across all subreddits and specifications. MBA is the only subreddit to reach significance individually (+0.0055, p = 0.032), driven especially by Cycle 2. GradAdmissions and MSCS estimates are positive but underpowered. Community breadth does not moderate the effect (RQ2 null).
