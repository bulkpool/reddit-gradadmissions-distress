# You're Freaking Out and It Scares Me Too:
## Understanding How Exposure to Negative Discourse on Admissions Subreddits Affects Distress

Graduate admissions is stressful. But does *reading* other people's rejection posts make it worse? This project uses causal inference across three Reddit communities — **r/GradAdmissions**, **r/MSCS**, and **r/MBA** — to find out, measuring whether exposure to high-distress posts in the Sep–Nov "anchor period" of an admission cycle raises a user's own mental-health-related distress language in the Dec–May decision season that follows.

**Short answer**: yes — a small but statistically significant positive effect, replicating across two independent identification strategies (PSM-DiD and Bayesian Causal Impact) and three cycles. Combined pooled DiD across all 3 subreddits × 3 cycles:
**ATT = +0.0045, 95 % CI [+0.0007, +0.0083], p = 0.021**.

---

## Key Findings (3-cycle BART pipeline, results last refreshed 2026-04-26)

| | |
|--|--|
| **RQ1 — combined pooled DiD (NB07 §9)** | ATT = +0.0045, p = 0.021* over 6,012 matched users × 12,416 observations × 3 subreddits × 3 cycles. The headline cross-community result. |
| **RQ1 — per subreddit** | r/GradAdmissions +0.0038 (p = 0.255, n.s.), r/MSCS +0.0065 (p = 0.270, n.s.), r/MBA +0.0055 (p = 0.032\*). MBA is the only individually significant subreddit; the others are underpowered but directionally consistent. |
| **RQ1 — robustness (NB08)** | Bayesian Causal Impact (BSTS) with the unexposed group as synthetic control shows positive relative effects across all subreddits and cycles. |
| **RQ2 — community breadth** | Three-way interaction `period × exposed × log(breadth)` is null in every subreddit; the stress-buffering hypothesis is not supported. |
| **Parallel trends** | Formally tested via a pre-period `week_number × exposed` regression. n.s. across all subreddits, cycles, and the pooled spec — assumption holds. |
| **Per-dimension (pooled)** | Anxiety +0.0045 (p = 0.023\*), Depression +0.0045 (p = 0.041\*), Stress +0.0045 (p = 0.030\*) — generalised distress, not single-emotion. |

Full result tables: [`docs/results.md`](docs/results.md).

---

## How It Works

```
Raw Reddit data (3 subreddits, Aug 2022 – Jul 2025)
       │
       ▼
NB01: Corpus cleaning + post_id linking via link_id
       │
       ▼
NB02: SVM mental-health classifiers (anx / dep / stress) — one-off
       │
       ▼
NB03: Anchor identification (keyword filter + BART NLI zero-shot)
      Thread-level exposure via link_id
       │
       ▼
NB04: Panel scoring (pre = Sep–Nov before first anchor comment, post = Dec–May)
NB05: Community breadth (Arctic Shift API)
       │
       ▼
NB06: PSM + DiD per subreddit + post-level DiD + dose-response + RQ2  ← main per-sub
NB07: Cross-subreddit comparison + combined pooled DiD                ← headline
NB08: Alt panel (re-scored raw text) + Causal Impact BSTS              ← robustness
NB09: Anchor characterisation (reads BART fields from NB03)
```

Run the whole pipeline with `~/venvs/jupyter/bin/python3 run_pipeline.py` — it injects the correct `SUBREDDIT` value into each notebook's config cell at runtime via nbconvert. Source notebooks are never modified. See [`docs/quickstart.md`](docs/quickstart.md) for options.

The distress measure (`mh_score`) is the mean of three LinearSVC classifiers trained on r/anxiety / r/depression / r/stress (positive) vs four general-topic control subreddits (negative). Scored via `sigmoid(decision_function(text))` ∈ (0, 1). Sensitive to the *style* of distressed writing rather than to negative words alone.

Anchor-post selection was switched from SVM-threshold to *keyword filter + zero-shot NLI* (`facebook/bart-large-mnli`) on 2026-04-23. This breaks the circularity between the SVM-defined treatment and the SVM-measured outcome — independent text features now drive each side of the identification.

---

## Repository Structure

```
├── notebooks/
│   ├── 00_exploratory_topic_sentiment.ipynb   # EDA only
│   ├── 01_clean_corpus.ipynb                  # Text cleaning + post_id mapping
│   ├── 02_train_classifiers.ipynb             # SVM classifier training (one-off)
│   ├── 03_exposure_labels.ipynb               # Keyword + BART → anchors → exposure
│   ├── 04_panel_scores.ipynb                  # Pre/post panel + post-level + dose
│   ├── 04a_exposure_checks.ipynb              # Diagnostic: attrition, span, timing
│   ├── 05_collect_community_breadth.ipynb     # Arctic Shift breadth fetch
│   ├── 05a_pipeline_funnel.ipynb              # Diagnostic: funnel waterfall
│   ├── 06_did_analysis.ipynb                  # MAIN: PSM + DiD per subreddit
│   ├── 06a_stratified_pre_exposure.ipynb      # Diagnostic: pre-period sensitivity
│   ├── 07_comparison_analysis.ipynb           # Cross-subreddit + combined pooled
│   ├── 08_alt_analysis.ipynb                  # Sep–Nov re-score + Causal Impact
│   └── 09_nli_anchor_validation.ipynb         # Anchor BART characterisation
│
├── data/
│   ├── raw/               # GA-2022, MSCS, MSCS-2022 raw JSONL (not in git)
│   ├── mba/               # Pre-cleaned MBA *.jsonl.gz (in git)
│   └── processed/{sub}/   # All per-subreddit pipeline outputs (parquet, in git)
│
├── models/                # Trained SVM classifiers (.joblib, in git)
├── figures/               # All output plots (.png, in git)
├── run_pipeline.py        # Orchestrator: injects SUBREDDIT + executes via nbconvert
└── docs/                  # Documentation
```

---

## Documentation

| Doc | What's in it |
|-----|--------------|
| [LLM Context](docs/LLM_CONTEXT.md) | Dense codebase snapshot for AI assistants — start here |
| [Research Writing Brief](docs/RESEARCH_WRITING_BRIEF.md) | Methods + numbers in prose-ready form for paper / blog authors |
| [Quickstart](docs/quickstart.md) | How to set up and run the pipeline |
| [Flow](docs/flow.md) | Step-by-step walkthrough with flowcharts |
| [Pipeline](docs/pipeline.md) | What each notebook does, inputs / outputs, runtimes |
| [Results](docs/results.md) | Full result tables, figures, interpretation |
| [Methodology](docs/methodology.md) | Terminology glossary, design choices, limitations |

---

## Dataset

All three subreddits cover **Aug 2022 – Jul 2025 (three admission cycles)**. Cycle numbering is chronological: cycle 1 = 2022, cycle 2 = 2023, cycle 3 = 2024. Raw JSONLs are not in git due to size; pre-cleaned MBA data is included as gzip. r/GradAdmissions and r/MSCS raw must be obtained separately. See [Quickstart](docs/quickstart.md) for details.

---

## References

- Low, D. M., et al. (2020). *Natural language processing reveals vulnerable mental health support patterns in a COVID-19 crisis forum.* JMIR Mental Health. — basis for the SVM `mh_score` design.
- Brodersen, K. H., et al. (2015). *Inferring causal impact using Bayesian structural time-series models.* Annals of Applied Statistics. — Causal Impact (NB08).
- Lewis, M., et al. (2020). *BART.* ACL. — `facebook/bart-large-mnli` used for zero-shot anchor NLI (NB03).
- Hutto, C. J., & Gilbert, E. E. (2014). *VADER.* ICWSM. — baseline comparison only.
- Arctic Shift API: https://github.com/ArthurHeitmann/arctic_shift — historical Reddit data source.
