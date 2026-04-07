# You're Freaking Out and It Scares Me Too:
## Understanding How Exposure to Negative Discourse on r/GradAdmissions Affects Anxiety

Graduate admissions is stressful. But does *reading* other people's rejection posts make it worse? This project uses causal inference to find out — measuring whether exposure to high-distress posts on r/GradAdmissions increases mental health distress in other users, and whether having a wider online social presence buffers that effect.

**Short answer**: The effect is small and directionally positive (exposure → slightly higher distress scores), but does not reach conventional significance thresholds in the v2 pipeline. The signal is consistent across two independent methods (PSM-DiD and Bayesian Causal Impact) and two admissions cycles.

---

## Key Findings (v2 Pipeline)

| | |
|--|--|
| **RQ1 — NB06** | PSM-matched DiD: exposed users show ~+0.007–0.010 increase in mh_score vs. matched unexposed; effects are directionally consistent but not statistically significant (p ≈ 0.21–0.48 per cycle) |
| **RQ1 — NB08** | Alternative pre-period (Sep–Nov): pooled DiD = +0.0076, p = 0.067 (borderline). CausalImpact: +2.1% (cycle 1) and +2.7% (cycle 2) relative effect on exposed group |
| **RQ2** | Community breadth moderation tested; directional coefficient varies by specification |
| **Coverage** | NB06 pre-period (August): ~6.5% user coverage, ~155 matched pairs/cycle. NB08 pre-period (Sep–Nov): 36.5% coverage, 668 matched pairs — 4× improvement |

---

## How It Works

```
Raw Reddit data (r/GradAdmissions, Aug 2023–Jul 2025)
        ↓
    NB01: Corpus cleaning (post_id linking via link_id)
        ↓
    NB02: SVM mental-health classifiers (anxiety, depression, stress)
        ↓
    NB03: Anchor event identification + thread-level exposure via link_id
        ↓
    NB04: Panel scoring (pre / post windows)
    NB05: Community breadth (Arctic Shift API)
        ↓
    NB06: PSM + DiD + post-level DiD + dose-response  ← main results
    NB08: Alternative pre-period + Causal Impact       ← robustness / power fix
```

The distress measure is a composite SVM score (`mh_score`) trained on r/anxiety, r/depression, and r/stress posts — more sensitive than VADER alone for domain-specific distress language.

---

## Repository Structure

```
├── notebooks/
│   ├── 00_exploratory_topic_sentiment.ipynb   # EDA only
│   ├── 01_clean_corpus.ipynb                  # Corpus cleaning + post_id mapping
│   ├── 02_train_classifiers.ipynb             # SVM classifier training
│   ├── 03_exposure_labels_v2.ipynb            # Anchor posts + exposure labels
│   ├── 04_panel_scores.ipynb                  # Pre/post panel scoring + post-level scores
│   ├── 05_collect_community_breadth.ipynb     # Arctic Shift API breadth collection
│   ├── 06_did_analysis_v2.ipynb               # Main: PSM + DiD + post-level + dose-response
│   ├── 07_did_analysis_vader_baseline.ipynb   # VADER baseline (optional comparison)
│   └── 08_alt_analysis.ipynb                  # Sep-Nov pre-period + Causal Impact
│
├── data/
│   ├── r_gradadmissions_posts.jsonl           # Raw posts (not in repo — large)
│   ├── r_gradadmissions_comments.jsonl        # Raw comments (not in repo — large)
│   └── processed_v2/                          # All intermediate parquets and clean JSONLs
│
├── models/                                    # Trained SVM classifiers (.joblib)
├── figures/                                   # All output plots (.png)
└── docs/                                      # Documentation
```

---
 
## Documentation

| Doc | What's in it |
|-----|-------------|
| [LLM Context](docs/LLM_CONTEXT.md) | Dense codebase snapshot for AI assistants — start here |
| [Quickstart](docs/quickstart.md) | How to set up and run the full pipeline from scratch |
| [Flow](docs/flow.md) | Step-by-step walkthrough with flowcharts |
| [Pipeline](docs/pipeline.md) | What each notebook does, inputs/outputs, runtimes |
| [Results](docs/results.md) | Full results tables, figures, and interpretation |
| [Methodology](docs/methodology.md) | Terminology glossary, design choices, limitations |

---

## Dataset

469,163 posts and comments from r/GradAdmissions (Aug 2023 – Jul 2025).
Raw data is not included in this repo due to file size. See [Quickstart](docs/quickstart.md) for how to obtain it.

---

## References

- Low et al. (2020). Natural language processing reveals vulnerable mental health support patterns in a COVID-19 crisis forum. *JMIR Mental Health*.
- Brodersen et al. (2015). Inferring causal impact using Bayesian structural time-series models. *Annals of Applied Statistics*. (CausalImpact)
- Hutto & Gilbert (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *ICWSM*.
- Arctic Shift API: https://github.com/ArthurHeitmann/arctic_shift
