# Methodology

This document covers the key design concepts, terminology, and limitations of the study.

---

## Core Concepts

### Anchor Post

A Reddit post identified as a **high-distress negative disclosure** related to graduate admissions. These are the treatment events — posts that expose their authors (and readers) to negative outcomes and anxious language.

A post qualifies as an anchor if:
1. It contains a negative keyword (reject, decline, waitlist, funding lost, anxiety, depressed, stress, falling apart...) **and** scores above the distress threshold (`vader_compound < −0.05` or `vader_neg > 0.10`)
2. **Or** the post's self-labeled outcome is Rejected/Waitlisted and it's above the distress threshold

**Result**: 7,075 anchor posts from 5,471 unique authors.

---

### Exposure

Whether a user was **treated** in a given event week.

- **Exposed**: authored an anchor post that week
- **Unexposed**: active on r/GradAdmissions the same week but didn't author any anchor post

**Why authorship?** The dataset doesn't include `link_id` on comments, so we can't reconstruct who read a specific post. Authorship is a conservative proxy — the author is unambiguously exposed. This means our DiD estimate is a **lower bound** on the true effect of reading a negative post.

---

### Difference-in-Differences (DiD)

A causal inference method that removes the influence of time trends by comparing *changes* across two groups.

**Logic**: If exposed and unexposed users are similar before the event, any divergence afterward can be attributed to the exposure.

**Regression**:
```
mh_score = β₀ + β₁·post + β₂·exposed + β₃·(post × exposed) + β₄·log(n_posts) + ε
```

`β₃` is the DiD estimate — the effect of exposure, net of time trends. Positive = exposure increases distress.

**Why not just compare before/after?** Distress naturally rises during peak admissions season. DiD removes this by using the unexposed group as a counterfactual experiencing the same time trends without the exposure.

---

### Parallel Trends Assumption

DiD's key assumption: absent treatment, the exposed and unexposed groups would have followed the same distress trajectory.

We verify this with the **event study plot** — plotting distress at each of the four observation weeks (−2, −1, +1, +2). The lines should be close and parallel in the pre-period and diverge only after the event. Our plots confirm this.

---

### Propensity Score Matching

Because exposed users (those who wrote distressed posts) may already be more anxious than average, a direct comparison would pick up baseline differences rather than the causal effect. Matching corrects for this.

**Process**:
1. For each (user, event_week), compute pre-event features: mean mh_score, total post count
2. Fit logistic regression predicting `exposed` from these features → **propensity score** (probability of being exposed given pre-treatment characteristics)
3. Match each exposed user to the unexposed user with the closest propensity score (caliper = 0.05)
4. Only matched pairs enter the DiD regression

**Result**: 2,843 matched pairs from ~2,844 eligible exposed observations.

---

### mh_score

The primary outcome variable. A continuous score ∈ (0, 1) measuring how closely a user's weekly writing resembles posts from mental health subreddits.

```
mh_score = mean(anx_score, dep_score, str_score)
```

Each component is an SVM classifier's `decision_function` output passed through a sigmoid. Aggregated to `mean_mh_score` at the user-week level.

**Validity check**: mh_score correctly ranks posts by self-reported outcome:
- Rejected (0.451) > Waitlisted (0.446) > Accepted (0.401)

---

### Community Breadth

Number of distinct subreddits a user posted or commented in during the study window, excluding r/GradAdmissions and their own profile sub. Log-transformed (`log(1 + breadth)`) in models due to right skew.

**Theoretical role**: proxy for social network diversity. Wider community engagement was predicted to buffer distress (stress-buffering hypothesis). This was not supported — see [Results](results.md).

---

### VADER vs SVM Classifiers

| | VADER | SVM (mh_score) |
|--|-------|----------------|
| Type | Rule-based lexicon | Trained classifier |
| Training | Hand-crafted word list | 14,000 Reddit posts from mental health subs |
| Fires on | Any negative word | Language *patterns* of distressed writing |
| Problem | "Don't stress about it" scores negative | — |
| RQ1 p-value | 0.077 (not significant) | < 0.0001 |

VADER is a useful first pass but lacks sensitivity for domain-specific distress language. The SVM approach follows Low et al. (2020), who used the same subreddit-as-label training strategy.

---

### HC3 Robust Standard Errors

Standard OLS assumes constant variance in residuals (homoskedasticity). Social media data violates this — some users post much more than others. HC3 corrects for this, giving reliable p-values without requiring distributional assumptions.

---

## Limitations

1. **Exposure proxy**: We measure authorship, not readership. Many truly exposed users (those who read but didn't write anchor posts) are coded as unexposed. This **attenuates** the estimated effect — the true impact of reading a negative post may be larger.

2. **Propensity score matching on observables**: Matching controls for pre-event distress and activity, but unobserved differences (personality, resilience, support networks) remain as potential confounders.

3. **Text-based distress**: mh_score measures *how someone writes*, not their direct psychological state. Users may be distressed without writing distressed posts, or write distressed posts strategically (seeking support or validation).

4. **Community breadth amplification (RQ2)**: The positive moderation coefficient is unexpected. Community breadth as a measure of social support is crude — it counts subreddits without regard for their relevance, quality, or reciprocity. A user active in r/gaming and r/cooking has "high breadth" but no relevant support for admissions anxiety.

5. **Reddit-specific generalizability**: r/GradAdmissions is a self-selected community of applicants willing to post publicly. Results may not generalize to all graduate school applicants.

6. **Data leakage prevention**: SVM training data (Jan 2022 – Jul 2023) predates the study window (Aug 2023 – Jul 2025) to avoid contamination.

---

## References

- Low, D.M., et al. (2020). Natural language processing reveals vulnerable mental health support patterns in a COVID-19 crisis forum. *JMIR Mental Health*, 7(6), e21236.
- Hutto, C.J. & Gilbert, E.E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *ICWSM*.
- Joachims, T. (1998). Text categorization with support vector machines. *ECML*.
- MacKinnon, J.G. & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators. *Journal of Econometrics*, 29(3), 305–325.
