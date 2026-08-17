# Newsletter Ad Factor Analysis – Methodology Brief

**Project:** What makes an ad clickable? Analyzing ad content CTR drivers
**Prepared by:** Biba Schwendt, Data & Analytics Intern at Morning Brew
**Ad data date range:** August 2024 – February 2026

## Overview

Morning Brew runs hundreds of newsletter ads, some perform well, others don't. This project analyzed thousands of real newsletter ad records (primary, secondary, and tertiary ad types across all newsletters) to identify the patterns behind high performance and translate them into guidance the content team can use.

## Approach

The analysis followed a four-stage process:

1. **Data cleaning** — Removed outliers to keep comparisons fair.
2. **Exploratory data analysis** — Moved from simple correlations to more robust statistical models.
3. **Hypothesis generation** — Identified six features of ad copy that might plausibly drive clicks.
4. **Hypothesis testing** — Tested each feature against actual click-through rate (CTR) data.

## Questions Explored

1. Does ad length affect click-through rate?
2. Do specific formatting choices (numbers, emojis, calls-to-action) affect performance?
3. Does more positive sentiment lead to more clicks?
4. Do emotionally extreme ads outperform neutral ones?
5. Are there specific words or phrases that consistently show up in high-performing ads?
6. Does running the same ad multiple times hurt performance?

## How the Data Was Handled

**Outlier removal:** A small number of ads had click rates extreme enough to distort results. These were trimmed using a standard statistical method (the Tukey method), which clips the extreme ends of the distribution, a known trade-off, since it also clips some genuine top performers.

**Placement as a control:** Where an ad sits in a newsletter strongly affects how many people see it. Placement was held constant across every hypothesis test so that comparisons were always copy-against-copy, not copy-against-position.

**Models used.**

- Linear regression — the primary tool, used to isolate how much a single factor (e.g., a word or feature) is associated with CTR while holding other variables constant.
- Cross-validated R² — used as the honest measure of predictive accuracy, since it tests the model against data it hasn't seen (rather than relying on in-sample fit alone).
- Random forest — tested as a comparison model; in most cases linear regression performed comparably or better, suggesting the underlying relationships are largely linear.

**Sentiment and tone tools used.**

- TextBlob and VADER — two rule-based sentiment scoring tools (scale of -1 to +1). VADER was originally built for social media and weighs punctuation more heavily. The two tools agreed only about a quarter of the time, which is why both were retained — agreement between them signals a more reliable finding.
- Hugging Face transformers (BERT) — a pre-trained language model run locally, fine-tuned on a six-label emotion dataset (joy, sadness, love, anger, fear, surprise), used to classify the dominant emotional tone of each ad.

## Note on Rigor

Throughout, data integrity was protected by keeping separate raw and working copies of the dataset per hypothesis, using left joins to avoid discarding valid records, and cross-validating results to guard against inflated in-sample accuracy.

---

*This brief covers methodology only. Specific findings and performance figures are available in the full deck only.*
