# Ad Copy Scorer

Paste newsletter ad copy and get a **relative performance score (0-100)**, a band (Top 20% ... Bottom 20%), and
prioritized edit suggestions (word count, numbers, CTA, tone, etc.). Built on the Newsletter Ad Factor Analysis.

## How it works
- `features.py`: text features from the analysis (word count with a 100-140 sweet-spot hinge, digits, CTA, VADER sentiment, exclamations, colon-headline opener, emoji).
- `train.py`: same cleaning as the notebook (Tukey 1.5xIQR trim), fits a Ridge regression on log(CTR), benchmarked against gradient boosting via 5-fold CV.
- `scorer.py`: recommendations by **counterfactual re-scoring**: change one feature, re-predict, report the score delta.
- `app.py`: Streamlit UI. Shows a percentile score rather than an absolute CTR (consistent with the privacy-modified charts).

## Run
```bash
pip install -r requirements.txt
python train.py --demo                      # synthetic data, just to try the app
python train.py --csv path/to/newsletter.csv  # real data, run locally
streamlit run app.py
```

## Data privacy
`newsletter.csv` and `model.joblib` are git-ignored on purpose. Do not commit raw data or a model trained on it to this public repo. The demo data is synthetic and its numbers are not real results.

## Limitations
Text explains only part of CTR variance (placement, audience, advertiser matter). Suggestions are associations, not causal. A person writes the final copy.
