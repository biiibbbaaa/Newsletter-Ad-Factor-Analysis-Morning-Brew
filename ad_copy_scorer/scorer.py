"""Score ad copy and generate counterfactual recommendations."""
import numpy as np, pandas as pd
from features import extract, refresh_hinges, FEATURES

def _pred(bundle, f):
    return float(bundle["model"].predict(pd.DataFrame([f])[FEATURES])[0])

def _pct(bundle, p):
    return 100.0 * np.searchsorted(bundle["ref"], p) / len(bundle["ref"])

def band(score):
    return ("Top 20%" if score >= 80 else "Above average" if score >= 60 else
            "Average" if score >= 40 else "Below average" if score >= 20 else "Bottom 20%")

def score(bundle, text):
    f = extract(text); p = _pred(bundle, f); return _pct(bundle, p), f, p

def recommend(bundle, text, max_n=5):
    s0, f, p0 = score(bundle, text)
    wc = f["word_count"]; cands = []
    def trial(label, change):
        g = refresh_hinges({**f, **change}); s1 = _pct(bundle, _pred(bundle, g)); d = s1 - s0
        if d >= 1.0: cands.append((d, label))
    if wc < 100: trial(f"Lengthen toward ~100-140 words (now {wc}); aim for about {100 - wc} more words.", {"word_count": 100})
    if wc > 140: trial(f"Trim toward ~100-140 words (now {wc}); cut about {wc - 140} words.", {"word_count": 140})
    if not f["has_number"]: trial("Add a concrete number or stat (digit).", {"has_number": 1, "digit_count": 2})
    if not f["has_cta"]: trial("Add a clear call to action (e.g. 'Get started', 'Sign up').", {"has_cta": 1})
    if f["exclaims"] > 1: trial("Tone down exclamation points.", {"exclaims": 0})
    if f["colon_headline"]: trial("Avoid a 'Label: text' colon opener.", {"colon_headline": 0})
    if not f["has_question"]: trial("Try opening with a question.", {"has_question": 1})
    cands.sort(reverse=True)
    return s0, f, [(l, round(float(d), 1)) for d, l in cands[:max_n]]
