"""Synthetic stand-in for newsletter.csv so the app runs without real data.
Planted effects are loosely based on the findings; NOT real Morning Brew numbers."""
import numpy as np, pandas as pd

def make_demo(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    subj = ["our new analytics platform", "the AI assistant for finance teams", "a smarter way to travel",
            "this cold brew subscription", "your next career move", "the app that tracks your savings"]
    hooks = ["Meet", "Introducing", "Tired of spreadsheets? Try", "Discover", "Looking for", "Say hello to"]
    filler = ("teams use it every day to save time and make better decisions with clear simple tools "
              "built for busy people who want results without the hassle").split()
    rows = []
    for _ in range(n):
        wc = int(np.clip(rng.normal(105, 45), 15, 260))
        body = " ".join(rng.choice(filler, size=wc))
        txt = f"{rng.choice(hooks)} {rng.choice(subj)}. {body}"
        if rng.random() < .45: txt += f" Over {rng.integers(2,95)}% of users saw gains."
        if rng.random() < .5: txt += " " + rng.choice(["Sign up today.", "Get started free.", "Learn more.", "Join now."])
        if rng.random() < .15: txt = "Quick update: " + txt
        if rng.random() < .2: txt += "!" * int(rng.integers(1, 4))
        rows.append(txt)
    df = pd.DataFrame({"FULL_AD_TEXT": rows})
    from features import build_matrix
    X = build_matrix(df.FULL_AD_TEXT)
    wc = X.word_count
    logit = (-7.4 - 0.0000*wc - 0.006*X.wc_below_100 - 0.008*X.wc_above_140
             + 0.30*X.has_number + 0.18*X.has_cta - 0.12*X.exclaims.clip(0,3)
             - 0.15*X.colon_headline + 0.10*X.has_question + rng.normal(0, 0.55, len(X)))
    df["TOTAL_AD_CTR_WITHIN_7_DAYS_HUMAN_CLICKS"] = np.exp(logit)
    return df
