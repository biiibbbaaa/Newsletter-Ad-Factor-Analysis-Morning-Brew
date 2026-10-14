"""Text feature extraction for the ad-copy scorer.

Column names match clean_data_exploration_v2.ipynb:
  FULL_AD_TEXT, TOTAL_AD_CTR_WITHIN_7_DAYS_HUMAN_CLICKS (-> CTR)
"""
import re
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_sia = SentimentIntensityAnalyzer()
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐✅]")
_COLON_HEAD = re.compile(r'^[^\n:]{1,60}:')              # "Headline: body" pattern (H-colon artifact)
# Starter CTA list. Swap for your data-driven cta_words_df terms from notebook H2.
CTA_TERMS = ["sign up", "get started", "try", "learn more", "download", "join",
             "start", "claim", "save", "free", "book", "subscribe", "shop", "get "]

FEATURES = ["word_count", "wc_below_100", "wc_above_140", "has_number", "digit_count",
            "has_cta", "has_question", "exclaims", "emoji_count",
            "colon_headline", "vader", "avg_sentence_len"]

# Features a copywriter can actually change -> used for recommendations.
EDITABLE = ["word_count", "has_number", "has_cta", "exclaims",
            "has_question", "colon_headline"]


def extract(text: str) -> dict:
    t = (text or "").strip()
    words = re.findall(r"\b[\w'’-]+\b", t)
    wc = len(words)
    sents = [s for s in re.split(r'[.!?]+\s', t) if s.strip()]
    low = t.lower()
    return {
        "word_count": wc,
        "wc_below_100": max(0, 100 - wc),   # hinge terms capture the 100-140 sweet spot
        "wc_above_140": max(0, wc - 140),
        "has_number": int(bool(re.search(r"\d", t))),
        "digit_count": len(re.findall(r"\d", t)),
        "has_cta": int(any(c in low for c in CTA_TERMS)),
        "has_question": int("?" in t),
        "exclaims": t.count("!"),
        "emoji_count": len(_EMOJI.findall(t)),
        "colon_headline": int(bool(_COLON_HEAD.match(t))),
        "vader": _sia.polarity_scores(t)["compound"] if t else 0.0,
        "avg_sentence_len": wc / max(len(sents), 1),
    }


def build_matrix(texts) -> pd.DataFrame:
    return pd.DataFrame([extract(t) for t in texts])[FEATURES]


def refresh_hinges(f: dict) -> dict:
    f["wc_below_100"] = max(0, 100 - f["word_count"])
    f["wc_above_140"] = max(0, f["word_count"] - 140)
    return f
