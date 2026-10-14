import joblib, streamlit as st
from pathlib import Path
from scorer import recommend, band

st.set_page_config(page_title="Ad Copy Scorer", page_icon="📈")
st.title("📈 Ad Copy Scorer")
st.caption("Paste newsletter ad copy. Get a relative performance score and suggested edits. Internal analytics project.")
mp = Path(__file__).parent / "model.joblib"
if not mp.exists():
    st.error("No model found. Run `python train.py --csv newsletter.csv` (or `--demo`) first."); st.stop()
b = joblib.load(mp)
if b["demo"]: st.warning("Running on SYNTHETIC demo data. Scores are not real Morning Brew results.")

text = st.text_area("Ad copy", height=220, placeholder="Paste your ad text here...")
if st.button("Score it", type="primary") and text.strip():
    s, f, recs = recommend(b, text)
    c1, c2 = st.columns(2); c1.metric("Relative score (0-100)", f"{s:.0f}"); c2.metric("Band", band(s))
    st.progress(min(int(s), 100))
    st.subheader("Suggested edits")
    if recs:
        for label, d in recs: st.markdown(f"- {label}  **(≈ +{d} pts)**")
    else: st.write("No single change clearly improves the score. Nice.")
    with st.expander("Detected features"): st.json(f)
st.divider()
st.caption(f"Model: ridge regression on log-CTR, trained on {b['n']:,} ads, cross-validated R² = {b['cv_r2']:.2f}. "
           "Text explains only part of CTR (placement, audience and advertiser matter too). Suggestions are "
           "associations, not proven causes, and are a starting point: a human writes the final copy.")
