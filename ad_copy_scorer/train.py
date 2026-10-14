"""Train the scorer.  python train.py --csv path/to/newsletter.csv   (or --demo)
Mirrors notebook cleaning: Tukey 1.5xIQR trim on CTR, drop blank text.
Target = log(CTR). Primary model = Ridge (explainable); benchmark = HistGradientBoosting.
Writes model.joblib (KEEP LOCAL - do not commit; see .gitignore)."""
import argparse, joblib, numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from features import build_matrix, FEATURES

TEXT, CTR = "FULL_AD_TEXT", "TOTAL_AD_CTR_WITHIN_7_DAYS_HUMAN_CLICKS"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv"); ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default="model.joblib"); a = ap.parse_args()
    if a.demo or not a.csv:
        from demo_data import make_demo; df = make_demo(); demo = True
    else:
        df = pd.read_csv(a.csv); demo = False
    df = df[df[TEXT].fillna("").str.strip() != ""].copy()
    q1, q3 = df[CTR].quantile([.25, .75]); iqr = q3 - q1
    df = df[(df[CTR] >= q1-1.5*iqr) & (df[CTR] <= q3+1.5*iqr) & (df[CTR] > 0)]
    X = build_matrix(df[TEXT]); y = np.log(df[CTR].values)
    cv = KFold(5, shuffle=True, random_state=0)
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
    gb = HistGradientBoostingRegressor(max_depth=3, learning_rate=.05, max_iter=300, random_state=0)
    r2_ridge = cross_val_score(ridge, X, y, cv=cv, scoring="r2").mean()
    r2_gb = cross_val_score(gb, X, y, cv=cv, scoring="r2").mean()
    print(f"rows={len(df)}  CV R^2  ridge={r2_ridge:.3f}  gradient-boost={r2_gb:.3f}")
    ridge.fit(X, y)
    ref = ridge.predict(X)   # reference distribution for percentile scoring
    coefs = pd.Series(ridge[-1].coef_ / ridge[0].scale_, index=FEATURES)  # per raw unit, log-CTR
    joblib.dump({"model": ridge, "ref": np.sort(ref), "coefs": coefs, "cv_r2": float(r2_ridge),
                 "cv_r2_gb": float(r2_gb), "demo": demo, "n": len(df)}, a.out)
    print(coefs.round(4).to_string()); print("saved", a.out, "(DEMO DATA)" if demo else "")
if __name__ == "__main__": main()
