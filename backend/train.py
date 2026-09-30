"""Compare models per target with repeated stratified CV, then save the best."""
import warnings
import joblib, json
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from common import (MODELS_DIR, ROOT, TARGETS, encode_features, load_raw,
                    make_preprocessor, make_targets)

warnings.filterwarnings("ignore")
SEED = 42
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def candidates(X, pos_weight):
    """name -> Pipeline. Trees don't need scaling."""
    return {
        "LogReg(C=1)": Pipeline([("pre", make_preprocessor(X)),
            ("clf", LogisticRegression(C=1, max_iter=3000, class_weight="balanced"))]),
        "LogReg(C=0.1)": Pipeline([("pre", make_preprocessor(X)),
            ("clf", LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))]),
        "RandomForest": Pipeline([("pre", make_preprocessor(X, scale=False)),
            ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                class_weight="balanced", random_state=SEED, n_jobs=-1))]),
        "XGBoost": Pipeline([("pre", make_preprocessor(X, scale=False)),
            ("clf", XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                scale_pos_weight=pos_weight, eval_metric="logloss",
                random_state=SEED, n_jobs=1))]),
    }


def main():
    df = load_raw()
    X = encode_features(df)
    y = make_targets(df)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    MODELS_DIR.mkdir(exist_ok=True)

    rows = []
    for t in TARGETS:
        pos_weight = (y[t] == 0).sum() / (y[t] == 1).sum()
        for name, pipe in candidates(X, pos_weight).items():
            s = cross_validate(pipe, X, y[t], cv=cv, scoring=SCORING, n_jobs=1)
            row = {"target": t, "model": name}
            for m in SCORING:
                row[m] = round(s[f"test_{m}"].mean(), 3)
                row[m + "_sd"] = round(s[f"test_{m}"].std(), 3)
            rows.append(row)
            print(f"{t:4s} {name:14s} AUC={row['roc_auc']:.3f} F1={row['f1']:.3f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "models" / "cv_results.csv", index=False)

    # Best model per target by ROC-AUC -> fit on all data, calibrate, save
    summary = {}
    for t in TARGETS:
        best = res[res.target == t].sort_values("roc_auc", ascending=False).iloc[0]
        pos_weight = (y[t] == 0).sum() / (y[t] == 1).sum()
        pipe = candidates(X, pos_weight)[best.model]
        fitted = clone(pipe).fit(X, y[t])                     # for SHAP
        calibrated = CalibratedClassifierCV(clone(pipe), method="sigmoid", cv=5).fit(X, y[t])
        joblib.dump({"pipeline": fitted, "calibrated": calibrated,
                     "features": list(X.columns), "model_name": best.model},
                    MODELS_DIR / f"{t}.joblib")
        summary[t] = {"model": best.model, "roc_auc": float(best.roc_auc)}
    json.dump(summary, open(MODELS_DIR / "best_models.json", "w"), indent=2)
    print("\nBest per target:", summary)


if __name__ == "__main__":
    main()