"""Compare models per target with repeated stratified CV, then save the best.

Candidates are fixed presets (several regularisation / depth settings per family) so that
pipeline steps stay named 'pre' and 'clf' (required by explain.py for SHAP).
evaluate.py re-runs the selection INSIDE each training fold (nested CV) to give an honest
estimate, so adding candidates here does not inflate the reported nested numbers.

Run:  python train.py
"""
import warnings
import joblib, json, time
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from common import (ENGINEERED, MODELS_DIR, ROOT, TARGETS, encode_features, load_raw,
                    make_preprocessor, make_targets)

warnings.filterwarnings("ignore")
SEED = 42
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def candidates(X, pos_weight):
    """name -> Pipeline. n_jobs=1 inside models: parallelism is at the CV-fold level."""
    def lin(**kw):
        return Pipeline([("pre", make_preprocessor(X)),
                         ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", **kw))])

    def tree(clf):
        return Pipeline([("pre", make_preprocessor(X, scale=False)), ("clf", clf)])

    def xgb(depth, n):
        return XGBClassifier(n_estimators=n, max_depth=depth, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                             scale_pos_weight=pos_weight, eval_metric="logloss",
                             random_state=SEED, n_jobs=1)

    return {
        "LogReg(C=1)": lin(C=1),
        "LogReg(C=0.1)": lin(C=0.1),
        "LogReg(C=0.03)": lin(C=0.03),
        "LogReg(L1,C=0.3)": lin(C=0.3, penalty="l1", solver="liblinear"),
        "RandomForest": tree(RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, class_weight="balanced",
            random_state=SEED, n_jobs=1)),
        "RandomForest(leaf=5)": tree(RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, max_features=0.3, class_weight="balanced",
            random_state=SEED, n_jobs=1)),
        "ExtraTrees": tree(ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=3, class_weight="balanced",
            random_state=SEED, n_jobs=1)),
        "XGBoost": tree(xgb(3, 300)),
        "XGBoost(d2)": tree(xgb(2, 400)),
    }


def main():
    t0 = time.time()
    df = load_raw()
    X = encode_features(df)
    y = make_targets(df)
    print("Engineered features in use:", [c for c in ENGINEERED if c in X.columns] or "none")
    print(f"{X.shape[1]} input features, {len(X)} patients")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    MODELS_DIR.mkdir(exist_ok=True)

    rows = []
    for t in TARGETS:
        pos_weight = (y[t] == 0).sum() / (y[t] == 1).sum()
        for name, pipe in candidates(X, pos_weight).items():
            s = cross_validate(pipe, X, y[t], cv=cv, scoring=SCORING, n_jobs=-1)
            row = {"target": t, "model": name}
            for m in SCORING:
                row[m] = round(s[f"test_{m}"].mean(), 3)
                row[m + "_sd"] = round(s[f"test_{m}"].std(), 3)
            rows.append(row)
            print(f"{t:4s} {name:22s} AUC={row['roc_auc']:.3f} F1={row['f1']:.3f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "models" / "cv_results.csv", index=False)

    # Best model per target by ROC-AUC (F1 breaks ties) -> fit on all data, calibrate, save
    summary = {}
    for t in TARGETS:
        best = res[res.target == t].sort_values(["roc_auc", "f1"], ascending=False).iloc[0]
        pos_weight = (y[t] == 0).sum() / (y[t] == 1).sum()
        pipe = candidates(X, pos_weight)[best.model]
        fitted = clone(pipe).fit(X, y[t])                     # used for SHAP
        # ensemble=False: one base model + sigmoid calibrator -> smaller file, faster inference
        calibrated = CalibratedClassifierCV(clone(pipe), method="sigmoid", cv=5, ensemble=False).fit(X, y[t])
        joblib.dump({"pipeline": fitted, "calibrated": calibrated,
                     "features": list(X.columns), "model_name": best.model},
                    MODELS_DIR / f"{t}.joblib")
        summary[t] = {"model": best.model, "roc_auc": float(best.roc_auc)}
    json.dump(summary, open(MODELS_DIR / "best_models.json", "w"), indent=2)
    print("\nBest per target:", summary)
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()