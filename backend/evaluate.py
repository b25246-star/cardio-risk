"""Stricter evaluation, saved to models/eval_extra.json (served by GET /evaluation).

1. Nested CV: model chosen INSIDE each training fold (inner 3-fold AUC), scored on the untouched outer fold.
2. Out-of-fold calibrated probabilities -> ROC curves + confusion matrices.
3. Operating points: default 0.5, sensitivity>=90% (screening) and Youden-J.
Folds run in parallel (joblib). Run:  python evaluate.py      (EVAL_REPEATS=1 for a quick test)
"""
import json, os, warnings
from collections import Counter
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score

from common import MODELS_DIR, TARGETS, encode_features, load_raw, make_targets
from train import SEED, candidates

warnings.filterwarnings("ignore")
REPEATS = int(os.environ.get("EVAL_REPEATS", 3))


def point(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: round(float(a / b), 3) if b else 0.0
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return {"threshold": round(float(thr), 3), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "accuracy": div(tp + tn, len(y)), "precision": prec, "recall": rec,
            "specificity": div(tn, tn + fp), "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0}


def run_fold(X, yt, pw, tr, te):
    cands = candidates(X, pw)
    inner = {n: cross_val_score(p, X.iloc[tr], yt[tr], cv=3, scoring="roc_auc").mean() for n, p in cands.items()}
    best = max(inner, key=inner.get)
    m = CalibratedClassifierCV(clone(cands[best]), method="sigmoid", cv=5, ensemble=False).fit(X.iloc[tr], yt[tr])
    p = m.predict_proba(X.iloc[te])[:, 1]
    return best, roc_auc_score(yt[te], p), p


def main():
    df = load_raw(); X = encode_features(df); y = make_targets(df)
    out = {"repeats": REPEATS, "nested": {}, "roc": {}, "operating_points": {}}
    for t in TARGETS:
        yt = y[t].values; pw = (yt == 0).sum() / (yt == 1).sum()
        jobs = [(r, tr, te) for r in range(REPEATS)
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(X, yt)]
        res = Parallel(n_jobs=-1)(delayed(run_fold)(X, yt, pw, tr, te) for _, tr, te in jobs)
        aucs, picks, oof = [], Counter(), np.zeros(len(yt))
        for (r, tr, te), (best, auc, p) in zip(jobs, res):
            aucs.append(auc); picks[best] += 1
            if r == 0: oof[te] = p
        fpr, tpr, thr = roc_curve(yt, oof)
        keep = np.unique(np.linspace(0, len(fpr) - 1, 60).astype(int))
        out["nested"][t] = {"auc_mean": round(float(np.mean(aucs)), 3), "auc_sd": round(float(np.std(aucs)), 3),
                            "folds": len(aucs), "selected": dict(picks)}
        out["roc"][t] = {"auc": round(float(roc_auc_score(yt, oof)), 3),
                         "fpr": [round(float(v), 3) for v in fpr[keep]], "tpr": [round(float(v), 3) for v in tpr[keep]]}
        ok = thr[tpr >= 0.90]
        t90 = ok[0] if len(ok) else thr[-1]
        out["operating_points"][t] = {"default": point(yt, oof, 0.5), "sens90": point(yt, oof, t90),
                                      "youden": point(yt, oof, thr[np.argmax(tpr - fpr)])}
        print(f"{t}: nested AUC {out['nested'][t]['auc_mean']} +/- {out['nested'][t]['auc_sd']}  picks={dict(picks)}", flush=True)
    json.dump(out, open(MODELS_DIR / "eval_extra.json", "w"), indent=1)
    print("Saved", MODELS_DIR / "eval_extra.json")


if __name__ == "__main__":
    main()