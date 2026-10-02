"""SHAP explanations per target. Works for linear and tree models.

Contributions are computed on the fitted pipeline and summed back from one-hot
columns to the original clinical features with a precomputed aggregation matrix.
Units: log-odds for LogReg/XGBoost, probability for RandomForest. Use the sign
(risk up/down) and the relative share, not the raw value across models.
"""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import shap

from common import CAT_COLS, MODELS_DIR, TARGETS, encode_features, load_raw

_cache = {}


def _dense(a):
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)


def _val(v):
    """Numeric when possible; categorical columns (e.g. BBB = 'LBBB') stay strings."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def _load(target):
    if target in _cache:
        return _cache[target]
    b = joblib.load(MODELS_DIR / f"{target}.joblib")
    pre, clf = b["pipeline"].named_steps["pre"], b["pipeline"].named_steps["clf"]
    Xt = _dense(pre.transform(encode_features(load_raw())[b["features"]]))
    if b["model_name"].startswith("LogReg"):
        explainer = shap.LinearExplainer(clf, shap.maskers.Independent(Xt, max_samples=len(Xt)))
    else:
        explainer = shap.TreeExplainer(clf)
    # transformed column -> original clinical feature, as one (n_transformed x n_original) matrix
    origin = []
    for n in pre.get_feature_names_out():
        n = n.split("__", 1)[1]
        origin.append(next((c for c in CAT_COLS if n.startswith(c + "_")), n))
    feats = list(dict.fromkeys(origin))
    M = np.zeros((len(origin), len(feats)))
    for i, o in enumerate(origin):
        M[i, feats.index(o)] = 1.0
    _cache[target] = dict(bundle=b, pre=pre, explainer=explainer, feats=feats, M=M, Xt=Xt)
    return _cache[target]


def _positive_class(sv):
    if isinstance(sv, list):          # older shap: [neg, pos]
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    return sv[..., 1] if sv.ndim == 3 else sv


def explain_patient(target, patient: pd.DataFrame, top_k=10):
    """patient: 1-row DataFrame of ENCODED features (see common.encode_features)."""
    m = _load(target)
    b = m["bundle"]
    row = patient[b["features"]]
    prob = float(b["calibrated"].predict_proba(row)[0, 1])
    sv = _positive_class(m["explainer"].shap_values(_dense(m["pre"].transform(row))))
    contrib = pd.Series(np.asarray(sv)[0] @ m["M"], index=m["feats"])
    total = contrib.abs().sum() or 1.0
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:top_k]
    return {
        "target": target,
        "probability": round(prob, 4),
        "model": b["model_name"],
        "factors": [
            {"feature": f, "value": _val(row.iloc[0][f]),
             "shap": round(float(v), 4),
             "direction": "raises risk" if v > 0 else "lowers risk",
             "share_pct": round(float(100 * abs(v) / total), 1)}
            for f, v in top.items()
        ],
    }


@lru_cache(maxsize=None)
def global_importance(target):
    """Mean |SHAP| per original feature over the whole dataset (computed once)."""
    m = _load(target)
    sv = _positive_class(m["explainer"].shap_values(m["Xt"]))
    return pd.DataFrame(sv @ m["M"], columns=m["feats"]).abs().mean().sort_values(ascending=False)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = load_raw()
    X = encode_features(df)
    print("Example patient 0:")
    for t in TARGETS:
        r = explain_patient(t, X.iloc[[0]], top_k=5)
        print(f"\n{t}: p={r['probability']} ({r['model']})")
        for f in r["factors"]:
            print(f"  {f['feature']:22s} value={f['value']!s:8} {f['shap']:+.3f} {f['direction']} ({f['share_pct']}%)")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, t in zip(axes.ravel(), TARGETS):
        g = global_importance(t).head(10)[::-1]
        ax.barh(g.index, g.values)
        ax.set_title(f"{t}: mean |SHAP| (top 10)")
    plt.tight_layout()
    out = MODELS_DIR / "global_importance.png"
    plt.savefig(out, dpi=140)
    print("\nSaved", out)