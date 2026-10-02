"""FastAPI backend for the cardio-risk dashboard.

Run from inside backend/:   uvicorn api:app --reload --port 8000
Interactive docs: http://localhost:8000/docs
Env: CORS_ORIGINS=comma,separated,origins  (default: Vite dev server)
"""
import json
import os
import sys
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

import explain
from common import (CONSTANT_COLS, LEAKAGE_COLS, MODELS_DIR, TARGETS,
                    encode_features, load_raw, make_targets)

DISCLAIMER = ("Decision-support / educational use only. Predictions are statistical "
              "estimates and are NOT a substitute for formal diagnostic imaging "
              "(e.g. coronary angiography) or clinical judgement.")
VESSELS = ["LAD", "LCX", "RCA"]
MAX_TOP_K = 30

# ------------------------------------------------------------------ data + schema (built once)
RAW = load_raw()
INPUT_COLS = [c for c in RAW.columns if c not in LEAKAGE_COLS + CONSTANT_COLS]
LABELS = {k: v.tolist() for k, v in make_targets(RAW).items()}   # was recomputed per request


def build_schema() -> dict:
    schema = {}
    for c in INPUT_COLS:
        s = RAW[c]
        if pd.api.types.is_numeric_dtype(s):
            is_int = pd.api.types.is_integer_dtype(s)
            med = float(s.median())
            schema[c] = {"type": "numeric", "integer": is_int,
                         "min": float(s.min()), "max": float(s.max()),
                         "default": int(round(med)) if is_int else med}
        else:
            schema[c] = {"type": "categorical",
                         "options": sorted(s.astype(str).unique().tolist()),
                         "default": str(s.mode().iloc[0])}
    return schema


SCHEMA = build_schema()


def build_row(features: dict[str, Any]):
    unknown = [k for k in features if k not in SCHEMA]
    if unknown:
        raise HTTPException(422, f"Unknown feature(s): {unknown}. See GET /schema.")
    values = {c: SCHEMA[c]["default"] for c in INPUT_COLS}
    warnings = []
    for k, v in features.items():
        spec = SCHEMA[k]
        if spec["type"] == "numeric":
            try:
                x = float(v)
            except (TypeError, ValueError):
                raise HTTPException(422, f"'{k}' must be numeric, got {v!r}")
            if not spec["min"] <= x <= spec["max"]:
                warnings.append(f"{k}={x} is outside the training range "
                                f"[{spec['min']}, {spec['max']}]; prediction may be unreliable.")
            values[k] = int(round(x)) if spec["integer"] else x
        else:
            match = next((o for o in spec["options"] if o.lower() == str(v).lower()), None)
            if match is None:
                raise HTTPException(422, f"'{k}' must be one of {spec['options']}, got {v!r}")
            values[k] = match
    row = pd.DataFrame([values])
    for c in INPUT_COLS:
        row[c] = row[c].astype(RAW[c].dtype)
    return row, [c for c in INPUT_COLS if c not in features], warnings


# ------------------------------------------------------------------ risk helpers
def risk_level(p: float) -> str:
    return "low" if p < 0.33 else "moderate" if p < 0.66 else "high"


def risk_color(p: float) -> str:
    g, a, r = (34, 197, 94), (250, 204, 21), (239, 68, 68)
    lo, hi, t = (g, a, p * 2) if p < 0.5 else (a, r, (p - 0.5) * 2)
    return "#%02x%02x%02x" % tuple(round(lo[i] + (hi[i] - lo[i]) * t) for i in range(3))


def summarize(res: dict) -> dict:
    p = res["probability"]
    return {**res, "percent": round(p * 100, 1), "level": risk_level(p), "color": risk_color(p)}


# ------------------------------------------------------------------ caches
_PRED, _PRED_LOCK = OrderedDict(), threading.Lock()


def predict_all(row: pd.DataFrame) -> dict:
    """Prediction + SHAP for every target; cached per unique input (top MAX_TOP_K factors)."""
    key = tuple(row.iloc[0].tolist())
    with _PRED_LOCK:
        if key in _PRED:
            _PRED.move_to_end(key)
            return _PRED[key]
    X = encode_features(row)
    out = {t: summarize(explain.explain_patient(t, X, top_k=MAX_TOP_K)) for t in TARGETS}
    with _PRED_LOCK:
        _PRED[key] = out
        while len(_PRED) > 256:
            _PRED.popitem(last=False)
    return out


_FILES = {}


def cached_file(path: Path, loader):
    """Re-read a results file only when it changes on disk."""
    mtime = path.stat().st_mtime
    hit = _FILES.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    val = loader(path)
    _FILES[path] = (mtime, val)
    return val


# ------------------------------------------------------------------ app
@asynccontextmanager
async def lifespan(app: FastAPI):
    for t in TARGETS:                      # warm models + SHAP explainers + global importance
        explain._load(t)
        explain.global_importance(t)
    yield


app = FastAPI(title="Cardio Risk API", version="1.1", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CORSMiddleware, allow_methods=["*"], allow_headers=["*"],
                   allow_origins=os.environ.get("CORS_ORIGINS",
                                                "http://localhost:5173,http://127.0.0.1:5173").split(","))


class PredictRequest(BaseModel):
    features: dict[str, Any] = Field(default_factory=dict,
        description="Raw feature values keyed by column name. Missing ones use dataset median/mode.")
    top_k: int = Field(10, ge=1, le=MAX_TOP_K)


@app.get("/health")
def health():
    return {"status": "ok", "models": TARGETS}


@app.get("/schema")
def schema():
    return {"features": SCHEMA, "disclaimer": DISCLAIMER}


@app.post("/predict")
def predict(req: PredictRequest):
    row, defaults_used, warnings = build_row(req.features)
    full = predict_all(row)
    out = {t: {**r, "factors": r["factors"][:req.top_k]} for t, r in full.items()}
    cad = out["CAD"]
    return {
        "cad": {**cad, "prediction": "CAD" if cad["probability"] >= 0.5 else "Normal"},
        "vessels": {v: out[v] for v in VESSELS},
        "defaults_used": defaults_used,
        "warnings": warnings,
        "disclaimer": DISCLAIMER,
    }


@app.get("/patients")
def patient_count():
    return {"count": len(RAW)}


@app.get("/patients/{idx}")
def get_patient(idx: int):
    if not 0 <= idx < len(RAW):
        raise HTTPException(404, f"Index must be 0..{len(RAW) - 1}")
    r = RAW.iloc[idx]
    conv = lambda v: v.item() if hasattr(v, "item") else v
    return {"index": idx,
            "features": {c: conv(r[c]) for c in INPUT_COLS},
            "ground_truth": {k: int(v[idx]) for k, v in LABELS.items()}}


@app.get("/importance/{target}")
def importance(target: str, top_k: int = Query(15, ge=1, le=50)):
    target = target.upper()
    if target not in TARGETS:
        raise HTTPException(404, f"target must be one of {TARGETS}")
    g = explain.global_importance(target).head(top_k)
    return {"target": target,
            "features": [{"feature": f, "mean_abs_shap": round(float(v), 4)} for f, v in g.items()]}


@app.get("/metrics")
def metrics():
    cv_path, best_path = MODELS_DIR / "cv_results.csv", MODELS_DIR / "best_models.json"
    if not cv_path.exists() or not best_path.exists():
        raise HTTPException(404, "Run backend/train.py first.")
    return {"cv_results": cached_file(cv_path, lambda p: pd.read_csv(p).to_dict(orient="records")),
            "best_models": cached_file(best_path, lambda p: json.loads(p.read_text())),
            "cv": "5-fold stratified, repeated 3x", "disclaimer": DISCLAIMER}


@app.get("/evaluation")
def evaluation():
    p = MODELS_DIR / "eval_extra.json"
    if not p.exists():
        raise HTTPException(404, "Run backend/evaluate.py first.")
    return cached_file(p, lambda q: json.loads(q.read_text()))