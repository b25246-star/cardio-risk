"""Shared data loading + preprocessing (used by training and the API)."""
import os
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "extention of Z-Alizadeh sani dataset.csv"
MODELS_DIR = ROOT / "models"

# Columns removed from the inputs: the targets (leakage) + a constant column
LEAKAGE_COLS = ["LAD", "LCX", "RCA", "Cath"]
CONSTANT_COLS = ["Exertional CP"]
CAT_COLS = ["BBB", "Region RWMA"]
TARGETS = ["CAD", "LAD", "LCX", "RCA"]

# Engineered features (computed from the clinical inputs only - no target information).
# Set env CARDIO_FE=0 to switch them off, e.g. to measure their effect with evaluate.py.
USE_FE = os.environ.get("CARDIO_FE", "1") == "1"
RISK_FACTORS = ["HTN", "DM", "Current Smoker", "EX-Smoker", "FH", "DLP", "Obesity"]
ECG_FLAGS = ["Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH", "Poor R Progression"]
ENGINEERED = ["LDL_HDL", "TG_HDL", "RiskFactorCount", "ECGAbnormalCount"]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


def make_targets(df: pd.DataFrame) -> dict:
    return {
        "CAD": (df["Cath"] == "CAD").astype(int),
        "LAD": (df["LAD"] == "Stenotic").astype(int),
        "LCX": (df["LCX"] == "Stenotic").astype(int),
        "RCA": (df["RCA"] == "Stenotic").astype(int),
    }


def add_engineered(X: pd.DataFrame) -> pd.DataFrame:
    """Ratios and counts built from already-encoded numeric inputs. Skips any that can't be built."""
    X = X.copy()
    if {"LDL", "HDL"} <= set(X.columns):
        X["LDL_HDL"] = X["LDL"] / X["HDL"].clip(lower=1)
    if {"TG", "HDL"} <= set(X.columns):
        X["TG_HDL"] = X["TG"] / X["HDL"].clip(lower=1)
    rf = [c for c in RISK_FACTORS if c in X.columns]
    if rf:
        X["RiskFactorCount"] = X[rf].sum(axis=1)
    ecg = [c for c in ECG_FLAGS if c in X.columns]
    if ecg:
        X["ECGAbnormalCount"] = X[ecg].sum(axis=1)
    return X


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw rows into the model input table (no targets)."""
    X = df.drop(columns=[c for c in LEAKAGE_COLS + CONSTANT_COLS if c in df.columns]).copy()
    X["Sex"] = (X["Sex"] == "Male").astype(int)
    X["VHD"] = X["VHD"].map({"N": 0, "mild": 1, "Moderate": 2, "Severe": 3})
    X["Region RWMA"] = X["Region RWMA"].astype(str)
    X["BBB"] = X["BBB"].astype(str)
    for c in X.columns:
        if c in CAT_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = (X[c] == "Y").astype(int)
    return add_engineered(X) if USE_FE else X


def make_preprocessor(X: pd.DataFrame, scale: bool = True) -> ColumnTransformer:
    num_cols = [c for c in X.columns if c not in CAT_COLS]
    num_tf = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num_tf, num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])