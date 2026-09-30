"""Shared data loading + preprocessing (used by training and, later, the API)."""
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


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


def make_targets(df: pd.DataFrame) -> dict:
    return {
        "CAD": (df["Cath"] == "CAD").astype(int),
        "LAD": (df["LAD"] == "Stenotic").astype(int),
        "LCX": (df["LCX"] == "Stenotic").astype(int),
        "RCA": (df["RCA"] == "Stenotic").astype(int),
    }


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
    return X


def make_preprocessor(X: pd.DataFrame, scale: bool = True) -> ColumnTransformer:
    num_cols = [c for c in X.columns if c not in CAT_COLS]
    num_tf = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num_tf, num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])