# Cardio Risk 3D — CAD & coronary stenosis prediction with an interactive 3D heart

Multimodal AI Hackathon 2026, Track A. Predicts overall coronary artery disease (CAD) and stenosis of the
LAD, LCX and RCA from clinical, ECG, laboratory and echo features (Z-Alizadeh Sani extension dataset, 303 patients),
maps the predicted probabilities onto a 3D heart, and explains each prediction with SHAP.

> **Disclaimer:** decision support / educational use only. Not a substitute for formal diagnostic imaging or clinical judgement.

## Structure
```
backend/   common.py (data + preprocessing)  train.py  evaluate.py  explain.py  api.py (FastAPI)
frontend/  Vite + React + React Three Fiber (src/App.jsx, Heart.jsx, Eval.jsx; public/heart.glb)
models/    trained models (*.joblib), cv_results.csv, eval_extra.json, best_models.json
data/      extention of Z-Alizadeh sani dataset.csv
```

## Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd frontend; npm install; cd ..
```

## Run
```powershell
# Optional: retrain / re-evaluate (trained models are already in models/)
python backend\train.py          # repeated stratified CV, picks + saves best model per target
python backend\evaluate.py       # nested CV, ROC curves, confusion matrices, operating thresholds

# Terminal 1 - API  (http://127.0.0.1:8000/docs)
uvicorn backend.api:app --reload --port 8000

# Terminal 2 - frontend (open the printed URL, usually http://localhost:5173)
cd frontend; npm run dev
```

## Method
- **Targets:** CAD (`Cath`), and LAD / LCX / RCA "Stenotic". These four columns plus a constant column are **excluded from the inputs** (no target leakage).
- **Preprocessing:** binary Y/N and Sex encoded 0/1, VHD ordinal, BBB and Region RWMA one-hot, numeric features standardised (linear models).
- **Models compared per target:** logistic regression (C=1, C=0.1), random forest, XGBoost; class-balanced; best by ROC-AUC, then probability-calibrated (sigmoid).
- **Validation:** 5-fold stratified CV repeated 3x, plus nested CV (model chosen inside each training fold).
- **Explainability:** SHAP (exact for linear and tree models), contributions summed back to original clinical features.

## Results (nested CV, ROC-AUC, mean ± SD over 15 folds)
| Target | AUC | Deployed model |
|---|---|---|
| CAD | 0.923 ± 0.043 | Logistic regression (C=0.1) |
| LAD | 0.853 ± 0.044 | Random forest |
| LCX | 0.738 ± 0.065 | XGBoost |
| RCA | 0.712 ± 0.038 | Logistic regression (C=0.1) |

Full tables, ROC curves and confusion matrices are in the app's **Model evaluation** tab.

## API
`GET /schema` · `POST /predict` · `GET /patients/{i}` · `GET /importance/{target}` · `GET /metrics` · `GET /evaluation` · `GET /health`

## Limitations
- LCX and RCA models are weak (AUC about 0.71-0.74); at the default threshold they miss over half of true stenoses. The UI marks them "lower confidence".
- Small dataset (303 patients, single centre); no external validation.
- Artery tubes in the 3D view are schematic overlays on a cutaway heart model, not anatomical vessel meshes; the dataset has no lesion location data.
- The form exposes 16 key inputs; all other features default to the dataset median/mode.

## Credits
- Dataset: Z-Alizadeh Sani (extension), UCI Machine Learning Repository.
- 3D heart model: "Beating heart" from Sketchfab — **add author name, license and link here**.