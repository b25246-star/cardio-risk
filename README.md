# CardioRisk 3D

Interactive coronary artery disease (CAD) and vessel-level stenosis risk visualization.
Multimodal AI Hackathon 2026, Track A.

Enter a patient's clinical data, press **Predict**, and watch the LAD, LCX and RCA arteries on an
interactive 3D heart change colour with their predicted stenosis probability. Click an artery to see
which measurements pushed the risk up or down (SHAP).

> **Clinical safety:** predictions are for decision support and educational use only. They are not a
> substitute for formal diagnostic imaging (e.g. coronary angiography) or clinical judgement.

Demo video: [[YouTube URL]] &nbsp;|&nbsp; Documentation: `docs/CardioRisk3D_Documentation.pdf`

## Features

- Calibrated classifiers for overall **CAD** and **LAD / LCX / RCA** stenosis
- No target leakage: `LAD`, `LCX`, `RCA` and `Cath` are never used as inputs
- Repeated stratified CV, **nested CV**, ROC curves, confusion matrices and operating points
- SHAP explanations with reference ranges and each factor's share of the prediction
- 3D heart (React Three Fiber): rotate, zoom, click arteries, colour-coded risk, reset view
- Dashboard with sidebar, grouped inputs, manual **Predict** button and optional live update
- FastAPI backend with schema-driven input form (new features appear automatically)

## Project structure

```
cardio-risk/
├── backend/            FastAPI app + training code
│   ├── api.py          REST API
│   ├── common.py       data loading, encoding, feature engineering
│   ├── train.py        model comparison, calibration, saving
│   ├── evaluate.py     nested CV, ROC, operating points
│   ├── explain.py      SHAP explanations
│   └── requirements.txt
├── frontend/           React + Vite dashboard
│   ├── public/heart_coronary.glb
│   └── src/            App.jsx, Heart.jsx, Eval.jsx, main.jsx, styles.css
├── data/               Z-Alizadeh Sani dataset (CSV)
├── models/             trained .joblib models + cv_results.csv, best_models.json, eval_extra.json
└── docs/               project documentation
```

## Setup

Requirements: Python 3.10+ and Node.js 18+.

Dataset: UCI *Extension of Z-Alizadeh Sani Dataset*. Place the CSV at
`data/extention of Z-Alizadeh sani dataset.csv` (already included if present in this repo).

```bash
# backend
cd backend
python -m venv venv
venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt

# frontend
cd ../frontend
npm install
```

## Run

Pre-trained models are in `models/`, so you can start straight away. Use two terminals.

```bash
# terminal 1: API  (http://127.0.0.1:8000, docs at /docs)
cd backend
uvicorn api:app --reload --port 8000

# terminal 2: dashboard  (http://localhost:5173)
cd frontend
npm run dev
```

If the frontend runs on another port, start the API with
`CORS_ORIGINS=http://localhost:<port>`. To deploy elsewhere, change `API` at the top of
`frontend/src/App.jsx`.

## Retrain and evaluate

```bash
cd backend
python train.py        # compares 9 candidates per target, calibrates and saves the best
python evaluate.py     # nested CV, ROC curves, confusion matrices (EVAL_REPEATS=1 for a quick run)
```

Engineered features can be switched off for comparison: set `CARDIO_FE=0` before running.
All random seeds are fixed (42).

## API

| Endpoint | Purpose |
|---|---|
| `GET /schema` | input definitions (type, range, options, default) |
| `POST /predict` | probabilities, risk level, colour and SHAP factors for CAD, LAD, LCX, RCA |
| `GET /patients/{i}` | a dataset record for demos |
| `GET /metrics`, `GET /evaluation` | CV results, nested CV, ROC, confusion matrices |
| `GET /importance/{target}` | global mean absolute SHAP |

## Results

[[Paste the final table: accuracy, precision, recall, F1, ROC-AUC and nested-CV AUC for CAD, LAD, LCX, RCA]]

LCX and RCA are harder to predict from routine clinical data; the dashboard marks vessels with
AUC below 0.75 as "lower confidence".

## Limitations

- Small single-source dataset (303 patients), no external validation.
- Artery paths are schematic centre-lines, not patient-specific anatomy. Artery colour shows
  vessel-level probability, not the location of a lesion.

## Credits

- Dataset: Z-Alizadeh Sani dataset (extension), UCI Machine Learning Repository.
- 3D heart model: [[name, author, source URL, license]]
- Built with scikit-learn, XGBoost, SHAP, FastAPI, React, Vite, Three.js and React Three Fiber.