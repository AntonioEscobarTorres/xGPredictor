# xG Predictor (end-to-end exercise)

This repository hosts an end-to-end data science exercise where I built a complete expected goals (xG) solution using real match data. The goal was to simulate a full pipeline—from data ingestion to inference—so I could practice a realistic project.

## Overview
- **Goal:** estimate the probability that a shot becomes a goal by combining spatial context, player posture, and blocker presence from real-event data.
- **End-to-end scope:** ingestion (`src/data_loader.ipynb` / StatsBomb data), preprocessing (`src/pre_processing.py` + `src/utils.py`), training/evaluation (`src/trainingModel.ipynb`), and inference/pipeline (`models/xg_pipeline_final.pkl`).
- **Motivation:** sharpen data science craft through a connected workflow that touches data engineering, derived features, and the final model in one project.

## Project structure
```
xgPredictor/
├── data/bronze/csv/statsbomb_shots_final.csv   # tabular inputs used in the notebooks (pre-downloaded StatsBomb shots)
├── models/xg_pipeline_final.pkl                # serialized pipeline for inference
└── src/
    ├── data_loader.ipynb                       # exploration and loading of the original data
    ├── pre_processing.py                       # `xGPreprocessor` class and feature engineering logic
    ├── utils.py                                # geometric helpers (normalization, distance, blockers, etc.)
    ├── trainingModel.ipynb                     # training/evaluation notebook
    └── testingWithWorldcupData.ipynb           # supplementary testing with specific datasets
```

## End-to-end flow
1. **Raw data:** the CSV in `data/bronze/csv` collects StatsBomb shot events and acts as the base dataset.
2. **Geometry normalization:** `src/utils.normalize_direction` ensures every shot faces the same goal so distances and angles are aligned.
3. **Derived features:** `xGPreprocessor.transform` adds indicators such as `is_inside_box`, `distance_to_goal`, `shot_angle`, `foot_alignment`, and `n_adversarios_frente` (based on `shot_freeze_frame`).
4. **Training:** the notebooks document CatBoost/XGBoost experiments and the final pipeline saved at `models/xg_pipeline_final.pkl`, compatible with the transformer and model steps.
5. **Inference:** the serialized pipeline applies preprocessing and returns goal probabilities for new shots.

## Reproducing the project
1. Set up the environment and install dependencies: `pip install -r requirements.txt`.
2. Run `src/data_loader.ipynb` to reprocess or download StatsBomb data (it uses `statsbombpy`).
3. Execute `src/trainingModel.ipynb` to retrain or explore variations. The notebooks document metrics and visualizations.
4. Validate the serialized pipeline with sample data via `src/testingWithWorldcupData.ipynb`.

## Suggested next steps
1. Automate loading new CSV files into `data/bronze` and version the datasets with checkpoints.
2. Build an interface (Streamlit or FastAPI) that serves predictions from `models/xg_pipeline_final.pkl` in real time.
3. Add explainability (SHAP) to highlight which features drive the xG prediction per shot context.

## Key dependencies
- `numpy`, `pandas`, `scikit-learn` (preprocessing and metrics)
- `xgboost`, `catboost` (gradient boosting models)
- `statsbombpy` (StatsBomb API connector)
- `shap`, `tqdm` (visualization and progress tracking)

This README captures the intent and outputs of this practical data science exercise, framed as an end-to-end project. Use it as a guide to reproduce, expand, or operationalize the xG pipeline.
