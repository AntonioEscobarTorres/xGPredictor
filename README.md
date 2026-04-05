# xG Predictor – Study Project

This repository captures a study project that walks through a full expected-goals (xG) workflow: data ingestion, feature engineering, modeling, and an interactive interface that compares the trained pipeline to StatsBomb’s published xG values. It was originally built as a hands-on exercise to practice real-world data science engineering and modeling skills.

## Getting started
1. **Clone & enter the repo**
   ```bash
   git clone <repo-url>
   cd xgPredictor
   ```
2. **Create a Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   .venv\\Scripts\\activate     # Windows
   ```
3. **Install the dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Project layout

```
xgPredictor/
├── data/bronze/csv/statsbomb_shots_final.csv   # StatsBomb shot events snapshot
├── data/bronze/shotsByMatches/                 # match‑level snapshots used during exploration
├── models/xg_pipeline_final.pkl                # saved pipeline (preprocessing + CatBoost model)
├── src/                                        # preprocessing/helpers/training notebooks
├── interface/app.py                            # Streamlit dashboard
└── requirements.txt                            # pinned dependencies
```

## Running the notebooks
- `src/data_loader.ipynb`: explore and/or reload StatsBomb open-data shot events (requires internet and statsbombpy credentials if private data is needed).
- `src/pre_processing.py` & `src/utils.py`: contain the `xGPreprocessor` and helper functions for geometry normalization, blocker counting, and feature derivation.
- `src/trainingModel.ipynb`: training experiments with CatBoost/XGBoost and saving the final pipeline (`models/xg_pipeline_final.pkl`).
- `src/testingWithWorldcupData.ipynb`: additional testing with curated datasets.

## Launching the Streamlit dashboard
1. Ensure the virtual environment (with dependencies) is active.
2. Run:
   ```bash
   streamlit run interface/app.py
   ```
3. Use the sidebar to select a StatsBomb competition + match (falls back to the local CSV snapshot when the API is unavailable) and compare:
   - **Model xG:** the probability produced by the saved pipeline (`models/xg_pipeline_final.pkl`).
   - **StatsBomb xG:** the published value embedded in the dataset.
4. The interface also shows shot metadata, location, freeze-frame blockers, and a mini visualization.

## Notes for future work
- The fully serialized pipeline already bundles preprocessing + model, so scoring a shot only requires the single file under `models/`.
- If you extend the project, try adding explainability (SHAP) or packaging the dashboard into a Streamlit app with hosting.
- Keep the data snapshots documented if you refresh them (CSV in `data/bronze/csv` and/or the match pickles).

## Why it’s a study project
This repo was crafted as a personal study project to practice end-to-end data science: from cleaning StatsBomb shots to fitting a gradient boosting model, then comparing it to industry-standard xG. It is not production software, but a living notebook-style exploration that connects engineering, modeling, and visualization in one place.
