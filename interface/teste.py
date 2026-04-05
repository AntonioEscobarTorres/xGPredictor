import sys
import os
from pathlib import Path

# Garante que o Python encontre o 'pre_processing' ou 'geometrias'
# Se o seu arquivo pre_processing.py estiver na pasta 'src'
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent / "src"))

import joblib
import pandas as pd

# Agora o load deve funcionar sem ModuleNotFoundError
pipeline = joblib.load("../models/xg_pipeline_final.pkl")

encoder = pipeline.named_steps["encoding"]
catboost_model = pipeline.named_steps["classifier"]

print("=== ColumnTransformer transformers_ ===")
for name, trans, cols in encoder.transformers_:
    print(f"  name={name!r}, cols={cols}")

print("\n=== feature_names_in_ do encoder ===")
print(list(encoder.feature_names_in_))

print("\n=== OHE feature names ===")
print(list(encoder.named_transformers_["onehot"].get_feature_names_out()))

print("\n=== CatBoost feature count ===")
print("n_features esperado pelo CatBoost:", catboost_model.n_features_in_)

print("\n=== xGPreprocessor: colunas de saída ===")
import pandas as pd
# Pega uma linha qualquer do CSV de treino para testar
df_sample = pd.read_csv("data/bronze/csv/statsbomb_shots_final.csv").head(1)
X_geo = pipeline.named_steps["geometrias"].transform(df_sample)
print("Colunas após xGPreprocessor:", list(X_geo.columns) if hasattr(X_geo, "columns") else X_geo.shape)
X_enc = pipeline.named_steps["encoding"].transform(X_geo)
print("Shape após encoding:", X_enc.shape)