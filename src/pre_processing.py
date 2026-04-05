import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
# Importe suas funções do utils
from utils import (
    normalize_direction, check_inside_box, check_inverted,  distance_to_goal, shot_angle, 
    count_blockers
)

class xGPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Definimos aqui as colunas que queremos manter ao final
        self.final_features = [
            'play_pattern',  
            'shot_aerial_won', 
            'shot_body_part',
            'shot_first_time',
            'shot_one_on_one',
            'shot_technique',
            'shot_type',
            'under_pressure',
            'shot_open_goal',
            'shot_follows_dribble',
            'x_norm',
            'y_norm', 
            'is_inside_box', 
            'foot_alignment', 
            'distance_to_goal',
           'shot_angle',
           'n_adversarios_frente'
                    ]
        



    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 1. Copiar para não alterar o original
        df = X.copy()

        # --- A JUSTE CRUCIAL: Garantir colunas obrigatórias ---
        # Lista de colunas que podem sumir dependendo do jogo/API
        cols_que_podem_faltar = [
            'shot_open_goal', 'shot_follows_dribble', 'under_pressure', 
            'shot_one_on_one', 'shot_first_time', 'shot_aerial_won'
        ]
        
        for col in cols_que_podem_faltar:
            if col not in df.columns:
                df[col] = False  # Cria a coluna com False se ela não existir
        # -----------------------------------------------------

        # 2. Preenchimento de NaNs básicos
        df["under_pressure"] = df["under_pressure"].fillna(False)
        df["shot_one_on_one"] = df["shot_one_on_one"].fillna(False)
        df["shot_open_goal"] = df["shot_open_goal"].fillna(False)
        # Adicione outros fillnas se necessário para colunas binárias

        # 3. Normalização e Geometria (Suas funções do utils)
        df = normalize_direction(df)
        
        df["is_inside_box"] = df.apply(check_inside_box, axis=1)
        df["foot_alignment"] = df.apply(check_inverted, axis=1)
        df["distance_to_goal"] = df.apply(distance_to_goal, axis=1)
        df["shot_angle"] = df.apply(shot_angle, axis=1)
        df["n_adversarios_frente"] = df.apply(count_blockers, axis=1)

        # 4. Seleção de colunas
        # Agora usamos self.final_features DIRETAMENTE. 
        # Se alguma coluna faltar aqui, o Python vai dar erro agora (o que é bom para avisar que algo está errado),
        # mas como garantimos as faltantes acima, isso não deve acontecer.
        return df[self.final_features]