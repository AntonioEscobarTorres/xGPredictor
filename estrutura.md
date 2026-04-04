xg-simulator/
├── data/               # Amostras de dados ou modelos salvos (.pkl, .joblib)
├── src/
│   ├── data_loader.py  # Funções para buscar dados do StatsBomb (statsbombpy)
│   ├── preprocessor.py # Transformação de coordenadas e criação de features
│   └── model.py        # Carregamento do modelo e lógica de inferência
├── app.py              # Interface principal do Streamlit
├── requirements.txt    # Dependências (streamlit, statsbombpy, xgboost, pandas)
└── README.md