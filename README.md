# xG Predictor (projeto e-2-e)

Este repositório reúne um projeto end-to-end de ciência de dados em que eu pratiquei a construção completa de uma solução de xG (expected goals) a partir de dados de partidas reais. A ideia foi simular um fluxo “de ponta a ponta” para entender desde a coleta até a entrega de inferências.

## Visão geral
- **Objetivo:** estimar a probabilidade de um chute virar gol usando dados de finalizações combinando contexto espacial, postura do jogador e presença de bloqueadores.
- **Escopo e-2-e:** ingestão (notebook `src/data_loader.ipynb`/dados StatsBomb), pré-processamento (`src/pre_processing.py` + `src/utils.py`), treinamento/avaliação (`src/trainingModel.ipynb`) e inferência/pipeline final (`models/xg_pipeline_final.pkl`).
- **Motivação:** praticar um projeto real de data science, articulando engenharia de dados, features derivadas e modelo final de forma conectada.

## Estrutura do projeto
```
xgPredictor/
├── data/bronze/csv/statsbomb_shots_final.csv   # entradas tabulares usadas nos notebooks (dados StatsBomb já baixados)
├── models/xg_pipeline_final.pkl                # pipeline treinado para inferência
└── src/
    ├── data_loader.ipynb                       # exploração e carga dos dados originais
    ├── pre_processing.py                       # classe `xGPreprocessor` e transformação de coordenadas/features
    ├── utils.py                                # funções geométricas (normalização, distância, bloqueadores etc.)
    ├── trainingModel.ipynb                     # notebook de treinamento/avaliação
    └── testingWithWorldcupData.ipynb           # testes adicionais com dados específicos
``` 

## Fluxo end-to-end resumido
1. **Dados brutos:** o CSV em `data/bronze/csv` reúne eventos de finalizações do StatsBomb, já tratado como base inicial.
2. **Normalização geométrica:** `src/utils.normalize_direction` garante que cada chute esteja orientado para o mesmo lado do campo, permitindo usar distâncias e ângulos consistentes.
3. **Features derivadas:** `xGPreprocessor.transform` cria indicadores como `is_inside_box`, `distance_to_goal`, `shot_angle`, `foot_alignment` e `n_adversarios_frente` (baseado no `shot_freeze_frame`).
4. **Treinamento:** os notebooks documentam o treinamento com CatBoost/XGBoost e o pipeline final salvo em `models/xg_pipeline_final.pkl`, compatível com o transformador e o modelo.
5. **Inferência:** o pipeline serializado aplica automaticamente o pré-processamento e retorna probabilidades de gol para novas finalizações.

## Como reproduzir
1. Criar o ambiente e instalar dependências: `pip install -r requirements.txt`.
2. Executar `src/data_loader.ipynb` para reprocessar ou baixar dados do StatsBomb (usa `statsbombpy`).
3. Rodar `src/trainingModel.ipynb` para re-treinar ou experimentar variações. Os notebooks já documentam métricas e visualizações.
4. Testar o pipeline salvo com dados de exemplo a partir do notebook `src/testingWithWorldcupData.ipynb`.

## Próximos passos sugeridos
1. Automatizar o carregamento de novos arquivos CSV no `data/bronze` e versionar os dados com checkpoints.
2. Criar uma interface (Streamlit ou API FastAPI) para servir previsões em tempo real usando `models/xg_pipeline_final.pkl`.
3. Adicionar explicabilidade (SHAP) para entender quais features influenciam mais o xG em cada contexto.

## Dependências principais
- `numpy`, `pandas`, `scikit-learn` (engine de pré-processamento/metrificação)
- `xgboost`, `catboost` (modelos de gradiente)
- `statsbombpy` (API para puxar os dados de eventos reais)
- `shap` e `tqdm` (visualização e progresso)

O README resume o racional e o resultado desse exercício prático de data science, organizado como um projeto e-2-e completo. Use-o como ponto de partida para replicar, expandir ou operacionalizar o pipeline.
