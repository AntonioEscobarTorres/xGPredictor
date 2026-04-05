"""
xG Dashboard - Comparação entre modelo próprio e StatsBomb
Coloque este arquivo dentro da pasta `interface/`
Execute com: streamlit run xg_dashboard.py
"""

import sys
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from mplsoccer import VerticalPitch
from pathlib import Path
from statsbombpy import sb

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
MODEL_PATH = ROOT_DIR / "models/xg_pipeline_final.pkl"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


# ─── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="xG Dashboard",
    page_icon="⚽",
    layout="wide",
)

# ─── Estilo customizado ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #313244;
    }
    .metric-label { font-size: 13px; color: #cdd6f4; margin-bottom: 6px; }
    .metric-value { font-size: 36px; font-weight: 700; }
    .metric-value.model    { color: #89b4fa; }
    .metric-value.statsbomb{ color: #a6e3a1; }
    .metric-value.diff     { color: #fab387; }
    .section-title {
        font-size: 18px; font-weight: 600;
        color: #cdd6f4; margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Carregamento do pipeline ─────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model_pipeline():
    return joblib.load(MODEL_PATH)


# ─── Helpers para extrair partes do pipeline ─────────────────────────────────

def get_feature_names(pipeline) -> list:
    """
    Usa encoder.get_feature_names_out() — método nativo do sklearn que retorna
    os nomes na ordem exata do array transformado, incluindo OHE e passthrough.

    Saída bruta: 'onehot__play_pattern_Regular Play', 'remainder__distance_to_goal'
    Limpamos o prefixo para deixar legível no waterfall plot.
    """
    encoder = pipeline.named_steps["encoding"]
    raw_names = encoder.get_feature_names_out()
    return [n.split("__", 1)[1] if "__" in n else n for n in raw_names]


def transform_shot(pipeline, shot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica xGPreprocessor + ColumnTransformer sobre um DataFrame de 1 linha
    e devolve um DataFrame com os nomes de feature corretos.
    Esse é o input que o CatBoost interno enxerga.
    """
    X_geo = pipeline.named_steps["geometrias"].transform(shot_df)
    X_enc = pipeline.named_steps["encoding"].transform(X_geo)
    feature_names = get_feature_names(pipeline)
    return pd.DataFrame(X_enc, columns=feature_names)


# ─── Predição e SHAP ─────────────────────────────────────────────────────────

def predict_xg(shot_row: pd.Series) -> float:
    pipeline = load_model_pipeline()
    shot_df = pd.DataFrame([shot_row])
    return float(pipeline.predict_proba(shot_df)[0, 1])


@st.cache_data(show_spinner=False)
def compute_shap_explanation(_pipeline, shot_row_dict: dict):
    """
    Calcula SHAP values locais para um único chute.

    Por que passamos numpy e não DataFrame?
    - CatBoost salvo via sklearn pipeline às vezes reporta n_features_in_=0
      quando acessado fora do pipeline (quirk de serialização do joblib).
    - Passando array numpy puro o TreeExplainer bypassa essa checagem.

    Por que shap_values() e não explainer()?
    - explainer() usa a API nova do SHAP que tenta inspecionar n_features_in_
      e falha com o bug acima. shap_values() usa a API legada, mais robusta.
    """
    shot_df = pd.DataFrame([shot_row_dict])
    X_transformed = transform_shot(_pipeline, shot_df)
    feature_names = list(X_transformed.columns)

    catboost_model = _pipeline.named_steps["classifier"]

    # Array numpy puro — sem nomes de coluna — para evitar o bug n_features_in_=0
    X_np = X_transformed.values.astype(float)

    # Calcula SHAP values em log-odds (única opção com tree_path_dependent)
    # e converte para probabilidade mantendo a propriedade aditiva do waterfall.
    explainer = shap.TreeExplainer(catboost_model, feature_perturbation="tree_path_dependent")

    # API nova (explainer(X)) retorna Explanation com .values, .base_values e .data
    # já estruturados — necessário para o waterfall funcionar corretamente.
    shap_explanation = explainer(X_np)

    # Seleciona a instância 0 (único chute) e faz cópia para não mutar o cache
    import copy
    sv = copy.deepcopy(shap_explanation[0])

    # ── Conversão log-odds → probabilidade ───────────────────────────────
    # SHAP aditivo em logit: base_logit + Σshap_i = f(x)_logit
    # Queremos converter para probabilidade mantendo aditividade:
    #   p_base  = sigmoid(base_logit)
    #   p_final = sigmoid(f(x)_logit)
    #   Cada shap_i_prob = shap_i_logit × (p_final - p_base) / Σshap_i_logit
    #
    # IMPORTANTE: o denominador é Σshap_i (com sinal), não Σ|shap_i|.
    # Isso preserva sinais e garante p_base + Σshap_i_prob == p_final exatamente.
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    base_logit  = float(sv.base_values)
    logit_sum   = float(sv.values.sum())           # Σshap em logit (com sinal)
    final_logit = base_logit + logit_sum

    p_base  = sigmoid(base_logit)
    p_final = sigmoid(final_logit)
    delta   = p_final - p_base                     # diferença em probabilidade

    # Reescala proporcional: cada feature mantém seu peso relativo e sinal
    if logit_sum != 0:
        sv.values = sv.values * (delta / logit_sum)
    # Se logit_sum == 0 todos os shap são zero — nada a converter

    sv.base_values = np.float64(p_base)

    # Adiciona nomes de feature (X_np não tem colunas, precisamos setar aqui)
    sv.feature_names = feature_names

    return sv


def build_shap_waterfall(pipeline, shot_row: pd.Series):
    """
    Gera a figura matplotlib do waterfall plot SHAP.
    Retorna (fig, error_msg). Se der erro, fig=None.
    """
    try:
        explanation = compute_shap_explanation(pipeline, shot_row.to_dict())

        # Fecha todas as figuras matplotlib abertas antes de chamar o waterfall.
        # Sem isso, plt.gcf() pode retornar fig_pitch ou fig_bar criadas antes,
        # e o st.pyplot acaba renderizando as figuras antigas junto com o SHAP.
        plt.close("all")

        shap.plots.waterfall(explanation, max_display=12, show=False)

        fig = plt.gcf()
        fig.patch.set_facecolor("#1e1e2e")
        for ax in fig.axes:
            ax.set_facecolor("#1e1e2e")
            ax.tick_params(colors="#cdd6f4")
            ax.xaxis.label.set_color("#cdd6f4")
            ax.yaxis.label.set_color("#cdd6f4")
            ax.title.set_color("#cdd6f4")
            for spine in ax.spines.values():
                spine.set_color("#585b70")

        plt.tight_layout()
        return fig, None

    except Exception as e:
        import traceback
        return None, traceback.format_exc()


# ─── StatsBomb helpers ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_competitions():
    return sb.competitions()

@st.cache_data(show_spinner=False)
def load_matches(competition_id: int, season_id: int):
    return sb.matches(competition_id=competition_id, season_id=season_id)

@st.cache_data(show_spinner=False)
def load_shots(match_id: int):
    events = sb.events(match_id=match_id)
    shots = events[events["type"] == "Shot"].copy()
    return shots.reset_index(drop=True)


# ─── Visualizações ────────────────────────────────────────────────────────────

def draw_pitch_with_shot(location, outcome_label):
    pitch = VerticalPitch(
        pitch_type="statsbomb", pitch_color="#1e1e2e",
        line_color="#585b70", half=True, goal_type="box", linewidth=1.5,
    )
    fig, ax = pitch.draw(figsize=(5, 5))
    fig.patch.set_facecolor("#1e1e2e")

    if location and len(location) == 2:
        x, y = location[0], location[1]
        color = "#a6e3a1" if "Goal" in str(outcome_label) else "#f38ba8"
        pitch.scatter(x, y, ax=ax, s=300, color=color,
                      edgecolors="#cdd6f4", linewidth=1.5, zorder=5)
        ax.annotate(outcome_label, xy=(x, y), xytext=(x + 3, y + 2),
                    fontsize=8, color="#cdd6f4",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#313244", alpha=0.8))
    return fig


def build_xg_bar_chart(model_xg: float, sb_xg: float):
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    bars = ax.barh(["Meu Modelo", "StatsBomb"], [model_xg, sb_xg],
                   color=["#89b4fa", "#a6e3a1"], height=0.4, edgecolor="#313244")
    ax.set_xlim(0, 1)
    ax.set_xlabel("xG", color="#cdd6f4", fontsize=10)
    ax.tick_params(colors="#cdd6f4")
    for s in ax.spines.values():
        s.set_color("#585b70")
    for bar, val in zip(bars, [model_xg, sb_xg]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left",
                color="#cdd6f4", fontsize=11, fontweight="bold")
    ax.set_title("Comparação de xG", color="#cdd6f4", fontsize=12, pad=10)
    plt.tight_layout()
    return fig


# ─── Layout principal ─────────────────────────────────────────────────────────

st.title("⚽ xG Dashboard")
st.caption("Compare o xG do seu modelo com os dados do StatsBomb")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔎 Filtros")

    with st.spinner("Carregando competições..."):
        competitions = load_competitions()

    comp_options = (competitions[["competition_name", "competition_id"]]
                    .drop_duplicates().sort_values("competition_name"))
    competition_name = st.selectbox("Competição", comp_options["competition_name"].tolist())
    competition_id = int(comp_options.loc[
        comp_options["competition_name"] == competition_name, "competition_id"].iloc[0])

    seasons = (competitions[competitions["competition_id"] == competition_id]
               [["season_name", "season_id"]].drop_duplicates()
               .sort_values("season_name", ascending=False))
    season_name = st.selectbox("Temporada", seasons["season_name"].tolist())
    season_id = int(seasons.loc[seasons["season_name"] == season_name, "season_id"].iloc[0])

    with st.spinner("Carregando partidas..."):
        matches = load_matches(competition_id, season_id)

    matches["label"] = (matches["home_team"] + " vs " + matches["away_team"]
                        + "  (" + matches["match_date"].astype(str) + ")")
    match_label = st.selectbox("Partida", matches["label"].tolist())
    match_id = int(matches.loc[matches["label"] == match_label, "match_id"].iloc[0])

    st.divider()
    st.caption("Dados fornecidos por StatsBomb Open Data")

# ── Finalizações ──────────────────────────────────────────────────────────────
with st.spinner("Carregando finalizações..."):
    shots = load_shots(match_id)

if shots.empty:
    st.warning("Nenhuma finalização encontrada nesta partida.")
    st.stop()

shots["shot_label"] = (
    shots["player"].astype(str) + " — "
    + shots.get("shot_outcome", pd.Series(["?"] * len(shots))).apply(
        lambda v: v if isinstance(v, str) else v.get("name", "?") if isinstance(v, dict) else "?"
    )
    + "  (" + shots["minute"].astype(str) + "')"
)

st.subheader("🎯 Selecione uma finalização")
shot_label = st.selectbox("Finalização", shots["shot_label"].tolist(), label_visibility="collapsed")
selected = shots[shots["shot_label"] == shot_label].iloc[0]

# ── xG ────────────────────────────────────────────────────────────────────────
pipeline = load_model_pipeline()
model_xg = predict_xg(selected)
sb_xg_raw = selected.get("shot_statsbomb_xg", None)
sb_xg = float(sb_xg_raw) if sb_xg_raw is not None and not (
    isinstance(sb_xg_raw, float) and np.isnan(sb_xg_raw)) else None
diff = round(model_xg - sb_xg, 4) if sb_xg is not None else None

# ── Métricas ──────────────────────────────────────────────────────────────────
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">⚙️ Meu Modelo</div>
        <div class="metric-value model">{model_xg:.4f}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    sb_display = f"{sb_xg:.4f}" if sb_xg is not None else "N/A"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">📊 StatsBomb xG</div>
        <div class="metric-value statsbomb">{sb_display}</div>
    </div>""", unsafe_allow_html=True)
with col3:
    diff_display = f"{diff:+.4f}" if diff is not None else "N/A"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Δ Diferença</div>
        <div class="metric-value diff">{diff_display}</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── Linha 1: campo + comparação xG ───────────────────────────────────────────
col_pitch, col_bar = st.columns([1, 1])

with col_pitch:
    st.markdown('<div class="section-title">📍 Localização do chute</div>', unsafe_allow_html=True)
    outcome_raw = selected.get("shot_outcome", "?")
    outcome = outcome_raw if isinstance(outcome_raw, str) else (
        outcome_raw.get("name", "?") if isinstance(outcome_raw, dict) else "?")
    fig_pitch = draw_pitch_with_shot(selected.get("location"), outcome)
    st.pyplot(fig_pitch, width="stretch")
    plt.close(fig_pitch)

with col_bar:
    if sb_xg is not None:
        st.markdown('<div class="section-title">📊 Comparação de xG</div>', unsafe_allow_html=True)
        fig_bar = build_xg_bar_chart(model_xg, sb_xg)
        st.pyplot(fig_bar, width="stretch")
        plt.close(fig_bar)
    else:
        st.info("StatsBomb não disponibilizou xG para esta finalização.")

st.divider()

# ── Linha 2: SHAP waterfall ───────────────────────────────────────────────────
st.markdown('<div class="section-title">🧠 Por que meu modelo deu esse xG?</div>',
            unsafe_allow_html=True)
st.caption(
    "Cada barra mostra o quanto aquela feature **empurrou** o xG para cima 🔴 "
    "ou para baixo 🔵 em relação ao valor base médio do modelo. "
    "A soma de todas as contribuições resulta no xG final."
)

with st.spinner("Calculando explicação SHAP..."):
    fig_shap, shap_error = build_shap_waterfall(pipeline, selected)

if shap_error:
    st.error(f"Não foi possível gerar explicação SHAP: `{shap_error}`")
    st.info(
        "Causas comuns: `xGPreprocessor` não importado corretamente, "
        "ou versão do sklearn incompatível com o pipeline salvo."
    )
else:
    st.pyplot(fig_shap, width="stretch")


# ── Glossário de features ─────────────────────────────────────────────────────
st.divider()
with st.expander("📖 Feature Glossary — what each variable means"):
    st.markdown("""
    #### 📐 Geometry & Position
    | Feature | Description |
    |---|---|
    | `distance_to_goal` | Straight-line distance (in meters) from the shot location to the center of the goal. Shorter distance = higher xG. |
    | `shot_angle` | The angular aperture of the goal as seen from the shot location. Wider angle = better shooting opportunity. |
    | `x_norm` | Normalized x-coordinate of the shot on the pitch (0 = own goal line, 120 = opponent goal line). |
    | `y_norm` | Normalized y-coordinate of the shot (0 = left touchline, 80 = right touchline). |
    | `is_inside_box` | Whether the shot was taken from inside the penalty box (1 = yes, 0 = no). |

    #### 👤 Situation & Context
    | Feature | Description |
    |---|---|
    | `play_pattern_*` | How the team gained possession before the shot (e.g. Regular Play, From Counter, From Corner, From Free Kick). |
    | `shot_type_*` | Type of shot attempt: Open Play, Free Kick, Corner, or Penalty. |
    | `shot_technique_*` | Technique used: Normal, Volley, Half Volley, Header, Lob, Backheel, Overhead Kick, etc. |
    | `shot_body_part_*` | Which body part was used: Right Foot, Left Foot, Head, or Other. |
    | `possession` | StatsBomb possession sequence ID — distinguishes between different attacks within the same match. |
    | `under_pressure` | Whether the shooter was being actively pressured by a defender at the moment of the shot (1 = yes). |
    | `n_adversarios_frente` | Number of opposing outfield players positioned between the ball and the goal at the moment of the shot. More defenders = lower xG. |

    #### 🦶 Foot & Alignment
    | Feature | Description |
    |---|---|
    | `foot_alignment_*` | Whether the shooter used their natural foot (Natural), their weaker foot (Inverted), or the feature is unavailable (Other). |

    #### ⚡ Shot Circumstances
    | Feature | Description |
    |---|---|
    | `shot_first_time` | Whether the shot was hit first-time without controlling the ball first (1 = yes). First-time shots are harder to save. |
    | `shot_one_on_one` | Whether the shooter was in a one-on-one situation with the goalkeeper (1 = yes). High-value opportunity. |
    | `shot_open_goal` | Whether the shot was directed at an open/unguarded goal (1 = yes). Highest-value situation. |
    | `shot_aerial_won` | Whether the shot came from an aerial duel that the attacker won (1 = yes, typically headers). |
    | `shot_follows_dribble` | Whether the shot immediately followed a successful dribble past a defender (1 = yes). |
    """)

# ── Detalhes brutos ───────────────────────────────────────────────────────────
st.divider()
with st.expander("🔍 Detalhes completos da finalização"):
    detail_fields = [
        "player", "team", "minute", "second", "location",
        "shot_outcome", "shot_technique", "shot_body_part",
        "shot_type", "under_pressure", "shot_statsbomb_xg",
    ]
    detail = {f: selected[f] for f in detail_fields if f in selected.index}
    st.json({k: (v if not (isinstance(v, float) and np.isnan(v)) else None)
             for k, v in detail.items()})