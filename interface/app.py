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
import matplotlib.patches as patches
from mplsoccer import VerticalPitch
from pathlib import Path
from statsbombpy import sb
import catboost

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
MODEL_PATH = ROOT_DIR / "models/xg_pipeline_final.pkl"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


@st.cache_resource(show_spinner=False)
def load_model_pipeline():
    """Load the saved pipeline that includes preprocessing + xG model."""
    return joblib.load(MODEL_PATH)

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
    .metric-label {
        font-size: 13px;
        color: #cdd6f4;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 36px;
        font-weight: 700;
    }
    .metric-value.model { color: #89b4fa; }
    .metric-value.statsbomb { color: #a6e3a1; }
    .metric-value.diff { color: #fab387; }
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #cdd6f4;
        margin-bottom: 12px;
    }
    div[data-testid="stSelectbox"] label {
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ─── Funções auxiliares ───────────────────────────────────────────────────────

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
    shots = shots.reset_index(drop=True)
    return shots


def predict_xg(shot_row: pd.Series) -> float:
    """
    ────────────────────────────────────────────────
    PONTO DE INTEGRAÇÃO DO SEU MODELO DE ML
    ────────────────────────────────────────────────
    Substitua o corpo desta função pela chamada real
    ao seu pipeline treinado, por exemplo:

        features = extract_features(shot_row)
        return model.predict_proba([features])[0][1]

    Por enquanto retorna um placeholder baseado
    na localização do chute (lógica simples).
    """
    pipeline = load_model_pipeline()
    shot_df = pd.DataFrame([shot_row])
    try:
        if hasattr(pipeline, "predict_proba"):
            return float(pipeline.predict_proba(shot_df)[0, 1])
        return float(pipeline.predict(shot_df)[0])
    except Exception as exc:  # pragma: no cover - best effort prediction
        st.error(f"Couldn't score shot with saved pipeline: {exc}")
        loc = shot_row.get("location", [60, 40])
        x, y = loc[0], loc[1]
        dist = np.sqrt((120 - x) ** 2 + (40 - y) ** 2)
        angle = np.arctan2(
            7.32 * (120 - x),
            (120 - x) ** 2 + (40 - y) ** 2 - (7.32 / 2) ** 2,
        )
        angle = max(angle, 0)
        return float(np.clip(0.05 + (1 / (dist + 1)) * 5 + angle * 0.3, 0.01, 0.99))


def draw_pitch_with_shot(location, outcome_label):
    """Desenha o campo com a posição do chute marcada."""
    pitch = VerticalPitch(
        pitch_type="statsbomb",
        pitch_color="#1e1e2e",
        line_color="#585b70",
        half=True,
        goal_type="box",
        linewidth=1.5,
    )
    fig, ax = pitch.draw(figsize=(5, 5))
    fig.patch.set_facecolor("#1e1e2e")

    if location and len(location) == 2:
        x, y = location[0], location[1]
        color = "#a6e3a1" if "Goal" in str(outcome_label) else "#f38ba8"
        pitch.scatter(
            x, y,
            ax=ax,
            s=300,
            color=color,
            edgecolors="#cdd6f4",
            linewidth=1.5,
            zorder=5,
        )
        ax.annotate(
            outcome_label,
            xy=(x, y),
            xytext=(x + 3, y + 2),
            fontsize=8,
            color="#cdd6f4",
            bbox=dict(boxstyle="round,pad=0.3", fc="#313244", alpha=0.8),
        )

    return fig


def build_xg_bar_chart(model_xg: float, sb_xg: float):
    """Gráfico de barras comparando os dois xG."""
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    labels = ["Meu Modelo", "StatsBomb"]
    values = [model_xg, sb_xg]
    colors = ["#89b4fa", "#a6e3a1"]

    bars = ax.barh(labels, values, color=colors, height=0.4, edgecolor="#313244")
    ax.set_xlim(0, 1)
    ax.set_xlabel("xG", color="#cdd6f4", fontsize=10)
    ax.tick_params(colors="#cdd6f4")
    for spine in ax.spines.values():
        ax.spines[spine.axis_name if hasattr(spine, "axis_name") else "top"].set_visible(False)
    for s in ax.spines.values():
        s.set_color("#585b70")

    for bar, val in zip(bars, values):
        ax.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left",
            color="#cdd6f4", fontsize=11, fontweight="bold",
        )

    ax.set_title("Comparação de xG", color="#cdd6f4", fontsize=12, pad=10)
    plt.tight_layout()
    return fig


# ─── Layout principal ─────────────────────────────────────────────────────────

st.title("⚽ xG Dashboard")
st.caption("Compare o xG do seu modelo com os dados do StatsBomb")
st.divider()

# ── Sidebar: Seleção de competição / temporada / partida ─────────────────────
with st.sidebar:
    st.header("🔎 Filtros")

    with st.spinner("Carregando competições..."):
        competitions = load_competitions()

    # Competição
    comp_options = (
        competitions[["competition_name", "competition_id"]]
        .drop_duplicates()
        .sort_values("competition_name")
    )
    competition_name = st.selectbox(
        "Competição",
        comp_options["competition_name"].tolist(),
    )
    competition_id = int(
        comp_options.loc[
            comp_options["competition_name"] == competition_name, "competition_id"
        ].iloc[0]
    )

    # Temporada
    seasons = competitions[competitions["competition_id"] == competition_id][
        ["season_name", "season_id"]
    ].drop_duplicates().sort_values("season_name", ascending=False)

    season_name = st.selectbox("Temporada", seasons["season_name"].tolist())
    season_id = int(
        seasons.loc[seasons["season_name"] == season_name, "season_id"].iloc[0]
    )

    # Partida
    with st.spinner("Carregando partidas..."):
        matches = load_matches(competition_id, season_id)

    matches["label"] = (
        matches["home_team"] + " vs " + matches["away_team"]
        + "  (" + matches["match_date"].astype(str) + ")"
    )
    match_label = st.selectbox("Partida", matches["label"].tolist())
    match_id = int(
        matches.loc[matches["label"] == match_label, "match_id"].iloc[0]
    )

    st.divider()
    st.caption("Dados fornecidos por StatsBomb Open Data")

# ── Carrega finalizações ──────────────────────────────────────────────────────
with st.spinner("Carregando finalizações..."):
    shots = load_shots(match_id)

if shots.empty:
    st.warning("Nenhuma finalização encontrada nesta partida.")
    st.stop()

# Formata label dos chutes
shots["shot_label"] = (
    shots["player"].astype(str)
    + " — "
    + shots.get("shot_outcome", pd.Series(["?"] * len(shots))).apply(
        lambda v: v if isinstance(v, str) else v.get("name", "?") if isinstance(v, dict) else "?"
    )
    + "  ("
    + shots["minute"].astype(str)
    + "')"
)

# ── Seleção de finalização ────────────────────────────────────────────────────
st.subheader("🎯 Selecione uma finalização")
shot_label = st.selectbox("Finalização", shots["shot_label"].tolist(), label_visibility="collapsed")

selected = shots[shots["shot_label"] == shot_label].iloc[0]

# ── Calcula xG ───────────────────────────────────────────────────────────────
model_xg = predict_xg(selected)
sb_xg_raw = selected.get("shot_statsbomb_xg", None)
sb_xg = float(sb_xg_raw) if sb_xg_raw is not None and not (isinstance(sb_xg_raw, float) and np.isnan(sb_xg_raw)) else None

diff = round(model_xg - sb_xg, 4) if sb_xg is not None else None

# ── Métricas ──────────────────────────────────────────────────────────────────
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">⚙️ Meu Modelo</div>
        <div class="metric-value model">{model_xg:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    sb_display = f"{sb_xg:.4f}" if sb_xg is not None else "N/A"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📊 StatsBomb xG</div>
        <div class="metric-value statsbomb">{sb_display}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    diff_display = f"{diff:+.4f}" if diff is not None else "N/A"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Δ Diferença</div>
        <div class="metric-value diff">{diff_display}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Visualizações ─────────────────────────────────────────────────────────────
col_pitch, col_bar = st.columns([1, 1])

with col_pitch:
    st.markdown('<div class="section-title">📍 Localização do chute</div>', unsafe_allow_html=True)
    outcome_raw = selected.get("shot_outcome", "?")
    outcome = outcome_raw if isinstance(outcome_raw, str) else (outcome_raw.get("name", "?") if isinstance(outcome_raw, dict) else "?")
    location = selected.get("location", None)
    fig_pitch = draw_pitch_with_shot(location, outcome)
    st.pyplot(fig_pitch, use_container_width=True)

with col_bar:
    if sb_xg is not None:
        st.markdown('<div class="section-title">📊 Comparação de xG</div>', unsafe_allow_html=True)
        fig_bar = build_xg_bar_chart(model_xg, sb_xg)
        st.pyplot(fig_bar, use_container_width=True)
    else:
        st.info("StatsBomb não disponibilizou xG para esta finalização.")

# ── Detalhes do chute ─────────────────────────────────────────────────────────
st.divider()
with st.expander("🔍 Detalhes completos da finalização"):
    detail_fields = [
        "player", "team", "minute", "second",
        "location", "shot_outcome", "shot_technique",
        "shot_body_part", "shot_type", "under_pressure",
        "shot_statsbomb_xg",
    ]
    detail = {
        f: selected[f]
        for f in detail_fields
        if f in selected.index
    }
    st.json({k: (v if not (isinstance(v, float) and np.isnan(v)) else None) for k, v in detail.items()})
