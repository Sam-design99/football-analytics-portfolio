"""
app.py
======
Dashboard principal Football Analytics — Streamlit
Lancer avec : streamlit run dashboard/app.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import pandas as pd

from data_pipeline import (
    get_competitions, get_matches, get_shots,
    get_events, compute_player_stats, compute_team_stats,
    get_passes
)
from visualizations import (
    shot_map, heatmap, player_radar,
    pass_network, xg_timeline, normalize_stats_for_radar
)

# ─────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Football Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label { font-size: 12px; color: #888; margin-bottom: 4px; }
    .metric-value { font-size: 26px; font-weight: 600; color: #ffffff; }
    .metric-delta { font-size: 11px; color: #1D9E75; margin-top: 2px; }
    h1, h2, h3 { color: #ffffff; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CACHE — évite de recharger à chaque interaction
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_competitions():
    return get_competitions()

@st.cache_data(show_spinner=False)
def cached_matches(competition_id, season_id):
    return get_matches(competition_id, season_id)

@st.cache_data(show_spinner=False)
def cached_events(match_id):
    events = get_events(match_id)
    events["x"] = events["location"].apply(lambda l: l[0] if isinstance(l, list) else None)
    events["y"] = events["location"].apply(lambda l: l[1] if isinstance(l, list) else None)
    return events

@st.cache_data(show_spinner=False)
def cached_shots(match_id):
    return get_shots(match_id)

@st.cache_data(show_spinner=False)
def cached_player_stats(match_id):
    return compute_player_stats(match_id)

@st.cache_data(show_spinner=False)
def cached_team_stats(match_id):
    return compute_team_stats(match_id)


# ─────────────────────────────────────────────
# SIDEBAR — Filtres
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚽ Football Analytics")
    st.markdown("---")

    # Chargement compétitions
    with st.spinner("Chargement des compétitions..."):
        comps = cached_competitions()

    comp_names = comps["competition_name"].unique().tolist()
    selected_comp = st.selectbox("🏆 Compétition", comp_names)

    comp_data = comps[comps["competition_name"] == selected_comp]
    season_names = comp_data["season_name"].tolist()
    selected_season = st.selectbox("📅 Saison", season_names)

    season_row = comp_data[comp_data["season_name"] == selected_season].iloc[0]
    competition_id = int(season_row["competition_id"])
    season_id = int(season_row["season_id"])

    # Chargement matchs
    with st.spinner("Chargement des matchs..."):
        matches = cached_matches(competition_id, season_id)

    if matches.empty:
        st.warning("Aucun match disponible pour cette sélection.")
        st.stop()

    matches["label"] = matches["home_team"] + " vs " + matches["away_team"] + " (" + matches["match_date"].astype(str).str[:10] + ")"
    match_labels = matches["label"].tolist()
    selected_match_label = st.selectbox("🎮 Match", match_labels)

    selected_match = matches[matches["label"] == selected_match_label].iloc[0]
    match_id = int(selected_match["match_id"])
    home_team = selected_match["home_team"]
    away_team = selected_match["away_team"]
    home_score = int(selected_match["home_score"])
    away_score = int(selected_match["away_score"])

    st.markdown("---")
    st.markdown(f"**{home_team}** {home_score} — {away_score} **{away_team}**")
    st.markdown(f"*{selected_match['match_date']}*" if 'match_date' in selected_match else "")


# ─────────────────────────────────────────────
# CHARGEMENT DONNÉES DU MATCH
# ─────────────────────────────────────────────

with st.spinner("Chargement des données du match..."):
    events   = cached_events(match_id)
    shots    = cached_shots(match_id)
    p_stats  = cached_player_stats(match_id)
    t_stats  = cached_team_stats(match_id)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown(f"# {home_team}  {home_score} — {away_score}  {away_team}")
st.markdown(f"*{selected_comp} · {selected_season}*")
st.markdown("---")

# KPIs globaux
home_t = t_stats[t_stats["team"] == home_team]
away_t = t_stats[t_stats["team"] == away_team]

def safe_val(df, col, default=0):
    return df[col].iloc[0] if not df.empty and col in df.columns else default

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.metric("Tirs (dom.)", safe_val(home_t, "total_shots"))
with col2:
    st.metric("xG (dom.)", f"{safe_val(home_t, 'total_xG'):.2f}")
with col3:
    st.metric("Passes (dom.)", safe_val(home_t, "total_passes"))
with col4:
    st.metric("Tirs (ext.)", safe_val(away_t, "total_shots"))
with col5:
    st.metric("xG (ext.)", f"{safe_val(away_t, 'total_xG'):.2f}")
with col6:
    st.metric("Passes (ext.)", safe_val(away_t, "total_passes"))

st.markdown("---")


# ─────────────────────────────────────────────
# ONGLETS PRINCIPAUX
# ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📊 Vue Match", "👤 Analyse Joueur", "🔗 Réseau de Passes"])


# ── TAB 1 : VUE MATCH ──────────────────────────────────────

with tab1:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Shot Map")
        team_filter = st.radio("Équipe", ["Les deux", home_team, away_team],
                               horizontal=True, key="shot_team")
        shots_filtered = shots.copy()
        if team_filter != "Les deux":
            shots_filtered = shots[shots["team"] == team_filter]

        fig_shots = shot_map(shots_filtered,
                             title=f"Shot Map — {team_filter}")
        st.plotly_chart(fig_shots, use_container_width=True)

    with col_right:
        st.subheader("xG Cumulé")
        fig_xg = xg_timeline(shots, home_team, away_team,
                             title="Évolution xG cumulé")
        st.plotly_chart(fig_xg, use_container_width=True)

    st.subheader("Statistiques comparatives")
    if not t_stats.empty:
        display_cols = [c for c in ["team", "total_shots", "total_xG", "total_goals",
                                     "total_passes", "avg_pass_pct", "total_dribbles"]
                        if c in t_stats.columns]
        st.dataframe(
            t_stats[display_cols].rename(columns={
                "team": "Équipe", "total_shots": "Tirs",
                "total_xG": "xG", "total_goals": "Buts",
                "total_passes": "Passes", "avg_pass_pct": "Passes %",
                "total_dribbles": "Dribbles"
            }),
            use_container_width=True, hide_index=True
        )


# ── TAB 2 : ANALYSE JOUEUR ─────────────────────────────────

with tab2:
    players = p_stats["player"].dropna().tolist()

    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.subheader("Sélection")
        selected_player = st.selectbox("Joueur principal", players)
        compare_players = st.multiselect(
            "Comparer avec (max 2)",
            [p for p in players if p != selected_player],
            max_selections=2
        )

        # Stats du joueur sélectionné
        player_row = p_stats[p_stats["player"] == selected_player]
        if not player_row.empty:
            r = player_row.iloc[0]
            st.markdown("**Stats du match**")
            st.metric("xG", f"{r.get('xG', 0):.2f}")
            st.metric("Tirs", int(r.get("shots", 0)))
            st.metric("Passes réussies", f"{r.get('passes_complete', 0)}/{r.get('passes_total', 0)}")
            st.metric("Passes %", f"{r.get('pass_pct', 0):.1f}%")
            st.metric("Dribbles", f"{r.get('dribbles_success', 0)}/{r.get('dribbles_attempted', 0)}")

    with col_b:
        st.subheader(f"Heatmap — {selected_player}")
        player_events = events[events["player"] == selected_player].dropna(subset=["x", "y"])
        fig_heat = heatmap(player_events, player_name=selected_player)
        st.plotly_chart(fig_heat, use_container_width=True)

    # Radar comparatif
    if compare_players:
        st.subheader("Radar comparatif")
        all_players = [selected_player] + compare_players
        radar_data = normalize_stats_for_radar(p_stats, all_players)
        if radar_data:
            fig_radar = player_radar(radar_data,
                                     title=f"Comparaison — {' vs '.join(all_players)}")
            st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader("Tableau complet des joueurs")
    display_p_cols = [c for c in ["player", "team", "shots", "xG", "passes_total",
                                   "pass_pct", "dribbles_attempted", "dribbles_success"]
                      if c in p_stats.columns]
    st.dataframe(
        p_stats[display_p_cols].rename(columns={
            "player": "Joueur", "team": "Équipe",
            "shots": "Tirs", "xG": "xG",
            "passes_total": "Passes", "pass_pct": "Passes %",
            "dribbles_attempted": "Dribbles", "dribbles_success": "Dribbles réussis"
        }),
        use_container_width=True, hide_index=True
    )


# ── TAB 3 : RÉSEAU DE PASSES ───────────────────────────────

with tab3:
    st.subheader("Réseau de passes")

    team_net = st.radio("Équipe", [home_team, away_team],
                        horizontal=True, key="net_team")

    passes_team = events[
        (events["type"] == "Pass") &
        (events["team"] == team_net)
    ].dropna(subset=["x", "y"])

    fig_net = pass_network(passes_team,
                           title=f"Réseau de passes — {team_net}")
    st.plotly_chart(fig_net, use_container_width=True)

    st.caption("Taille des nœuds = nombre de passes effectuées · Épaisseur des liens = fréquence de la combinaison")