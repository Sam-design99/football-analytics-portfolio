"""
visualizations.py
=================
Visualisations interactives pour le dashboard football.
Toutes les fonctions retournent un objet Plotly Figure.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde


# ─────────────────────────────────────────────
# CONSTANTES TERRAIN
# ─────────────────────────────────────────────

# StatsBomb : terrain 120x80 yards
PITCH_LENGTH = 120
PITCH_WIDTH = 80

PITCH_DARK = "#1a6b3a"
PITCH_LINE = "rgba(255,255,255,0.7)"

# Couleurs pour les équipes / outcomes
COLORS = {
    "goal":       "#EF9F27",
    "saved":      "#378ADD",
    "missed":     "#E24B4A",
    "blocked":    "#888780",
    "home":       "#1D9E75",
    "away":       "#D85A30",
}


# ─────────────────────────────────────────────
# UTILITAIRES — dessin du terrain
# ─────────────────────────────────────────────

def _draw_pitch(fig, pitch_color=PITCH_DARK):
    """Ajoute les lignes du terrain à une figure Plotly existante."""

    def line(x0, y0, x1, y1):
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color=PITCH_LINE, width=1.5))

    def rect(x0, y0, x1, y1):
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color=PITCH_LINE, width=1.5),
                      fillcolor="rgba(0,0,0,0)")

    def circle(cx, cy, r):
        fig.add_shape(type="circle", x0=cx-r, y0=cy-r, x1=cx+r, y1=cy+r,
                      line=dict(color=PITCH_LINE, width=1.5),
                      fillcolor="rgba(0,0,0,0)")

    def dot(x, y):
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers",
                                 marker=dict(color=PITCH_LINE, size=4),
                                 showlegend=False, hoverinfo="skip"))

    # Contour
    rect(0, 0, PITCH_LENGTH, PITCH_WIDTH)
    # Ligne médiane
    line(PITCH_LENGTH/2, 0, PITCH_LENGTH/2, PITCH_WIDTH)
    # Cercle central
    circle(PITCH_LENGTH/2, PITCH_WIDTH/2, 10)
    dot(PITCH_LENGTH/2, PITCH_WIDTH/2)

    # Surface de réparation gauche
    rect(0, 18, 18, 62)
    rect(0, 30, 6, 50)
    circle(11, PITCH_WIDTH/2, 10)
    dot(11, PITCH_WIDTH/2)

    # Surface de réparation droite
    rect(PITCH_LENGTH, 18, PITCH_LENGTH-18, 62)
    rect(PITCH_LENGTH, 30, PITCH_LENGTH-6, 50)
    circle(PITCH_LENGTH-11, PITCH_WIDTH/2, 10)
    dot(PITCH_LENGTH-11, PITCH_WIDTH/2)

    # Buts
    rect(0, 36, -2, 44)
    rect(PITCH_LENGTH, 36, PITCH_LENGTH+2, 44)


def _pitch_layout(fig, title="", pitch_color=PITCH_DARK):
    """Applique le style terrain à la figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#ffffff"), x=0.5),
        paper_bgcolor="#111111",
        plot_bgcolor=pitch_color,
        font=dict(color="#ffffff", family="sans-serif"),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(range=[-5, PITCH_LENGTH+5], showgrid=False,
                   zeroline=False, showticklabels=False),
        yaxis=dict(range=[-5, PITCH_WIDTH+5], showgrid=False,
                   zeroline=False, showticklabels=False, scaleanchor="x"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", font=dict(size=11)),
    )
    return fig


# ─────────────────────────────────────────────
# 1. SHOT MAP — tirs & xG
# ─────────────────────────────────────────────

def shot_map(shots_df: pd.DataFrame, title: str = "Shot Map") -> go.Figure:
    """
    Carte des tirs avec xG.
    - Taille du cercle = probabilité de but (xG)
    - Couleur = résultat (but / arrêté / raté / bloqué)
    - Étoile = but
    """
    fig = go.Figure()
    _draw_pitch(fig)

    outcome_map = {
        "Goal":    ("goal",    "★ But"),
        "Saved":   ("saved",   "Arrêté"),
        "Off T":   ("missed",  "Raté"),
        "Missed":  ("missed",  "Raté"),
        "Blocked": ("blocked", "Bloqué"),
        "Wayward": ("missed",  "Raté"),
    }

    for outcome, (color_key, label) in outcome_map.items():
        subset = shots_df[shots_df["shot_outcome"] == outcome] if "shot_outcome" in shots_df.columns else pd.DataFrame()
        if subset.empty:
            continue

        xg_vals = subset["shot_statsbomb_xg"].fillna(0.05) if "shot_statsbomb_xg" in subset.columns else pd.Series([0.05]*len(subset))
        sizes = (xg_vals * 400).clip(8, 120)
        symbol = "star" if outcome == "Goal" else "circle"

        hover = [
            f"<b>{row.get('player', 'N/A')}</b><br>"
            f"Minute : {row.get('minute', '?')}'<br>"
            f"xG : {row.get('shot_statsbomb_xg', 0):.2f}<br>"
            f"Technique : {row.get('shot_technique', 'N/A')}"
            for _, row in subset.iterrows()
        ]

        fig.add_trace(go.Scatter(
            x=subset["x"], y=subset["y"],
            mode="markers",
            name=label,
            marker=dict(
                symbol=symbol,
                size=sizes,
                color=COLORS[color_key],
                line=dict(color="white", width=1),
                opacity=0.85,
            ),
            text=hover,
            hoverinfo="text",
        ))

    _pitch_layout(fig, title)
    return fig


# ─────────────────────────────────────────────
# 2. HEATMAP — densité de présence
# ─────────────────────────────────────────────

def heatmap(actions_df: pd.DataFrame, player_name: str = "",
            title: str = "Heatmap") -> go.Figure:
    """
    Heatmap de densité des actions d'un joueur/équipe.
    Utilise un KDE (kernel density estimation) pour lisser.
    """
    fig = go.Figure()

    df = actions_df.dropna(subset=["x", "y"])
    if df.empty:
        fig.add_annotation(text="Pas de données disponibles",
                           x=60, y=40, showarrow=False,
                           font=dict(color="white", size=14))
        _pitch_layout(fig, title)
        _draw_pitch(fig)
        return fig

    x = df["x"].values
    y = df["y"].values

    # KDE sur une grille
    xi = np.linspace(0, PITCH_LENGTH, 120)
    yi = np.linspace(0, PITCH_WIDTH, 80)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    positions = np.vstack([xi_grid.ravel(), yi_grid.ravel()])

    try:
        kernel = gaussian_kde(np.vstack([x, y]), bw_method=0.15)
        zi = kernel(positions).reshape(xi_grid.shape)
    except Exception:
        zi = np.zeros_like(xi_grid)

    fig.add_trace(go.Heatmap(
        x=xi, y=yi, z=zi,
        colorscale=[
            [0.0, "rgba(0,0,0,0)"],
            [0.3, "rgba(255,200,0,0.3)"],
            [0.6, "rgba(255,120,0,0.6)"],
            [1.0, "rgba(220,30,30,0.9)"],
        ],
        showscale=False,
        hoverinfo="skip",
    ))

    # Points individuels (semi-transparents)
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(color="white", size=3, opacity=0.2),
        showlegend=False,
        hoverinfo="skip",
    ))

    _draw_pitch(fig)
    t = title if title else f"Heatmap — {player_name}"
    _pitch_layout(fig, t)
    return fig


# ─────────────────────────────────────────────
# 3. RADAR — comparaison de joueurs
# ─────────────────────────────────────────────

def player_radar(players_stats: dict, title: str = "Comparaison joueurs") -> go.Figure:
    """
    Radar chart pour comparer jusqu'à 3 joueurs.

    players_stats : dict {nom_joueur: {stat: valeur, ...}}
    Les valeurs doivent être normalisées entre 0 et 100.

    Exemple :
        players_stats = {
            "Messi": {"xG": 85, "Passes%": 90, "Dribbles": 95, "Tirs": 80, "Vitesse": 70},
            "Ronaldo": {"xG": 90, "Passes%": 75, "Dribbles": 70, "Tirs": 95, "Vitesse": 85},
        }
    """
    if not players_stats:
        return go.Figure()

    # Récupère les catégories depuis le premier joueur
    categories = list(next(iter(players_stats.values())).keys())
    if not categories:
        return go.Figure()
    categories_closed = categories + [categories[0]]  # fermer le polygone

    palette = ["#1D9E75", "#EF9F27", "#378ADD", "#E24B4A"]
    fig = go.Figure()

    for i, (player, stats) in enumerate(players_stats.items()):
        values = [stats.get(cat, 0) for cat in categories]
        values_closed = values + [values[0]]
        color = palette[i % len(palette)]

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            name=player,
            line=dict(color=color, width=2),
            fillcolor=color.replace(")", ", 0.15)").replace("rgb", "rgba") if "rgb" in color else color + "26",
            marker=dict(size=5, color=color),
            hovertemplate="<b>%{theta}</b><br>%{r:.0f}/100<extra>" + player + "</extra>",
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickfont=dict(size=9, color="#aaaaaa"),
                gridcolor="rgba(255,255,255,0.15)",
                linecolor="rgba(255,255,255,0.1)",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#ffffff"),
                gridcolor="rgba(255,255,255,0.15)",
                linecolor="rgba(255,255,255,0.2)",
            ),
            bgcolor="#1a1a2e",
        ),
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        title=dict(text=title, font=dict(size=15, color="#ffffff"), x=0.5),
        font=dict(color="#ffffff"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", font=dict(size=11)),
        margin=dict(l=60, r=60, t=60, b=60),
    )
    return fig


def normalize_stats_for_radar(season_df: pd.DataFrame,
                               player_names: list,
                               stats_cols: list = None) -> dict:
    """
    Normalise les stats de joueurs sélectionnés (0-100) pour le radar.
    Basé sur le percentile dans le dataset complet.
    """
    if stats_cols is None:
        stats_cols = ["total_xG", "avg_pass_pct", "total_dribbles_success",
                      "shots_per_match", "xG_per_match"]

    labels = {
        "total_xG": "xG total",
        "avg_pass_pct": "Passes %",
        "total_dribbles_success": "Dribbles réussis",
        "shots_per_match": "Tirs/match",
        "xG_per_match": "xG/match",
        "dribble_success_rate": "Réussite dribble",
    }

    available = [c for c in stats_cols if c in season_df.columns]
    result = {}

    for player in player_names:
        row = season_df[season_df["player"] == player]
        if row.empty:
            continue
        stats = {}
        for col in available:
            col_max = season_df[col].max()
            val = row[col].iloc[0]
            stats[labels.get(col, col)] = round((val / col_max * 100) if col_max > 0 else 0, 1)
        result[player] = stats

    return result


# ─────────────────────────────────────────────
# 4. RÉSEAU DE PASSES
# ─────────────────────────────────────────────

def pass_network(passes_df: pd.DataFrame, events_df: pd.DataFrame = None,
                 title: str = "Réseau de passes") -> go.Figure:
    """
    Réseau de passes entre joueurs.
    - Nœuds = joueurs (position moyenne sur le terrain)
    - Liens = nombre de passes entre deux joueurs (épaisseur = fréquence)
    """
    fig = go.Figure()
    _draw_pitch(fig)

    df = passes_df.dropna(subset=["x", "y", "player"])
    if df.empty or "pass_recipient" not in df.columns:
        _pitch_layout(fig, title)
        return fig

    df = df.dropna(subset=["pass_recipient"])

    # Position moyenne de chaque joueur (là où il fait ses passes)
    avg_pos = df.groupby("player")[["x", "y"]].mean().reset_index()
    avg_pos.columns = ["player", "avg_x", "avg_y"]

    # Compter les combinaisons de passes
    pairs = df.groupby(["player", "pass_recipient"]).size().reset_index(name="count")
    pairs = pairs[pairs["count"] >= 2]  # filtrer les anecdotiques

    max_count = pairs["count"].max() if not pairs.empty else 1

    # Tracer les liens
    for _, row in pairs.iterrows():
        p1 = avg_pos[avg_pos["player"] == row["player"]]
        p2 = avg_pos[avg_pos["player"] == row["pass_recipient"]]
        if p1.empty or p2.empty:
            continue

        width = (row["count"] / max_count) * 6 + 0.5
        opacity = 0.3 + (row["count"] / max_count) * 0.5

        fig.add_trace(go.Scatter(
            x=[p1["avg_x"].values[0], p2["avg_x"].values[0]],
            y=[p1["avg_y"].values[0], p2["avg_y"].values[0]],
            mode="lines",
            line=dict(color=f"rgba(255,255,255,{opacity:.2f})", width=width),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Compter le total de passes par joueur (taille des nœuds)
    pass_count = df.groupby("player").size().reset_index(name="total_passes")
    avg_pos = avg_pos.merge(pass_count, on="player", how="left")
    avg_pos["total_passes"] = avg_pos["total_passes"].fillna(0)
    max_passes = avg_pos["total_passes"].max()

    # Tracer les nœuds
    node_sizes = ((avg_pos["total_passes"] / max_passes) * 25 + 8).clip(8, 35)
    short_names = avg_pos["player"].apply(
        lambda n: n.split()[-1] if isinstance(n, str) and " " in n else n
    )

    fig.add_trace(go.Scatter(
        x=avg_pos["avg_x"],
        y=avg_pos["avg_y"],
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color="#1D9E75",
            line=dict(color="white", width=1.5),
            opacity=0.9,
        ),
        text=short_names,
        textposition="top center",
        textfont=dict(size=9, color="white"),
        name="Joueur",
        customdata=avg_pos[["player", "total_passes"]].values,
        hovertemplate="<b>%{customdata[0]}</b><br>Passes : %{customdata[1]:.0f}<extra></extra>",
    ))

    _pitch_layout(fig, title)
    return fig


# ─────────────────────────────────────────────
# 5. XG TIMELINE — évolution xG cumulé dans le match
# ─────────────────────────────────────────────

def xg_timeline(shots_df: pd.DataFrame, home_team: str, away_team: str,
                title: str = "Évolution xG cumulé") -> go.Figure:
    """
    Graphique d'évolution du xG cumulé au fil du match pour les deux équipes.
    """
    if shots_df.empty or "shot_statsbomb_xg" not in shots_df.columns:
        return go.Figure()

    shots = shots_df.copy().sort_values("minute")
    shots["xg"] = shots["shot_statsbomb_xg"].fillna(0)

    home = shots[shots["team"] == home_team].copy()
    away = shots[shots["team"] == away_team].copy()

    # Points de début et fin pour chaque équipe
    def build_cumulative(team_shots):
        minutes = [0] + list(team_shots["minute"]) + [90]
        xg_cum = [0] + list(team_shots["xg"].cumsum()) + [team_shots["xg"].sum()]
        return minutes, xg_cum

    home_min, home_xg = build_cumulative(home)
    away_min, away_xg = build_cumulative(away)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=home_min, y=home_xg,
        mode="lines",
        name=home_team,
        line=dict(color=COLORS["home"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(29,158,117,0.15)",
        hovertemplate="Minute %{x}' · xG cumulé : %{y:.2f}<extra>" + home_team + "</extra>",
    ))

    fig.add_trace(go.Scatter(
        x=away_min, y=away_xg,
        mode="lines",
        name=away_team,
        line=dict(color=COLORS["away"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(216,90,48,0.15)",
        hovertemplate="Minute %{x}' · xG cumulé : %{y:.2f}<extra>" + away_team + "</extra>",
    ))

    # Annotations buts
    goals = shots_df[shots_df.get("shot_outcome", pd.Series()) == "Goal"] if "shot_outcome" in shots_df.columns else pd.DataFrame()
    for _, g in goals.iterrows():
        color = COLORS["home"] if g["team"] == home_team else COLORS["away"]
        fig.add_vline(x=g["minute"], line=dict(color=color, dash="dot", width=1.5),
                      annotation_text=f"⚽ {g['minute']}'",
                      annotation_font=dict(size=9, color=color))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#ffffff"), x=0.5),
        paper_bgcolor="#111111",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#ffffff"),
        xaxis=dict(title="Minute", range=[0, 95], gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="xG cumulé", gridcolor="rgba(255,255,255,0.08)"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)"),
        margin=dict(l=50, r=30, t=50, b=50),
    )
    return fig


# ─────────────────────────────────────────────
# TEST RAPIDE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    from data_pipeline import get_matches, get_shots, get_passes, get_events

    COMPETITION_ID = 43
    SEASON_ID = 3

    print("Chargement des donnees...")
    matches = get_matches(COMPETITION_ID, SEASON_ID)
    match = matches.iloc[0]
    match_id = int(match["match_id"])
    home = match["home_team"]
    away = match["away_team"]
    print(f"Match selectionne : {home} vs {away} (id={match_id})")

    # Chargement events + coordonnees
    print("Chargement des evenements...")
    events = get_events(match_id)
    events["x"] = events["location"].apply(lambda l: l[0] if isinstance(l, list) else None)
    events["y"] = events["location"].apply(lambda l: l[1] if isinstance(l, list) else None)

    shots = get_shots(match_id)
    passes_full = events[events["type"] == "Pass"].dropna(subset=["x", "y"])

    # 1. Shot map
    print("Generation shot map...")
    fig_shots = shot_map(shots, title=f"Shot Map - {home} vs {away}")
    fig_shots.write_html("test_shot_map.html")
    print("  -> test_shot_map.html OK")

    # 2. Heatmap premier joueur
    print("Generation heatmap...")
    first_player = events["player"].dropna().iloc[0]
    player_ev = events[events["player"] == first_player].dropna(subset=["x", "y"])
    fig_heat = heatmap(player_ev, player_name=first_player,
                       title=f"Heatmap - {first_player}")
    fig_heat.write_html("test_heatmap.html")
    print(f"  -> test_heatmap.html OK ({first_player})")

    # 3. xG timeline
    print("Generation xG timeline...")
    fig_xg = xg_timeline(shots, home, away,
                         title=f"xG cumule - {home} vs {away}")
    fig_xg.write_html("test_xg_timeline.html")
    print("  -> test_xg_timeline.html OK")

    # 4. Reseau de passes
    print("Generation reseau de passes...")
    passes_home = passes_full[passes_full["team"] == home]
    fig_net = pass_network(passes_home, title=f"Reseau de passes - {home}")
    fig_net.write_html("test_pass_network.html")
    print("  -> test_pass_network.html OK")

    print("\nTous les graphiques ont ete generes avec succes !")
    print("Lance : start test_shot_map.html pour les ouvrir")