"""
data_pipeline.py
================
Pipeline de données football - Source : StatsBomb Open Data
Fonctions de récupération, nettoyage et structuration des données.
"""

import pandas as pd
import numpy as np
from statsbombpy import sb


# ─────────────────────────────────────────────
# 1. EXPLORATION — lister les compétitions disponibles
# ─────────────────────────────────────────────

def get_competitions() -> pd.DataFrame:
    """Retourne toutes les compétitions disponibles en open data."""
    comps = sb.competitions()
    return comps[["competition_id", "season_id", "competition_name",
                  "season_name", "competition_gender"]].sort_values("competition_name")


def get_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Retourne tous les matchs d'une compétition/saison."""
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    cols = ["match_id", "match_date", "home_team", "away_team",
            "home_score", "away_score", "competition", "season"]
    available = [c for c in cols if c in matches.columns]
    return matches[available].sort_values("match_date", ascending=False)


# ─────────────────────────────────────────────
# 2. ÉVÉNEMENTS — cœur de l'analyse
# ─────────────────────────────────────────────

def get_events(match_id: int) -> pd.DataFrame:
    """Récupère tous les événements d'un match."""
    events = sb.events(match_id=match_id)
    return events


def get_shots(match_id: int) -> pd.DataFrame:
    """Extrait et structure les tirs d'un match avec xG."""
    events = get_events(match_id)
    shots = events[events["type"] == "Shot"].copy()

    # Extraire les données imbriquées (dict dans certaines colonnes)
    if "location" in shots.columns:
        shots["x"] = shots["location"].apply(lambda l: l[0] if isinstance(l, list) else None)
        shots["y"] = shots["location"].apply(lambda l: l[1] if isinstance(l, list) else None)

    if "shot_end_location" in shots.columns:
        shots["end_x"] = shots["shot_end_location"].apply(lambda l: l[0] if isinstance(l, list) else None)
        shots["end_y"] = shots["shot_end_location"].apply(lambda l: l[1] if isinstance(l, list) else None)

    keep = ["id", "minute", "player", "team", "x", "y", "end_x", "end_y",
            "shot_statsbomb_xg", "shot_outcome", "shot_technique", "shot_body_part"]
    available = [c for c in keep if c in shots.columns]
    return shots[available].reset_index(drop=True)


def get_passes(match_id: int, team: str = None) -> pd.DataFrame:
    """Extrait les passes d'un match, filtrables par équipe."""
    events = get_events(match_id)
    passes = events[events["type"] == "Pass"].copy()

    if team:
        passes = passes[passes["team"] == team]

    if "location" in passes.columns:
        passes["x"] = passes["location"].apply(lambda l: l[0] if isinstance(l, list) else None)
        passes["y"] = passes["location"].apply(lambda l: l[1] if isinstance(l, list) else None)

    if "pass_end_location" in passes.columns:
        passes["end_x"] = passes["pass_end_location"].apply(lambda l: l[0] if isinstance(l, list) else None)
        passes["end_y"] = passes["pass_end_location"].apply(lambda l: l[1] if isinstance(l, list) else None)

    keep = ["id", "minute", "player", "team", "x", "y", "end_x", "end_y",
            "pass_outcome", "pass_length", "pass_angle", "pass_recipient"]
    available = [c for c in keep if c in passes.columns]
    return passes[available].reset_index(drop=True)


def get_player_actions(match_id: int, player_name: str) -> pd.DataFrame:
    """Toutes les actions d'un joueur sur un match."""
    events = get_events(match_id)
    player_events = events[events["player"] == player_name].copy()

    if "location" in player_events.columns:
        player_events["x"] = player_events["location"].apply(lambda l: l[0] if isinstance(l, list) else None)
        player_events["y"] = player_events["location"].apply(lambda l: l[1] if isinstance(l, list) else None)

    return player_events.reset_index(drop=True)


# ─────────────────────────────────────────────
# 3. STATISTIQUES — agrégations par joueur / équipe
# ─────────────────────────────────────────────

def compute_player_stats(match_id: int) -> pd.DataFrame:
    """
    Calcule les stats clés par joueur sur un match :
    xG, tirs, passes réussies, dribbles, touches.
    """
    events = get_events(match_id)
    players = events["player"].dropna().unique()
    stats = []

    for player in players:
        p_events = events[events["player"] == player]
        team = p_events["team"].iloc[0] if not p_events.empty else None

        # Tirs & xG
        shots = p_events[p_events["type"] == "Shot"]
        xg = shots["shot_statsbomb_xg"].sum() if "shot_statsbomb_xg" in shots.columns else 0
        goals = shots[shots.get("shot_outcome", pd.Series()) == "Goal"].shape[0] if "shot_outcome" in shots.columns else 0

        # Passes
        passes = p_events[p_events["type"] == "Pass"]
        passes_total = len(passes)
        if "pass_outcome" in passes.columns:
            passes_complete = passes[passes["pass_outcome"].isna()].shape[0]  # NaN = succès chez StatsBomb
        else:
            passes_complete = 0
        pass_pct = round(passes_complete / passes_total * 100, 1) if passes_total > 0 else 0

        # Dribbles
        dribbles = p_events[p_events["type"] == "Dribble"]
        dribbles_success = dribbles[dribbles.get("dribble_outcome", pd.Series()) == "Complete"].shape[0] if "dribble_outcome" in dribbles.columns else 0

        stats.append({
            "player": player,
            "team": team,
            "minutes": int(p_events["minute"].max()) if not p_events.empty else 0,
            "touches": len(p_events),
            "shots": len(shots),
            "goals": goals,
            "xG": round(float(xg), 2),
            "passes_total": passes_total,
            "passes_complete": passes_complete,
            "pass_pct": pass_pct,
            "dribbles_attempted": len(dribbles),
            "dribbles_success": dribbles_success,
        })

    df = pd.DataFrame(stats).sort_values("xG", ascending=False)
    return df.reset_index(drop=True)


def compute_team_stats(match_id: int) -> pd.DataFrame:
    """Stats agrégées par équipe pour un match."""
    player_stats = compute_player_stats(match_id)
    team_stats = player_stats.groupby("team").agg(
        total_shots=("shots", "sum"),
        total_xG=("xG", "sum"),
        total_goals=("goals", "sum"),
        total_passes=("passes_total", "sum"),
        avg_pass_pct=("pass_pct", "mean"),
        total_dribbles=("dribbles_attempted", "sum"),
    ).reset_index()
    team_stats["total_xG"] = team_stats["total_xG"].round(2)
    team_stats["avg_pass_pct"] = team_stats["avg_pass_pct"].round(1)
    return team_stats


# ─────────────────────────────────────────────
# 4. DONNÉES MULTI-MATCHS — analyse sur une saison
# ─────────────────────────────────────────────

def compute_season_player_stats(competition_id: int, season_id: int,
                                 max_matches: int = 10) -> pd.DataFrame:
    """
    Agrège les stats d'un joueur sur plusieurs matchs.
    max_matches : limite pour éviter des temps de chargement trop longs.
    """
    matches = get_matches(competition_id, season_id)
    all_stats = []

    for match_id in matches["match_id"].head(max_matches):
        try:
            stats = compute_player_stats(match_id)
            stats["match_id"] = match_id
            all_stats.append(stats)
        except Exception as e:
            print(f"Erreur match {match_id} : {e}")
            continue

    if not all_stats:
        return pd.DataFrame()

    combined = pd.concat(all_stats, ignore_index=True)
    season_stats = combined.groupby(["player", "team"]).agg(
        matchs_joues=("match_id", "nunique"),
        total_shots=("shots", "sum"),
        total_goals=("goals", "sum"),
        total_xG=("xG", "sum"),
        total_passes=("passes_total", "sum"),
        avg_pass_pct=("pass_pct", "mean"),
        total_dribbles=("dribbles_attempted", "sum"),
        total_dribbles_success=("dribbles_success", "sum"),
    ).reset_index()

    season_stats["xG_per_match"] = (season_stats["total_xG"] / season_stats["matchs_joues"]).round(2)
    season_stats["shots_per_match"] = (season_stats["total_shots"] / season_stats["matchs_joues"]).round(1)
    season_stats["dribble_success_rate"] = (
        season_stats["total_dribbles_success"] / season_stats["total_dribbles"].replace(0, np.nan) * 100
    ).round(1)
    season_stats["total_xG"] = season_stats["total_xG"].round(2)
    season_stats["avg_pass_pct"] = season_stats["avg_pass_pct"].round(1)

    return season_stats.sort_values("total_xG", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# 5. TEST RAPIDE — exécution directe du script
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import io
    # Fix encodage Windows (CMD ne supporte pas UTF-8 par defaut)
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    COMPETITION_ID = 43  # Coupe du Monde 2018
    SEASON_ID = 3

    print("=== Competitions disponibles ===")
    comps = get_competitions()
    print(comps[["competition_name", "season_name", "competition_id", "season_id"]].head(10).to_string())

    print("\n=== Matchs CDM 2018 ===")
    matches = get_matches(COMPETITION_ID, SEASON_ID)
    print(matches[["match_id", "home_team", "away_team", "home_score", "away_score"]].head(5).to_string())

    if not matches.empty:
        match_id = matches["match_id"].iloc[0]
        home = matches["home_team"].iloc[0]
        away = matches["away_team"].iloc[0]
        print(f"\n=== Tirs : {home} vs {away} ===")
        shots = get_shots(match_id)
        print(shots[["minute", "player", "team", "shot_statsbomb_xg", "shot_outcome"]].head(10).to_string())

        print(f"\n=== Stats joueurs ===")
        player_stats = compute_player_stats(match_id)
        print(player_stats[["player", "team", "shots", "xG", "passes_total", "pass_pct"]].head(10).to_string())

        print(f"\n=== Stats equipes ===")
        team_stats = compute_team_stats(match_id)
        print(team_stats.to_string())

    print("\nPipeline OK - toutes les donnees chargees avec succes !")