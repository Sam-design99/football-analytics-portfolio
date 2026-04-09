"""
event_extractor.py
==================
Extraction automatique d'événements footballistiques
à partir des données de tracking (trajectoires joueurs).

Événements détectés :
    - Sprint        : accélération > seuil pendant N frames
    - Arrêt         : vitesse quasi nulle après mouvement
    - Changement de direction : angle brusque dans la trajectoire
    - Regroupement  : plusieurs joueurs proches (duel potentiel)
    - Zone d'action : zones fréquentées sur le terrain
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ─────────────────────────────────────────────
# CONFIGURATION — seuils paramétrables
# ─────────────────────────────────────────────

# Seuils de vitesse (en pixels/frame)
SPRINT_THRESHOLD    = 8.0    # vitesse min pour un sprint
WALK_THRESHOLD      = 2.0    # vitesse max pour considérer arrêt/marche
SPRINT_MIN_FRAMES   = 5      # durée min d'un sprint (frames consécutives)

# Seuil de proximité (en pixels) pour détecter un duel
DUEL_DISTANCE_PX    = 60

# Seuil d'angle pour changement de direction (degrés)
DIR_CHANGE_THRESHOLD = 60.0


# ─────────────────────────────────────────────
# 1. CALCUL DE VITESSE ET DIRECTION
# ─────────────────────────────────────────────

def compute_kinematics(
    tracking_df: pd.DataFrame,
    fps: float = 25.0,
    smooth: bool = True
) -> pd.DataFrame:
    """
    Ajoute les colonnes cinématiques au DataFrame de tracking :
        speed      : vitesse en pixels/frame (lissée)
        direction  : angle de déplacement en degrés
        accel      : accélération (variation de vitesse)

    smooth=True : applique un filtre Savitzky-Golay pour réduire le bruit.
    """
    df = tracking_df.copy().sort_values(['track_id', 'frame'])

    df['dx'] = df.groupby('track_id')['cx'].diff().fillna(0)
    df['dy'] = df.groupby('track_id')['cy'].diff().fillna(0)
    df['speed_raw'] = np.sqrt(df['dx']**2 + df['dy']**2)

    # Lissage par joueur
    if smooth:
        def _smooth(series):
            n = len(series)
            if n < 7:
                return series
            wl = min(7, n if n % 2 != 0 else n - 1)
            return pd.Series(
                savgol_filter(series.values, window_length=wl, polyorder=2),
                index=series.index
            )
        df['speed'] = df.groupby('track_id')['speed_raw'].transform(_smooth)
    else:
        df['speed'] = df['speed_raw']

    df['speed'] = df['speed'].clip(lower=0).round(3)

    # Direction (angle en degrés, 0 = droite, 90 = bas)
    df['direction'] = np.degrees(np.arctan2(df['dy'], df['dx'])).round(1)

    # Accélération (variation de vitesse)
    df['accel'] = df.groupby('track_id')['speed'].diff().fillna(0).round(3)

    # Vitesse en km/h (estimation — dépend de la résolution vidéo)
    # On fournit la valeur brute en px/frame, la conversion nécessite
    # la calibration du terrain (pixels → mètres)
    df['speed_px_s'] = (df['speed'] * fps).round(2)

    return df


# ─────────────────────────────────────────────
# 2. DÉTECTION DES SPRINTS
# ─────────────────────────────────────────────

def detect_sprints(
    kinematics_df: pd.DataFrame,
    threshold: float = SPRINT_THRESHOLD,
    min_frames: int = SPRINT_MIN_FRAMES
) -> pd.DataFrame:
    """
    Détecte les séquences de sprint par joueur.

    Retourne un DataFrame avec :
        track_id, sprint_id, start_frame, end_frame,
        duration_frames, peak_speed, avg_speed, cx_start, cy_start
    """
    sprints = []

    for track_id, group in kinematics_df.groupby('track_id'):
        group = group.sort_values('frame').reset_index(drop=True)
        is_sprint = group['speed'] >= threshold

        sprint_id = 0
        in_sprint = False
        start_idx = None

        for i, sprinting in enumerate(is_sprint):
            if sprinting and not in_sprint:
                in_sprint = True
                start_idx = i
            elif not sprinting and in_sprint:
                in_sprint = False
                segment = group.iloc[start_idx:i]
                if len(segment) >= min_frames:
                    sprint_id += 1
                    sprints.append({
                        'track_id':       track_id,
                        'sprint_id':      sprint_id,
                        'start_frame':    int(segment['frame'].iloc[0]),
                        'end_frame':      int(segment['frame'].iloc[-1]),
                        'start_time_sec': round(float(segment['time_sec'].iloc[0]), 2),
                        'end_time_sec':   round(float(segment['time_sec'].iloc[-1]), 2),
                        'duration_frames': len(segment),
                        'peak_speed':     round(float(segment['speed'].max()), 2),
                        'avg_speed':      round(float(segment['speed'].mean()), 2),
                        'cx_start':       int(segment['cx'].iloc[0]),
                        'cy_start':       int(segment['cy'].iloc[0]),
                        'cx_end':         int(segment['cx'].iloc[-1]),
                        'cy_end':         int(segment['cy'].iloc[-1]),
                    })

        # Sprint en cours à la fin de la vidéo
        if in_sprint and start_idx is not None:
            segment = group.iloc[start_idx:]
            if len(segment) >= min_frames:
                sprint_id += 1
                sprints.append({
                    'track_id':       track_id,
                    'sprint_id':      sprint_id,
                    'start_frame':    int(segment['frame'].iloc[0]),
                    'end_frame':      int(segment['frame'].iloc[-1]),
                    'start_time_sec': round(float(segment['time_sec'].iloc[0]), 2),
                    'end_time_sec':   round(float(segment['time_sec'].iloc[-1]), 2),
                    'duration_frames': len(segment),
                    'peak_speed':     round(float(segment['speed'].max()), 2),
                    'avg_speed':      round(float(segment['speed'].mean()), 2),
                    'cx_start':       int(segment['cx'].iloc[0]),
                    'cy_start':       int(segment['cy'].iloc[0]),
                    'cx_end':         int(segment['cx'].iloc[-1]),
                    'cy_end':         int(segment['cy'].iloc[-1]),
                })

    return pd.DataFrame(sprints)


# ─────────────────────────────────────────────
# 3. DÉTECTION DES CHANGEMENTS DE DIRECTION
# ─────────────────────────────────────────────

def detect_direction_changes(
    kinematics_df: pd.DataFrame,
    angle_threshold: float = DIR_CHANGE_THRESHOLD,
    min_speed: float = WALK_THRESHOLD
) -> pd.DataFrame:
    """
    Détecte les changements de direction brusques.
    Utile pour identifier dribbles et feintes.
    """
    changes = []

    for track_id, group in kinematics_df.groupby('track_id'):
        group = group.sort_values('frame').reset_index(drop=True)

        # Ne considérer que les frames avec du mouvement
        moving = group[group['speed'] >= min_speed].copy()
        if len(moving) < 3:
            continue

        dir_diff = moving['direction'].diff().abs()

        # Corriger les sauts 360° → -0°
        dir_diff = dir_diff.apply(lambda a: 360 - a if a > 180 else a)

        sharp_turns = moving[dir_diff >= angle_threshold]

        for _, row in sharp_turns.iterrows():
            changes.append({
                'track_id':   track_id,
                'frame':      int(row['frame']),
                'time_sec':   row['time_sec'],
                'cx':         int(row['cx']),
                'cy':         int(row['cy']),
                'angle_change': round(float(dir_diff.loc[row.name]), 1),
                'speed':      round(float(row['speed']), 2),
                'event_type': 'direction_change',
            })

    return pd.DataFrame(changes)


# ─────────────────────────────────────────────
# 4. DÉTECTION DES DUELS (joueurs proches)
# ─────────────────────────────────────────────

def detect_duels(
    tracking_df: pd.DataFrame,
    distance_threshold: float = DUEL_DISTANCE_PX
) -> pd.DataFrame:
    """
    Détecte les frames où deux joueurs sont très proches (duel potentiel).
    """
    duels = []

    for frame_nb, frame_group in tracking_df.groupby('frame'):
        players = frame_group[['track_id', 'cx', 'cy']].values

        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                id1, x1, y1 = players[i]
                id2, x2, y2 = players[j]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                if dist <= distance_threshold:
                    duels.append({
                        'frame':      int(frame_nb),
                        'track_id_1': int(id1),
                        'track_id_2': int(id2),
                        'distance_px': round(float(dist), 1),
                        'cx_mid':     int((x1 + x2) / 2),
                        'cy_mid':     int((y1 + y2) / 2),
                        'event_type': 'duel',
                    })

    return pd.DataFrame(duels)


# ─────────────────────────────────────────────
# 5. ZONES D'ACTION — heatmap par joueur
# ─────────────────────────────────────────────

def compute_zones(
    tracking_df: pd.DataFrame,
    frame_width: int = 1280,
    frame_height: int = 720,
    grid_cols: int = 6,
    grid_rows: int = 4
) -> pd.DataFrame:
    """
    Divise le terrain en zones et compte les passages de chaque joueur.
    Retourne un DataFrame avec la distribution spatiale par track_id.
    """
    zone_records = []

    col_w = frame_width  / grid_cols
    row_h = frame_height / grid_rows

    for track_id, group in tracking_df.groupby('track_id'):
        zone_counts = np.zeros((grid_rows, grid_cols), dtype=int)

        for _, row in group.iterrows():
            col_idx = min(int(row['cx'] / col_w), grid_cols - 1)
            row_idx = min(int(row['cy'] / row_h), grid_rows - 1)
            zone_counts[row_idx, col_idx] += 1

        total = zone_counts.sum()
        dominant_zone = np.unravel_index(zone_counts.argmax(), zone_counts.shape)

        zone_records.append({
            'track_id':       track_id,
            'total_frames':   int(total),
            'dominant_col':   int(dominant_zone[1]),
            'dominant_row':   int(dominant_zone[0]),
            'zone_matrix':    zone_counts.tolist(),
            'left_pct':       round(zone_counts[:, :grid_cols//2].sum() / total * 100, 1) if total else 0,
            'right_pct':      round(zone_counts[:, grid_cols//2:].sum() / total * 100, 1) if total else 0,
            'attack_pct':     round(zone_counts[:, grid_cols*2//3:].sum() / total * 100, 1) if total else 0,
            'defense_pct':    round(zone_counts[:, :grid_cols//3].sum() / total * 100, 1) if total else 0,
        })

    return pd.DataFrame(zone_records)


# ─────────────────────────────────────────────
# 6. EXPORT FORMAT STATSBOMB-COMPATIBLE
# ─────────────────────────────────────────────

def export_events(
    sprints_df: pd.DataFrame,
    direction_changes_df: pd.DataFrame,
    duels_df: pd.DataFrame,
    output_path: str = None,
    fps: float = 25.0,
    video_name: str = "video"
) -> pd.DataFrame:
    """
    Combine tous les événements détectés dans un DataFrame unifié
    au format proche de StatsBomb pour intégration avec le dashboard.

    Colonnes : type, track_id, frame, time_sec, minute, cx, cy, data
    """
    events = []

    # Sprints
    for _, row in sprints_df.iterrows():
        events.append({
            'source':   'video',
            'video':    video_name,
            'type':     'Sprint',
            'track_id': row['track_id'],
            'frame':    row['start_frame'],
            'time_sec': row['start_time_sec'],
            'minute':   int(row['start_time_sec'] // 60),
            'cx':       row.get('cx_start', None),
            'cy':       row.get('cy_start', None),
            'data':     f"peak_speed={row['peak_speed']} duration={row['duration_frames']}f",
        })

    # Changements de direction
    for _, row in direction_changes_df.iterrows():
        events.append({
            'source':   'video',
            'video':    video_name,
            'type':     'Direction change',
            'track_id': row['track_id'],
            'frame':    row['frame'],
            'time_sec': row['time_sec'],
            'minute':   int(row['time_sec'] // 60),
            'cx':       row.get('cx', None),
            'cy':       row.get('cy', None),
            'data':     f"angle={row['angle_change']}deg speed={row['speed']}",
        })

    # Duels
    for _, row in duels_df.iterrows():
        events.append({
            'source':   'video',
            'video':    video_name,
            'type':     'Duel',
            'track_id': row['track_id_1'],
            'frame':    row['frame'],
            'time_sec': round(row['frame'] / fps, 2),
            'minute':   int(row['frame'] / fps // 60),
            'cx':       row.get('cx_mid', None),
            'cy':       row.get('cy_mid', None),
            'data':     f"opponent={row['track_id_2']} dist={row['distance_px']}px",
        })

    df = pd.DataFrame(events).sort_values('time_sec').reset_index(drop=True)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Événements exportés : {output_path} ({len(df)} événements)")

    return df


# ─────────────────────────────────────────────
# 7. PIPELINE COMPLET
# ─────────────────────────────────────────────

def run_full_analysis(
    tracking_csv: str,
    fps: float = 25.0,
    output_dir: str = None,
    frame_width: int = 1280,
    frame_height: int = 720,
) -> dict:
    """
    Pipeline complet d'extraction d'événements depuis un CSV de tracking.

    Paramètres :
        tracking_csv : chemin vers le CSV généré par video_tracker.py
        fps          : framerate de la vidéo originale
        output_dir   : dossier de sortie pour les CSV d'événements

    Retourne un dict avec : kinematics, sprints, direction_changes, duels, zones, events
    """
    print(f"Lecture tracking : {tracking_csv}")
    df = pd.read_csv(tracking_csv)
    print(f"  {len(df)} lignes · {df['track_id'].nunique()} joueurs · {df['frame'].nunique()} frames")

    out_dir = output_dir or os.path.dirname(tracking_csv)
    base    = os.path.splitext(os.path.basename(tracking_csv))[0].replace('_tracking', '')

    print("\n1. Calcul cinématique...")
    kine = compute_kinematics(df, fps=fps)

    print("2. Détection sprints...")
    sprints = detect_sprints(kine)
    print(f"   → {len(sprints)} sprints détectés")

    print("3. Détection changements de direction...")
    dir_changes = detect_direction_changes(kine)
    print(f"   → {len(dir_changes)} changements détectés")

    print("4. Détection duels...")
    duels = detect_duels(df)
    duels_dedup = duels.drop_duplicates(subset=['frame', 'track_id_1', 'track_id_2'])
    print(f"   → {len(duels_dedup)} duels détectés")

    print("5. Calcul zones d'action...")
    zones = compute_zones(df, frame_width=frame_width, frame_height=frame_height)

    print("6. Export événements unifiés...")
    events_path = os.path.join(out_dir, f"{base}_events.csv")
    events = export_events(sprints, dir_changes, duels_dedup,
                           output_path=events_path, fps=fps, video_name=base)

    # Sauvegardes individuelles
    sprints.to_csv(os.path.join(out_dir, f"{base}_sprints.csv"), index=False)
    dir_changes.to_csv(os.path.join(out_dir, f"{base}_direction_changes.csv"), index=False)
    zones.to_csv(os.path.join(out_dir, f"{base}_zones.csv"), index=False)

    print(f"\nAnalyse terminée — {len(events)} événements extraits au total.")

    return {
        'kinematics':        kine,
        'sprints':           sprints,
        'direction_changes': dir_changes,
        'duels':             duels_dedup,
        'zones':             zones,
        'events':            events,
    }


# ─────────────────────────────────────────────
# TEST RAPIDE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import glob

    processed_dir = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'processed'
    )

    csvs = glob.glob(os.path.join(processed_dir, '*_tracking.csv'))
    if not csvs:
        print("Aucun fichier de tracking trouvé.")
        print("Lance d'abord video_tracker.py pour générer un CSV.")
    else:
        results = run_full_analysis(
            tracking_csv=csvs[0],
            fps=25.0,
            output_dir=processed_dir,
        )

        print("\n=== Résumé sprints ===")
        if not results['sprints'].empty:
            print(results['sprints'].head(10).to_string())

        print("\n=== Résumé zones ===")
        if not results['zones'].empty:
            print(results['zones'][['track_id', 'total_frames',
                                    'attack_pct', 'defense_pct']].head(10).to_string())