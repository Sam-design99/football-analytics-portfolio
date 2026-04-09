"""
pitch_calibrator.py
===================
Calibration terrain par homographie manuelle par segment.
Interface deux vues : frame video + schema terrain FIFA.
L'utilisateur pointe des correspondances pour chaque segment
de camera principale detecte par scene_detector.py.
"""

import cv2
import numpy as np
import pandas as pd
import os
import json
from typing import List, Optional, Tuple, Dict
from scene_detector import Segment, get_main_camera_segments, extract_representative_frame


# ─────────────────────────────────────────────
# CONSTANTES TERRAIN FIFA
# ─────────────────────────────────────────────

PITCH_LENGTH = 105.0
PITCH_WIDTH  = 68.0

REFERENCE_POINTS = {
    "corner_bl":   (0.0,    0.0,    "Coin BG"),
    "corner_br":   (105.0,  0.0,    "Coin BD"),
    "corner_tl":   (0.0,    68.0,   "Coin HG"),
    "corner_tr":   (105.0,  68.0,   "Coin HD"),
    "mid_left":    (52.5,   0.0,    "Milieu bas"),
    "mid_right":   (52.5,   68.0,   "Milieu haut"),
    "center":      (52.5,   34.0,   "Centre"),
    "box_l_bl":    (0.0,    13.84,  "Surface G bas-g"),
    "box_l_br":    (16.5,   13.84,  "Surface G bas-d"),
    "box_l_tl":    (0.0,    54.16,  "Surface G haut-g"),
    "box_l_tr":    (16.5,   54.16,  "Surface G haut-d"),
    "box_r_bl":    (88.5,   13.84,  "Surface D bas-g"),
    "box_r_br":    (105.0,  13.84,  "Surface D bas-d"),
    "box_r_tl":    (88.5,   54.16,  "Surface D haut-g"),
    "box_r_tr":    (105.0,  54.16,  "Surface D haut-d"),
    "small_l_bl":  (0.0,    24.84,  "Petit rect G bas-g"),
    "small_l_br":  (5.5,    24.84,  "Petit rect G bas-d"),
    "small_l_tl":  (0.0,    43.16,  "Petit rect G haut-g"),
    "small_l_tr":  (5.5,    43.16,  "Petit rect G haut-d"),
    "small_r_bl":  (99.5,   24.84,  "Petit rect D bas-g"),
    "small_r_br":  (105.0,  24.84,  "Petit rect D bas-d"),
    "small_r_tl":  (99.5,   43.16,  "Petit rect D haut-g"),
    "small_r_tr":  (105.0,  43.16,  "Petit rect D haut-d"),
    "penalty_l":   (11.0,   34.0,   "Penalty G"),
    "penalty_r":   (94.0,   34.0,   "Penalty D"),
}


# ─────────────────────────────────────────────
# HOMOGRAPHIE
# ─────────────────────────────────────────────

def compute_homography(
    image_points: List[Tuple[float, float]],
    real_points:  List[Tuple[float, float]],
) -> Tuple[Optional[np.ndarray], float]:
    if len(image_points) < 4:
        return None, float('inf')

    src = np.float32(image_points)
    dst = np.float32(real_points)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=2.0)

    if H is None:
        return None, float('inf')

    src_pt  = np.float32([[p] for p in image_points])
    dst_hat = cv2.perspectiveTransform(src_pt, H)
    errors  = [np.linalg.norm(np.array(real_points[i]) - dst_hat[i][0])
               for i in range(len(real_points))]

    return H, float(np.mean(errors))


def pixel_to_real(cx: float, cy: float, H: np.ndarray) -> Tuple[float, float]:
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    t  = cv2.perspectiveTransform(pt, H)
    return (
        float(np.clip(t[0][0][0], 0, PITCH_LENGTH)),
        float(np.clip(t[0][0][1], 0, PITCH_WIDTH)),
    )


def compute_speed_kmh(rx1, ry1, rx2, ry2, dt_seconds) -> float:
    if dt_seconds <= 0:
        return 0.0
    dist = np.sqrt((rx2-rx1)**2 + (ry2-ry1)**2)
    return float(min(dist / dt_seconds * 3.6, 40.0))


# ─────────────────────────────────────────────
# STOCKAGE DES CALIBRATIONS
# ─────────────────────────────────────────────

class CalibrationStore:
    """Stocke les calibrations par segment, serialisable JSON."""

    def __init__(self):
        self.calibrations: Dict[int, dict] = {}

    def add_point(self, seg_idx, image_pt, real_pt, name, frame_nb):
        if seg_idx not in self.calibrations:
            self.calibrations[seg_idx] = {
                'image_points': [], 'real_points': [],
                'point_names': [], 'H': None,
                'reproj_error': None, 'frame_nb': frame_nb,
            }
        cal = self.calibrations[seg_idx]
        if name not in cal['point_names']:
            cal['image_points'].append(list(image_pt))
            cal['real_points'].append(list(real_pt))
            cal['point_names'].append(name)

    def remove_point(self, seg_idx, name):
        cal = self.calibrations.get(seg_idx)
        if not cal or name not in cal['point_names']:
            return
        i = cal['point_names'].index(name)
        for k in ['image_points', 'real_points', 'point_names']:
            cal[k].pop(i)

    def compute(self, seg_idx) -> Tuple[bool, float]:
        cal = self.calibrations.get(seg_idx)
        if not cal or len(cal['image_points']) < 4:
            return False, float('inf')
        H, err = compute_homography(cal['image_points'], cal['real_points'])
        if H is not None:
            cal['H'] = H.tolist()
            cal['reproj_error'] = round(err, 3)
            return True, err
        return False, float('inf')

    def get_H(self, seg_idx) -> Optional[np.ndarray]:
        cal = self.calibrations.get(seg_idx)
        return np.array(cal['H']) if cal and cal['H'] else None

    def is_calibrated(self, seg_idx) -> bool:
        return self.get_H(seg_idx) is not None

    def n_points(self, seg_idx) -> int:
        return len(self.calibrations.get(seg_idx, {}).get('image_points', []))

    def get_point_names(self, seg_idx) -> List[str]:
        return self.calibrations.get(seg_idx, {}).get('point_names', [])

    def get_nearest_H(self, seg_idx, all_indices) -> Optional[np.ndarray]:
        H = self.get_H(seg_idx)
        if H is not None:
            return H
        calibrated = [i for i in all_indices if self.is_calibrated(i)]
        if not calibrated:
            return None
        return self.get_H(min(calibrated, key=lambda i: abs(i - seg_idx)))

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.calibrations, f, indent=2)

    def load(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            self.calibrations = {int(k): v for k, v in data.items()}
            return True
        return False

    def summary(self) -> dict:
        return {
            'n_segments_calibrated': sum(1 for s in self.calibrations.values() if s['H']),
            'n_points_total': sum(len(s['image_points']) for s in self.calibrations.values()),
        }


# ─────────────────────────────────────────────
# RENDU SVG DU TERRAIN
# ─────────────────────────────────────────────

def render_pitch_svg(
    width: int = 580,
    height: int = 360,
    calibrated_points: Optional[List[str]] = None,
    highlight_point: Optional[str] = None,
) -> str:
    """
    Genere un SVG interactif du terrain FIFA.
    Points verts = calibres, orange = selectionne, blanc = disponibles.
    Chaque point est cliquable via onclick.
    """
    calibrated_points = calibrated_points or []
    mx, my = 40, 30
    pw, ph = width - 2*mx, height - 2*my

    def sx(rx): return mx + rx / PITCH_LENGTH * pw
    def sy(ry): return my + (1 - ry / PITCH_WIDTH) * ph

    lines = [
        ((0,0),(105,0)),((105,0),(105,68)),((105,68),(0,68)),((0,68),(0,0)),
        ((52.5,0),(52.5,68)),
        ((0,13.84),(16.5,13.84)),((16.5,13.84),(16.5,54.16)),((16.5,54.16),(0,54.16)),
        ((88.5,13.84),(105,13.84)),((88.5,13.84),(88.5,54.16)),((88.5,54.16),(105,54.16)),
        ((0,24.84),(5.5,24.84)),((5.5,24.84),(5.5,43.16)),((5.5,43.16),(0,43.16)),
        ((99.5,24.84),(105,24.84)),((99.5,24.84),(99.5,43.16)),((99.5,43.16),(105,43.16)),
    ]

    body = ""
    for (x1,y1),(x2,y2) in lines:
        body += f'<line x1="{sx(x1):.1f}" y1="{sy(y1):.1f}" x2="{sx(x2):.1f}" y2="{sy(y2):.1f}" stroke="rgba(255,255,255,0.7)" stroke-width="1.5"/>\n'

    # Cercle central
    r = pw * 9.15 / PITCH_LENGTH
    body += f'<circle cx="{sx(52.5):.1f}" cy="{sy(34):.1f}" r="{r:.1f}" fill="none" stroke="rgba(255,255,255,0.7)" stroke-width="1.5"/>\n'

    # Points cliquables
    for key, (rx, ry, label) in REFERENCE_POINTS.items():
        x, y = sx(rx), sy(ry)
        if key == highlight_point:
            fill, r_pt, stroke_w = "#EF9F27", 8, 2.5
        elif key in calibrated_points:
            fill, r_pt, stroke_w = "#1D9E75", 7, 2
        else:
            fill, r_pt, stroke_w = "rgba(255,255,255,0.4)", 5, 1.5

        body += (
            f'<g style="cursor:pointer" onclick="selectPitchPoint(\'{key}\',\'{label}\',{rx},{ry})">\n'
            f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="{r_pt}" fill="{fill}" stroke="white" stroke-width="{stroke_w}"/>\n'
            f'  <text x="{x+9:.1f}" y="{y+4:.1f}" font-size="8.5" fill="rgba(255,255,255,0.85)" font-family="sans-serif">{label}</text>\n'
            f'</g>\n'
        )

    return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"><rect width="{width}" height="{height}" fill="#1a6b3a" rx="8"/>{body}</svg>'


# ─────────────────────────────────────────────
# ENRICHISSEMENT TRACKING
# ─────────────────────────────────────────────

def enrich_tracking_with_calibration(
    tracking_df: pd.DataFrame,
    segments: List[Segment],
    store: CalibrationStore,
    fps: float = 24.0,
) -> pd.DataFrame:
    """Enrichit le tracking avec coordonnees reelles et vitesses km/h."""
    df = tracking_df.copy()
    for col in ['rx','ry','speed_kmh','speed_ms','segment_idx','segment_type','is_calibrated']:
        df[col] = np.nan if col in ['rx','ry','speed_kmh','speed_ms'] else (-1 if col == 'segment_idx' else ('unknown' if col == 'segment_type' else False))

    all_idx = list(range(len(segments)))

    for i, seg in enumerate(segments):
        mask = (df['frame'] >= seg.start_frame) & (df['frame'] <= seg.end_frame)
        df.loc[mask, 'segment_idx']  = i
        df.loc[mask, 'segment_type'] = seg.segment_type

    for track_id, group in df.groupby('track_id'):
        group  = group.sort_values('frame')
        prev   = {'rx': None, 'ry': None, 'frame': None}

        for idx, row in group.iterrows():
            si = int(row['segment_idx'])
            if si < 0:
                continue
            H = store.get_nearest_H(si, all_idx)
            if H is None:
                continue
            try:
                rx, ry = pixel_to_real(row['cx'], row['cy'], H)
                df.at[idx, 'rx'] = rx
                df.at[idx, 'ry'] = ry
                df.at[idx, 'is_calibrated'] = store.is_calibrated(si)

                if prev['rx'] is not None:
                    dt = (row['frame'] - prev['frame']) / fps
                    if dt > 0:
                        spd = compute_speed_kmh(prev['rx'], prev['ry'], rx, ry, dt)
                        df.at[idx, 'speed_kmh'] = spd
                        df.at[idx, 'speed_ms']  = round(spd / 3.6, 2)

                prev = {'rx': rx, 'ry': ry, 'frame': row['frame']}
            except Exception:
                prev = {'rx': None, 'ry': None, 'frame': None}

    return df


def draw_calibration_points(
    frame: np.ndarray,
    image_points: List[Tuple[float, float]],
    point_names: List[str],
) -> np.ndarray:
    """Dessine les points de calibration sur la frame video."""
    out = frame.copy()
    for i, (px, py) in enumerate(image_points):
        px, py = int(px), int(py)
        cv2.circle(out, (px, py), 8, (29, 158, 117), -1)
        cv2.circle(out, (px, py), 8, (255, 255, 255), 2)
        label = point_names[i] if i < len(point_names) else str(i)
        cv2.putText(out, label, (px+10, py-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return out


def compute_real_player_stats(enriched_df: pd.DataFrame, fps: float = 24.0) -> pd.DataFrame:
    """Stats reelles par joueur (frames calibrees uniquement)."""
    df = enriched_df[enriched_df['is_calibrated'] == True].dropna(subset=['rx','ry','speed_kmh'])
    results = []

    for track_id, group in df.groupby('track_id'):
        group  = group.sort_values('frame')
        if len(group) < 5:
            continue
        dx     = group['rx'].diff().fillna(0)
        dy     = group['ry'].diff().fillna(0)
        dists  = np.sqrt(dx**2 + dy**2)
        speeds = group['speed_kmh'].dropna()
        n      = len(group)

        results.append({
            'track_id':      track_id,
            'frames':        n,
            'duration_s':    round(n / fps, 1),
            'distance_m':    round(float(dists.sum()), 1),
            'speed_avg_kmh': round(float(speeds.mean()), 1) if len(speeds) else 0,
            'speed_max_kmh': round(float(speeds.max()),  1) if len(speeds) else 0,
            'pct_standing':  round((speeds <= 2).sum()                    / n * 100, 1),
            'pct_walking':   round(((speeds > 2)  & (speeds <= 7)).sum()  / n * 100, 1),
            'pct_jogging':   round(((speeds > 7)  & (speeds <= 14)).sum() / n * 100, 1),
            'pct_running':   round(((speeds > 14) & (speeds <= 21)).sum() / n * 100, 1),
            'pct_sprinting': round((speeds > 21).sum()                    / n * 100, 1),
        })

    return pd.DataFrame(results).sort_values('distance_m', ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# TEST RAPIDE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys, glob
    from scene_detector import detect_scenes, load_segments, save_segments

    videos_dir    = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'videos')
    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

    videos = glob.glob(os.path.join(videos_dir, '*.mp4'))
    if not videos:
        print("Aucune video."); sys.exit(0)

    video_path = videos[0]
    base       = os.path.splitext(os.path.basename(video_path))[0]
    seg_csv    = os.path.join(processed_dir, f"{base}_segments.csv")

    segments = (load_segments(seg_csv)
                if os.path.exists(seg_csv)
                else detect_scenes(video_path, frame_skip=2, max_frames=5000))

    main_segs = get_main_camera_segments(segments)
    print(f"Segments principaux : {len(main_segs)}")

    store    = CalibrationStore()
    cal_path = os.path.join(processed_dir, f"{base}_calibration.json")
    store.load(cal_path)

    if main_segs:
        seg     = main_segs[0]
        seg_idx = segments.index(seg)
        frame   = extract_representative_frame(video_path, seg)

        if frame is not None:
            h, w = frame.shape[:2]
            # Simulation 4 points
            for (px,py), name in [
                ((w*.15, h*.75), "box_l_bl"),
                ((w*.85, h*.75), "box_r_br"),
                ((w*.85, h*.25), "box_r_tr"),
                ((w*.15, h*.25), "box_l_tl"),
            ]:
                rx, ry, _ = REFERENCE_POINTS[name]
                store.add_point(seg_idx, (px,py), (rx,ry), name, seg.start_frame)

            ok, err = store.compute(seg_idx)
            print(f"Homographie : {'OK' if ok else 'ECHEC'} | erreur={err:.3f}m")

            if ok:
                store.save(cal_path)
                tracks = glob.glob(os.path.join(processed_dir, '*_tracking.csv'))
                if tracks:
                    df_track  = pd.read_csv(tracks[0])
                    enriched  = enrich_tracking_with_calibration(df_track, segments, store, fps=24.0)
                    stats     = compute_real_player_stats(enriched)
                    print("\n=== Stats joueurs reelles ===")
                    print(stats.head(10).to_string())
                    enriched.to_csv(os.path.join(processed_dir,'tracking_enriched.csv'), index=False)
                    print("CSV enrichi sauvegarde.")