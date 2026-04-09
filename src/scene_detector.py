"""
scene_detector.py
=================
Detection automatique des changements de camera dans une video broadcast.
Utilise la difference inter-frames pour identifier les coupures nettes
et les changements progressifs (pan, zoom).

Types de segments detectes :
    - main_camera   : vue d'ensemble du terrain (calibrable)
    - cut           : changement brutal de camera
    - slow_motion   : ralenti (frames similaires repetees)
    - close_up      : gros plan (peu de terrain visible)
"""

import cv2
import numpy as np
import pandas as pd
import os
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Seuil de difference inter-frames pour detecter une coupure
CUT_THRESHOLD       = 35.0   # diff moyenne en niveaux de gris (0-255)
SLOW_MO_THRESHOLD   = 3.0    # diff tres faible = ralenti ou image figee
MIN_SEGMENT_FRAMES  = 15     # segment minimum viable (< 15 frames = ignore)
GREEN_RATIO_MIN     = 0.25   # % minimum de vert pour une vue terrain


# ─────────────────────────────────────────────
# STRUCTURES DE DONNEES
# ─────────────────────────────────────────────

@dataclass
class Segment:
    """Represente un segment de video avec un type de camera."""
    start_frame:  int
    end_frame:    int
    segment_type: str          # 'main_camera', 'cut', 'slow_motion', 'close_up', 'unknown'
    green_ratio:  float = 0.0  # proportion de vert dans le segment
    avg_diff:     float = 0.0  # difference inter-frames moyenne
    calibration_H: Optional[np.ndarray] = None  # matrice homographie si calibre
    is_calibrated: bool = False

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def is_main_camera(self) -> bool:
        return self.segment_type == 'main_camera'

    @property
    def is_usable(self) -> bool:
        """Segment utilisable pour le calcul de vitesse."""
        return self.is_main_camera and self.is_calibrated

    def to_dict(self) -> dict:
        return {
            'start_frame':   self.start_frame,
            'end_frame':     self.end_frame,
            'duration_frames': self.duration_frames,
            'segment_type':  self.segment_type,
            'green_ratio':   round(self.green_ratio, 3),
            'avg_diff':      round(self.avg_diff, 2),
            'is_calibrated': self.is_calibrated,
        }


# ─────────────────────────────────────────────
# 1. ANALYSE D'UNE FRAME
# ─────────────────────────────────────────────

def compute_frame_diff(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Calcule la difference moyenne entre deux frames consecutives.
    Retourne une valeur entre 0 (identiques) et 255 (completement differentes).
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).astype(float)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(float)
    return float(np.mean(np.abs(gray1 - gray2)))


def compute_green_ratio(frame: np.ndarray) -> float:
    """
    Calcule la proportion de pixels verts (gazon) dans une frame.
    Utile pour distinguer vue terrain vs gros plan / tribune.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask  = cv2.inRange(hsv, lower_green, upper_green)
    return float(green_mask.sum() / 255) / (frame.shape[0] * frame.shape[1])


def classify_frame_type(green_ratio: float, avg_diff: float) -> str:
    """
    Classifie le type de frame selon le ratio vert et la difference.
    """
    if avg_diff > CUT_THRESHOLD:
        return 'cut'
    if avg_diff < SLOW_MO_THRESHOLD:
        return 'slow_motion'
    if green_ratio >= GREEN_RATIO_MIN:
        return 'main_camera'
    return 'close_up'


# ─────────────────────────────────────────────
# 2. DETECTION DES SCENES
# ─────────────────────────────────────────────

def detect_scenes(
    video_path: str,
    frame_skip: int = 2,
    max_frames: int = None,
    progress_callback=None,
) -> List[Segment]:
    """
    Analyse une video et retourne la liste des segments detectes.

    Parametres :
        video_path     : chemin vers le fichier MP4
        frame_skip     : analyser 1 frame sur N
        max_frames     : limite de frames (None = video entiere)
        progress_callback : fonction(pct) pour Streamlit

    Retourne une liste de Segment tries par start_frame.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video introuvable : {video_path}")

    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    limit        = max_frames or total_frames

    print(f"Analyse scenes : {os.path.basename(video_path)}")
    print(f"Frames : {total_frames} | FPS : {fps:.1f} | Limite : {limit}")

    # Lecture de toutes les frames avec differences
    frame_data = []  # [(frame_nb, green_ratio, diff)]
    prev_frame = None
    frame_nb   = 0

    while cap.isOpened() and frame_nb < limit:
        ret, frame = cap.read()
        if not ret:
            break

        frame_nb += 1

        if frame_nb % frame_skip != 0:
            continue

        # Redimensionner pour aller plus vite
        small = cv2.resize(frame, (320, 180))

        green_ratio = compute_green_ratio(small)
        diff = compute_frame_diff(prev_frame, small) if prev_frame is not None else 0.0

        frame_data.append({
            'frame':       frame_nb,
            'green_ratio': green_ratio,
            'diff':        diff,
            'frame_type':  classify_frame_type(green_ratio, diff),
        })

        prev_frame = small

        if progress_callback and frame_nb % 100 == 0:
            progress_callback(round(frame_nb / limit * 100, 1))

        if frame_nb % 500 == 0:
            print(f"  [{round(frame_nb/limit*100)}%] Frame {frame_nb}")

    cap.release()

    if not frame_data:
        return []

    df = pd.DataFrame(frame_data)

    # ── Segmentation ──────────────────────────────────────────────────────────
    segments = []
    current_type  = df.iloc[0]['frame_type']
    segment_start = int(df.iloc[0]['frame'])
    segment_diffs = [df.iloc[0]['diff']]
    segment_greens = [df.iloc[0]['green_ratio']]

    for i in range(1, len(df)):
        row      = df.iloc[i]
        new_type = row['frame_type']

        if new_type != current_type:
            # Fin du segment courant
            seg_end = int(df.iloc[i-1]['frame'])
            duration = seg_end - segment_start

            if duration >= MIN_SEGMENT_FRAMES:
                segments.append(Segment(
                    start_frame  = segment_start,
                    end_frame    = seg_end,
                    segment_type = current_type,
                    green_ratio  = float(np.mean(segment_greens)),
                    avg_diff     = float(np.mean(segment_diffs)),
                ))

            # Nouveau segment
            current_type   = new_type
            segment_start  = int(row['frame'])
            segment_diffs  = []
            segment_greens = []

        segment_diffs.append(float(row['diff']))
        segment_greens.append(float(row['green_ratio']))

    # Dernier segment
    if segment_diffs:
        segments.append(Segment(
            start_frame  = segment_start,
            end_frame    = int(df.iloc[-1]['frame']),
            segment_type = current_type,
            green_ratio  = float(np.mean(segment_greens)),
            avg_diff     = float(np.mean(segment_diffs)),
        ))

    # ── Fusion des segments courts ────────────────────────────────────────────
    segments = _merge_short_segments(segments)

    print(f"\nSegments detectes : {len(segments)}")
    for s in segments:
        print(f"  [{s.start_frame:5d}-{s.end_frame:5d}] "
              f"{s.segment_type:<15} "
              f"duree={s.duration_frames:4d}f "
              f"vert={s.green_ratio:.2f}")

    return segments


def _merge_short_segments(segments: List[Segment], min_frames: int = 30) -> List[Segment]:
    """
    Fusionne les segments trop courts avec leur voisin le plus proche.
    Evite les micro-segments de quelques frames.
    """
    if len(segments) <= 1:
        return segments

    merged = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        if seg.duration_frames < min_frames and i > 0:
            # Fusionner avec le segment precedent
            prev = merged[-1]
            merged[-1] = Segment(
                start_frame  = prev.start_frame,
                end_frame    = seg.end_frame,
                segment_type = prev.segment_type,
                green_ratio  = (prev.green_ratio + seg.green_ratio) / 2,
                avg_diff     = (prev.avg_diff + seg.avg_diff) / 2,
            )
        else:
            merged.append(seg)
        i += 1

    return merged


# ─────────────────────────────────────────────
# 3. EXPORT ET UTILITAIRES
# ─────────────────────────────────────────────

def segments_to_dataframe(segments: List[Segment]) -> pd.DataFrame:
    """Convertit la liste de segments en DataFrame."""
    return pd.DataFrame([s.to_dict() for s in segments])


def get_main_camera_segments(segments: List[Segment]) -> List[Segment]:
    """Retourne uniquement les segments camera principale."""
    return [s for s in segments if s.is_main_camera]


def get_segment_for_frame(segments: List[Segment], frame_nb: int) -> Optional[Segment]:
    """Retourne le segment auquel appartient une frame donnee."""
    for seg in segments:
        if seg.start_frame <= frame_nb <= seg.end_frame:
            return seg
    return None


def save_segments(segments: List[Segment], output_path: str):
    """Sauvegarde les segments en CSV."""
    df = segments_to_dataframe(segments)
    df.to_csv(output_path, index=False)
    print(f"Segments sauvegardes : {output_path}")


def load_segments(csv_path: str) -> List[Segment]:
    """Charge les segments depuis un CSV."""
    df = pd.read_csv(csv_path)
    segments = []
    for _, row in df.iterrows():
        segments.append(Segment(
            start_frame  = int(row['start_frame']),
            end_frame    = int(row['end_frame']),
            segment_type = row['segment_type'],
            green_ratio  = float(row['green_ratio']),
            avg_diff     = float(row['avg_diff']),
            is_calibrated= bool(row.get('is_calibrated', False)),
        ))
    return segments


def extract_representative_frame(
    video_path: str,
    segment: Segment,
    position: float = 0.3,
) -> Optional[np.ndarray]:
    """
    Extrait une frame representative d'un segment.
    position : 0.0 = debut, 0.5 = milieu, 1.0 = fin
    """
    target_frame = int(
        segment.start_frame +
        (segment.end_frame - segment.start_frame) * position
    )
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def get_scene_summary(segments: List[Segment]) -> dict:
    """Retourne un resume statistique des scenes."""
    total = sum(s.duration_frames for s in segments)
    by_type = {}
    for s in segments:
        if s.segment_type not in by_type:
            by_type[s.segment_type] = {'count': 0, 'frames': 0}
        by_type[s.segment_type]['count']  += 1
        by_type[s.segment_type]['frames'] += s.duration_frames

    summary = {
        'total_segments': len(segments),
        'total_frames':   total,
        'by_type': {}
    }
    for t, v in by_type.items():
        summary['by_type'][t] = {
            'count':   v['count'],
            'frames':  v['frames'],
            'pct':     round(v['frames'] / total * 100, 1) if total > 0 else 0,
        }
    return summary


# ─────────────────────────────────────────────
# TEST RAPIDE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import glob

    videos_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'videos')
    videos = glob.glob(os.path.join(videos_dir, '*.mp4'))

    if not videos:
        print("Aucune video disponible dans data/raw/videos/")
        sys.exit(0)

    video_path = videos[0]
    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # Analyse les 5000 premieres frames (~3 minutes)
    segments = detect_scenes(
        video_path=video_path,
        frame_skip=2,
        max_frames=5000,
    )

    # Resume
    summary = get_scene_summary(segments)
    print("\n=== Resume des scenes ===")
    for t, v in summary['by_type'].items():
        print(f"  {t:<15} : {v['count']:2d} segments | "
              f"{v['frames']:5d} frames | {v['pct']:.1f}%")

    # Segments camera principale
    main_segs = get_main_camera_segments(segments)
    print(f"\nSegments camera principale : {len(main_segs)}")
    print(f"Frames calibrables         : {sum(s.duration_frames for s in main_segs)}")

    # Sauvegarde
    base = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(processed_dir, f"{base}_segments.csv")
    save_segments(segments, csv_path)

    # Extraction frame representative du premier segment principal
    if main_segs:
        frame = extract_representative_frame(video_path, main_segs[0])
        if frame is not None:
            debug_path = os.path.join(processed_dir, 'debug_segment.jpg')
            cv2.imwrite(debug_path, frame)
            print(f"\nFrame representative sauvegardee : {debug_path}")