"""
video_tracker.py
================
Tracking persistant des joueurs sur 90 minutes.
Combine YOLO v8 (détection) + ByteTrack (tracking local)
+ Re-ID par couleur de maillot (identité persistante).

Fonctionnement :
    1. YOLO détecte les joueurs frame par frame
    2. ByteTrack assigne des IDs temporaires locaux
    3. Re-ID compare la couleur du maillot à la banque connue
    4. Si similarité > seuil → même joueur → ID persistant réattribué
    5. Sinon → nouveau joueur ajouté à la banque (max 26)
"""

import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import supervision as sv


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DEFAULT_MODEL       = "yolov8n.pt"
PERSON_CLASS_ID     = 0
MIN_CONFIDENCE      = 0.45

MIN_BOX_HEIGHT      = 40
MAX_BOX_HEIGHT      = 300
MIN_BOX_WIDTH       = 15
MAX_BOX_WIDTH       = 150

PITCH_MARGIN_TOP    = 0.10
PITCH_MARGIN_BOTTOM = 0.90
PITCH_MARGIN_LEFT   = 0.02
PITCH_MARGIN_RIGHT  = 0.98

REID_SIMILARITY_THRESHOLD = 0.75
MAX_PLAYERS               = 26
REID_UPDATE_FREQ          = 10
REID_REGION_TOP           = 0.15
REID_REGION_BOTTOM        = 0.65

DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'processed'
)


# ─────────────────────────────────────────────
# 1. SIGNATURE COULEUR MAILLOT
# ─────────────────────────────────────────────

def extract_jersey_signature(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """
    Extrait un histogramme HSV normalisé de la zone torse du joueur.
    Retourne un vecteur de 96 dimensions.
    """
    x1, y1, x2, y2 = bbox.astype(int)
    h = y2 - y1
    fh, fw = frame.shape[:2]

    crop_y1 = max(0, y1 + int(h * REID_REGION_TOP))
    crop_y2 = min(fh, y1 + int(h * REID_REGION_BOTTOM))
    crop_x1 = max(0, x1 + int((x2 - x1) * 0.1))
    crop_x2 = min(fw, x2 - int((x2 - x1) * 0.1))

    if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
        return np.zeros(96)

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return np.zeros(96)

    hsv   = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()

    sig  = np.concatenate([hist_h, hist_s, hist_v])
    norm = np.linalg.norm(sig)
    return sig / norm if norm > 0 else sig


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ─────────────────────────────────────────────
# 2. BANQUE DE JOUEURS
# ─────────────────────────────────────────────

class PlayerBank:
    """
    Maintient une banque de signatures visuelles.
    Assigne des IDs persistants en comparant les couleurs de maillot.
    """

    def __init__(self, max_players=MAX_PLAYERS, threshold=REID_SIMILARITY_THRESHOLD):
        self.max_players  = max_players
        self.threshold    = threshold
        self.signatures   = {}
        self.frame_counts = {}
        self.next_id      = 1

    def find_or_create(self, signature: np.ndarray) -> int:
        if np.all(signature == 0):
            return -1

        best_id, best_score = None, 0.0
        for pid, stored in self.signatures.items():
            score = cosine_similarity(signature, stored)
            if score > best_score:
                best_score, best_id = score, pid

        if best_score >= self.threshold and best_id is not None:
            # Mise à jour progressive de la signature
            alpha = 0.85
            self.signatures[best_id] = alpha * self.signatures[best_id] + (1 - alpha) * signature
            norm = np.linalg.norm(self.signatures[best_id])
            if norm > 0:
                self.signatures[best_id] /= norm
            self.frame_counts[best_id] = self.frame_counts.get(best_id, 0) + 1
            return best_id

        if len(self.signatures) < self.max_players:
            new_id = self.next_id
            self.next_id += 1
            self.signatures[new_id]   = signature.copy()
            self.frame_counts[new_id] = 1
            return new_id

        return best_id if best_id is not None else -1

    @property
    def player_count(self):
        return len(self.signatures)


# ─────────────────────────────────────────────
# 3. DÉTECTION
# ─────────────────────────────────────────────

def load_model(model_name=DEFAULT_MODEL) -> YOLO:
    print(f"Chargement modèle : {model_name}")
    model = YOLO(model_name)
    print("Modèle prêt.")
    return model


def detect_players_frame(frame, model, confidence=MIN_CONFIDENCE):
    h, w = frame.shape[:2]
    results    = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    if len(detections) == 0:
        return detections

    mask_class = (detections.class_id == PERSON_CLASS_ID) & (detections.confidence >= confidence)

    boxes = detections.xyxy
    box_w = boxes[:, 2] - boxes[:, 0]
    box_h = boxes[:, 3] - boxes[:, 1]
    mask_size = (
        (box_h >= MIN_BOX_HEIGHT) & (box_h <= MAX_BOX_HEIGHT) &
        (box_w >= MIN_BOX_WIDTH)  & (box_w <= MAX_BOX_WIDTH)
    )

    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    mask_zone = (
        (cy >= h * PITCH_MARGIN_TOP)    & (cy <= h * PITCH_MARGIN_BOTTOM) &
        (cx >= w * PITCH_MARGIN_LEFT)   & (cx <= w * PITCH_MARGIN_RIGHT)
    )

    return detections[mask_class & mask_size & mask_zone]


# ─────────────────────────────────────────────
# 4. TRACKING AVEC RE-ID
# ─────────────────────────────────────────────

def track_video(
    video_path: str,
    model: YOLO = None,
    output_dir: str = None,
    save_video: bool = True,
    frame_skip: int = 2,
    max_frames: int = None,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Tracking persistant avec Re-ID couleur maillot.
    Retourne un DataFrame avec IDs persistants sur toute la vidéo.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Vidéo introuvable : {video_path}")

    if model is None:
        model = load_model()

    out_dir = output_dir or DEFAULT_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    limit        = max_frames or total_frames

    print(f"Vidéo      : {base_name}")
    print(f"Résolution : {width}x{height} | FPS : {fps:.1f} | Frames : {total_frames}")
    print(f"Frame skip : {frame_skip} | Re-ID seuil : {REID_SIMILARITY_THRESHOLD}\n")

    tracker           = sv.ByteTrack()
    bank              = PlayerBank()
    local_to_pid      = {}

    box_annotator   = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=40)

    video_writer = None
    if save_video:
        out_path = os.path.join(out_dir, f"{base_name}_tracked.mp4")
        video_writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
        )

    records  = []
    frame_nb = 0

    while cap.isOpened() and frame_nb < limit:
        ret, frame = cap.read()
        if not ret:
            break

        frame_nb += 1

        if frame_nb % 100 == 0:
            pct = round(frame_nb / limit * 100, 1)
            print(f"  [{pct}%] Frame {frame_nb}/{limit} — joueurs identifiés : {bank.player_count}")

        if progress_callback and frame_nb % 10 == 0:
            progress_callback(pct=round(frame_nb / limit * 100, 1),
                              frame=frame_nb, total=limit)

        if frame_nb % frame_skip != 0:
            if video_writer:
                video_writer.write(frame)
            continue

        detections = detect_players_frame(frame, model)
        if len(detections) == 0:
            if video_writer:
                video_writer.write(frame)
            continue

        detections = tracker.update_with_detections(detections)
        if detections.tracker_id is None:
            if video_writer:
                video_writer.write(frame)
            continue

        persistent_ids = []
        for i in range(len(detections)):
            local_id = int(detections.tracker_id[i])
            bbox     = detections.xyxy[i]

            if local_id not in local_to_pid or frame_nb % REID_UPDATE_FREQ == 0:
                sig = extract_jersey_signature(frame, bbox)
                pid = bank.find_or_create(sig)
                local_to_pid[local_id] = pid
            else:
                pid = local_to_pid[local_id]

            persistent_ids.append(pid)

            if pid == -1:
                continue

            x1, y1, x2, y2 = bbox.astype(int)
            records.append({
                'frame':      frame_nb,
                'time_sec':   round(frame_nb / fps, 2),
                'track_id':   pid,
                'local_id':   local_id,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'cx':         int((x1 + x2) / 2),
                'cy':         int((y1 + y2) / 2),
                'width':      x2 - x1,
                'height':     y2 - y1,
                'confidence': round(float(detections.confidence[i]), 3),
            })

        if video_writer:
            labels    = [f"P{pid}" if pid != -1 else "?" for pid in persistent_ids]
            annotated = trace_annotator.annotate(frame.copy(), detections)
            annotated = box_annotator.annotate(annotated, detections)
            annotated = label_annotator.annotate(annotated, detections, labels=labels)
            video_writer.write(annotated)

    cap.release()
    if video_writer:
        video_writer.release()
        print(f"\nVidéo annotée sauvegardée.")

    df = pd.DataFrame(records)

    if not df.empty:
        csv_path = os.path.join(out_dir, f"{base_name}_tracking.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nRésultats Re-ID :")
        print(f"  IDs locaux ByteTrack  : {df['local_id'].nunique()}")
        print(f"  IDs persistants Re-ID : {df['track_id'].nunique()}")
        print(f"  Réduction             : {df['local_id'].nunique()} → {df['track_id'].nunique()}")
        print(f"  Frames analysées      : {df['frame'].nunique()}")
        print(f"  CSV sauvegardé        : {csv_path}")

    return df


# ─────────────────────────────────────────────
# 5. TRAJECTOIRES
# ─────────────────────────────────────────────

def compute_trajectories(tracking_df: pd.DataFrame, fps: float = 25.0) -> pd.DataFrame:
    if tracking_df.empty:
        return pd.DataFrame()

    results = []
    for track_id, group in tracking_df.groupby('track_id'):
        group = group.sort_values('frame')
        dx    = group['cx'].diff().fillna(0)
        dy    = group['cy'].diff().fillna(0)
        dists = np.sqrt(dx**2 + dy**2)

        results.append({
            'track_id':      track_id,
            'frames':        len(group),
            'duration_sec':  round(len(group) / fps, 2),
            'total_dist_px': round(dists.sum(), 1),
            'speed_avg':     round(dists.mean(), 2),
            'speed_max':     round(dists.max(), 2),
            'cx_mean':       round(group['cx'].mean(), 1),
            'cy_mean':       round(group['cy'].mean(), 1),
        })

    return pd.DataFrame(results).sort_values('total_dist_px', ascending=False)


def get_player_path(tracking_df: pd.DataFrame, track_id: int) -> pd.DataFrame:
    return tracking_df[tracking_df['track_id'] == track_id].sort_values('frame')


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from video_downloader import list_downloaded_videos
    videos = list_downloaded_videos()

    if not videos:
        print("Aucune vidéo disponible. Place un MP4 dans data/raw/videos/")
        sys.exit(0)

    video_path = videos[0]['path']
    print(f"Analyse de : {video_path}\n")

    model = load_model("yolov8n.pt")

    df = track_video(
        video_path=video_path,
        model=model,
        save_video=True,
        frame_skip=2,
        max_frames=3000,   # ~2 minutes — retire pour analyser tout le match
    )

    if not df.empty:
        print("\n=== Trajectoires joueurs persistants ===")
        trajs = compute_trajectories(df)
        print(trajs.head(26).to_string())