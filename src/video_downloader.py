"""
video_downloader.py
===================
Téléchargement de vidéos YouTube pour analyse football.
Utilise yt-dlp pour extraire la meilleure qualité disponible.
"""

import os
import re
import yt_dlp


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'raw', 'videos'
)


def _sanitize_filename(name: str) -> str:
    """Supprime les caractères interdits dans un nom de fichier."""
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()


# ─────────────────────────────────────────────
# 1. INFORMATIONS SUR LA VIDÉO
# ─────────────────────────────────────────────

def get_video_info(url: str) -> dict:
    """
    Récupère les métadonnées d'une vidéo YouTube sans la télécharger.
    Retourne : titre, durée, résolutions disponibles, vignette.
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    formats = info.get('formats', [])
    resolutions = sorted(set(
        f.get('height') for f in formats
        if f.get('height') and f.get('vcodec') != 'none'
    ), reverse=True)

    return {
        'title':       info.get('title', 'N/A'),
        'duration':    info.get('duration', 0),
        'duration_fmt': _fmt_duration(info.get('duration', 0)),
        'uploader':    info.get('uploader', 'N/A'),
        'view_count':  info.get('view_count', 0),
        'resolutions': resolutions,
        'thumbnail':   info.get('thumbnail', ''),
        'url':         url,
    }


def _fmt_duration(seconds: int) -> str:
    """Formate une durée en mm:ss ou hh:mm:ss."""
    if not seconds:
        return '00:00'
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ─────────────────────────────────────────────
# 2. TÉLÉCHARGEMENT
# ─────────────────────────────────────────────

def download_video(
    url: str,
    output_dir: str = None,
    max_height: int = 720,
    progress_callback=None
) -> str:
    """
    Télécharge une vidéo YouTube en MP4.

    Paramètres :
        url            : URL YouTube
        output_dir     : dossier de destination (défaut : data/raw/videos/)
        max_height     : résolution max en pixels (360, 480, 720, 1080)
        progress_callback : fonction appelée avec un dict de progression

    Retourne : chemin absolu du fichier téléchargé.
    """
    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # Récupère le titre pour nommer le fichier
    info = get_video_info(url)
    safe_title = _sanitize_filename(info['title'])[:80]
    output_path = os.path.join(out_dir, f"{safe_title}.mp4")

    # Si déjà téléchargé, retourne directement
    if os.path.exists(output_path):
        print(f"Déjà téléchargé : {output_path}")
        return output_path

    def _progress_hook(d):
        if progress_callback and d['status'] == 'downloading':
            downloaded = d.get('downloaded_bytes', 0)
            total      = d.get('total_bytes') or d.get('total_bytes_estimate', 1)
            speed      = d.get('speed', 0) or 0
            progress_callback({
                'percent':   round(downloaded / total * 100, 1) if total else 0,
                'speed_kb':  round(speed / 1024, 1),
                'eta':       d.get('eta', 0),
                'filename':  d.get('filename', ''),
            })
        elif progress_callback and d['status'] == 'finished':
            progress_callback({'percent': 100, 'speed_kb': 0, 'eta': 0, 'finished': True})

    ydl_opts = {
        # Meilleure qualité MP4 jusqu'à max_height
        'format': (
            f'bestvideo[height<={max_height}][ext=mp4]+'
            f'bestaudio[ext=m4a]/'
            f'bestvideo[height<={max_height}]+bestaudio/'
            f'best[height<={max_height}]'
        ),
        'outtmpl':          os.path.join(out_dir, f"{safe_title}.%(ext)s"),
        'merge_output_format': 'mp4',
        'quiet':            False,
        'no_warnings':      True,
        'progress_hooks':   [_progress_hook],
        'noprogress':       progress_callback is None,
    }

    print(f"Téléchargement : {info['title']}")
    print(f"Durée          : {info['duration_fmt']}")
    print(f"Résolution max : {max_height}p")
    print(f"Destination    : {output_path}")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Cherche le fichier généré (l'extension peut varier)
    for f in os.listdir(out_dir):
        if f.startswith(safe_title) and f.endswith('.mp4'):
            final_path = os.path.join(out_dir, f)
            print(f"\nTéléchargement terminé : {final_path}")
            return final_path

    raise FileNotFoundError(f"Fichier MP4 introuvable dans {out_dir}")


# ─────────────────────────────────────────────
# 3. DÉCOUPAGE — extrait un clip d'une durée donnée
# ─────────────────────────────────────────────

def download_clip(
    url: str,
    start_time: str,
    end_time: str,
    output_dir: str = None,
    max_height: int = 720,
) -> str:
    """
    Télécharge uniquement un extrait de la vidéo.

    Paramètres :
        start_time : format 'mm:ss' ou 'hh:mm:ss' (ex: '05:30')
        end_time   : format 'mm:ss' ou 'hh:mm:ss' (ex: '07:45')

    Utile pour les highlights ou séquences spécifiques.
    """
    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    info = get_video_info(url)
    safe_title = _sanitize_filename(info['title'])[:60]
    clip_name  = f"{safe_title}_{start_time.replace(':', '')}_{end_time.replace(':', '')}.mp4"
    output_path = os.path.join(out_dir, clip_name)

    if os.path.exists(output_path):
        print(f"Clip déjà présent : {output_path}")
        return output_path

    ydl_opts = {
        'format': f'bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best',
        'outtmpl': output_path,
        'merge_output_format': 'mp4',
        'quiet': False,
        'no_warnings': True,
        'download_ranges': yt_dlp.utils.download_range_func(
            None, [(
                _time_to_seconds(start_time),
                _time_to_seconds(end_time)
            )]
        ),
        'force_keyframes_at_cuts': True,
    }

    print(f"Téléchargement clip : {start_time} → {end_time}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return output_path


def _time_to_seconds(t: str) -> int:
    """Convertit 'mm:ss' ou 'hh:mm:ss' en secondes."""
    parts = list(map(int, t.split(':')))
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


# ─────────────────────────────────────────────
# 4. LISTE DES VIDÉOS TÉLÉCHARGÉES
# ─────────────────────────────────────────────

def list_downloaded_videos(output_dir: str = None) -> list:
    """
    Retourne la liste des vidéos MP4 disponibles localement.
    """
    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    if not os.path.exists(out_dir):
        return []

    videos = []
    for f in os.listdir(out_dir):
        if f.endswith('.mp4'):
            path = os.path.join(out_dir, f)
            size_mb = round(os.path.getsize(path) / (1024 * 1024), 1)
            videos.append({
                'filename': f,
                'path':     path,
                'size_mb':  size_mb,
            })
    return sorted(videos, key=lambda x: x['filename'])


# ─────────────────────────────────────────────
# TEST RAPIDE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("   Football Analytics — Téléchargeur vidéo")
    print("=" * 50)

    url = input("\nColle l'URL YouTube du match : ").strip()
    if not url:
        print("Aucune URL fournie. Abandon.")
        exit(0)

    print("\n--- Récupération des infos ---")
    try:
        info = get_video_info(url)
        print(f"Titre      : {info['title']}")
        print(f"Durée      : {info['duration_fmt']}")
        print(f"Résolutions: {info['resolutions']}")
        print(f"Uploader   : {info['uploader']}")
    except Exception as e:
        print(f"Erreur lors de la récupération : {e}")
        exit(1)

    print("\n--- Résolutions disponibles ---")
    for r in info['resolutions']:
        print(f"  {r}p")

    res_input = input("\nRésolution souhaitée (ex: 720) [défaut: 720] : ").strip()
    max_height = int(res_input) if res_input.isdigit() else 720

    confirm = input(f"\nTélécharger en {max_height}p ? (o/n) : ").strip().lower()
    if confirm != 'o':
        print("Téléchargement annulé.")
        exit(0)

    print("\n--- Téléchargement en cours ---")
    try:
        path = download_video(url, max_height=max_height)
        print(f"\nFichier prêt : {path}")
    except Exception as e:
        print(f"Erreur téléchargement : {e}")

    print("\n--- Vidéos disponibles localement ---")
    videos = list_downloaded_videos()
    if videos:
        for v in videos:
            print(f"  {v['filename']} ({v['size_mb']} MB)")
    else:
        print("  Aucune vidéo locale.")