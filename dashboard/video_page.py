"""
video_page.py
=============
Onglet Analyse Vidéo du dashboard Football Analytics.
Intègre le tracking Re-ID et l'extraction d'événements
directement dans l'interface Streamlit.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import glob
import tempfile
import cv2

from video_tracker import load_model, track_video, compute_trajectories
from event_extractor import run_full_analysis


# ─────────────────────────────────────────────
# CACHE MODÈLE
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Chargement du modèle YOLO v8...")
def get_model():
    return load_model("yolov8n.pt")


# ─────────────────────────────────────────────
# VISUALISATIONS VIDÉO
# ─────────────────────────────────────────────

def plot_trajectories(tracking_df: pd.DataFrame, frame_width: int, frame_height: int) -> go.Figure:
    """Trajectoires de tous les joueurs sur le terrain vidéo."""
    fig = go.Figure()

    colors = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel

    for i, (track_id, group) in enumerate(tracking_df.groupby('track_id')):
        group = group.sort_values('frame')
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=group['cx'], y=group['cy'],
            mode='lines+markers',
            name=f"Joueur P{track_id}",
            line=dict(color=color, width=1.5),
            marker=dict(size=3, color=color),
            hovertemplate=(
                f"<b>Joueur P{track_id}</b><br>"
                "Frame : %{customdata[0]}<br>"
                "Temps : %{customdata[1]}s<br>"
                "Position : (%{x}, %{y})<extra></extra>"
            ),
            customdata=group[['frame', 'time_sec']].values,
        ))

    fig.update_layout(
        title="Trajectoires des joueurs",
        paper_bgcolor="#111111",
        plot_bgcolor="#1a6b3a",
        font=dict(color="#ffffff"),
        xaxis=dict(range=[0, frame_width], showgrid=False,
                   zeroline=False, title="X (pixels)"),
        yaxis=dict(range=[frame_height, 0], showgrid=False,
                   zeroline=False, title="Y (pixels)", scaleanchor="x"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(size=10)),
        margin=dict(l=20, r=20, t=50, b=20),
        height=450,
    )
    return fig


def plot_speed_timeline(tracking_df: pd.DataFrame, selected_ids: list) -> go.Figure:
    """Timeline de vitesse par joueur sélectionné."""
    fig = go.Figure()
    colors = px.colors.qualitative.Set2

    for i, tid in enumerate(selected_ids):
        group = tracking_df[tracking_df['track_id'] == tid].sort_values('frame')
        if group.empty:
            continue

        dx = group['cx'].diff().fillna(0)
        dy = group['cy'].diff().fillna(0)
        speed = np.sqrt(dx**2 + dy**2)

        fig.add_trace(go.Scatter(
            x=group['time_sec'],
            y=speed.round(2),
            mode='lines',
            name=f"P{tid}",
            line=dict(color=colors[i % len(colors)], width=2),
            fill='tozeroy',
            fillcolor=colors[i % len(colors)].replace('rgb', 'rgba').replace(')', ',0.1)'),
            hovertemplate="Temps : %{x}s<br>Vitesse : %{y} px/frame<extra></extra>",
        ))

    fig.update_layout(
        title="Vitesse au fil du temps",
        paper_bgcolor="#111111",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#ffffff"),
        xaxis=dict(title="Temps (secondes)", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="Vitesse (px/frame)", gridcolor="rgba(255,255,255,0.08)"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=300,
    )
    return fig


def plot_heatmap_video(tracking_df: pd.DataFrame, track_id: int,
                       frame_width: int, frame_height: int) -> go.Figure:
    """Heatmap de présence d'un joueur sur la vidéo."""
    group = tracking_df[tracking_df['track_id'] == track_id]

    fig = go.Figure(go.Histogram2dContour(
        x=group['cx'], y=group['cy'],
        colorscale=[
            [0.0, "rgba(0,0,0,0)"],
            [0.3, "rgba(255,200,0,0.4)"],
            [0.7, "rgba(255,100,0,0.7)"],
            [1.0, "rgba(220,30,30,0.95)"],
        ],
        showscale=False,
        ncontours=20,
    ))

    fig.update_layout(
        title=f"Heatmap — Joueur P{track_id}",
        paper_bgcolor="#111111",
        plot_bgcolor="#1a6b3a",
        font=dict(color="#ffffff"),
        xaxis=dict(range=[0, frame_width], showgrid=False, zeroline=False),
        yaxis=dict(range=[frame_height, 0], showgrid=False, zeroline=False, scaleanchor="x"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=380,
    )
    return fig


def plot_events_timeline(events_df: pd.DataFrame) -> go.Figure:
    """Timeline des événements détectés."""
    if events_df.empty:
        return go.Figure()

    color_map = {
        'Sprint':            '#EF9F27',
        'Direction change':  '#378ADD',
        'Duel':              '#E24B4A',
    }

    fig = go.Figure()

    for event_type, color in color_map.items():
        subset = events_df[events_df['type'] == event_type]
        if subset.empty:
            continue

        fig.add_trace(go.Scatter(
            x=subset['time_sec'],
            y=subset['track_id'],
            mode='markers',
            name=event_type,
            marker=dict(color=color, size=8, symbol='circle'),
            hovertemplate=(
                f"<b>{event_type}</b><br>"
                "Joueur : P%{y}<br>"
                "Temps : %{x}s<br>"
                "%{text}<extra></extra>"
            ),
            text=subset['data'],
        ))

    fig.update_layout(
        title="Timeline des événements détectés",
        paper_bgcolor="#111111",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#ffffff"),
        xaxis=dict(title="Temps (secondes)", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="Joueur (ID persistant)",
                   gridcolor="rgba(255,255,255,0.08)",
                   tickprefix="P"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)"),
        margin=dict(l=50, r=20, t=50, b=40),
        height=350,
    )
    return fig


# ─────────────────────────────────────────────
# PAGE PRINCIPALE
# ─────────────────────────────────────────────

def render_video_page():
    """Rendu complet de l'onglet Analyse Vidéo."""

    st.markdown("# Analyse Vidéo")
    st.markdown("*Tracking joueurs par Re-ID couleur maillot · YOLO v8 · ByteTrack*")
    st.markdown("---")

    # ── Section 1 : Source vidéo ──────────────────────────────────────────────
    st.subheader("Source vidéo")

    source = st.radio(
        "Choisir la source",
        ["Vidéo locale (data/raw/videos/)", "Uploader un fichier MP4"],
        horizontal=True
    )

    video_path = None

    if source == "Vidéo locale (data/raw/videos/)":
        videos_dir = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'raw', 'videos'
        )
        mp4_files = glob.glob(os.path.join(videos_dir, "*.mp4"))

        if not mp4_files:
            st.warning("Aucune vidéo trouvée dans `data/raw/videos/`. "
                       "Place un fichier MP4 dans ce dossier.")
        else:
            labels     = [os.path.basename(f) for f in mp4_files]
            selected   = st.selectbox("Sélectionne une vidéo", labels)
            video_path = mp4_files[labels.index(selected)]
            st.info(f"Vidéo sélectionnée : **{selected}**")

    else:
        uploaded = st.file_uploader("Upload un fichier MP4", type=["mp4"])
        if uploaded:
            tmp_dir  = os.path.join(
                os.path.dirname(__file__), '..', 'data', 'raw', 'videos'
            )
            os.makedirs(tmp_dir, exist_ok=True)
            video_path = os.path.join(tmp_dir, uploaded.name)
            with open(video_path, 'wb') as f:
                f.write(uploaded.read())
            st.success(f"Fichier uploadé : {uploaded.name}")

    if not video_path:
        st.stop()

    # ── Section 2 : Paramètres ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Paramètres d'analyse")

    col1, col2, col3 = st.columns(3)
    with col1:
        frame_skip = st.slider(
            "Frame skip", 1, 5, 2,
            help="Analyser 1 frame sur N (plus élevé = plus rapide, moins précis)"
        )
    with col2:
        max_minutes = st.slider(
            "Durée analysée (minutes)", 1, 10, 2,
            help="Nombre de minutes à analyser depuis le début"
        )
    with col3:
        save_video = st.checkbox("Sauvegarder vidéo annotée", value=True)

    # Convertir minutes en frames (estimation à 25fps)
    max_frames = max_minutes * 60 * 25

    st.markdown("---")

    # ── Section 3 : Lancement ────────────────────────────────────────────────
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run = st.button("Lancer l'analyse", type="primary", use_container_width=True)
    with col_info:
        st.caption(
            f"Estimation : ~{max_minutes * 2}-{max_minutes * 4} minutes de traitement "
            f"pour {max_minutes} minute(s) de vidéo."
        )

    # ── Vérifie si un CSV existe déjà pour cette vidéo ───────────────────────
    base_name   = os.path.splitext(os.path.basename(video_path))[0]
    out_dir     = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    csv_path    = os.path.join(out_dir, f"{base_name}_tracking.csv")
    events_path = os.path.join(out_dir, f"{base_name}_events.csv")

    tracking_df = pd.DataFrame()
    events_df   = pd.DataFrame()

    if os.path.exists(csv_path) and not run:
        st.info("Résultats existants trouvés — affichage des données précédentes. "
                "Clique sur **Lancer l'analyse** pour relancer.")
        tracking_df = pd.read_csv(csv_path)
        if os.path.exists(events_path):
            events_df = pd.read_csv(events_path)

    elif run:
        model = get_model()

        progress_bar  = st.progress(0)
        status_text   = st.empty()

        def on_progress(pct, frame, total):
            progress_bar.progress(min(int(pct), 99))
            status_text.text(f"Tracking... {pct}% — frame {frame}/{total}")

        with st.spinner("Analyse en cours..."):
            tracking_df = track_video(
                video_path=video_path,
                model=model,
                output_dir=out_dir,
                save_video=save_video,
                frame_skip=frame_skip,
                max_frames=max_frames,
                progress_callback=on_progress,
            )

        progress_bar.progress(100)
        status_text.text("Tracking terminé !")

        if not tracking_df.empty:
            with st.spinner("Extraction des événements..."):
                results   = run_full_analysis(
                    tracking_csv=csv_path,
                    fps=25.0,
                    output_dir=out_dir,
                )
                events_df = results['events']

            st.success(
                f"Analyse terminée — "
                f"{tracking_df['track_id'].nunique()} joueurs identifiés · "
                f"{len(events_df)} événements extraits"
            )

    if tracking_df.empty:
        st.stop()

    # ── Section 4 : KPIs ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Résumé")

    trajs = compute_trajectories(tracking_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Joueurs identifiés", tracking_df['track_id'].nunique())
    c2.metric("Frames analysées",   tracking_df['frame'].nunique())
    c3.metric("Durée analysée",
              f"{tracking_df['time_sec'].max():.0f}s")
    c4.metric("Événements extraits", len(events_df) if not events_df.empty else 0)

    st.markdown("---")

    # ── Section 5 : Visualisations ────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Trajectoires", "Joueur individuel", "Événements", "Données brutes", "Calibration terrain"
    ])

    frame_width  = int(tracking_df['x2'].max())
    frame_height = int(tracking_df['y2'].max())

    with tab1:
        all_ids = sorted(tracking_df['track_id'].unique())
        selected_traj = st.multiselect(
            "Joueurs à afficher",
            options=all_ids,
            default=all_ids[:min(6, len(all_ids))],
            format_func=lambda x: f"P{x}",
        )
        if selected_traj:
            filtered = tracking_df[tracking_df['track_id'].isin(selected_traj)]
            fig_traj = plot_trajectories(filtered, frame_width, frame_height)
            st.plotly_chart(fig_traj, use_container_width=True)

            st.subheader("Vitesse")
            fig_speed = plot_speed_timeline(filtered, selected_traj)
            st.plotly_chart(fig_speed, use_container_width=True)

    with tab2:
        all_ids_sorted = trajs.sort_values('total_dist_px', ascending=False)['track_id'].tolist()
        selected_player = st.selectbox(
            "Sélectionne un joueur",
            options=all_ids_sorted,
            format_func=lambda x: f"P{x}"
        )

        if selected_player:
            player_row = trajs[trajs['track_id'] == selected_player]
            if not player_row.empty:
                r = player_row.iloc[0]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Frames trackées", int(r['frames']))
                m2.metric("Durée",           f"{r['duration_sec']:.1f}s")
                m3.metric("Distance totale", f"{r['total_dist_px']:.0f} px")
                m4.metric("Vitesse max",     f"{r['speed_max']:.1f} px/f")

            col_h, col_t = st.columns(2)
            with col_h:
                fig_heat = plot_heatmap_video(
                    tracking_df, selected_player, frame_width, frame_height
                )
                st.plotly_chart(fig_heat, use_container_width=True)
            with col_t:
                player_data = tracking_df[tracking_df['track_id'] == selected_player]
                fig_spd = plot_speed_timeline(player_data, [selected_player])
                st.plotly_chart(fig_spd, use_container_width=True)

    with tab3:
        if events_df.empty:
            st.info("Lance d'abord l'analyse pour extraire les événements.")
        else:
            st.subheader("Timeline des événements")
            fig_ev = plot_events_timeline(events_df)
            st.plotly_chart(fig_ev, use_container_width=True)

            col_e1, col_e2, col_e3 = st.columns(3)
            sprints = events_df[events_df['type'] == 'Sprint']
            dirs    = events_df[events_df['type'] == 'Direction change']
            duels   = events_df[events_df['type'] == 'Duel']
            col_e1.metric("Sprints",              len(sprints))
            col_e2.metric("Changements direction", len(dirs))
            col_e3.metric("Duels",                len(duels))

            st.subheader("Détail des événements")
            st.dataframe(
                events_df[['type', 'track_id', 'time_sec', 'minute', 'data']].rename(columns={
                    'type': 'Type', 'track_id': 'Joueur',
                    'time_sec': 'Temps (s)', 'minute': 'Minute', 'data': 'Détails'
                }),
                use_container_width=True, hide_index=True
            )

    with tab4:
        st.subheader("Trajectoires par joueur")
        st.dataframe(
            trajs.rename(columns={
                'track_id': 'Joueur', 'frames': 'Frames',
                'duration_sec': 'Durée (s)', 'total_dist_px': 'Distance (px)',
                'speed_avg': 'Vitesse moy.', 'speed_max': 'Vitesse max',
                'cx_mean': 'X moyen', 'cy_mean': 'Y moyen',
            }),
            use_container_width=True, hide_index=True
        )

        st.subheader("Données de tracking brutes")
        st.dataframe(
            tracking_df.head(500),
            use_container_width=True, hide_index=True
        )
        st.caption(f"Affichage limité à 500 lignes sur {len(tracking_df)}")

    with tab5:
        render_calibration_tab(video_path, base_name, out_dir)


# ─────────────────────────────────────────────
# ONGLET CALIBRATION TERRAIN
# ─────────────────────────────────────────────

def render_calibration_tab(video_path: str, base_name: str, out_dir: str):
    """
    Interface de calibration terrain par segment.
    Deux colonnes : frame video cliquable + schema terrain FIFA.
    """
    import streamlit.components.v1 as components
    from scene_detector import (
        detect_scenes, load_segments, save_segments,
        get_main_camera_segments, extract_representative_frame,
        get_scene_summary
    )
    from pitch_calibrator import (
        CalibrationStore, REFERENCE_POINTS,
        render_pitch_svg, enrich_tracking_with_calibration,
        compute_real_player_stats, draw_calibration_points
    )

    seg_csv  = os.path.join(out_dir, f"{base_name}_segments.csv")
    cal_json = os.path.join(out_dir, f"{base_name}_calibration.json")

    # Init session state pour les coordonnees de clic
    if 'click_x' not in st.session_state:
        st.session_state['click_x'] = 320
    if 'click_y' not in st.session_state:
        st.session_state['click_y'] = 180

    # ── Chargement / détection des segments ──────────────────────────────────
    if os.path.exists(seg_csv):
        segments = load_segments(seg_csv)
        st.info(f"{len(segments)} segments chargés depuis le cache.")
    else:
        if st.button("Détecter les changements de caméra", type="primary"):
            with st.spinner("Analyse des scènes en cours..."):
                bar = st.progress(0)
                segments = detect_scenes(
                    video_path, frame_skip=2,
                    progress_callback=lambda p: bar.progress(min(int(p), 99))
                )
                save_segments(segments, seg_csv)
                bar.progress(100)
            st.success(f"{len(segments)} segments détectés !")
            st.rerun()
        else:
            st.info("Lance la détection pour commencer la calibration.")
            return
        return

    main_segs = get_main_camera_segments(segments)
    summary   = get_scene_summary(segments)

    # KPIs scènes
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total segments", len(segments))
    c2.metric("Caméra principale",
              f"{summary['by_type'].get('main_camera', {}).get('count', 0)} segs")
    c3.metric("Calibrables",
              f"{summary['by_type'].get('main_camera', {}).get('pct', 0):.1f}%")
    c4.metric("Gros plans / ralentis",
              summary['by_type'].get('slow_motion', {}).get('count', 0) +
              summary['by_type'].get('close_up', {}).get('count', 0))

    st.markdown("---")

    # ── Store de calibration ──────────────────────────────────────────────────
    store = CalibrationStore()
    store.load(cal_json)

    # ── Sélection du segment à calibrer ──────────────────────────────────────
    st.subheader("Calibration par segment")

    seg_options = [
        f"Segment {segments.index(s)} — frames {s.start_frame}-{s.end_frame} "
        f"({'✅ calibré' if store.is_calibrated(segments.index(s)) else '⏳ à calibrer'})"
        for s in main_segs
    ]

    if not seg_options:
        st.warning("Aucun segment caméra principale trouvé.")
        return

    selected_label = st.selectbox("Choisir un segment à calibrer", seg_options)
    selected_idx   = seg_options.index(selected_label)
    selected_seg   = main_segs[selected_idx]
    seg_global_idx = segments.index(selected_seg)

    # Slider pour choisir la frame dans le segment
    frame_position = st.slider(
        "Position dans le segment",
        min_value=selected_seg.start_frame,
        max_value=selected_seg.end_frame,
        value=selected_seg.start_frame + (selected_seg.end_frame - selected_seg.start_frame) // 3,
        step=2,
        help="Choisis une frame où le terrain est bien visible"
    )

    # Extraire la frame sélectionnée
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
    ret, current_frame = cap.read()
    cap.release()

    if not ret:
        st.error("Impossible de lire cette frame.")
        return

    # Dessiner les points déjà calibrés sur la frame
    cal_pts    = store.calibrations.get(seg_global_idx, {})
    img_points = cal_pts.get('image_points', [])
    pt_names   = cal_pts.get('point_names', [])
    annotated  = draw_calibration_points(current_frame, img_points, pt_names)

    # ── Interface deux colonnes ───────────────────────────────────────────────
    col_video, col_pitch = st.columns(2)

    with col_video:
        st.markdown("**Frame vidéo — clique sur un point du terrain**")

        try:
            from streamlit_image_coordinates import streamlit_image_coordinates
            from PIL import Image as PILImage

            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil_img   = PILImage.fromarray(frame_rgb)

            coords = streamlit_image_coordinates(
                pil_img,
                key=f"frame_click_{seg_global_idx}_{frame_position}",
            )

            if coords:
                st.session_state['click_x'] = coords['x']
                st.session_state['click_y'] = coords['y']
                st.success(f"Point sélectionné : X={coords['x']} Y={coords['y']}")

        except ImportError:
            st.warning(
                "Installe `streamlit-image-coordinates` pour le clic interactif :\n"
                "```\npip install streamlit-image-coordinates\n```"
            )
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, use_container_width=True)
            st.caption("Passe ta souris sur l'image → les coordonnées s'affichent en bas à droite.")

        st.caption("Clique sur le terrain pour sélectionner un point.")

        # Coordonnées — se mettent à jour automatiquement après clic
        st.markdown("**Coordonnées sélectionnées**")
        c_px, c_py = st.columns(2)
        px = c_px.number_input(
            "X (pixels)", min_value=0, max_value=current_frame.shape[1],
            value=int(st.session_state.get('click_x', current_frame.shape[1]//2)),
            key=f"px_{seg_global_idx}_{frame_position}"
        )
        py = c_py.number_input(
            "Y (pixels)", min_value=0, max_value=current_frame.shape[0],
            value=int(st.session_state.get('click_y', current_frame.shape[0]//2)),
            key=f"py_{seg_global_idx}_{frame_position}"
        )

    with col_pitch:
        st.markdown("**Schéma terrain FIFA — clique le point correspondant**")

        calibrated_names = store.get_point_names(seg_global_idx)
        pitch_svg = render_pitch_svg(
            width=560, height=360,
            calibrated_points=calibrated_names,
        )

        # Interface HTML interactive pour le clic sur le terrain
        pitch_html = f"""
        <div id="pitch-container" style="position:relative;display:inline-block">
            {pitch_svg}
            <div id="selected-info" style="
                margin-top:8px;font-size:12px;
                color:#1D9E75;font-family:sans-serif;
                min-height:20px;font-weight:500
            ">Clique sur un point du terrain</div>
        </div>
        <script>
        var selectedKey = null;
        var selectedLabel = null;
        var selectedRx = null;
        var selectedRy = null;

        function selectPitchPoint(key, label, rx, ry) {{
            selectedKey   = key;
            selectedLabel = label;
            selectedRx    = rx;
            selectedRy    = ry;
            document.getElementById('selected-info').innerHTML =
                '&#10003; ' + label + ' (' + rx + 'm, ' + ry + 'm)';
            document.getElementById('selected-key').value   = key;
            document.getElementById('selected-label').value = label;
            document.getElementById('selected-rx').value    = rx;
            document.getElementById('selected-ry').value    = ry;
        }}
        </script>
        <input type="hidden" id="selected-key"   value="">
        <input type="hidden" id="selected-label" value="">
        <input type="hidden" id="selected-rx"    value="">
        <input type="hidden" id="selected-ry"    value="">
        """
        components.html(pitch_html, height=420, scrolling=False)

        # Sélection du point terrain via selectbox (alternative au clic)
        pt_options = {v[2]: k for k, v in REFERENCE_POINTS.items()}
        selected_pt_label = st.selectbox(
            "Ou sélectionne le point terrain ici",
            list(pt_options.keys()),
        )
        selected_pt_key = pt_options[selected_pt_label]
        rx_real, ry_real, _ = REFERENCE_POINTS[selected_pt_key]

    # ── Ajout du point ────────────────────────────────────────────────────────
    st.markdown("---")
    col_add, col_clear, col_compute = st.columns(3)

    with col_add:
        if st.button("Ajouter ce point de correspondance", type="primary",
                     use_container_width=True):
            store.add_point(
                seg_global_idx,
                (float(px), float(py)),
                (rx_real, ry_real),
                selected_pt_key,
                frame_position,
            )
            store.save(cal_json)
            st.success(f"Point '{selected_pt_label}' ajouté ! "
                       f"Video: ({px},{py}) → Terrain: ({rx_real}m, {ry_real}m)")
            st.rerun()

    with col_clear:
        if st.button("Supprimer un point", use_container_width=True):
            names = store.get_point_names(seg_global_idx)
            if names:
                store.remove_point(seg_global_idx, names[-1])
                store.save(cal_json)
                st.info(f"Point '{names[-1]}' supprimé.")
                st.rerun()

    with col_compute:
        n_pts = store.n_points(seg_global_idx)
        if st.button(
            f"Calculer l'homographie ({n_pts} points)",
            disabled=n_pts < 4,
            type="primary" if n_pts >= 4 else "secondary",
            use_container_width=True,
        ):
            ok, err = store.compute(seg_global_idx)
            store.save(cal_json)
            if ok:
                st.success(f"Homographie calculée ! Erreur de reprojection : {err:.3f}m")
            else:
                st.error("Echec — ajoute au moins 4 points.")

    # ── Points actuels ────────────────────────────────────────────────────────
    cal = store.calibrations.get(seg_global_idx, {})
    if cal.get('point_names'):
        st.markdown("**Points enregistrés pour ce segment**")
        rows = []
        for i, name in enumerate(cal['point_names']):
            px_s, py_s = cal['image_points'][i]
            rx_s, ry_s = cal['real_points'][i]
            rows.append({
                'Point': REFERENCE_POINTS.get(name, ('','',''))[2],
                'Video X': int(px_s), 'Video Y': int(py_s),
                'Terrain X (m)': rx_s, 'Terrain Y (m)': ry_s,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if store.is_calibrated(seg_global_idx):
        err = cal.get('reproj_error', 'N/A')
        st.success(f"✅ Segment calibré — erreur reprojection : {err}m")

    st.markdown("---")

    # ── Application aux données de tracking ──────────────────────────────────
    st.subheader("Application au tracking")

    track_csv = os.path.join(out_dir, f"{base_name}_tracking.csv")
    n_calibrated = store.summary()['n_segments_calibrated']

    if not os.path.exists(track_csv):
        st.warning("Lance d'abord l'analyse vidéo pour générer le tracking.")
        return

    st.info(f"{n_calibrated} segments calibrés sur {len(main_segs)} segments principaux.")

    if st.button("Appliquer la calibration au tracking", type="primary",
                 disabled=n_calibrated == 0):
        with st.spinner("Calcul des coordonnées réelles..."):
            tracking_df = pd.read_csv(track_csv)
            enriched    = enrich_tracking_with_calibration(
                tracking_df, segments, store, fps=24.0
            )
            enriched.to_csv(
                os.path.join(out_dir, f"{base_name}_tracking_enriched.csv"),
                index=False
            )
            stats = compute_real_player_stats(enriched, fps=24.0)

        st.success("Coordonnées réelles calculées !")

        if not stats.empty:
            st.subheader("Stats joueurs réelles (km/h)")
            st.dataframe(
                stats.rename(columns={
                    'track_id': 'Joueur', 'frames': 'Frames',
                    'duration_s': 'Durée (s)', 'distance_m': 'Distance (m)',
                    'speed_avg_kmh': 'Vitesse moy. (km/h)',
                    'speed_max_kmh': 'Vitesse max (km/h)',
                    'pct_standing': 'Arrêt %', 'pct_walking': 'Marche %',
                    'pct_jogging': 'Trot %', 'pct_running': 'Course %',
                    'pct_sprinting': 'Sprint %',
                }),
                use_container_width=True, hide_index=True
            )