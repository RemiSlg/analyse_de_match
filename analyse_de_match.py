import base64
import io
import mimetypes
import re
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.visualization.pitch import PitchVisualizer
from src.data.playersLoader import PlayerLoader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Analyse de match",
    page_icon="⚽",
    layout="wide",
)

# Version de démonstration : données allégées (quelques matchs, colonnes utiles uniquement)
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "demo_data" / "EPL_2011-12"
EVENTS_PATH = str(DATA_DIR / "Events")
LOGOS_PATH = str(DATA_DIR / "Logos")
PLAYERS_PATH = str(DATA_DIR / "Players")

TEAM_COLORS = ["#4FA3E0", "#E4605A"]  # équipe 1, équipe 2
FALLBACK_COLOR = "#8FA0B2"

# Image de fond (chemin relatif au dossier du projet). Mettre None pour garder la couleur unie.
BACKGROUND_IMAGE = str(BASE_DIR / "assets" / "fond.jpg")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600;700&display=swap');

:root {
    --surface: rgba(128, 128, 128, 0.10);
    --line: rgba(128, 128, 128, 0.30);
    --pitch: #3DAA7D;
}

/* Structure générale */
.block-container {
    max-width: 1280px;
    padding: 3.5rem 1.5rem 3rem 1.5rem !important;
}
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }

/* Couleurs de l'app (modifiables ici) : fond de page, widgets, accent */
html, body, .stApp, [data-testid="stApp"], [data-testid="stAppViewContainer"],
[data-testid="stMain"], section.main, .main, [data-testid="stMainBlockContainer"],
.block-container, [data-testid="stBottom"], [data-testid="stBottom"] > div {
    background-color: #D3E5D9 !important;
}
[data-baseweb="select"] > div { background-color: #F4F5F6 !important; border-color: rgba(128,128,128,.35) !important; }
button[data-baseweb="tab"][aria-selected="true"] p { color: #2F8553; }
[data-baseweb="tab-highlight"] { background-color: #2F8553 !important; }

/* En-tête */
h1.app-title {
    font-family: 'Barlow Semi Condensed', sans-serif;
    font-size: clamp(1.6rem, 4vw, 2.2rem);
    font-weight: 700;
    line-height: 1.1;
    padding: 0 !important;
    margin: 0 !important;
}
.app-sub { opacity: .65; margin: .15rem 0 0 0; }

.section-title {
    font-family: 'Barlow Semi Condensed', sans-serif;
    font-size: clamp(1.7rem, 4.5vw, 2.3rem);
    font-weight: 700;
    line-height: 1.1;
    margin: 1.8rem 0 .9rem 0;
    padding-bottom: .3rem;
    border-bottom: 3px solid #2F8553;
}

/* Scoreboard : reste sur une ligne, même sur mobile */
.scoreboard {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: center;
    gap: clamp(.5rem, 3vw, 2rem);
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: clamp(.9rem, 2.5vw, 1.6rem) clamp(.75rem, 3vw, 2rem);
    margin: .75rem 0 1.25rem 0;
}
.sb-team {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: .5rem;
    min-width: 0;
    text-align: center;
    padding-bottom: .6rem;
    border-bottom: 3px solid transparent;
}
.sb-logo, .sb-badge {
    width: clamp(52px, 11vw, 92px);
    height: clamp(52px, 11vw, 92px);
}
.sb-logo { object-fit: contain; display: block; }
.sb-badge {
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    border: 2px solid currentColor;
    font-family: 'Barlow Semi Condensed', sans-serif;
    font-weight: 700;
    font-size: clamp(1rem, 3vw, 1.6rem);
}
.sb-name {
    font-family: 'Barlow Semi Condensed', sans-serif;
    font-weight: 600;
    font-size: clamp(.95rem, 2.4vw, 1.35rem);
    line-height: 1.15;
    overflow-wrap: anywhere;
}
.sb-score {
    display: flex;
    align-items: baseline;
    gap: .6rem;
    font-family: 'Barlow Semi Condensed', sans-serif;
    font-weight: 700;
    font-size: clamp(2.4rem, 9vw, 4.5rem);
    line-height: 1;
    font-variant-numeric: tabular-nums;
}
.sb-sep { opacity: .5; font-weight: 400; }

/* Buteurs */
.scorers-head {
    font-family: 'Barlow Semi Condensed', sans-serif;
    font-weight: 600;
    font-size: 1.15rem;
    border-bottom: 2px solid currentColor;
    padding-bottom: .35rem;
    margin-bottom: .6rem;
}
.scorer {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    background: var(--surface);
    border: 1px solid var(--line);
    border-left: 3px solid var(--team);
    border-radius: 8px;
    padding: .55rem .8rem;
    margin-bottom: .4rem;
}
.scorer-name { font-weight: 600; }
.scorer-min {
    opacity: .65;
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
}
.empty { opacity: .65; font-style: italic; margin: 0; }

/* Chronologie */
.tl { position: relative; display: flex; flex-direction: column; gap: .4rem; padding: .25rem 0; }
.tl::before {
    content: "";
    position: absolute;
    top: 0; bottom: 0; left: 50%;
    width: 2px;
    transform: translateX(-50%);
    background: var(--line);
}
.tl-row { position: relative; display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: .6rem; }
.tl-cell { display: flex; min-width: 0; }
.tl-cell.left { justify-content: flex-end; }
.tl-min, .tl-break {
    position: relative;
    z-index: 1;
    background: #fff;
    color: #14241B;
    border: 1px solid var(--line);
    border-radius: 999px;
    font-family: 'Barlow Semi Condensed', sans-serif;
    font-weight: 600;
}
.tl-min { min-width: 3.2rem; text-align: center; padding: .1rem .6rem; font-variant-numeric: tabular-nums; }
.tl-break { align-self: center; display: flex; gap: .6rem; padding: .15rem .9rem; margin: .25rem 0; }
.tl-ev {
    display: flex;
    align-items: center;
    gap: .55rem;
    max-width: 100%;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: .4rem .65rem;
}
.tl-ev.left { text-align: right; border-right: 3px solid transparent; }
.tl-ev.right { border-left: 3px solid transparent; }
.tl-txt { min-width: 0; }
.tl-name { font-weight: 600; overflow-wrap: anywhere; line-height: 1.2; }
.tl-sub { opacity: .65; font-size: .85rem; font-variant-numeric: tabular-nums; }
.tl-in { color: #1E8449; }
.tl-out { opacity: .65; }
.tl-ico { font-size: 1.15rem; line-height: 1; flex: none; }
.tl-card { flex: none; width: .75rem; height: 1.05rem; border-radius: 2px; }
.tl-card.yellow { background: #F2C94C; }
.tl-card.red { background: #D64545; }

/* Faits marquants : version réduite (buts et cartons rouges) */
.cp { display: flex; flex-direction: column; gap: .8rem; padding: .25rem 0 .5rem 0; }
.cp-row { display: grid; grid-template-columns: 1fr auto 1fr; align-items: start; gap: .75rem; }
.cp-col { display: flex; flex-direction: column; gap: .3rem; min-width: 0; }
.cp-col.left { text-align: left; }
.cp-col.right { text-align: right; }
.cp-name { font-weight: 600; overflow-wrap: anywhere; }
.cp-min { opacity: .65; margin-left: .35rem; font-variant-numeric: tabular-nums; white-space: nowrap; }
.cp-ico { display: flex; justify-content: center; min-width: 2rem; padding-top: .1rem; }

/* Flèche « Détails » sous la chronologie */
.st-key-timeline_toggle_box { align-items: center; gap: 0 !important; }
.st-key-timeline_toggle_box [data-testid="stButton"] { display: flex; justify-content: center; }
.st-key-timeline_toggle_box button { border: none; background: transparent; padding: 0 1.5rem; min-height: 0; }
.st-key-timeline_toggle_box button span[data-testid="stIconMaterial"] { font-size: 2rem; }
.st-key-timeline_toggle_box button:hover { background: var(--surface); }
.arrow-label { text-align: center; opacity: .65; font-size: .9rem; margin-top: -.35rem; }

/* Statistiques d'équipe */
.st-table { display: flex; flex-direction: column; gap: .55rem; max-width: 760px; margin: 0 auto 1.2rem auto; }
.st-head, .st-row { display: grid; grid-template-columns: minmax(4rem, 1fr) minmax(0, 2fr) minmax(4rem, 1fr); align-items: center; gap: .5rem; }
.st-team { font-family: 'Barlow Semi Condensed', sans-serif; font-weight: 600; font-size: 1.05rem; }
.st-team.right { text-align: right; }
.st-cell { display: flex; }
.st-cell.left { justify-content: flex-start; }
.st-cell.right { justify-content: flex-end; }
.st-label { text-align: center; }
.st-val { min-width: 2.6rem; text-align: center; font-weight: 600; font-variant-numeric: tabular-nums; }
.st-pill { color: #fff; border-radius: 999px; padding: .15rem .7rem; }

/* Fiche joueur */
.pl-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: .8rem; margin: .8rem 0 1rem 0; }
.pl-card { background: var(--surface); border: 1px solid var(--line); border-top: 4px solid var(--line); border-radius: 12px; padding: .75rem .9rem .5rem .9rem; }
.pl-head { font-family: 'Barlow Semi Condensed', sans-serif; font-size: 1.45rem; font-weight: 700; line-height: 1.15; }
.pl-dot { display: inline-block; width: .7rem; height: .7rem; border-radius: 50%; margin-right: .5rem; vertical-align: .1rem; }
.pl-team { opacity: .65; margin: .1rem 0 .6rem 0; }
.pl-chips { display: flex; flex-wrap: wrap; gap: .45rem; margin-bottom: .3rem; }
.pl-chip { background: rgba(255, 255, 255, 0.45); border: 1px solid var(--line); border-radius: 10px; padding: .3rem .65rem; text-align: center; min-width: 3.9rem; }
.pl-chip strong { display: block; font-size: 1.1rem; font-variant-numeric: tabular-nums; }
.pl-chip span { opacity: .65; font-size: .78rem; }
.radar-title { font-family: 'Barlow Semi Condensed', sans-serif; font-weight: 600; font-size: 1.1rem; text-align: center; }

/* Compositions */
.bench-title { font-weight: 600; margin: .6rem 0 .25rem 0; }
.bench-row { display: flex; flex-wrap: wrap; gap: .2rem .6rem; padding: .3rem 0; border-top: 1px solid var(--line); }

/* Navigation entre sections : même aspect que les onglets de Streamlit (soulignement vert) */
.st-key-main_nav [data-testid="stButtonGroup"] {
    width: 100%;
    gap: 0 !important;
    border-bottom: 1px solid var(--line);
}
.st-key-main_nav button[data-testid^="stBaseButton-segmented_control"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    margin-bottom: -1px;
    min-height: 0 !important;
    padding: .5rem 1.1rem !important;
}
.st-key-main_nav button[data-testid^="stBaseButton-segmented_control"] p { font-size: 1rem; font-weight: 600; opacity: .7; }
.st-key-main_nav button[data-testid^="stBaseButton-segmented_control"]:hover p { opacity: 1; }
.st-key-main_nav button[data-testid="stBaseButton-segmented_controlActive"] { border-bottom-color: #2F8553 !important; }
.st-key-main_nav button[data-testid="stBaseButton-segmented_controlActive"] p { color: #2F8553; opacity: 1; }

/* Widgets */
button[data-baseweb="tab"] p { font-size: 1rem; font-weight: 600; }
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] { gap: .4rem; }
[data-testid="stPlotlyChart"] { border-radius: 12px; overflow: hidden; }

/* Mobile */
@media (max-width: 640px) {
    .block-container { padding: 3.25rem .75rem 2rem .75rem !important; }
    .scoreboard { border-radius: 12px; }
    .scorer { padding: .5rem .65rem; }
    .tl-row { gap: .4rem; }
    .tl-ev { padding: .35rem .5rem; gap: .4rem; }
}
</style>
"""


# ---------------------------------------------------------------------------
# Fonctions utilitaires d'affichage
# ---------------------------------------------------------------------------
@st.cache_data
def background_css(path):
    """CSS d'image de fond, ou chaîne vide si l'image est absente."""
    if not path or not Path(path).exists():
        return ""
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    data = base64.b64encode(Path(path).read_bytes()).decode()
    css = """
<style>
html, body, .stApp, [data-testid="stApp"], [data-testid="stAppViewContainer"],
[data-testid="stMain"], section.main, .main {
    background-image: url("__IMG__") !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
}
/* Panneau translucide derrière le contenu, pour rester lisible sur n'importe quelle image */
.block-container {
    background-color: rgba(255, 255, 255, 0.78) !important;
    backdrop-filter: blur(6px);
    border-radius: 16px;
    margin-top: 1rem;
}
@media (max-width: 640px) {
    html, body, .stApp, [data-testid="stApp"], [data-testid="stAppViewContainer"],
    [data-testid="stMain"], section.main, .main { background-attachment: scroll !important; }
    .block-container { border-radius: 0; margin-top: 0; }
}
</style>
"""
    return css.replace("__IMG__", f"data:{mime};base64,{data}")


def match_label(filename: str) -> str:
    return filename.replace("- Events.csv", "").replace(".csv", "").strip()


def logo_to_data_uri(logo):
    """Convertit un logo (chemin, bytes, image PIL ou tableau numpy) en data URI."""
    if logo is None:
        return None
    try:
        if isinstance(logo, (str, Path)):
            if str(logo).startswith(("http://", "https://", "data:")):
                return str(logo)
            mime = mimetypes.guess_type(str(logo))[0] or "image/png"
            data = Path(logo).read_bytes()
        elif isinstance(logo, (bytes, bytearray)):
            mime, data = "image/png", bytes(logo)
        else:
            from PIL import Image

            img = logo if isinstance(logo, Image.Image) else Image.fromarray(np.asarray(logo))
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            mime, data = "image/png", buffer.getvalue()
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception:
        return None


def render_scoreboard(teams, logos, scores, colors):
    def team_block(team, logo):
        uri = logo_to_data_uri(logo)
        if uri:
            visual = f'<img class="sb-logo" src="{uri}" alt="Logo {escape(team)}">'
        else:
            visual = f'<div class="sb-badge" style="color:{colors[team]}">{escape(team[:3].upper())}</div>'
        return (
            f'<div class="sb-team" style="border-bottom-color:{colors[team]}">'
            f"{visual}"
            f'<span class="sb-name">{escape(team)}</span>'
            f"</div>"
        )

    html = (
        '<div class="scoreboard">'
        + team_block(teams[0], logos[0])
        + f'<div class="sb-score"><span>{scores[0]}</span>'
        + '<span class="sb-sep">–</span>'
        + f"<span>{scores[1]}</span></div>"
        + team_block(teams[1], logos[1])
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def style_figure(fig, transparent_plot=True):
    """Rend une figure Plotly fluide et alignée sur le thème sombre."""
    fig.update_layout(
        title=dict(text=""),
        autosize=True,
        width=None,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    if transparent_plot:
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    return fig


SUB_PATTERN = r"\bsub|substitut|replace"

# (type, motif à inclure, motif à exclure) : le premier qui correspond gagne
TIMELINE_KINDS = [
    ("goal", r"\bgoal\b", r"kick|keeper|assist"),
    ("red", r"\bred\b|second.?yellow", ""),
    ("yellow", r"yellow", ""),
    ("sub", SUB_PATTERN, ""),
]


def classify_event(name):
    for kind, include, exclude in TIMELINE_KINDS:
        if re.search(include, str(name), re.I) and not (exclude and re.search(exclude, str(name), re.I)):
            return kind
    return None


def player_team_map(df):
    """Équipe de chaque joueur d'après ses événements courants.

    L'équipe indiquée sur un remplacement est parfois fausse dans les données,
    on l'ignore donc pour déterminer à quelle équipe appartient un joueur.
    """
    names = df["Event Name"].astype(str)
    core = df[~names.str.contains(SUB_PATTERN, case=False, regex=True)]
    core = core.dropna(subset=["Player1 Name", "Player1 Team"])
    return core.groupby("Player1 Name")["Player1 Team"].agg(lambda s: s.mode().iat[0]).to_dict()


def build_timeline(df, teams):
    """Buts, cartons et remplacements dans l'ordre du match.

    Time est en secondes et repart de 0 en seconde période (Half vaut 0 puis 1).
    Pour un remplacement, Player1 est le joueur sortant et Player2 l'entrant.
    """
    ptm = player_team_map(df)
    data = df.assign(_kind=df["Event Name"].map(classify_event)).dropna(subset=["_kind"])
    data = data.sort_values(["Half", "Time"])

    score = {teams[0]: 0, teams[1]: 0}
    events = []
    for _, row in data.iterrows():
        player = row["Player1 Name"]
        team = ptm.get(player, row["Player1 Team"])
        if team not in score:
            continue
        event = {
            "minute": int(row["Time"] // 60 + 45 * row["Half"]),
            "half": int(row["Half"]),
            "team": team,
            "kind": row["_kind"],
            "player": player,
            "other": row["Player2 Name"] if isinstance(row["Player2 Name"], str) else None,
        }
        if event["kind"] == "goal":
            score[team] += 1
        event["score"] = (score[teams[0]], score[teams[1]])
        events.append(event)
    return events


def render_timeline(events, teams, colors, kinds=None):
    """Affiche la chronologie. `kinds` limite les types d'événements montrés (None = tous).

    Les scores de mi-temps et de fin de match restent ceux du match complet.
    """
    if not events:
        st.markdown('<p class="empty">Aucun événement marquant pour ce match.</p>', unsafe_allow_html=True)
        return
    if kinds is not None and not any(ev["kind"] in kinds for ev in events):
        st.markdown('<p class="empty">Aucun but ni carton pour ce match.</p>', unsafe_allow_html=True)
        return

    def event_html(ev, side):
        kind = ev["kind"]
        player = escape(str(ev["player"]))
        if kind == "goal":
            icon = '<span class="tl-ico">⚽</span>'
            main, detail = player, f'{ev["score"][0]} - {ev["score"][1]}'
        elif kind in ("yellow", "red"):
            icon = f'<span class="tl-card {kind}"></span>'
            main, detail = player, "Carton jaune" if kind == "yellow" else "Carton rouge"
        else:  # remplacement : Player1 sort, Player2 entre
            icon = '<span class="tl-ico">⇄</span>'
            if ev["other"]:
                main = f'<span class="tl-in">↑ {escape(ev["other"])}</span>'
                detail = f'<span class="tl-out">↓ {player}</span>'
            else:
                main, detail = f'<span class="tl-out">↓ {player}</span>', "Remplacement"
        text = f'<div class="tl-txt"><div class="tl-name">{main}</div><div class="tl-sub">{detail}</div></div>'
        body = text + icon if side == "left" else icon + text
        edge = "border-right-color" if side == "left" else "border-left-color"
        return f'<div class="tl-ev {side}" style="{edge}:{colors[ev["team"]]}">{body}</div>'

    def break_html(label, score):
        return f'<div class="tl-break"><span>{label}</span><strong>{score[0]} - {score[1]}</strong></div>'

    parts = ['<div class="tl">']
    half_time_score = (0, 0)
    half_time_shown = False
    for ev in events:
        if ev["half"] == 0:
            half_time_score = ev["score"]
        if kinds is not None and ev["kind"] not in kinds:
            continue
        if ev["half"] == 1 and not half_time_shown:
            parts.append(break_html("Mi-temps", half_time_score))
            half_time_shown = True
        side = "left" if ev["team"] == teams[0] else "right"
        left = event_html(ev, "left") if side == "left" else ""
        right = event_html(ev, "right") if side == "right" else ""
        parts.append(
            '<div class="tl-row">'
            f'<div class="tl-cell left">{left}</div>'
            f'<div class="tl-min">{ev["minute"]}\'</div>'
            f'<div class="tl-cell right">{right}</div>'
            "</div>"
        )
    if not half_time_shown:
        parts.append(break_html("Mi-temps", half_time_score))
    parts.append(break_html("Fin du match", events[-1]["score"]))
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def short_name(name):
    """Nom de famille (écrit en majuscules dans les données) : « Morten GAMST PEDERSEN » -> « GAMST PEDERSEN »."""
    parts = [t for t in str(name).split() if t.isupper()]
    return " ".join(parts) if parts else str(name).split()[-1]


def lineup_for_team(df_players, team, events):
    """Titulaires (placés par leur position moyenne) et remplacements d'une équipe.

    Un titulaire est un joueur qui n'entre jamais en cours de match (il n'est le
    Player2 d'aucun remplacement). Les joueurs sortis sont repérés par « ↓ minute ».
    """
    subs = [ev for ev in events if ev["kind"] == "sub" and ev["team"] == team]
    entrants = {ev["other"] for ev in subs if ev["other"]}
    left_at = {ev["player"]: ev["minute"] for ev in subs}

    starters = []
    for _, row in df_players[df_players["Team"] == team].iterrows():
        name = row["Player Name"]
        if name in entrants or pd.isna(row["Average X"]) or pd.isna(row["Average Y"]):
            continue
        label = short_name(name)
        if name in left_at:
            label += f" ↓{left_at[name]}'"
        starters.append((label, float(row["Average X"]), float(row["Average Y"])))
    return starters, subs


def render_bench(subs):
    if not subs:
        return
    rows = []
    for ev in subs:
        incoming = escape(str(ev["other"])) if ev["other"] else "?"
        rows.append(
            '<div class="bench-row">'
            f'<span class="tl-in">↑ {incoming}</span>'
            f'<span class="cp-min">{ev["minute"]}\'</span>'
            f'<span class="tl-out">↓ {escape(str(ev["player"]))}</span>'
            "</div>"
        )
    st.markdown('<div class="bench-title">Remplacements</div>' + "".join(rows), unsafe_allow_html=True)


SHOT_PATTERN = r"\bshot\b"
PASS_PATTERN = r"pass|cross"

# (libellé, clé, format, « plus c'est élevé, mieux c'est »)
STAT_ROWS = [
    ("Tirs", "shots", "n", True),
    ("Tirs cadrés", "on_target", "n", True),
    ("Possession", "possession", "pct", True),
    ("Passes", "passes", "n", True),
    ("Précision des passes", "pass_acc", "pct", True),
    ("Fautes", "fouls", "n", False),
    ("Cartons jaunes", "yellow", "n", False),
    ("Cartons rouges", "red", "n", False),
    ("Hors-jeu", "offside", "n", False),
    ("Corners", "corners", "n", True),
]


def event_masks(d, team):
    """Masques d'événements communs aux statistiques d'équipe et de joueur."""
    names = d["Event Name"].astype(str)
    is_shot = names.str.contains(SHOT_PATTERN, case=False, regex=True)
    is_goal = names.map(classify_event) == "goal"

    # Relie chaque but au tir qui l'a précédé (même joueur, même mi-temps, quelques secondes avant,
    # quelques événements plus tôt) : le but ne doit pas être compté comme un second tir
    time = pd.to_numeric(d["Time"], errors="coerce")
    scoring_shot = pd.Series(False, index=d.index)
    linked_goal = pd.Series(False, index=d.index)
    shot_to_goal = {}
    for g in d.index[is_goal]:
        lo = max(0, g - 10)
        window = d.iloc[lo:g]
        found = window[
            is_shot.iloc[lo:g]
            & (window["Player1 Name"] == d.at[g, "Player1 Name"])
            & (window["Half"] == d.at[g, "Half"])
            & (d.at[g, "Time"] - time.iloc[lo:g]).between(0, 15)
        ]
        if not found.empty:
            shot_index = found.index[-1]
            scoring_shot[shot_index] = True
            linked_goal[g] = True
            shot_to_goal[shot_index] = g

    is_pass = names.str.contains(PASS_PATTERN, case=False, regex=True)
    if "Possession Number" in d.columns:
        continues = d["Possession Number"].shift(-1) == d["Possession Number"]
    else:
        continues = team.shift(-1) == team
    interrupted = names.shift(-1).fillna("").str.contains(r"out of play|offside|end of half", case=False, regex=True)

    return {
        "names": names,
        "shot": is_shot,
        "goal": is_goal,
        "extra_goal": is_goal & ~linked_goal,
        "scoring_shot": scoring_shot,
        "shot_to_goal": shot_to_goal,
        "save": names.str.contains(r"goalkeeper save", case=False, regex=True),
        "pass": is_pass,
        "pass_ok": is_pass & continues & ~interrupted,
    }


def compute_match_stats(df, teams, events):
    """Statistiques d'équipe calculées à partir des événements du match.

    Certaines valeurs sont des estimations (les données n'ont pas de colonne de résultat) :
    - tirs cadrés = buts + arrêts du gardien adverse ;
    - précision des passes = part des passes suivies d'un événement de la même possession ;
    - possession = temps pendant lequel chaque équipe a le ballon (colonne Possession).
    """
    t0, t1 = teams[:2]
    other = {t0: t1, t1: t0}
    ptm = player_team_map(df)
    d = df.reset_index(drop=True)
    team = d["Player1 Name"].map(ptm).fillna(d["Player1 Team"])
    m = event_masks(d, team)
    names = m["names"]

    def count(mask, who):
        return int((mask & (team == who)).sum())

    # Possession : temps passé avec le ballon (écarts plafonnés à 30 s pour ignorer les arrêts de jeu)
    hold = None
    if {"Possession", "Team A", "Team B", "Time", "Half"}.issubset(d.columns):
        owner = d["Possession"].map({"A": d["Team A"].iloc[0], "B": d["Team B"].iloc[0]})
        gap = (d["Time"].shift(-1) - d["Time"]).where(d["Half"].shift(-1) == d["Half"])
        gap = gap.clip(lower=0, upper=30).fillna(0)
        hold = {t: float(gap[owner == t].sum()) for t in (t0, t1)}

    stats = {}
    for t in (t0, t1):
        shots = count(m["shot"], t) + count(m["extra_goal"], t)
        passes = count(m["pass"], t)
        stats[t] = {
            "shots": shots,
            "on_target": min(count(m["goal"], t) + count(m["save"], other[t]), shots),
            "passes": passes,
            "pass_acc": round(100 * count(m["pass_ok"], t) / passes) if passes else None,
            "fouls": count(names.str.lower() == "foul", t),
            "offside": count(names.str.lower() == "offside", t),
            "corners": count(names.str.contains("corner", case=False, regex=False), t),
            "yellow": sum(1 for ev in events if ev["kind"] == "yellow" and ev["team"] == t),
            "red": sum(1 for ev in events if ev["kind"] == "red" and ev["team"] == t),
            "possession": None,
        }
    if hold and sum(hold.values()) > 0:
        share = round(100 * hold[t0] / sum(hold.values()))
        stats[t0]["possession"], stats[t1]["possession"] = share, 100 - share
    return stats


def render_stats_table(stats, teams, colors):
    t0, t1 = teams[:2]

    def cell(value, fmt, highlighted, team):
        text = "–" if value is None else (f"{value} %" if fmt == "pct" else str(value))
        if highlighted:
            return f'<span class="st-val st-pill" style="background:{colors[team]}">{text}</span>'
        return f'<span class="st-val">{text}</span>'

    rows = [
        '<div class="st-head">'
        f'<span class="st-team" style="color:{colors[t0]}">{escape(str(t0))}</span><span></span>'
        f'<span class="st-team right" style="color:{colors[t1]}">{escape(str(t1))}</span>'
        "</div>"
    ]
    for label, key, fmt, higher_is_better in STAT_ROWS:
        a, b = stats[t0][key], stats[t1][key]
        if a is None and b is None:
            continue
        win0 = win1 = False
        if a is not None and b is not None and a != b:
            win0 = (a > b) == higher_is_better
            win1 = not win0
        rows.append(
            '<div class="st-row">'
            f'<div class="st-cell left">{cell(a, fmt, win0, t0)}</div>'
            f'<div class="st-label">{label}</div>'
            f'<div class="st-cell right">{cell(b, fmt, win1, t1)}</div>'
            "</div>"
        )
    st.markdown('<div class="st-table">' + "".join(rows) + "</div>", unsafe_allow_html=True)


# Colonnes reprises du fichier joueurs quand elles existent
PLAYER_FILE_COLUMNS = ["Minutes Played", "Goals", "xGoals Shot", "Player Points", "HP Regains"]

# Un seul radar : (libellé, colonne, ramené à 90 minutes ?)
RADAR_AXES = [
    ("Tirs", "Tirs", True), ("xG", "xGoals Shot", True), ("Buts", "Goals", True),
    ("Passes", "Passes", True), ("Précision des passes", "Précision des passes", False), ("Dribbles", "Dribbles", True),
    ("Centres", "Centres", True), ("Tacles", "Tacles", True), ("Blocs", "Blocs", True),
    ("Pressing", "HP Regains", True),
]


def compute_player_table(df, df_players, events):
    """Une ligne par joueur : actions comptées dans les événements + colonnes du fichier joueurs."""
    ptm = player_team_map(df)
    d = df.reset_index(drop=True)
    team = d["Player1 Name"].map(ptm).fillna(d["Player1 Team"])
    m = event_masks(d, team)
    lower = m["names"].str.lower()

    flags = pd.DataFrame({
        "Joueur": d["Player1 Name"],
        "Touches": 1,
        "Tirs": (m["shot"] | m["extra_goal"]).astype(int),
        "Passes": m["pass"].astype(int),
        "Passes réussies": m["pass_ok"].astype(int),
        "Centres": (lower == "cross").astype(int),
        "Dribbles": (lower == "dribble").astype(int),
        "Tacles": (lower == "tackle").astype(int),
        "Blocs": (lower == "block").astype(int),
        "Dégagements": (lower == "clearance").astype(int),
        "Têtes": (lower == "header").astype(int),
        "Fautes": (lower == "foul").astype(int),
        "Arrêts": lower.str.contains("goalkeeper save", regex=False).astype(int),
        "Prises": lower.isin(["goalkeeper catch", "goalkeeper pick up"]).astype(int),
        "Relances": lower.isin(["goalkeeper kick", "goalkeeper throw"]).astype(int),
    })
    table = flags.dropna(subset=["Joueur"]).groupby("Joueur").sum()
    table["Actions déf."] = table["Tacles"] + table["Blocs"] + table["Dégagements"]
    table["Actions gardien"] = table["Arrêts"] + table["Prises"] + table["Relances"]
    table["Équipe"] = table.index.map(ptm)
    table["Jaunes"] = [sum(1 for ev in events if ev["kind"] == "yellow" and ev["player"] == p) for p in table.index]
    table["Rouges"] = [sum(1 for ev in events if ev["kind"] == "red" and ev["player"] == p) for p in table.index]
    # Précision estimée, seulement si le joueur a fait assez de passes pour que ce soit parlant
    table["Précision des passes"] = (100 * table["Passes réussies"] / table["Passes"]).where(table["Passes"] >= 10)

    file_stats = df_players.set_index("Player Name")
    for col in PLAYER_FILE_COLUMNS:
        if col in file_stats.columns:
            table[col] = file_stats[col].reindex(table.index)
    return table


def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def percentile_of(pool, value):
    """Part (0 à 100) des joueurs de référence que la valeur dépasse ; 0 si elle ne dépasse personne."""
    pool = pool.dropna()
    if pool.empty or pd.isna(value):
        return 0.0
    return float(min(100.0, 100 * (pool < value).sum() / max(len(pool) - 1, 1)))


def radar_figure(table, selected, axes):
    """Radar de un ou deux joueurs sur un thème. selected : liste de (joueur, couleur).

    Chaque axe compare le joueur aux joueurs du match ayant joué au moins 20 minutes ;
    les volumes sont ramenés à 90 minutes.
    """
    minutes = table["Minutes Played"].where(table["Minutes Played"] > 0)
    in_pool = minutes >= 20
    axes = [axis for axis in axes if axis[1] in table.columns]
    labels = [axis[0] for axis in axes]

    fig = go.Figure()
    for player, color in selected:
        values, hovers = [], []
        for label, column, per90 in axes:
            series = table[column] / minutes * 90 if per90 else table[column]
            raw = series.get(player)
            values.append(percentile_of(series[in_pool], raw))
            if pd.isna(raw):
                shown = "–"
            else:
                shown = f"{raw:.2f} / 90 min" if per90 else f"{raw:.0f} %"
            hovers.append(f"<b>{escape(str(player))}</b><br>{label} : {shown}")
        fig.add_trace(go.Scatterpolar(
            r=values + values[:1],
            theta=labels + labels[:1],
            hovertext=hovers + hovers[:1],
            hoverinfo="text",
            fill="toself",
            name=player,
            line=dict(color=color, width=2.5),
            fillcolor=hex_to_rgba(color, 0.25),
            marker=dict(size=5),
        ))

    grid = "rgba(128, 128, 128, 0.3)"
    fig.update_layout(
        height=480,
        # Marges larges : les libellés des axes (en haut, en bas, sur les côtés) ne sont plus coupés
        margin=dict(l=70, r=70, t=45, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                range=[0, 100], tickvals=[25, 50, 75, 100], showticklabels=False,
                ticks="", showline=False, gridcolor=grid,
            ),
            angularaxis=dict(gridcolor=grid, linecolor=grid, tickfont=dict(size=12), ticks=""),
        ),
    )
    return fig


def chips_html(items):
    chips = "".join(
        f'<div class="pl-chip"><strong>{value}</strong><span>{label}</span></div>' for label, value in items
    )
    return f'<div class="pl-chips">{chips}</div>'


def player_cards_html(table, selected):
    """Une carte par joueur sélectionné : nom, équipe et chiffres clés."""
    cards = []
    for player, color in selected:
        row = table.loc[player]
        precision = "–" if pd.isna(row["Précision des passes"]) else f"{row['Précision des passes']:.0f} %"
        minutes = f"{row['Minutes Played']:.0f}"
        if row["Actions gardien"] >= 5:
            items = [
                ("Min", minutes), ("Arrêts", int(row["Arrêts"])), ("Prises", int(row["Prises"])),
                ("Relances", int(row["Relances"])), ("Passes", int(row["Passes"])), ("Précision des passes", precision),
            ]
        else:
            goals = int(row["Goals"]) if "Goals" in row.index and pd.notna(row["Goals"]) else 0
            items = [
                ("Min", minutes), ("Buts", goals), ("Tirs", int(row["Tirs"])), ("Passes", int(row["Passes"])),
                ("Précision des passes", precision), ("Fautes", int(row["Fautes"])),
                ("Cartons", int(row["Jaunes"] + row["Rouges"])),
            ]
        cards.append(
            f'<div class="pl-card" style="border-top-color:{color}">'
            f'<div class="pl-head"><span class="pl-dot" style="background:{color}"></span>{escape(str(player))}</div>'
            f'<div class="pl-team">{escape(str(row["Équipe"]))}</div>'
            f"{chips_html(items)}"
            "</div>"
        )
    return '<div class="pl-cards">' + "".join(cards) + "</div>"


def render_player_section(df, df_players, events, teams, colors, match_id):
    if not {"Player Name", "Minutes Played"}.issubset(df_players.columns):
        st.info("Les statistiques par joueur ne sont pas disponibles pour ce match.")
        return

    table = compute_player_table(df, df_players, events)
    keys = [f"player_pick_{i}_{match_id}" for i in range(2)]
    order_key = f"player_order_{match_id}"

    rosters = []
    for team in teams[:2]:
        sub = table[(table["Équipe"] == team) & (table["Minutes Played"].fillna(0) > 0)]
        rosters.append(list(sub.sort_values("Minutes Played", ascending=False).index))

    def on_pick():
        # Deux joueurs au maximum, de n'importe quelle équipe : un troisième remplace le plus ancien
        current = []
        for key in keys:
            current += st.session_state.get(key) or []
        previous = st.session_state.get(order_key, [])
        ordered = [p for p in previous if p in current] + [p for p in current if p not in previous]
        ordered = ordered[-2:]
        st.session_state[order_key] = ordered
        for key, roster in zip(keys, rosters):
            st.session_state[key] = [p for p in ordered if p in roster]

    for team, roster, key in zip(teams[:2], rosters, keys):
        st.markdown(
            f'<div class="scorers-head" style="color:{colors[team]}">{escape(str(team))}</div>',
            unsafe_allow_html=True,
        )
        st.pills(
            str(team),
            roster,
            selection_mode="multi",
            key=key,
            format_func=short_name,
            on_change=on_pick,
            label_visibility="collapsed",
        )

    chosen = [p for p in st.session_state.get(order_key, []) if p in table.index]
    if not chosen:
        st.caption("Choisis un joueur, ou deux pour les comparer (de la même équipe ou non). Un troisième remplace le plus ancien.")
        return

    # Couleur d'équipe ; si les deux joueurs sont dans la même équipe, le second prend une couleur distincte
    selected = []
    for player in chosen:
        color = colors.get(table.loc[player, "Équipe"], FALLBACK_COLOR)
        if any(color == c for _, c in selected):
            color = "#8E6CC8"
        selected.append((player, color))

    st.markdown(player_cards_html(table, selected), unsafe_allow_html=True)

    keepers = [p for p, _ in selected if table.loc[p, "Actions gardien"] >= 5]
    outfield = [(p, c) for p, c in selected if p not in keepers]
    short = [p for p, _ in outfield if table.loc[p, "Minutes Played"] < 20]
    if keepers:
        st.caption("Pas de radar pour un gardien : le comparer aux joueurs de champ n'aurait pas de sens.")
    if short:
        st.caption(f"Moins de 20 minutes jouées pour {', '.join(short)} : son radar est peu fiable.")
    if not outfield:
        return

    axes = [axis for axis in RADAR_AXES if axis[1] in table.columns]
    if len(axes) >= 3:
        with st.container(border=True):
            st.plotly_chart(
                radar_figure(table, outfield, axes),
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"radar_{match_id}",
            )
    st.caption(
        "Chaque axe compare le joueur aux autres joueurs du match ayant joué au moins 20 minutes "
        "(100 = le meilleur). Les volumes sont ramenés à 90 minutes."
    )


MAP_CHOICES = ["Passes", "Tirs", "Carte d'activité", "Récupérations", "Réseau de passes"]


def event_minutes(d):
    """Minute de jeu de chaque événement (Time est en secondes et repart de 0 en seconde période)."""
    return pd.to_numeric(d["Time"], errors="coerce") / 60 + d["Half"].astype(float) * 45


def map_events(df, kind, players, period):
    """Événements à placer sur le terrain, pour des joueurs et une période (en minutes) donnés."""
    ptm = player_team_map(df)
    d = df.reset_index(drop=True)
    team = d["Player1 Name"].map(ptm).fillna(d["Player1 Team"])
    m = event_masks(d, team)
    minute = event_minutes(d)

    def flag(column):
        if column not in d.columns:
            return pd.Series(False, index=d.index)
        return d[column].fillna(False).astype(bool)

    mask = (
        d["X"].notna() & d["Y"].notna()
        & d["Player1 Name"].isin(players)
        & minute.between(period[0], period[1])
    )
    if kind == "Passes":
        mask &= m["pass"]
    elif kind == "Tirs":
        mask &= m["shot"] | m["extra_goal"]
        # xG du tir ; s'il est porté par l'événement « But » qui le suit, on le récupère
        xg = pd.to_numeric(d["xG Score"], errors="coerce") if "xG Score" in d.columns else pd.Series(np.nan, index=d.index)
        for shot_index, goal_index in m["shot_to_goal"].items():
            if not xg.get(shot_index, 0) > 0 and xg.get(goal_index, 0) > 0:
                xg[shot_index] = xg[goal_index]
        is_goal_shot = m["scoring_shot"] | m["extra_goal"]
    elif kind == "Récupérations":
        mask &= flag("Possession Regain")

    out = d[mask].copy()
    out["Player1 Team"] = team[mask]
    out["Time"] = minute[mask].round()  # minutes, pour l'infobulle du terrain
    if kind == "Tirs":
        out["xg"] = xg[mask]
        out["goal"] = is_goal_shot[mask]
    if kind == "Passes":
        # Une passe va jusqu'à l'événement suivant (même mi-temps)
        same_half = d["Half"].shift(-1) == d["Half"]
        out["end_x"] = d["X"].shift(-1)[mask]
        out["end_y"] = d["Y"].shift(-1)[mask]
        keep = same_half[mask] & out["end_x"].notna() & out["end_y"].notna()
        out = out[keep]
    return out


def pass_network(df, teams, colors, players, period, min_links):
    """Réseau de passes : joueurs placés à leur position moyenne, reliés selon leurs échanges de passes.

    Le destinataire d'une passe est estimé : Player2 s'il est renseigné, sinon l'auteur de l'événement
    suivant (même équipe, même mi-temps, dans les 10 secondes). Les liens comptent les passes dans
    les deux sens. Retourne (joueurs, liens, nombre de passes échangées).
    """
    ptm = player_team_map(df)
    d = df.reset_index(drop=True)
    team = d["Player1 Name"].map(ptm).fillna(d["Player1 Team"])
    m = event_masks(d, team)
    in_period = event_minutes(d).between(period[0], period[1])
    player = d["Player1 Name"]

    next_player = player.shift(-1)
    next_ok = (
        (team.shift(-1) == team)
        & (d["Half"].shift(-1) == d["Half"])
        & (d["Time"].shift(-1) - d["Time"]).between(0, 10)
        & next_player.notna()
        & (next_player != player)
    )
    receiver = next_player.where(next_ok)
    if "Player2 Name" in d.columns:
        second = d["Player2 Name"]
        explicit = second.notna() & (second.map(ptm) == team) & (second != player)
        receiver = second.where(explicit, receiver)

    linked = m["pass"] & in_period & player.isin(players) & receiver.isin(players)
    directed = pd.DataFrame({"a": player[linked], "b": receiver[linked]}).groupby(["a", "b"]).size().to_dict()

    pair_total = {}
    involvement = {}
    for (a, b), n in directed.items():
        key = tuple(sorted((a, b)))
        pair_total[key] = pair_total.get(key, 0) + n
        involvement[a] = involvement.get(a, 0) + n
        involvement[b] = involvement.get(b, 0) + n

    # Positions moyennes (coordonnées normalisées comme sur les autres cartes, Y inversé comme dans les compositions)
    valid = d["X"].notna() & d["Y"].notna()
    nx = ((d["X"] + 53) * (100 / 105.8)).clip(0, 100)
    ny = ((-d["Y"] + 34.5) * (100 / 68)).clip(0, 100)
    flipped = d["Half"] == 1
    nx, ny = nx.where(~flipped, 100 - nx), ny.where(~flipped, 100 - ny)
    t0, t1 = teams[:2]
    if nx[valid & (team == t0)].mean() > nx[valid & (team == t1)].mean():
        nx, ny = 100 - nx, 100 - ny  # la première équipe attaque vers la droite, comme dans les compositions
    present = valid & in_period & player.isin(players)
    position = pd.DataFrame({"x": nx[present], "y": ny[present], "p": player[present]}).groupby("p").mean()

    top = max(involvement.values(), default=1)
    nodes = []
    for name, row in position.iterrows():
        count = involvement.get(name, 0)
        nodes.append({
            "name": name,
            "label": short_name(name),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "color": colors.get(ptm.get(name), FALLBACK_COLOR),
            "size": 16 + 22 * float(np.sqrt(count / top)) if count else 14,
            "hover": f"<b>{escape(str(name))}</b><br>{count} passes échangées",
        })

    edges = []
    for (a, b), total in pair_total.items():
        if total < min_links or a not in position.index or b not in position.index:
            continue
        edges.append({
            "x0": float(position.at[a, "x"]), "y0": float(position.at[a, "y"]),
            "x1": float(position.at[b, "x"]), "y1": float(position.at[b, "y"]),
            "count": total,
            "color": colors.get(ptm.get(a), FALLBACK_COLOR),
            "hover": (
                f"<b>{escape(str(a))} ↔ {escape(str(b))}</b><br>{total} passes "
                f"({escape(short_name(a))} → {escape(short_name(b))} : {directed.get((a, b), 0)}, "
                f"{escape(short_name(b))} → {escape(short_name(a))} : {directed.get((b, a), 0)})"
            ),
        })
    return nodes, edges, sum(pair_total.values())


def map_figure(events, kind, teams):
    viz = PitchVisualizer(team_colors=TEAM_COLORS)
    if kind == "Passes":
        return viz.create_vector_plot(events, kind, teams)
    if kind == "Carte d'activité":
        return viz.creat_heat_map(events, kind)
    if kind == "Tirs":
        return viz.create_shot_plot(events, teams)
    return viz.create_point_plot(events, kind, teams)


def player_selector(team, starters, subs):
    """Cases à cocher d'une équipe + « tout sélectionner ». Renvoie la liste choisie."""
    all_key = f"{team}_select_all"

    def toggle_all():
        value = st.session_state[all_key]
        for kind, group in (("titu", starters), ("sub", subs)):
            for player in group:
                st.session_state[f"{team}_{kind}_{player}"] = value

    st.checkbox("Sélectionner toute l'équipe", key=all_key, on_change=toggle_all)

    selected = []
    st.markdown("**Titulaires**")
    for player in starters:
        if st.checkbox(player, key=f"{team}_titu_{player}"):
            selected.append(player)

    with st.expander(f"Remplaçants ({len(subs)})"):
        for player in subs:
            if st.checkbox(player, key=f"{team}_sub_{player}"):
                selected.append(player)

    return selected


def render_terrain_section(df, teams, colors, rosters):
    minutes = event_minutes(df)
    last = int(np.ceil(minutes.max())) if minutes.notna().any() else 90

    ctrl_left, ctrl_right = st.columns([1, 2], vertical_alignment="bottom")
    with ctrl_left:
        kind = st.selectbox("Données à afficher", MAP_CHOICES)
    with ctrl_right:
        period = st.slider("Période du match (minutes)", 0, last, (0, last))

    team1, team2 = list(rosters)[:2]
    players_col, pitch_col = st.columns([1, 2.4])

    with players_col:
        st.markdown('<div class="section-title">Joueurs</div>', unsafe_allow_html=True)
        with st.container(height=420, border=True):
            tab_t1, tab_t2 = st.tabs([team1, team2])
            with tab_t1:
                team1_selections = player_selector(team1, *rosters[team1])
            with tab_t2:
                team2_selections = player_selector(team2, *rosters[team2])
        selected_players = team1_selections + team2_selections
        st.caption(f"{len(selected_players)} joueur(s) sélectionné(s)")

    with pitch_col:
        if not selected_players:
            st.info("Sélectionnez des joueurs dans la liste pour afficher leurs actions.")

        if kind == "Réseau de passes":
            min_links = st.slider("Passes minimum entre deux joueurs", 1, 10, 3)
            nodes, edges, exchanged = pass_network(df, teams, colors, selected_players, period, min_links)
            figure = PitchVisualizer(team_colors=TEAM_COLORS).create_pass_network(nodes, edges)
            note = (
                f"{len(edges)} lien(s) affiché(s), {exchanged} passes échangées entre la {period[0]}e et la {period[1]}e minute. "
                "Joueurs à leur position moyenne, trait d'autant plus épais qu'ils s'échangent de passes (destinataire estimé)."
            )
        else:
            shown = map_events(df, kind, selected_players, period)
            figure = map_figure(shown, kind, teams)
            note = f"{len(shown)} événement(s) entre la {period[0]}e et la {period[1]}e minute."
            if kind == "Tirs":
                note += " Taille du point = xG ; doré = but, blanc = sans but ; anneau = équipe."
        # On garde le fond du terrain, on ne rend fluide que le cadre autour
        st.plotly_chart(
            style_figure(figure, transparent_plot=False),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption(note)


def render_compact_timeline(events, teams):
    """Version réduite : une ligne par type (buts, cartons rouges). Les cartons jaunes restent dans le détail.

    Première équipe à gauche, seconde à droite, icône au centre, minute après le nom.
    """
    kinds = (
        ("goal", '<span class="tl-ico">⚽</span>'),
        ("red", '<span class="tl-card red"></span>'),
    )
    rows = []
    for kind, icon in kinds:
        selected = [ev for ev in events if ev["kind"] == kind]
        if not selected:
            continue
        columns = []
        for team in teams[:2]:
            minutes_by_player = {}
            for ev in selected:
                if ev["team"] == team:
                    minutes_by_player.setdefault(ev["player"], []).append(ev["minute"])
            lines = []
            for player, minutes in minutes_by_player.items():
                times = ", ".join(str(m) + "'" for m in minutes)
                lines.append(
                    f'<div><span class="cp-name">{escape(str(player))}</span>'
                    f'<span class="cp-min">{times}</span></div>'
                )
            columns.append("".join(lines))
        rows.append(
            '<div class="cp-row">'
            f'<div class="cp-col left">{columns[0]}</div>'
            f'<div class="cp-ico">{icon}</div>'
            f'<div class="cp-col right">{columns[1]}</div>'
            "</div>"
        )

    if not rows:
        st.markdown('<p class="empty">Aucun but ni carton rouge pour ce match.</p>', unsafe_allow_html=True)
        return
    st.markdown('<div class="cp">' + "".join(rows) + "</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(background_css(BACKGROUND_IMAGE), unsafe_allow_html=True)

loader = DataLoader(EVENTS_PATH, LOGOS_PATH)
players_loader = PlayerLoader(PLAYERS_PATH)
processor = DataProcessor()

try:
    match_files = loader.load_match_files()

    # En-tête : titre + choix du match
    head_left, head_right = st.columns([1, 1.2], vertical_alignment="bottom")
    with head_left:
        st.markdown(
            '<h1 class="app-title">Analyse de match</h1>'
            '<p class="app-sub">Premier League 2011-12</p>',
            unsafe_allow_html=True,
        )
    with head_right:
        selected_match = st.selectbox("Match", match_files, format_func=match_label)

    # Données du match
    df = loader.load_match_data(selected_match)
    teams = loader.get_teams(df)
    logos = [loader.load_logos(teams[0]), loader.load_logos(teams[1])]
    df_players = players_loader.load_player_data(
        selected_match.replace("- Events.csv", "- Players.csv")
    )
    team_stats = processor.get_team_stats(df, teams)
    colors = {teams[0]: TEAM_COLORS[0], teams[1]: TEAM_COLORS[1]}

    # Score
    scores = [
        team_stats[team].score if hasattr(team_stats[team], "score") else 0
        for team in teams[:2]
    ]
    render_scoreboard(teams, logos, scores, colors)

    # Effectifs
    players = processor.get_players(df, teams)
    team1, team2 = players.columns
    rosters = {}
    player_team = player_team_map(df)
    for team in (team1, team2):
        # On retire les joueurs rattachés à tort à cette équipe (équipe fausse sur un remplacement)
        names = [n for n in players[team].dropna().tolist() if player_team.get(n, team) == team]
        extras = [n for n, t in player_team.items() if t == team and n not in names]
        rosters[team] = (names[:11], names[11:] + extras)

    # Contenu principal
    timeline_events = build_timeline(df, teams)

    # Navigation entre sections. On évite st.tabs : en changeant de contenu (par exemple en choisissant
    # un joueur), Streamlit peut revenir au premier onglet. Ici la section active est mémorisée.
    sections = ["Vue d'ensemble", "Statistiques", "Terrain"]
    with st.container(key="main_nav"):
        choice = st.segmented_control(
            "Section", sections, default=sections[0], key="section_choice", label_visibility="collapsed"
        )
    if choice:  # un clic sur la section active la désélectionne : on garde alors la précédente
        st.session_state["last_section"] = choice
    section = st.session_state.get("last_section", sections[0])

    if section == sections[0]:
        if "timeline_expanded" not in st.session_state:
            st.session_state["timeline_expanded"] = False

        def toggle_timeline():
            st.session_state["timeline_expanded"] = not st.session_state["timeline_expanded"]

        expanded = st.session_state["timeline_expanded"]
        title = "Chronologie du match" if expanded else "Faits marquants"
        st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

        if expanded:
            render_timeline(timeline_events, teams, colors)
        else:
            render_compact_timeline(timeline_events, teams)

        # Flèche sous la chronologie : « Détails » en version réduite, « Réduire » en version détaillée
        with st.container(key="timeline_toggle_box"):
            icon = "keyboard_arrow_up" if expanded else "keyboard_arrow_down"
            st.button(f":material/{icon}:", key="timeline_toggle", type="tertiary", on_click=toggle_timeline)
            st.markdown(
                f'<div class="arrow-label">{"Réduire" if expanded else "Détails"}</div>',
                unsafe_allow_html=True,
            )

        # Compositions, sous la chronologie
        st.markdown('<div class="section-title">Compositions</div>', unsafe_allow_html=True)
        needed = {"Player Name", "Team", "Average X", "Average Y"}
        if not needed.issubset(df_players.columns):
            st.info("Les positions moyennes ne sont pas disponibles pour ce match.")
        else:
            st.caption(
                "Titulaires placés à leur position moyenne sur l'ensemble de leur temps de jeu. "
                "La première équipe attaque vers la droite, la seconde vers la gauche."
            )
            for index, (column, team) in enumerate(zip(st.columns(2), teams[:2])):
                with column:
                    starters, team_subs = lineup_for_team(df_players, team, timeline_events)
                    st.markdown(
                        f'<div class="scorers-head" style="color:{colors[team]}">{escape(str(team))}</div>',
                        unsafe_allow_html=True,
                    )
                    lineup_fig = PitchVisualizer(team_colors=TEAM_COLORS).create_lineup_plot(starters, colors[team], attack_left=(index == 1))
                    st.plotly_chart(
                        style_figure(lineup_fig, transparent_plot=False),
                        use_container_width=True,
                        config={"displayModeBar": False},
                        key=f"lineup_{team}",
                    )
                    render_bench(team_subs)

    elif section == sections[1]:
        st.markdown('<div class="section-title">Statistiques du match</div>', unsafe_allow_html=True)
        match_stats = compute_match_stats(df, teams, timeline_events)
        render_stats_table(match_stats, teams, colors)

        st.markdown('<div class="section-title">Joueurs</div>', unsafe_allow_html=True)
        render_player_section(df, df_players, timeline_events, teams, colors, selected_match)

    else:
        render_terrain_section(df, teams, colors, rosters)

except Exception as e:
    st.error(f"Une erreur est survenue : {e}")
    with st.expander("Détails techniques"):
        st.exception(e)