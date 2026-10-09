import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Sequence


class PitchVisualizer:
    # Couleurs du terrain
    GRASS_LIGHT = "#2F8553"
    GRASS_DARK = "#2A7A4B"
    LINE_COLOR = "rgba(255, 255, 255, 0.85)"

    def __init__(self, team_colors: Sequence[str] = ("#4FA3E0", "#E4605A"), height: int = 440):
        self.pitch_width = 100
        self.pitch_height = 100
        self.real_pitch_width = 105.8
        self.real_pitch_height = 68
        self.team_colors = team_colors
        self.height = height

        # Mètres -> unités du graphique (axes normalisés de 0 à 100)
        self.kx = self.pitch_width / self.real_pitch_width
        self.ky = self.pitch_height / self.real_pitch_height

    # ------------------------------------------------------------------
    # Terrain
    # ------------------------------------------------------------------
    def _team_color(self, team, teams) -> str:
        return self.team_colors[1] if team == teams[1] else self.team_colors[0]

    def _line(self, fig, **kwargs):
        fig.add_shape(line=dict(color=self.LINE_COLOR, width=2), layer="below", **kwargs)

    def _arc_path(self, cx, cy, rx, ry, a0, a1, mirror=False, n=40) -> str:
        points = []
        for angle in np.linspace(np.radians(a0), np.radians(a1), n):
            x = cx + rx * np.cos(angle)
            y = cy + ry * np.sin(angle)
            points.append((100 - x if mirror else x, y))
        return "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in points)

    def _create_base_pitch(self, background_color: str = None) -> go.Figure:
        """Crée un terrain rayé avec ses lignes, à l'échelle réelle."""
        fig = go.Figure()

        # Pelouse rayée
        stripes = 10
        for i in range(stripes):
            fig.add_shape(
                type="rect",
                x0=i * 100 / stripes, x1=(i + 1) * 100 / stripes, y0=0, y1=100,
                fillcolor=background_color or (self.GRASS_LIGHT if i % 2 == 0 else self.GRASS_DARK),
                line=dict(width=0),
                layer="below",
            )

        # Contour et ligne médiane
        self._line(fig, type="rect", x0=0, y0=0, x1=100, y1=100)
        self._line(fig, type="line", x0=50, y0=0, x1=50, y1=100)

        # Rond central et point central
        rx, ry = 9.15 * self.kx, 9.15 * self.ky
        self._line(fig, type="circle", x0=50 - rx, y0=50 - ry, x1=50 + rx, y1=50 + ry)
        fig.add_shape(
            type="circle", x0=50 - 0.5, x1=50 + 0.5, y0=50 - 0.75, y1=50 + 0.75,
            fillcolor=self.LINE_COLOR, line=dict(width=0), layer="below",
        )

        # Surfaces, six mètres, point de penalty, arc, but (de chaque côté)
        pen_depth, pen_half = 16.5 * self.kx, 20.16 * self.ky
        box_depth, box_half = 5.5 * self.kx, 9.16 * self.ky
        spot = 11 * self.kx
        goal_depth, goal_half = 1.8, 3.66 * self.ky
        arc_angle = np.degrees(np.arccos((pen_depth - spot) / rx))

        for mirror in (False, True):
            def mx(x):
                return 100 - x if mirror else x

            self._line(fig, type="rect", x0=mx(0), x1=mx(pen_depth), y0=50 - pen_half, y1=50 + pen_half)
            self._line(fig, type="rect", x0=mx(0), x1=mx(box_depth), y0=50 - box_half, y1=50 + box_half)
            self._line(fig, type="rect", x0=mx(0), x1=mx(-goal_depth), y0=50 - goal_half, y1=50 + goal_half)

            fig.add_shape(
                type="circle",
                x0=mx(spot) - 0.5, x1=mx(spot) + 0.5, y0=50 - 0.75, y1=50 + 0.75,
                fillcolor=self.LINE_COLOR, line=dict(width=0), layer="below",
            )
            self._line(
                fig, type="path",
                path=self._arc_path(spot, 50, rx, ry, -arc_angle, arc_angle, mirror=mirror),
            )

        return fig

    def _normalize_coordinates(self, x: float, y: float) -> tuple[float, float]:
        """Normalise les coordonnées du terrain réel vers l'affichage."""
        norm_x = (x + 53) * (self.pitch_width / self.real_pitch_width)
        norm_y = (y + 34.5) * (self.pitch_height / self.real_pitch_height)
        return np.clip(norm_x, 0, 100), np.clip(norm_y, 0, 100)

    # ------------------------------------------------------------------
    # Graphiques
    # ------------------------------------------------------------------
    def create_vector_plot(self, df: pd.DataFrame, title: str, teams) -> go.Figure:
        fig = self._create_base_pitch()

        if df.empty:
            self._update_layout(fig, title)
            return fig

        start_points = []
        hover_texts = []
        colors = []

        for _, row in df.iterrows():
            start_x, start_y = self._normalize_coordinates(row['X'], row['Y'])
            end_x, end_y = self._normalize_coordinates(row['end_x'], row['end_y'])

            # Miroir pour les événements joués dans l'autre sens
            if row['Half'] == 1:
                start_x, start_y = 100 - start_x, 100 - start_y
                end_x, end_y = 100 - end_x, 100 - end_y

            start_points.append((start_x, start_y))
            hover_texts.append(f"{row['Player1 Name']}")
            colors.append(self._team_color(row['Player1 Team'], teams))

            # Flèche : pas de marqueur au départ, une pointe à l'arrivée
            fig.add_trace(go.Scatter(
                x=[start_x, end_x],
                y=[start_y, end_y],
                mode='lines+markers',
                line=dict(width=2, color='rgba(255, 255, 255, 0.85)'),
                marker=dict(
                    symbol="arrow",
                    size=[0, 12],
                    color='white',
                    angleref="previous",
                ),
                hoverinfo='none',
            ))

        # Points de départ, colorés selon l'équipe
        fig.add_trace(go.Scatter(
            x=[p[0] for p in start_points],
            y=[p[1] for p in start_points],
            mode='markers',
            marker=dict(size=9, color=colors, opacity=0.95, line=dict(color='white', width=1)),
            text=hover_texts,
            hoverinfo='text',
        ))

        self._update_layout(fig, title)
        return fig

    def create_point_plot(self, df: pd.DataFrame, title: str, teams) -> go.Figure:
        fig = self._create_base_pitch()

        if df.empty:
            self._update_layout(fig, title)
            return fig

        x_coords, y_coords, hover_texts, colors = [], [], [], []

        for _, row in df.iterrows():
            x, y = self._normalize_coordinates(row['X'], row['Y'])

            if row['Half'] == 1:
                x, y = 100 - x, 100 - y

            x_coords.append(x)
            y_coords.append(y)
            hover_texts.append(
                f"{row['Player1 Name']}<br>{row['Event Name']}, {int(row['Time'])}'"
            )
            colors.append(self._team_color(row['Player1 Team'], teams))

        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(size=9, color=colors, opacity=0.95, line=dict(color='white', width=1)),
            text=hover_texts,
            hoverinfo='text',
        ))

        self._update_layout(fig, title)
        return fig

    @staticmethod
    def _rgba(color: str, alpha: float) -> str:
        h = color.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha:.2f})"

    def create_pass_network(self, nodes, edges) -> go.Figure:
        """Réseau de passes : joueurs placés à leur position moyenne, reliés selon leurs échanges.

        nodes : liste de dict (label, x, y, color, size, hover), coordonnées déjà normalisées (0 à 100).
        edges : liste de dict (x0, y0, x1, y1, count, color, hover). L'épaisseur du trait suit le nombre de passes.
        """
        fig = self._create_base_pitch()
        if not nodes:
            self._update_layout(fig, "")
            return fig

        top = max((e["count"] for e in edges), default=1)
        for e in sorted(edges, key=lambda e: e["count"]):  # les liens les plus forts sont dessinés au-dessus
            share = e["count"] / top
            fig.add_trace(go.Scatter(
                x=[e["x0"], e["x1"]], y=[e["y0"], e["y1"]], mode="lines",
                line=dict(width=1.5 + 9 * share, color=self._rgba(e["color"], 0.35 + 0.55 * share)),
                hoverinfo="skip", showlegend=False,
            ))
        if edges:
            fig.add_trace(go.Scatter(
                x=[(e["x0"] + e["x1"]) / 2 for e in edges],
                y=[(e["y0"] + e["y1"]) / 2 for e in edges],
                mode="markers", marker=dict(size=18, opacity=0),
                hovertext=[e["hover"] for e in edges], hoverinfo="text", showlegend=False,
            ))

        ordered = sorted(nodes, key=lambda n: n["x"])
        positions = ["top center" if i % 2 == 0 else "bottom center" for i in range(len(ordered))]
        fig.add_trace(go.Scatter(
            x=[n["x"] for n in ordered],
            y=[n["y"] for n in ordered],
            mode="markers+text",
            text=[n["label"] for n in ordered],
            textposition=positions,
            textfont=dict(color="white", size=12),
            marker=dict(size=[n["size"] for n in ordered], color=[n["color"] for n in ordered],
                        line=dict(color="white", width=2)),
            hovertext=[n["hover"] for n in ordered], hoverinfo="text", showlegend=False,
            cliponaxis=False,
        ))
        self._update_layout(fig, "")
        return fig

    def create_shot_plot(self, df: pd.DataFrame, teams) -> go.Figure:
        """Tirs : taille du point selon le xG, couleur selon l'issue, anneau à la couleur de l'équipe.

        df doit contenir X, Y, Half, Time (en minutes), Player1 Name, Player1 Team, xg et goal.
        """
        fig = self._create_base_pitch()
        if df.empty:
            self._update_layout(fig, "")
            return fig

        groups = {"Sans but": [], "But": []}  # les buts sont dessinés en dernier, donc au-dessus
        for _, row in df.iterrows():
            x, y = self._normalize_coordinates(row["X"], row["Y"])
            if row["Half"] == 1:
                x, y = 100 - x, 100 - y
            xg = row.get("xg", np.nan)
            size = 10 + 46 * float(np.sqrt(min(max(xg, 0.0), 1.0))) if pd.notna(xg) else 12
            outcome = "But" if bool(row.get("goal", False)) else "Sans but"
            xg_text = "xG n/d" if pd.isna(xg) else f"xG {xg:.2f}"
            hover = f"{row['Player1 Name']}<br>{int(row['Time'])}' — {xg_text}<br>{outcome}"
            groups[outcome].append((x, y, size, self._team_color(row["Player1 Team"], teams), hover))

        fills = {"Sans but": "rgba(255, 255, 255, 0.55)", "But": "#F2C94C"}
        for outcome, points in groups.items():
            if not points:
                continue
            fig.add_trace(go.Scatter(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                mode="markers",
                name=outcome,
                marker=dict(size=[p[2] for p in points], color=fills[outcome],
                            line=dict(color=[p[3] for p in points], width=3)),
                hovertext=[p[4] for p in points],
                hoverinfo="text",
            ))

        self._update_layout(fig, "")
        fig.update_layout(
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.35)", font=dict(color="white")),
        )
        return fig

    def creat_heat_map(self, df: pd.DataFrame, title: str):
        fig = self._create_base_pitch()

        mat = np.zeros((100, 100))

        for _, row in df.iterrows():
            x, y = self._normalize_coordinates(row['X'], row['Y'])

            if row['Half'] == 1:
                x, y = 100 - x, 100 - y

            self._add_gaussian(mat, int(x), int(y), sigma=2.0)

        heatmap = go.Heatmap(
            z=mat.T,
            x=np.linspace(0, 100, 100),
            y=np.linspace(0, 100, 100),
            colorscale='YlOrRd',
            showscale=False,
            opacity=0.8,
            hoverinfo='skip',
            zmin=0.1,  # les valeurs sous ce seuil restent transparentes
            zauto=False,
            xgap=0,
            ygap=0,
        )

        fig.add_trace(heatmap)
        self._update_layout(fig, title)
        return fig

    def _add_gaussian(self, M, x_center, y_center, sigma=100):
        """
        Ajoute une distribution gaussienne 2D centrée sur (x_center, y_center)

        Paramètres:
        - M: la matrice 100x100 sur laquelle ajouter la gaussienne
        - x_center, y_center: coordonnées du centre de la gaussienne (déjà normalisées)
        - sigma: écart-type de la gaussienne
        """
        if not (0 <= x_center < 100 and 0 <= y_center < 100):
            return M

        kernel_size = int(3 * sigma)

        x_min = max(0, x_center - kernel_size)
        x_max = min(99, x_center + kernel_size)
        y_min = max(0, y_center - kernel_size)
        y_max = min(99, y_center + kernel_size)

        for i in range(x_min, x_max + 1):
            for j in range(y_min, y_max + 1):
                dx = i - x_center
                dy = j - y_center
                M[i, j] += np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))

        return M

    def create_lineup_plot(self, players, color: str, height: int = 360, attack_left: bool = False) -> go.Figure:
        """Positions moyennes d'une équipe sur le terrain.

        L'équipe attaque vers la droite, ou vers la gauche si attack_left est vrai
        (le terrain est alors retourné de 180°, ce qui garde gauche et droite corrects).

        players : liste de (étiquette, x, y) en coordonnées réelles (mètres), dans le repère
        propre à l'équipe (elle attaque vers les X positifs). L'axe Y des données est inversé
        par rapport à l'affichage : un joueur de gauche a un Y négatif.
        """
        fig = self._create_base_pitch()

        points = []
        for label, x, y in players:
            nx, ny = self._normalize_coordinates(x, -y)
            if attack_left:
                nx, ny = 100 - nx, 100 - ny
            points.append((float(nx), float(ny), label))

        # Étiquettes alternées au-dessus / en dessous pour limiter les chevauchements
        points.sort(key=lambda p: p[0])
        positions = ["top center" if i % 2 == 0 else "bottom center" for i in range(len(points))]

        fig.add_trace(go.Scatter(
            x=[p[0] for p in points],
            y=[p[1] for p in points],
            mode="markers+text",
            text=[p[2] for p in points],
            textposition=positions,
            textfont=dict(color="white", size=12),
            marker=dict(size=18, color=color, line=dict(color="white", width=2)),
            hovertext=[p[2] for p in points],
            hoverinfo="text",
            cliponaxis=False,
        ))

        self._update_layout(fig, "")
        fig.update_layout(height=height)
        return fig

    def _update_layout(self, fig: go.Figure, title: str = None):
        """Terrain à l'échelle réelle, fluide en largeur, sans titre ni fond fixe."""
        # 1 unité en y = (68/100) / (105.8/100) unités en x, pour garder les vraies proportions
        ratio = self.real_pitch_height / self.real_pitch_width
        pad = 3  # marge autour du terrain pour voir les buts

        fig.update_layout(
            title=dict(text=""),
            autosize=True,
            width=None,
            height=self.height,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            xaxis=dict(
                range=[-pad, 100 + pad],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                fixedrange=True,
                constrain="domain",
            ),
            yaxis=dict(
                range=[-pad, 100 + pad],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                fixedrange=True,
                scaleanchor="x",
                scaleratio=ratio,
                constrain="domain",
            ),
        )