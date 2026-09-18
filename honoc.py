# -*- coding: utf-8 -*-
"""
Tableau de bord Conformal Prediction — version Dash
Remplace l'ancienne version ipywidgets (incompatible Domino WebApp).

Usage local :
    python tableau_de_bord_dash.py
Puis ouvrir http://localhost:8888

Usage Domino :
    Commande de démarrage : python tableau_de_bord_dash.py
    Le dashboard charge donnees_dashboard.parquet (ou .csv) à la racine du projet.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc

# ══════════════════════════════════════════════════════════════════════════════
# CLASSE SOCLE — préparation des données
# ══════════════════════════════════════════════════════════════════════════════

class Socle:
    """Couche de données : filtrage, agrégation, hiérarchie, historique."""

    def __init__(self, df: pd.DataFrame, id_cols: list, target: str):
        self.df_brut  = df.copy()
        self.id_cols  = id_cols
        self.target   = target
        self._valider_colonnes()

    def _valider_colonnes(self):
        obligatoires = {"score_composite", "y_pred", "borne_basse",
                        "borne_haute", self.target, "year", "quarter"}
        manquantes = obligatoires - set(self.df_brut.columns)
        if manquantes:
            raise ValueError(f"Colonnes manquantes dans le DataFrame : {manquantes}")

    # ── Filtrage ──────────────────────────────────────────────────────────────
    def filtrer(self, axes: list, valeurs: list) -> pd.DataFrame:
        """Filtre le DataFrame sur les axes et valeurs sélectionnés."""
        df = self.df_brut.copy()
        if axes and valeurs:
            masque = pd.Series([False] * len(df), index=df.index)
            for val in valeurs:
                parties = val.split(" | ")
                cond = pd.Series([True] * len(df), index=df.index)
                for col, partie in zip(axes, parties):
                    cond &= df[col].astype(str) == partie
                masque |= cond
            df = df[masque]
        return df

    # ── Préparation agrégation ────────────────────────────────────────────────
    def preparer(self, df: pd.DataFrame, axes: list, maille: str) -> pd.DataFrame:
        """Agrège df selon les axes et la maille temporelle."""
        if df.empty:
            return df

        grp_cols = list(axes) if axes else []
        if maille == "Trimestre":
            grp_cols += ["year", "quarter"]
        elif maille == "Année":
            grp_cols += ["year"]

        if not grp_cols:
            return df

        agg = df.groupby(grp_cols, observed=True).agg(
            target_sum    =(self.target,       "sum"),
            pred_sum      =("y_pred",          "sum"),
            borne_basse   =("borne_basse",     "mean"),
            borne_haute   =("borne_haute",     "mean"),
            score_moy     =("score_composite", "mean"),
            n_anomalies   =("score_composite", lambda x: (x == 1).sum()),
            n_total       =("score_composite", "count"),
        ).reset_index()

        # Libellé lisible pour l'axe X
        if len(axes) == 1:
            libelle_series = agg[axes[0]].astype(str)
        elif len(axes) > 1:
            libelle_series = agg[axes].apply(
                lambda row: " | ".join(row.astype(str)), axis=1
            )
        else:
            libelle_series = pd.Series(["Total"] * len(agg), index=agg.index)

        if maille == "Trimestre":
            suffixe = agg["year"].astype(str) + " T" + agg["quarter"].astype(str)
            agg["libelle"] = libelle_series + " — " + suffixe if len(axes) > 0 else suffixe
        elif maille == "Année":
            suffixe = agg["year"].astype(str)
            agg["libelle"] = libelle_series + " — " + suffixe if len(axes) > 0 else suffixe
        else:
            agg["libelle"] = libelle_series

        agg["taux_anomalie"] = agg["n_anomalies"] / agg["n_total"].replace(0, np.nan)
        return agg

    # ── Options de filtre ─────────────────────────────────────────────────────
    def options_valeurs(self, axes: list) -> list:
        """Retourne les options disponibles pour les axes sélectionnés."""
        if not axes:
            return []
        if len(axes) == 1:
            valeurs = self.df_brut[axes[0]].dropna().astype(str).unique()
            return sorted([{"label": v, "value": v} for v in valeurs],
                          key=lambda x: x["label"])
        else:
            combos = (self.df_brut[axes]
                      .drop_duplicates()
                      .dropna()
                      .apply(lambda r: " | ".join(r.astype(str)), axis=1)
                      .unique())
            return sorted([{"label": v, "value": v} for v in combos],
                          key=lambda x: x["label"])

    # ── Historique d'une ligne ────────────────────────────────────────────────
    def historique(self, df: pd.DataFrame, axes: list) -> pd.DataFrame:
        """Construit la série temporelle trimestrielle pour un sous-ensemble."""
        grp = ["year", "quarter"] + (list(axes) if axes else [])
        h = df.groupby(grp, observed=True).agg(
            target_sum =(self.target,       "sum"),
            pred_sum   =("y_pred",          "sum"),
            bb         =("borne_basse",     "mean"),
            bh         =("borne_haute",     "mean"),
            n_anom     =("score_composite", lambda x: (x == 1).sum()),
            n_total    =("score_composite", "count"),
        ).reset_index()
        h["periode"] = h["year"].astype(str) + " T" + h["quarter"].astype(str)
        h["taux"]    = h["n_anom"] / h["n_total"].replace(0, np.nan)
        return h.sort_values(["year", "quarter"])


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSANTS VISUELS
# ══════════════════════════════════════════════════════════════════════════════

def composant_cartes(df_filtre: pd.DataFrame, target: str) -> html.Div:
    """Cartes KPI — utilise des composants Dash natifs (pas de dangerouslySetInnerHTML)."""
    if df_filtre.empty:
        return html.Div("Aucune donnée", className="text-muted")

    n_total   = len(df_filtre)
    n_anom    = int((df_filtre["score_composite"] == 1).sum())
    taux      = n_anom / n_total * 100 if n_total > 0 else 0
    score_moy = df_filtre["score_composite"].mean()
    target_sum= df_filtre[target].sum()
    pred_sum  = df_filtre["y_pred"].sum()
    ecart_pct = abs(target_sum - pred_sum) / abs(target_sum) * 100 if target_sum != 0 else 0

    def carte(titre, valeur, couleur, icone=""):
        return dbc.Card([
            dbc.CardBody([
                html.P(f"{icone} {titre}", className="text-muted small mb-1"),
                html.H4(valeur, className="mb-0 fw-bold",
                        style={"color": couleur}),
            ])
        ], className="shadow-sm h-100")

    return dbc.Row([
        dbc.Col(carte("Observations",    f"{n_total:,}".replace(",", " "),
                      "#1565C0", "📊"), width=3),
        dbc.Col(carte("Anomalies",       f"{n_anom:,}".replace(",", " "),
                      "#C62828", "⚠️"), width=3),
        dbc.Col(carte("Taux d'anomalie", f"{taux:.1f} %",
                      "#E65100", "📈"), width=3),
        dbc.Col(carte("Écart préd/réel", f"{ecart_pct:.1f} %",
                      "#2E7D32", "🎯"), width=3),
    ], className="g-2 mb-3")


def fig_barres(agg: pd.DataFrame, titre: str) -> go.Figure:
    """Graphique en barres : valeur réelle vs prédiction."""
    if agg.empty:
        return go.Figure().update_layout(title=titre, template="plotly_white")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["libelle"], y=agg["target_sum"],
                         name="Réel", marker_color="#1565C0", opacity=0.85))
    fig.add_trace(go.Bar(x=agg["libelle"], y=agg["pred_sum"],
                         name="Prédit", marker_color="#90CAF9", opacity=0.85))
    fig.update_layout(
        title=titre, barmode="group", template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=80),
        xaxis_tickangle=-30,
    )
    return fig


def fig_intervalles(agg: pd.DataFrame, titre: str) -> go.Figure:
    """Graphique : valeur réelle + intervalle conformal (borne basse/haute)."""
    if agg.empty:
        return go.Figure().update_layout(title=titre, template="plotly_white")
    fig = go.Figure()
    # Zone intervalle conformal
    fig.add_trace(go.Scatter(
        x=list(agg["libelle"]) + list(agg["libelle"])[::-1],
        y=list(agg["borne_haute"]) + list(agg["borne_basse"])[::-1],
        fill="toself", fillcolor="rgba(144,202,249,0.3)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Intervalle conformal", hoverinfo="skip",
    ))
    # Borne haute
    fig.add_trace(go.Scatter(x=agg["libelle"], y=agg["borne_haute"],
                              mode="lines", line=dict(color="#1565C0", dash="dash", width=1),
                              name="Borne haute"))
    # Borne basse
    fig.add_trace(go.Scatter(x=agg["libelle"], y=agg["borne_basse"],
                              mode="lines", line=dict(color="#1565C0", dash="dash", width=1),
                              name="Borne basse"))
    # Valeur réelle
    fig.add_trace(go.Scatter(x=agg["libelle"], y=agg["target_sum"],
                              mode="lines+markers", line=dict(color="#C62828", width=2),
                              marker=dict(size=6), name="Réel"))
    fig.update_layout(
        title=titre, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=80),
        xaxis_tickangle=-30,
    )
    return fig


def fig_taux_anomalie(agg: pd.DataFrame, titre: str) -> go.Figure:
    """Graphique : taux d'anomalie en barres colorées."""
    if agg.empty:
        return go.Figure().update_layout(title=titre, template="plotly_white")
    couleurs = ["#C62828" if t > 0.1 else "#E65100" if t > 0.05 else "#2E7D32"
                for t in agg["taux_anomalie"].fillna(0)]
    fig = go.Figure(go.Bar(
        x=agg["libelle"], y=agg["taux_anomalie"] * 100,
        marker_color=couleurs,
        text=(agg["taux_anomalie"] * 100).round(1).astype(str) + "%",
        textposition="outside",
    ))
    fig.update_layout(
        title=titre, template="plotly_white", yaxis_title="Taux (%)",
        margin=dict(l=40, r=20, t=60, b=80),
        xaxis_tickangle=-30,
    )
    return fig


def fig_historique(hist: pd.DataFrame, titre: str) -> go.Figure:
    """Graphique historique trimestriel avec intervalles conformaux."""
    if hist.empty:
        return go.Figure().update_layout(title=titre, template="plotly_white")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(hist["periode"]) + list(hist["periode"])[::-1],
        y=list(hist["bh"]) + list(hist["bb"])[::-1],
        fill="toself", fillcolor="rgba(144,202,249,0.3)",
        line=dict(color="rgba(0,0,0,0)"), name="Intervalle", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=hist["periode"], y=hist["bh"],
                              mode="lines", line=dict(color="#1565C0", dash="dash", width=1),
                              name="Borne haute"))
    fig.add_trace(go.Scatter(x=hist["periode"], y=hist["bb"],
                              mode="lines", line=dict(color="#1565C0", dash="dash", width=1),
                              name="Borne basse"))
    fig.add_trace(go.Scatter(x=hist["periode"], y=hist["target_sum"],
                              mode="lines+markers", line=dict(color="#C62828", width=2),
                              marker=dict(
                                  size=8,
                                  color=["#C62828" if a > 0 else "#2E7D32"
                                         for a in hist["n_anom"]],
                              ), name="Réel"))
    fig.add_trace(go.Scatter(x=hist["periode"], y=hist["pred_sum"],
                              mode="lines", line=dict(color="#1565C0", width=1.5, dash="dot"),
                              name="Prédit"))
    fig.update_layout(
        title=titre, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=80),
        xaxis_tickangle=-30,
    )
    return fig


def fig_scatter_anomalies(df: pd.DataFrame, target: str, titre: str) -> go.Figure:
    """Scatter : réel vs prédit, anomalies en rouge."""
    if df.empty:
        return go.Figure().update_layout(title=titre, template="plotly_white")
    couleurs = df["score_composite"].map({1: "#C62828", 0: "#1565C0"})
    fig = go.Figure(go.Scatter(
        x=df["y_pred"], y=df[target],
        mode="markers",
        marker=dict(color=couleurs, size=5, opacity=0.6),
        text=df["score_composite"].map({1: "Anomalie", 0: "Normal"}),
        hovertemplate="Prédit: %{x:.0f}<br>Réel: %{y:.0f}<br>%{text}<extra></extra>",
    ))
    # Droite y = x
    vmin = min(df["y_pred"].min(), df[target].min())
    vmax = max(df["y_pred"].max(), df[target].max())
    fig.add_trace(go.Scatter(x=[vmin, vmax], y=[vmin, vmax],
                              mode="lines", line=dict(color="grey", dash="dash"),
                              name="y = x", showlegend=True))
    fig.update_layout(
        title=titre, template="plotly_white",
        xaxis_title="Valeur prédite", yaxis_title="Valeur réelle",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUCTION DE L'APPLICATION DASH
# ══════════════════════════════════════════════════════════════════════════════

def creer_app(df: pd.DataFrame, id_cols: list, target: str) -> Dash:
    """Construit et retourne l'application Dash."""

    socle = Socle(df, id_cols, target)

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        title="Dashboard Conformal Prediction",
    )

    # ── Options initiales ────────────────────────────────────────────────────
    opts_axes   = [{"label": c, "value": c} for c in id_cols]
    opts_maille = [{"label": m, "value": m}
                   for m in ["Agrégat", "Année", "Trimestre"]]

    # ── Layout ───────────────────────────────────────────────────────────────
    app.layout = dbc.Container(fluid=True, children=[

        # En-tête
        dbc.Row(dbc.Col(html.Div([
            html.H3("Tableau de bord — Détection d'anomalies conformes",
                    className="mb-0 text-white"),
            html.Small("IFRS 17 / Solvabilité II — Conformal Prediction",
                       className="text-white-50"),
        ], style={"background": "#1565C0", "padding": "16px 24px",
                  "borderRadius": "8px", "marginBottom": "16px"}))),

        # Filtres
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Axe(s) d'analyse", className="fw-bold small"),
                    dcc.Dropdown(
                        id="dd-axes",
                        options=opts_axes,
                        value=[id_cols[0]] if id_cols else [],
                        multi=True,
                        placeholder="Choisir un ou plusieurs axes…",
                    ),
                ], md=4),
                dbc.Col([
                    html.Label("Maille temporelle", className="fw-bold small"),
                    dcc.Dropdown(
                        id="dd-maille",
                        options=opts_maille,
                        value="Agrégat",
                        clearable=False,
                    ),
                ], md=3),
                dbc.Col([
                    html.Label("Valeur(s) filtrée(s)", className="fw-bold small"),
                    dcc.Dropdown(
                        id="dd-valeur",
                        options=[],
                        value=[],
                        multi=True,
                        placeholder="Toutes (aucun filtre)",
                    ),
                ], md=5),
            ], className="g-2"),
        ]), className="mb-3 shadow-sm"),

        # Cartes KPI
        html.Div(id="zone-cartes", className="mb-3"),

        # Graphiques principaux (2×2)
        dbc.Row([
            dbc.Col(dcc.Graph(id="graph-barres"), md=6),
            dbc.Col(dcc.Graph(id="graph-intervalles"), md=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="graph-taux"), md=6),
            dbc.Col(dcc.Graph(id="graph-scatter"), md=6),
        ], className="mb-3"),

        # Historique
        dbc.Row(dbc.Col(dcc.Graph(id="graph-historique"), md=12), className="mb-3"),

        # Tableau détail
        dbc.Card([
            dbc.CardHeader(html.Strong("Détail des observations anormales")),
            dbc.CardBody(html.Div(id="zone-tableau")),
        ], className="mb-4 shadow-sm"),

    ], style={"padding": "16px"})

    # ── Callbacks ─────────────────────────────────────────────────────────────

    @app.callback(
        Output("dd-valeur", "options"),
        Output("dd-valeur", "value"),
        Input("dd-axes", "value"),
    )
    def maj_options_valeur(axes):
        axes = axes or []
        return socle.options_valeurs(axes), []

    @app.callback(
        Output("zone-cartes",      "children"),
        Output("graph-barres",     "figure"),
        Output("graph-intervalles","figure"),
        Output("graph-taux",       "figure"),
        Output("graph-scatter",    "figure"),
        Output("graph-historique", "figure"),
        Input("dd-axes",   "value"),
        Input("dd-maille", "value"),
        Input("dd-valeur", "value"),
    )
    def maj_panneaux(axes, maille, valeurs):
        axes    = axes    or []
        valeurs = valeurs or []
        maille  = maille  or "Agrégat"

        df_f = socle.filtrer(axes, valeurs)
        agg  = socle.preparer(df_f, axes, maille)
        hist = socle.historique(df_f, axes)

        cartes     = composant_cartes(df_f, target)
        f_barres   = fig_barres(agg,    f"Réel vs Prédit ({maille})")
        f_interv   = fig_intervalles(agg, f"Intervalles conformaux ({maille})")
        f_taux     = fig_taux_anomalie(agg, f"Taux d'anomalie ({maille})")
        f_scatter  = fig_scatter_anomalies(df_f, target, "Réel vs Prédit (individuel)")
        f_hist     = fig_historique(hist, "Évolution trimestrielle")

        return cartes, f_barres, f_interv, f_taux, f_scatter, f_hist

    @app.callback(
        Output("zone-tableau", "children"),
        Input("dd-axes",   "value"),
        Input("dd-maille", "value"),
        Input("dd-valeur", "value"),
    )
    def maj_detail(axes, maille, valeurs):
        axes    = axes    or []
        valeurs = valeurs or []

        df_f = socle.filtrer(axes, valeurs)
        df_anom = df_f[df_f["score_composite"] == 1].copy()

        if df_anom.empty:
            return html.P("Aucune anomalie dans la sélection.",
                          className="text-success")

        cols_aff = list(dict.fromkeys(
            axes + [target, "y_pred", "borne_basse", "borne_haute",
                    "score_composite", "year", "quarter"]
        ))
        cols_aff = [c for c in cols_aff if c in df_anom.columns]
        df_show  = df_anom[cols_aff].head(500)

        for col in [target, "y_pred", "borne_basse", "borne_haute"]:
            if col in df_show.columns:
                df_show[col] = df_show[col].round(2)

        return dash_table.DataTable(
            data=df_show.to_dict("records"),
            columns=[{"name": c, "id": c} for c in df_show.columns],
            page_size=15,
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#1565C0", "color": "white",
                          "fontWeight": "bold"},
            style_data_conditional=[
                {"if": {"filter_query": "{score_composite} = 1"},
                 "backgroundColor": "#FFEBEE", "color": "#C62828"},
            ],
            style_cell={"fontSize": "12px", "padding": "6px",
                        "fontFamily": "sans-serif"},
        )

    return app


def lancer(df: pd.DataFrame, id_cols: list, target: str, port: int = 8888):
    """Lance l'application Dash."""
    app = creer_app(df, id_cols, target)
    app.run(debug=False, port=port, host="0.0.0.0")


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE — chargement automatique des données
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os

    # Chercher le fichier de données dans le répertoire courant
    if os.path.exists("donnees_dashboard.parquet"):
        print("Chargement : donnees_dashboard.parquet")
        df = pd.read_parquet("donnees_dashboard.parquet")
    elif os.path.exists("donnees_dashboard.csv"):
        print("Chargement : donnees_dashboard.csv")
        df = pd.read_csv("donnees_dashboard.csv")
    else:
        raise FileNotFoundError(
            "Fichier de données introuvable.\n"
            "Placer donnees_dashboard.parquet (ou .csv) dans le même dossier."
        )

    print(f"Données chargées : {len(df):,} lignes, {df.shape[1]} colonnes"
          .replace(",", " "))

    # ── Adapter selon les colonnes disponibles ────────────────────────────────
    # ID_COLS : colonnes d'identification (LOB, partenaire, produit…)
    # TARGET  : variable cible (sinistres, primes, provisions…)
    # Modifier ces deux lignes selon le nom réel des colonnes dans vos données.

    COLONNES_RESERVEES = {"score_composite", "y_pred", "borne_basse",
                          "borne_haute", "year", "quarter"}

    # Détection automatique du target (première colonne numérique hors réservées)
    candidats_target = [
        c for c in df.select_dtypes(include="number").columns
        if c not in COLONNES_RESERVEES
    ]
    TARGET = candidats_target[0] if candidats_target else "y_pred"

    # Détection automatique des ID_COLS (colonnes objet/catégorie hors réservées)
    ID_COLS = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c not in COLONNES_RESERVEES
    ]
    if not ID_COLS:
        ID_COLS = ["year"]   # fallback minimal

    print(f"Target : {TARGET}")
    print(f"ID_COLS : {ID_COLS}")

    lancer(df, ID_COLS, TARGET, port=8888)
