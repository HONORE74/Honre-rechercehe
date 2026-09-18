# -*- coding: utf-8 -*-
"""
Tableau de bord Conformal Prediction — version Dash
Compatible Domino WebApp (remplace ipywidgets).

Lancement : python tableau_de_bord_dash.py
Accès     : http://localhost:8888
"""

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

# ══════════════════════════════════════════════════════════════════════════════
#  CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

base_path     = Path("/domino/datasets/local/conformal_pred_actuariat/DataSet")
target_folder = base_path / "Dossier_concatenee"

df_anomalies = pd.read_pickle(target_folder / "anomalies_prio.pkl")
results_v2   = pd.read_pickle(target_folder / "results_v2.pkl")

df_anomalies.drop(columns=["annee"], inplace=True, errors="ignore")
results_v2.drop(columns=["annee"],   inplace=True, errors="ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TARGET  = "Claims_incurred"
GWP     = "Earned_Premium"
LEAKS   = ["Conso", "Currencies", "period", "Reinsurance"]

ID_COLS = [c for c in ["Partner", "Companies", "Lob", "Activity", "Periodicity", "Risk"]
           if c in df_anomalies.columns]

EXCLUDE_COLS = {TARGET, GWP, "time_idx", "year", "quarter", "annee",
                "score_composite", "y_pred", "borne_basse", "borne_haute"} | set(LEAKS)

# Convertir colonnes catégorielles
for c in ID_COLS:
    if df_anomalies[c].dtype == object:
        df_anomalies[c] = df_anomalies[c].astype("category")

DF = df_anomalies.copy()

# ══════════════════════════════════════════════════════════════════════════════
#  FONCTIONS DE DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

def filtrer(axes: list, valeurs: list) -> pd.DataFrame:
    df = DF.copy()
    if axes and valeurs:
        masque = pd.Series(False, index=df.index)
        for val in valeurs:
            parties = val.split(" | ")
            cond = pd.Series(True, index=df.index)
            for col, partie in zip(axes, parties):
                cond &= df[col].astype(str) == partie
            masque |= cond
        df = df[masque]
    return df


def options_valeurs(axes: list) -> list:
    if not axes:
        return []
    if len(axes) == 1:
        vals = DF[axes[0]].dropna().astype(str).unique()
        return sorted([{"label": v, "value": v} for v in vals], key=lambda x: x["label"])
    combos = (DF[axes].drop_duplicates().dropna()
              .apply(lambda r: " | ".join(r.astype(str)), axis=1).unique())
    return sorted([{"label": v, "value": v} for v in combos], key=lambda x: x["label"])


def agreger(df: pd.DataFrame, axes: list, maille: str) -> pd.DataFrame:
    if df.empty:
        return df

    grp = list(axes)
    if maille == "Trimestre":
        grp += ["year", "quarter"]
    elif maille == "Année":
        grp += ["year"]

    if not grp:
        return df

    agg = df.groupby(grp, observed=True).agg(
        target_sum  =(TARGET,            "sum"),
        pred_sum    =("y_pred",          "sum"),
        borne_basse =("borne_basse",     "mean"),
        borne_haute =("borne_haute",     "mean"),
        n_anomalies =("score_composite", lambda x: (x == 1).sum()),
        n_total     =("score_composite", "count"),
    ).reset_index()

    # Libellé axe X
    if len(axes) == 1:
        lib = agg[axes[0]].astype(str)
    elif len(axes) > 1:
        lib = agg[axes].apply(lambda r: " | ".join(r.astype(str)), axis=1)
    else:
        lib = pd.Series(["Total"] * len(agg), index=agg.index)

    if maille == "Trimestre":
        suf = agg["year"].astype(str) + " T" + agg["quarter"].astype(str)
        agg["libelle"] = (lib + " — " + suf) if len(axes) > 0 else suf
    elif maille == "Année":
        suf = agg["year"].astype(str)
        agg["libelle"] = (lib + " — " + suf) if len(axes) > 0 else suf
    else:
        agg["libelle"] = lib

    agg["taux_anom"] = agg["n_anomalies"] / agg["n_total"].replace(0, np.nan)
    return agg


def historique(df: pd.DataFrame, axes: list) -> pd.DataFrame:
    grp = ["year", "quarter"] + list(axes)
    h = df.groupby(grp, observed=True).agg(
        target_sum=(TARGET,            "sum"),
        pred_sum  =("y_pred",          "sum"),
        bb        =("borne_basse",     "mean"),
        bh        =("borne_haute",     "mean"),
        n_anom    =("score_composite", lambda x: (x == 1).sum()),
        n_total   =("score_composite", "count"),
    ).reset_index()
    h["periode"] = h["year"].astype(str) + " T" + h["quarter"].astype(str)
    return h.sort_values(["year", "quarter"])


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPHIQUES
# ══════════════════════════════════════════════════════════════════════════════

BLANC = "plotly_white"

def fig_barres(agg: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not agg.empty:
        fig.add_trace(go.Bar(x=agg["libelle"], y=agg["target_sum"],
                             name="Réel", marker_color="#1565C0", opacity=0.85))
        fig.add_trace(go.Bar(x=agg["libelle"], y=agg["pred_sum"],
                             name="Prédit", marker_color="#90CAF9", opacity=0.85))
    fig.update_layout(title="Réel vs Prédit", barmode="group", template=BLANC,
                      legend=dict(orientation="h", y=1.08),
                      margin=dict(l=40, r=10, t=60, b=80), xaxis_tickangle=-30)
    return fig


def fig_intervalles(agg: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not agg.empty:
        x  = list(agg["libelle"])
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=list(agg["borne_haute"]) + list(agg["borne_basse"])[::-1],
            fill="toself", fillcolor="rgba(144,202,249,0.3)",
            line=dict(color="rgba(0,0,0,0)"), name="Intervalle conformal",
        ))
        fig.add_trace(go.Scatter(x=x, y=agg["borne_haute"], mode="lines",
                                  line=dict(color="#1565C0", dash="dash"), name="Borne haute"))
        fig.add_trace(go.Scatter(x=x, y=agg["borne_basse"], mode="lines",
                                  line=dict(color="#1565C0", dash="dash"), name="Borne basse"))
        fig.add_trace(go.Scatter(x=x, y=agg["target_sum"], mode="lines+markers",
                                  line=dict(color="#C62828", width=2),
                                  marker=dict(size=6), name="Réel"))
    fig.update_layout(title="Intervalles conformaux", template=BLANC,
                      legend=dict(orientation="h", y=1.08),
                      margin=dict(l=40, r=10, t=60, b=80), xaxis_tickangle=-30)
    return fig


def fig_taux(agg: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not agg.empty:
        couleurs = ["#C62828" if t > 0.1 else "#E65100" if t > 0.05 else "#2E7D32"
                    for t in agg["taux_anom"].fillna(0)]
        fig.add_trace(go.Bar(
            x=agg["libelle"], y=agg["taux_anom"] * 100,
            marker_color=couleurs,
            text=(agg["taux_anom"] * 100).round(1).astype(str) + "%",
            textposition="outside",
        ))
    fig.update_layout(title="Taux d'anomalie (%)", template=BLANC,
                      margin=dict(l=40, r=10, t=60, b=80), xaxis_tickangle=-30)
    return fig


def fig_scatter(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not df.empty:
        couleurs = df["score_composite"].map({1: "#C62828", 0: "#1565C0"})
        fig.add_trace(go.Scatter(
            x=df["y_pred"], y=df[TARGET], mode="markers",
            marker=dict(color=couleurs, size=5, opacity=0.6),
            text=df["score_composite"].map({1: "Anomalie", 0: "Normal"}),
            hovertemplate="Prédit: %{x:.0f}<br>Réel: %{y:.0f}<br>%{text}<extra></extra>",
        ))
        vmin = min(df["y_pred"].min(), df[TARGET].min())
        vmax = max(df["y_pred"].max(), df[TARGET].max())
        fig.add_trace(go.Scatter(x=[vmin, vmax], y=[vmin, vmax], mode="lines",
                                  line=dict(color="grey", dash="dash"), name="y = x"))
    fig.update_layout(title="Réel vs Prédit (individuel)", template=BLANC,
                      xaxis_title="Prédit", yaxis_title="Réel",
                      margin=dict(l=40, r=10, t=60, b=40))
    return fig


def fig_hist(hist: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not hist.empty:
        x = list(hist["periode"])
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=list(hist["bh"]) + list(hist["bb"])[::-1],
            fill="toself", fillcolor="rgba(144,202,249,0.3)",
            line=dict(color="rgba(0,0,0,0)"), name="Intervalle",
        ))
        fig.add_trace(go.Scatter(x=x, y=hist["bh"], mode="lines",
                                  line=dict(color="#1565C0", dash="dash"), name="Borne haute"))
        fig.add_trace(go.Scatter(x=x, y=hist["bb"], mode="lines",
                                  line=dict(color="#1565C0", dash="dash"), name="Borne basse"))
        fig.add_trace(go.Scatter(
            x=x, y=hist["target_sum"], mode="lines+markers",
            line=dict(color="#C62828", width=2),
            marker=dict(size=8, color=["#C62828" if a > 0 else "#2E7D32"
                                        for a in hist["n_anom"]]),
            name="Réel",
        ))
        fig.add_trace(go.Scatter(x=x, y=hist["pred_sum"], mode="lines",
                                  line=dict(color="#1565C0", width=1.5, dash="dot"),
                                  name="Prédit"))
    fig.update_layout(title="Évolution trimestrielle", template=BLANC,
                      legend=dict(orientation="h", y=1.08),
                      margin=dict(l=40, r=10, t=60, b=80), xaxis_tickangle=-30)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  CARTES KPI
# ══════════════════════════════════════════════════════════════════════════════

def cartes_kpi(df: pd.DataFrame) -> dbc.Row:
    if df.empty:
        return html.Div("Aucune donnée", className="text-muted")

    n_total  = len(df)
    n_anom   = int((df["score_composite"] == 1).sum())
    taux     = n_anom / n_total * 100 if n_total > 0 else 0
    t_sum    = df[TARGET].sum()
    p_sum    = df["y_pred"].sum()
    ecart    = abs(t_sum - p_sum) / abs(t_sum) * 100 if t_sum != 0 else 0

    def carte(label, valeur, couleur):
        return dbc.Col(dbc.Card(dbc.CardBody([
            html.P(label, className="text-muted small mb-1"),
            html.H4(valeur, className="mb-0 fw-bold", style={"color": couleur}),
        ]), className="shadow-sm h-100"), width=3)

    return dbc.Row([
        carte("Observations",     f"{n_total:,}".replace(",", " "), "#1565C0"),
        carte("Anomalies",        f"{n_anom:,}".replace(",", " "), "#C62828"),
        carte("Taux d'anomalie",  f"{taux:.1f} %",                 "#E65100"),
        carte("Écart préd/réel",  f"{ecart:.1f} %",                "#2E7D32"),
    ], className="g-2 mb-3")


# ══════════════════════════════════════════════════════════════════════════════
#  APPLICATION DASH
# ══════════════════════════════════════════════════════════════════════════════

app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True,
           title="Dashboard Conformal Prediction")

opts_axes   = [{"label": c, "value": c} for c in ID_COLS]
opts_maille = [{"label": m, "value": m} for m in ["Agrégat", "Année", "Trimestre"]]

app.layout = dbc.Container(fluid=True, style={"padding": "16px"}, children=[

    # ── En-tête ────────────────────────────────────────────────────────────
    dbc.Row(dbc.Col(html.Div([
        html.H4("Tableau de bord — Détection d'anomalies conformes",
                className="mb-0 text-white"),
        html.Small("Claims Incurred · IFRS 17 / Solvabilité II — Conformal Prediction",
                   className="text-white-50"),
    ], style={"background": "#1565C0", "padding": "14px 20px",
              "borderRadius": "8px", "marginBottom": "14px"}))),

    # ── Filtres ────────────────────────────────────────────────────────────
    dbc.Card(dbc.CardBody(dbc.Row([
        dbc.Col([
            html.Label("Axe(s) d'analyse", className="fw-bold small"),
            dcc.Dropdown(id="dd-axes", options=opts_axes,
                         value=[ID_COLS[0]] if ID_COLS else [],
                         multi=True, placeholder="Choisir un axe…"),
        ], md=4),
        dbc.Col([
            html.Label("Maille temporelle", className="fw-bold small"),
            dcc.Dropdown(id="dd-maille", options=opts_maille,
                         value="Agrégat", clearable=False),
        ], md=3),
        dbc.Col([
            html.Label("Valeur(s)", className="fw-bold small"),
            dcc.Dropdown(id="dd-valeur", options=[], value=[],
                         multi=True, placeholder="Toutes"),
        ], md=5),
    ], className="g-2")), className="mb-3 shadow-sm"),

    # ── KPI ────────────────────────────────────────────────────────────────
    html.Div(id="zone-kpi", className="mb-3"),

    # ── Graphiques (2 × 2) ─────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(dcc.Graph(id="g-barres"),     md=6),
        dbc.Col(dcc.Graph(id="g-intervalles"),md=6),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="g-taux"),  md=6),
        dbc.Col(dcc.Graph(id="g-scatter"),md=6),
    ], className="mb-3"),

    # ── Historique ─────────────────────────────────────────────────────────
    dbc.Row(dbc.Col(dcc.Graph(id="g-historique"), md=12), className="mb-3"),

    # ── Tableau détail ─────────────────────────────────────────────────────
    dbc.Card([
        dbc.CardHeader(html.Strong("Détail des observations anormales")),
        dbc.CardBody(html.Div(id="zone-tableau")),
    ], className="mb-4 shadow-sm"),
])


# ── Callback 1 : mise à jour des valeurs de filtre selon l'axe ────────────
@app.callback(
    Output("dd-valeur", "options"),
    Output("dd-valeur", "value"),
    Input("dd-axes", "value"),
)
def cb_options(axes):
    return options_valeurs(axes or []), []


# ── Callback 2 : mise à jour de tous les panneaux ─────────────────────────
@app.callback(
    Output("zone-kpi",       "children"),
    Output("g-barres",       "figure"),
    Output("g-intervalles",  "figure"),
    Output("g-taux",         "figure"),
    Output("g-scatter",      "figure"),
    Output("g-historique",   "figure"),
    Input("dd-axes",   "value"),
    Input("dd-maille", "value"),
    Input("dd-valeur", "value"),
)
def cb_panneaux(axes, maille, valeurs):
    axes    = axes    or []
    valeurs = valeurs or []
    maille  = maille  or "Agrégat"

    df_f = filtrer(axes, valeurs)
    agg  = agreger(df_f, axes, maille)
    hist = historique(df_f, axes)

    return (cartes_kpi(df_f),
            fig_barres(agg),
            fig_intervalles(agg),
            fig_taux(agg),
            fig_scatter(df_f),
            fig_hist(hist))


# ── Callback 3 : tableau des anomalies ────────────────────────────────────
@app.callback(
    Output("zone-tableau", "children"),
    Input("dd-axes",   "value"),
    Input("dd-valeur", "value"),
)
def cb_tableau(axes, valeurs):
    axes    = axes    or []
    valeurs = valeurs or []

    df_f   = filtrer(axes, valeurs)
    df_anom = df_f[df_f["score_composite"] == 1].copy()

    if df_anom.empty:
        return html.P("Aucune anomalie dans la sélection.", className="text-success")

    cols = list(dict.fromkeys(
        axes + [TARGET, "y_pred", "borne_basse", "borne_haute",
                "score_composite", "year", "quarter"]
    ))
    cols = [c for c in cols if c in df_anom.columns]
    df_show = df_anom[cols].head(500)

    for c in [TARGET, "y_pred", "borne_basse", "borne_haute"]:
        if c in df_show.columns:
            df_show[c] = df_show[c].round(2)

    return dash_table.DataTable(
        data=df_show.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df_show.columns],
        page_size=15,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#1565C0", "color": "white",
                      "fontWeight": "bold"},
        style_data_conditional=[{
            "if": {"filter_query": "{score_composite} = 1"},
            "backgroundColor": "#FFEBEE", "color": "#C62828",
        }],
        style_cell={"fontSize": "12px", "padding": "6px", "fontFamily": "sans-serif"},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  LANCEMENT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Données chargées : {len(DF):,} lignes".replace(",", " "))
    print(f"Anomalies        : {int((DF['score_composite']==1).sum()):,}".replace(",", " "))
    print(f"ID_COLS          : {ID_COLS}")
    print(f"TARGET           : {TARGET}")
    print("Dashboard sur    : http://localhost:8888")
    app.run(debug=False, port=8888, host="0.0.0.0")
