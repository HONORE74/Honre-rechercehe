Tou premeir 

# =========================================================================
# BANDE CONFORME — style reference (y_pred en abscisse)
#   Ligne noire = droite y = x (la prediction, par construction)
#   Bande       = [borne_basse ; borne_haute]
#   Points      : vert = dans l'intervalle / rouge = hors intervalle
# =========================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output

N_MAX    = 90      # observations affichees (reduit pour une figure lisible)
LISSAGE  = 0       # 0 = bornes brutes (exact). 5-11 = mediane glissante (voir note)
LOG_AXES = True    # log-log : recommande pour des montants a queue lourde


def bande_style_reference(df, niveau=None, groupe_col="groupe_largeur",
                          n_max=N_MAX, lissage=LISSAGE, log_axes=LOG_AXES,
                          random_state=42, out=None):
    d = df if (niveau in (None, "TOUS")) else df[df[groupe_col] == niveau]
    d = d.copy()
    if len(d) < 5:
        print(f"Niveau '{niveau}' : {len(d)} observations, trop peu.")
        return
    if len(d) > n_max:
        d = d.sample(n_max, random_state=random_state)

    d = d.sort_values("y_pred").reset_index(drop=True)
    xs = d["y_pred"].values.astype(float)
    lo = d["borne_basse"].values.astype(float).copy()
    hi = d["borne_haute"].values.astype(float).copy()
    obs = d["y_obs"].values.astype(float)
    dedans = d["dans_intervalle"].values.astype(bool)

    # Lissage optionnel : rend l'enveloppe plus reguliere, mais un point rouge
    # peut alors sembler tomber dans la bande. Le statut vert/rouge reste
    # calcule sur les bornes REELLES, jamais sur les bornes lissees.
    if lissage and lissage >= 3:
        w = int(lissage)
        lo = pd.Series(lo).rolling(w, center=True, min_periods=1).median().values
        hi = pd.Series(hi).rolling(w, center=True, min_periods=1).median().values

    log_ok = log_axes and (xs > 0).all() and (obs > 0).all()
    if log_ok:
        lo = np.maximum(lo, np.nanmin(obs[obs > 0]) * 1e-2)

    hover = [(f"<b>{' | '.join(str(r[c]) for c in ID_COLS if c in d.columns)}</b><br>"
              f"Observe   : {r['y_obs']:,.0f}<br>"
              f"Predit    : {r['y_pred']:,.0f}<br>"
              f"Intervalle: [{r['borne_basse']:,.0f} ; {r['borne_haute']:,.0f}]")
             for _, r in d.iterrows()]

    fig = go.Figure()

    # Bande conforme
    fig.add_trace(go.Scatter(x=xs, y=lo, mode="lines",
                              line=dict(color="#7b7fd6", width=1.3, dash="dot"),
                              showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=hi, mode="lines",
                              line=dict(color="#7b7fd6", width=1.3, dash="dot"),
                              fill="tonexty", fillcolor="rgba(150,150,235,0.35)",
                              name=f"Prediction Interval ({100*(1-ALPHA):.0f} %)",
                              hoverinfo="skip"))

    # Droite de prediction : y = x, exactement comme la droite noire de reference
    fig.add_trace(go.Scatter(x=[xs.min(), xs.max()], y=[xs.min(), xs.max()],
                              mode="lines", line=dict(color="black", width=2),
                              name="Prediction", hoverinfo="skip"))

    # Observations
    fig.add_trace(go.Scatter(
        x=xs[dedans], y=obs[dedans], mode="markers",
        marker=dict(size=6, color="#1b7f2c"),
        name=f"Observation (inside PI) — {dedans.sum()}",
        text=[h for h, k in zip(hover, dedans) if k],
        hovertemplate="%{text}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=xs[~dedans], y=obs[~dedans], mode="markers",
        marker=dict(size=7, color="#e02020"),
        name=f"Observation (outside PI) — {(~dedans).sum()}",
        text=[h for h, k in zip(hover, dedans) if not k],
        hovertemplate="%{text}<extra></extra>"))

    titre = f"Intervalle de Conformal Prediction — {len(d)} observations"
    if niveau not in (None, "TOUS"):
        titre += f"  |  niveau : {niveau}"
    fig.update_layout(
        title=titre,
        xaxis=dict(title="Prediction du modele" + ("  (log)" if log_ok else ""),
                   type="log" if log_ok else "linear"),
        yaxis=dict(title=f"{TARGET} observe" + ("  (log)" if log_ok else ""),
                   type="log" if log_ok else "linear"),
        template="plotly_white", height=620, hovermode="closest",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="black", borderwidth=1))

    if out is not None:
        with out:
            clear_output(wait=True)
            fig.show()
    else:
        fig.show()


niveaux = ["TOUS"] + (results_v2["groupe_largeur"].value_counts()
                      .loc[lambda s: s >= 10].index.tolist()
                      if "groupe_largeur" in results_v2.columns else [])

out = widgets.Output()
dd = widgets.Dropdown(options=niveaux, description="Niveau :",
                      layout=widgets.Layout(width="500px"))
dd.observe(lambda ch: bande_style_reference(results_v2, ch["new"], out=out)
          if ch["name"] == "value" else None, names="value")
display(dd, out)
bande_style_reference(results_v2, "TOUS", out=out)




Deuxièmeme

# ================================================================
# CONFORMAL PREDICTION PAR CATEGORIE — bornes individuelles exactes
#
# X : categorie (Lob / Risk / Partner / ...) — 5 modalites principales
# Y : RBNS_eop observe
#
# Dans chaque categorie, les observations sont ordonnees par prediction
# croissante et etalees sur la largeur du slot. Le ruban trace est fait
# des VRAIES bornes CP individuelles : aucune mediane, aucun lissage.
# Un point rouge est donc necessairement hors du ruban a son abscisse.
# ================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output

OBS_COL    = "y_obs"
PRED_COL   = "y_pred"
LOWER_COL  = "borne_basse"
UPPER_COL  = "borne_haute"
INSIDE_COL = "dans_intervalle"

N_CATEGORIES  = 5
N_POINTS_CAT  = 35
DEMI_LARGEUR  = 0.38     # demi-largeur d'un slot de categorie
POINT_SIZE    = 5


def plot_cp_par_categorie(df, category_col, n_categories=N_CATEGORIES,
                          n_points=N_POINTS_CAT, log_y=True,
                          random_state=42, out=None):
    d = df.dropna(subset=[category_col, OBS_COL, PRED_COL,
                          LOWER_COL, UPPER_COL]).copy()

    cats = d[category_col].value_counts().head(n_categories).index.tolist()
    if not cats:
        print(f"Aucune modalite exploitable dans '{category_col}'.")
        return

    fig = go.Figure()
    premier = True
    xticks, xlabels, annotations = [], [], []

    for i, cat in enumerate(cats):
        sub_all = d[d[category_col] == cat]
        couverture = sub_all[INSIDE_COL].mean()          # sur TOUTES les obs

        sub = sub_all.sample(min(n_points, len(sub_all)), random_state=random_state)
        sub = sub.sort_values(PRED_COL).reset_index(drop=True)
        if len(sub) < 2:
            continue

        # Etalement des observations sur la largeur du slot
        xs = i + np.linspace(-DEMI_LARGEUR, DEMI_LARGEUR, len(sub))
        lo = sub[LOWER_COL].values.astype(float)
        hi = sub[UPPER_COL].values.astype(float)
        pred = sub[PRED_COL].values.astype(float)
        obs = sub[OBS_COL].values.astype(float)
        dedans = sub[INSIDE_COL].values.astype(bool)

        hover = [(f"<b>{cat}</b><br>"
                  f"RBNS observe : {o:,.0f}<br>"
                  f"Prediction   : {p:,.0f}<br>"
                  f"Intervalle CP: [{l:,.0f} ; {h:,.0f}]")
                 for o, p, l, h in zip(obs, pred, lo, hi)]

        # Ruban : vraies bornes individuelles
        fig.add_trace(go.Scatter(x=xs, y=lo, mode="lines",
                                 line=dict(color="#355CDE", width=2),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=hi, mode="lines",
                                 line=dict(color="#355CDE", width=2),
                                 fill="tonexty", fillcolor="rgba(70,110,230,0.20)",
                                 name=f"Intervalle CP ({100*(1-ALPHA):.0f} %)",
                                 showlegend=premier, hoverinfo="skip"))

        # Prediction
        fig.add_trace(go.Scatter(x=xs, y=pred, mode="lines",
                                 line=dict(color="black", width=2),
                                 name="Prediction", showlegend=premier,
                                 hoverinfo="skip"))

        # Observations
        fig.add_trace(go.Scatter(
            x=xs[dedans], y=obs[dedans], mode="markers",
            marker=dict(size=POINT_SIZE, color="#1769E0", opacity=0.85),
            name="Observation dans le CP", showlegend=premier,
            text=[h for h, k in zip(hover, dedans) if k],
            hovertemplate="%{text}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=xs[~dedans], y=obs[~dedans], mode="markers",
            marker=dict(size=POINT_SIZE + 2, color="#E53935", opacity=0.95),
            name="Observation hors CP", showlegend=premier,
            text=[h for h, k in zip(hover, dedans) if not k],
            hovertemplate="%{text}<extra></extra>"))

        premier = False
        xticks.append(i)
        xlabels.append(str(cat))
        annotations.append(dict(x=i, y=1.0, yref="paper", yanchor="bottom",
                                text=f"couverture {100*couverture:.1f} %<br>"
                                     f"<sub>n = {len(sub_all)}</sub>",
                                showarrow=False, font=dict(size=10)))

    # Separateurs entre categories
    for i in range(len(xticks) - 1):
        fig.add_vline(x=xticks[i] + 0.5, line=dict(color="lightgray", width=1))

    obs_all = d[d[category_col].isin(cats)][OBS_COL]
    log_ok = log_y and (obs_all > 0).all()

    fig.update_layout(
        title=dict(text=f"Conformal Prediction par {category_col}"
                        f"<br><sup>Bleu : observation couverte | Rouge : non couverte "
                        f"| bornes CP individuelles, sans agregation</sup>", x=0.5),
        xaxis=dict(title=category_col, tickmode="array",
                   tickvals=xticks, ticktext=xlabels,
                   range=[-0.6, len(xticks) - 0.4], showgrid=False),
        yaxis=dict(title=OBS_COL + ("  (log)" if log_ok else ""),
                   type="log" if log_ok else "linear",
                   gridcolor="rgba(0,0,0,0.08)"),
        template="plotly_white", height=680, hovermode="closest",
        annotations=annotations,
        legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="center", x=0.5),
        margin=dict(t=140))

    if out is not None:
        with out:
            clear_output(wait=True)
            fig.show()
    else:
        fig.show()


# --- Selecteur interactif de la colonne de categorie ---------------------
cols_dispo = [c for c in ID_COLS if c in results_v2.columns]

out = widgets.Output()
dd = widgets.Dropdown(options=cols_dispo, value=cols_dispo[0],
                      description="Categorie :",
                      layout=widgets.Layout(width="450px"))
dd.observe(lambda ch: plot_cp_par_categorie(results_v2, ch["new"], out=out)
           if ch["name"] == "value" else None, names="value")
display(dd, out)
plot_cp_par_categorie(results_v2, cols_dispo[0], out=out)




Troisème Chat


# ================================================================
# CONFORMAL PREDICTION — VISUALISATION PAR CATEGORIE
#
# X      : LOB / Risk / Partner / autre ID_COL
# Y      : RBNS_eop observe
#
# Bande : vraie borne basse / vraie borne haute CP
# Ligne : prediction
#
# Bleu  : observation DANS l'intervalle CP
# Rouge : observation HORS de l'intervalle CP
#
# Objectif :
#   - peu de categories (ex. 5)
#   - bornes larges et lisibles
#   - prediction au milieu
#   - observations visibles
#   - graphique interactif
# ================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output


# ================================================================
# PARAMETRES
# ================================================================

TARGET_COL = "RBNS_eop"

PRED_COL = "y_pred"

LOWER_COL = "borne_basse"

UPPER_COL = "borne_haute"

INSIDE_COL = "dans_intervalle"

# Nombre maximum de catégories affichées
N_CATEGORIES = 5

# Nombre maximum de points par catégorie
N_POINTS_PAR_CATEGORIE = 35

# Taille des points
POINT_SIZE = 5


# ================================================================
# FONCTION PRINCIPALE
# ================================================================

def plot_cp_by_category(
    df,
    category_col,
    n_categories=5,
    n_points_per_category=35,
    random_state=42
):

    d = df.copy()

    # ------------------------------------------------------------
    # Vérification des colonnes
    # ------------------------------------------------------------

    required = [
        category_col,
        TARGET_COL,
        PRED_COL,
        LOWER_COL,
        UPPER_COL,
        INSIDE_COL
    ]

    missing = [c for c in required if c not in d.columns]

    if missing:
        raise ValueError(
            f"Colonnes absentes du DataFrame : {missing}"
        )

    # ------------------------------------------------------------
    # Nettoyage
    # ------------------------------------------------------------

    d = d.dropna(
        subset=[
            category_col,
            TARGET_COL,
            PRED_COL,
            LOWER_COL,
            UPPER_COL
        ]
    ).copy()

    # ------------------------------------------------------------
    # Sélection des catégories les plus représentées
    # ------------------------------------------------------------

    top_categories = (
        d[category_col]
        .value_counts()
        .head(n_categories)
        .index
        .tolist()
    )

    d = d[d[category_col].isin(top_categories)].copy()

    # ------------------------------------------------------------
    # Ordre des catégories
    # ------------------------------------------------------------

    category_order = top_categories

    d[category_col] = pd.Categorical(
        d[category_col],
        categories=category_order,
        ordered=True
    )

    # ------------------------------------------------------------
    # Sous-échantillonnage pour la lisibilité
    #
    # IMPORTANT :
    # on ne modifie PAS les bornes CP.
    # On réduit seulement le nombre de points affichés.
    # ------------------------------------------------------------

    pieces = []

    for cat in category_order:

        tmp = d[d[category_col] == cat].copy()

        if len(tmp) > n_points_per_category:
            tmp = tmp.sample(
                n_points_per_category,
                random_state=random_state
            )

        pieces.append(tmp)

    d_plot = pd.concat(pieces, ignore_index=True)

    # ------------------------------------------------------------
    # Calcul des valeurs centrales de chaque catégorie
    #
    # Pour la représentation :
    # prediction = médiane des prédictions
    # borne basse = médiane des bornes basses
    # borne haute = médiane des bornes hautes
    #
    # ATTENTION :
    # les couleurs des observations restent basées
    # sur le vrai statut CP individuel.
    # ------------------------------------------------------------

    summary = (
        d.groupby(category_col)
        .agg(
            prediction=(PRED_COL, "median"),
            borne_basse=(LOWER_COL, "median"),
            borne_haute=(UPPER_COL, "median"),
            n=(TARGET_COL, "size")
        )
        .reindex(category_order)
        .reset_index()
    )

    # ------------------------------------------------------------
    # Position numérique des catégories
    # ------------------------------------------------------------

    summary["x"] = np.arange(len(summary))

    # Mapping catégorie -> x
    mapping = {
        cat: i
        for i, cat in enumerate(category_order)
    }

    d_plot["x"] = d_plot[category_col].map(mapping).astype(float)

    # Petit jitter horizontal
    rng = np.random.default_rng(random_state)

    d_plot["x_jitter"] = (
        d_plot["x"]
        + rng.uniform(
            -0.18,
            0.18,
            size=len(d_plot)
        )
    )

    # ============================================================
    # FIGURE
    # ============================================================

    fig = go.Figure()

    # ------------------------------------------------------------
    # 1. BANDES CP
    #
    # Une bande par catégorie.
    # ------------------------------------------------------------

    for _, r in summary.iterrows():

        x = r["x"]

        # largeur horizontale de la bande
        x_left = x - 0.32
        x_right = x + 0.32

        # borne inférieure
        fig.add_trace(
            go.Scatter(
                x=[x_left, x_right],
                y=[
                    r["borne_basse"],
                    r["borne_basse"]
                ],
                mode="lines",
                line=dict(
                    color="#355CDE",
                    width=3
                ),
                showlegend=False,
                hoverinfo="skip"
            )
        )

        # borne supérieure
        fig.add_trace(
            go.Scatter(
                x=[x_left, x_right],
                y=[
                    r["borne_haute"],
                    r["borne_haute"]
                ],
                mode="lines",
                line=dict(
                    color="#355CDE",
                    width=3
                ),
                fill="tonexty",
                fillcolor="rgba(70,110,230,0.18)",
                showlegend=False,
                hoverinfo="skip"
            )
        )

    # ------------------------------------------------------------
    # 2. LIGNE DE PREDICTION
    # ------------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=summary["x"],
            y=summary["prediction"],
            mode="lines+markers",
            line=dict(
                color="black",
                width=3
            ),
            marker=dict(
                size=7,
                color="black"
            ),
            name="Prediction",
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Prediction : %{y:,.0f}<br>"
                "Borne basse : %{customdata[1]:,.0f}<br>"
                "Borne haute : %{customdata[2]:,.0f}"
                "<extra></extra>"
            ),
            customdata=np.column_stack([
                summary[category_col].astype(str),
                summary["borne_basse"],
                summary["borne_haute"]
            ])
        )
    )

    # ------------------------------------------------------------
    # 3. OBSERVATIONS DANS L'INTERVALLE
    # ------------------------------------------------------------

    inside = d_plot[d_plot[INSIDE_COL] == True]

    fig.add_trace(
        go.Scatter(
            x=inside["x_jitter"],
            y=inside[TARGET_COL],
            mode="markers",
            marker=dict(
                size=POINT_SIZE,
                color="#1769E0",
                opacity=0.75
            ),
            name="Observation dans le CP",
            customdata=np.column_stack([
                inside[category_col].astype(str),
                inside[PRED_COL],
                inside[LOWER_COL],
                inside[UPPER_COL]
            ]),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "RBNS_eop observé : %{y:,.0f}<br>"
                "Prediction : %{customdata[1]:,.0f}<br>"
                "CP : [%{customdata[2]:,.0f} ; "
                "%{customdata[3]:,.0f}]"
                "<extra></extra>"
            )
        )
    )

    # ------------------------------------------------------------
    # 4. OBSERVATIONS HORS INTERVALLE
    # ------------------------------------------------------------

    outside = d_plot[d_plot[INSIDE_COL] == False]

    fig.add_trace(
        go.Scatter(
            x=outside["x_jitter"],
            y=outside[TARGET_COL],
            mode="markers",
            marker=dict(
                size=POINT_SIZE + 1,
                color="#E53935",
                opacity=0.85
            ),
            name="Observation hors CP",
            customdata=np.column_stack([
                outside[category_col].astype(str),
                outside[PRED_COL],
                outside[LOWER_COL],
                outside[UPPER_COL]
            ]),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "RBNS_eop observé : %{y:,.0f}<br>"
                "Prediction : %{customdata[1]:,.0f}<br>"
                "CP : [%{customdata[2]:,.0f} ; "
                "%{customdata[3]:,.0f}]"
                "<extra></extra>"
            )
        )
    )

    # ============================================================
    # MISE EN FORME
    # ============================================================

    fig.update_layout(

        title=dict(
            text=(
                f"Conformal Prediction — {category_col}<br>"
                "<sup>"
                "Bleu : observations dans l'intervalle | "
                "Rouge : observations hors intervalle"
                "</sup>"
            ),
            x=0.5
        ),

        xaxis=dict(
            title=category_col,
            tickmode="array",
            tickvals=summary["x"],
            ticktext=summary[category_col].astype(str),
            range=[
                -0.55,
                len(summary) - 0.45
            ],
            showgrid=False
        ),

        yaxis=dict(
            title=TARGET_COL,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)"
        ),

        template="plotly_white",

        height=700,

        hovermode="closest",

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),

        margin=dict(
            l=80,
            r=40,
            t=110,
            b=80
        )
    )

    fig.show()

    return d_plot, summary




d_plot, summary = plot_cp_by_category(
    results_v2,
    category_col="LOB",
    n_categories=5,
    n_points_per_category=35
)
