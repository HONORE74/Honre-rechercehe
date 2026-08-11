# ================================================================
# CONFORMAL PREDICTION — VISUALISATION CONTINUE DES INTERVALLES CP
#
# DataFrame : anomalies_prio
#
# AXE X :
#   Rang de priorité des anomalies
#   (score de priorisation décroissant)
#
# AXE Y :
#   RBNS_eop observé
#
# BANDE :
#   borne_basse  -> borne_haute
#   propre à chaque observation
#
# LIGNE NOIRE :
#   prédiction
#
# POINTS :
#   bleu  = observation dans son intervalle CP
#   rouge = observation hors de son intervalle CP
#
# IMPORTANT :
#   - aucun max/min global
#   - aucune médiane
#   - aucune régression artificielle
#   - aucun intervalle constant
# ================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ================================================================
# PARAMETRES
# ================================================================

TARGET_COL = "RBNS_eop"
SCORE_COL = "score_priorisation"
PRED_COL = "y_pred"
LOWER_COL = "borne_basse"
UPPER_COL = "borne_haute"

# Nombre d'anomalies affichées
N_ANOMALIES = 40

# True :
# score élevé = anomalie plus prioritaire
SCORE_DESCENDING = True


# ================================================================
# FONCTION
# ================================================================

def plot_cp_priorisation(
    df,
    score_col=SCORE_COL,
    target_col=TARGET_COL,
    pred_col=PRED_COL,
    lower_col=LOWER_COL,
    upper_col=UPPER_COL,
    n_anomalies=N_ANOMALIES,
    descending=SCORE_DESCENDING
):

    data = df.copy()

    # ============================================================
    # 1. VERIFICATION DES COLONNES
    # ============================================================

    required = [
        score_col,
        target_col,
        pred_col,
        lower_col,
        upper_col
    ]

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:

        raise ValueError(
            "\nLes colonnes suivantes sont absentes :\n"
            + "\n".join(f"  • {col}" for col in missing)
            + "\n\nColonnes disponibles :\n"
            + "\n".join(f"  • {col}" for col in data.columns)
        )

    # ============================================================
    # 2. NETTOYAGE
    # ============================================================

    data = data.dropna(
        subset=required
    ).copy()

    # ============================================================
    # 3. CONVERSION NUMERIQUE
    # ============================================================

    for col in required:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.dropna(
        subset=required
    ).copy()

    # ============================================================
    # 4. TRI PAR SCORE DE PRIORISATION
    # ============================================================

    data = data.sort_values(
        by=score_col,
        ascending=not descending
    ).copy()

    # ============================================================
    # 5. SELECTION DES ANOMALIES
    # ============================================================

    if n_anomalies is not None:

        data = data.head(
            n_anomalies
        ).copy()

    # ============================================================
    # 6. NOUVEAU RANG POUR L'AXE X
    # ============================================================

    data = data.reset_index(drop=False)

    data["rang_priorite"] = np.arange(
        1,
        len(data) + 1
    )

    x = data["rang_priorite"].values

    # ============================================================
    # 7. STATUT DE CHAQUE OBSERVATION
    # ============================================================

    data["dans_cp"] = (
        (data[target_col] >= data[lower_col])
        &
        (data[target_col] <= data[upper_col])
    )

    # ============================================================
    # 8. VALEURS CP
    # ============================================================

    lower = data[lower_col].values
    upper = data[upper_col].values
    prediction = data[pred_col].values

    # ============================================================
    # 9. GRILLE FINE
    #
    # On ne crée PAS de nouvelles bornes.
    #
    # On interpole seulement entre les observations existantes
    # afin d'obtenir visuellement une bande continue.
    # ============================================================

    if len(data) > 1:

        x_fine = np.linspace(
            x.min(),
            x.max(),
            max(500, len(data) * 20)
        )

        lower_fine = np.interp(
            x_fine,
            x,
            lower
        )

        upper_fine = np.interp(
            x_fine,
            x,
            upper
        )

        prediction_fine = np.interp(
            x_fine,
            x,
            prediction
        )

    else:

        x_fine = x
        lower_fine = lower
        upper_fine = upper
        prediction_fine = prediction

    # ============================================================
    # 10. FIGURE
    # ============================================================

    fig = go.Figure()

    # ============================================================
    # 11. BORNE HAUTE
    # ============================================================

    fig.add_trace(
        go.Scatter(
            x=x_fine,
            y=upper_fine,
            mode="lines",
            line=dict(
                color="#4169E1",
                width=2.5
            ),
            name="Borne haute CP",
            hoverinfo="skip"
        )
    )

    # ============================================================
    # 12. BORNE BASSE + REMPLISSAGE
    # ============================================================

    fig.add_trace(
        go.Scatter(
            x=x_fine,
            y=lower_fine,
            mode="lines",
            line=dict(
                color="#4169E1",
                width=2.5
            ),
            fill="tonexty",
            fillcolor="rgba(65,105,225,0.20)",
            name="Intervalle CP",
            hoverinfo="skip"
        )
    )

    # ============================================================
    # 13. LIGNE DE PREDICTION
    # ============================================================

    fig.add_trace(
        go.Scatter(
            x=x_fine,
            y=prediction_fine,
            mode="lines",
            line=dict(
                color="black",
                width=3
            ),
            name="Prédiction",
            hoverinfo="skip"
        )
    )

    # ============================================================
    # 14. OBSERVATIONS DANS LE CP
    # ============================================================

    inside = data[
        data["dans_cp"]
    ].copy()

    fig.add_trace(
        go.Scatter(
            x=inside["rang_priorite"],
            y=inside[target_col],
            mode="markers",

            marker=dict(
                size=6,
                color="#087F23",
                opacity=0.90,
                line=dict(
                    color="white",
                    width=0.6
                )
            ),

            name="Observation dans le CP",

            customdata=inside[
                [
                    score_col,
                    pred_col,
                    lower_col,
                    upper_col
                ]
            ].values,

            hovertemplate=(
                "<b>Observation dans le CP</b>"
                "<br><br>"
                "<b>Rang :</b> %{x}"
                "<br>"
                "<b>Score de priorisation :</b> "
                "%{customdata[0]:,.3f}"
                "<br>"
                "<b>RBNS_eop observé :</b> "
                "%{y:,.0f}"
                "<br>"
                "<b>Prédiction :</b> "
                "%{customdata[1]:,.0f}"
                "<br>"
                "<b>Borne basse :</b> "
                "%{customdata[2]:,.0f}"
                "<br>"
                "<b>Borne haute :</b> "
                "%{customdata[3]:,.0f}"
                "<extra></extra>"
            )
        )
    )

    # ============================================================
    # 15. OBSERVATIONS HORS CP
    # ============================================================

    outside = data[
        ~data["dans_cp"]
    ].copy()

    fig.add_trace(
        go.Scatter(
            x=outside["rang_priorite"],
            y=outside[target_col],
            mode="markers",

            marker=dict(
                size=7,
                color="#E31B23",
                opacity=0.95,
                line=dict(
                    color="white",
                    width=0.7
                )
            ),

            name="Observation hors CP",

            customdata=outside[
                [
                    score_col,
                    pred_col,
                    lower_col,
                    upper_col
                ]
            ].values,

            hovertemplate=(
                "<b>⚠ Observation hors CP</b>"
                "<br><br>"
                "<b>Rang :</b> %{x}"
                "<br>"
                "<b>Score de priorisation :</b> "
                "%{customdata[0]:,.3f}"
                "<br>"
                "<b>RBNS_eop observé :</b> "
                "%{y:,.0f}"
                "<br>"
                "<b>Prédiction :</b> "
                "%{customdata[1]:,.0f}"
                "<br>"
                "<b>Borne basse :</b> "
                "%{customdata[2]:,.0f}"
                "<br>"
                "<b>Borne haute :</b> "
                "%{customdata[3]:,.0f}"
                "<extra></extra>"
            )
        )
    )

    # ============================================================
    # 16. AXE X
    #
    # On affiche le SCORE et non seulement le rang.
    # ============================================================

    # Maximum 10 labels pour garder l'axe lisible

    n_ticks = min(
        10,
        len(data)
    )

    if n_ticks > 0:

        tick_indices = np.linspace(
            0,
            len(data) - 1,
            n_ticks,
            dtype=int
        )

        tick_indices = np.unique(
            tick_indices
        )

        tickvals = (
            data.iloc[tick_indices]["rang_priorite"]
            .tolist()
        )

        ticktext = [
            f"{score:.2f}"
            for score in
            data.iloc[tick_indices][score_col]
        ]

    else:

        tickvals = []
        ticktext = []

    # ============================================================
    # 17. MISE EN FORME
    # ============================================================

    fig.update_layout(

        template="plotly_white",

        title=dict(
            text=(
                "<b>Conformal Prediction — "
                "Priorisation des anomalies</b>"
                "<br>"
                "<sup>"
                "Bande bleue : intervalle de prédiction conforme | "
                "Ligne noire : prédiction"
                "</sup>"
            ),
            x=0.5,
            xanchor="center"
        ),

        xaxis=dict(
            title=(
                "Score de priorisation "
                "(du plus prioritaire au moins prioritaire)"
            ),

            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,

            showgrid=False,
            zeroline=False
        ),

        yaxis=dict(
            title="RBNS_eop observé",

            showgrid=True,

            gridcolor="rgba(0,0,0,0.08)",

            zeroline=False
        ),

        height=720,

        hovermode="closest",

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),

        margin=dict(
            l=90,
            r=40,
            t=120,
            b=100
        )
    )

    # ============================================================
    # 18. AFFICHAGE
    # ============================================================

    fig.show()

    # ============================================================
    # 19. RETOUR DES DONNEES
    # ============================================================

    return data, fig

data_cp_plot, fig_cp = plot_cp_priorisation(
    anomalies_prio,
    n_anomalies=40,
    descending=True
)
