# ================================================================
# REPARTITION DES ANOMALIES (uniquement les anomalies, rien d'autre)
#   Chaque segment = une part des N anomalies totales
#   Pas de taux, pas de comparaison au reste de la base
#   Ex: sur 312 anomalies, 40% sont en Lob=Auto, dont 60% en Risk=X
# ================================================================

import plotly.express as px


def sunburst_repartition_anomalies(anomalies_df, chemin=None):
    chemin = chemin or [c for c in ["Lob", "Risk", "Partner"]
                        if c in anomalies_df.columns][:2]

    dd = anomalies_df.dropna(subset=chemin).copy()
    for c in chemin:
        dd[c] = dd[c].astype(str)

    n_total = len(dd)

    fig = px.sunburst(dd, path=chemin)   # values = nombre de lignes par defaut
    fig.update_traces(
        texttemplate="%{label}<br>%{percentRoot:.1%}",
        hovertemplate="<b>%{label}</b><br>"
                      "%{value} anomalies<br>"
                      f"soit %{{percentRoot}} des {n_total} anomalies au total<br>"
                      "%{percentParent} de sa branche parente"
                      "<extra></extra>",
        insidetextorientation="radial")
    fig.update_layout(
        title=f"Repartition des {n_total} anomalies"
              f"<br><sup>{' -> '.join(chemin)}  |  "
              "la taille d'un segment = sa part parmi les anomalies</sup>",
        template="plotly_white", height=700)
    afficher(fig)
    return dd


sunburst_repartition_anomalies(anomalies_prio)

























def treemap_repartition_anomalies(anomalies_df, chemin=None):
    chemin = chemin or [c for c in ["Lob", "Risk", "Partner"]
                        if c in anomalies_df.columns][:2]
    dd = anomalies_df.dropna(subset=chemin).copy()
    for c in chemin:
        dd[c] = dd[c].astype(str)
    n_total = len(dd)

    fig = px.treemap(dd, path=chemin)
    fig.update_traces(
        texttemplate="%{label}<br>%{value} (%{percentRoot:.1%})",
        hovertemplate="<b>%{label}</b><br>%{value} anomalies<br>"
                      f"%{{percentRoot}} des {n_total} au total<extra></extra>")
    fig.update_layout(
        title=f"Repartition des {n_total} anomalies (treemap)",
        template="plotly_white", height=600)
    afficher(fig)


treemap_repartition_anomalies(anomalies_prio)


























