# ================================================================
# SUNBURST — CONCENTRATION DES ANOMALIES
#   Taille  : nombre d'anomalies (les branches sans anomalie disparaissent)
#   Couleur : taux d'anomalie (calcule sur TOUTES les observations)
#   Survol  : GWP total de la branche
# ================================================================

import numpy as np
import pandas as pd
import plotly.express as px


def sunburst_concentration(expl, chemin=None, top_n_par_niveau=8):
    chemin = chemin or [c for c in ["Lob", "Partner", "Risk", "Activity"]
                        if c in expl.columns][:3]

    d = expl.dropna(subset=chemin).copy()
    for c in chemin:
        d[c] = d[c].astype(str)
        top = d[c].value_counts().head(top_n_par_niveau).index
        d[c] = np.where(d[c].isin(top), d[c], "Autres")

    agg = {"n_total": ("dans_intervalle", "size"),
          "n_anomalies": ("est_anomalie", "sum")}
    if GWP_COL in d.columns:
        agg["gwp_total"] = (GWP_COL, "sum")

    stats = d.groupby(chemin, observed=True).agg(**agg).reset_index()
    stats["taux_anomalie"] = stats["n_anomalies"] / stats["n_total"]
    stats = stats[stats["n_anomalies"] > 0].copy()   # ne garder que ce qui a des anomalies

    if "gwp_total" not in stats.columns:
        stats["gwp_total"] = np.nan

    fig = px.sunburst(
        stats, path=chemin, values="n_anomalies",
        color="taux_anomalie", color_continuous_scale="RdYlGn_r",
        range_color=[0, min(1.0, max(0.3, stats["taux_anomalie"].quantile(0.9) * 1.2))],
        custom_data=["n_total", "gwp_total", "taux_anomalie"])
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>"
                      "Anomalies : %{value} / %{customdata[0]} obs  (%{customdata[2]:.1%})<br>"
                      "GWP total : %{customdata[1]:,.0f}<extra></extra>",
        insidetextorientation="radial")
    fig.update_layout(
        title="Concentration des anomalies"
              "<br><sup>Taille = nombre d'anomalies | Couleur = taux d'anomalie | "
              "seules les branches ayant au moins une anomalie sont affichees</sup>",
        template="plotly_white", height=720,
        coloraxis_colorbar=dict(title="Taux<br>anomalie", tickformat=".0%"))
    afficher(fig)
    return stats


stats_concentration = sunburst_concentration(expl, chemin=["Lob", "Partner", "Risk"])












# ================================================================
# QUADRANT — TAUX D'ANOMALIE vs VOLUME (GWP)
#   Un point par modalite d'une categorie (Activity, Lob, Risk...)
#   Taille du point = nombre d'anomalies
#   Quadrant haut-droit = taux eleve ET gros volume -> le plus preoccupant
#   Quadrant haut-gauche = taux eleve mais petit volume -> a surveiller,
#                           potentiellement du bruit (peu d'observations)
# ================================================================

import plotly.graph_objects as go


def stats_par_categorie(expl, categorie):
    d = expl.dropna(subset=[categorie]).copy()
    d[categorie] = d[categorie].astype(str)
    agg = {"n_total": ("dans_intervalle", "size"),
          "n_anomalies": ("est_anomalie", "sum")}
    if GWP_COL in d.columns:
        agg["gwp_total"] = (GWP_COL, "sum")
    g = d.groupby(categorie, observed=True).agg(**agg).reset_index()
    g["taux_anomalie"] = g["n_anomalies"] / g["n_total"]
    return g


def quadrant_taux_vs_gwp(expl, categorie, n_min_obs=1):
    g = stats_par_categorie(expl, categorie)
    g = g[g["n_total"] >= n_min_obs].copy()
    if "gwp_total" not in g.columns or g["gwp_total"].isna().all():
        print(f"Colonne {GWP_COL} indisponible.")
        return g

    med_taux = g["taux_anomalie"].median()
    med_gwp = g["gwp_total"].median()

    fig = go.Figure()
    fig.add_vline(x=med_taux, line=dict(color="gray", dash="dash", width=1))
    fig.add_hline(y=med_gwp, line=dict(color="gray", dash="dash", width=1))

    fig.add_trace(go.Scatter(
        x=g["taux_anomalie"], y=g["gwp_total"], mode="markers+text",
        marker=dict(size=8 + 24 * g["n_anomalies"].rank(pct=True).fillna(0),
                   color=g["taux_anomalie"], colorscale="RdYlGn_r",
                   showscale=True, colorbar=dict(title="Taux", tickformat=".0%"),
                   line=dict(width=0.6, color="white")),
        text=g[categorie], textposition="top center", textfont=dict(size=9),
        customdata=np.column_stack([g["n_anomalies"], g["n_total"]]),
        hovertemplate=f"<b>%{{text}}</b> ({categorie})<br>"
                      "Taux anomalie : %{x:.1%}<br>GWP total : %{y:,.0f}<br>"
                      "Anomalies : %{customdata[0]:.0f} / %{customdata[1]:.0f}<extra></extra>"))

    fig.add_annotation(x=0.98, y=0.98, xref="paper", yref="paper", xanchor="right", yanchor="top",
                       text="<b>CRITIQUE</b><br>taux eleve + gros volume", showarrow=False,
                       font=dict(size=10, color="#b71c1c"), align="right",
                       bgcolor="rgba(255,255,255,0.8)")
    fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper", xanchor="left", yanchor="top",
                       text="<b>A surveiller</b><br>taux eleve, petit volume<br>"
                            "(verifier n avant conclusion)",
                       showarrow=False, font=dict(size=10, color="#e65100"), align="left",
                       bgcolor="rgba(255,255,255,0.8)")

    fig.update_layout(
        title=f"Taux d'anomalie vs Volume (GWP) — par {categorie}"
              "<br><sup>Taille = nombre d'anomalies | "
              "Quadrant haut-droit = le plus preoccupant financierement</sup>",
        xaxis=dict(title="Taux d'anomalie", tickformat=".0%"),
        yaxis=dict(title=f"{GWP_COL} total (volume)", type="log"),
        template="plotly_white", height=650, hovermode="closest")
    afficher(fig)
    return g


stats_activity = quadrant_taux_vs_gwp(expl, "Activity")


















# ================================================================
# BLOC 1 — STATISTIQUES AVANCEES PAR CATEGORIE
#   Pour chaque modalite : taux, intervalle de confiance de Wilson,
#   test binomial contre ALPHA, GWP total et GWP porte par les anomalies
# ================================================================

import numpy as np
import pandas as pd
from scipy.stats import binom


def _wilson(k, n, conf=0.95):
    """Intervalle de confiance de Wilson pour une proportion.
    Bien plus fiable que l'approximation normale sur petits effectifs."""
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - conf) / 2)
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    marge = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return np.clip(centre - marge, 0, 1), np.clip(centre + marge, 0, 1)


def _bh(pvals, q=0.05):
    p = np.asarray(pvals, float)
    m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    ordre = np.argsort(p)
    passe = p[ordre] <= q * np.arange(1, m + 1) / m
    k = int(np.max(np.where(passe)[0]) + 1) if passe.any() else 0
    rej = np.zeros(m, bool)
    if k:
        rej[ordre[:k]] = True
    return rej


def stats_avancees(expl, categorie, n_min=3, q_fdr=0.05):
    d = expl.dropna(subset=[categorie]).copy()
    d[categorie] = d[categorie].astype(str)

    agg = {"n_total": ("dans_intervalle", "size"),
           "n_anomalies": ("est_anomalie", "sum")}
    if GWP_COL in d.columns:
        agg["gwp_total"] = (GWP_COL, "sum")

    g = d.groupby(categorie, observed=True).agg(**agg).reset_index()

    if GWP_COL in d.columns:
        risque = (d[d["est_anomalie"] == 1]
                  .groupby(categorie, observed=True)[GWP_COL].sum()
                  .rename("gwp_a_risque").reset_index())
        g = g.merge(risque, on=categorie, how="left")
        g["gwp_a_risque"] = g["gwp_a_risque"].fillna(0.0)
        g["part_gwp_risque"] = g["gwp_a_risque"] / g["gwp_total"].replace(0, np.nan)
    else:
        g["gwp_total"] = np.nan
        g["gwp_a_risque"] = np.nan
        g["part_gwp_risque"] = np.nan

    g = g[g["n_total"] >= n_min].copy()
    g["taux"] = g["n_anomalies"] / g["n_total"]
    g["ic_bas"], g["ic_haut"] = _wilson(g["n_anomalies"], g["n_total"])

    # Test binomial : P(observer au moins k anomalies | taux reel = ALPHA)
    g["p_value"] = binom.sf(g["n_anomalies"] - 1, g["n_total"], ALPHA)
    g["significatif"] = _bh(g["p_value"].values, q=q_fdr)

    # Materialite : au-dessus de la mediane du GWP a risque
    seuil_mat = g["gwp_a_risque"].median() if g["gwp_a_risque"].notna().any() else np.nan
    g["materiel"] = g["gwp_a_risque"] > seuil_mat

    g["verdict"] = np.select(
        [g["significatif"] & g["materiel"],
         g["significatif"] & ~g["materiel"],
         ~g["significatif"] & g["materiel"]],
        ["PRIORITAIRE", "Reel mais faible enjeu", "Gros enjeu, non prouve"],
        default="Non concluant")

    return g.sort_values(["significatif", "gwp_a_risque"],
                         ascending=[False, False]).reset_index(drop=True)















# ================================================================
# BLOC 2 — MATRICE DE DECISION
#   X : taux d'anomalie, avec intervalle de confiance de Wilson
#   Y : GWP porte par les anomalies (l'enjeu financier reel)
#   Taille : nombre d'observations (= fiabilite de l'estimation)
#   Couleur : verdict croise (significatif x materiel)
# ================================================================

import plotly.graph_objects as go

COULEURS_VERDICT = {
    "PRIORITAIRE":            "#c62828",
    "Reel mais faible enjeu": "#ef6c00",
    "Gros enjeu, non prouve": "#1565c0",
    "Non concluant":          "#9e9e9e"}


def matrice_decision(expl, categorie, n_min=3, max_labels=12):
    g = stats_avancees(expl, categorie, n_min=n_min)
    if g.empty:
        print("Aucune modalite avec assez d'observations.")
        return g

    seuil_mat = g["gwp_a_risque"].median()

    fig = go.Figure()
    fig.add_vline(x=ALPHA, line=dict(color="black", dash="dash", width=2),
                  annotation_text=f"taux attendu par hasard ({100*ALPHA:.0f} %)",
                  annotation_position="top")
    if np.isfinite(seuil_mat) and seuil_mat > 0:
        fig.add_hline(y=seuil_mat, line=dict(color="gray", dash="dot", width=1.4),
                      annotation_text="mediane du GWP a risque", annotation_position="right")

    top_lab = set(g.nlargest(max_labels, "gwp_a_risque")[categorie]) | \
              set(g[g["significatif"]].nlargest(max_labels, "taux")[categorie])

    for verdict, coul in COULEURS_VERDICT.items():
        sub = g[g["verdict"] == verdict]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["taux"], y=np.maximum(sub["gwp_a_risque"], 1),
            error_x=dict(type="data", symmetric=False,
                         array=sub["ic_haut"] - sub["taux"],
                         arrayminus=sub["taux"] - sub["ic_bas"],
                         color=coul, thickness=1.3, width=3),
            mode="markers+text",
            marker=dict(size=8 + 22 * sub["n_total"].rank(pct=True),
                        color=coul, opacity=0.78,
                        line=dict(width=0.8, color="white")),
            text=[t if t in top_lab else "" for t in sub[categorie]],
            textposition="top center", textfont=dict(size=9),
            name=f"{verdict} ({len(sub)})",
            customdata=np.column_stack([sub[categorie], sub["n_anomalies"],
                                        sub["n_total"], sub["gwp_total"],
                                        sub["p_value"], sub["ic_bas"], sub["ic_haut"]]),
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "Taux : %{x:.1%}  (IC 95 % : %{customdata[5]:.1%} – %{customdata[6]:.1%})<br>"
                          "Anomalies : %{customdata[1]:.0f} / %{customdata[2]:.0f} obs<br>"
                          "GWP a risque : %{y:,.0f}<br>"
                          "GWP total : %{customdata[3]:,.0f}<br>"
                          "p-value (test binomial) : %{customdata[4]:.2e}"
                          "<extra></extra>"))

    fig.update_layout(
        title=f"Matrice de decision — {categorie}"
              "<br><sup>Barres horizontales = intervalle de confiance a 95 %. "
              "Une barre qui ne touche pas la ligne noire = taux reellement superieur au hasard. "
              "Taille du point = nombre d'observations</sup>",
        xaxis=dict(title="Taux d'anomalie", tickformat=".0%"),
        yaxis=dict(title=f"{GWP_COL} porte par les anomalies", type="log"),
        template="plotly_white", height=700, hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5))
    afficher(fig)

    print(f"\n{'=' * 92}")
    print(f"VERDICT PAR MODALITE — {categorie}")
    print("=" * 92)
    cols = [categorie, "n_total", "n_anomalies", "taux", "ic_bas", "ic_haut",
            "p_value", "gwp_a_risque", "verdict"]
    aff = g[cols].copy()
    for c in ["taux", "ic_bas", "ic_haut"]:
        aff[c] = (100 * aff[c]).round(1).astype(str) + " %"
    aff["p_value"] = aff["p_value"].map(lambda v: f"{v:.1e}")
    aff["gwp_a_risque"] = aff["gwp_a_risque"].map(lambda v: f"{v:,.0f}")
    print(aff.head(20).to_string(index=False))
    print(f"\nPRIORITAIRES : {(g['verdict'] == 'PRIORITAIRE').sum()} modalite(s)")
    return g


verdicts = matrice_decision(expl, "Activity")



















# ================================================================
# BLOC 3 — FOREST PLOT : le taux est-il vraiment superieur au hasard ?
#   Chaque ligne = une modalite, avec son intervalle de confiance.
#   Si la barre ne croise pas la ligne verticale, le taux est reel.
# ================================================================

def forest_taux(expl, categorie, top_n=20, trier_par="gwp_a_risque"):
    g = stats_avancees(expl, categorie)
    if g.empty:
        return g
    g = g.nlargest(top_n, trier_par).sort_values("taux").reset_index(drop=True)
    y = np.arange(len(g))

    fig = go.Figure()
    fig.add_vline(x=ALPHA, line=dict(color="black", dash="dash", width=2),
                  annotation_text=f"attendu par hasard ({100*ALPHA:.0f} %)")

    for _, r in g.iterrows():
        i = g.index[g[categorie] == r[categorie]][0]
        coul = COULEURS_VERDICT[r["verdict"]]
        fig.add_trace(go.Scatter(
            x=[r["ic_bas"], r["ic_haut"]], y=[i, i], mode="lines",
            line=dict(color=coul, width=3), showlegend=False, hoverinfo="skip"))

    fig.add_trace(go.Scatter(
        x=g["taux"], y=y, mode="markers",
        marker=dict(size=9 + 16 * g["gwp_a_risque"].rank(pct=True).fillna(0),
                    color=[COULEURS_VERDICT[v] for v in g["verdict"]],
                    line=dict(width=0.8, color="white")),
        showlegend=False,
        customdata=np.column_stack([g[categorie], g["n_anomalies"], g["n_total"],
                                    g["gwp_a_risque"], g["p_value"], g["verdict"]]),
        hovertemplate="<b>%{customdata[0]}</b><br>Taux : %{x:.1%}<br>"
                      "Anomalies : %{customdata[1]:.0f} / %{customdata[2]:.0f}<br>"
                      "GWP a risque : %{customdata[3]:,.0f}<br>"
                      "p-value : %{customdata[4]:.2e}<br>"
                      "<b>%{customdata[5]}</b><extra></extra>"))

    fig.update_layout(
        title=f"Taux d'anomalie et incertitude — {categorie}"
              "<br><sup>Barre entierement a droite de la ligne noire = taux significativement "
              "superieur au hasard. Barre large = trop peu d'observations pour conclure</sup>",
        xaxis=dict(title="Taux d'anomalie (IC 95 % de Wilson)", tickformat=".0%"),
        yaxis=dict(tickmode="array", tickvals=y,
                   ticktext=[f"{r[categorie]}  <sub>n={int(r['n_total'])}</sub>"
                             for _, r in g.iterrows()],
                   tickfont=dict(size=9)),
        template="plotly_white", height=max(420, 26 * len(g) + 200))
    afficher(fig)
    return g


forest_taux(expl, "Activity", top_n=20)














