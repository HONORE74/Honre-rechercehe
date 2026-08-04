import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_couverture_hierarchique(results, niveau1="Lob", niveau2="Risk",
                                 alpha=0.10, n_min=15, tolerance=0.05,
                                 max_sous_seg=28, fichier=None):
    """Style du graphique de reference : axe hierarchique a 2 niveaux,
    ligne de liaison, pastilles rouge/vert, bande de tolerance."""
    cible = 1 - alpha

    t = (results.groupby([niveau1, niveau2], observed=True)["dans_intervalle"]
         .agg(k="sum", n="size").reset_index())
    t = t[t["n"] >= n_min].copy()
    t["cov"] = t["k"] / t["n"]
    t = (t.sort_values([niveau1, "cov"]).groupby(niveau1, observed=True)
         .head(max_sous_seg // max(t[niveau1].nunique(), 1)).reset_index(drop=True))
    t = t.sort_values([niveau1, niveau2]).reset_index(drop=True)
    x = np.arange(len(t))

    fig, ax = plt.subplots(figsize=(max(13, 0.62*len(t)), 8))

    # Bande de tolerance (le "rose" du graphique de reference)
    ax.axhspan(cible - tolerance, cible + tolerance,
               color="#E8A0A0", alpha=0.30, zorder=0)
    ax.axhline(cible, color="#333333", ls="--", lw=1.8, zorder=2)

    # Ligne de liaison entre les points
    ax.plot(x, t["cov"], color="#2F2F2F", lw=1.9, zorder=3, alpha=0.85)

    # Pastilles : vert si conforme, rouge sinon
    conforme = t["cov"] >= cible - tolerance
    coul = np.where(conforme, "#8FD98F", "#F0563C")
    ax.scatter(x, t["cov"], s=135, c=coul, edgecolor="#2F2F2F", lw=1.1, zorder=6)

    # Valeurs au-dessus des points
    for xi, row in zip(x, t.itertuples()):
        dy = 0.035 if row.cov >= cible else -0.055
        ax.annotate(f"{row.cov:.2%}", (xi, row.cov + dy), ha="center",
                    fontsize=8.5, fontweight="bold",
                    color="#2F2F2F" if row.cov >= cible else "#B22222")
        ax.annotate(f"n={row.n}", (xi, row.cov - 0.028 if row.cov >= cible else row.cov + 0.022),
                    ha="center", fontsize=6.8, color="#777777")

    # --- Axe niveau 2 (sous-segments) ---
    ax.set_xticks(x)
    ax.set_xticklabels(t[niveau2].astype(str), rotation=55, ha="right", fontsize=8.5)
    ax.set_xlim(-0.7, len(t) - 0.3)

    # --- Axe niveau 1 (groupes) : labels centres + separateurs ---
    bornes, debut = [], 0
    for g in t[niveau1].astype(str).unique():
        idx = np.where(t[niveau1].astype(str).values == g)[0]
        bornes.append((g, idx.min(), idx.max()))
    for g, i0, i1 in bornes:
        centre = (i0 + i1) / 2
        ax.annotate(g, xy=(centre, -0.20), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=11, fontweight="bold",
                    color="#444444", annotation_clip=False)
        ax.annotate("", xy=(i0 - 0.4, -0.165), xytext=(i1 + 0.4, -0.165),
                    xycoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=1.1),
                    annotation_clip=False)
    for _, _, i1 in bornes[:-1]:
        ax.axvline(i1 + 0.5, color="#DDDDDD", lw=1, zorder=1)

    ax.set_ylabel("Taux de couverture observe", fontsize=12)
    ax.set_ylim(max(0, t["cov"].min() - 0.14), min(1.06, t["cov"].max() + 0.12))
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    ax.grid(axis="y", ls=":", alpha=0.35); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    glob = results["dans_intervalle"].mean()
    ax.set_title(f"Couverture conditionnelle  |  {niveau1} > {niveau2}\n"
                 f"Cible {cible:.0%}   -   Marginale globale {glob:.1%}   -   "
                 f"{(~conforme).sum()} sous-segment(s) hors tolerance",
                 fontsize=13, fontweight="bold", pad=16)
    ax.legend(handles=[
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#8FD98F",
               markeredgecolor="#2F2F2F", markersize=11, label=f"Conforme (+/- {tolerance:.0%})"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#F0563C",
               markeredgecolor="#2F2F2F", markersize=11, label="Hors tolerance"),
        Line2D([0],[0], color="#333333", ls="--", lw=1.8, label=f"Cible {cible:.0%}")],
        fontsize=9, loc="lower right", framealpha=0.95)

    plt.subplots_adjust(bottom=0.30)
    plt.savefig(fichier or f"couverture_hierarchique_{niveau1}_{niveau2}.png",
                dpi=220, bbox_inches="tight")
    plt.show()
    return t

tab_h = plot_couverture_hierarchique(results_test, niveau1="Lob", niveau2="Risk", alpha=ALPHA)

















import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_interactif_couverture(results, dimensions=("Lob","Risk","Activity","Periodicity"),
                               alpha=0.10, n_min=20, tolerance=0.05, fichier="couverture_interactive.html"):
    """Graphique interactif : survol pour le detail, menu deroulant pour changer de dimension."""
    cible = 1 - alpha
    dims = [d for d in dimensions if d in results.columns]
    fig = go.Figure()
    boutons, n_traces = [], []

    for k, dim in enumerate(dims):
        t = (results.groupby(dim, observed=True)["dans_intervalle"]
             .agg(kk="sum", n="size").reset_index())
        t = t[t["n"] >= n_min].copy()
        t["cov"] = t["kk"] / t["n"]
        z = 1.96
        d_ = 1 + z**2 / t["n"]
        c_ = (t["cov"] + z**2/(2*t["n"])) / d_
        h_ = (z/d_) * np.sqrt(t["cov"]*(1-t["cov"])/t["n"] + z**2/(4*t["n"]**2))
        t["lo"], t["hi"] = (c_ - h_).clip(0), (c_ + h_).clip(upper=1)
        t = t.sort_values("cov").reset_index(drop=True)
        vis = (k == 0)

        fig.add_trace(go.Scatter(
            x=t[dim].astype(str), y=t["cov"], mode="lines",
            line=dict(color="#2F2F2F", width=2), name="Liaison",
            showlegend=False, visible=vis, hoverinfo="skip"))

        fig.add_trace(go.Scatter(
            x=t[dim].astype(str), y=t["cov"], mode="markers+text",
            marker=dict(size=np.clip(12 + 26*t["n"]/t["n"].max(), 12, 34),
                        color=np.where(t["cov"] >= cible - tolerance, "#8FD98F", "#F0563C"),
                        line=dict(color="#2F2F2F", width=1.4)),
            text=[f"{v:.1%}" for v in t["cov"]], textposition="top center",
            textfont=dict(size=10),
            error_y=dict(type="data", symmetric=False,
                         array=t["hi"]-t["cov"], arrayminus=t["cov"]-t["lo"],
                         color="rgba(120,120,120,0.6)", thickness=1.2, width=4),
            customdata=np.column_stack([t["n"], t["lo"], t["hi"], t["cov"]-cible]),
            hovertemplate=("<b>%{x}</b><br>"
                           "Couverture : %{y:.2%}<br>"
                           "Effectif   : %{customdata[0]:,.0f}<br>"
                           "IC 95%%     : [%{customdata[1]:.2%} ; %{customdata[2]:.2%}]<br>"
                           "Ecart cible: %{customdata[3]:+.2%}<extra></extra>"),
            name=dim, visible=vis, showlegend=False))
        n_traces.append(2)

    for k, dim in enumerate(dims):
        v = []
        for j, nt in enumerate(n_traces):
            v.extend([j == k] * nt)
        boutons.append(dict(label=dim, method="update",
                            args=[{"visible": v},
                                  {"title": f"Couverture conditionnelle par {dim}"}]))

    fig.add_hrect(y0=cible-tolerance, y1=cible+tolerance,
                  fillcolor="#E8A0A0", opacity=0.25, line_width=0)
    fig.add_hline(y=cible, line_dash="dash", line_color="#333333", line_width=2,
                  annotation_text=f"Cible {cible:.0%}", annotation_position="right")

    fig.update_layout(
        title=dict(text=f"Couverture conditionnelle par {dims[0]}", font=dict(size=17)),
        updatemenus=[dict(buttons=boutons, direction="down", x=1.0, xanchor="right",
                          y=1.16, yanchor="top", showactive=True)],
        yaxis=dict(title="Taux de couverture observe", tickformat=".0%", range=[0, 1.05]),
        xaxis=dict(title="Segment", tickangle=-45),
        template="plotly_white", height=650,
        hovermode="closest", margin=dict(t=110, b=130))

    fig.write_html(fichier)
    fig.show()
    print(f"Sauvegarde : {fichier}  (ouvrable dans un navigateur, partageable)")
    return fig

fig_int = plot_interactif_couverture(results_test, alpha=ALPHA)



















import plotly.express as px

def sunburst_couverture(results, hierarchie=("Lob","Risk"), alpha=0.10, n_min=10):
    """Chaque anneau = un niveau de hierarchie. Clic pour zoomer dans une branche."""
    cible = 1 - alpha
    cols = [c for c in hierarchie if c in results.columns]
    t = (results.groupby(cols, observed=True)["dans_intervalle"]
         .agg(cov="mean", n="size").reset_index())
    t = t[t["n"] >= n_min]

    fig = px.sunburst(t, path=cols, values="n", color="cov",
                      color_continuous_scale="RdYlGn",
                      range_color=[cible-0.25, cible+0.06],
                      hover_data={"cov": ":.2%", "n": True})
    fig.update_layout(
        title=f"Couverture conditionnelle hierarchique  ({' > '.join(cols)})<br>"
              f"<sub>Taille = effectif | Couleur = couverture | Cible {cible:.0%}</sub>",
        coloraxis_colorbar=dict(title="Couverture", tickformat=".0%"),
        height=720, template="plotly_white")
    fig.write_html("sunburst_couverture.html"); fig.show()
    return t

sun = sunburst_couverture(results_test, ("Lob","Risk"), ALPHA)
