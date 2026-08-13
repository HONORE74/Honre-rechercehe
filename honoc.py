# ================================================================
# REPARTITION DES ANOMALIES PONDEREE PAR IMPORTANCE
#   Taille  : somme du score_composite (= A x B x GWP) par segment
#   Couleur : score moyen par anomalie — BLEU = peu grave, ROUGE = grave
# ================================================================

import numpy as np
import pandas as pd
import plotly.express as px


def sunburst_gravite_bluered(anomalies_df, chemin=None):
    chemin = chemin or [c for c in ["Lob", "Risk", "Partner"]
                        if c in anomalies_df.columns][:2]

    dd = anomalies_df.dropna(subset=chemin + ["score_composite"]).copy()
    for c in chemin:
        dd[c] = dd[c].astype(str)

    n_total = len(dd)
    score_total_global = dd["score_composite"].sum()

    agg = {"n_anomalies": ("score_composite", "size"),
          "score_total": ("score_composite", "sum"),
          "score_moyen": ("score_composite", "mean")}
    if GWP_COL in dd.columns:
        agg["gwp_total"] = (GWP_COL, "sum")

    stats = dd.groupby(chemin, observed=True).agg(**agg).reset_index()
    stats["part_du_score"] = stats["score_total"] / score_total_global

    if "gwp_total" not in stats.columns:
        stats["gwp_total"] = np.nan

    fig = px.sunburst(
        stats, path=chemin, values="score_total",
        color="score_moyen", color_continuous_scale="Bluered",
        custom_data=["n_anomalies", "gwp_total", "part_du_score", "score_moyen"])
    fig.update_traces(
        texttemplate="%{label}<br>%{customdata[2]:.1%} du score",
        hovertemplate="<b>%{label}</b><br>"
                      "Nombre d'anomalies : %{customdata[0]:.0f}<br>"
                      "Score cumule : %{value:.3f}  (%{customdata[2]:.1%} du total)<br>"
                      "Score moyen/anomalie : %{customdata[3]:.4f}<br>"
                      f"{GWP_COL} total : %{{customdata[1]:,.0f}}"
                      "<extra></extra>",
        insidetextorientation="radial")
    fig.update_layout(
        title=f"Repartition des {n_total} anomalies, ponderee par importance"
              "<br><sup>Taille = score cumule (A x B x GWP) | "
              "Couleur : bleu = peu grave, rouge = grave</sup>",
        template="plotly_white", height=720,
        coloraxis_colorbar=dict(
            title="Gravite<br>moyenne", thickness=18, len=0.75,
            tickfont=dict(size=10)))
    fig.show()
    return stats


stats_bluered = sunburst_gravite_bluered(anomalies_prio)































# ================================================================
# REPARTITION DES ANOMALIES PONDEREE PAR IMPORTANCE
#   Taille   : somme du score_composite (= A x B x GWP) par segment
#              -> 1 anomalie a 1M pese plus que 10 anomalies a 10 euros
#   Couleur  : score moyen par anomalie du segment
#              -> distingue "beaucoup de petites" vs "une grosse"
# ================================================================

import numpy as np
import pandas as pd
import plotly.express as px


def sunburst_gravite_ylorrd(anomalies_df, chemin=None):
    chemin = chemin or [c for c in ["Lob", "Risk", "Partner"]
                        if c in anomalies_df.columns][:2]

    dd = anomalies_df.dropna(subset=chemin + ["score_composite"]).copy()
    for c in chemin:
        dd[c] = dd[c].astype(str)

    n_total = len(dd)
    score_total_global = dd["score_composite"].sum()

    agg = {"n_anomalies": ("score_composite", "size"),
          "score_total": ("score_composite", "sum"),
          "score_moyen": ("score_composite", "mean")}
    if GWP_COL in dd.columns:
        agg["gwp_total"] = (GWP_COL, "sum")

    stats = dd.groupby(chemin, observed=True).agg(**agg).reset_index()
    stats["part_du_score"] = stats["score_total"] / score_total_global

    if "gwp_total" not in stats.columns:
        stats["gwp_total"] = np.nan

    fig = px.sunburst(
        stats, path=chemin, values="score_total",
        color="score_moyen", color_continuous_scale="YlOrRd",
        custom_data=["n_anomalies", "gwp_total", "part_du_score", "score_moyen"])
    fig.update_traces(
        texttemplate="%{label}<br>%{customdata[2]:.1%} du score",
        hovertemplate="<b>%{label}</b><br>"
                      "Nombre d'anomalies : %{customdata[0]:.0f}<br>"
                      "Score cumule : %{value:.3f}  (%{customdata[2]:.1%} du total)<br>"
                      "Score moyen/anomalie : %{customdata[3]:.4f}<br>"
                      f"{GWP_COL} total : %{{customdata[1]:,.0f}}"
                      "<extra></extra>",
        insidetextorientation="radial")
    fig.update_layout(
        title=f"Repartition des {n_total} anomalies, ponderee par importance"
              "<br><sup>Taille = score cumule (A x B x GWP) | "
              "Couleur = score moyen par anomalie (concentration de la gravite)</sup>",
        template="plotly_white", height=720,
        coloraxis_colorbar=dict(title="Score<br>moyen"))
    fig.show()
    return stats


stats_ylorrd = sunburst_gravite_ylorrd(anomalies_prio)


























# ================================================================
# REPARTITION DES ANOMALIES PONDEREE PAR IMPORTANCE
#   Taille  : somme du score_composite (= A x B x GWP) par segment
#   Couleur : RANG PERCENTILE de gravite — bleu = 0, rouge = 100
#             (etale les teintes uniformement malgre la queue lourde)
# ================================================================

import numpy as np
import pandas as pd
import plotly.express as px


def sunburst_gravite_rang(anomalies_df, chemin=None):
    chemin = chemin or [c for c in ["Lob", "Risk", "Partner"]
                        if c in anomalies_df.columns][:2]

    dd = anomalies_df.dropna(subset=chemin + ["score_composite"]).copy()
    for c in chemin:
        dd[c] = dd[c].astype(str)

    n_total = len(dd)
    score_total_global = dd["score_composite"].sum()

    agg = {"n_anomalies": ("score_composite", "size"),
          "score_total": ("score_composite", "sum"),
          "score_moyen": ("score_composite", "mean")}
    if GWP_COL in dd.columns:
        agg["gwp_total"] = (GWP_COL, "sum")

    stats = dd.groupby(chemin, observed=True).agg(**agg).reset_index()
    stats["part_du_score"] = stats["score_total"] / score_total_global
    stats["rang_gravite"] = stats["score_moyen"].rank(pct=True) * 100

    if "gwp_total" not in stats.columns:
        stats["gwp_total"] = np.nan

    fig = px.sunburst(
        stats, path=chemin, values="score_total",
        color="rang_gravite", color_continuous_scale="Bluered",
        range_color=[0, 100],
        custom_data=["n_anomalies", "gwp_total", "part_du_score",
                     "score_moyen", "rang_gravite"])
    fig.update_traces(
        texttemplate="%{label}<br>%{customdata[2]:.1%} du score",
        hovertemplate="<b>%{label}</b><br>"
                      "Rang de gravite : %{customdata[4]:.0f} / 100<br>"
                      "─────────────<br>"
                      "Nombre d'anomalies : %{customdata[0]:.0f}<br>"
                      "Score cumule : %{value:.3f}  (%{customdata[2]:.1%} du total)<br>"
                      "Score moyen/anomalie : %{customdata[3]:.4f}<br>"
                      f"{GWP_COL} total : %{{customdata[1]:,.0f}}"
                      "<extra></extra>",
        insidetextorientation="radial")
    fig.update_layout(
        title=f"Repartition des {n_total} anomalies, ponderee par importance"
              "<br><sup>Taille = score cumule (A x B x GWP) | "
              "Couleur = rang de gravite : bleu = les moins graves, rouge = les plus graves</sup>",
        template="plotly_white", height=720,
        coloraxis_colorbar=dict(
            title="Rang de<br>gravite", thickness=18, len=0.75,
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0<br>(min)", "25", "50", "75", "100<br>(max)"],
            tickfont=dict(size=10)))
    fig.show()
    return stats


stats_rang = sunburst_gravite_rang(anomalies_prio)















# ================================================================
# REPARTITION DES ANOMALIES PONDEREE PAR IMPORTANCE
#   Taille  : somme du score_composite (= A x B x GWP) par segment
#   Couleur : RANG PERCENTILE de gravite — bleu = 0, rouge = 100
#             (etale les teintes uniformement malgre la queue lourde)
# ================================================================

import numpy as np
import pandas as pd
import plotly.express as px


def sunburst_gravite_rang(anomalies_df, chemin=None):
    chemin = chemin or [c for c in ["Lob", "Risk", "Partner"]
                        if c in anomalies_df.columns][:2]

    dd = anomalies_df.dropna(subset=chemin + ["score_composite"]).copy()
    for c in chemin:
        dd[c] = dd[c].astype(str)

    n_total = len(dd)
    score_total_global = dd["score_composite"].sum()

    agg = {"n_anomalies": ("score_composite", "size"),
          "score_total": ("score_composite", "sum"),
          "score_moyen": ("score_composite", "mean")}
    if GWP_COL in dd.columns:
        agg["gwp_total"] = (GWP_COL, "sum")

    stats = dd.groupby(chemin, observed=True).agg(**agg).reset_index()
    stats["part_du_score"] = stats["score_total"] / score_total_global
    stats["rang_gravite"] = stats["score_moyen"].rank(pct=True) * 100

    if "gwp_total" not in stats.columns:
        stats["gwp_total"] = np.nan

    fig = px.sunburst(
        stats, path=chemin, values="score_total",
        color="rang_gravite", color_continuous_scale="Bluered",
        range_color=[0, 100],
        custom_data=["n_anomalies", "gwp_total", "part_du_score",
                     "score_moyen", "rang_gravite"])
    fig.update_traces(
        texttemplate="%{label}<br>%{customdata[2]:.1%} du score",
        hovertemplate="<b>%{label}</b><br>"
                      "Rang de gravite : %{customdata[4]:.0f} / 100<br>"
                      "─────────────<br>"
                      "Nombre d'anomalies : %{customdata[0]:.0f}<br>"
                      "Score cumule : %{value:.3f}  (%{customdata[2]:.1%} du total)<br>"
                      "Score moyen/anomalie : %{customdata[3]:.4f}<br>"
                      f"{GWP_COL} total : %{{customdata[1]:,.0f}}"
                      "<extra></extra>",
        insidetextorientation="radial")
    fig.update_layout(
        title=f"Repartition des {n_total} anomalies, ponderee par importance"
              "<br><sup>Taille = score cumule (A x B x GWP) | "
              "Couleur = rang de gravite : bleu = les moins graves, rouge = les plus graves</sup>",
        template="plotly_white", height=720,
        coloraxis_colorbar=dict(
            title="Rang de<br>gravite", thickness=18, len=0.75,
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0<br>(min)", "25", "50", "75", "100<br>(max)"],
            tickfont=dict(size=10)))
    fig.show()
    return stats


stats_rang = sunburst_gravite_rang(anomalies_prio)

















# ================================================================
# BLOC AVANCE — TABLEAU DE BORD LIE
#   Cliquez sur un segment du sunburst : les panneaux du bas se
#   recomposent sur ce segment (anomalies les plus critiques,
#   decomposition du score, position dans la distribution globale).
#   Clic au centre = retour au niveau global.
# ================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML


ECHELLE = "Bluered"


def _construire_hierarchie(dd, chemin):
    """Assemble ids / labels / parents pour un go.Sunburst avec branchvalues='total'."""
    lignes = []
    for prof in range(1, len(chemin) + 1):
        cols = chemin[:prof]
        agg = {"score_total": ("score_composite", "sum"),
               "score_moyen": ("score_composite", "mean"),
               "score_max":   ("score_composite", "max"),
               "n":           ("score_composite", "size")}
        if GWP_COL in dd.columns:
            agg["gwp"] = (GWP_COL, "sum")
        g = dd.groupby(cols, observed=True).agg(**agg).reset_index()
        for _, r in g.iterrows():
            vals = [str(r[c]) for c in cols]
            lignes.append({
                "id": "/".join(vals),
                "label": vals[-1],
                "parent": "/".join(vals[:-1]) if prof > 1 else "",
                "profondeur": prof,
                "score_total": r["score_total"],
                "score_moyen": r["score_moyen"],
                "score_max": r["score_max"],
                "n": int(r["n"]),
                "gwp": r.get("gwp", np.nan)})
    h = pd.DataFrame(lignes)
    h["rang_gravite"] = h["score_moyen"].rank(pct=True) * 100
    return h


def _filtrer_par_noeud(dd, chemin, node_id):
    """Restreint le DataFrame aux lignes correspondant au chemin clique."""
    if not node_id:
        return dd, "Ensemble des anomalies"
    parts = node_id.split("/")
    masque = pd.Series(True, index=dd.index)
    for col, val in zip(chemin, parts):
        masque &= dd[col].astype(str) == val
    return dd[masque], " › ".join(parts)


def _cartes_html(sub, dd_global, titre):
    """Bandeau de statistiques en HTML."""
    part = sub["score_composite"].sum() / max(dd_global["score_composite"].sum(), 1e-12)
    pire = sub.loc[sub["score_composite"].idxmax()] if len(sub) else None
    gwp = f"{sub[GWP_COL].sum():,.0f}" if GWP_COL in sub.columns else "n/a"
    pire_txt = (f"#{int(pire['rank'])}" if pire is not None
                and pd.notna(pire.get("rank", np.nan)) else "—")

    cartes = [("Anomalies", f"{len(sub):,}", "#37474f"),
              ("Score cumule", f"{sub['score_composite'].sum():,.3g}", "#1565c0"),
              ("Part du score global", f"{100*part:.1f} %", "#c62828"),
              (f"{GWP_COL} total", gwp, "#2e7d32"),
              ("Pire anomalie", pire_txt, "#6a1b9a")]

    blocs = "".join(
        f"<div style='flex:1;min-width:140px;background:#fff;border:1px solid #e0e0e0;"
        f"border-left:5px solid {c};border-radius:7px;padding:12px 14px;"
        f"box-shadow:0 1px 3px rgba(0,0,0,.07)'>"
        f"<div style='font-size:11px;color:#78909c;text-transform:uppercase;"
        f"letter-spacing:.6px'>{t}</div>"
        f"<div style='font-size:21px;font-weight:600;color:{c};margin-top:4px'>{v}</div>"
        f"</div>" for t, v, c in cartes)

    return HTML(
        f"<div style='font-family:system-ui,sans-serif;margin:6px 0 14px 0'>"
        f"<div style='font-size:15px;font-weight:600;color:#263238;margin-bottom:10px'>"
        f"📍 {titre}</div>"
        f"<div style='display:flex;gap:10px;flex-wrap:wrap'>{blocs}</div></div>")


def _figure_details(sub, dd_global, titre, top_n=12):
    """Deux panneaux : anomalies les plus critiques + decomposition du score."""
    if len(sub) == 0:
        print("Aucune anomalie dans ce segment.")
        return

    top = sub.nlargest(min(top_n, len(sub)), "score_composite").iloc[::-1]
    id_cols = [c for c in ID_COLS if c in top.columns]
    labels = top[id_cols].astype(str).agg(" | ".join, axis=1).str.slice(0, 34)

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.58, 0.42], horizontal_spacing=0.13,
        subplot_titles=(f"Les {len(top)} anomalies les plus critiques",
                        "Ce qui porte le score (rang percentile global)"))

    rangs_couleur = top["score_composite"].rank(pct=True)
    fig.add_trace(go.Bar(
        x=top["score_composite"], y=labels, orientation="h",
        marker=dict(color=rangs_couleur, colorscale=ECHELLE, cmin=0, cmax=1,
                    line=dict(width=0.5, color="white")),
        customdata=np.column_stack([
            top["rank"].fillna(-1), top["y_obs"], top["y_pred"],
            top["borne_basse"], top["borne_haute"],
            top[GWP_COL] if GWP_COL in top.columns else np.full(len(top), np.nan)]),
        hovertemplate="<b>%{y}</b><br>Score : %{x:.4g}<br>"
                      "Rang global : #%{customdata[0]:.0f}<br>"
                      "Observe : %{customdata[1]:,.0f}<br>"
                      "Predit : %{customdata[2]:,.0f}<br>"
                      "Intervalle : [%{customdata[3]:,.0f} ; %{customdata[4]:,.0f}]<br>"
                      f"{GWP_COL} : %{{customdata[5]:,.0f}}<extra></extra>",
        showlegend=False), row=1, col=1)

    facteurs = {"A — ecart borne": "A_ecart_borne",
                "B — erreur modele": "B_erreur_modele",
                f"{GWP_COL} — exposition": GWP_COL}
    facteurs = {k: v for k, v in facteurs.items() if v in dd_global.columns}

    z = np.column_stack([
        dd_global[v].rank(pct=True).reindex(top.index).values * 100
        for v in facteurs.values()])

    fig.add_trace(go.Heatmap(
        z=z, x=list(facteurs.keys()), y=labels,
        colorscale=ECHELLE, zmin=0, zmax=100,
        text=np.round(z, 0), texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(title="Rang<br>percentile", thickness=13, len=0.72, x=1.02),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Rang : %{z:.0f}/100<extra></extra>"),
        row=1, col=2)

    fig.update_xaxes(title_text="Score composite", row=1, col=1)
    fig.update_yaxes(tickfont=dict(size=9), row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_layout(
        title=dict(text=f"Detail — {titre}", font=dict(size=14)),
        template="plotly_white", height=max(420, 34 * len(top) + 190),
        margin=dict(l=10, r=90, t=110, b=50))
    fig.show()


def dashboard_critique(anomalies_df, chemin=None, top_n=12):
    chemin = chemin or [c for c in ["Lob", "Risk", "Partner"]
                        if c in anomalies_df.columns][:2]

    dd = anomalies_df.dropna(subset=chemin + ["score_composite"]).copy()
    for c in chemin:
        dd[c] = dd[c].astype(str)

    h = _construire_hierarchie(dd, chemin)
    n_total = len(dd)
    score_global = dd["score_composite"].sum()

    survol = [
        f"<b>{r['label']}</b><br>"
        f"Rang de gravite : {r['rang_gravite']:.0f}/100<br>"
        f"─────────────<br>"
        f"Anomalies : {r['n']}<br>"
        f"Score cumule : {r['score_total']:.4g}  ({100*r['score_total']/score_global:.1f} % du total)<br>"
        f"Score moyen : {r['score_moyen']:.4g}<br>"
        f"Pire anomalie : {r['score_max']:.4g}<br>"
        + (f"{GWP_COL} : {r['gwp']:,.0f}" if pd.notna(r["gwp"]) else "")
        for _, r in h.iterrows()]

    sunburst = go.Sunburst(
        ids=h["id"], labels=h["label"], parents=h["parent"],
        values=h["score_total"], branchvalues="total",
        marker=dict(colors=h["rang_gravite"], colorscale=ECHELLE,
                    cmin=0, cmax=100, line=dict(color="white", width=1.6),
                    colorbar=dict(title="Rang de<br>gravite", thickness=16,
                                  len=0.7, tickvals=[0, 50, 100],
                                  ticktext=["faible", "moyen", "critique"])),
        text=[f"{100*v/score_global:.0f} %" for v in h["score_total"]],
        texttemplate="%{label}<br>%{text}",
        hovertext=survol, hoverinfo="text",
        insidetextorientation="radial",
        maxdepth=len(chemin))

    fw = go.FigureWidget(data=[sunburst])
    fw.update_layout(
        title=dict(text=f"Repartition des {n_total} anomalies — cliquez un segment"
                        "<br><sup>Taille = score cumule | "
                        "Couleur = gravite (bleu faible, rouge critique)</sup>",
                   font=dict(size=15)),
        template="plotly_white", height=640, margin=dict(t=110, b=20))

    zone_cartes = widgets.Output()
    zone_detail = widgets.Output()

    def _rafraichir(node_id=""):
        sub, titre = _filtrer_par_noeud(dd, chemin, node_id)
        with zone_cartes:
            clear_output(wait=True)
            display(_cartes_html(sub, dd, titre))
        with zone_detail:
            clear_output(wait=True)
            _figure_details(sub, dd, titre, top_n=top_n)

    def _au_clic(trace, points, state):
        if not points.point_inds:
            return
        _rafraichir(trace.ids[points.point_inds[0]])

    try:
        fw.data[0].on_click(_au_clic)
        clic_actif = True
    except Exception:
        clic_actif = False

    # Selecteur de repli : garantit l'acces meme si le clic n'est pas capte
    options = [("— Ensemble des anomalies —", "")] + [
        (f"{'   ' * (r['profondeur'] - 1)}{r['label']}  ({r['n']})", r["id"])
        for _, r in h.sort_values(["profondeur", "score_total"],
                                  ascending=[True, False]).iterrows()]
    selecteur = widgets.Dropdown(options=options, value="",
                                 description="Segment :",
                                 layout=widgets.Layout(width="520px"))
    selecteur.observe(lambda c: _rafraichir(c["new"]) if c["name"] == "value" else None,
                      names="value")

    display(HTML(
        "<div style='font-family:system-ui,sans-serif;font-size:12px;color:#546e7a;"
        "background:#eceff1;padding:8px 12px;border-radius:6px;margin-bottom:8px'>"
        + ("💡 Cliquez un segment du sunburst — ou utilisez le menu — "
           "pour recomposer les panneaux du bas."
           if clic_actif else
           "💡 Le clic direct n'est pas disponible dans cet environnement : "
           "utilisez le menu deroulant ci-dessous.")
        + "</div>"))
    display(fw, selecteur, zone_cartes, zone_detail)
    _rafraichir("")
    return h


hierarchie = dashboard_critique(anomalies_prio, chemin=["Lob", "Risk", "Partner"])












