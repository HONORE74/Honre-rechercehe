# ================================================================
# BLOC AVANCE — TABLEAU DE BORD LIE (v2)
#   Deux menus lies :
#     1) Dimension  : choisit l'axe d'analyse (Lob, Partner, Risk,
#        Companies... selon ce qui existe reellement dans vos donnees)
#     2) Valeur      : se repeuple automatiquement avec les valeurs
#        de la dimension choisie, triees par gravite decroissante.
#        L'option "Vue generale" laisse la dimension entiere, non
#        filtree.
#   Le sunburst est a un seul niveau (celui de la dimension choisie).
#   Cliquer une part met a jour le menu Valeur, qui met a jour les
#   panneaux du bas. Un seul graphique sous le cercle : la barre des
#   anomalies les plus critiques (couleur = rang percentile).
# ================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML


ECHELLE = "Bluered"

# Colonnes candidates pour le menu "Dimension". Seules celles
# reellement presentes dans anomalies_df seront proposees.
DIMENSIONS_CANDIDATES = ["Lob", "Partner", "Risk", "Companies"]


def _construire_niveau(dd, colonne):
    """Agrege le score par valeur d'une seule colonne (un niveau, pas de hierarchie)."""
    agg = {"score_total": ("score_composite", "sum"),
           "score_moyen": ("score_composite", "mean"),
           "score_max":   ("score_composite", "max"),
           "n":           ("score_composite", "size")}
    if GWP_COL in dd.columns:
        agg["gwp"] = (GWP_COL, "sum")
    g = dd.groupby(colonne, observed=True).agg(**agg).reset_index()
    g = g.rename(columns={colonne: "label"})
    g["rang_gravite"] = g["score_moyen"].rank(pct=True) * 100
    return g.sort_values("score_total", ascending=False)


def _filtrer(dd, colonne, valeur):
    """Restreint le DataFrame a une valeur precise de la dimension, ou pas de filtre si valeur vide."""
    if not valeur:
        return dd, f"{colonne} — vue generale"
    return dd[dd[colonne].astype(str) == valeur], f"{colonne} = {valeur}"


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


def _figure_barre(sub, titre, top_n=12):
    """Un seul panneau : les anomalies les plus critiques, couleur = rang percentile."""
    if len(sub) == 0:
        print("Aucune anomalie dans ce segment.")
        return

    top = sub.nlargest(min(top_n, len(sub)), "score_composite").iloc[::-1]
    id_cols = [c for c in ID_COLS if c in top.columns]
    labels = top[id_cols].astype(str).agg(" | ".join, axis=1).str.slice(0, 40)

    rangs_couleur = top["score_composite"].rank(pct=True)
    fig = go.Figure(go.Bar(
        x=top["score_composite"], y=labels, orientation="h",
        marker=dict(color=rangs_couleur, colorscale=ECHELLE, cmin=0, cmax=1,
                    line=dict(width=0.5, color="white"),
                    colorbar=dict(title="Rang<br>percentile", thickness=13, len=0.85)),
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
        showlegend=False))

    fig.update_layout(
        title=dict(text=f"Les {len(top)} anomalies les plus critiques — {titre}", font=dict(size=14)),
        xaxis_title="Score composite",
        yaxis=dict(tickfont=dict(size=9)),
        template="plotly_white", height=max(420, 30 * len(top) + 120),
        margin=dict(l=10, r=90, t=70, b=50))
    fig.show()


def dashboard_critique(anomalies_df, dimensions=None, top_n=12):
    dimensions = dimensions or [c for c in DIMENSIONS_CANDIDATES if c in anomalies_df.columns]
    if not dimensions:
        raise ValueError("Aucune des colonnes de DIMENSIONS_CANDIDATES n'existe dans anomalies_df. "
                          "Passez explicitement dimensions=[...] avec vos noms de colonnes.")

    dd_all = anomalies_df.dropna(subset=["score_composite"]).copy()
    for c in dimensions:
        dd_all[c] = dd_all[c].astype(str)

    zone_cartes = widgets.Output()
    zone_detail = widgets.Output()

    selecteur_dim = widgets.Dropdown(
        options=dimensions, value=dimensions[0],
        description="Dimension :", layout=widgets.Layout(width="320px"))
    selecteur_val = widgets.Dropdown(
        options=[("— Vue generale —", "")], value="",
        description="Valeur :", layout=widgets.Layout(width="420px"))

    fw = go.FigureWidget()
    fw.update_layout(template="plotly_white", height=560, margin=dict(t=90, b=20))

    _en_cours = {"flag": False}  # evite les rafraichissements en boucle lors des maj programmatiques

    def _dessiner_sunburst(colonne):
        niveau = _construire_niveau(dd_all, colonne)
        with fw.batch_update():
            fw.data = []
            fw.add_trace(go.Sunburst(
                ids=niveau["label"], labels=niveau["label"], parents=[""] * len(niveau),
                values=niveau["score_total"],
                marker=dict(colors=niveau["rang_gravite"], colorscale=ECHELLE,
                            cmin=0, cmax=100, line=dict(color="white", width=1.6),
                            colorbar=dict(title="Rang de<br>gravite", thickness=16,
                                          len=0.7, tickvals=[0, 50, 100],
                                          ticktext=["faible", "moyen", "critique"])),
                text=[f"{100*v/max(dd_all['score_composite'].sum(),1e-12):.0f} %"
                      for v in niveau["score_total"]],
                texttemplate="%{label}<br>%{text}",
                hovertext=[
                    f"<b>{r['label']}</b><br>Rang de gravite : {r['rang_gravite']:.0f}/100<br>"
                    f"─────────────<br>Anomalies : {r['n']}<br>"
                    f"Score cumule : {r['score_total']:.4g}<br>"
                    f"Score moyen : {r['score_moyen']:.4g}<br>"
                    f"Pire anomalie : {r['score_max']:.4g}"
                    + (f"<br>{GWP_COL} : {r['gwp']:,.0f}" if pd.notna(r.get("gwp", np.nan)) else "")
                    for _, r in niveau.iterrows()],
                hoverinfo="text", insidetextorientation="radial"))
            fw.layout.title = dict(
                text=f"Repartition par {colonne} — cliquez une part"
                     "<br><sup>Taille = score cumule | Couleur = gravite (bleu faible, rouge critique)</sup>",
                font=dict(size=15))

        try:
            fw.data[0].on_click(_au_clic)
        except Exception:
            pass

        return niveau

    def _rafraichir_panneaux():
        colonne = selecteur_dim.value
        valeur = selecteur_val.value
        sub, titre = _filtrer(dd_all, colonne, valeur)
        with zone_cartes:
            clear_output(wait=True)
            display(_cartes_html(sub, dd_all, titre))
        with zone_detail:
            clear_output(wait=True)
            _figure_barre(sub, titre, top_n=top_n)

    def _changer_dimension(change=None):
        colonne = selecteur_dim.value
        niveau = _dessiner_sunburst(colonne)
        _en_cours["flag"] = True
        selecteur_val.options = [("— Vue generale —", "")] + [
            (f"{r['label']}  ({r['n']})", r["label"]) for _, r in niveau.iterrows()]
        selecteur_val.value = ""
        _en_cours["flag"] = False
        _rafraichir_panneaux()

    def _changer_valeur(change=None):
        if _en_cours["flag"]:
            return
        _rafraichir_panneaux()

    def _au_clic(trace, points, state):
        if not points.point_inds:
            return
        selecteur_val.value = trace.labels[points.point_inds[0]]  # declenche _changer_valeur

    selecteur_dim.observe(lambda c: _changer_dimension() if c["name"] == "value" else None, names="value")
    selecteur_val.observe(lambda c: _changer_valeur() if c["name"] == "value" else None, names="value")

    display(HTML(
        "<div style='font-family:system-ui,sans-serif;font-size:12px;color:#546e7a;"
        "background:#eceff1;padding:8px 12px;border-radius:6px;margin-bottom:8px'>"
        "💡 Choisissez d'abord une dimension, puis une valeur (ou laissez « Vue generale »). "
        "Cliquer une part du cercle fait aussi avancer le menu Valeur."
        "</div>"))
    display(widgets.HBox([selecteur_dim, selecteur_val]))
    display(fw, zone_cartes, zone_detail)

    _changer_dimension()


dashboard_critique(anomalies_prio, dimensions=["Lob", "Partner", "Risk"])






ECHELLE = "RdYlGn_r"
