

























# ================================================================
# TABLEAU DE BORD — version revisee
#   ① Cercle a couches empilables — couleur = GRAVITE MOYENNE
#   ② Maille (ligne 1) puis valeur (ligne 2), independantes du cercle
#   ③ Panneaux : cartes -> bar plot des scores -> FOREST PLOT
#   (heatmap "rang percentile" supprimee, panneau CQR remplace)
# ================================================================
# ┌─────────────────── PARAMETRES A MODIFIER ───────────────────┐
TOP_N_PANNEAUX = 12       # Nombre d'anomalies dans le bar plot ET le forest plot
                          # (meme granularite volontairement)
ECHELLE        = "Bluered"   # Bleu = peu grave, rouge = grave
LOG_FOREST     = True     # Echelle log sur l'axe des montants du forest plot
COL_BARPLOT    = "score_composite"   # Grandeur des barres.
                          # Alternative : "abs_z" pour la gravite pure sans GWP
# └──────────────────────────────────────────────────────────────┘

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

VUE_GENERALE = ""


# ------------------------------------------------------------- helpers

def _labels(d, maxlen=34):
    id_cols = [c for c in ID_COLS if c in d.columns]
    return d[id_cols].astype(str).agg(" | ".join, axis=1).str.slice(0, maxlen)


def _hover(r, id_cols):
    t = (f"<b>{' | '.join(str(r[c]) for c in id_cols)}</b><br>"
         f"Observe    : {r['y_obs']:,.0f}<br>"
         f"Predit     : {r['y_pred']:,.0f}<br>"
         f"Intervalle : [{r['borne_basse']:,.0f} ; {r['borne_haute']:,.0f}]")
    if "rank" in r.index and pd.notna(r["rank"]):
        t += f"<br>Rang priorite : #{int(r['rank'])}"
    if "score_composite" in r.index:
        t += f"<br>Score : {r['score_composite']:.4g}"
    if GWP_COL in r.index:
        t += f"<br>{GWP_COL} : {r[GWP_COL]:,.0f}"
    return t


def _hierarchie(dd, chemin):
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
            lignes.append({"id": "/".join(vals), "label": vals[-1],
                           "parent": "/".join(vals[:-1]) if prof > 1 else "",
                           "profondeur": prof, "score_total": r["score_total"],
                           "score_moyen": r["score_moyen"], "score_max": r["score_max"],
                           "n": int(r["n"]), "gwp": r.get("gwp", np.nan)})
    return pd.DataFrame(lignes)


def _filtrer_maille(df, colonne, valeur):
    if not valeur:
        return df, f"{colonne} — vue generale"
    return df[df[colonne].astype(str) == str(valeur)], f"{colonne} = {valeur}"


def _cartes(sub, dd_global, titre, sub_expl=None):
    part = sub["score_composite"].sum() / max(dd_global["score_composite"].sum(), 1e-12)
    gwp = (f"{sub[GWP_COL].sum():,.0f}"
           if GWP_COL in sub.columns and sub[GWP_COL].notna().any() else "n/a")
    grav = (f"{sub['score_composite'].mean():,.4g}" if len(sub) else "—")
    pire = "—"
    if len(sub) and "rank" in sub.columns:
        rk = sub.loc[sub["score_composite"].idxmax(), "rank"]
        if pd.notna(rk):
            pire = f"#{int(rk)}"
    couv = (f"{100*sub_expl['dans_intervalle'].mean():.1f} %"
            if sub_expl is not None and len(sub_expl) else "n/a")

    cartes = [("Anomalies", f"{len(sub):,}", "#37474f"),
              ("Gravite moyenne", grav, "#c62828"),
              ("Score cumule", f"{sub['score_composite'].sum():,.3g}", "#1565c0"),
              ("Part du score global", f"{100*part:.1f} %", "#ad1457"),
              (f"{GWP_COL} total", gwp, "#2e7d32"),
              ("Couverture CQR", couv, "#00838f"),
              ("Pire anomalie", pire, "#6a1b9a")]
    blocs = "".join(
        f"<div style='flex:1;min-width:130px;background:#fff;border:1px solid #e0e0e0;"
        f"border-left:5px solid {c};border-radius:7px;padding:11px 13px;"
        f"box-shadow:0 1px 3px rgba(0,0,0,.07)'>"
        f"<div style='font-size:10.5px;color:#78909c;text-transform:uppercase;"
        f"letter-spacing:.6px'>{t}</div>"
        f"<div style='font-size:19px;font-weight:600;color:{c};margin-top:4px'>{v}</div>"
        f"</div>" for t, v, c in cartes)
    return HTML(f"<div style='font-family:system-ui,sans-serif;margin:6px 0 14px 0'>"
                f"<div style='font-size:15px;font-weight:600;color:#263238;"
                f"margin-bottom:10px'>📍 {titre}</div>"
                f"<div style='display:flex;gap:9px;flex-wrap:wrap'>{blocs}</div></div>")


# ------------------- panneau 1 : bar plot seul (heatmap supprimee)

def _creer_fw_bar():
    fig = go.Figure(go.Bar(x=[], y=[], orientation="h", showlegend=False,
                           marker=dict(colorscale=ECHELLE, cmin=0, cmax=1,
                                       line=dict(width=0.5, color="white"))))
    fig.update_layout(template="plotly_white", height=520,
                      xaxis_title="Score composite",
                      yaxis=dict(tickfont=dict(size=9)),
                      margin=dict(l=10, r=40, t=90, b=45))
    return go.FigureWidget(fig)


def _maj_fw_bar(fw, sub, titre, top_n=TOP_N_PANNEAUX, col=COL_BARPLOT):
    if len(sub) == 0:
        with fw.batch_update():
            fw.data[0].x, fw.data[0].y = [], []
            fw.layout.title = dict(text=f"{titre}<br><sup>Aucune anomalie</sup>",
                                   font=dict(size=14))
            fw.layout.height = 260
        return
    cle = col if col in sub.columns else "score_composite"
    top = sub.nlargest(min(top_n, len(sub)), cle).iloc[::-1]
    labels = _labels(top).tolist()
    id_cols = [c for c in ID_COLS if c in top.columns]
    with fw.batch_update():
        fw.data[0].x = top[cle].tolist()
        fw.data[0].y = labels
        fw.data[0].marker.color = top[cle].rank(pct=True).tolist()
        fw.data[0].text = [_hover(r, id_cols) for _, r in top.iterrows()]
        fw.data[0].hovertemplate = "%{text}<extra></extra>"
        fw.layout.xaxis.title.text = ("Score composite" if cle == "score_composite"
                                      else cle)
        fw.layout.title = dict(
            text=f"Les {len(top)} anomalies les plus critiques — {titre}",
            font=dict(size=14))
        fw.layout.height = max(360, 34 * len(top) + 150)


# ------------------- panneau 2 : FOREST PLOT (remplace le panneau CQR)

def _creer_fw_forest():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                             line=dict(color="#3a6bbf", width=10), opacity=0.3,
                             name=f"Intervalle conforme ({100*(1-ALPHA):.0f} %)",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                             line=dict(color="#c0392b", width=2, dash="dot"),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name="Prediction",
                             marker=dict(symbol="diamond", size=10, color="white",
                                         line=dict(color="black", width=1.5))))
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name="Valeur comptabilisee",
                             marker=dict(size=13, color="#c0392b",
                                         line=dict(color="#7b241c", width=1.3))))
    fig.update_layout(template="plotly_white", height=520,
                      xaxis=dict(title=TARGET),
                      legend=dict(orientation="h", yanchor="bottom", y=1.03,
                                  xanchor="center", x=0.5),
                      margin=dict(l=10, r=40, t=110, b=50))
    return go.FigureWidget(fig)


def _maj_fw_forest(fw, sub, titre, top_n=TOP_N_PANNEAUX, log_x=LOG_FOREST):
    if len(sub) == 0:
        with fw.batch_update():
            for t in fw.data:
                t.x, t.y = [], []
            fw.layout.title = dict(text=f"{titre}<br><sup>Aucune anomalie</sup>",
                                   font=dict(size=14))
            fw.layout.height = 260
        return

    d = sub.nlargest(min(top_n, len(sub)), "score_composite").iloc[::-1].reset_index(drop=True)
    y = list(range(len(d)))
    id_cols = [c for c in ID_COLS if c in d.columns]
    lo = d["borne_basse"].values.astype(float)
    hi = d["borne_haute"].values.astype(float)
    obs = d["y_obs"].values.astype(float)
    pred = d["y_pred"].values.astype(float)
    log_ok = bool(log_x and (obs > 0).all() and (pred > 0).all() and (lo > 0).all())

    xs_band, ys_band = [], []
    for yi, l, h in zip(y, lo, hi):
        xs_band += [l, h, None]
        ys_band += [yi, yi, None]

    xs_over, ys_over = [], []
    for yi, o, l, h in zip(y, obs, lo, hi):
        cible = h if o > h else l
        xs_over += [cible, o, None]
        ys_over += [yi, yi, None]

    textes = [_hover(r, id_cols) for _, r in d.iterrows()]
    ticks = [f"#{int(r)}  {l}" if pd.notna(r) else l
             for r, l in zip(d.get("rank", pd.Series([np.nan] * len(d))),
                             _labels(d))]

    with fw.batch_update():
        fw.data[0].x, fw.data[0].y = xs_band, ys_band
        fw.data[1].x, fw.data[1].y = xs_over, ys_over
        fw.data[2].x, fw.data[2].y = pred, y
        fw.data[2].text = textes
        fw.data[2].hovertemplate = "%{text}<extra></extra>"
        fw.data[3].x, fw.data[3].y = obs, y
        fw.data[3].text = textes
        fw.data[3].hovertemplate = "%{text}<extra></extra>"
        fw.layout.xaxis.type = "log" if log_ok else "linear"
        fw.layout.xaxis.title.text = TARGET + ("  (log)" if log_ok else "")
        fw.layout.yaxis = dict(tickmode="array", tickvals=y, ticktext=ticks,
                               tickfont=dict(size=9))
        fw.layout.title = dict(
            text=f"Intervalle conforme, prediction et valeur observee — {titre}"
                 "<br><sup>Une unite statistique par ligne, meme granularite que "
                 "le graphique ci-dessus</sup>", font=dict(size=14))
        fw.layout.height = max(400, 40 * len(d) + 170)


# ------------------------------------------------------------- dashboard

def dashboard_complet(anomalies_prio, expl, top_n=TOP_N_PANNEAUX):
    cols = [c for c in ID_COLS if c in anomalies_prio.columns and c in expl.columns]
    if not cols:
        print("Aucune colonne d'identification commune.")
        return

    dd = anomalies_prio.dropna(subset=["score_composite"]).copy()
    ex = expl.copy()
    for c in cols:
        dd[c] = dd[c].astype(str)
        ex[c] = ex[c].astype(str)
    score_global = dd["score_composite"].sum()
    verrou = {"actif": False}

    # ---- ① Couches du cercle ------------------------------------------
    prefs = [c for c in ["Lob", "Partner", "Companies", "Risk"] if c in cols]
    defauts = [prefs[0] if prefs else cols[0], None, None, None]
    niveaux = [widgets.Dropdown(
        options=[("— aucun —", None)] + [(c, c) for c in cols],
        value=defauts[i], description=f"Couche {i+1} :",
        layout=widgets.Layout(width="255px"),
        style={"description_width": "72px"}) for i in range(4)]

    fw_cercle = go.FigureWidget(data=[go.Sunburst(ids=[], labels=[], parents=[],
                                                  values=[], branchvalues="total")])
    fw_cercle.update_layout(template="plotly_white", height=620, margin=dict(t=110, b=20))

    # ---- ② Maille puis valeur -----------------------------------------
    maille_defaut = prefs[0] if prefs else cols[0]
    sel_maille = widgets.Dropdown(
        options=[(c, c) for c in cols], value=maille_defaut,
        description="1 · Maille :", layout=widgets.Layout(width="330px"),
        style={"description_width": "90px"})
    sel_valeur = widgets.Dropdown(
        options=[("— vue generale —", VUE_GENERALE)], value=VUE_GENERALE,
        description="2 · Valeur :", layout=widgets.Layout(width="480px"),
        style={"description_width": "90px"})

    z_cartes = widgets.Output()
    fw_bar = _creer_fw_bar()
    fw_forest = _creer_fw_forest()

    def _chemin():
        vus, out = set(), []
        for w in niveaux:
            if w.value and w.value not in vus:
                out.append(w.value)
                vus.add(w.value)
        return out

    def _maj_cercle(*_):
        chemin = _chemin()
        if not chemin:
            with fw_cercle.batch_update():
                t = fw_cercle.data[0]
                t.ids, t.labels, t.parents, t.values = [], [], [], []
                fw_cercle.layout.title = dict(text="Activez au moins une couche.",
                                              font=dict(size=15))
            return
        h = _hierarchie(dd, chemin)
        # CONDITION : la couleur est la GRAVITE MOYENNE en valeur reelle,
        # bornee au P95 pour que la queue lourde n'ecrase pas l'echelle
        cmax = float(np.nanpercentile(h["score_moyen"], 95)) or float(h["score_moyen"].max())
        survol = [f"<b>{r['label']}</b><br>"
                  f"Gravite moyenne : {r['score_moyen']:.4g}<br>"
                  f"─────────────<br>Anomalies : {r['n']}<br>"
                  f"Score cumule : {r['score_total']:.4g} "
                  f"({100*r['score_total']/score_global:.1f} % du total)<br>"
                  f"Pire anomalie : {r['score_max']:.4g}"
                  + (f"<br>{GWP_COL} : {r['gwp']:,.0f}" if pd.notna(r["gwp"]) else "")
                  for _, r in h.iterrows()]
        with fw_cercle.batch_update():
            t = fw_cercle.data[0]
            t.ids, t.labels = h["id"].tolist(), h["label"].tolist()
            t.parents, t.values = h["parent"].tolist(), h["score_total"].tolist()
            t.text = [f"{100*v/score_global:.0f} %" for v in h["score_total"]]
            t.texttemplate = "%{label}<br>%{text}"
            t.hovertext, t.hoverinfo = survol, "text"
            t.insidetextorientation = "radial"
            t.maxdepth = len(chemin)
            t.marker = dict(colors=h["score_moyen"].tolist(), colorscale=ECHELLE,
                            cmin=0, cmax=cmax, line=dict(color="white", width=1.6),
                            colorbar=dict(title="Gravite<br>moyenne", thickness=16,
                                          len=0.7, tickformat=".2g"))
            fw_cercle.layout.title = dict(
                text=f"Repartition des {len(dd)} anomalies — {len(chemin)} couche(s) : "
                     f"{' › '.join(chemin)}"
                     "<br><sup>Taille = score cumule | Couleur = gravite moyenne "
                     "(bleu faible, rouge elevee)</sup>", font=dict(size=15))

    def _maj_panneaux(*_):
        colonne, valeur = sel_maille.value, sel_valeur.value
        sub, titre = _filtrer_maille(dd, colonne, valeur)
        sub_ex, _ = _filtrer_maille(ex, colonne, valeur)
        with z_cartes:
            clear_output(wait=True)
            display(_cartes(sub, dd, titre, sub_ex))
        _maj_fw_bar(fw_bar, sub, titre, top_n=top_n)
        _maj_fw_forest(fw_forest, sub, titre, top_n=top_n)

    def _options_valeurs(colonne):
        g = (dd.groupby(colonne, observed=True)["score_composite"]
               .agg(["size", "sum"]).reset_index().sort_values("sum", ascending=False))
        return [("— vue generale —", VUE_GENERALE)] + [
            (f"{r[colonne]}  ({int(r['size'])} anomalies)", str(r[colonne]))
            for _, r in g.iterrows()]

    def _maj_valeurs(*_):
        if verrou["actif"]:
            return
        verrou["actif"] = True
        try:
            sel_valeur.options = _options_valeurs(sel_maille.value)
            sel_valeur.value = VUE_GENERALE
        finally:
            verrou["actif"] = False
        _maj_panneaux()

    def _au_clic(trace, points, state):
        if not points.point_inds:
            return
        parts = trace.ids[points.point_inds[0]].split("/")
        chemin = _chemin()
        if not parts or len(parts) > len(chemin):
            return
        colonne, valeur = chemin[len(parts) - 1], parts[-1]
        verrou["actif"] = True
        try:
            sel_maille.value = colonne
            sel_valeur.options = _options_valeurs(colonne)
            dispo = [v for _, v in sel_valeur.options]
            sel_valeur.value = valeur if valeur in dispo else VUE_GENERALE
        finally:
            verrou["actif"] = False
        _maj_panneaux()

    try:
        fw_cercle.data[0].on_click(_au_clic)
        clic = True
    except Exception:
        clic = False

    for w in niveaux:
        w.observe(lambda c: _maj_cercle() if c["name"] == "value" else None, names="value")
    sel_maille.observe(lambda c: _maj_valeurs() if c["name"] == "value" else None,
                       names="value")
    sel_valeur.observe(lambda c: (_maj_panneaux() if not verrou["actif"] else None)
                       if c["name"] == "value" else None, names="value")

    def _bandeau(txt, fond="#eceff1", coul="#37474f"):
        return HTML(f"<div style='font-family:system-ui,sans-serif;font-size:12.5px;"
                    f"color:{coul};background:{fond};padding:9px 13px;border-radius:6px;"
                    f"margin:14px 0 8px 0'>{txt}</div>")

    display(_bandeau(
        "<b>① COUCHES DU CERCLE</b> — chaque couche activee ajoute un anneau. "
        "Couleur = gravite moyenne des anomalies du segment."))
    display(widgets.HBox(niveaux[:2]), widgets.HBox(niveaux[2:]))
    display(fw_cercle)

    display(_bandeau(
        "<b>② PANNEAUX DE DETAIL</b> — commandes independantes du cercle. "
        "Maille (ligne 1) puis valeur (ligne 2). "
        f"Maille active par defaut : <b>{maille_defaut}</b>."
        + ("<br>Le clic sur un segment du cercle synchronise ces deux commandes."
           if clic else ""),
        fond="#e3f2fd", coul="#0d47a1"))
    display(sel_maille)
    display(sel_valeur)
    display(z_cartes, fw_bar, fw_forest)

    _maj_cercle()
    _maj_valeurs()
    return {"cercle": fw_cercle, "bar": fw_bar, "forest": fw_forest,
            "maille": sel_maille, "valeur": sel_valeur}


controles = dashboard_complet(anomalies_prio, expl)



















# ================================================================
# TABLEAU DE BORD — cercle a couches empilables + panneaux a maille propre
#
#   HAUT  : 4 niveaux -> chaque niveau actif ajoute un anneau au cercle
#   BAS   : ligne 1 = maille (toujours definie par defaut)
#           ligne 2 = valeur dans cette maille
#           -> pilote les cartes, le top des anomalies et la vue CQR
#   Le clic sur un segment du cercle synchronise le bas (bonus, non bloquant)
# ================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

ECHELLE = "Bluered"
N_CQR_MAX = 120
VUE_GENERALE = ""


# ------------------------------------------------------------- helpers

def _hierarchie(dd, chemin):
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
            lignes.append({"id": "/".join(vals), "label": vals[-1],
                           "parent": "/".join(vals[:-1]) if prof > 1 else "",
                           "profondeur": prof, "score_total": r["score_total"],
                           "score_moyen": r["score_moyen"], "score_max": r["score_max"],
                           "n": int(r["n"]), "gwp": r.get("gwp", np.nan)})
    h = pd.DataFrame(lignes)
    h["rang_gravite"] = h["score_moyen"].rank(pct=True) * 100
    return h


def _filtrer_maille(df, colonne, valeur):
    """Filtre sur UNE maille et UNE valeur. Valeur vide = vue generale."""
    if not valeur:
        return df, f"{colonne} — vue generale"
    return df[df[colonne].astype(str) == str(valeur)], f"{colonne} = {valeur}"


def _cartes(sub, dd_global, titre, sub_expl=None):
    part = sub["score_composite"].sum() / max(dd_global["score_composite"].sum(), 1e-12)
    gwp = (f"{sub[GWP_COL].sum():,.0f}"
           if GWP_COL in sub.columns and sub[GWP_COL].notna().any() else "n/a")
    pire = "—"
    if len(sub) and "rank" in sub.columns:
        rk = sub.loc[sub["score_composite"].idxmax(), "rank"]
        if pd.notna(rk):
            pire = f"#{int(rk)}"
    couv = (f"{100*sub_expl['dans_intervalle'].mean():.1f} %"
            if sub_expl is not None and len(sub_expl) else "n/a")

    cartes = [("Anomalies", f"{len(sub):,}", "#37474f"),
              ("Score cumule", f"{sub['score_composite'].sum():,.3g}", "#1565c0"),
              ("Part du score global", f"{100*part:.1f} %", "#c62828"),
              (f"{GWP_COL} total", gwp, "#2e7d32"),
              ("Couverture CQR", couv, "#00838f"),
              ("Pire anomalie", pire, "#6a1b9a")]
    blocs = "".join(
        f"<div style='flex:1;min-width:135px;background:#fff;border:1px solid #e0e0e0;"
        f"border-left:5px solid {c};border-radius:7px;padding:11px 13px;"
        f"box-shadow:0 1px 3px rgba(0,0,0,.07)'>"
        f"<div style='font-size:10.5px;color:#78909c;text-transform:uppercase;"
        f"letter-spacing:.6px'>{t}</div>"
        f"<div style='font-size:20px;font-weight:600;color:{c};margin-top:4px'>{v}</div>"
        f"</div>" for t, v, c in cartes)
    return HTML(f"<div style='font-family:system-ui,sans-serif;margin:6px 0 14px 0'>"
                f"<div style='font-size:15px;font-weight:600;color:#263238;"
                f"margin-bottom:10px'>📍 {titre}</div>"
                f"<div style='display:flex;gap:9px;flex-wrap:wrap'>{blocs}</div></div>")


def _figure_top(sub, dd_global, titre, top_n=12):
    if len(sub) == 0:
        print("Aucune anomalie dans cette selection.")
        return
    top = sub.nlargest(min(top_n, len(sub)), "score_composite").iloc[::-1]
    id_cols = [c for c in ID_COLS if c in top.columns]
    labels = top[id_cols].astype(str).agg(" | ".join, axis=1).str.slice(0, 34)

    fig = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42],
                        horizontal_spacing=0.13,
                        subplot_titles=(f"Les {len(top)} anomalies les plus critiques",
                                        "Ce qui porte le score (rang percentile global)"))
    fig.add_trace(go.Bar(
        x=top["score_composite"], y=labels, orientation="h",
        marker=dict(color=top["score_composite"].rank(pct=True), colorscale=ECHELLE,
                    cmin=0, cmax=1, line=dict(width=0.5, color="white")),
        customdata=np.column_stack([
            top["rank"].fillna(-1) if "rank" in top.columns else np.full(len(top), -1),
            top["y_obs"], top["y_pred"], top["borne_basse"], top["borne_haute"],
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
    if facteurs:
        z = np.column_stack([dd_global[v].rank(pct=True).reindex(top.index).values * 100
                             for v in facteurs.values()])
        fig.add_trace(go.Heatmap(
            z=z, x=list(facteurs.keys()), y=labels, colorscale=ECHELLE, zmin=0, zmax=100,
            text=np.round(z, 0), texttemplate="%{text}", textfont=dict(size=9),
            colorbar=dict(title="Rang<br>percentile", thickness=13, len=0.72, x=1.02),
            hovertemplate="<b>%{y}</b><br>%{x}<br>Rang : %{z:.0f}/100<extra></extra>"),
            row=1, col=2)

    fig.update_xaxes(title_text="Score composite", row=1, col=1)
    fig.update_yaxes(tickfont=dict(size=9), row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_layout(title=dict(text=f"Detail — {titre}", font=dict(size=14)),
                      template="plotly_white", height=max(400, 34 * len(top) + 190),
                      margin=dict(l=10, r=90, t=105, b=45))
    fig.show()


def _figure_cqr(sub_expl, titre, n_max=N_CQR_MAX):
    d = sub_expl.dropna(subset=["y_obs", "y_pred", "borne_basse", "borne_haute"]).copy()
    if len(d) < 2:
        print("Trop peu d'observations pour la vue CQR.")
        return
    couv, n_reel = d["dans_intervalle"].mean(), len(d)
    if len(d) > n_max:
        d = d.sample(n_max, random_state=42)
    d = d.sort_values("y_pred").reset_index(drop=True)

    x = np.arange(len(d))
    obs, pred = d["y_obs"].values.astype(float), d["y_pred"].values.astype(float)
    lo, hi = d["borne_basse"].values.astype(float), d["borne_haute"].values.astype(float)
    dedans = d["dans_intervalle"].values.astype(bool)
    centre, demi = (lo + hi) / 2, np.maximum((hi - lo) / 2, 1e-9)
    z = (obs - centre) / demi
    log_ok = (obs > 0).all() and (pred > 0).all()

    id_cols = [c for c in ID_COLS if c in d.columns]
    hover = [(f"<b>{' | '.join(str(r[c]) for c in id_cols)}</b><br>"
              f"Observe : {r['y_obs']:,.0f}<br>Predit : {r['y_pred']:,.0f}<br>"
              f"Intervalle : [{r['borne_basse']:,.0f} ; {r['borne_haute']:,.0f}]<br>"
              f"z = {zz:+.2f}") for (_, r), zz in zip(d.iterrows(), z)]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        row_heights=[0.62, 0.38],
                        subplot_titles=("Observations et intervalle de prediction",
                                        "Severite normalisee (echelle constante)"))
    fig.add_trace(go.Scatter(
        x=x, y=pred, mode="markers",
        error_y=dict(type="data", symmetric=False, array=hi - pred, arrayminus=pred - lo,
                     color="rgba(90,130,200,0.55)", thickness=1.5, width=2),
        marker=dict(symbol="diamond", size=5, color="#37474f"),
        name="Prediction + intervalle CQR", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x[dedans], y=obs[dedans], mode="markers",
        marker=dict(size=7, color="#1769E0", line=dict(width=0.5, color="white")),
        name=f"Dans l'intervalle ({int(dedans.sum())})",
        text=[h for h, k in zip(hover, dedans) if k],
        hovertemplate="%{text}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x[~dedans], y=obs[~dedans], mode="markers",
        marker=dict(size=10, color="#E53935", line=dict(width=0.7, color="#7b0000")),
        name=f"HORS intervalle ({int((~dedans).sum())})",
        text=[h for h, k in zip(hover, dedans) if not k],
        hovertemplate="%{text}<extra></extra>"), row=1, col=1)

    fig.add_hrect(y0=-1, y1=1, fillcolor="rgba(70,110,230,0.16)", line_width=0, row=2, col=1)
    for yv in (1, -1):
        fig.add_hline(y=yv, line=dict(color="#355CDE", width=1.5), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dash"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x[dedans], y=z[dedans], mode="markers",
                             marker=dict(size=6, color="#1769E0"), showlegend=False,
                             text=[h for h, k in zip(hover, dedans) if k],
                             hovertemplate="%{text}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x[~dedans], y=z[~dedans], mode="markers",
                             marker=dict(size=9, color="#E53935"), showlegend=False,
                             text=[h for h, k in zip(hover, dedans) if not k],
                             hovertemplate="%{text}<extra></extra>"), row=2, col=1)

    fig.update_yaxes(title_text=TARGET + ("  (log)" if log_ok else ""),
                     type="log" if log_ok else "linear", row=1, col=1)
    fig.update_yaxes(title_text="z",
                     range=[max(-6, np.nanmin(z) - 0.5), min(10, np.nanmax(z) + 0.7)],
                     row=2, col=1)
    fig.update_xaxes(title_text="Observations triees par prediction croissante",
                     showticklabels=False, row=2, col=1)
    fig.update_layout(
        title=dict(text=f"Vue CQR — {titre}<br><sup>Couverture : {100*couv:.1f} % "
                        f"(cible {100*(1-ALPHA):.0f} %) sur {n_reel:,} observations | "
                        f"{len(d)} affichees</sup>", font=dict(size=14)),
        template="plotly_white", height=680, hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
        margin=dict(t=140))
    fig.show()


# ------------------------------------------------------------- dashboard

def dashboard_complet(anomalies_prio, expl, top_n=12):
    cols = [c for c in ID_COLS if c in anomalies_prio.columns and c in expl.columns]
    if not cols:
        print("Aucune colonne d'identification commune entre anomalies_prio et expl.")
        return

    dd = anomalies_prio.dropna(subset=["score_composite"]).copy()
    ex = expl.copy()
    for c in cols:
        dd[c] = dd[c].astype(str)
        ex[c] = ex[c].astype(str)
    score_global = dd["score_composite"].sum()
    verrou = {"actif": False}

    # ---- HAUT : couches du cercle ------------------------------------
    prefs = [c for c in ["Lob", "Partner", "Companies", "Risk"] if c in cols]
    defauts = [prefs[0] if prefs else cols[0], None, None, None]
    niveaux = [widgets.Dropdown(
        options=[("— aucun —", None)] + [(c, c) for c in cols],
        value=defauts[i], description=f"Couche {i+1} :",
        layout=widgets.Layout(width="255px"),
        style={"description_width": "72px"}) for i in range(4)]

    fw = go.FigureWidget(data=[go.Sunburst(ids=[], labels=[], parents=[], values=[],
                                           branchvalues="total")])
    fw.update_layout(template="plotly_white", height=620, margin=dict(t=110, b=20))

    # ---- BAS : maille (ligne 1) puis valeur (ligne 2) ------------------
    maille_defaut = prefs[0] if prefs else cols[0]
    selecteur_maille = widgets.Dropdown(
        options=[(c, c) for c in cols], value=maille_defaut,
        description="1 · Maille :", layout=widgets.Layout(width="330px"),
        style={"description_width": "90px"})
    selecteur_valeur = widgets.Dropdown(
        options=[("— vue generale —", VUE_GENERALE)], value=VUE_GENERALE,
        description="2 · Valeur :", layout=widgets.Layout(width="480px"),
        style={"description_width": "90px"})

    z_cartes, z_top, z_cqr = (widgets.Output(), widgets.Output(), widgets.Output())

    def _chemin():
        vus, out = set(), []
        for w in niveaux:
            if w.value and w.value not in vus:
                out.append(w.value)
                vus.add(w.value)
        return out

    # ---- Mise a jour du cercle ---------------------------------------
    def _maj_cercle(*_):
        chemin = _chemin()
        if not chemin:
            with fw.batch_update():
                fw.data[0].ids, fw.data[0].labels = [], []
                fw.data[0].parents, fw.data[0].values = [], []
                fw.layout.title = dict(text="Activez au moins une couche.",
                                       font=dict(size=15))
            return
        h = _hierarchie(dd, chemin)
        survol = [f"<b>{r['label']}</b><br>Rang de gravite : {r['rang_gravite']:.0f}/100"
                  f"<br>─────────────<br>Anomalies : {r['n']}<br>"
                  f"Score cumule : {r['score_total']:.4g} "
                  f"({100*r['score_total']/score_global:.1f} % du total)<br>"
                  f"Score moyen : {r['score_moyen']:.4g}<br>"
                  f"Pire anomalie : {r['score_max']:.4g}"
                  + (f"<br>{GWP_COL} : {r['gwp']:,.0f}" if pd.notna(r["gwp"]) else "")
                  for _, r in h.iterrows()]
        with fw.batch_update():
            t = fw.data[0]
            t.ids, t.labels = h["id"].tolist(), h["label"].tolist()
            t.parents, t.values = h["parent"].tolist(), h["score_total"].tolist()
            t.text = [f"{100*v/score_global:.0f} %" for v in h["score_total"]]
            t.texttemplate = "%{label}<br>%{text}"
            t.hovertext, t.hoverinfo = survol, "text"
            t.insidetextorientation = "radial"
            t.maxdepth = len(chemin)
            t.marker = dict(colors=h["rang_gravite"].tolist(), colorscale=ECHELLE,
                            cmin=0, cmax=100, line=dict(color="white", width=1.6),
                            colorbar=dict(title="Rang de<br>gravite", thickness=16,
                                          len=0.7, tickvals=[0, 50, 100],
                                          ticktext=["faible", "moyen", "critique"]))
            fw.layout.title = dict(
                text=f"Repartition des {len(dd)} anomalies — {len(chemin)} couche(s) : "
                     f"{' › '.join(chemin)}"
                     "<br><sup>Taille = score cumule | Couleur = gravite "
                     "(bleu faible, rouge critique)</sup>", font=dict(size=15))

    # ---- Mise a jour des panneaux du bas -------------------------------
    def _maj_panneaux(*_):
        colonne, valeur = selecteur_maille.value, selecteur_valeur.value
        sub, titre = _filtrer_maille(dd, colonne, valeur)
        sub_ex, _ = _filtrer_maille(ex, colonne, valeur)
        with z_cartes:
            clear_output(wait=True)
            display(_cartes(sub, dd, titre, sub_ex))
        with z_top:
            clear_output(wait=True)
            _figure_top(sub, dd, titre, top_n=top_n)
        with z_cqr:
            clear_output(wait=True)
            _figure_cqr(sub_ex, titre)

    def _maj_valeurs(*_):
        """Change de maille -> repeuple la liste des valeurs (ligne 2)."""
        if verrou["actif"]:
            return
        verrou["actif"] = True
        try:
            colonne = selecteur_maille.value
            g = (dd.groupby(colonne, observed=True)["score_composite"]
                   .agg(["size", "sum"]).reset_index()
                   .sort_values("sum", ascending=False))
            selecteur_valeur.options = [("— vue generale —", VUE_GENERALE)] + [
                (f"{r[colonne]}  ({int(r['size'])} anomalies)", str(r[colonne]))
                for _, r in g.iterrows()]
            selecteur_valeur.value = VUE_GENERALE
        finally:
            verrou["actif"] = False
        _maj_panneaux()

    def _au_clic(trace, points, state):
        """Bonus : le clic sur le cercle synchronise la maille et la valeur du bas."""
        if not points.point_inds:
            return
        node = trace.ids[points.point_inds[0]]
        chemin = _chemin()
        parts = node.split("/")
        if not parts or len(parts) > len(chemin):
            return
        colonne, valeur = chemin[len(parts) - 1], parts[-1]
        verrou["actif"] = True
        try:
            selecteur_maille.value = colonne
            g = (dd.groupby(colonne, observed=True)["score_composite"]
                   .agg(["size", "sum"]).reset_index()
                   .sort_values("sum", ascending=False))
            selecteur_valeur.options = [("— vue generale —", VUE_GENERALE)] + [
                (f"{r[colonne]}  ({int(r['size'])} anomalies)", str(r[colonne]))
                for _, r in g.iterrows()]
            selecteur_valeur.value = valeur if valeur in [
                v for _, v in selecteur_valeur.options] else VUE_GENERALE
        finally:
            verrou["actif"] = False
        _maj_panneaux()

    try:
        fw.data[0].on_click(_au_clic)
        clic = True
    except Exception:
        clic = False

    for w in niveaux:
        w.observe(lambda c: _maj_cercle() if c["name"] == "value" else None, names="value")
    selecteur_maille.observe(lambda c: _maj_valeurs() if c["name"] == "value" else None,
                             names="value")
    selecteur_valeur.observe(lambda c: (_maj_panneaux() if not verrou["actif"] else None)
                             if c["name"] == "value" else None, names="value")

    # ---- Affichage -----------------------------------------------------
    def _bandeau(txt, fond="#eceff1", coul="#37474f"):
        return HTML(f"<div style='font-family:system-ui,sans-serif;font-size:12.5px;"
                    f"color:{coul};background:{fond};padding:9px 13px;border-radius:6px;"
                    f"margin:14px 0 8px 0'>{txt}</div>")

    display(_bandeau(
        "<b>① COUCHES DU CERCLE</b> — chaque couche activee ajoute un anneau. "
        "Couche 1 seule = un cercle simple ; ajoutez la couche 2, puis 3, puis 4 "
        "pour affiner la granularite."))
    display(widgets.HBox(niveaux[:2]), widgets.HBox(niveaux[2:]))
    display(fw)

    display(_bandeau(
        "<b>② PANNEAUX DE DETAIL</b> — commandes independantes du cercle. "
        "Choisissez d'abord la <b>maille</b> (ligne 1), puis la <b>valeur</b> (ligne 2). "
        f"Une maille est deja active par defaut : <b>{maille_defaut}</b>."
        + ("<br>Le clic sur un segment du cercle synchronise automatiquement ces deux "
           "commandes." if clic else ""),
        fond="#e3f2fd", coul="#0d47a1"))
    display(selecteur_maille)
    display(selecteur_valeur)
    display(z_cartes, z_top, z_cqr)

    _maj_cercle()
    _maj_valeurs()
    return {"niveaux": niveaux, "maille": selecteur_maille, "valeur": selecteur_valeur}


controles = dashboard_complet(anomalies_prio, expl)










Bloc 2



# ================================================================
# TABLEAU DE BORD — figures mises a jour EN PLACE (aucune accumulation)
#
#   Les panneaux sont des FigureWidget crees UNE SEULE FOIS.
#   Chaque changement de selection reecrit leur contenu : il n'existe
#   jamais plus d'une figure par panneau, quel que soit l'environnement.
# ================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

ECHELLE = "Bluered"
N_CQR_MAX = 120
VUE_GENERALE = ""


# ------------------------------------------------------------- helpers

def _hierarchie(dd, chemin):
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
            lignes.append({"id": "/".join(vals), "label": vals[-1],
                           "parent": "/".join(vals[:-1]) if prof > 1 else "",
                           "profondeur": prof, "score_total": r["score_total"],
                           "score_moyen": r["score_moyen"], "score_max": r["score_max"],
                           "n": int(r["n"]), "gwp": r.get("gwp", np.nan)})
    h = pd.DataFrame(lignes)
    h["rang_gravite"] = h["score_moyen"].rank(pct=True) * 100
    return h


def _filtrer_maille(df, colonne, valeur):
    if not valeur:
        return df, f"{colonne} — vue generale"
    return df[df[colonne].astype(str) == str(valeur)], f"{colonne} = {valeur}"


def _cartes(sub, dd_global, titre, sub_expl=None):
    part = sub["score_composite"].sum() / max(dd_global["score_composite"].sum(), 1e-12)
    gwp = (f"{sub[GWP_COL].sum():,.0f}"
           if GWP_COL in sub.columns and sub[GWP_COL].notna().any() else "n/a")
    pire = "—"
    if len(sub) and "rank" in sub.columns:
        rk = sub.loc[sub["score_composite"].idxmax(), "rank"]
        if pd.notna(rk):
            pire = f"#{int(rk)}"
    couv = (f"{100*sub_expl['dans_intervalle'].mean():.1f} %"
            if sub_expl is not None and len(sub_expl) else "n/a")

    cartes = [("Anomalies", f"{len(sub):,}", "#37474f"),
              ("Score cumule", f"{sub['score_composite'].sum():,.3g}", "#1565c0"),
              ("Part du score global", f"{100*part:.1f} %", "#c62828"),
              (f"{GWP_COL} total", gwp, "#2e7d32"),
              ("Couverture CQR", couv, "#00838f"),
              ("Pire anomalie", pire, "#6a1b9a")]
    blocs = "".join(
        f"<div style='flex:1;min-width:135px;background:#fff;border:1px solid #e0e0e0;"
        f"border-left:5px solid {c};border-radius:7px;padding:11px 13px;"
        f"box-shadow:0 1px 3px rgba(0,0,0,.07)'>"
        f"<div style='font-size:10.5px;color:#78909c;text-transform:uppercase;"
        f"letter-spacing:.6px'>{t}</div>"
        f"<div style='font-size:20px;font-weight:600;color:{c};margin-top:4px'>{v}</div>"
        f"</div>" for t, v, c in cartes)
    return HTML(f"<div style='font-family:system-ui,sans-serif;margin:6px 0 14px 0'>"
                f"<div style='font-size:15px;font-weight:600;color:#263238;"
                f"margin-bottom:10px'>📍 {titre}</div>"
                f"<div style='display:flex;gap:9px;flex-wrap:wrap'>{blocs}</div></div>")


# ------------------------- panneau TOP : creation puis mise a jour

def _creer_fw_top():
    fig = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42],
                        horizontal_spacing=0.13,
                        subplot_titles=("Anomalies les plus critiques",
                                        "Ce qui porte le score (rang percentile global)"))
    fig.add_trace(go.Bar(x=[], y=[], orientation="h", showlegend=False,
                         marker=dict(colorscale=ECHELLE, cmin=0, cmax=1,
                                     line=dict(width=0.5, color="white"))),
                  row=1, col=1)
    fig.add_trace(go.Heatmap(z=[[0]], x=[" "], y=[" "], colorscale=ECHELLE,
                             zmin=0, zmax=100, texttemplate="%{text}",
                             textfont=dict(size=9),
                             colorbar=dict(title="Rang<br>percentile",
                                           thickness=13, len=0.72, x=1.02)),
                  row=1, col=2)
    fig.update_xaxes(title_text="Score composite", row=1, col=1)
    fig.update_yaxes(tickfont=dict(size=9), row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_layout(template="plotly_white", height=560,
                      margin=dict(l=10, r=90, t=105, b=45))
    return go.FigureWidget(fig)


def _maj_fw_top(fw, sub, dd_global, titre, top_n=12):
    facteurs = {"A — ecart borne": "A_ecart_borne",
                "B — erreur modele": "B_erreur_modele",
                f"{GWP_COL} — exposition": GWP_COL}
    facteurs = {k: v for k, v in facteurs.items() if v in dd_global.columns}

    if len(sub) == 0:
        with fw.batch_update():
            fw.data[0].x, fw.data[0].y = [], []
            fw.data[1].z, fw.data[1].x, fw.data[1].y = [[0]], [" "], [" "]
            fw.data[1].text = [[0]]
            fw.layout.title = dict(text=f"Detail — {titre}<br>"
                                        "<sup>Aucune anomalie dans cette selection</sup>",
                                   font=dict(size=14))
            fw.layout.height = 300
        return

    top = sub.nlargest(min(top_n, len(sub)), "score_composite").iloc[::-1]
    id_cols = [c for c in ID_COLS if c in top.columns]
    labels = top[id_cols].astype(str).agg(" | ".join, axis=1).str.slice(0, 34).tolist()

    cd = np.column_stack([
        top["rank"].fillna(-1) if "rank" in top.columns else np.full(len(top), -1),
        top["y_obs"], top["y_pred"], top["borne_basse"], top["borne_haute"],
        top[GWP_COL] if GWP_COL in top.columns else np.full(len(top), np.nan)])

    z = (np.column_stack([dd_global[v].rank(pct=True).reindex(top.index).values * 100
                          for v in facteurs.values()])
         if facteurs else np.zeros((len(top), 1)))

    with fw.batch_update():
        fw.data[0].x = top["score_composite"].tolist()
        fw.data[0].y = labels
        fw.data[0].marker.color = top["score_composite"].rank(pct=True).tolist()
        fw.data[0].customdata = cd
        fw.data[0].hovertemplate = (
            "<b>%{y}</b><br>Score : %{x:.4g}<br>"
            "Rang global : #%{customdata[0]:.0f}<br>"
            "Observe : %{customdata[1]:,.0f}<br>"
            "Predit : %{customdata[2]:,.0f}<br>"
            "Intervalle : [%{customdata[3]:,.0f} ; %{customdata[4]:,.0f}]<br>"
            f"{GWP_COL} : %{{customdata[5]:,.0f}}<extra></extra>")

        fw.data[1].z = z
        fw.data[1].x = list(facteurs.keys()) if facteurs else [" "]
        fw.data[1].y = labels
        fw.data[1].text = np.round(z, 0)
        fw.data[1].hovertemplate = ("<b>%{y}</b><br>%{x}<br>"
                                    "Rang : %{z:.0f}/100<extra></extra>")

        fw.layout.annotations[0].text = f"Les {len(top)} anomalies les plus critiques"
        fw.layout.title = dict(text=f"Detail — {titre}", font=dict(size=14))
        fw.layout.height = max(400, 34 * len(top) + 190)


# ------------------------- panneau CQR : creation puis mise a jour

def _creer_fw_cqr():
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        row_heights=[0.62, 0.38],
                        subplot_titles=("Observations et intervalle de prediction",
                                        "Severite normalisee (echelle constante)"))
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name="Prediction + intervalle CQR",
                             marker=dict(symbol="diamond", size=5, color="#37474f"),
                             error_y=dict(type="data", symmetric=False,
                                          color="rgba(90,130,200,0.55)",
                                          thickness=1.5, width=2),
                             hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name="Dans l'intervalle",
                             marker=dict(size=7, color="#1769E0",
                                         line=dict(width=0.5, color="white"))),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name="HORS intervalle",
                             marker=dict(size=10, color="#E53935",
                                         line=dict(width=0.7, color="#7b0000"))),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False,
                             marker=dict(size=6, color="#1769E0")), row=2, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False,
                             marker=dict(size=9, color="#E53935")), row=2, col=1)

    fig.add_hrect(y0=-1, y1=1, fillcolor="rgba(70,110,230,0.16)", line_width=0,
                  row=2, col=1)
    for yv in (1, -1):
        fig.add_hline(y=yv, line=dict(color="#355CDE", width=1.5), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dash"), row=2, col=1)

    fig.update_yaxes(title_text="z", row=2, col=1)
    fig.update_xaxes(title_text="Observations triees par prediction croissante",
                     showticklabels=False, row=2, col=1)
    fig.update_layout(template="plotly_white", height=680, hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.04,
                                  xanchor="center", x=0.5),
                      margin=dict(t=140))
    return go.FigureWidget(fig)


def _maj_fw_cqr(fw, sub_expl, titre, n_max=N_CQR_MAX):
    d = sub_expl.dropna(subset=["y_obs", "y_pred", "borne_basse", "borne_haute"]).copy()
    if len(d) < 2:
        with fw.batch_update():
            for t in fw.data:
                t.x, t.y = [], []
            fw.layout.title = dict(text=f"Vue CQR — {titre}<br>"
                                        "<sup>Trop peu d'observations</sup>",
                                   font=dict(size=14))
        return

    couv, n_reel = d["dans_intervalle"].mean(), len(d)
    if len(d) > n_max:
        d = d.sample(n_max, random_state=42)
    d = d.sort_values("y_pred").reset_index(drop=True)

    x = np.arange(len(d))
    obs, pred = d["y_obs"].values.astype(float), d["y_pred"].values.astype(float)
    lo, hi = d["borne_basse"].values.astype(float), d["borne_haute"].values.astype(float)
    dedans = d["dans_intervalle"].values.astype(bool)
    centre, demi = (lo + hi) / 2, np.maximum((hi - lo) / 2, 1e-9)
    z = (obs - centre) / demi
    log_ok = bool((obs > 0).all() and (pred > 0).all())

    id_cols = [c for c in ID_COLS if c in d.columns]
    hover = [(f"<b>{' | '.join(str(r[c]) for c in id_cols)}</b><br>"
              f"Observe : {r['y_obs']:,.0f}<br>Predit : {r['y_pred']:,.0f}<br>"
              f"Intervalle : [{r['borne_basse']:,.0f} ; {r['borne_haute']:,.0f}]<br>"
              f"z = {zz:+.2f}") for (_, r), zz in zip(d.iterrows(), z)]
    h_in = [h for h, k in zip(hover, dedans) if k]
    h_out = [h for h, k in zip(hover, dedans) if not k]

    with fw.batch_update():
        fw.data[0].x, fw.data[0].y = x, pred
        fw.data[0].error_y.array = hi - pred
        fw.data[0].error_y.arrayminus = pred - lo

        fw.data[1].x, fw.data[1].y = x[dedans], obs[dedans]
        fw.data[1].name = f"Dans l'intervalle ({int(dedans.sum())})"
        fw.data[1].text = h_in
        fw.data[1].hovertemplate = "%{text}<extra></extra>"

        fw.data[2].x, fw.data[2].y = x[~dedans], obs[~dedans]
        fw.data[2].name = f"HORS intervalle ({int((~dedans).sum())})"
        fw.data[2].text = h_out
        fw.data[2].hovertemplate = "%{text}<extra></extra>"

        fw.data[3].x, fw.data[3].y = x[dedans], z[dedans]
        fw.data[3].text = h_in
        fw.data[3].hovertemplate = "%{text}<extra></extra>"

        fw.data[4].x, fw.data[4].y = x[~dedans], z[~dedans]
        fw.data[4].text = h_out
        fw.data[4].hovertemplate = "%{text}<extra></extra>"

        fw.layout.yaxis.type = "log" if log_ok else "linear"
        fw.layout.yaxis.title.text = TARGET + ("  (log)" if log_ok else "")
        fw.layout.yaxis2.range = [max(-6, float(np.nanmin(z)) - 0.5),
                                  min(10, float(np.nanmax(z)) + 0.7)]
        fw.layout.title = dict(
            text=f"Vue CQR — {titre}<br><sup>Couverture : {100*couv:.1f} % "
                 f"(cible {100*(1-ALPHA):.0f} %) sur {n_reel:,} observations | "
                 f"{len(d)} affichees</sup>", font=dict(size=14))


# ------------------------------------------------------------- dashboard

def dashboard_complet(anomalies_prio, expl, top_n=12):
    cols = [c for c in ID_COLS if c in anomalies_prio.columns and c in expl.columns]
    if not cols:
        print("Aucune colonne d'identification commune entre anomalies_prio et expl.")
        return

    dd = anomalies_prio.dropna(subset=["score_composite"]).copy()
    ex = expl.copy()
    for c in cols:
        dd[c] = dd[c].astype(str)
        ex[c] = ex[c].astype(str)
    score_global = dd["score_composite"].sum()
    verrou = {"actif": False}

    # ---- ① Couches du cercle ------------------------------------------
    prefs = [c for c in ["Lob", "Partner", "Companies", "Risk"] if c in cols]
    defauts = [prefs[0] if prefs else cols[0], None, None, None]
    niveaux = [widgets.Dropdown(
        options=[("— aucun —", None)] + [(c, c) for c in cols],
        value=defauts[i], description=f"Couche {i+1} :",
        layout=widgets.Layout(width="255px"),
        style={"description_width": "72px"}) for i in range(4)]

    fw_cercle = go.FigureWidget(data=[go.Sunburst(ids=[], labels=[], parents=[],
                                                  values=[], branchvalues="total")])
    fw_cercle.update_layout(template="plotly_white", height=620, margin=dict(t=110, b=20))

    # ---- ② Maille (ligne 1) puis valeur (ligne 2) ----------------------
    maille_defaut = prefs[0] if prefs else cols[0]
    sel_maille = widgets.Dropdown(
        options=[(c, c) for c in cols], value=maille_defaut,
        description="1 · Maille :", layout=widgets.Layout(width="330px"),
        style={"description_width": "90px"})
    sel_valeur = widgets.Dropdown(
        options=[("— vue generale —", VUE_GENERALE)], value=VUE_GENERALE,
        description="2 · Valeur :", layout=widgets.Layout(width="480px"),
        style={"description_width": "90px"})

    z_cartes = widgets.Output()
    fw_top = _creer_fw_top()
    fw_cqr = _creer_fw_cqr()

    def _chemin():
        vus, out = set(), []
        for w in niveaux:
            if w.value and w.value not in vus:
                out.append(w.value)
                vus.add(w.value)
        return out

    def _maj_cercle(*_):
        chemin = _chemin()
        if not chemin:
            with fw_cercle.batch_update():
                t = fw_cercle.data[0]
                t.ids, t.labels, t.parents, t.values = [], [], [], []
                fw_cercle.layout.title = dict(text="Activez au moins une couche.",
                                              font=dict(size=15))
            return
        h = _hierarchie(dd, chemin)
        survol = [f"<b>{r['label']}</b><br>Rang de gravite : {r['rang_gravite']:.0f}/100"
                  f"<br>─────────────<br>Anomalies : {r['n']}<br>"
                  f"Score cumule : {r['score_total']:.4g} "
                  f"({100*r['score_total']/score_global:.1f} % du total)<br>"
                  f"Score moyen : {r['score_moyen']:.4g}<br>"
                  f"Pire anomalie : {r['score_max']:.4g}"
                  + (f"<br>{GWP_COL} : {r['gwp']:,.0f}" if pd.notna(r["gwp"]) else "")
                  for _, r in h.iterrows()]
        with fw_cercle.batch_update():
            t = fw_cercle.data[0]
            t.ids, t.labels = h["id"].tolist(), h["label"].tolist()
            t.parents, t.values = h["parent"].tolist(), h["score_total"].tolist()
            t.text = [f"{100*v/score_global:.0f} %" for v in h["score_total"]]
            t.texttemplate = "%{label}<br>%{text}"
            t.hovertext, t.hoverinfo = survol, "text"
            t.insidetextorientation = "radial"
            t.maxdepth = len(chemin)
            t.marker = dict(colors=h["rang_gravite"].tolist(), colorscale=ECHELLE,
                            cmin=0, cmax=100, line=dict(color="white", width=1.6),
                            colorbar=dict(title="Rang de<br>gravite", thickness=16,
                                          len=0.7, tickvals=[0, 50, 100],
                                          ticktext=["faible", "moyen", "critique"]))
            fw_cercle.layout.title = dict(
                text=f"Repartition des {len(dd)} anomalies — {len(chemin)} couche(s) : "
                     f"{' › '.join(chemin)}"
                     "<br><sup>Taille = score cumule | Couleur = gravite "
                     "(bleu faible, rouge critique)</sup>", font=dict(size=15))

    def _maj_panneaux(*_):
        colonne, valeur = sel_maille.value, sel_valeur.value
        sub, titre = _filtrer_maille(dd, colonne, valeur)
        sub_ex, _ = _filtrer_maille(ex, colonne, valeur)
        with z_cartes:
            clear_output(wait=True)
            display(_cartes(sub, dd, titre, sub_ex))
        _maj_fw_top(fw_top, sub, dd, titre, top_n=top_n)
        _maj_fw_cqr(fw_cqr, sub_ex, titre)

    def _options_valeurs(colonne):
        g = (dd.groupby(colonne, observed=True)["score_composite"]
               .agg(["size", "sum"]).reset_index().sort_values("sum", ascending=False))
        return [("— vue generale —", VUE_GENERALE)] + [
            (f"{r[colonne]}  ({int(r['size'])} anomalies)", str(r[colonne]))
            for _, r in g.iterrows()]

    def _maj_valeurs(*_):
        if verrou["actif"]:
            return
        verrou["actif"] = True
        try:
            sel_valeur.options = _options_valeurs(sel_maille.value)
            sel_valeur.value = VUE_GENERALE
        finally:
            verrou["actif"] = False
        _maj_panneaux()

    def _au_clic(trace, points, state):
        if not points.point_inds:
            return
        parts = trace.ids[points.point_inds[0]].split("/")
        chemin = _chemin()
        if not parts or len(parts) > len(chemin):
            return
        colonne, valeur = chemin[len(parts) - 1], parts[-1]
        verrou["actif"] = True
        try:
            sel_maille.value = colonne
            sel_valeur.options = _options_valeurs(colonne)
            dispo = [v for _, v in sel_valeur.options]
            sel_valeur.value = valeur if valeur in dispo else VUE_GENERALE
        finally:
            verrou["actif"] = False
        _maj_panneaux()

    try:
        fw_cercle.data[0].on_click(_au_clic)
        clic = True
    except Exception:
        clic = False

    for w in niveaux:
        w.observe(lambda c: _maj_cercle() if c["name"] == "value" else None, names="value")
    sel_maille.observe(lambda c: _maj_valeurs() if c["name"] == "value" else None,
                       names="value")
    sel_valeur.observe(lambda c: (_maj_panneaux() if not verrou["actif"] else None)
                       if c["name"] == "value" else None, names="value")

    def _bandeau(txt, fond="#eceff1", coul="#37474f"):
        return HTML(f"<div style='font-family:system-ui,sans-serif;font-size:12.5px;"
                    f"color:{coul};background:{fond};padding:9px 13px;border-radius:6px;"
                    f"margin:14px 0 8px 0'>{txt}</div>")

    display(_bandeau(
        "<b>① COUCHES DU CERCLE</b> — chaque couche activee ajoute un anneau. "
        "Couche 1 seule = un cercle simple ; ajoutez la couche 2, puis 3, puis 4."))
    display(widgets.HBox(niveaux[:2]), widgets.HBox(niveaux[2:]))
    display(fw_cercle)

    display(_bandeau(
        "<b>② PANNEAUX DE DETAIL</b> — commandes independantes du cercle. "
        "Choisissez d'abord la <b>maille</b> (ligne 1), puis la <b>valeur</b> (ligne 2). "
        f"Maille active par defaut : <b>{maille_defaut}</b>."
        + ("<br>Le clic sur un segment du cercle synchronise ces deux commandes."
           if clic else ""),
        fond="#e3f2fd", coul="#0d47a1"))
    display(sel_maille)
    display(sel_valeur)
    display(z_cartes, fw_top, fw_cqr)

    _maj_cercle()
    _maj_valeurs()
    return {"cercle": fw_cercle, "top": fw_top, "cqr": fw_cqr,
            "maille": sel_maille, "valeur": sel_valeur}


controles = dashboard_complet(anomalies_prio, expl)





Bloc 3




# ================================================================
# BLOC A — ECHELLE INDIVIDUELLE : tableau de priorisation
#   Colonnes : IDGroup | Y_OBS | Y_Pred | CP_Lower | CP_Upper |
#              dans_interval | Score
# ================================================================

import numpy as np
import pandas as pd


def tableau_priorisation(anomalies_prio, top_n=50, style=True):
    d = anomalies_prio.copy()
    id_cols = [c for c in ID_COLS if c in d.columns]
    d["IDGroup"] = d[id_cols].astype(str).agg(" | ".join, axis=1)

    t = pd.DataFrame({
        "Rang":           d["rank"].astype("Int64") if "rank" in d else range(1, len(d) + 1),
        "IDGroup":        d["IDGroup"],
        "Y_OBS":          d["y_obs"],
        "Y_Pred":         d["y_pred"],
        "CP_Lower":       d["borne_basse"],
        "CP_Upper":       d["borne_haute"],
        "dans_interval":  d["dans_intervalle"],
        "Score":          d["score_composite"]})
    if GWP_COL in d.columns:
        t[GWP_COL] = d[GWP_COL]

    t = t.sort_values("Score", ascending=False).head(top_n).reset_index(drop=True)

    if not style:
        return t
    try:
        return (t.style
                 .background_gradient(subset=["Score"], cmap="Reds")
                 .format({"Y_OBS": "{:,.0f}", "Y_Pred": "{:,.0f}",
                          "CP_Lower": "{:,.0f}", "CP_Upper": "{:,.0f}",
                          "Score": "{:,.4g}",
                          **({GWP_COL: "{:,.0f}"} if GWP_COL in t.columns else {})})
                 .set_caption(f"Tableau de priorisation — top {len(t)} sur "
                              f"{len(anomalies_prio)} anomalies"))
    except Exception:
        return t


tableau_priorisation(anomalies_prio, top_n=50)




Bloc 4

# ================================================================
# BLOC A — ECHELLE INDIVIDUELLE : tableau de priorisation
#   Colonnes : IDGroup | Y_OBS | Y_Pred | CP_Lower | CP_Upper |
#              dans_interval | Score
# ================================================================

import numpy as np
import pandas as pd


def tableau_priorisation(anomalies_prio, top_n=50, style=True):
    d = anomalies_prio.copy()
    id_cols = [c for c in ID_COLS if c in d.columns]
    d["IDGroup"] = d[id_cols].astype(str).agg(" | ".join, axis=1)

    t = pd.DataFrame({
        "Rang":           d["rank"].astype("Int64") if "rank" in d else range(1, len(d) + 1),
        "IDGroup":        d["IDGroup"],
        "Y_OBS":          d["y_obs"],
        "Y_Pred":         d["y_pred"],
        "CP_Lower":       d["borne_basse"],
        "CP_Upper":       d["borne_haute"],
        "dans_interval":  d["dans_intervalle"],
        "Score":          d["score_composite"]})
    if GWP_COL in d.columns:
        t[GWP_COL] = d[GWP_COL]

    t = t.sort_values("Score", ascending=False).head(top_n).reset_index(drop=True)

    if not style:
        return t
    try:
        return (t.style
                 .background_gradient(subset=["Score"], cmap="Reds")
                 .format({"Y_OBS": "{:,.0f}", "Y_Pred": "{:,.0f}",
                          "CP_Lower": "{:,.0f}", "CP_Upper": "{:,.0f}",
                          "Score": "{:,.4g}",
                          **({GWP_COL: "{:,.0f}"} if GWP_COL in t.columns else {})})
                 .set_caption(f"Tableau de priorisation — top {len(t)} sur "
                              f"{len(anomalies_prio)} anomalies"))
    except Exception:
        return t


tableau_priorisation(anomalies_prio, top_n=50)





Bloc 5 

# ================================================================
# BLOC B — FUNNEL PLOT : quels segments sont REELLEMENT aberrants ?
#
#   X : nombre d'observations du segment
#   Y : taux d'anomalie du segment
#   Courbes : limites de controle a 95 % et 99.8 %, qui se resserrent
#             quand n augmente -> un taux eleve sur 9 obs reste dans
#             l'entonnoir, un taux modere sur 800 obs en sort.
#
#   C'est LA reponse a "78 % sur 9 observations : reel ou hasard ?"
# ================================================================

import plotly.graph_objects as go


def funnel_plot(expl, categorie, n_min=3, reference="observee"):
    """reference : "observee" -> taux global constate | "alpha" -> cible theorique"""
    d = expl.dropna(subset=[categorie]).copy()
    d[categorie] = d[categorie].astype(str)

    agg = {"n": ("dans_intervalle", "size"), "n_ano": ("est_anomalie", "sum")}
    if GWP_COL in d.columns:
        agg["gwp"] = (GWP_COL, "sum")
    g = d.groupby(categorie, observed=True).agg(**agg).reset_index()
    g = g[g["n"] >= n_min].copy()
    if g.empty:
        print("Aucun segment avec assez d'observations.")
        return g
    g["taux"] = g["n_ano"] / g["n"]

    p0 = ALPHA if reference == "alpha" else d["est_anomalie"].mean()

    # Limites de controle : p0 +- z * sqrt(p0(1-p0)/n)
    n_grid = np.logspace(np.log10(max(g["n"].min(), 1)),
                         np.log10(g["n"].max() * 1.15), 250)
    limites = {}
    for z, lab in [(1.96, "95 %"), (3.09, "99.8 %")]:
        se = np.sqrt(p0 * (1 - p0) / n_grid)
        limites[lab] = (np.clip(p0 - z * se, 0, 1), np.clip(p0 + z * se, 0, 1))

    # Statut de chaque segment
    se_seg = np.sqrt(p0 * (1 - p0) / g["n"])
    g["z_score"] = (g["taux"] - p0) / np.maximum(se_seg, 1e-12)
    g["statut"] = np.select(
        [g["z_score"] > 3.09, g["z_score"] > 1.96, g["z_score"] < -1.96],
        ["Aberrant (99.8 %)", "Eleve (95 %)", "Anormalement bas"],
        default="Conforme")

    couleurs = {"Aberrant (99.8 %)": "#b71c1c", "Eleve (95 %)": "#ef6c00",
                "Conforme": "#1769E0", "Anormalement bas": "#2e7d32"}

    fig = go.Figure()
    for lab, (bas, haut) in limites.items():
        opac = 0.30 if lab == "95 %" else 0.16
        fig.add_trace(go.Scatter(x=n_grid, y=haut, mode="lines",
                                 line=dict(color="#90a4ae", width=1, dash="dot"),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=n_grid, y=bas, mode="lines",
                                 line=dict(color="#90a4ae", width=1, dash="dot"),
                                 fill="tonexty", fillcolor=f"rgba(144,164,174,{opac})",
                                 name=f"Limites {lab}", hoverinfo="skip"))

    fig.add_hline(y=p0, line=dict(color="black", width=2),
                  annotation_text=f"reference {'theorique' if reference=='alpha' else 'observee'}"
                                  f" = {100*p0:.1f} %", annotation_position="right")
    if reference != "alpha":
        fig.add_hline(y=ALPHA, line=dict(color="#455a64", width=1.4, dash="dash"),
                      annotation_text=f"cible CQR = {100*ALPHA:.0f} %",
                      annotation_position="left")

    for statut, coul in couleurs.items():
        sub = g[g["statut"] == statut]
        if sub.empty:
            continue
        gwp_col = sub["gwp"] if "gwp" in sub.columns else np.full(len(sub), np.nan)
        fig.add_trace(go.Scatter(
            x=sub["n"], y=sub["taux"], mode="markers",
            marker=dict(size=9 + 15 * sub["n_ano"].rank(pct=True),
                        color=coul, opacity=0.85, line=dict(width=0.7, color="white")),
            name=f"{statut} ({len(sub)})",
            customdata=np.column_stack([sub[categorie], sub["n_ano"],
                                        sub["z_score"], gwp_col]),
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "Taux : %{y:.1%}  sur %{x} observations<br>"
                          "Anomalies : %{customdata[1]:.0f}<br>"
                          "Ecart standardise : %{customdata[2]:+.2f} sigma<br>"
                          f"{GWP_COL} : %{{customdata[3]:,.0f}}<extra></extra>"))

    fig.update_layout(
        title=f"Funnel plot — {categorie}"
              "<br><sup>Un point DANS l'entonnoir est compatible avec le hasard, "
              "quel que soit son taux apparent. Seuls les points au-dessus des "
              "courbes sont reellement aberrants</sup>",
        xaxis=dict(title="Nombre d'observations du segment", type="log"),
        yaxis=dict(title="Taux d'anomalie", tickformat=".0%",
                   range=[0, min(1.02, g["taux"].max() * 1.15 + 0.05)]),
        template="plotly_white", height=650, hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5))
    fig.show()

    aberrants = g[g["z_score"] > 1.96].sort_values("z_score", ascending=False)
    print(f"Segments hors entonnoir : {len(aberrants)} / {len(g)}")
    if len(aberrants):
        cols = [categorie, "n", "n_ano", "taux", "z_score"] + \
               (["gwp"] if "gwp" in g.columns else [])
        aff = aberrants[cols].copy()
        aff["taux"] = (100 * aff["taux"]).round(1).astype(str) + " %"
        aff["z_score"] = aff["z_score"].round(2)
        print(aff.head(15).to_string(index=False))
    return g


funnel = funnel_plot(expl, "Activity")








Bloc 6



# ================================================================
# BLOC C — ARBRE DE DETECTION DES SEGMENTS (icicle hierarchique)
#   Taille  : nombre d'anomalies
#   Couleur : taux d'anomalie du segment
# ================================================================

import plotly.express as px


def arbre_segments(expl, chemin=None, top_n_par_niveau=10):
    chemin = chemin or [c for c in ["Lob", "Partner", "Risk", "Activity"]
                        if c in expl.columns][:3]
    d = expl.dropna(subset=chemin).copy()
    for c in chemin:
        d[c] = d[c].astype(str)
        top = d[c].value_counts().head(top_n_par_niveau).index
        d[c] = np.where(d[c].isin(top), d[c], "Autres")

    agg = {"n_total": ("dans_intervalle", "size"), "n_ano": ("est_anomalie", "sum")}
    if GWP_COL in d.columns:
        agg["gwp"] = (GWP_COL, "sum")
    stats = d.groupby(chemin, observed=True).agg(**agg).reset_index()
    stats["taux"] = stats["n_ano"] / stats["n_total"]
    stats = stats[stats["n_ano"] > 0].copy()
    if "gwp" not in stats.columns:
        stats["gwp"] = np.nan

    fig = px.icicle(stats, path=chemin, values="n_ano",
                    color="taux", color_continuous_scale="Bluered",
                    range_color=[0, min(1.0, max(0.3, stats["taux"].quantile(0.9) * 1.2))],
                    custom_data=["n_total", "taux", "gwp"])
    fig.update_traces(
        texttemplate="%{label}<br>%{value} anomalies",
        hovertemplate="<b>%{label}</b><br>Anomalies : %{value} / %{customdata[0]}<br>"
                      "Taux : %{customdata[1]:.1%}<br>"
                      f"{GWP_COL} : %{{customdata[2]:,.0f}}<extra></extra>",
        tiling=dict(orientation="h"))
    fig.update_layout(
        title=f"Arbre de detection — {' › '.join(chemin)}"
              "<br><sup>Largeur = nombre d'anomalies | Couleur = taux d'anomalie</sup>",
        template="plotly_white", height=680,
        coloraxis_colorbar=dict(title="Taux", tickformat=".0%"))
    fig.show()
    return stats


arbre = arbre_segments(expl)




Bloc 7


# ================================================================
# BLOC D — PROPORTION D'UNITES PAR ETAT ET PAR GROUPE
#   Barres empilees 100 % : part couverte / hors intervalle par groupe
# ================================================================

def proportion_par_groupe(expl, categorie, n_min=5, top_n=25, trier="taux"):
    d = expl.dropna(subset=[categorie]).copy()
    d[categorie] = d[categorie].astype(str)
    g = (d.groupby(categorie, observed=True)
           .agg(n=("dans_intervalle", "size"), n_ano=("est_anomalie", "sum"))
           .reset_index())
    g = g[g["n"] >= n_min].copy()
    g["n_ok"] = g["n"] - g["n_ano"]
    g["taux"] = g["n_ano"] / g["n"]
    g = g.nlargest(top_n, "n_ano" if trier == "volume" else "taux")
    g = g.sort_values("taux").reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=g[categorie], x=g["n_ok"] / g["n"], orientation="h",
        name="Couverte", marker_color="#1769E0",
        customdata=np.column_stack([g["n_ok"], g["n"]]),
        hovertemplate="<b>%{y}</b><br>Couvertes : %{customdata[0]:.0f} / "
                      "%{customdata[1]:.0f}  (%{x:.1%})<extra></extra>"))
    fig.add_trace(go.Bar(
        y=g[categorie], x=g["taux"], orientation="h",
        name="Hors intervalle", marker_color="#E53935",
        customdata=np.column_stack([g["n_ano"], g["n"]]),
        hovertemplate="<b>%{y}</b><br>Anomalies : %{customdata[0]:.0f} / "
                      "%{customdata[1]:.0f}  (%{x:.1%})<extra></extra>"))

    fig.add_vline(x=ALPHA, line=dict(color="black", width=2, dash="dash"))
    fig.add_annotation(x=ALPHA, y=1.02, yref="paper", xanchor="left",
                       text=f" taux attendu ({100*ALPHA:.0f} %)",
                       showarrow=False, font=dict(size=10))

    fig.update_layout(
        barmode="stack",
        title=f"Proportion d'unites statistiques par etat — {categorie}"
              f"<br><sup>Segments d'au moins {n_min} observations, "
              "tries par taux d'anomalie croissant</sup>",
        xaxis=dict(title="Proportion", tickformat=".0%", range=[0, 1]),
        yaxis=dict(title=categorie, tickfont=dict(size=9)),
        template="plotly_white", height=max(420, 24 * len(g) + 190),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5))
    fig.show()
    return g


prop = proportion_par_groupe(expl, "Lob")




Boc 8


# ================================================================
# BLOC E — DECOMPOSITION SHAP
#   pip install shap  (si absent)
#
#   IMPORTANT : passez le modele qui a produit y_pred, ainsi que le
#   DataFrame de features correspondant. Chez vous il s'agit du modele
#   de prediction ponctuelle, PAS de pipeline_lo / pipeline_hi.
# ================================================================

def shap_anomalies(modele, X_test, anomalies_prio, expl, top_n=10,
                   feature_cols=None):
    """
    modele    : estimateur LightGBM entraine (ou pipeline avec .named_steps["model"])
    X_test    : DataFrame de features aligne sur expl (meme index / meme ordre)
    """
    try:
        import shap
    except ImportError:
        print("Installez shap :  pip install shap")
        return None

    mdl = getattr(getattr(modele, "named_steps", {}), "get", lambda k: None)("model") or modele
    feature_cols = feature_cols or list(X_test.columns)

    explainer = shap.TreeExplainer(mdl)
    sv = explainer.shap_values(X_test[feature_cols])
    sv = sv[0] if isinstance(sv, list) else sv

    # Vue globale : importance moyenne
    imp = pd.Series(np.abs(sv).mean(axis=0), index=feature_cols).sort_values()
    fig = go.Figure(go.Bar(x=imp.tail(20).values, y=imp.tail(20).index,
                           orientation="h", marker_color="#1769E0"))
    fig.update_layout(title="Importance SHAP moyenne (|valeur SHAP|)",
                      xaxis_title="Impact moyen sur la prediction",
                      template="plotly_white", height=560)
    fig.show()

    # Vue locale : decomposition des pires anomalies
    id_cols = [c for c in ID_COLS if c in anomalies_prio.columns]
    top = anomalies_prio.nlargest(top_n, "score_composite")
    cles = id_cols + [c for c in ["year", "quarter"] if c in expl.columns]
    idx_map = expl.reset_index().set_index(cles)["index"]

    for _, r in top.iterrows():
        cle = tuple(r[c] for c in cles)
        if cle not in idx_map.index:
            continue
        i = expl.index.get_loc(idx_map.loc[cle])
        contrib = pd.Series(sv[i], index=feature_cols)
        contrib = contrib.reindex(contrib.abs().sort_values().index).tail(12)
        nom = " | ".join(str(r[c]) for c in id_cols)
        fig = go.Figure(go.Bar(
            x=contrib.values, y=contrib.index, orientation="h",
            marker_color=np.where(contrib.values > 0, "#E53935", "#1769E0")))
        fig.update_layout(
            title=f"Decomposition SHAP — #{int(r['rank'])}  {nom}"
                  f"<br><sup>Observe {r['y_obs']:,.0f} | Predit {r['y_pred']:,.0f} | "
                  f"Rouge = pousse la prediction a la hausse</sup>",
            xaxis_title="Contribution a la prediction",
            template="plotly_white", height=440)
        fig.show()
    return sv


# Exemple d'appel — adaptez les deux premiers arguments a vos objets :
# shap_anomalies(votre_modele_point, expl[FEATURE_COLS], anomalies_prio, expl, top_n=5)






Bon àpar tire de ce niveau 

# ================================================================
# BLOC A — TABLEAU DE PRIORISATION (echelle individuelle)
# ================================================================
# ┌─────────────────── PARAMETRES A MODIFIER ───────────────────┐
TOP_N_TABLEAU  = 50      # Nombre de lignes affichees. Mettre len(anomalies_prio)
                         # pour tout voir. Au-dela de ~200 l'affichage ralentit.
STYLE_TABLEAU  = True    # True  = degrade de couleur sur la colonne Score
                         # False = DataFrame brut (utile pour .to_csv / .to_excel)
COL_SCORE_TRI  = "score_composite"   # Colonne de tri. Alternatives possibles :
                         # "A_ecart_borne" (gravite pure), "B_erreur_modele",
                         # GWP_COL (par exposition financiere seule)
# └──────────────────────────────────────────────────────────────┘

import numpy as np
import pandas as pd


def tableau_priorisation(anomalies_prio, top_n=TOP_N_TABLEAU,
                         style=STYLE_TABLEAU, col_tri=COL_SCORE_TRI):
    d = anomalies_prio.copy()
    id_cols = [c for c in ID_COLS if c in d.columns]
    d["IDGroup"] = d[id_cols].astype(str).agg(" | ".join, axis=1)

    t = pd.DataFrame({
        "Rang":          d["rank"].astype("Int64") if "rank" in d else range(1, len(d) + 1),
        "IDGroup":       d["IDGroup"],
        "Y_OBS":         d["y_obs"],
        "Y_Pred":        d["y_pred"],
        "CP_Lower":      d["borne_basse"],
        "CP_Upper":      d["borne_haute"],
        "dans_interval": d["dans_intervalle"],
        "Score":         d["score_composite"]})
    if GWP_COL in d.columns:
        t[GWP_COL] = d[GWP_COL]

    # CONDITION : le tri se fait sur col_tri s'il existe, sinon sur Score
    cle = col_tri if col_tri in d.columns else "score_composite"
    t = t.assign(_tri=d[cle].values).sort_values("_tri", ascending=False)
    t = t.drop(columns="_tri").head(top_n).reset_index(drop=True)

    if not style:
        return t
    try:
        return (t.style
                 .background_gradient(subset=["Score"], cmap="Reds")
                 .format({"Y_OBS": "{:,.0f}", "Y_Pred": "{:,.0f}",
                          "CP_Lower": "{:,.0f}", "CP_Upper": "{:,.0f}",
                          "Score": "{:,.4g}",
                          **({GWP_COL: "{:,.0f}"} if GWP_COL in t.columns else {})})
                 .set_caption(f"Tableau de priorisation — top {len(t)} sur "
                              f"{len(anomalies_prio)} anomalies"))
    except Exception:
        return t     # CONDITION : repli si le Styler n'est pas rendu par l'environnement


tableau_priorisation(anomalies_prio)

# Export si besoin :
# tableau_priorisation(anomalies_prio, top_n=len(anomalies_prio), style=False) \
#     .to_csv("tableau_priorisation.csv", index=False)


# ================================================================
# BLOC B — FUNNEL PLOT : quels segments sont REELLEMENT aberrants ?
# ================================================================
# ┌─────────────────── PARAMETRES A MODIFIER ───────────────────┐
CAT_FUNNEL       = "Activity"   # Colonne analysee. Essayer aussi : "Lob",
                                # "Partner", "Risk", "Companies"
N_MIN_FUNNEL     = 3            # Effectif minimum pour qu'un segment apparaisse.
                                # Monter a 10-20 pour ne garder que le fiable.
REFERENCE_FUNNEL = "observee"   # "observee" -> compare au taux global CONSTATE
                                #               (detecte les segments atypiques)
                                # "alpha"    -> compare a la cible THEORIQUE du CQR
                                #               (detecte les depassements de couverture)
Z_LIMITES        = [(1.96, "95 %"), (3.09, "99.8 %")]
                                # Seuils des entonnoirs. 1.96 = 5 % de faux positifs,
                                # 3.09 = 0.2 %. Ajouter (2.58, "99 %") si souhaite.
LOG_X_FUNNEL     = True         # False si vos effectifs sont peu etales
# └──────────────────────────────────────────────────────────────┘

import plotly.graph_objects as go


def funnel_plot(expl, categorie=CAT_FUNNEL, n_min=N_MIN_FUNNEL,
                reference=REFERENCE_FUNNEL, z_limites=Z_LIMITES,
                log_x=LOG_X_FUNNEL):
    d = expl.dropna(subset=[categorie]).copy()
    d[categorie] = d[categorie].astype(str)

    agg = {"n": ("dans_intervalle", "size"), "n_ano": ("est_anomalie", "sum")}
    if GWP_COL in d.columns:
        agg["gwp"] = (GWP_COL, "sum")
    g = d.groupby(categorie, observed=True).agg(**agg).reset_index()

    # CONDITION : on ecarte les segments trop petits (voir N_MIN_FUNNEL)
    g = g[g["n"] >= n_min].copy()
    if g.empty:
        print(f"Aucun segment avec au moins {n_min} observations.")
        return g
    g["taux"] = g["n_ano"] / g["n"]

    # CONDITION : reference theorique (ALPHA) ou empirique (taux global)
    p0 = ALPHA if reference == "alpha" else d["est_anomalie"].mean()

    n_grid = np.logspace(np.log10(max(g["n"].min(), 1)),
                         np.log10(g["n"].max() * 1.15), 250)
    limites = {}
    for z, lab in z_limites:
        se = np.sqrt(p0 * (1 - p0) / n_grid)
        limites[lab] = (np.clip(p0 - z * se, 0, 1), np.clip(p0 + z * se, 0, 1))

    se_seg = np.sqrt(p0 * (1 - p0) / g["n"])
    g["z_score"] = (g["taux"] - p0) / np.maximum(se_seg, 1e-12)

    # CONDITION : classement selon les seuils de Z_LIMITES
    z_haut = max(z for z, _ in z_limites)
    z_bas = min(z for z, _ in z_limites)
    g["statut"] = np.select(
        [g["z_score"] > z_haut, g["z_score"] > z_bas, g["z_score"] < -z_bas],
        ["Aberrant", "Eleve", "Anormalement bas"], default="Conforme")

    couleurs = {"Aberrant": "#b71c1c", "Eleve": "#ef6c00",
                "Conforme": "#1769E0", "Anormalement bas": "#2e7d32"}

    fig = go.Figure()
    for i, (lab, (bas, haut)) in enumerate(limites.items()):
        opac = 0.30 if i == 0 else 0.16     # premier entonnoir plus dense
        fig.add_trace(go.Scatter(x=n_grid, y=haut, mode="lines",
                                 line=dict(color="#90a4ae", width=1, dash="dot"),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=n_grid, y=bas, mode="lines",
                                 line=dict(color="#90a4ae", width=1, dash="dot"),
                                 fill="tonexty", fillcolor=f"rgba(144,164,174,{opac})",
                                 name=f"Limites {lab}", hoverinfo="skip"))

    fig.add_hline(y=p0, line=dict(color="black", width=2),
                  annotation_text=f"reference = {100*p0:.1f} %",
                  annotation_position="right")
    if reference != "alpha":
        fig.add_hline(y=ALPHA, line=dict(color="#455a64", width=1.4, dash="dash"),
                      annotation_text=f"cible CQR = {100*ALPHA:.0f} %",
                      annotation_position="left")

    for statut, coul in couleurs.items():
        sub = g[g["statut"] == statut]
        if sub.empty:
            continue
        gwp_col = sub["gwp"] if "gwp" in sub.columns else np.full(len(sub), np.nan)
        fig.add_trace(go.Scatter(
            x=sub["n"], y=sub["taux"], mode="markers",
            marker=dict(size=9 + 15 * sub["n_ano"].rank(pct=True),
                        color=coul, opacity=0.85, line=dict(width=0.7, color="white")),
            name=f"{statut} ({len(sub)})",
            customdata=np.column_stack([sub[categorie], sub["n_ano"],
                                        sub["z_score"], gwp_col]),
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "Taux : %{y:.1%}  sur %{x} observations<br>"
                          "Anomalies : %{customdata[1]:.0f}<br>"
                          "Ecart standardise : %{customdata[2]:+.2f} sigma<br>"
                          f"{GWP_COL} : %{{customdata[3]:,.0f}}<extra></extra>"))

    fig.update_layout(
        title=f"Funnel plot — {categorie}"
              "<br><sup>Un point DANS l'entonnoir est compatible avec le hasard, "
              "quel que soit son taux apparent</sup>",
        xaxis=dict(title="Nombre d'observations du segment",
                   type="log" if log_x else "linear"),
        yaxis=dict(title="Taux d'anomalie", tickformat=".0%",
                   range=[0, min(1.02, g["taux"].max() * 1.15 + 0.05)]),
        template="plotly_white", height=650, hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5))
    fig.show()

    aberrants = g[g["z_score"] > z_bas].sort_values("z_score", ascending=False)
    print(f"Segments hors entonnoir : {len(aberrants)} / {len(g)}")
    if len(aberrants):
        cols = [categorie, "n", "n_ano", "taux", "z_score"] + \
               (["gwp"] if "gwp" in g.columns else [])
        aff = aberrants[cols].copy()
        aff["taux"] = (100 * aff["taux"]).round(1).astype(str) + " %"
        aff["z_score"] = aff["z_score"].round(2)
        print(aff.head(15).to_string(index=False))
    return g


funnel = funnel_plot(expl)

# Pour analyser une autre dimension sans toucher au code :
# funnel_plot(expl, categorie="Lob", n_min=10)






# ================================================================
# BLOC C — ARBRE DE DETECTION DES SEGMENTS
# ================================================================
# ┌─────────────────── PARAMETRES A MODIFIER ───────────────────┐
CHEMIN_ARBRE      = None    # None = detection auto des 3 premieres colonnes
                            # dispo parmi Lob, Partner, Risk, Activity.
                            # Sinon imposer : ["Lob", "Risk", "Partner", "Activity"]
TOP_N_PAR_NIVEAU  = 10      # Modalites conservees par niveau ; le reste devient
                            # "Autres". Baisser a 5-6 si l'arbre est illisible.
ORIENT_ARBRE      = "h"     # "h" = branches horizontales, "v" = verticales
SEUIL_COULEUR     = None    # None = echelle auto (P90 du taux x 1.2).
                            # Imposer 0.5 pour figer l'echelle a 0-50 %.
# └──────────────────────────────────────────────────────────────┘

import plotly.express as px


def arbre_segments(expl, chemin=CHEMIN_ARBRE, top_n_par_niveau=TOP_N_PAR_NIVEAU,
                   orientation=ORIENT_ARBRE, seuil_couleur=SEUIL_COULEUR):
    # CONDITION : chemin auto si non impose
    chemin = chemin or [c for c in ["Lob", "Partner", "Risk", "Activity"]
                        if c in expl.columns][:3]
    d = expl.dropna(subset=chemin).copy()
    for c in chemin:
        d[c] = d[c].astype(str)
        # CONDITION : regroupement des modalites rares sous "Autres"
        top = d[c].value_counts().head(top_n_par_niveau).index
        d[c] = np.where(d[c].isin(top), d[c], "Autres")

    agg = {"n_total": ("dans_intervalle", "size"), "n_ano": ("est_anomalie", "sum")}
    if GWP_COL in d.columns:
        agg["gwp"] = (GWP_COL, "sum")
    stats = d.groupby(chemin, observed=True).agg(**agg).reset_index()
    stats["taux"] = stats["n_ano"] / stats["n_total"]

    # CONDITION : on n'affiche que les branches contenant au moins une anomalie
    stats = stats[stats["n_ano"] > 0].copy()
    if "gwp" not in stats.columns:
        stats["gwp"] = np.nan

    borne = seuil_couleur or min(1.0, max(0.3, stats["taux"].quantile(0.9) * 1.2))

    fig = px.icicle(stats, path=chemin, values="n_ano",
                    color="taux", color_continuous_scale="Bluered",
                    range_color=[0, borne],
                    custom_data=["n_total", "taux", "gwp"])
    fig.update_traces(
        texttemplate="%{label}<br>%{value} anomalies",
        hovertemplate="<b>%{label}</b><br>Anomalies : %{value} / %{customdata[0]}<br>"
                      "Taux : %{customdata[1]:.1%}<br>"
                      f"{GWP_COL} : %{{customdata[2]:,.0f}}<extra></extra>",
        tiling=dict(orientation=orientation))
    fig.update_layout(
        title=f"Arbre de detection — {' › '.join(chemin)}"
              "<br><sup>Largeur = nombre d'anomalies | Couleur = taux d'anomalie</sup>",
        template="plotly_white", height=680,
        coloraxis_colorbar=dict(title="Taux", tickformat=".0%"))
    fig.show()
    return stats


arbre = arbre_segments(expl)

# Pour imposer 4 niveaux :
# arbre_segments(expl, chemin=["Lob", "Partner", "Risk", "Activity"], top_n_par_niveau=6)




# ================================================================
# BLOC D — PROPORTION D'UNITES PAR ETAT ET PAR GROUPE
# ================================================================
# ┌─────────────────── PARAMETRES A MODIFIER ───────────────────┐
CAT_PROPORTION = "Lob"     # Colonne analysee : "Lob", "Partner", "Risk", "Activity"
N_MIN_PROP     = 5         # Effectif minimum du segment pour etre affiche
TOP_N_PROP     = 25        # Nombre de barres affichees
TRIER_PROP     = "taux"    # "taux"   -> garde les segments au plus fort taux
                           # "volume" -> garde ceux au plus grand NOMBRE d'anomalies
# └──────────────────────────────────────────────────────────────┘


def proportion_par_groupe(expl, categorie=CAT_PROPORTION, n_min=N_MIN_PROP,
                          top_n=TOP_N_PROP, trier=TRIER_PROP):
    d = expl.dropna(subset=[categorie]).copy()
    d[categorie] = d[categorie].astype(str)
    g = (d.groupby(categorie, observed=True)
           .agg(n=("dans_intervalle", "size"), n_ano=("est_anomalie", "sum"))
           .reset_index())

    # CONDITION : effectif minimum
    g = g[g["n"] >= n_min].copy()
    if g.empty:
        print(f"Aucun segment avec au moins {n_min} observations.")
        return g
    g["n_ok"] = g["n"] - g["n_ano"]
    g["taux"] = g["n_ano"] / g["n"]

    # CONDITION : selection selon TRIER_PROP
    g = g.nlargest(top_n, "n_ano" if trier == "volume" else "taux")
    g = g.sort_values("taux").reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=g[categorie], x=g["n_ok"] / g["n"], orientation="h",
        name="Couverte", marker_color="#1769E0",
        customdata=np.column_stack([g["n_ok"], g["n"]]),
        hovertemplate="<b>%{y}</b><br>Couvertes : %{customdata[0]:.0f} / "
                      "%{customdata[1]:.0f}  (%{x:.1%})<extra></extra>"))
    fig.add_trace(go.Bar(
        y=g[categorie], x=g["taux"], orientation="h",
        name="Hors intervalle", marker_color="#E53935",
        customdata=np.column_stack([g["n_ano"], g["n"]]),
        hovertemplate="<b>%{y}</b><br>Anomalies : %{customdata[0]:.0f} / "
                      "%{customdata[1]:.0f}  (%{x:.1%})<extra></extra>"))

    fig.add_vline(x=ALPHA, line=dict(color="black", width=2, dash="dash"))
    fig.add_annotation(x=ALPHA, y=1.02, yref="paper", xanchor="left",
                       text=f" taux attendu ({100*ALPHA:.0f} %)",
                       showarrow=False, font=dict(size=10))

    fig.update_layout(
        barmode="stack",
        title=f"Proportion d'unites statistiques par etat — {categorie}"
              f"<br><sup>Segments d'au moins {n_min} observations, "
              "tries par taux d'anomalie croissant</sup>",
        xaxis=dict(title="Proportion", tickformat=".0%", range=[0, 1]),
        yaxis=dict(title=categorie, tickfont=dict(size=9)),
        template="plotly_white", height=max(420, 24 * len(g) + 190),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5))
    fig.show()
    return g


prop = proportion_par_groupe(expl)

# Variante par volume d'anomalies plutot que par taux :
# proportion_par_groupe(expl, categorie="Partner", trier="volume", top_n=20)







# ================================================================
# BLOC E — DECOMPOSITION SHAP
#   pip install shap   (si absent)
# ================================================================
# ┌─────────────────── PARAMETRES A MODIFIER ───────────────────┐
MODELE_SHAP    = None    # OBLIGATOIRE : le modele qui a produit y_pred.
                         # PAS pipeline_lo / pipeline_hi (ce sont les quantiles).
                         # Exemple : MODELE_SHAP = best_model_pipeline
X_SHAP         = None    # OBLIGATOIRE : DataFrame de features, meme index que expl.
                         # Exemple : X_SHAP = expl[FEATURE_COLS]
TOP_N_SHAP     = 5       # Nombre d'anomalies detaillees individuellement.
                         # Chaque anomalie = 1 figure : ne pas monter trop haut.
N_FEATURES_GLOB = 20     # Barres affichees sur le graphique d'importance globale
N_FEATURES_LOC  = 12     # Barres affichees par decomposition individuelle
# └──────────────────────────────────────────────────────────────┘


def shap_anomalies(modele=MODELE_SHAP, X_test=X_SHAP, anomalies_prio=None, expl=None,
                   top_n=TOP_N_SHAP, feature_cols=None,
                   n_glob=N_FEATURES_GLOB, n_loc=N_FEATURES_LOC):
    # CONDITION : les deux objets modele / features doivent etre fournis
    if modele is None or X_test is None:
        print("Renseignez MODELE_SHAP et X_SHAP en tete de bloc avant d'executer.")
        return None
    try:
        import shap
    except ImportError:
        print("Installez shap :  pip install shap")
        return None

    # CONDITION : extraction du modele si c'est un Pipeline sklearn
    mdl = modele
    if hasattr(modele, "named_steps") and "model" in modele.named_steps:
        mdl = modele.named_steps["model"]

    feature_cols = feature_cols or list(X_test.columns)
    explainer = shap.TreeExplainer(mdl)
    sv = explainer.shap_values(X_test[feature_cols])
    sv = sv[0] if isinstance(sv, list) else sv

    # --- Vue globale ---
    imp = pd.Series(np.abs(sv).mean(axis=0), index=feature_cols).sort_values()
    fig = go.Figure(go.Bar(x=imp.tail(n_glob).values, y=imp.tail(n_glob).index,
                           orientation="h", marker_color="#1769E0"))
    fig.update_layout(title="Importance SHAP moyenne (|valeur SHAP|)",
                      xaxis_title="Impact moyen sur la prediction",
                      template="plotly_white", height=560)
    fig.show()

    # --- Vue locale sur les pires anomalies ---
    id_cols = [c for c in ID_COLS if c in anomalies_prio.columns]
    cles = id_cols + [c for c in ["year", "quarter"] if c in expl.columns]
    idx_map = expl.reset_index().set_index(cles)["index"]

    for _, r in anomalies_prio.nlargest(top_n, "score_composite").iterrows():
        cle = tuple(r[c] for c in cles)
        if cle not in idx_map.index:      # CONDITION : anomalie introuvable dans expl
            continue
        i = expl.index.get_loc(idx_map.loc[cle])
        contrib = pd.Series(sv[i], index=feature_cols)
        contrib = contrib.reindex(contrib.abs().sort_values().index).tail(n_loc)
        nom = " | ".join(str(r[c]) for c in id_cols)
        fig = go.Figure(go.Bar(
            x=contrib.values, y=contrib.index, orientation="h",
            marker_color=np.where(contrib.values > 0, "#E53935", "#1769E0")))
        fig.update_layout(
            title=f"Decomposition SHAP — #{int(r['rank'])}  {nom}"
                  f"<br><sup>Observe {r['y_obs']:,.0f} | Predit {r['y_pred']:,.0f} | "
                  "Rouge = pousse la prediction a la hausse</sup>",
            xaxis_title="Contribution a la prediction",
            template="plotly_white", height=440)
        fig.show()
    return sv


# Decommenter apres avoir renseigne MODELE_SHAP et X_SHAP :
# shap_anomalies(anomalies_prio=anomalies_prio, expl=expl)








# ================================================================
# BLOC F1 — TIME EVOLUTION : preparation
# ================================================================
# ┌─────────────────── PARAMETRES A MODIFIER ───────────────────┐
DF_HISTORIQUE   = df        # Base COMPLETE avec tous les trimestres.
                            # Doit contenir : ID_COLS, TARGET, time_idx, year, quarter
N_TRIMESTRES    = 10        # Profondeur d'historique affichee
TOP_N_VARS      = 5         # Nombre de variables importantes suivies
HIST_PREDICTION = None      # Optionnel : DataFrame de predictions historiques
                            # (ex. prediction_history du rolling). None = ignore.
                            # Doit contenir ID_COLS + year + quarter + Valeur_predite
MODELE_TE       = None      # Optionnel : modele pour identifier les variables via SHAP.
                            # Si None -> repli sur l'importance globale du modele,
                            # puis sur VARIABLES_MANUELLES.
X_TE            = None      # Features alignees sur expl (ex. expl[FEATURE_COLS])
VARIABLES_MANUELLES = []    # Dernier repli : imposer vous-meme les variables
                            # ex. ["GWP", "Nb_contrats", "Duree_moy"]
# └──────────────────────────────────────────────────────────────┘

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML


def _cles_unite(expl_df):
    """Colonnes qui definissent une unite statistique."""
    return [c for c in ID_COLS if c in expl_df.columns and c in DF_HISTORIQUE.columns]


def _historique_unite(cle_valeurs, cles, n_trim=N_TRIMESTRES):
    """Extrait l'historique d'une unite, tries par periode, limite aux n derniers."""
    d = DF_HISTORIQUE
    masque = np.logical_and.reduce(
        [d[c].astype(str).values == str(v) for c, v in zip(cles, cle_valeurs)])
    h = d[masque].sort_values("time_idx")
    # CONDITION : on ne garde que les N_TRIMESTRES derniers trimestres disponibles
    return h.tail(n_trim).copy()


def _libelle_periode(h):
    return h["year"].astype(int).astype(str) + "-T" + h["quarter"].astype(int).astype(str)


def _variables_importantes(cle_valeurs, cles, expl_df, n=TOP_N_VARS):
    """
    Ordre de priorite :
      1. SHAP local sur l'observation de test        (MODELE_TE + X_TE fournis)
      2. Importance globale du modele                 (MODELE_TE seul)
      3. VARIABLES_MANUELLES                          (repli)
    Retourne (liste_variables, methode_utilisee)
    """
    # --- 1. SHAP local ---
    if MODELE_TE is not None and X_TE is not None:
        try:
            import shap
            mdl = MODELE_TE
            if hasattr(mdl, "named_steps") and "model" in mdl.named_steps:
                mdl = mdl.named_steps["model"]
            masque = np.logical_and.reduce(
                [expl_df[c].astype(str).values == str(v)
                 for c, v in zip(cles, cle_valeurs)])
            pos = np.where(masque)[0]
            if len(pos):
                sv = shap.TreeExplainer(mdl).shap_values(X_TE.iloc[[pos[0]]])
                sv = sv[0] if isinstance(sv, list) else sv
                s = pd.Series(np.abs(np.asarray(sv).ravel()), index=X_TE.columns)
                cand = [v for v in s.sort_values(ascending=False).index
                        if v in DF_HISTORIQUE.columns
                        and pd.api.types.is_numeric_dtype(DF_HISTORIQUE[v])]
                if cand:
                    return cand[:n], "SHAP local"
        except Exception as e:
            print(f"SHAP local indisponible ({str(e)[:60]}) — repli sur l'importance globale.")

    # --- 2. Importance globale ---
    if MODELE_TE is not None:
        try:
            mdl = MODELE_TE
            if hasattr(mdl, "named_steps") and "model" in mdl.named_steps:
                mdl = mdl.named_steps["model"]
            noms = (list(X_TE.columns) if X_TE is not None
                    else list(getattr(mdl, "feature_name_", [])))
            imp = pd.Series(mdl.booster_.feature_importance("gain"), index=noms)
            cand = [v for v in imp.sort_values(ascending=False).index
                    if v in DF_HISTORIQUE.columns
                    and pd.api.types.is_numeric_dtype(DF_HISTORIQUE[v])]
            if cand:
                return cand[:n], "importance globale (gain)"
        except Exception:
            pass

    # --- 3. Repli manuel ---
    cand = [v for v in VARIABLES_MANUELLES if v in DF_HISTORIQUE.columns]
    return cand[:n], "liste manuelle"













# ================================================================
# BLOC F2 — FIGURE : evolution de la TARGET + des variables cles
#   Ligne 1  : TARGET sur N_TRIMESTRES, avec intervalle CP sur la
#              periode validee (point bleu si couvert, rouge sinon)
#   Lignes suivantes : les TOP_N_VARS variables les plus determinantes
#   Axe des abscisses partage : tout est aligne dans le temps
# ================================================================

COUL_HIST   = "#37474f"
COUL_VAR    = "#1565c0"
COUL_OK     = "#1769E0"
COUL_ANO    = "#E53935"
COUL_BANDE  = "rgba(70,110,230,0.22)"


def evolution_unite(cle_valeurs, expl_df, anomalies_df=None,
                    n_trim=N_TRIMESTRES, n_vars=TOP_N_VARS):
    cles = _cles_unite(expl_df)
    h = _historique_unite(cle_valeurs, cles, n_trim)
    if len(h) == 0:
        print("Aucun historique trouve pour cette unite.")
        return

    periodes = _libelle_periode(h).tolist()
    nom = " | ".join(f"{c}={v}" for c, v in zip(cles, cle_valeurs))

    # --- Ligne de test : bornes CP, prediction, statut ---
    m_test = np.logical_and.reduce(
        [expl_df[c].astype(str).values == str(v) for c, v in zip(cles, cle_valeurs)])
    test = expl_df[m_test]
    per_test, y_pred_t, lo_t, hi_t, obs_t, couvert = None, None, None, None, None, None
    if len(test):
        t = test.iloc[0]
        per_test = f"{int(t['year'])}-T{int(t['quarter'])}"
        y_pred_t, lo_t, hi_t = t["y_pred"], t["borne_basse"], t["borne_haute"]
        obs_t, couvert = t["y_obs"], bool(t["dans_intervalle"])

    variables, methode = _variables_importantes(cle_valeurs, cles, expl_df, n_vars)
    n_rows = 1 + len(variables)
    hauteurs = [0.40] + [0.60 / max(len(variables), 1)] * len(variables)

    titres = [f"TARGET — {TARGET}"] + [f"{v}   ({i+1}ᵉ variable)"
                                       for i, v in enumerate(variables)]
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.045, row_heights=hauteurs,
                        subplot_titles=titres)

    # ---------- Ligne 1 : TARGET ----------
    fig.add_trace(go.Scatter(
        x=periodes, y=h[TARGET], mode="lines+markers",
        line=dict(color=COUL_HIST, width=2),
        marker=dict(size=7, color=COUL_HIST),
        name="TARGET historique",
        hovertemplate="%{x}<br>" + TARGET + " : %{y:,.0f}<extra></extra>"),
        row=1, col=1)

    # CONDITION : predictions historiques seulement si HIST_PREDICTION est fourni
    if HIST_PREDICTION is not None:
        try:
            mh = np.logical_and.reduce(
                [HIST_PREDICTION[c].astype(str).values == str(v)
                 for c, v in zip(cles, cle_valeurs) if c in HIST_PREDICTION.columns])
            ph = HIST_PREDICTION[mh].copy()
            if len(ph):
                ph["per"] = (ph["year"].astype(int).astype(str) + "-T"
                             + ph["quarter"].astype(int).astype(str))
                ph = ph[ph["per"].isin(periodes)]
                col_p = ("Valeur_predite" if "Valeur_predite" in ph.columns else "y_pred")
                fig.add_trace(go.Scatter(
                    x=ph["per"], y=ph[col_p], mode="lines+markers",
                    line=dict(color="#78909c", width=1.6, dash="dash"),
                    marker=dict(size=5, symbol="diamond"),
                    name="Prediction historique",
                    hovertemplate="%{x}<br>Predit : %{y:,.0f}<extra></extra>"),
                    row=1, col=1)
        except Exception:
            pass

    # CONDITION : bloc CP affiche uniquement si l'unite figure dans le test
    if per_test is not None and per_test in periodes:
        fig.add_trace(go.Scatter(
            x=[per_test], y=[y_pred_t], mode="markers",
            error_y=dict(type="data", symmetric=False,
                         array=[hi_t - y_pred_t], arrayminus=[y_pred_t - lo_t],
                         color="#355CDE", thickness=2.4, width=8),
            marker=dict(symbol="diamond", size=11, color="white",
                        line=dict(color="#355CDE", width=2)),
            name=f"Prediction + intervalle CP ({100*(1-ALPHA):.0f} %)",
            hovertemplate=f"{per_test}<br>Predit : {y_pred_t:,.0f}<br>"
                          f"Intervalle : [{lo_t:,.0f} ; {hi_t:,.0f}]<extra></extra>"),
            row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[per_test], y=[obs_t], mode="markers",
            marker=dict(size=15, color=COUL_OK if couvert else COUL_ANO,
                        line=dict(width=1.4, color="white"),
                        symbol="circle" if couvert else "x"),
            name="Observe " + ("(couvert)" if couvert else "(HORS intervalle)"),
            hovertemplate=f"{per_test}<br>Observe : {obs_t:,.0f}<br>"
                          + ("Dans l'intervalle" if couvert else "HORS intervalle")
                          + "<extra></extra>"),
            row=1, col=1)

    # ---------- Lignes suivantes : variables importantes ----------
    for i, v in enumerate(variables, start=2):
        fig.add_trace(go.Scatter(
            x=periodes, y=h[v], mode="lines+markers",
            line=dict(color=COUL_VAR, width=1.8),
            marker=dict(size=6, color=COUL_VAR), showlegend=False,
            hovertemplate="%{x}<br>" + v + " : %{y:,.4g}<extra></extra>"),
            row=i, col=1)

    # ---------- Bande verticale sur la periode validee ----------
    if per_test is not None and per_test in periodes:
        k = periodes.index(per_test)
        fig.add_vrect(x0=k - 0.45, x1=k + 0.45, fillcolor=COUL_BANDE,
                      line_width=0, layer="below")
        fig.add_annotation(x=per_test, y=1.0, yref="paper", yanchor="bottom",
                           text="periode validee", showarrow=False,
                           font=dict(size=10, color="#355CDE"))

    rang = ""
    if anomalies_df is not None and len(test):
        ma = np.logical_and.reduce(
            [anomalies_df[c].astype(str).values == str(v)
             for c, v in zip(cles, cle_valeurs) if c in anomalies_df.columns])
        sa = anomalies_df[ma]
        if len(sa) and "rank" in sa.columns and pd.notna(sa.iloc[0]["rank"]):
            rang = f"  |  rang de priorite #{int(sa.iloc[0]['rank'])}"

    fig.update_xaxes(tickangle=45, row=n_rows, col=1)
    fig.update_yaxes(title_text=TARGET, row=1, col=1)
    fig.update_layout(
        title=dict(text=f"Evolution sur {len(h)} trimestres — {nom}{rang}"
                        f"<br><sup>Variables selectionnees par : {methode}</sup>",
                   font=dict(size=14)),
        template="plotly_white", height=260 + 150 * len(variables),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.04,
                    xanchor="center", x=0.5),
        margin=dict(t=145))
    fig.show()
    return h, variables
















# ================================================================
# BLOC F3 — SELECTEUR INTERACTIF
#   Choisissez une unite : sa trajectoire complete se redessine.
#   Figure mise a jour en place -> aucune accumulation.
# ================================================================
# ┌─────────────────── PARAMETRES A MODIFIER ───────────────────┐
N_UNITES_LISTE = 40      # Nombre d'anomalies proposees dans le menu
TRI_LISTE      = "score_composite"   # Critere de tri du menu.
                                     # Alternatives : "A_ecart_borne", GWP_COL
# └──────────────────────────────────────────────────────────────┘


def dashboard_evolution(anomalies_prio, expl_df, n_unites=N_UNITES_LISTE,
                        tri=TRI_LISTE):
    cles = _cles_unite(expl_df)
    if not cles:
        print("Aucune colonne d'identification commune entre df et expl.")
        return

    col_tri = tri if tri in anomalies_prio.columns else "score_composite"
    top = anomalies_prio.nlargest(min(n_unites, len(anomalies_prio)), col_tri)

    options = []
    for _, r in top.iterrows():
        vals = tuple(str(r[c]) for c in cles)
        rang = f"#{int(r['rank'])} " if "rank" in r.index and pd.notna(r["rank"]) else ""
        options.append((f"{rang}{' | '.join(vals)}", vals))

    if not options:
        print("Aucune anomalie a afficher.")
        return

    selecteur = widgets.Dropdown(options=options, value=options[0][1],
                                 description="Unite :",
                                 layout=widgets.Layout(width="760px"),
                                 style={"description_width": "60px"})
    zone = widgets.Output()

    def _maj(*_):
        with zone:
            clear_output(wait=True)
            evolution_unite(selecteur.value, expl_df, anomalies_prio)

    selecteur.observe(lambda c: _maj() if c["name"] == "value" else None, names="value")

    display(HTML(
        "<div style='font-family:system-ui,sans-serif;font-size:12.5px;color:#0d47a1;"
        "background:#e3f2fd;padding:9px 13px;border-radius:6px;margin-bottom:8px'>"
        f"<b>EVOLUTION TEMPORELLE</b> — {N_TRIMESTRES} derniers trimestres. "
        "La bande bleue marque la periode validee : point bleu = observation couverte, "
        "croix rouge = hors intervalle.</div>"))
    display(selecteur, zone)
    _maj()
    return selecteur


selecteur_evolution = dashboard_evolution(anomalies_prio, expl)




















# ================================================================
# BLOC F3 — SELECTEUR INTERACTIF
#   Choisissez une unite : sa trajectoire complete se redessine.
#   Figure mise a jour en place -> aucune accumulation.
# ================================================================
# ┌─────────────────── PARAMETRES A MODIFIER ───────────────────┐
N_UNITES_LISTE = 40      # Nombre d'anomalies proposees dans le menu
TRI_LISTE      = "score_composite"   # Critere de tri du menu.
                                     # Alternatives : "A_ecart_borne", GWP_COL
# └──────────────────────────────────────────────────────────────┘


def dashboard_evolution(anomalies_prio, expl_df, n_unites=N_UNITES_LISTE,
                        tri=TRI_LISTE):
    cles = _cles_unite(expl_df)
    if not cles:
        print("Aucune colonne d'identification commune entre df et expl.")
        return

    col_tri = tri if tri in anomalies_prio.columns else "score_composite"
    top = anomalies_prio.nlargest(min(n_unites, len(anomalies_prio)), col_tri)

    options = []
    for _, r in top.iterrows():
        vals = tuple(str(r[c]) for c in cles)
        rang = f"#{int(r['rank'])} " if "rank" in r.index and pd.notna(r["rank"]) else ""
        options.append((f"{rang}{' | '.join(vals)}", vals))

    if not options:
        print("Aucune anomalie a afficher.")
        return

    selecteur = widgets.Dropdown(options=options, value=options[0][1],
                                 description="Unite :",
                                 layout=widgets.Layout(width="760px"),
                                 style={"description_width": "60px"})
    zone = widgets.Output()

    def _maj(*_):
        with zone:
            clear_output(wait=True)
            evolution_unite(selecteur.value, expl_df, anomalies_prio)

    selecteur.observe(lambda c: _maj() if c["name"] == "value" else None, names="value")

    display(HTML(
        "<div style='font-family:system-ui,sans-serif;font-size:12.5px;color:#0d47a1;"
        "background:#e3f2fd;padding:9px 13px;border-radius:6px;margin-bottom:8px'>"
        f"<b>EVOLUTION TEMPORELLE</b> — {N_TRIMESTRES} derniers trimestres. "
        "La bande bleue marque la periode validee : point bleu = observation couverte, "
        "croix rouge = hors intervalle.</div>"))
    display(selecteur, zone)
    _maj()










dernier




# ================================================================
# DIAGNOSTIC DES ANOMALIES — anomalie reelle ou defaillance du modele ?
#
#   Principe : la couverture LOCALE du segment separe les deux cas.
#     - segment bien couvert + point isole hors intervalle -> anomalie
#     - segment mal couvert -> le modele echoue ici (epistemique)
# ================================================================
# ┌─────────────────── PARAMETRES A MODIFIER ───────────────────┐
MAILLE_DIAG   = "Lob"     # Maille de calcul de la couverture locale.
                          # Plus fine (Partner) = plus precis mais effectifs faibles.
N_MIN_DIAG    = 20        # Effectif minimum pour juger la couverture d'un segment.
                          # En dessous, verdict "indeterminable".
SEUIL_Z_FORT  = 2.0       # |z| au-dela duquel l'ecart est juge severe
MARGE_COUV    = 0.05      # Tolerance sur la couverture : un segment est declare
                          # defaillant si sa couverture est significativement
                          # inferieure a (1 - ALPHA - MARGE_COUV)
# └──────────────────────────────────────────────────────────────┘

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm


def _wilson_bas(k, n, conf=0.95):
    """Borne INFERIEURE de Wilson : couverture minimale plausible du segment."""
    z = norm.ppf(1 - (1 - conf) / 2)
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    marge = (z / den) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return np.clip(centre - marge, 0, 1)


def diagnostiquer_anomalies(expl, anomalies_prio, maille=MAILLE_DIAG,
                            n_min=N_MIN_DIAG, seuil_z=SEUIL_Z_FORT,
                            marge=MARGE_COUV):
    ex = expl.dropna(subset=["y_obs", "y_pred", "borne_basse", "borne_haute"]).copy()
    ex[maille] = ex[maille].astype(str)

    # --- 1. Couverture locale de chaque segment ---
    seg = (ex.groupby(maille, observed=True)
             .agg(n_seg=("dans_intervalle", "size"),
                  n_couv=("dans_intervalle", "sum"))
             .reset_index())
    seg["couverture"] = seg["n_couv"] / seg["n_seg"]
    seg["couv_ic_bas"] = _wilson_bas(seg["n_couv"], seg["n_seg"])
    # CONDITION : segment declare defaillant si sa couverture PLAUSIBLE MAXIMALE
    # reste sous la cible -> ce n'est pas du hasard, le modele echoue ici
    seg["modele_defaillant"] = (seg["couv_ic_bas"] < (1 - ALPHA - marge)) & \
                               (seg["n_seg"] >= n_min)
    seg["effectif_suffisant"] = seg["n_seg"] >= n_min

    # --- 2. Largeur relative mediane du segment (proxy d'incertitude du modele) ---
    ex["largeur_rel"] = ((ex["borne_haute"] - ex["borne_basse"])
                         / np.maximum(ex["y_pred"].abs(), 1e-9))
    med_larg = (ex.groupby(maille, observed=True)["largeur_rel"]
                  .median().rename("largeur_rel_med").reset_index())
    seg = seg.merge(med_larg, on=maille, how="left")

    # --- 3. Jointure sur les anomalies ---
    d = anomalies_prio.copy()
    d[maille] = d[maille].astype(str)
    centre = (d["borne_haute"] + d["borne_basse"]) / 2
    demi = np.maximum((d["borne_haute"] - d["borne_basse"]) / 2, 1e-9)
    d["z"] = (d["y_obs"] - centre) / demi
    d["abs_z"] = d["z"].abs()
    d["largeur_rel"] = ((d["borne_haute"] - d["borne_basse"])
                        / np.maximum(d["y_pred"].abs(), 1e-9))
    d = d.merge(seg, on=maille, how="left")

    # --- 4. Verdict ---
    # CONDITION 1 : effectif insuffisant -> on ne peut rien conclure
    # CONDITION 2 : segment defaillant -> l'ecart traduit l'echec du modele
    # CONDITION 3 : segment sain + ecart severe -> anomalie probable
    # CONDITION 4 : segment sain + ecart modere -> a surveiller
    d["verdict"] = np.select(
        [~d["effectif_suffisant"].fillna(False),
         d["modele_defaillant"].fillna(False),
         d["abs_z"] >= seuil_z,
         d["abs_z"] < seuil_z],
        ["Indeterminable (effectif insuffisant)",
         "Defaillance du modele (segment mal couvert)",
         "ANOMALIE PROBABLE (segment sain, ecart severe)",
         "Ecart limite (segment sain, ecart modere)"],
        default="Non classe")

    # --- 5. Signal complementaire : intervalle anormalement etroit ---
    # Un ecart severe alors que le modele se disait TRES sur de lui est le
    # signal le plus fort : il ne peut pas s'expliquer par l'incertitude.
    d["modele_confiant"] = d["largeur_rel"] < d["largeur_rel_med"]
    d["signal_fort"] = (d["verdict"].str.startswith("ANOMALIE")) & d["modele_confiant"]

    # --- 6. Synthese ---
    print("=" * 84)
    print(f"DIAGNOSTIC DES {len(d)} ANOMALIES — maille : {maille}")
    print("=" * 84)
    for v, n in d["verdict"].value_counts().items():
        print(f"  {v:<52} {n:>5}  ({100*n/len(d):>5.1f} %)")
    print("-" * 84)
    print(f"  dont SIGNAL FORT (modele confiant + ecart severe) : "
          f"{int(d['signal_fort'].sum())}")
    print("=" * 84)

    nb_def = int(seg["modele_defaillant"].sum())
    if nb_def:
        print(f"\n/!\\ {nb_def} segment(s) ou le modele est structurellement defaillant :")
        print(seg[seg["modele_defaillant"]]
              .sort_values("couverture")[[maille, "n_seg", "couverture", "couv_ic_bas"]]
              .head(10).to_string(index=False))
        print("    -> Dans ces segments, les ecarts traduisent un probleme de MODELE,")
        print("       pas de donnee. Ne les presentez pas comme des anomalies metier.")

    return d, seg


diag, segments = diagnostiquer_anomalies(expl, anomalies_prio)


















# ================================================================
# CARTE DE DIAGNOSTIC — visualiser la separation
#   X : couverture du segment (le modele fonctionne-t-il ici ?)
#   Y : |z| de l'anomalie (a quel point l'ecart est severe)
#   Le quadrant en haut a droite = anomalies defendables
# ================================================================

def carte_diagnostic(diag, seuil_z=SEUIL_Z_FORT, marge=MARGE_COUV):
    couleurs = {
        "ANOMALIE PROBABLE (segment sain, ecart severe)": "#c62828",
        "Ecart limite (segment sain, ecart modere)":       "#ef6c00",
        "Defaillance du modele (segment mal couvert)":     "#1565c0",
        "Indeterminable (effectif insuffisant)":           "#9e9e9e"}

    fig = go.Figure()
    fig.add_vrect(x0=0, x1=1 - ALPHA - marge, fillcolor="rgba(21,101,192,0.10)",
                  line_width=0, annotation_text="zone de defaillance du modele",
                  annotation_position="top left")
    fig.add_vline(x=1 - ALPHA, line=dict(color="black", width=2, dash="dash"),
                  annotation_text=f"couverture cible {100*(1-ALPHA):.0f} %")
    fig.add_hline(y=seuil_z, line=dict(color="#c62828", width=1.6, dash="dot"),
                  annotation_text=f"|z| = {seuil_z}")

    for v, c in couleurs.items():
        sub = diag[diag["verdict"] == v]
        if sub.empty:
            continue
        taille = (8 + 16 * sub["score_composite"].rank(pct=True)
                  if "score_composite" in sub.columns else 9)
        id_cols = [x for x in ID_COLS if x in sub.columns]
        fig.add_trace(go.Scatter(
            x=sub["couverture"], y=sub["abs_z"], mode="markers",
            marker=dict(size=taille, color=c, opacity=0.75,
                        line=dict(width=0.5, color="white"),
                        symbol=np.where(sub["signal_fort"], "star", "circle")),
            name=f"{v.split('(')[0].strip()} ({len(sub)})",
            customdata=np.column_stack([
                sub[id_cols].astype(str).agg(" | ".join, axis=1),
                sub["n_seg"], sub["largeur_rel"], sub["y_obs"], sub["y_pred"]]),
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "Couverture du segment : %{x:.1%} (n=%{customdata[1]:.0f})<br>"
                          "|z| : %{y:.2f}<br>"
                          "Largeur relative : %{customdata[2]:.1%}<br>"
                          "Observe : %{customdata[3]:,.0f} | "
                          "Predit : %{customdata[4]:,.0f}<extra></extra>"))

    fig.update_layout(
        title="Carte de diagnostic — anomalie reelle ou defaillance du modele ?"
              "<br><sup>A droite de la ligne noire : le modele est fiable dans ce segment, "
              "l'ecart est imputable a la donnee. Etoile = signal fort "
              "(modele confiant ET ecart severe)</sup>",
        xaxis=dict(title="Couverture du segment", tickformat=".0%", range=[0, 1.02]),
        yaxis=dict(title="Severite de l'ecart  |z|"),
        template="plotly_white", height=660, hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5))
    fig.show()


carte_diagnostic(diag)

















    return selecteur


selecteur_evolution = dashboard_evolution(anomalies_prio, expl)
