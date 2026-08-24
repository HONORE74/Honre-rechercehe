# ================================================================
# TABLEAU DE BORD — version compatible (sans FigureWidget)
#   ① Couches du cercle | ② Filtre | ③ Granularite d'affichage
# ================================================================
# ┌─────────────────── PARAMETRES A MODIFIER ───────────────────┐
TOP_N_PANNEAUX    = 12
ECHELLE           = "Bluered"
LOG_FOREST        = True
COL_BARPLOT       = "score_total"     # ou "score_moyen"
N_SEGMENTS        = 4
TEXTE_DANS_CERCLE = False
LABELS_AXE_BAS    = True
# └──────────────────────────────────────────────────────────────┘

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

VUE_GENERALE = ""


def _f(v, defaut=0.0):
    """Convertit en float natif ; NaN/Inf -> defaut. Evite les erreurs JS."""
    try:
        v = float(v)
        return v if np.isfinite(v) else defaut
    except Exception:
        return defaut


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
                           "profondeur": prof,
                           "score_total": _f(r["score_total"]),
                           "score_moyen": _f(r["score_moyen"]),
                           "score_max": _f(r["score_max"]),
                           "n": int(r["n"]), "gwp": _f(r.get("gwp", np.nan))})
    return pd.DataFrame(lignes)


def _filtrer_maille(df, colonne, valeur):
    if not valeur:
        return df, f"{colonne} — vue generale"
    return df[df[colonne].astype(str) == str(valeur)], f"{colonne} = {valeur}"


def _agreger(sub, gran, maxlen=38):
    if sub.empty:
        return sub.assign(_label="", score_total=np.nan, score_moyen=np.nan, n_groupe=0)
    stats = (sub.groupby(gran, observed=True)
               .agg(score_total=("score_composite", "sum"),
                    score_moyen=("score_composite", "mean"),
                    n_groupe=("score_composite", "size")).reset_index())
    idx = sub.groupby(gran, observed=True)["score_composite"].idxmax()
    out = sub.loc[idx].copy().merge(stats, on=gran, how="left")
    out["_label"] = out[gran].astype(str).agg(" | ".join, axis=1).str.slice(0, maxlen)
    return out


def _hover_groupe(r, gran):
    ident = " | ".join(f"{c}={r[c]}" for c in gran)
    t = (f"<b>{ident}</b><br>Anomalies du groupe : {int(r['n_groupe'])}<br>"
         f"Score cumule : {_f(r['score_total']):.4g}<br>"
         f"Gravite moyenne : {_f(r['score_moyen']):.4g}<br>─────────────<br>"
         f"<i>Pire anomalie :</i><br>Observe : {_f(r['y_obs']):,.0f}<br>"
         f"Predit : {_f(r['y_pred']):,.0f}<br>"
         f"Intervalle : [{_f(r['borne_basse']):,.0f} ; {_f(r['borne_haute']):,.0f}]")
    if "rank" in r.index and pd.notna(r["rank"]):
        t += f"<br>Rang priorite : #{int(r['rank'])}"
    if GWP_COL in r.index and pd.notna(r[GWP_COL]):
        t += f"<br>{GWP_COL} : {_f(r[GWP_COL]):,.0f}"
    return t


def _cartes(sub, dd_global, titre, sub_expl=None, gran=None):
    part = _f(sub["score_composite"].sum()) / max(_f(dd_global["score_composite"].sum()), 1e-12)
    gwp = (f"{_f(sub[GWP_COL].sum()):,.0f}"
           if GWP_COL in sub.columns and sub[GWP_COL].notna().any() else "n/a")
    grav = f"{_f(sub['score_composite'].mean()):,.4g}" if len(sub) else "—"
    n_grp = (sub.groupby(gran, observed=True).ngroups if gran and len(sub) else len(sub))
    pire = "—"
    if len(sub) and "rank" in sub.columns:
        rk = sub.loc[sub["score_composite"].idxmax(), "rank"]
        if pd.notna(rk):
            pire = f"#{int(rk)}"
    couv = (f"{100*sub_expl['dans_intervalle'].mean():.1f} %"
            if sub_expl is not None and len(sub_expl) else "n/a")

    cartes = [("Anomalies", f"{len(sub):,}", "#37474f"),
              ("Groupes affiches", f"{n_grp:,}", "#455a64"),
              ("Gravite moyenne", grav, "#c62828"),
              ("Score cumule", f"{_f(sub['score_composite'].sum()):,.3g}", "#1565c0"),
              ("Part du score global", f"{100*part:.1f} %", "#ad1457"),
              (f"{GWP_COL} total", gwp, "#2e7d32"),
              ("Couverture CQR", couv, "#00838f"),
              ("Pire anomalie", pire, "#6a1b9a")]
    blocs = "".join(
        f"<div style='flex:1;min-width:128px;background:#fff;border:1px solid #e0e0e0;"
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


# ------------------- figures : go.Figure classiques

def _fig_cercle(dd, chemin, score_global):
    h = _hierarchie(dd, chemin)
    sm = h["score_moyen"].replace([np.inf, -np.inf], np.nan).dropna()
    cmax = _f(np.nanpercentile(sm, 95) if len(sm) else 1.0, 1.0) or 1.0

    survol = [f"<b>{r['label']}</b><br>Gravite moyenne : {r['score_moyen']:.4g}<br>"
              f"─────────────<br>Anomalies : {r['n']}<br>"
              f"Score cumule : {r['score_total']:.4g} "
              f"({100*r['score_total']/max(score_global,1e-12):.1f} % du total)<br>"
              f"Pire anomalie : {r['score_max']:.4g}<br>{GWP_COL} : {r['gwp']:,.0f}"
              for _, r in h.iterrows()]

    trace = dict(ids=h["id"].tolist(), labels=h["label"].tolist(),
                 parents=h["parent"].tolist(), values=h["score_total"].tolist(),
                 branchvalues="total", hovertext=survol, hoverinfo="text",
                 maxdepth=len(chemin),
                 marker=dict(colors=h["score_moyen"].tolist(), colorscale=ECHELLE,
                             cmin=0, cmax=cmax, line=dict(color="white", width=1.6),
                             colorbar=dict(title="Gravite<br>moyenne", thickness=16,
                                           len=0.7, tickformat=".2g")))
    if TEXTE_DANS_CERCLE:
        trace["text"] = [f"{100*v/max(score_global,1e-12):.0f} %" for v in h["score_total"]]
        trace["texttemplate"] = "%{label}<br>%{text}"
        trace["insidetextorientation"] = "radial"
    else:
        trace["textinfo"] = "none"

    fig = go.Figure(go.Sunburst(**trace))
    fig.update_layout(
        title=dict(text=f"Repartition des {len(dd)} anomalies — {len(chemin)} couche(s) : "
                        f"{' › '.join(chemin)}"
                        "<br><sup>Taille = score cumule | Couleur = gravite moyenne | "
                        "survolez pour le detail</sup>", font=dict(size=15)),
        template="plotly_white", height=620, margin=dict(t=110, b=20))
    return fig


def _fig_bar(sub, titre, gran, top_n=TOP_N_PANNEAUX, col=COL_BARPLOT):
    if len(sub) == 0:
        return go.Figure().update_layout(
            title=f"{titre} — aucune anomalie", template="plotly_white", height=260)
    agg = _agreger(sub, gran)
    cle = col if col in agg.columns else "score_total"
    top = agg.nlargest(min(top_n, len(agg)), cle).iloc[::-1]

    fig = go.Figure(go.Bar(
        x=[_f(v) for v in top[cle]], y=top["_label"].tolist(), orientation="h",
        showlegend=False, textposition="none",
        marker=dict(color=top[cle].rank(pct=True).fillna(0.5).tolist(),
                    colorscale=ECHELLE, cmin=0, cmax=1,
                    line=dict(width=0.5, color="white")),
        text=[_hover_groupe(r, gran) for _, r in top.iterrows()],
        hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(
        title=dict(text=f"Top {len(top)} — {titre}"
                        f"<br><sup>Granularite : {' | '.join(gran)}</sup>", font=dict(size=14)),
        xaxis_title=("Score cumule du groupe" if cle == "score_total"
                     else "Gravite moyenne du groupe"),
        yaxis=dict(tickfont=dict(size=9), showticklabels=LABELS_AXE_BAS),
        template="plotly_white", height=max(360, 34 * len(top) + 160),
        margin=dict(l=10, r=40, t=95, b=45))
    return fig


def _fig_forest(sub, titre, gran, top_n=TOP_N_PANNEAUX, log_x=LOG_FOREST, col=COL_BARPLOT):
    if len(sub) == 0:
        return go.Figure().update_layout(
            title=f"{titre} — aucune anomalie", template="plotly_white", height=260)
    agg = _agreger(sub, gran)
    cle = col if col in agg.columns else "score_total"
    d = agg.nlargest(min(top_n, len(agg)), cle).iloc[::-1].reset_index(drop=True)

    y = list(range(len(d)))
    lo = np.array([_f(v) for v in d["borne_basse"]])
    hi = np.array([_f(v) for v in d["borne_haute"]])
    obs = np.array([_f(v) for v in d["y_obs"]])
    pred = np.array([_f(v) for v in d["y_pred"]])
    log_ok = bool(log_x and (obs > 0).all() and (pred > 0).all() and (lo > 0).all())

    xs_band, ys_band, xs_over, ys_over = [], [], [], []
    for yi, o, l, h in zip(y, obs, lo, hi):
        xs_band += [l, h, None]; ys_band += [yi, yi, None]
        cible = h if o > h else l
        xs_over += [cible, o, None]; ys_over += [yi, yi, None]

    textes = [_hover_groupe(r, gran) for _, r in d.iterrows()]
    ticks = [f"{lab}" + (f"  ({int(n)})" if n > 1 else "")
             for lab, n in zip(d["_label"], d["n_groupe"])]
    multi = bool((d["n_groupe"] > 1).any())

    fig = go.Figure()
    fig.add_scatter(x=xs_band, y=ys_band, mode="lines", opacity=0.3, hoverinfo="skip",
                    line=dict(color="#3a6bbf", width=10),
                    name=f"Intervalle conforme ({100*(1-ALPHA):.0f} %)")
    fig.add_scatter(x=xs_over, y=ys_over, mode="lines", showlegend=False, hoverinfo="skip",
                    line=dict(color="#c0392b", width=2, dash="dot"))
    fig.add_scatter(x=pred, y=y, mode="markers", name="Prediction", text=textes,
                    marker=dict(symbol="diamond", size=10, color="white",
                                line=dict(color="black", width=1.5)),
                    hovertemplate="%{text}<extra></extra>")
    fig.add_scatter(x=obs, y=y, mode="markers", name="Valeur comptabilisee", text=textes,
                    marker=dict(size=13, color="#c0392b",
                                line=dict(color="#7b241c", width=1.3)),
                    hovertemplate="%{text}<extra></extra>")
    fig.update_layout(
        title=dict(text=f"Intervalle conforme, prediction et valeur observee — {titre}"
                        f"<br><sup>Granularite : {' | '.join(gran)}"
                        + ("  —  chaque ligne montre la PIRE anomalie du groupe "
                           "(effectif entre parentheses)" if multi else "") + "</sup>",
                   font=dict(size=14)),
        xaxis=dict(title=TARGET + ("  (log)" if log_ok else ""),
                   type="log" if log_ok else "linear"),
        yaxis=dict(tickmode="array", tickvals=y, ticktext=ticks,
                   tickfont=dict(size=9), showticklabels=LABELS_AXE_BAS),
        template="plotly_white", height=max(400, 40 * len(d) + 175),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5),
        margin=dict(l=10, r=40, t=115, b=50))
    return fig


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
    score_global = _f(dd["score_composite"].sum(), 1.0)
    verrou = {"actif": False}

    prefs = [c for c in ["Lob", "Partner", "Companies", "Risk"] if c in cols]
    def_cercle = [prefs[0] if prefs else cols[0], None, None, None]
    niveaux = [widgets.Dropdown(
        options=[("— aucun —", None)] + [(c, c) for c in cols], value=def_cercle[i],
        description=f"Couche {i+1} :", layout=widgets.Layout(width="255px"),
        style={"description_width": "72px"}) for i in range(4)]

    maille_defaut = prefs[0] if prefs else cols[0]
    sel_maille = widgets.Dropdown(options=[(c, c) for c in cols], value=maille_defaut,
        description="1 · Maille :", layout=widgets.Layout(width="330px"),
        style={"description_width": "90px"})
    sel_valeur = widgets.Dropdown(options=[("— vue generale —", VUE_GENERALE)],
        value=VUE_GENERALE, description="2 · Valeur :",
        layout=widgets.Layout(width="480px"), style={"description_width": "90px"})

    ordre_def = [c for c in ["Partner", "Companies", "Lob", "Activity", "Risk"]
                 if c in cols] or cols
    def_seg = [ordre_def[i] if i < min(3, len(ordre_def)) else None for i in range(N_SEGMENTS)]
    segments = [widgets.Dropdown(
        options=([(c, c) for c in cols] if i == 0
                 else [("— aucun —", None)] + [(c, c) for c in cols]),
        value=def_seg[i] or (cols[0] if i == 0 else None),
        description=f"Segment {i+1} :", layout=widgets.Layout(width="255px"),
        style={"description_width": "80px"}) for i in range(N_SEGMENTS)]

    z_cercle, z_cartes = widgets.Output(), widgets.Output()
    z_bar, z_forest = widgets.Output(), widgets.Output()

    def _chemin():
        vus, out = set(), []
        for w in niveaux:
            if w.value and w.value not in vus:
                out.append(w.value); vus.add(w.value)
        return out

    def _granularite():
        vus, out = set(), []
        for w in segments:
            if w.value and w.value not in vus:
                out.append(w.value); vus.add(w.value)
        return out or [cols[0]]

    def _maj_cercle(*_):
        chemin = _chemin()
        with z_cercle:
            clear_output(wait=True)
            if not chemin:
                print("Activez au moins une couche.")
            else:
                _fig_cercle(dd, chemin, score_global).show()

    def _maj_panneaux(*_):
        gran = _granularite()
        sub, titre = _filtrer_maille(dd, sel_maille.value, sel_valeur.value)
        sub_ex, _ = _filtrer_maille(ex, sel_maille.value, sel_valeur.value)
        with z_cartes:
            clear_output(wait=True); display(_cartes(sub, dd, titre, sub_ex, gran))
        with z_bar:
            clear_output(wait=True); _fig_bar(sub, titre, gran, top_n).show()
        with z_forest:
            clear_output(wait=True); _fig_forest(sub, titre, gran, top_n).show()

    def _options_valeurs(colonne):
        g = (dd.groupby(colonne, observed=True)["score_composite"].agg(["size", "sum"])
               .reset_index().sort_values("sum", ascending=False))
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

    for w in niveaux:
        w.observe(lambda c: _maj_cercle() if c["name"] == "value" else None, names="value")
    sel_maille.observe(lambda c: _maj_valeurs() if c["name"] == "value" else None, names="value")
    sel_valeur.observe(lambda c: (_maj_panneaux() if not verrou["actif"] else None)
                       if c["name"] == "value" else None, names="value")
    for w in segments:
        w.observe(lambda c: _maj_panneaux() if c["name"] == "value" else None, names="value")

    def _b(txt, fond="#eceff1", coul="#37474f"):
        return HTML(f"<div style='font-family:system-ui,sans-serif;font-size:12.5px;"
                    f"color:{coul};background:{fond};padding:9px 13px;border-radius:6px;"
                    f"margin:14px 0 8px 0'>{txt}</div>")

    display(_b("<b>① COUCHES DU CERCLE</b> — chaque couche activee ajoute un anneau. "
               "Couleur = gravite moyenne. Survolez un segment pour le detail."))
    display(widgets.HBox(niveaux[:2]), widgets.HBox(niveaux[2:]), z_cercle)

    display(_b("<b>② FILTRE</b> — quelles anomalies retenir. Maille puis valeur. "
               f"Maille par defaut : <b>{maille_defaut}</b>.", "#e3f2fd", "#0d47a1"))
    display(sel_maille, sel_valeur)

    display(_b("<b>③ SELECTIONNEZ LA MAILLE SOUHAITEE</b> — granularite des unites "
               "affichees en bas. Segment 1 toujours actif ; ajoutez 2 a 4 pour affiner.",
               "#f1f8e9", "#33691e"))
    display(widgets.HBox(segments[:2]), widgets.HBox(segments[2:]))
    display(z_cartes, z_bar, z_forest)

    _maj_cercle()
    _maj_valeurs()
    return {"maille": sel_maille, "valeur": sel_valeur, "segments": segments,
            "niveaux": niveaux}


controles = dashboard_complet(anomalies_prio, expl)
