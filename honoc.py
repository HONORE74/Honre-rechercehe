import joblib, json, lightgbm as lgb, sklearn
from pathlib import Path
from datetime import datetime

CHEMIN_FINAL = Path(DOSSIER_ARTEFACTS) / "modele_final.joblib"

# Catégories exactes vues à l'entraînement — indispensable, sinon LightGBM
# réencode différemment au rechargement et les prédictions sont fausses.
categories = {c: list(X_tr[c].cat.categories)
              for c in X_tr.columns if str(X_tr[c].dtype) == "category"}

modele_final_paquet = {
    "modele":        modele_final,
    "features":      list(X_tr.columns),
    "categorielles": list(categories.keys()),
    "categories":    categories,
    "params":        params_finaux,
    "metriques_test": metriques("Test", y_te, pred_test),
    "target":        TARGET,
    "date":          datetime.now().isoformat(timespec="seconds"),
    "versions":      {"lightgbm": lgb.__version__,
                      "sklearn":  sklearn.__version__,
                      "joblib":   joblib.__version__},
}

joblib.dump(modele_final_paquet, CHEMIN_FINAL)
print(f"Sauvegardé : {CHEMIN_FINAL.resolve()}")
print(f"Taille     : {CHEMIN_FINAL.stat().st_size/1e6:.2f} Mo")
print(f"Variables  : {len(modele_final_paquet['features'])}")
print(f"MAE test   : {modele_final_paquet['metriques_test']['MAE']:,.0f}")
print(f"LightGBM   : {lgb.__version__}")

# --- Vérification immédiate : rechargement et comparaison ---
p = joblib.load(CHEMIN_FINAL)
Xv = X_te[p["features"]].copy()
for c in p["categorielles"]:
    Xv[c] = pd.Categorical(Xv[c].astype(str), categories=p["categories"][c])
ecart = np.abs(np.clip(p["modele"].predict(Xv), 0, None) - pred_test).max()
print(f"\nÉcart après rechargement : {ecart:.10f}  {'OK' if ecart < 1e-6 else '/!\\ ANOMALIE'}")























import joblib, lightgbm as lgb, pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime

CHEMIN = Path(DOSSIER_ARTEFACTS) / "modele_final.joblib"

modele_final = modele_apres
pred_test = predire(modele_final, X_te)

paquet = {
    "modele":        modele_final,
    "features":      list(X_tr.columns),
    "categorielles": [c for c in X_tr.columns if str(X_tr[c].dtype) == "category"],
    "categories":    {c: list(X_tr[c].cat.categories)
                      for c in X_tr.columns if str(X_tr[c].dtype) == "category"},
    "params":        params_finaux,
    "target":        TARGET,
    "id_cols":       [c for c in ID_COLS if c in df.columns],
    "X_test":        X_te.copy(),                                  # features du test
    "y_test":        np.asarray(y_te, float),                      # cible du test
    "infos_test":    df.loc[X_te.index, [c for c in ID_COLS + ["year", "quarter"]
                                         if c in df.columns]].copy(),
    "mae_test":      float(np.abs(pred_test - np.asarray(y_te, float)).mean()),
    "date":          datetime.now().isoformat(timespec="seconds"),
    "version_lgb":   lgb.__version__,
}

joblib.dump(paquet, CHEMIN)
print(f"Enregistre : {CHEMIN.resolve()}")
print(f"Taille     : {CHEMIN.stat().st_size/1e6:.2f} Mo")
print(f"Test       : {len(X_te):,} lignes | MAE {paquet['mae_test']:,.0f}")













import joblib, numpy as np, pandas as pd

paquet = joblib.load("artefacts_modele/modele_final.joblib")   # a adapter

# --- Preparation des features du test (types identiques a l'entrainement) ---
X = paquet["X_test"][paquet["features"]].copy()
for c in paquet["categorielles"]:
    X[c] = pd.Categorical(X[c].astype(str), categories=paquet["categories"][c])

# --- PREDICTION SUR LE TEST ---
predict_test = np.clip(paquet["modele"].predict(X), 0, None)

# --- Tableau observe / predit ---
res = paquet["infos_test"].reset_index(drop=True)
res["y_obs"]     = paquet["y_test"]
res["y_pred"]    = predict_test
res["ecart"]     = res.y_pred - res.y_obs
res["ecart_abs"] = res.ecart.abs()
res["ratio"]     = res.y_pred / res.y_obs.replace(0, np.nan)
res = res.sort_values("ecart_abs", ascending=False).reset_index(drop=True)

mae = res.ecart_abs.mean()
print(f"Modele du {paquet['date']} | {len(predict_test):,} predictions")
print(f"MAE recalculee : {mae:,.0f}   (enregistree : {paquet['mae_test']:,.0f})")
print(f"Controle       : {'OK' if abs(mae - paquet['mae_test']) < 1 else '/!\\ ECART'}")
print(f"Somme observee : {res.y_obs.sum():,.0f}")
print(f"Somme predite  : {res.y_pred.sum():,.0f}  (ratio {res.y_pred.sum()/res.y_obs.sum():.4f})\n")
print(res.head(20).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

res.to_csv("predictions_test.csv", index=False)



















AAAAAAAAAAAAAAAAAAAA



# ================================================================
# TABLEAU DE BORD — version revisee + granularite d'affichage
#   ① Couches du cercle          (empilement des anneaux)
#   ② Maille + valeur            (FILTRE : quelles anomalies retenir)
#   ③ Maille souhaitee           (AFFICHAGE : granularite des unites en bas)
#   Panneaux : cartes -> bar plot -> forest plot
# ================================================================
# ┌─────────────────── PARAMETRES A MODIFIER ───────────────────┐
TOP_N_PANNEAUX = 12       # Unites affichees dans le bar plot ET le forest plot
ECHELLE        = "Bluered"
LOG_FOREST     = True     # Echelle log sur l'axe des montants du forest plot
COL_BARPLOT    = "score_total"   # Grandeur des barres apres agregation.
                          # "score_total" = somme du groupe (defaut)
                          # "score_moyen" = gravite moyenne du groupe
N_SEGMENTS     = 4        # Nombre de selecteurs de granularite
# └──────────────────────────────────────────────────────────────┘

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

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
    return pd.DataFrame(lignes)


def _filtrer_maille(df, colonne, valeur):
    if not valeur:
        return df, f"{colonne} — vue generale"
    return df[df[colonne].astype(str) == str(valeur)], f"{colonne} = {valeur}"


def _agreger(sub, gran, maxlen=38):
    """
    Agrege les anomalies a la granularite choisie.

    Pour chaque groupe on retient :
      - score_total / score_moyen / n  (statistiques du groupe)
      - la PIRE anomalie du groupe et son intervalle CQR REEL
        -> aucune moyenne de bornes, aucune deformation
    Si gran == toutes les ID_COLS, chaque groupe contient une seule
    anomalie : le comportement est identique a l'affichage individuel.
    """
    if sub.empty:
        return sub.assign(_label="", score_total=np.nan,
                          score_moyen=np.nan, n_groupe=0)

    stats = (sub.groupby(gran, observed=True)
               .agg(score_total=("score_composite", "sum"),
                    score_moyen=("score_composite", "mean"),
                    n_groupe=("score_composite", "size"))
               .reset_index())

    # CONDITION : ligne representative = anomalie de plus fort score du groupe
    idx_pires = sub.groupby(gran, observed=True)["score_composite"].idxmax()
    pires = sub.loc[idx_pires].copy()

    out = pires.merge(stats, on=gran, how="left")
    out["_label"] = out[gran].astype(str).agg(" | ".join, axis=1).str.slice(0, maxlen)
    return out


def _hover_groupe(r, gran):
    ident = " | ".join(f"{c}={r[c]}" for c in gran)
    t = (f"<b>{ident}</b><br>"
         f"Anomalies du groupe : {int(r['n_groupe'])}<br>"
         f"Score cumule : {r['score_total']:.4g}<br>"
         f"Gravite moyenne : {r['score_moyen']:.4g}<br>"
         "─────────────<br>"
         f"<i>Pire anomalie du groupe :</i><br>"
         f"Observe    : {r['y_obs']:,.0f}<br>"
         f"Predit     : {r['y_pred']:,.0f}<br>"
         f"Intervalle : [{r['borne_basse']:,.0f} ; {r['borne_haute']:,.0f}]")
    if "rank" in r.index and pd.notna(r["rank"]):
        t += f"<br>Rang priorite : #{int(r['rank'])}"
    if GWP_COL in r.index and pd.notna(r[GWP_COL]):
        t += f"<br>{GWP_COL} : {r[GWP_COL]:,.0f}"
    return t


def _cartes(sub, dd_global, titre, sub_expl=None, gran=None):
    part = sub["score_composite"].sum() / max(dd_global["score_composite"].sum(), 1e-12)
    gwp = (f"{sub[GWP_COL].sum():,.0f}"
           if GWP_COL in sub.columns and sub[GWP_COL].notna().any() else "n/a")
    grav = f"{sub['score_composite'].mean():,.4g}" if len(sub) else "—"
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
              ("Score cumule", f"{sub['score_composite'].sum():,.3g}", "#1565c0"),
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


# ------------------- panneau 1 : bar plot

def _creer_fw_bar():
    fig = go.Figure(go.Bar(x=[], y=[], orientation="h", showlegend=False,
                           marker=dict(colorscale=ECHELLE, cmin=0, cmax=1,
                                       line=dict(width=0.5, color="white"))))
    fig.update_layout(template="plotly_white", height=520,
                      yaxis=dict(tickfont=dict(size=9)),
                      margin=dict(l=10, r=40, t=95, b=45))
    return go.FigureWidget(fig)


def _maj_fw_bar(fw, sub, titre, gran, top_n=TOP_N_PANNEAUX, col=COL_BARPLOT):
    if len(sub) == 0:
        with fw.batch_update():
            fw.data[0].x, fw.data[0].y = [], []
            fw.layout.title = dict(text=f"{titre}<br><sup>Aucune anomalie</sup>",
                                   font=dict(size=14))
            fw.layout.height = 260
        return

    agg = _agreger(sub, gran)
    cle = col if col in agg.columns else "score_total"
    top = agg.nlargest(min(top_n, len(agg)), cle).iloc[::-1]

    with fw.batch_update():
        fw.data[0].x = top[cle].tolist()
        fw.data[0].y = top["_label"].tolist()
        fw.data[0].marker.color = top[cle].rank(pct=True).tolist()
        fw.data[0].text = [_hover_groupe(r, gran) for _, r in top.iterrows()]
        fw.data[0].hovertemplate = "%{text}<extra></extra>"
        fw.layout.xaxis.title.text = ("Score cumule du groupe" if cle == "score_total"
                                      else "Gravite moyenne du groupe")
        fw.layout.title = dict(
            text=f"Top {len(top)} — {titre}"
                 f"<br><sup>Granularite : {' | '.join(gran)}</sup>",
            font=dict(size=14))
        fw.layout.height = max(360, 34 * len(top) + 160)


# ------------------- panneau 2 : forest plot

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
                      margin=dict(l=10, r=40, t=115, b=50))
    return go.FigureWidget(fig)


def _maj_fw_forest(fw, sub, titre, gran, top_n=TOP_N_PANNEAUX,
                   log_x=LOG_FOREST, col=COL_BARPLOT):
    if len(sub) == 0:
        with fw.batch_update():
            for t in fw.data:
                t.x, t.y = [], []
            fw.layout.title = dict(text=f"{titre}<br><sup>Aucune anomalie</sup>",
                                   font=dict(size=14))
            fw.layout.height = 260
        return

    agg = _agreger(sub, gran)
    cle = col if col in agg.columns else "score_total"
    # CONDITION : memes unites que le bar plot, meme ordre
    d = agg.nlargest(min(top_n, len(agg)), cle).iloc[::-1].reset_index(drop=True)

    y = list(range(len(d)))
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

    textes = [_hover_groupe(r, gran) for _, r in d.iterrows()]
    ticks = [f"{lab}" + (f"  ({int(n)})" if n > 1 else "")
             for lab, n in zip(d["_label"], d["n_groupe"])]

    multi = (d["n_groupe"] > 1).any()
    sous_titre = (f"Granularite : {' | '.join(gran)}"
                  + ("  —  chaque ligne montre la PIRE anomalie du groupe "
                     "(effectif entre parentheses)" if multi else ""))

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
                 f"<br><sup>{sous_titre}</sup>", font=dict(size=14))
        fw.layout.height = max(400, 40 * len(d) + 175)


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
    def_cercle = [prefs[0] if prefs else cols[0], None, None, None]
    niveaux = [widgets.Dropdown(
        options=[("— aucun —", None)] + [(c, c) for c in cols],
        value=def_cercle[i], description=f"Couche {i+1} :",
        layout=widgets.Layout(width="255px"),
        style={"description_width": "72px"}) for i in range(4)]

    fw_cercle = go.FigureWidget(data=[go.Sunburst(ids=[], labels=[], parents=[],
                                                  values=[], branchvalues="total")])
    fw_cercle.update_layout(template="plotly_white", height=620, margin=dict(t=110, b=20))

    # ---- ② Filtre : maille puis valeur ---------------------------------
    maille_defaut = prefs[0] if prefs else cols[0]
    sel_maille = widgets.Dropdown(
        options=[(c, c) for c in cols], value=maille_defaut,
        description="1 · Maille :", layout=widgets.Layout(width="330px"),
        style={"description_width": "90px"})
    sel_valeur = widgets.Dropdown(
        options=[("— vue generale —", VUE_GENERALE)], value=VUE_GENERALE,
        description="2 · Valeur :", layout=widgets.Layout(width="480px"),
        style={"description_width": "90px"})

    # ---- ③ Granularite d'affichage des unites --------------------------
    # CONDITION : par defaut, les 3 premieres colonnes disponibles.
    # Segment 1 est toujours renseigne -> la granularite n'est jamais vide.
    ordre_def = [c for c in ["Partner", "Companies", "Lob", "Activity", "Risk"]
                 if c in cols] or cols
    def_seg = [ordre_def[i] if i < min(3, len(ordre_def)) else None
               for i in range(N_SEGMENTS)]
    segments = [widgets.Dropdown(
        options=([(c, c) for c in cols] if i == 0
                 else [("— aucun —", None)] + [(c, c) for c in cols]),
        value=def_seg[i] or (cols[0] if i == 0 else None),
        description=f"Segment {i+1} :", layout=widgets.Layout(width="255px"),
        style={"description_width": "80px"}) for i in range(N_SEGMENTS)]

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

    def _granularite():
        vus, out = set(), []
        for w in segments:
            if w.value and w.value not in vus:
                out.append(w.value)
                vus.add(w.value)
        return out or [cols[0]]      # CONDITION : jamais vide

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
        gran = _granularite()
        sub, titre = _filtrer_maille(dd, colonne, valeur)
        sub_ex, _ = _filtrer_maille(ex, colonne, valeur)
        with z_cartes:
            clear_output(wait=True)
            display(_cartes(sub, dd, titre, sub_ex, gran))
        _maj_fw_bar(fw_bar, sub, titre, gran, top_n=top_n)
        _maj_fw_forest(fw_forest, sub, titre, gran, top_n=top_n)

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
    for w in segments:
        w.observe(lambda c: _maj_panneaux() if c["name"] == "value" else None,
                  names="value")

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
        "<b>② FILTRE</b> — quelles anomalies retenir. Maille (ligne 1) puis valeur "
        f"(ligne 2). Maille active par defaut : <b>{maille_defaut}</b>."
        + ("<br>Le clic sur un segment du cercle synchronise ces deux commandes."
           if clic else ""),
        fond="#e3f2fd", coul="#0d47a1"))
    display(sel_maille)
    display(sel_valeur)

    display(_bandeau(
        "<b>③ SELECTIONNEZ LA MAILLE SOUHAITEE</b> — granularite des unites affichees "
        "en bas des graphiques. Segment 1 est toujours actif ; ajoutez les segments 2 a 4 "
        "pour affiner. Une seule colonne (ex. Partner) regroupe toutes les anomalies "
        "de ce partenaire en une ligne.",
        fond="#f1f8e9", coul="#33691e"))
    display(widgets.HBox(segments[:2]), widgets.HBox(segments[2:]))

    display(z_cartes, fw_bar, fw_forest)

    _maj_cercle()
    _maj_valeurs()
    return {"cercle": fw_cercle, "bar": fw_bar, "forest": fw_forest,
            "maille": sel_maille, "valeur": sel_valeur, "segments": segments}


controles = dashboard_complet(anomalies_prio, expl)
















bbbbbbbbbbbbbbbbbbbbbbbb


# ================================================================
# TABLEAU DE BORD — version revisee + granularite d'affichage
#   ① Couches du cercle          (empilement des anneaux)
#   ② Maille + valeur            (FILTRE : quelles anomalies retenir)
#   ③ Maille souhaitee           (AFFICHAGE : granularite des unites en bas)
#   Panneaux : cartes -> bar plot -> forest plot
# ================================================================
# ┌─────────────────── PARAMETRES A MODIFIER ───────────────────┐
TOP_N_PANNEAUX    = 12       # Unites affichees dans le bar plot ET le forest plot
ECHELLE           = "Bluered"
LOG_FOREST        = True     # Echelle log sur l'axe des montants du forest plot
COL_BARPLOT       = "score_total"   # "score_total" = somme du groupe
                             # "score_moyen" = gravite moyenne du groupe
N_SEGMENTS        = 4        # Nombre de selecteurs de granularite
TEXTE_DANS_CERCLE = False    # False = AUCUN nom ecrit dans les segments du cercle
                             #         (tout reste lisible au survol)
                             # True  = libelle + pourcentage ecrits dedans
LABELS_AXE_BAS    = True     # True = libelles des unites sur l'axe des graphiques
                             #        du bas. False = axe nu, survol seul.
# └──────────────────────────────────────────────────────────────┘

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

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
    return pd.DataFrame(lignes)


def _filtrer_maille(df, colonne, valeur):
    if not valeur:
        return df, f"{colonne} — vue generale"
    return df[df[colonne].astype(str) == str(valeur)], f"{colonne} = {valeur}"


def _agreger(sub, gran, maxlen=38):
    """
    Agrege les anomalies a la granularite choisie.
    Pour chaque groupe : statistiques + la PIRE anomalie et son intervalle REEL
    (aucune moyenne de bornes, aucune deformation).
    """
    if sub.empty:
        return sub.assign(_label="", score_total=np.nan,
                          score_moyen=np.nan, n_groupe=0)

    stats = (sub.groupby(gran, observed=True)
               .agg(score_total=("score_composite", "sum"),
                    score_moyen=("score_composite", "mean"),
                    n_groupe=("score_composite", "size"))
               .reset_index())

    # CONDITION : ligne representative = anomalie de plus fort score du groupe
    idx_pires = sub.groupby(gran, observed=True)["score_composite"].idxmax()
    pires = sub.loc[idx_pires].copy()

    out = pires.merge(stats, on=gran, how="left")
    out["_label"] = out[gran].astype(str).agg(" | ".join, axis=1).str.slice(0, maxlen)
    return out


def _hover_groupe(r, gran):
    ident = " | ".join(f"{c}={r[c]}" for c in gran)
    t = (f"<b>{ident}</b><br>"
         f"Anomalies du groupe : {int(r['n_groupe'])}<br>"
         f"Score cumule : {r['score_total']:.4g}<br>"
         f"Gravite moyenne : {r['score_moyen']:.4g}<br>"
         "─────────────<br>"
         f"<i>Pire anomalie du groupe :</i><br>"
         f"Observe    : {r['y_obs']:,.0f}<br>"
         f"Predit     : {r['y_pred']:,.0f}<br>"
         f"Intervalle : [{r['borne_basse']:,.0f} ; {r['borne_haute']:,.0f}]")
    if "rank" in r.index and pd.notna(r["rank"]):
        t += f"<br>Rang priorite : #{int(r['rank'])}"
    if GWP_COL in r.index and pd.notna(r[GWP_COL]):
        t += f"<br>{GWP_COL} : {r[GWP_COL]:,.0f}"
    return t


def _cartes(sub, dd_global, titre, sub_expl=None, gran=None):
    part = sub["score_composite"].sum() / max(dd_global["score_composite"].sum(), 1e-12)
    gwp = (f"{sub[GWP_COL].sum():,.0f}"
           if GWP_COL in sub.columns and sub[GWP_COL].notna().any() else "n/a")
    grav = f"{sub['score_composite'].mean():,.4g}" if len(sub) else "—"
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
              ("Score cumule", f"{sub['score_composite'].sum():,.3g}", "#1565c0"),
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


# ------------------- panneau 1 : bar plot

def _creer_fw_bar():
    fig = go.Figure(go.Bar(x=[], y=[], orientation="h", showlegend=False,
                           marker=dict(colorscale=ECHELLE, cmin=0, cmax=1,
                                       line=dict(width=0.5, color="white"))))
    fig.update_layout(template="plotly_white", height=520,
                      yaxis=dict(tickfont=dict(size=9)),
                      margin=dict(l=10, r=40, t=95, b=45))
    return go.FigureWidget(fig)


def _maj_fw_bar(fw, sub, titre, gran, top_n=TOP_N_PANNEAUX, col=COL_BARPLOT):
    if len(sub) == 0:
        with fw.batch_update():
            fw.data[0].x, fw.data[0].y = [], []
            fw.layout.title = dict(text=f"{titre}<br><sup>Aucune anomalie</sup>",
                                   font=dict(size=14))
            fw.layout.height = 260
        return

    agg = _agreger(sub, gran)
    cle = col if col in agg.columns else "score_total"
    top = agg.nlargest(min(top_n, len(agg)), cle).iloc[::-1]

    with fw.batch_update():
        fw.data[0].x = top[cle].tolist()
        fw.data[0].y = top["_label"].tolist()
        fw.data[0].marker.color = top[cle].rank(pct=True).tolist()
        fw.data[0].text = [_hover_groupe(r, gran) for _, r in top.iterrows()]
        fw.data[0].hovertemplate = "%{text}<extra></extra>"
        # CONDITION : aucun texte ecrit sur les barres elles-memes
        fw.data[0].textposition = "none"
        fw.layout.xaxis.title.text = ("Score cumule du groupe" if cle == "score_total"
                                      else "Gravite moyenne du groupe")
        fw.layout.yaxis.showticklabels = LABELS_AXE_BAS
        fw.layout.title = dict(
            text=f"Top {len(top)} — {titre}"
                 f"<br><sup>Granularite : {' | '.join(gran)}</sup>",
            font=dict(size=14))
        fw.layout.height = max(360, 34 * len(top) + 160)


# ------------------- panneau 2 : forest plot

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
                      margin=dict(l=10, r=40, t=115, b=50))
    return go.FigureWidget(fig)


def _maj_fw_forest(fw, sub, titre, gran, top_n=TOP_N_PANNEAUX,
                   log_x=LOG_FOREST, col=COL_BARPLOT):
    if len(sub) == 0:
        with fw.batch_update():
            for t in fw.data:
                t.x, t.y = [], []
            fw.layout.title = dict(text=f"{titre}<br><sup>Aucune anomalie</sup>",
                                   font=dict(size=14))
            fw.layout.height = 260
        return

    agg = _agreger(sub, gran)
    cle = col if col in agg.columns else "score_total"
    d = agg.nlargest(min(top_n, len(agg)), cle).iloc[::-1].reset_index(drop=True)

    y = list(range(len(d)))
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

    textes = [_hover_groupe(r, gran) for _, r in d.iterrows()]
    ticks = [f"{lab}" + (f"  ({int(n)})" if n > 1 else "")
             for lab, n in zip(d["_label"], d["n_groupe"])]

    multi = (d["n_groupe"] > 1).any()
    sous_titre = (f"Granularite : {' | '.join(gran)}"
                  + ("  —  chaque ligne montre la PIRE anomalie du groupe "
                     "(effectif entre parentheses)" if multi else ""))

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
                               tickfont=dict(size=9),
                               showticklabels=LABELS_AXE_BAS)
        fw.layout.title = dict(
            text=f"Intervalle conforme, prediction et valeur observee — {titre}"
                 f"<br><sup>{sous_titre}</sup>", font=dict(size=14))
        fw.layout.height = max(400, 40 * len(d) + 175)


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
    def_cercle = [prefs[0] if prefs else cols[0], None, None, None]
    niveaux = [widgets.Dropdown(
        options=[("— aucun —", None)] + [(c, c) for c in cols],
        value=def_cercle[i], description=f"Couche {i+1} :",
        layout=widgets.Layout(width="255px"),
        style={"description_width": "72px"}) for i in range(4)]

    fw_cercle = go.FigureWidget(data=[go.Sunburst(ids=[], labels=[], parents=[],
                                                  values=[], branchvalues="total")])
    fw_cercle.update_layout(template="plotly_white", height=620, margin=dict(t=110, b=20))

    # ---- ② Filtre : maille puis valeur ---------------------------------
    maille_defaut = prefs[0] if prefs else cols[0]
    sel_maille = widgets.Dropdown(
        options=[(c, c) for c in cols], value=maille_defaut,
        description="1 · Maille :", layout=widgets.Layout(width="330px"),
        style={"description_width": "90px"})
    sel_valeur = widgets.Dropdown(
        options=[("— vue generale —", VUE_GENERALE)], value=VUE_GENERALE,
        description="2 · Valeur :", layout=widgets.Layout(width="480px"),
        style={"description_width": "90px"})

    # ---- ③ Granularite d'affichage des unites --------------------------
    ordre_def = [c for c in ["Partner", "Companies", "Lob", "Activity", "Risk"]
                 if c in cols] or cols
    def_seg = [ordre_def[i] if i < min(3, len(ordre_def)) else None
               for i in range(N_SEGMENTS)]
    segments = [widgets.Dropdown(
        options=([(c, c) for c in cols] if i == 0
                 else [("— aucun —", None)] + [(c, c) for c in cols]),
        value=def_seg[i] or (cols[0] if i == 0 else None),
        description=f"Segment {i+1} :", layout=widgets.Layout(width="255px"),
        style={"description_width": "80px"}) for i in range(N_SEGMENTS)]

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

    def _granularite():
        vus, out = set(), []
        for w in segments:
            if w.value and w.value not in vus:
                out.append(w.value)
                vus.add(w.value)
        return out or [cols[0]]      # CONDITION : jamais vide

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
            t.hovertext, t.hoverinfo = survol, "text"
            t.maxdepth = len(chemin)
            # CONDITION : texte dans les segments seulement si demande
            if TEXTE_DANS_CERCLE:
                t.text = [f"{100*v/score_global:.0f} %" for v in h["score_total"]]
                t.texttemplate = "%{label}<br>%{text}"
                t.textinfo = None
                t.insidetextorientation = "radial"
            else:
                t.text = None
                t.texttemplate = None
                t.textinfo = "none"
            t.marker = dict(colors=h["score_moyen"].tolist(), colorscale=ECHELLE,
                            cmin=0, cmax=cmax, line=dict(color="white", width=1.6),
                            colorbar=dict(title="Gravite<br>moyenne", thickness=16,
                                          len=0.7, tickformat=".2g"))
            fw_cercle.layout.title = dict(
                text=f"Repartition des {len(dd)} anomalies — {len(chemin)} couche(s) : "
                     f"{' › '.join(chemin)}"
                     "<br><sup>Taille = score cumule | Couleur = gravite moyenne "
                     "(bleu faible, rouge elevee) | survolez pour le detail</sup>",
                font=dict(size=15))

    def _maj_panneaux(*_):
        colonne, valeur = sel_maille.value, sel_valeur.value
        gran = _granularite()
        sub, titre = _filtrer_maille(dd, colonne, valeur)
        sub_ex, _ = _filtrer_maille(ex, colonne, valeur)
        with z_cartes:
            clear_output(wait=True)
            display(_cartes(sub, dd, titre, sub_ex, gran))
        _maj_fw_bar(fw_bar, sub, titre, gran, top_n=top_n)
        _maj_fw_forest(fw_forest, sub, titre, gran, top_n=top_n)

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
    for w in segments:
        w.observe(lambda c: _maj_panneaux() if c["name"] == "value" else None,
                  names="value")

    def _bandeau(txt, fond="#eceff1", coul="#37474f"):
        return HTML(f"<div style='font-family:system-ui,sans-serif;font-size:12.5px;"
                    f"color:{coul};background:{fond};padding:9px 13px;border-radius:6px;"
                    f"margin:14px 0 8px 0'>{txt}</div>")

    display(_bandeau(
        "<b>① COUCHES DU CERCLE</b> — chaque couche activee ajoute un anneau. "
        "Couleur = gravite moyenne des anomalies du segment. "
        "Survolez un segment pour son identite et ses statistiques."))
    display(widgets.HBox(niveaux[:2]), widgets.HBox(niveaux[2:]))
    display(fw_cercle)

    display(_bandeau(
        "<b>② FILTRE</b> — quelles anomalies retenir. Maille (ligne 1) puis valeur "
        f"(ligne 2). Maille active par defaut : <b>{maille_defaut}</b>."
        + ("<br>Le clic sur un segment du cercle synchronise ces deux commandes."
           if clic else ""),
        fond="#e3f2fd", coul="#0d47a1"))
    display(sel_maille)
    display(sel_valeur)

    display(_bandeau(
        "<b>③ SELECTIONNEZ LA MAILLE SOUHAITEE</b> — granularite des unites affichees "
        "en bas des graphiques. Segment 1 est toujours actif ; ajoutez les segments 2 a 4 "
        "pour affiner. Une seule colonne (ex. Partner) regroupe toutes les anomalies "
        "de ce partenaire en une ligne.",
        fond="#f1f8e9", coul="#33691e"))
    display(widgets.HBox(segments[:2]), widgets.HBox(segments[2:]))

    display(z_cartes, fw_bar, fw_forest)

    _maj_cercle()
    _maj_valeurs()
    return {"cercle": fw_cercle, "bar": fw_bar, "forest": fw_forest,
            "maille": sel_maille, "valeur": sel_valeur, "segments": segments}


controles = dashboard_complet(anomalies_prio, expl)












