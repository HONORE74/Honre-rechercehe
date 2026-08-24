for nom in ["df", "results_v2", "anomalies", "anomalies_prio", "expl",
            "ID_COLS", "TARGET", "ALPHA", "GWP_COL"]:
    print(f"  {'OK    ' if nom in globals() else 'ABSENT'}  {nom}")

if "results_v2" in globals():
    print(f"\nresults_v2 : {len(results_v2):,} lignes")
    print("ID_COLS presentes :", [c for c in ID_COLS if c in results_v2.columns])
    print("\nColonnes contenant PREMIUM / GWP / EARNED :")
    print([c for c in results_v2.columns
           if any(k in c.upper() for k in ("PREMIUM", "GWP", "EARNED"))])








import numpy as np, pandas as pd

# ┌─── PARAMETRE ───┐
GWP_COL = "EARNED_PREMIUM"     # ajustez si le BLOC 1 montre un autre nom
# └─────────────────┘

if GWP_COL not in results_v2.columns:
    cand = [c for c in results_v2.columns
            if any(k in c.upper() for k in ("PREMIUM", "GWP", "EARNED"))]
    raise KeyError(f"'{GWP_COL}' absente. Candidates : {cand}")

id_ok = [c for c in ID_COLS if c in results_v2.columns]
cols_ano = id_ok + ["year", "quarter", "y_obs", "y_pred",
                    "borne_basse", "borne_haute", "dans_intervalle", GWP_COL]
cols_ano += [c for c in ["time_idx", "groupe_largeur", "largeur_intervalle"]
             if c in results_v2.columns]
cols_ano = list(dict.fromkeys([c for c in cols_ano if c in results_v2.columns]))

anomalies = results_v2.loc[~results_v2["dans_intervalle"], cols_ano].copy()

anomalies["ecart_intervalle"] = np.where(
    anomalies["y_obs"] > anomalies["borne_haute"],
    anomalies["y_obs"] - anomalies["borne_haute"],
    anomalies["borne_basse"] - anomalies["y_obs"])
anomalies["sens"] = np.where(anomalies["y_obs"] > anomalies["borne_haute"],
                             "Hors Haut", "Hors Bas")
anomalies = anomalies.sort_values("ecart_intervalle", ascending=False).reset_index(drop=True)

print(f"{len(anomalies)} anomalies / {len(results_v2)} observations "
      f"({100*len(anomalies)/len(results_v2):.1f} %)")
print(f"Couverture : {100*results_v2['dans_intervalle'].mean():.1f} % "
      f"(cible {100*(1-ALPHA):.0f} %)")
print(anomalies["sens"].value_counts().to_string())
























borne_franchie = np.where(anomalies["y_obs"] > anomalies["borne_haute"],
                          anomalies["borne_haute"], anomalies["borne_basse"])

anomalies_prio = anomalies.copy()
anomalies_prio["A_ecart_borne"] = (np.abs(anomalies_prio["ecart_intervalle"])
                                   / np.maximum(np.abs(borne_franchie), 1e-6))
anomalies_prio["B_erreur_modele"] = (np.abs(anomalies_prio["y_obs"] - anomalies_prio["y_pred"])
                                     / np.maximum(np.abs(anomalies_prio["y_pred"]), 1e-6))
anomalies_prio["score_composite"] = (anomalies_prio["A_ecart_borne"]
                                     * anomalies_prio["B_erreur_modele"]
                                     * anomalies_prio[GWP_COL])

anomalies_prio = anomalies_prio.sort_values("score_composite", ascending=False).reset_index(drop=True)
anomalies_prio["rank"] = np.arange(1, len(anomalies_prio) + 1)

print(f"anomalies_prio : {len(anomalies_prio)} lignes")
print(f"Score : min {anomalies_prio.score_composite.min():,.4g} | "
      f"median {anomalies_prio.score_composite.median():,.4g} | "
      f"max {anomalies_prio.score_composite.max():,.4g}")
print("\nTop 5 :")
print(anomalies_prio[id_ok + ["y_obs", "y_pred", "A_ecart_borne",
                              "B_erreur_modele", GWP_COL, "score_composite"]]
      .head(5).to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
































cles = [c for c in ID_COLS if c in results_v2.columns and c in anomalies_prio.columns]
cles += [c for c in ["year", "quarter"]
         if c in results_v2.columns and c in anomalies_prio.columns]

expl = results_v2.merge(
    anomalies_prio[cles + ["score_composite", "rank",
                           "A_ecart_borne", "B_erreur_modele"]],
    on=cles, how="left")

centre = (expl["borne_haute"] + expl["borne_basse"]) / 2
demi   = np.maximum((expl["borne_haute"] - expl["borne_basse"]) / 2, 1e-9)
expl["z"]            = (expl["y_obs"] - centre) / demi
expl["largeur"]      = expl["borne_haute"] - expl["borne_basse"]
expl["largeur_rel"]  = expl["largeur"] / np.maximum(expl["y_pred"].abs(), 1e-9)
expl["est_anomalie"] = (~expl["dans_intervalle"]).astype(int)
expl["statut"]       = np.where(expl["dans_intervalle"], "Couverte", "Hors intervalle")
expl["identite"]     = expl[id_ok].astype(str).agg(" | ".join, axis=1)

print(f"expl : {len(expl):,} lignes | {int(expl.est_anomalie.sum()):,} anomalies")
print(f"Scores rattaches : {expl.score_composite.notna().sum():,} "
      f"(doit egaler {len(anomalies_prio)})")
print(f"ID disponibles : {id_ok}")





























# ┌─── PARAMETRES ───┐
TOP_N, LOG_X, HAUTEUR = 10, True, 980
# └──────────────────┘

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

_dd = anomalies_prio.dropna(subset=["score_composite"]).copy()
_cols = [c for c in ID_COLS if c in _dd.columns]
for c in _cols:
    _dd[c] = _dd[c].astype(str)


def construire_figure(couches, filtre_col, filtre_val, granularite,
                      top_n=TOP_N, log_x=LOG_X):
    sub = _dd if not filtre_val else _dd[_dd[filtre_col] == str(filtre_val)]
    titre_f = f"{filtre_col} — TOUS" if not filtre_val else f"{filtre_col} = {filtre_val}"
    fig = make_subplots(rows=2, cols=2, vertical_spacing=.13, horizontal_spacing=.10,
        specs=[[{"type": "domain"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("1. Repartition", "2. Top unites par score",
                        "3. Intervalles conformes", "4. Score vs exposition"))
    if len(sub) == 0:
        return fig.update_layout(title="Aucune anomalie pour cette selection",
                                 template="plotly_white", height=300)
    total = float(sub["score_composite"].sum()) or 1.0

    lignes = []
    for prof in range(1, len(couches) + 1):
        cc = couches[:prof]
        g = sub.groupby(cc, observed=True).agg(
            st=("score_composite", "sum"), sm=("score_composite", "mean"),
            n=("score_composite", "size")).reset_index()
        for _, r in g.iterrows():
            v = [str(r[c]) for c in cc]
            lignes.append(dict(id="/".join(v), label=v[-1],
                               parent="/".join(v[:-1]) if prof > 1 else "",
                               st=float(r.st), sm=float(r.sm), n=int(r.n)))
    h = pd.DataFrame(lignes)
    cmax = float(np.nanpercentile(h.sm, 95)) or float(h.sm.max()) or 1.0
    fig.add_trace(go.Sunburst(
        ids=h.id.tolist(), labels=h.label.tolist(), parents=h.parent.tolist(),
        values=h.st.tolist(), branchvalues="total", textinfo="none", hoverinfo="text",
        hovertext=[f"<b>{r.label}</b><br>Gravite moyenne : {r.sm:,.4g}<br>"
                   f"Anomalies : {r.n}<br>Score : {r.st:,.4g} "
                   f"({100*r.st/total:.1f} %)" for _, r in h.iterrows()],
        marker=dict(colors=h.sm.tolist(), colorscale="Bluered", cmin=0, cmax=cmax,
                    line=dict(color="white", width=1.4),
                    colorbar=dict(title="Gravite<br>moyenne", thickness=13,
                                  len=.38, x=.44, y=.79))), row=1, col=1)

    st = (sub.groupby(granularite, observed=True)
            .agg(score_total=("score_composite", "sum"),
                 score_moyen=("score_composite", "mean"),
                 n_groupe=("score_composite", "size")).reset_index())
    pires = sub.loc[sub.groupby(granularite, observed=True)["score_composite"].idxmax()]
    a = pires.merge(st, on=granularite, how="left")
    a["lab"] = a[granularite].astype(str).agg(" | ".join, axis=1).str.slice(0, 34)
    top = a.nlargest(min(top_n, len(a)), "score_total").iloc[::-1].reset_index(drop=True)

    fig.add_trace(go.Bar(
        x=top.score_total.tolist(), y=top.lab.tolist(), orientation="h",
        showlegend=False, textposition="none",
        marker=dict(color=top.score_total.rank(pct=True).tolist(),
                    colorscale="Bluered", cmin=0, cmax=1,
                    line=dict(color="white", width=.6)),
        text=[f"<b>{r.lab}</b><br>Anomalies : {int(r.n_groupe)}<br>"
              f"Score cumule : {r.score_total:,.4g}<br>"
              f"Gravite moyenne : {r.score_moyen:,.4g}" for _, r in top.iterrows()],
        hovertemplate="%{text}<extra></extra>"), row=1, col=2)

    yy = list(range(len(top)))
    lo, hi = top.borne_basse.astype(float).values, top.borne_haute.astype(float).values
    obs, pred = top.y_obs.astype(float).values, top.y_pred.astype(float).values
    ok_log = bool(log_x and (obs > 0).all() and (pred > 0).all() and (lo > 0).all())
    xb, yb, xo, yo = [], [], [], []
    for i, (o, l, hh) in enumerate(zip(obs, lo, hi)):
        xb += [l, hh, None]; yb += [i, i, None]
        xo += [hh if o > hh else l, o, None]; yo += [i, i, None]
    hov = [f"<b>{r.lab}</b><br>Observe : {r.y_obs:,.0f}<br>Predit : {r.y_pred:,.0f}<br>"
           f"Intervalle : [{r.borne_basse:,.0f} ; {r.borne_haute:,.0f}]"
           for _, r in top.iterrows()]
    fig.add_trace(go.Scatter(x=xb, y=yb, mode="lines", opacity=.3, hoverinfo="skip",
                             line=dict(color="#3a6bbf", width=9), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=xo, y=yo, mode="lines", hoverinfo="skip", showlegend=False,
                             line=dict(color="#c0392b", width=2, dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=pred, y=yy, mode="markers", text=hov, showlegend=False,
                             marker=dict(symbol="diamond", size=9, color="white",
                                         line=dict(color="black", width=1.4)),
                             hovertemplate="%{text}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=obs, y=yy, mode="markers", text=hov, showlegend=False,
                             marker=dict(size=12, color="#c0392b",
                                         line=dict(color="#7b241c", width=1.2)),
                             hovertemplate="%{text}<extra></extra>"), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=sub[GWP_COL].astype(float), y=sub.score_composite.astype(float),
        mode="markers", showlegend=False,
        marker=dict(size=7, color=sub.score_composite.rank(pct=True),
                    colorscale="Bluered", cmin=0, cmax=1, opacity=.7,
                    line=dict(color="white", width=.4)),
        text=sub[granularite].astype(str).agg(" | ".join, axis=1),
        hovertemplate="<b>%{text}</b><br>" + GWP_COL +
                      " : %{x:,.0f}<br>Score : %{y:,.4g}<extra></extra>"), row=2, col=2)

    fig.update_xaxes(title_text="Score cumule", row=1, col=2)
    fig.update_yaxes(tickfont=dict(size=8), row=1, col=2)
    fig.update_xaxes(title_text=TARGET + ("  (log)" if ok_log else ""),
                     type="log" if ok_log else "linear", row=2, col=1)
    fig.update_yaxes(tickmode="array", tickvals=yy,
                     ticktext=[f"{r.lab}" + (f"  ({int(r.n_groupe)})"
                                             if r.n_groupe > 1 else "")
                               for _, r in top.iterrows()],
                     tickfont=dict(size=8), row=2, col=1)
    fig.update_xaxes(title_text=GWP_COL, type="log", row=2, col=2)
    fig.update_yaxes(title_text="Score composite", type="log", row=2, col=2)
    fig.update_layout(
        title=f"<b>{len(sub)} anomalies — {titre_f}</b>"
              f"<br><sup>Anneaux : {' › '.join(couches)}  |  "
              f"Granularite : {' | '.join(granularite)}</sup>",
        template="plotly_white", height=HAUTEUR, showlegend=False, margin=dict(t=140))
    return fig


_prefs = [c for c in ["Lob", "Partner", "Companies", "Risk"] if c in _cols]
_gran0 = [c for c in ["Partner", "Companies", "Lob"] if c in _cols] or _cols[:1]

w_couche = [widgets.Dropdown(
    options=([(c, c) for c in _cols] if i == 0
             else [("— aucun —", None)] + [(c, c) for c in _cols]),
    value=(_prefs[0] if _prefs else _cols[0]) if i == 0
          else (_prefs[1] if i == 1 and len(_prefs) > 1 else None),
    description=f"Couche {i+1} :", layout=widgets.Layout(width="250px"),
    style={"description_width": "72px"}) for i in range(3)]

w_maille = widgets.Dropdown(options=[(c, c) for c in _cols],
    value=_prefs[0] if _prefs else _cols[0], description="1 · Maille :",
    layout=widgets.Layout(width="320px"), style={"description_width": "88px"})
w_valeur = widgets.Dropdown(options=[("— TOUS —", "")], value="",
    description="2 · Valeur :", layout=widgets.Layout(width="460px"),
    style={"description_width": "88px"})

w_seg = [widgets.Dropdown(
    options=([(c, c) for c in _cols] if i == 0
             else [("— aucun —", None)] + [(c, c) for c in _cols]),
    value=_gran0[i] if i < len(_gran0) else None,
    description=f"Segment {i+1} :", layout=widgets.Layout(width="250px"),
    style={"description_width": "80px"}) for i in range(3)]

zone = widgets.Output()
_verrou = {"on": False}


def _uniques(ws, defaut):
    vus, out = set(), []
    for w in ws:
        if w.value and w.value not in vus:
            out.append(w.value); vus.add(w.value)
    return out or [defaut]


def _redessiner(*_):
    if _verrou["on"]:
        return
    fig = construire_figure(_uniques(w_couche, _cols[0]), w_maille.value,
                            w_valeur.value, _uniques(w_seg, _cols[0]))
    with zone:
        clear_output(wait=True)
        fig.show()


def _maj_valeurs(*_):
    _verrou["on"] = True
    try:
        g = (_dd.groupby(w_maille.value, observed=True)["score_composite"]
               .agg(["size", "sum"]).reset_index().sort_values("sum", ascending=False))
        w_valeur.options = [("— TOUS —", "")] + [
            (f"{r[w_maille.value]}  ({int(r['size'])})", str(r[w_maille.value]))
            for _, r in g.iterrows()]
        w_valeur.value = ""
    finally:
        _verrou["on"] = False
    _redessiner()


for w in w_couche + w_seg + [w_valeur]:
    w.observe(lambda c: _redessiner() if c["name"] == "value" else None, names="value")
w_maille.observe(lambda c: _maj_valeurs() if c["name"] == "value" else None, names="value")


def _b(t, f, c):
    return HTML(f"<div style='font-family:system-ui,sans-serif;font-size:12.5px;"
                f"color:{c};background:{f};padding:8px 12px;border-radius:6px;"
                f"margin:12px 0 6px 0'>{t}</div>")

display(_b("<b>① ANNEAUX DU CERCLE</b> — panneau 1", "#eceff1", "#37474f"))
display(widgets.HBox(w_couche))
display(_b("<b>② FILTRE</b> — quelles anomalies retenir", "#e3f2fd", "#0d47a1"))
display(widgets.HBox([w_maille, w_valeur]))
display(_b("<b>③ MAILLE SOUHAITEE</b> — granularite des panneaux 2 et 3",
           "#f1f8e9", "#33691e"))
display(widgets.HBox(w_seg))
display(zone)

_maj_valeurs()






























