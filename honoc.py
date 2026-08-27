# ================================================================
# EVOLUTION RBNS_eop — graphique principal soigne
#   X : trimestres  |  Y : RBNS_eop
#   Reticule au survol : la valeur exacte s'affiche au croisement
# ================================================================
# ┌─────────────────── PARAMETRES A MODIFIER ───────────────────┐
RANG_UNITE   = 1        # 1 = anomalie la plus grave
N_PER        = 12       # trimestres d'historique (3 a 24)
VARS_SECOND  = 3        # variables secondaires sous le graphe (0 = aucune)
AFFICHER_VAL = True     # etiquettes chiffrees sur chaque point
# └──────────────────────────────────────────────────────────────┘

import numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ENCRE, ACCENT, OK  = "#141B34", "#FF5A5F", "#3D5A9E"
BLEU, GRILLE, GRIS = "#636EFA", "#EDF1F7", "#8A93A5"
DOUX = ["#6C8EBF", "#82B366", "#C08552", "#9673A6", "#5F9EA0", "#B85C7E"]

# --- unite selectionnee ---
_c = [c for c in ID_COLS if c in expl.columns and c in df.columns]
UNITE = tuple(str(anomalies_prio.iloc[RANG_UNITE - 1][x]) for x in _c)
nom = " · ".join(UNITE)

# --- historique ---
_m = np.logical_and.reduce([df[c].astype(str).values == v for c, v in zip(_c, UNITE)])
h = df[_m].sort_values("time_idx").tail(N_PER).copy()
if len(h) == 0:
    raise ValueError("Aucun historique pour cette unite.")
per = (h["year"].astype(int).astype(str) + "-T" + h["quarter"].astype(int).astype(str)).tolist()
val = h[TARGET].values.astype(float)

# --- contexte de la periode validee ---
_mt = np.logical_and.reduce([expl[c].astype(str).values == v for c, v in zip(_c, UNITE)])
_t = expl[_mt]
ctx = None
if len(_t):
    r = _t.iloc[0]
    ctx = dict(per=f"{int(r['year'])}-T{int(r['quarter'])}", pred=float(r["y_pred"]),
               lo=float(r["borne_basse"]), hi=float(r["borne_haute"]),
               obs=float(r["y_obs"]), couvert=bool(r["dans_intervalle"]))

# --- variables secondaires ---
secondaires = []
if VARS_SECOND > 0:
    num = [c for c in h.columns
           if c != TARGET and pd.api.types.is_numeric_dtype(h[c])
           and h[c].notna().all() and h[c].nunique() > 1
           and c not in ("time_idx", "year", "quarter")]
    if "MODELE_TE" in globals() and MODELE_TE is not None:
        try:
            mdl = MODELE_TE.named_steps["model"] if hasattr(MODELE_TE, "named_steps") else MODELE_TE
            imp = pd.Series(mdl.booster_.feature_importance("gain"), index=mdl.feature_name_)
            num = [v for v in imp.sort_values(ascending=False).index if v in num] or num
        except Exception:
            pass
    secondaires = num[:VARS_SECOND]

# ---------------------------------------------------------------- figure
n_rows = 1 + len(secondaires)
fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=.06,
                    row_heights=[.58] + [.42/max(len(secondaires),1)]*len(secondaires)
                                if secondaires else [1.0])

# Bande verticale sur la periode validee
if ctx and ctx["per"] in per:
    k = per.index(ctx["per"])
    for rr in range(1, n_rows + 1):
        fig.add_vrect(x0=k-.5, x1=k+.5, fillcolor="rgba(99,110,250,0.055)",
                      line_width=0, layer="below", row=rr, col=1)

# Aire sous la courbe, avec degrade si la version de plotly le permet
aire = dict(x=per, y=val, mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
            fill="tozeroy", showlegend=False, hoverinfo="skip")
try:
    fig.add_scatter(**aire, fillgradient=dict(type="vertical", colorscale=[
        [0, "rgba(99,110,250,0.02)"], [1, "rgba(99,110,250,0.30)"]]), row=1, col=1)
except (TypeError, ValueError):
    fig.add_scatter(**aire, fillcolor="rgba(99,110,250,0.14)", row=1, col=1)

# Intervalle conforme sur la periode validee
if ctx and ctx["per"] in per:
    fig.add_scatter(x=[ctx["per"], ctx["per"]], y=[ctx["lo"], ctx["hi"]], mode="lines",
                    line=dict(color=BLEU, width=16), opacity=.22,
                    name=f"Intervalle CP {100*(1-ALPHA):.0f} %",
                    hovertemplate=f"Intervalle<br>[{ctx['lo']:,.0f} ; {ctx['hi']:,.0f}]"
                                  "<extra></extra>", row=1, col=1)
    fig.add_scatter(x=[ctx["per"]], y=[ctx["pred"]], mode="markers",
                    marker=dict(size=10, symbol="diamond", color="white",
                                line=dict(color=ENCRE, width=1.8)),
                    name="Prediction",
                    hovertemplate=f"Prediction<br>{ctx['pred']:,.0f}<extra></extra>",
                    row=1, col=1)

# Courbe principale
fig.add_scatter(x=per, y=val, mode="lines+markers+text" if AFFICHER_VAL else "lines+markers",
                line=dict(color=ENCRE, width=2.8, shape="spline", smoothing=.55),
                marker=dict(size=9, color="white", line=dict(color=ENCRE, width=2.2)),
                text=[f"{v:,.0f}" for v in val] if AFFICHER_VAL else None,
                textposition="top center", textfont=dict(size=9.5, color=GRIS),
                name=TARGET,
                hovertemplate="<b>%{x}</b><br>" + TARGET + " : <b>%{y:,.0f}</b><extra></extra>",
                row=1, col=1)

# Halo sur la periode validee
if ctx and ctx["per"] in per:
    coul = ACCENT if not ctx["couvert"] else OK
    halo = "rgba(255,90,95,0.20)" if not ctx["couvert"] else "rgba(61,90,158,0.18)"
    for taille, c in ((38, halo), (24, halo)):
        fig.add_scatter(x=[ctx["per"]], y=[ctx["obs"]], mode="markers",
                        marker=dict(size=taille, color=c), showlegend=False,
                        hoverinfo="skip", row=1, col=1)
    fig.add_scatter(x=[ctx["per"]], y=[ctx["obs"]], mode="markers",
                    marker=dict(size=13, color=coul, line=dict(color="white", width=2.4)),
                    name="Hors intervalle" if not ctx["couvert"] else "Couvert",
                    hovertemplate=f"<b>{ctx['per']}</b><br>Observe : {ctx['obs']:,.0f}<br>"
                                  + ("HORS intervalle" if not ctx["couvert"] else "Couvert")
                                  + "<extra></extra>", row=1, col=1)

# Variables secondaires
for i, v in enumerate(secondaires, start=2):
    c = DOUX[(i-2) % len(DOUX)]
    vv = h[v].values.astype(float)
    fig.add_scatter(x=per, y=vv, mode="lines+markers",
                    line=dict(color=c, width=2.2, shape="spline", smoothing=.55),
                    marker=dict(size=6, color="white", line=dict(color=c, width=1.8)),
                    name=v, showlegend=False,
                    hovertemplate=f"<b>%{{x}}</b><br>{v} : <b>%{{y:,.4g}}</b><extra></extra>",
                    row=i, col=1)
    fig.add_annotation(xref="paper", x=1.005, y=vv[-1], xanchor="left",
                       text=f"<b>{v}</b>", showarrow=False,
                       font=dict(size=10, color=c), row=i, col=1)

# ---------------------------------------------------------------- mise en forme
fig.update_xaxes(showgrid=False, showspikes=True, spikemode="across",
                 spikethickness=1.2, spikedash="dot", spikecolor=GRIS,
                 tickfont=dict(size=11), linecolor=GRILLE)
fig.update_yaxes(gridcolor=GRILLE, zeroline=False, showspikes=True,
                 spikemode="across", spikethickness=1.2, spikedash="dot",
                 spikecolor=GRIS, tickformat=",.0f", tickfont=dict(size=11))
fig.update_yaxes(title_text=f"<b>{TARGET}</b>", title_font=dict(size=12), row=1, col=1)
fig.update_xaxes(title_text="<b>Trimestre</b>", title_font=dict(size=12),
                 row=n_rows, col=1)

delta = 100*(val[-1] - val[0]) / abs(val[0]) if val[0] else np.nan
fleche = "▲" if delta >= 0 else "▼"

fig.update_layout(
    title=dict(text=f"<b style='font-size:19px;color:{ENCRE}'>{nom}</b>"
                    f"<br><span style='font-size:12px;color:{GRIS}'>"
                    f"{TARGET} · {len(h)} trimestres · "
                    f"{per[0]} → {per[-1]} · "
                    f"<span style='color:{ACCENT if delta<0 else OK}'>{fleche} "
                    f"{abs(delta):.1f} %</span> sur la periode</span>",
               x=.015, xanchor="left", y=.96),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="white", bordercolor=GRILLE, font=dict(size=12.5,
                    family="Inter, system-ui, sans-serif", color=ENCRE), align="left"),
    template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color=GRIS),
    height=430 + 130*len(secondaires),
    legend=dict(orientation="h", y=1.04, x=1, xanchor="right",
                bgcolor="rgba(255,255,255,0)", font=dict(size=11)),
    margin=dict(l=80, r=110, t=120, b=60))
fig.show()

print(f"Rang #{RANG_UNITE} · {nom}")
print(f"Min {val.min():,.0f} | Median {np.median(val):,.0f} | Max {val.max():,.0f}")
if ctx:
    print(f"Periode validee {ctx['per']} : observe {ctx['obs']:,.0f} | "
          f"predit {ctx['pred']:,.0f} | intervalle [{ctx['lo']:,.0f} ; {ctx['hi']:,.0f}]"
          f" -> {'COUVERT' if ctx['couvert'] else 'HORS INTERVALLE'}")








# ================================================================
# BLOC F1 — PALETTE ET HELPERS
# ================================================================
# ┌─────────────────── PARAMETRES A MODIFIER ───────────────────┐
DF_HISTORIQUE   = df          # base complete avec tous les trimestres
N_PERIODES      = 10          # nombre de trimestres affiches par defaut
TOP_N_VARS      = 5           # nombre de variables suivies par defaut
MODELE_TE       = None        # modele pour SHAP local (optionnel)
X_TE            = None        # features alignees sur expl (optionnel)
VARIABLES_MANUELLES = []      # repli si ni SHAP ni importance disponibles
# └──────────────────────────────────────────────────────────────┘

import numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PAL = dict(
    encre   = "#141B34",
    bande   = "rgba(99,110,250,0.16)",
    bordure = "rgba(99,110,250,0.55)",
    ok      = "#3D5A9E",
    halo    = "rgba(255,90,95,0.22)",
    ano     = "#FF5A5F",
    doux    = ["#6C8EBF", "#82B366", "#C08552", "#9673A6", "#5F9EA0", "#B85C7E"],
    grille  = "#EDF1F7",
    texte   = "#8A93A5",
    aire    = "rgba(0,0,0,0.035)",
    fond    = "#FFFFFF")

MISE_EN_FORME = dict(
    template="plotly_white",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color=PAL["texte"]),
    paper_bgcolor=PAL["fond"], plot_bgcolor=PAL["fond"],
    margin=dict(l=70, r=100, t=110, b=55))


def _cles_unite(expl_df):
    return [c for c in ID_COLS if c in expl_df.columns and c in DF_HISTORIQUE.columns]


def _historique_unite(cle_valeurs, cles, n=None):
    n = n or N_PERIODES
    d = DF_HISTORIQUE
    m = np.logical_and.reduce([d[c].astype(str).values == str(v)
                               for c, v in zip(cles, cle_valeurs)])
    return d[m].sort_values("time_idx").tail(n).copy()


def _periodes(h):
    return (h["year"].astype(int).astype(str) + "-T"
            + h["quarter"].astype(int).astype(str)).tolist()


def _variables_importantes(cle_valeurs, cles, expl_df, n=None):
    """SHAP local -> importance globale -> liste manuelle."""
    n = TOP_N_VARS if n is None else n
    if n == 0:
        return [], "aucune"

    if MODELE_TE is not None and X_TE is not None:
        try:
            import shap
            mdl = MODELE_TE.named_steps["model"] if hasattr(MODELE_TE, "named_steps") \
                  else MODELE_TE
            m = np.logical_and.reduce([expl_df[c].astype(str).values == str(v)
                                       for c, v in zip(cles, cle_valeurs)])
            pos = np.where(m)[0]
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
            print(f"SHAP indisponible ({str(e)[:50]}) -> importance globale")

    if MODELE_TE is not None:
        try:
            mdl = MODELE_TE.named_steps["model"] if hasattr(MODELE_TE, "named_steps") \
                  else MODELE_TE
            noms = list(X_TE.columns) if X_TE is not None else list(mdl.feature_name_)
            imp = pd.Series(mdl.booster_.feature_importance("gain"), index=noms)
            cand = [v for v in imp.sort_values(ascending=False).index
                    if v in DF_HISTORIQUE.columns
                    and pd.api.types.is_numeric_dtype(DF_HISTORIQUE[v])]
            if cand:
                return cand[:n], "importance globale"
        except Exception:
            pass

    manuelles = [v for v in VARIABLES_MANUELLES if v in DF_HISTORIQUE.columns]
    if manuelles:
        return manuelles[:n], "liste manuelle"

    auto = [c for c in DF_HISTORIQUE.columns
            if c != TARGET and pd.api.types.is_numeric_dtype(DF_HISTORIQUE[c])
            and c not in ("time_idx", "year", "quarter")]
    return auto[:n], "colonnes numeriques (defaut)"


def _contexte_test(cle_valeurs, cles, expl_df):
    """Bornes CP, prediction et statut sur la periode validee."""
    m = np.logical_and.reduce([expl_df[c].astype(str).values == str(v)
                               for c, v in zip(cles, cle_valeurs)])
    t = expl_df[m]
    if not len(t):
        return None
    r = t.iloc[0]
    return dict(periode=f"{int(r['year'])}-T{int(r['quarter'])}",
                pred=float(r["y_pred"]), lo=float(r["borne_basse"]),
                hi=float(r["borne_haute"]), obs=float(r["y_obs"]),
                couvert=bool(r["dans_intervalle"]))


print("Definies :", ["_cles_unite", "_historique_unite", "_periodes",
                     "_variables_importantes", "_contexte_test"])
















# ================================================================
# BLOC F2 — CINQ STYLES DE VISUALISATION
#   evolution_unite() ne retourne RIEN quand elle affiche : pas de doublon.
# ================================================================

def _fan(h, per, ctx, variables, nom, methode):
    """STYLE 1 — bande douce, ligne d'encre, halo lumineux."""
    n_rows = 1 + len(variables)
    hauteurs = [.48] + [.52/len(variables)]*len(variables) if variables else [1.0]
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                        vertical_spacing=.055, row_heights=hauteurs)

    if ctx and ctx["periode"] in per:
        k = per.index(ctx["periode"])
        for r in range(1, n_rows + 1):
            fig.add_vrect(x0=k-.5, x1=k+.5, fillcolor="rgba(99,110,250,0.055)",
                          line_width=0, layer="below", row=r, col=1)
        fig.add_scatter(x=[per[k], per[k]], y=[ctx["lo"], ctx["hi"]], mode="lines",
                        line=dict(color=PAL["bordure"], width=15), opacity=.28,
                        name=f"Intervalle CP {100*(1-ALPHA):.0f} %",
                        hovertemplate=f"[{ctx['lo']:,.0f} ; {ctx['hi']:,.0f}]<extra></extra>",
                        row=1, col=1)
        fig.add_scatter(x=[per[k]], y=[ctx["pred"]], mode="markers",
                        marker=dict(size=10, symbol="diamond", color="white",
                                    line=dict(color=PAL["encre"], width=1.8)),
                        name="Prediction",
                        hovertemplate=f"Prediction {ctx['pred']:,.0f}<extra></extra>",
                        row=1, col=1)

    fig.add_scatter(x=per, y=h[TARGET], mode="lines", line=dict(width=0),
                    fill="tozeroy", fillcolor="rgba(99,110,250,0.12)",
                    showlegend=False, hoverinfo="skip", row=1, col=1)
    fig.add_scatter(x=per, y=h[TARGET], mode="lines+markers",
                    line=dict(color=PAL["encre"], width=2.8, shape="spline", smoothing=.55),
                    marker=dict(size=8, color="white",
                                line=dict(color=PAL["encre"], width=2.2)),
                    name=TARGET,
                    hovertemplate="<b>%{x}</b><br>" + TARGET +
                                  " : <b>%{y:,.0f}</b><extra></extra>", row=1, col=1)

    if ctx and ctx["periode"] in per:
        c = PAL["ano"] if not ctx["couvert"] else PAL["ok"]
        halo = PAL["halo"] if not ctx["couvert"] else "rgba(61,90,158,0.18)"
        for taille in (36, 24):
            fig.add_scatter(x=[ctx["periode"]], y=[ctx["obs"]], mode="markers",
                            marker=dict(size=taille, color=halo), showlegend=False,
                            hoverinfo="skip", row=1, col=1)
        fig.add_scatter(x=[ctx["periode"]], y=[ctx["obs"]], mode="markers",
                        marker=dict(size=12, color=c, line=dict(color="white", width=2.4)),
                        name="Hors intervalle" if not ctx["couvert"] else "Couvert",
                        hovertemplate=f"Observe {ctx['obs']:,.0f}<extra></extra>",
                        row=1, col=1)

    for i, v in enumerate(variables, start=2):
        c = PAL["doux"][(i-2) % len(PAL["doux"])]
        fig.add_scatter(x=per, y=h[v], mode="lines",
                        line=dict(color=c, width=2.2, shape="spline", smoothing=.55),
                        fill="tozeroy", fillcolor=PAL["aire"], showlegend=False,
                        hovertemplate=f"<b>%{{x}}</b><br>{v} : <b>%{{y:,.4g}}</b>"
                                      "<extra></extra>", row=i, col=1)
        fig.add_annotation(xref="paper", x=1.005, y=h[v].iloc[-1], xanchor="left",
                           text=f"<b>{v}</b><br>{h[v].iloc[-1]:,.4g}", showarrow=False,
                           align="left", font=dict(size=10, color=c), row=i, col=1)

    fig.update_yaxes(gridcolor=PAL["grille"], zeroline=False, tickformat=",.0f")
    fig.update_xaxes(showgrid=False)
    fig.update_layout(
        title=dict(text=f"<b style='font-size:18px;color:{PAL['encre']}'>{nom}</b>"
                        f"<br><span style='font-size:11.5px'>{len(h)} trimestres · "
                        f"variables via {methode}</span>", x=.015, xanchor="left"),
        height=300 + 110*len(variables), hovermode="x unified",
        hoverlabel=dict(bgcolor="white", bordercolor=PAL["grille"],
                        font=dict(size=12.5, color=PAL["encre"])),
        legend=dict(orientation="h", y=1.03, x=1, xanchor="right"), **MISE_EN_FORME)
    return fig


def _sparkline(h, per, ctx, variables, nom, methode):
    """STYLE 2 — mini-courbes empilees, valeur actuelle a droite."""
    series = [(TARGET, PAL["encre"])] + [
        (v, PAL["doux"][i % len(PAL["doux"])]) for i, v in enumerate(variables)]
    fig = make_subplots(rows=len(series), cols=1, shared_xaxes=True, vertical_spacing=.035)

    for i, (v, c) in enumerate(series, start=1):
        vals = h[v].values.astype(float)
        fig.add_scatter(x=per, y=vals, mode="lines",
                        line=dict(color=c, width=2.4, shape="spline", smoothing=.65),
                        fill="tozeroy", fillcolor=PAL["aire"], showlegend=False,
                        row=i, col=1,
                        hovertemplate=f"<b>%{{x}}</b><br>{v} : <b>%{{y:,.4g}}</b>"
                                      "<extra></extra>")
        fig.add_scatter(x=[per[-1]], y=[vals[-1]], mode="markers",
                        marker=dict(size=8, color=c), showlegend=False,
                        hoverinfo="skip", row=i, col=1)
        fig.add_annotation(xref="paper", x=1.012, y=vals[-1], xanchor="left",
                           text=f"<b>{vals[-1]:,.4g}</b>"
                                f"<br><span style='font-size:9px'>{v}</span>",
                           showarrow=False, align="left",
                           font=dict(size=13, color=c), row=i, col=1)
        if i == 1 and ctx and ctx["periode"] in per:
            c2 = PAL["ano"] if not ctx["couvert"] else PAL["ok"]
            halo = PAL["halo"] if not ctx["couvert"] else "rgba(61,90,158,0.18)"
            fig.add_scatter(x=[ctx["periode"]], y=[ctx["obs"]], mode="markers",
                            marker=dict(size=24, color=halo), showlegend=False,
                            hoverinfo="skip", row=1, col=1)
            fig.add_scatter(x=[ctx["periode"]], y=[ctx["obs"]], mode="markers",
                            marker=dict(size=10, color=c2,
                                        line=dict(color="white", width=2)),
                            showlegend=False, row=1, col=1,
                            hovertemplate=f"Observe {ctx['obs']:,.0f}<extra></extra>")

    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_xaxes(showticklabels=True, row=len(series), col=1)
    fig.update_layout(
        title=dict(text=f"<b style='font-size:18px;color:{PAL['encre']}'>{nom}</b>"
                        f"<br><span style='font-size:11.5px'>{len(h)} trimestres · "
                        f"variables via {methode}</span>", x=.015, xanchor="left"),
        height=105*len(series) + 140, hovermode="x unified",
        hoverlabel=dict(bgcolor="white", bordercolor=PAL["grille"],
                        font=dict(size=12.5, color=PAL["encre"])), **MISE_EN_FORME)
    return fig


def _horizon(h, per, ctx, variables, nom, methode):
    """STYLE 3 — bandes d'intensite repliees, tres compact."""
    series = [TARGET] + list(variables)
    fig = make_subplots(rows=len(series), cols=1, shared_xaxes=True, vertical_spacing=.018)
    tons = ["rgba(99,110,250,0.20)", "rgba(99,110,250,0.42)",
            "rgba(99,110,250,0.68)", "rgba(255,90,95,0.55)"]

    for i, v in enumerate(series, start=1):
        vals = h[v].values.astype(float)
        z = (vals - np.median(vals)) / (np.std(vals) or 1.0)
        for k, ton in enumerate(tons):
            bas, haut = k*.8, (k+1)*.8
            fig.add_scatter(x=per, y=np.clip(np.abs(z), bas, haut) - bas, mode="lines",
                            line=dict(width=0), fill="tozeroy", fillcolor=ton,
                            showlegend=False, hoverinfo="skip", row=i, col=1)
        fig.add_scatter(x=per, y=np.abs(z), mode="lines",
                        line=dict(color="rgba(0,0,0,0)", width=0), showlegend=False,
                        hovertemplate=f"<b>%{{x}}</b><br>{v}<br>ecart : %{{y:.2f}} sigma"
                                      "<extra></extra>", row=i, col=1)
        fig.add_annotation(xref="paper", x=-.012, y=.4, xanchor="right",
                           text=f"<b>{v}</b>", showarrow=False,
                           font=dict(size=10, color=PAL["texte"]), row=i, col=1)

    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, range=[0, .85])
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_xaxes(showticklabels=True, row=len(series), col=1)
    fig.update_layout(
        title=dict(text=f"<b style='font-size:18px;color:{PAL['encre']}'>{nom}</b> "
                        "<span style='font-size:12px'>— vue horizon</span>"
                        f"<br><span style='font-size:11.5px'>Intensite = ecart a la "
                        f"mediane · {len(h)} trimestres</span>", x=.015, xanchor="left"),
        height=56*len(series) + 155, hovermode="x unified",
        hoverlabel=dict(bgcolor="white", bordercolor=PAL["grille"],
                        font=dict(size=12.5, color=PAL["encre"])),
        **{**MISE_EN_FORME, "margin": dict(l=150, r=60, t=115, b=55)})
    return fig


def _dot(h, per, ctx, variables, nom, methode):
    """STYLE 4 — points relies, taille proportionnelle, tres sobre."""
    n_rows = 1 + len(variables)
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=.06)

    vals = h[TARGET].values.astype(float)
    etendue = max(vals.max() - vals.min(), 1e-9)
    taille = 9 + 17*(vals - vals.min())/etendue
    fig.add_scatter(x=per, y=vals, mode="lines", line=dict(color=PAL["grille"], width=6),
                    showlegend=False, hoverinfo="skip", row=1, col=1)
    fig.add_scatter(x=per, y=vals, mode="markers+text",
                    marker=dict(size=taille, color=PAL["encre"],
                                line=dict(color="white", width=2.2)),
                    text=[f"{v:,.0f}" for v in vals], textposition="top center",
                    textfont=dict(size=9.5, color=PAL["texte"]), name=TARGET,
                    hovertemplate="<b>%{x}</b><br>" + TARGET +
                                  " : <b>%{y:,.0f}</b><extra></extra>", row=1, col=1)

    if ctx and ctx["periode"] in per:
        c = PAL["ano"] if not ctx["couvert"] else PAL["ok"]
        halo = PAL["halo"] if not ctx["couvert"] else "rgba(61,90,158,0.18)"
        fig.add_scatter(x=[ctx["periode"]], y=[ctx["obs"]], mode="markers",
                        marker=dict(size=36, color=halo), showlegend=False,
                        hoverinfo="skip", row=1, col=1)
        fig.add_scatter(x=[ctx["periode"]], y=[ctx["obs"]], mode="markers",
                        marker=dict(size=14, color=c, line=dict(color="white", width=2.6)),
                        name="Hors intervalle" if not ctx["couvert"] else "Couvert",
                        hovertemplate=f"Observe {ctx['obs']:,.0f}<extra></extra>",
                        row=1, col=1)

    for i, v in enumerate(variables, start=2):
        vv = h[v].values.astype(float)
        c = PAL["doux"][(i-2) % len(PAL["doux"])]
        fig.add_scatter(x=per, y=vv, mode="lines", line=dict(color=PAL["grille"], width=4),
                        showlegend=False, hoverinfo="skip", row=i, col=1)
        fig.add_scatter(x=per, y=vv, mode="markers",
                        marker=dict(size=9, color=c, line=dict(color="white", width=1.8)),
                        showlegend=False, row=i, col=1,
                        hovertemplate=f"<b>%{{x}}</b><br>{v} : <b>%{{y:,.4g}}</b>"
                                      "<extra></extra>")
        fig.add_annotation(xref="paper", x=1.005, y=vv[-1], xanchor="left",
                           text=f"<b>{v}</b>", showarrow=False,
                           font=dict(size=10, color=c), row=i, col=1)

    fig.update_yaxes(gridcolor=PAL["grille"], zeroline=False, tickformat=",.0f")
    fig.update_xaxes(showgrid=False)
    fig.update_layout(
        title=dict(text=f"<b style='font-size:18px;color:{PAL['encre']}'>{nom}</b>"
                        f"<br><span style='font-size:11.5px'>{len(h)} trimestres · "
                        f"variables via {methode}</span>", x=.015, xanchor="left"),
        height=280 + 105*len(variables), hovermode="x unified",
        hoverlabel=dict(bgcolor="white", bordercolor=PAL["grille"],
                        font=dict(size=12.5, color=PAL["encre"])),
        legend=dict(orientation="h", y=1.03, x=1, xanchor="right"), **MISE_EN_FORME)
    return fig


def _heatmap(h, per, ctx, variables, nom, methode):
    """STYLE 5 — matrice variables x trimestres, cible en bandeau superieur."""
    series = list(variables)
    z = []
    for v in series:
        vv = h[v].values.astype(float)
        rng = vv.max() - vv.min()
        z.append((vv - vv.min())/rng if rng > 0 else np.zeros_like(vv))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.10,
                        row_heights=[.36, .64],
                        subplot_titles=(f"{TARGET}", "Variables determinantes"))

    vals = h[TARGET].values.astype(float)
    fig.add_scatter(x=per, y=vals, mode="lines", line=dict(width=0), fill="tozeroy",
                    fillcolor="rgba(99,110,250,0.10)", showlegend=False,
                    hoverinfo="skip", row=1, col=1)
    fig.add_scatter(x=per, y=vals, mode="lines+markers",
                    line=dict(color=PAL["encre"], width=2.8, shape="spline", smoothing=.55),
                    marker=dict(size=8, color="white",
                                line=dict(color=PAL["encre"], width=2.2)),
                    showlegend=False, row=1, col=1,
                    hovertemplate="<b>%{x}</b><br>" + TARGET +
                                  " : <b>%{y:,.0f}</b><extra></extra>")
    if ctx and ctx["periode"] in per:
        c = PAL["ano"] if not ctx["couvert"] else PAL["ok"]
        halo = PAL["halo"] if not ctx["couvert"] else "rgba(61,90,158,0.18)"
        fig.add_scatter(x=[ctx["periode"]], y=[ctx["obs"]], mode="markers",
                        marker=dict(size=30, color=halo), showlegend=False,
                        hoverinfo="skip", row=1, col=1)
        fig.add_scatter(x=[ctx["periode"]], y=[ctx["obs"]], mode="markers",
                        marker=dict(size=12, color=c, line=dict(color="white", width=2.2)),
                        showlegend=False, row=1, col=1,
                        hovertemplate=f"Observe {ctx['obs']:,.0f}<extra></extra>")

    if z:
        fig.add_heatmap(z=z, x=per, y=series, colorscale="RdBu_r", zmid=.5,
                        xgap=3, ygap=3, showscale=True,
                        colorbar=dict(title="Niveau<br>relatif", thickness=13, len=.5,
                                      y=.27, tickvals=[0, .5, 1],
                                      ticktext=["bas", "median", "haut"]),
                        hovertemplate="<b>%{y}</b><br>%{x} : %{z:.0%}<extra></extra>",
                        row=2, col=1)

    fig.update_yaxes(gridcolor=PAL["grille"], zeroline=False, tickformat=",.0f",
                     row=1, col=1)
    fig.update_xaxes(showgrid=False)
    fig.update_layout(
        title=dict(text=f"<b style='font-size:18px;color:{PAL['encre']}'>{nom}</b>"
                        f"<br><span style='font-size:11.5px'>{len(h)} trimestres · "
                        f"variables via {methode}</span>", x=.015, xanchor="left"),
        height=210 + 46*max(len(series), 1) + 210,
        hoverlabel=dict(bgcolor="white", bordercolor=PAL["grille"],
                        font=dict(size=12.5, color=PAL["encre"])), **MISE_EN_FORME)
    return fig


STYLES = {"fan": _fan, "sparkline": _sparkline, "horizon": _horizon,
          "dot": _dot, "heatmap": _heatmap}


def evolution_unite(cle_valeurs, expl_df, style="fan",
                    n_periodes=N




# ┌─── PARAMETRES ───┐
RANG_UNITE, N_PER, N_VAR = 1, 12, 6
# └──────────────────┘
UNITE = unite_par_rang(RANG_UNITE)
print(f"Rang #{RANG_UNITE} — {' · '.join(UNITE)}")
evolution_unite(UNITE, expl, style="sparkline", n_periodes=N_PER, n_vars=N_VAR);
