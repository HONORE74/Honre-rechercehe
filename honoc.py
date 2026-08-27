# ================================================================
# BLOC F1 — PALETTE ET HELPERS
# ================================================================
# ┌─────────────────── PARAMETRES A MODIFIER ───────────────────┐
DF_HISTORIQUE   = df          # base complete avec tous les trimestres
N_PERIODES      = 10          # 5, 8, 10, 12... nombre de trimestres affiches
TOP_N_VARS      = 5           # nombre de variables suivies
MODELE_TE       = None        # modele pour SHAP local (optionnel)
X_TE            = None        # features alignees sur expl (optionnel)
VARIABLES_MANUELLES = []      # repli si ni SHAP ni importance disponibles
# └──────────────────────────────────────────────────────────────┘

import numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PAL = dict(
    encre   = "#1A1A2E",                    # ligne principale
    bande   = "rgba(99,110,250,0.16)",      # interieur de l'intervalle
    bordure = "rgba(99,110,250,0.55)",
    ok      = "#3D5A9E",
    halo    = "rgba(255,107,107,0.22)",     # auréole d'anomalie
    ano     = "#FF5A5F",
    doux    = ["#6C8EBF", "#82B366", "#B8860B", "#9673A6", "#D6795A", "#5F9EA0"],
    grille  = "#F0F3F8",
    texte   = "#6B7688",
    fond    = "#FFFFFF")

MISE_EN_FORME = dict(
    template="plotly_white", font=dict(family="Inter, system-ui, sans-serif",
                                       size=12, color=PAL["texte"]),
    paper_bgcolor=PAL["fond"], plot_bgcolor=PAL["fond"],
    margin=dict(l=60, r=90, t=110, b=50))


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
    n = n or TOP_N_VARS
    if MODELE_TE is not None and X_TE is not None:
        try:
            import shap
            mdl = MODELE_TE
            if hasattr(mdl, "named_steps") and "model" in mdl.named_steps:
                mdl = mdl.named_steps["model"]
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
            mdl = MODELE_TE
            if hasattr(mdl, "named_steps") and "model" in mdl.named_steps:
                mdl = mdl.named_steps["model"]
            noms = list(X_TE.columns) if X_TE is not None else list(mdl.feature_name_)
            imp = pd.Series(mdl.booster_.feature_importance("gain"), index=noms)
            cand = [v for v in imp.sort_values(ascending=False).index
                    if v in DF_HISTORIQUE.columns
                    and pd.api.types.is_numeric_dtype(DF_HISTORIQUE[v])]
            if cand:
                return cand[:n], "importance globale"
        except Exception:
            pass

    return [v for v in VARIABLES_MANUELLES if v in DF_HISTORIQUE.columns][:n], "liste manuelle"


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


print("Definies :", [n for n in ["_cles_unite", "_historique_unite", "_periodes",
                                 "_variables_importantes", "_contexte_test"]
                     if n in globals()])







# ================================================================
# BLOC F2 — CINQ STYLES DE VISUALISATION
# ================================================================

def _fan(h, per, ctx, variables, nom, methode):
    """STYLE 1 — bande douce, ligne sombre, halo lumineux sur l'anomalie."""
    n_rows = 1 + len(variables)
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=.05,
                        row_heights=[.45] + [.55/max(len(variables),1)]*len(variables))

    if ctx and ctx["periode"] in per:
        k = per.index(ctx["periode"])
        for r in range(1, n_rows + 1):
            fig.add_vrect(x0=k-.5, x1=k+.5, fillcolor="rgba(99,110,250,0.05)",
                          line_width=0, layer="below", row=r, col=1)
        fig.add_scatter(x=[per[k], per[k]], y=[ctx["lo"], ctx["hi"]], mode="lines",
                        line=dict(color=PAL["bordure"], width=14), opacity=.35,
                        showlegend=False, hoverinfo="skip", row=1, col=1)
        fig.add_scatter(x=[per[k]], y=[ctx["pred"]], mode="markers",
                        marker=dict(size=9, color="white",
                                    line=dict(color=PAL["encre"], width=1.6)),
                        name="Prediction", row=1, col=1)

    fig.add_scatter(x=per, y=h[TARGET], mode="lines",
                    line=dict(color=PAL["encre"], width=2.6, shape="spline",
                              smoothing=.6),
                    name=TARGET, row=1, col=1)
    fig.add_scatter(x=per, y=h[TARGET], mode="markers",
                    marker=dict(size=6, color="white",
                                line=dict(color=PAL["encre"], width=1.6)),
                    showlegend=False, row=1, col=1)

    if ctx and ctx["periode"] in per:
        c = PAL["ano"] if not ctx["couvert"] else PAL["ok"]
        halo = PAL["halo"] if not ctx["couvert"] else "rgba(61,90,158,0.18)"
        fig.add_scatter(x=[ctx["periode"]], y=[ctx["obs"]], mode="markers",
                        marker=dict(size=30, color=halo), showlegend=False,
                        hoverinfo="skip", row=1, col=1)
        fig.add_scatter(x=[ctx["periode"]], y=[ctx["obs"]], mode="markers",
                        marker=dict(size=11, color=c,
                                    line=dict(color="white", width=2)),
                        name="Couvert" if ctx["couvert"] else "Hors intervalle",
                        row=1, col=1)
        fig.add_annotation(x=ctx["periode"], y=ctx["obs"], row=1, col=1,
                           text=f"<b>{ctx['obs']:,.0f}</b>", showarrow=False,
                           yshift=26, font=dict(size=12, color=c))

    for i, v in enumerate(variables, start=2):
        fig.add_scatter(x=per, y=h[v], mode="lines",
                        line=dict(color=PAL["doux"][(i-2) % len(PAL["doux"])],
                                  width=2, shape="spline", smoothing=.6),
                        fill="tozeroy",
                        fillcolor=PAL["doux"][(i-2) % len(PAL["doux"])].replace(")", ",0.10)")
                                  .replace("#", "rgba(") if False else "rgba(0,0,0,0.03)",
                        showlegend=False, row=i, col=1)
        fig.add_annotation(xref="paper", x=1.005, y=h[v].iloc[-1], xanchor="left",
                           text=f"<b>{v}</b><br>{h[v].iloc[-1]:,.4g}", showarrow=False,
                           align="left", font=dict(size=10,
                           color=PAL["doux"][(i-2) % len(PAL["doux"])]), row=i, col=1)

    fig.update_yaxes(gridcolor=PAL["grille"], zeroline=False)
    fig.update_xaxes(showgrid=False, tickangle=0, row=n_rows, col=1)
    fig.update_layout(
        title=dict(text=f"<b>{nom}</b><br><sup>{len(h)} trimestres · "
                        f"variables via {methode}</sup>", x=.02, xanchor="left"),
        height=280 + 110*len(variables), hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"), **MISE_EN_FORME)
    return fig


def _sparkline(h, per, ctx, variables, nom, methode):
    """STYLE 2 — mini-courbes empilees, valeur actuelle en gros a droite."""
    series = [(TARGET, PAL["encre"])] + [
        (v, PAL["doux"][i % len(PAL["doux"])]) for i, v in enumerate(variables)]
    fig = make_subplots(rows=len(series), cols=1, shared_xaxes=True,
                        vertical_spacing=.03)

    for i, (v, c) in enumerate(series, start=1):
        vals = h[v].values.astype(float)
        fig.add_scatter(x=per, y=vals, mode="lines", line=dict(color=c, width=2.2,
                        shape="spline", smoothing=.7), fill="tozeroy",
                        fillcolor=c.replace("#", "rgba(").replace(")", "")
                        if False else "rgba(0,0,0,0.035)",
                        showlegend=False, row=i, col=1,
                        hovertemplate=f"<b>{v}</b><br>%{{x}} : %{{y:,.4g}}<extra></extra>")
        fig.add_scatter(x=[per[-1]], y=[vals[-1]], mode="markers",
                        marker=dict(size=8, color=c), showlegend=False,
                        hoverinfo="skip", row=i, col=1)
        fig.add_annotation(xref="paper", x=1.01, y=vals[-1], xanchor="left",
                           text=f"<b>{vals[-1]:,.4g}</b><br>"
                                f"<span style='font-size:9px'>{v}</span>",
                           showarrow=False, align="left",
                           font=dict(size=13, color=c), row=i, col=1)
        if i == 1 and ctx and ctx["periode"] in per:
            c2 = PAL["ano"] if not ctx["couvert"] else PAL["ok"]
            fig.add_scatter(x=[ctx["periode"]], y=[ctx["obs"]], mode="markers",
                            marker=dict(size=22, color=PAL["halo"] if not ctx["couvert"]
                                        else "rgba(61,90,158,0.18)"),
                            showlegend=False, hoverinfo="skip", row=1, col=1)
            fig.add_scatter(x=[ctx["periode"]], y=[ctx["obs"]], mode="markers",
                            marker=dict(size=10, color=c2,
                                        line=dict(color="white", width=2)),
                            showlegend=False, row=1, col=1)

    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_xaxes(showticklabels=True, row=len(series), col=1)
    fig.update_layout(
        title=dict(text=f"<b>{nom}</b><br><sup>{len(h)} trimestres · "
                        f"variables via {methode}</sup>", x=.02, xanchor="left"),
        height=110 * len(series) + 130, hovermode="x unified", **MISE_EN_FORME)
    return fig


def _horizon(h, per, ctx, variables, nom, methode):
    """STYLE 3 — bandes d'intensite repliees : tres compact."""
    series = [TARGET] + list(variables)
    fig = make_subplots(rows=len(series), cols=1, shared_xaxes=True, vertical_spacing=.015)
    tons = ["rgba(99,110,250,0.20)", "rgba(99,110,250,0.42)", "rgba(99,110,250,0.68)",
            "rgba(255,90,95,0.55)"]

    for i, v in enumerate(series, start=1):
        vals = h[v].values.astype(float)
        med = np.median(vals)
        ecart = np.std(vals) or 1.0
        z = (vals - med) / ecart
        for k, ton in enumerate(tons):
            bas, haut = k * .8, (k + 1) * .8
            band = np.clip(np.abs(z), bas, haut) - bas
            fig.add_scatter(x=per, y=band, mode="lines", line=dict(width=0),
                            fill="tozeroy", fillcolor=ton, showlegend=False,
                            hoverinfo="skip", row=i, col=1)
        fig.add_scatter(x=per, y=np.abs(z), mode="lines",
                        line=dict(color="rgba(0,0,0,0)", width=0), showlegend=False,
                        hovertemplate=f"<b>{v}</b><br>%{{x}}<br>ecart : %{{y:.2f}} sigma"
                                      "<extra></extra>", row=i, col=1)
        fig.add_annotation(xref="paper", x=-.01, y=.4, xanchor="right",
                           text=f"<b>{v}</b>", showarrow=False,
                           font=dict(size=10, color=PAL["texte"]), row=i, col=1)

    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, range=[0, .85])
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_xaxes(showticklabels=True, row=len(series), col=1)
    fig.update_layout(
        title=dict(text=f"<b>{nom}</b> — vue horizon"
                        f"<br><sup>Intensite = ecart a la mediane · {len(h)} trimestres</sup>",
                   x=.02, xanchor="left"),
        height=58 * len(series) + 150, **MISE_EN_FORME,
        margin=dict(l=140, r=60, t=110, b=50))
    return fig


def _dot(h, per, ctx, variables, nom, methode):
    """STYLE 4 — points relies, taille proportionnelle, tres sobre."""
    n_rows = 1 + len(variables)
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=.055)

    vals = h[TARGET].values.astype(float)
    taille = 8 + 16 * (vals - vals.min()) / max(vals.max() - vals.min(), 1e-9)
    fig.add_scatter(x=per, y=vals, mode="lines", line=dict(color=PAL["grille"], width=6),
                    showlegend=False, hoverinfo="skip", row=1, col=1)
    fig.add_scatter(x=per, y=vals, mode="markers+text",
                    marker=dict(size=taille, color=PAL["encre"],
                                line=dict(color="white", width=2)),
                    text=[f"{v:,.0f}" for v in vals], textposition="top center",
                    textfont=dict(size=9, color=PAL["texte"]),
                    name=TARGET, row=1, col=1)

    if ctx and ctx["periode"] in per:
        c = PAL["ano"] if not ctx["couvert"] else PAL["ok"]
        fig.add_scatter(x=[ctx["periode"]], y=[ctx["obs"]], mode="markers",
                        marker=dict(size=34, color=PAL["halo"] if not ctx["couvert"]
                                    else "rgba(61,90,158,0.18)"),
                        showlegend=False, hoverinfo="skip", row=1, col=1)
        fig.add_scatter(x=[ctx["periode"]], y=[ctx["obs"]], mode="markers",
                        marker=dict(size=14, color=c, line=dict(color="white", width=2.5)),
                        name="Hors intervalle" if not ctx["couvert"] else "Couvert",
                        row=1, col=1)

    for i, v in enumerate(variables, start=2):
        vv = h[v].values.astype(float)
        c = PAL["doux"][(i-2) % len(PAL["doux"])]
        fig.add_scatter(x=per, y=vv, mode="lines", line=dict(color=PAL["grille"], width=4),
                        showlegend=False, hoverinfo="skip", row=i, col=1)
        fig.add_scatter(x=per, y=vv, mode="markers",
                        marker=dict(size=9, color=c, line=dict(color="white", width=1.6)),
                        showlegend=False, row=i, col=1,
                        hovertemplate=f"<b>{v}</b><br>%{{x}} : %{{y:,.4g}}<extra></extra>")
        fig.add_annotation(xref="paper", x=1.005, y=vv[-1], xanchor="left",
                           text=f"<b>{v}</b>", showarrow=False,
                           font=dict(size=10, color=c), row=i, col=1)

    fig.update_yaxes(gridcolor=PAL["grille"], zeroline=False)
    fig.update_xaxes(showgrid=False)
    fig.update_layout(
        title=dict(text=f"<b>{nom}</b><br><sup>{len(h)} trimestres · "
                        f"variables via {methode}</sup>", x=.02, xanchor="left"),
        height=250 + 105*len(variables), hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"), **MISE_EN_FORME)
    return fig


def _heatmap(h, per, ctx, variables, nom, methode):
    """STYLE 5 — matrice variables x trimestres, cible en bandeau superieur."""
    series = list(variables)
    z = []
    for v in series:
        vv = h[v].values.astype(float)
        rng = vv.max() - vv.min()
        z.append((vv - vv.min()) / rng if rng > 0 else np.zeros_like(vv))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.09,
                        row_heights=[.34, .66],
                        subplot_titles=(f"{TARGET}", "Variables determinantes"))

    vals = h[TARGET].values.astype(float)
    fig.add_scatter(x=per, y=vals, mode="lines+markers",
                    line=dict(color=PAL["encre"], width=2.6, shape="spline", smoothing=.6),
                    marker=dict(size=7, color="white",
                                line=dict(color=PAL["encre"], width=1.6)),
                    showlegend=False, row=1, col=1,
                    hovertemplate="%{x} : %{y:,.0f}<extra></extra>")
    if ctx and ctx["periode"] in per:
        c = PAL["ano"] if not ctx["couvert"] else PAL["ok"]
        fig.add_scatter(x=[ctx["periode"]], y=[ctx["obs"]], mode="markers",
                        marker=dict(size=28, color=PAL["halo"] if not ctx["couvert"]
                                    else "rgba(61,90,158,0.18)"),
                        showlegend=False, hoverinfo="skip", row=1, col=1)
        fig.add_scatter(x=[ctx["periode"]], y=[ctx["obs"]], mode="markers",
                        marker=dict(size=12, color=c, line=dict(color="white", width=2)),
                        showlegend=False, row=1, col=1)

    if z:
        fig.add_heatmap(z=z, x=per, y=series, colorscale="RdBu_r", zmid=.5,
                        xgap=3, ygap=3, showscale=True,
                        colorbar=dict(title="Niveau<br>relatif", thickness=13, len=.5,
                                      y=.28, tickvals=[0, .5, 1],
                                      ticktext=["bas", "median", "haut"]),
                        hovertemplate="<b>%{y}</b><br>%{x} : %{z:.0%}<extra></extra>",
                        row=2, col=1)

    fig.update_yaxes(gridcolor=PAL["grille"], zeroline=False, row=1, col=1)
    fig.update_xaxes(showgrid=False)
    fig.update_layout(
        title=dict(text=f"<b>{nom}</b><br><sup>{len(h)} trimestres · "
                        f"variables via {methode}</sup>", x=.02, xanchor="left"),
        height=200 + 46*max(len(series), 1) + 200, **MISE_EN_FORME)
    return fig


STYLES = {"fan": _fan, "sparkline": _sparkline, "horizon": _horizon,
          "dot": _dot, "heatmap": _heatmap}


def evolution_unite(cle_valeurs, expl_df, style="fan",
                    n_periodes=None, n_vars=None, afficher=True):
    cles = _cles_unite(expl_df)
    h = _historique_unite(cle_valeurs, cles, n_periodes)
    if len(h) == 0:
        print("Aucun historique pour cette unite.")
        return None
    per = _periodes(h)
    nom = " · ".join(str(v) for v in cle_valeurs)
    variables, methode = _variables_importantes(cle_valeurs, cles, expl_df, n_vars)
    variables = [v for v in variables if v in h.columns]
    ctx = _contexte_test(cle_valeurs, cles, expl_df)

    fig = STYLES.get(style, _fan)(h, per, ctx, variables, nom, methode)
    if afficher:
        fig.show()
    return fig


print("Styles disponibles :", list(STYLES))








# ================================================================
# BLOC F3 — SELECTEURS : unite, style, nombre de periodes
# ================================================================
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

N_UNITES_LISTE = 40
TRI_LISTE      = "score_composite"

_cles = _cles_unite(expl)
_top = anomalies_prio.nlargest(min(N_UNITES_LISTE, len(anomalies_prio)),
                               TRI_LISTE if TRI_LISTE in anomalies_prio.columns
                               else "score_composite")
_options = [(f"#{int(r['rank'])} · " + " · ".join(str(r[c]) for c in _cles)
             if "rank" in r.index and pd.notna(r["rank"])
             else " · ".join(str(r[c]) for c in _cles),
             tuple(str(r[c]) for c in _cles)) for _, r in _top.iterrows()]

w_unite = widgets.Dropdown(options=_options, value=_options[0][1],
    description="Unite :", layout=widgets.Layout(width="720px"),
    style={"description_width": "60px"})
w_style = widgets.ToggleButtons(
    options=[("Fan chart", "fan"), ("Sparklines", "sparkline"),
             ("Horizon", "horizon"), ("Dot plot", "dot"), ("Matrice", "heatmap")],
    value="fan", description="Style :", style={"description_width": "60px",
                                               "button_width": "115px"})
w_per = widgets.IntSlider(value=N_PERIODES, min=3, max=24, step=1,
    description="Trimestres :", continuous_update=False,
    layout=widgets.Layout(width="420px"), style={"description_width": "90px"})
w_var = widgets.IntSlider(value=TOP_N_VARS, min=0, max=8, step=1,
    description="Variables :", continuous_update=False,
    layout=widgets.Layout(width="420px"), style={"description_width": "90px"})

zone = widgets.Output()


def _maj(*_):
    with zone:
        clear_output(wait=True)
        evolution_unite(w_unite.value, expl, style=w_style.value,
                        n_periodes=w_per.value, n_vars=w_var.value)


for w in (w_unite, w_style, w_per, w_var):
    w.observe(lambda c: _maj() if c["name"] == "value" else None, names="value")

display(HTML("<div style='font-family:system-ui,sans-serif;font-size:12.5px;"
             "color:#3D5A9E;background:#F0F3FB;padding:10px 14px;border-radius:8px;"
             "margin-bottom:10px'><b>EVOLUTION TEMPORELLE</b> — choisissez l'unite, "
             "le style de rendu, la profondeur d'historique et le nombre de variables. "
             "Le halo colore marque la periode validee.</div>"))
display(w_unite, w_style, widgets.HBox([w_per, w_var]), zone)
_maj()
































# ================================================================
# STYLE 1 — FAN CHART : bande douce, ligne d'encre, halo lumineux
#   Le plus elegant pour une unite seule. Figure principale du memoire.
# ================================================================
# ┌─── PARAMETRES ───┐
RANG_UNITE = 1      # 1 = anomalie la plus grave, 2 = la suivante...
N_PER      = 10     # trimestres d'historique (3 a 24)
N_VAR      = 5      # variables suivies (0 = cible seule, tres epure)
# └──────────────────┘

_c = _cles_unite(expl)
UNITE = tuple(str(anomalies_prio.iloc[RANG_UNITE - 1][x]) for x in _c)
print(f"Rang #{RANG_UNITE} — {' · '.join(UNITE)}")

evolution_unite(UNITE, expl, style="fan", n_periodes=N_PER, n_vars=N_VAR)










# ================================================================
# STYLE 2 — SPARKLINES : mini-courbes compactes, valeur actuelle a droite
#   Style tableau de bord financier. Ideal pour lire 6 variables d'un coup.
# ================================================================
# ┌─── PARAMETRES ───┐
RANG_UNITE = 1
N_PER      = 12     # les sparklines supportent bien un historique long
N_VAR      = 6
# └──────────────────┘

_c = _cles_unite(expl)
UNITE = tuple(str(anomalies_prio.iloc[RANG_UNITE - 1][x]) for x in _c)
print(f"Rang #{RANG_UNITE} — {' · '.join(UNITE)}")

evolution_unite(UNITE, expl, style="sparkline", n_periodes=N_PER, n_vars=N_VAR)



















# ================================================================
# STYLE 3 — HORIZON : bandes d'intensite repliees, tres compact
#   L'intensite code l'ecart a la mediane. Divise la hauteur par 4.
#   A privilegier quand vous suivez beaucoup de variables.
# ================================================================
# ┌─── PARAMETRES ───┐
RANG_UNITE = 1
N_PER      = 16     # le horizon chart gagne avec un historique long
N_VAR      = 8
# └──────────────────┘

_c = _cles_unite(expl)
UNITE = tuple(str(anomalies_prio.iloc[RANG_UNITE - 1][x]) for x in _c)
print(f"Rang #{RANG_UNITE} — {' · '.join(UNITE)}")

evolution_unite(UNITE, expl, style="horizon", n_periodes=N_PER, n_vars=N_VAR)



















# ================================================================
# STYLE 4 — DOT PLOT : points relies, taille proportionnelle, aucune bande
#   Le plus sobre. Le message passe en une seconde. Ideal en slide.
# ================================================================
# ┌─── PARAMETRES ───┐
RANG_UNITE = 1
N_PER      = 8      # peu de periodes : les points restent lisibles
N_VAR      = 3
# └──────────────────┘

_c = _cles_unite(expl)
UNITE = tuple(str(anomalies_prio.iloc[RANG_UNITE - 1][x]) for x in _c)
print(f"Rang #{RANG_UNITE} — {' · '.join(UNITE)}")

evolution_unite(UNITE, expl, style="dot", n_periodes=N_PER, n_vars=N_VAR)











# ================================================================
# STYLE 5 — MATRICE : variables x trimestres, cible en bandeau superieur
#   Pour reperer quelle variable decroche EN MEME TEMPS que la cible.
# ================================================================
# ┌─── PARAMETRES ───┐
RANG_UNITE = 1
N_PER      = 12
N_VAR      = 8      # la matrice supporte beaucoup de lignes
# └──────────────────┘

_c = _cles_unite(expl)
UNITE = tuple(str(anomalies_prio.iloc[RANG_UNITE - 1][x]) for x in _c)
print(f"Rang #{RANG_UNITE} — {' · '.join(UNITE)}")

evolution_unite(UNITE, expl, style="heatmap", n_periodes=N_PER, n_vars=N_VAR)

















# ================================================================
# COMPARAISON — les N pires anomalies dans le style choisi
# ================================================================
# ┌─── PARAMETRES ───┐
STYLE   = "fan"     # "fan" | "sparkline" | "horizon" | "dot" | "heatmap"
N_TOP   = 3         # nombre d'unites a comparer
N_PER   = 10
N_VAR   = 3
# └──────────────────┘

_c = _cles_unite(expl)
for i in range(1, N_TOP + 1):
    u = tuple(str(anomalies_prio.iloc[i - 1][x]) for x in _c)
    print(f"\n{'─'*70}\nRang #{i} — {' · '.join(u)}\n{'─'*70}")
    evolution_unite(u, expl, style=STYLE, n_periodes=N_PER, n_vars=N_VAR)
