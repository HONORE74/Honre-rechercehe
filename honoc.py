# -*- coding: utf-8 -*-
# =============================================================================
#  MISSION 1 - EVOLUTION DE LA TARGET POUR UN SOUS-PORTEFEUILLE
#
#  Votre code, avec trois modifications integrees :
#    1. format des nombres en francais (1 249 441 et non 1,249,441), cote Python
#    2. format des nombres en francais cote Plotly (axes et infobulles)
#    3. zone decorative sous la courbe passee en gris neutre, pour ne plus
#       etre confondue avec la bande bleue de l'intervalle conforme
#
#  Prerequis dans la session : df, expl, anomalies_prio, ID_COLS, TARGET
# =============================================================================

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

# ┌─────────────────────────── PARAMETRES ────────────────────────────┐
RANG          = 1        # 1 = l'anomalie la plus grave  (*)
N_trimestre   = 8        # nombre de trimestres d'historique affiches
VARS_SECOND   = 0        # variables secondaires sous la courbe (0 = aucune)
AFFICHER_VAL  = True     # afficher la valeur au-dessus de chaque point
# └────────────────────────────────────────────────────────────────────┘
#  (*) cette ligne etait coupee sur votre capture, remettez votre valeur.

ENCRE, ACCENT, OK  = "#141B34", "#FF5A5F", "#3D5A9E"
BLEU, GRILLE, GRIS = "#636EFA", "#EDF1F7", "#8A93A5"
DOUX = ["#6C8EBF", "#82B366", "#C08552", "#9673A6", "#5F9EA0", "#B85C7E"]

# --- MODIFICATION 1 ----------------------------------------------------------
#  Format francais des nombres cote Python. Remplace tous les f"{v:,.0f}", qui
#  produisaient le format anglo-saxon a virgules (1,249,441).
# -----------------------------------------------------------------------------
def fmt(v):
    """1249441.0 -> '1 249 441'  (separateur de milliers francais)."""
    return f"{v:,.0f}".replace(",", " ")


# ---------------------------------------------------- selection du sous-portefeuille
_c = [c for c in ID_COLS if c in expl.columns and c in df.columns]
UNITE = tuple(str(anomalies_prio.iloc[RANG - 1][x]) for x in _c)
nom = " · ".join(UNITE)

_m = np.logical_and.reduce([df[c].astype(str).values == v
                            for c, v in zip(_c, UNITE)])
h = df[_m].sort_values("time_idx").tail(N_trimestre).copy()
if len(h) == 0:
    raise ValueError("Aucun historique pour cette unite.")
if len(h) < N_trimestre:
    print(f"ATTENTION : seulement {len(h)} trimestre(s) disponible(s) pour "
          f"{nom} (demande : {N_trimestre}). Ce sous-portefeuille est "
          f"probablement recent ou incomplet dans la base.")

per = (h["year"].astype(int).astype(str) + "-T"
       + h["quarter"].astype(int).astype(str)).tolist()
val = h[TARGET].values.astype(float)

# ------------------------------------------- contexte conforme (periode validee)
_mt = np.logical_and.reduce([expl[c].astype(str).values == v
                             for c, v in zip(_c, UNITE)])
_t = expl[_mt]
ctx = None
if len(_t):
    r = _t.iloc[0]
    ctx = dict(per=f"{int(r['year'])}-T{int(r['quarter'])}",
               pred=float(r["y_pred"]),
               lo=float(r["borne_basse"]), hi=float(r["borne_haute"]),
               obs=float(r["y_obs"]), couvert=bool(r["dans_intervalle"]))

# ------------------------------------------------------- variables secondaires
secondaires = []
if VARS_SECOND > 0:
    num = [c for c in h.columns
           if c != TARGET and pd.api.types.is_numeric_dtype(h[c])
           and h[c].notna().all() and h[c].nunique() > 1
           and c not in ("time_idx", "year", "quarter")]
    if "MODELE_TE" in globals() and MODELE_TE is not None:
        try:
            mdl = (MODELE_TE.named_steps["model"]
                   if hasattr(MODELE_TE, "named_steps") else MODELE_TE)
            imp = pd.Series(mdl.booster_.feature_importance("gain"),
                            index=mdl.feature_name_)
            num = [v for v in imp.sort_values(ascending=False).index
                   if v in num] or num
        except Exception:
            pass
    secondaires = num[:VARS_SECOND]

# ------------------------------------------------------------------- figure
n_rows = 1 + len(secondaires)
fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=.06,
                    row_heights=[.58] + [.42 / max(len(secondaires), 1)]
                                * len(secondaires) if secondaires else [1.0])

if ctx and ctx["per"] in per:
    k = per.index(ctx["per"])
    for rr in range(1, n_rows + 1):
        fig.add_vrect(x0=k - .5, x1=k + .5, fillcolor="rgba(99,110,250,0.055)",
                      line_width=0, layer="below", row=rr, col=1)

# --- MODIFICATION 3 ----------------------------------------------------------
#  La zone sous la courbe etait bleue, donc de la meme famille que la bande de
#  l'intervalle conforme. Un lecteur pouvait croire que TOUT l'historique etait
#  couvert par un intervalle, alors qu'il n'y en a qu'un, sur la periode validee.
#  Elle passe en gris neutre. Pour la supprimer completement plutot que la
#  neutraliser, commentez le bloc try/except ci-dessous.
# -----------------------------------------------------------------------------
aire = dict(x=per, y=val, mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
            fill="tozeroy", showlegend=False, hoverinfo="skip")
try:
    fig.add_scatter(**aire, fillgradient=dict(type="vertical", colorscale=[
        (0, "rgba(140,147,165,0.02)"), (1, "rgba(140,147,165,0.14)")]),
        row=1, col=1)
except Exception:
    fig.add_scatter(**aire, fillcolor="rgba(140,147,165,0.10)", row=1, col=1)

# ----------------------------------------------------------------------------
#  BLOC RECONSTRUIT : coupe entre vos 2e et 3e captures. Reconstitue d'apres la
#  legende de votre graphique (Intervalle CP 90 %, Prediction) et la bande
#  verticale bleue visible sur 2024-T4. A verifier contre votre original.
# ----------------------------------------------------------------------------
if ctx and ctx["per"] in per:
    fig.add_scatter(x=[ctx["per"], ctx["per"]], y=[ctx["lo"], ctx["hi"]],
                    mode="lines", line=dict(color=BLEU, width=15), opacity=.26,
                    name="Intervalle CP 90 %",
                    hovertemplate=f"Intervalle<br>[{fmt(ctx['lo'])} ; "
                                  f"{fmt(ctx['hi'])}]<extra></extra>",
                    row=1, col=1)
    fig.add_scatter(x=[ctx["per"]], y=[ctx["pred"]], mode="markers",
                    marker=dict(size=10, symbol="diamond", color="white",
                                line=dict(color=ENCRE, width=1.8)),
                    name="Prediction",
                    hovertemplate=f"Prediction<br>{fmt(ctx['pred'])}"
                                  "<extra></extra>",
                    row=1, col=1)
# ------------------------------------------------- fin du bloc reconstruit ---

fig.add_scatter(x=per, y=val,
                mode="lines+markers+text" if AFFICHER_VAL else "lines+markers",
                line=dict(color=ENCRE, width=2.8, shape="spline", smoothing=.55),
                marker=dict(size=9, color="white",
                            line=dict(color=ENCRE, width=2.2)),
                text=[fmt(v) for v in val] if AFFICHER_VAL else None,
                textposition="top center", textfont=dict(size=9.5, color=GRIS),
                name=TARGET,
                hovertemplate="<b>%{x}</b><br>" + TARGET +
                              " : <b>%{y:,.0f}</b><extra></extra>",
                row=1, col=1)

if ctx and ctx["per"] in per:
    coul = ACCENT if not ctx["couvert"] else OK
    halo = ("rgba(255,90,95,0.20)" if not ctx["couvert"]
            else "rgba(61,90,158,0.18)")
    for taille, c in ((38, halo), (24, halo)):
        fig.add_scatter(x=[ctx["per"]], y=[ctx["obs"]], mode="markers",
                        marker=dict(size=taille, color=c), showlegend=False,
                        hoverinfo="skip", row=1, col=1)
    fig.add_scatter(x=[ctx["per"]], y=[ctx["obs"]], mode="markers",
                    marker=dict(size=13, color=coul,
                                line=dict(color="white", width=2.4)),
                    name="Hors intervalle" if not ctx["couvert"] else "Couvert",
                    hovertemplate=f"<b>{ctx['per']}</b><br>"
                                  f"Observe : {fmt(ctx['obs'])}<br>"
                                  + ("HORS intervalle" if not ctx["couvert"]
                                     else "Couvert")
                                  + "<extra></extra>", row=1, col=1)

for i, v in enumerate(secondaires, start=2):
    c = DOUX[(i - 2) % len(DOUX)]
    vv = h[v].values.astype(float)
    fig.add_scatter(x=per, y=vv, mode="lines+markers",
                    line=dict(color=c, width=2.2, shape="spline", smoothing=.55),
                    marker=dict(size=6, color="white",
                                line=dict(color=c, width=1.8)),
                    name=v, showlegend=False,
                    # ligne reconstruite : fin coupee sur votre 3e capture
                    hovertemplate=f"<b>%{{x}}</b><br>{v} : <b>%{{y:,.4g}}</b>"
                                  "<extra></extra>",
                    row=i, col=1)
    fig.add_annotation(xref="paper", x=1.005, y=vv[-1], xanchor="left",
                       text=f"<b>{v}</b>", showarrow=False,
                       font=dict(size=10, color=c), row=i, col=1)

fig.update_xaxes(showgrid=False, showspikes=True, spikemode="across",
                 spikethickness=1.2, spikedash="dot", spikecolor=GRIS,
                 tickfont=dict(size=11), linecolor=GRILLE)
fig.update_yaxes(gridcolor=GRILLE, zeroline=False, showspikes=True,
                 spikemode="across", spikethickness=1.2, spikedash="dot",
                 spikecolor=GRIS, tickformat=",.0f", tickfont=dict(size=11))
fig.update_yaxes(title_text=f"<b>{TARGET}</b>", title_font=dict(size=12),
                 row=1, col=1)
fig.update_xaxes(title_text="<b>Trimestre</b>", title_font=dict(size=12),
                 row=n_rows, col=1)

delta = 100 * (val[-1] - val[0]) / abs(val[0]) if val[0] else np.nan
fleche = "▲" if delta >= 0 else "▼"

fig.update_layout(
    title=dict(text=f"<b style='font-size:19px;color:{ENCRE}'>{nom}</b>"
                    f"<br><span style='font-size:12px;color:{GRIS}'>"
                    f"{TARGET} · {len(h)} trimestres · "
                    f"{per[0]} → {per[-1]} · "
                    f"<span style='color:{ACCENT if delta < 0 else OK}'>{fleche} "
                    f"{abs(delta):.1f} %</span> sur la periode</span>",
               x=.015, xanchor="left", y=.96),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="white", bordercolor=GRILLE,
                    font=dict(size=12.5, family="Inter, system-ui, sans-serif",
                              color=ENCRE), align="left"),
    template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color=GRIS),
    height=430 + 130 * len(secondaires),
    legend=dict(orientation="h", y=1.04, x=1, xanchor="right",
                bgcolor="rgba(255,255,255,0)", font=dict(size=11)),
    margin=dict(l=80, r=110, t=120, b=60),
    # --- MODIFICATION 2 ------------------------------------------------------
    #  Format francais cote Plotly. Les %{y:,.0f} des hovertemplate et le
    #  tickformat des axes sont calcules par le navigateur, pas par Python :
    #  fmt() ne les touche pas. Cette ligne les regle tous d'un coup, virgule
    #  pour les decimales et espace pour les milliers.
    # -------------------------------------------------------------------------
    separators=", ")

fig.show()

print(f"Rang #{RANG} · {nom}")
print(f"Min {fmt(val.min())} | Median {fmt(np.median(val))} | "
      f"Max {fmt(val.max())}")
if ctx:
    print(f"Periode validee {ctx['per']} : observe {fmt(ctx['obs'])} | "
          f"predit {fmt(ctx['pred'])} | "
          f"intervalle [{fmt(ctx['lo'])} ; {fmt(ctx['hi'])}]"
          f" -> {'COUVERT' if ctx['couvert'] else 'HORS INTERVALLE'}")








Mission 2





# -*- coding: utf-8 -*-
# =============================================================================
#  MISSION 2 - TIME EVOLUTION OF THE FIRST IMPORTANT VARIABLES
#
#  Enonce :
#    - SHAP local (contribution / decomposition) pour identifier les 5 variables
#      les plus determinantes dans la prediction de la target ;
#    - visualiser leur evolution sur les 10 derniers trimestres et sur la
#      periode a valider, comme pour la target.
#
#  A EXECUTER JUSTE APRES la cellule de la mission 1 : reutilise UNITE, h, per,
#  ctx et nom, deja construits pour le sous-portefeuille choisi.
#  Prerequis : pip install shap
# =============================================================================

import joblib
import numpy as np
import pandas as pd
import shap
from plotly.subplots import make_subplots

# ┌──────────────────────────── PARAMETRES ────────────────────────────┐
N_VARS  = 5                                    # enonce : 5 variables
CHEMIN  = "artefacts_modele/modele_final.joblib"
#  MODELE et X_MODEL sont repris de la session s'ils y existent deja. Pour les
#  fixer, definissez-les dans une cellule AVANT celle-ci (MODELE = mon_modele),
#  plutot que d'editer ces deux lignes.
MODELE  = globals().get("MODELE")    # None -> charge l'artefact CHEMIN
X_MODEL = globals().get("X_MODEL")   # features exactes de la prediction, indexees
                                     # comme df ; None -> reconstruites depuis h
# └─────────────────────────────────────────────────────────────────────┘

for _v in ("h", "per", "nom"):
    if _v not in globals():
        raise NameError(f"`{_v}` absent : executez d'abord la cellule mission 1.")

if "fmt" not in globals():
    def fmt(v):
        return f"{v:,.0f}".replace(",", " ")

ENCRE  = globals().get("ENCRE", "#141B34")
GRIS   = globals().get("GRIS", "#8A93A5")
GRILLE = globals().get("GRILLE", "#EDF1F7")
ACCENT = globals().get("ACCENT", "#FF5A5F")
DOUX   = globals().get("DOUX", ["#6C8EBF", "#82B366", "#C08552",
                                "#9673A6", "#5F9EA0"])


# %% -------------------------------------------------------------------------
#  1. SHAP LOCAL : decomposition de LA prediction du sous-portefeuille
# -----------------------------------------------------------------------------
if MODELE is None:
    _art = joblib.load(CHEMIN)
    MODELE = (next(v for v in _art.values() if hasattr(v, "predict"))
              if isinstance(_art, dict) else _art)
mdl = MODELE.named_steps["model"] if hasattr(MODELE, "named_steps") else MODELE
FEATS = (list(mdl.feature_name_) if hasattr(mdl, "feature_name_")
         else list(mdl.feature_names_in_))

ligne = h.iloc[[-1]]                       # dernier trimestre = periode a valider
if X_MODEL is not None:
    x = X_MODEL.loc[ligne.index, FEATS]
else:
    absentes = [f for f in FEATS if f not in ligne.columns]
    if absentes:
        raise KeyError(f"{len(absentes)} variable(s) du modele absente(s) de df "
                       f"(ex. {absentes[:3]}). Renseignez X_MODEL.")
    x = ligne[FEATS].copy()
    for c in x.columns:                    # LightGBM natif attend des category
        if x[c].dtype == object:
            x[c] = x[c].astype("category")

explainer = shap.TreeExplainer(mdl)
sv = np.asarray(explainer.shap_values(x)).ravel()
base = float(np.ravel(explainer.expected_value)[0])
contrib = pd.Series(sv, index=FEATS)

#  Controle de decomposition : base + somme des contributions doit redonner la
#  prediction du modele. C'est ce qui rend l'explication opposable en soutenance.
pred_mdl = float(mdl.predict(x)[0])
ecart = abs(base + contrib.sum() - pred_mdl)

print(f"{nom}  |  periode expliquee {per[-1]}")
print(f"Valeur de base (moyenne du modele) : {fmt(base)}")
print(f"Somme des contributions SHAP       : {fmt(contrib.sum())}")
print(f"Prediction reconstituee            : {fmt(base + contrib.sum())}")
print(f"Prediction du modele               : {fmt(pred_mdl)}")
print(f"Ecart de reconstitution            : {ecart:.6g}"
      f"   {'(decomposition exacte)' if ecart < 1e-6 * max(abs(pred_mdl), 1) else '(A VERIFIER)'}")

top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(N_VARS)
part = 100 * top.abs() / contrib.abs().sum()

print(f"\nLes {N_VARS} variables les plus determinantes pour cette prediction :")
recap = pd.DataFrame({
    "valeur": [ligne[v].iloc[0] if v in ligne.columns else np.nan for v in top.index],
    "contribution": top.values,
    "sens": np.where(top.values >= 0, "pousse a la hausse", "pousse a la baisse"),
    "part_abs_%": part.round(1).values}, index=top.index)
print(recap.to_string(float_format=lambda v: f"{v:,.2f}".replace(",", " ")))


# %% -------------------------------------------------------------------------
#  2. EVOLUTION DE CES 5 VARIABLES, meme principe que la target
# -----------------------------------------------------------------------------
tracables = [v for v in top.index
             if v in h.columns and pd.api.types.is_numeric_dtype(h[v])]
ecartees = [v for v in top.index if v not in tracables]
if ecartees:
    print(f"\nNon tracables (categorielles ou absentes de l'historique) : {ecartees}")
if not tracables:
    raise ValueError("Aucune des variables retenues n'est numerique et suivie "
                     "dans l'historique : rien a tracer.")

k_valid = per.index(ctx["per"]) if globals().get("ctx") and ctx["per"] in per else len(per) - 1

fig2 = make_subplots(rows=len(tracables), cols=1, shared_xaxes=True,
                     vertical_spacing=.05)

for i, v in enumerate(tracables, start=1):
    c = DOUX[(i - 1) % len(DOUX)]
    vv = h[v].values.astype(float)
    fig2.add_vrect(x0=k_valid - .5, x1=k_valid + .5,
                   fillcolor="rgba(99,110,250,0.055)", line_width=0,
                   layer="below", row=i, col=1)
    fig2.add_scatter(x=per, y=vv, mode="lines+markers",
                     line=dict(color=c, width=2.4, shape="spline", smoothing=.55),
                     marker=dict(size=7, color="white",
                                 line=dict(color=c, width=1.9)),
                     name=v, showlegend=False,
                     hovertemplate=f"<b>%{{x}}</b><br>{v} : <b>%{{y:,.4g}}</b>"
                                   "<extra></extra>", row=i, col=1)
    fig2.add_scatter(x=[per[k_valid]], y=[vv[k_valid]], mode="markers",
                     marker=dict(size=13, color=c,
                                 line=dict(color="white", width=2.4)),
                     showlegend=False, hoverinfo="skip", row=i, col=1)
    signe = "+" if top[v] >= 0 else "-"
    fig2.add_annotation(xref="paper", x=1.008, y=vv[k_valid], xanchor="left",
                        text=f"<b>{v}</b><br>"
                             f"<span style='font-size:10px;color:{GRIS}'>"
                             f"SHAP {signe}{fmt(abs(top[v]))} "
                             f"({part[v]:.0f} %)</span>",
                        showarrow=False, align="left",
                        font=dict(size=11, color=c), row=i, col=1)

fig2.update_xaxes(showgrid=False, showticklabels=False, linecolor=GRILLE)
fig2.update_xaxes(title_text="<b>Trimestre</b>", title_font=dict(size=12),
                  showticklabels=True, tickfont=dict(size=11),
                  row=len(tracables), col=1)
fig2.update_yaxes(gridcolor=GRILLE, zeroline=False, tickformat=",.4g",
                  tickfont=dict(size=10))

fig2.update_layout(
    title=dict(text=f"<b style='font-size:19px;color:{ENCRE}'>{nom}</b>"
                    f"<br><span style='font-size:12px;color:{GRIS}'>"
                    f"{len(tracables)} variables les plus determinantes "
                    f"(SHAP local) · {len(h)} trimestres · {per[0]} → {per[-1]} · "
                    f"periode a valider <b style='color:{ACCENT}'>{per[k_valid]}"
                    f"</b> surlignee</span>", x=.015, xanchor="left", y=.97),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="white", bordercolor=GRILLE, align="left",
                    font=dict(size=12.5, family="Inter, system-ui, sans-serif",
                              color=ENCRE)),
    template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color=GRIS),
    height=150 * len(tracables) + 150,
    margin=dict(l=80, r=150, t=125, b=60), separators=", ")

fig2.show()
