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






























Tavbleua













print(f"results_v2      : {len(results_v2):>10,} lignes")
print(f"anomalies       : {len(anomalies):>10,} lignes" if "anomalies" in globals() else "anomalies : ABSENT")
print(f"anomalies_prio  : {len(anomalies_prio):>10,} lignes")
print(f"expl            : {len(expl):>10,} lignes" if "expl" in globals() else "expl : ABSENT")

if "expl" in globals():
    ratio = len(expl) / len(results_v2)
    print(f"\nRatio expl / results_v2 : {ratio:.2f}")
    print("  -> " + ("OK, pas d'explosion" if ratio < 1.05
                     else "EXPLOSION DE LA FUSION — c'est la cause du blocage"))

cles = [c for c in ID_COLS if c in anomalies_prio.columns]
cles += [c for c in ["year", "quarter"] if c in anomalies_prio.columns]
dup = anomalies_prio.duplicated(subset=cles).sum()
print(f"\nCles de fusion : {cles}")
print(f"Doublons dans anomalies_prio : {dup}"
      + ("  -> CAUSE CONFIRMEE" if dup else "  -> OK"))



rEMPLACER EXPLSIDOIBLONS 

cles = [c for c in ID_COLS if c in results_v2.columns and c in anomalies_prio.columns]
cles += [c for c in ["year", "quarter"]
         if c in results_v2.columns and c in anomalies_prio.columns]

# CONDITION : on dedoublonne AVANT la fusion, sinon explosion combinatoire
ap = (anomalies_prio.sort_values("score_composite", ascending=False)
                    .drop_duplicates(subset=cles, keep="first"))
print(f"anomalies_prio : {len(anomalies_prio)} -> {len(ap)} apres dedoublonnage")

expl = results_v2.merge(
    ap[cles + ["score_composite", "rank", "A_ecart_borne", "B_erreur_modele"]],
    on=cles, how="left", validate="one_to_one")     # leve une erreur si ca explose

centre = (expl["borne_haute"] + expl["borne_basse"]) / 2
demi   = np.maximum((expl["borne_haute"] - expl["borne_basse"]) / 2, 1e-9)
expl["z"]            = (expl["y_obs"] - centre) / demi
expl["largeur"]      = expl["borne_haute"] - expl["borne_basse"]
expl["largeur_rel"]  = expl["largeur"] / np.maximum(expl["y_pred"].abs(), 1e-9)
expl["est_anomalie"] = (~expl["dans_intervalle"]).astype(int)
expl["statut"]       = np.where(expl["dans_intervalle"], "Couverte", "Hors intervalle")
expl["identite"]     = expl[[c for c in ID_COLS if c in expl.columns]].astype(str).agg(" | ".join, axis=1)

print(f"expl : {len(expl):,} lignes (doit egaler results_v2 = {len(results_v2):,})")














import numpy as np, pandas as pd

# ┌─── PARAMETRE ───┐
GWP_COL = "EARNED_PREMIUM"     # ajustez si le nom differe dans vos donnees
# └─────────────────┘

if GWP_COL not in results_v2.columns:
    cand = [c for c in results_v2.columns
            if any(k in c.upper() for k in ("PREMIUM", "GWP", "EARNED"))]
    raise KeyError(f"'{GWP_COL}' absente de results_v2. Candidates : {cand}")

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

# CONDITION : dedoublonnage AVANT la fusion, sinon explosion combinatoire
ap = (anomalies_prio.sort_values("score_composite", ascending=False)
                    .drop_duplicates(subset=cles, keep="first"))
if len(ap) < len(anomalies_prio):
    print(f"/!\\ {len(anomalies_prio)-len(ap)} doublons sur {cles} -> retires pour la fusion")

expl = results_v2.merge(
    ap[cles + ["score_composite", "rank", "A_ecart_borne", "B_erreur_modele"]],
    on=cles, how="left", validate="one_to_one")     # erreur immediate si ca explose

centre = (expl["borne_haute"] + expl["borne_basse"]) / 2
demi   = np.maximum((expl["borne_haute"] - expl["borne_basse"]) / 2, 1e-9)
expl["z"]            = (expl["y_obs"] - centre) / demi
expl["largeur"]      = expl["borne_haute"] - expl["borne_basse"]
expl["largeur_rel"]  = expl["largeur"] / np.maximum(expl["y_pred"].abs(), 1e-9)
expl["est_anomalie"] = (~expl["dans_intervalle"]).astype(int)
expl["statut"]       = np.where(expl["dans_intervalle"], "Couverte", "Hors intervalle")
expl["identite"]     = expl[id_ok].astype(str).agg(" | ".join, axis=1)

print(f"expl : {len(expl):,} lignes  (results_v2 = {len(results_v2):,})")
print(f"Scores rattaches : {expl.score_composite.notna().sum():,}")
print(f"Anomalies : {int(expl.est_anomalie.sum()):,} "
      f"({100*expl.est_anomalie.mean():.1f} %)")

















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

    # --- 1. Cercle ---
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

    # --- Agregation a la granularite choisie ---
    st = (sub.groupby(granularite, observed=True)
            .agg(score_total=("score_composite", "sum"),
                 score_moyen=("score_composite", "mean"),
                 n_groupe=("score_composite", "size")).reset_index())
    pires = sub.loc[sub.groupby(granularite, observed=True)["score_composite"].idxmax()]
    a = pires.merge(st, on=granularite, how="left")
    a["lab"] = a[granularite].astype(str).agg(" | ".join, axis=1).str.slice(0, 34)
    top = a.nlargest(min(top_n, len(a)), "score_total").iloc[::-1].reset_index(drop=True)

    # --- 2. Barres ---
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

    # --- 3. Forest plot ---
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

    # --- 4. Score vs exposition ---
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


# ------------------------------- selecteurs

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
