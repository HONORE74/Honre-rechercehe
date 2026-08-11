Celluele A

# ================================================================
# FIGURE CANONIQUE CQR — recette du tutoriel officiel MAPIE
#   X : valeur observee (vraie)
#   Y : prediction, avec barre d'erreur = intervalle CQR individuel
#   Diagonale x=y : une barre qui la touche = observation couverte
#   Bleu = couverte / Rouge = non couverte
#   Taille des points rouges proportionnelle au score de priorisation
# ================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go

N_AFFICHE = 350      # nb d'observations tracees (MAPIE en trace 2 %)
LOG_AXES  = True


def _injecter_score(results_v2, anomalies_prio):
    """Rapatrie score_composite dans results_v2 (NaN pour les non-anomalies)."""
    cles = [c for c in ID_COLS if c in results_v2.columns and c in anomalies_prio.columns]
    for c in ["year", "quarter"]:
        if c in results_v2.columns and c in anomalies_prio.columns:
            cles.append(c)
    if "score_composite" not in anomalies_prio.columns or not cles:
        return results_v2.assign(score_composite=np.nan, rank=np.nan)
    return results_v2.merge(anomalies_prio[cles + ["score_composite", "rank"]],
                            on=cles, how="left")


def figure_cqr_canonique(results_v2, anomalies_prio, n_affiche=N_AFFICHE,
                         log_axes=LOG_AXES, random_state=42):
    d = _injecter_score(results_v2, anomalies_prio)
    d = d.dropna(subset=["y_obs", "y_pred", "borne_basse", "borne_haute"])

    # Statistiques calculees sur TOUT le test, pas sur l'echantillon affiche
    couverture = d["dans_intervalle"].mean()
    largeur_moy = (d["borne_haute"] - d["borne_basse"]).mean()
    n_total = len(d)

    if len(d) > n_affiche:
        d = d.sample(n_affiche, random_state=random_state)
    d = d.reset_index(drop=True)

    obs = d["y_obs"].values.astype(float)
    pred = d["y_pred"].values.astype(float)
    lo = d["borne_basse"].values.astype(float)
    hi = d["borne_haute"].values.astype(float)
    dedans = d["dans_intervalle"].values.astype(bool)

    # Taille des points rouges : croissante avec le score de priorisation
    sc = d["score_composite"].values.astype(float)
    sc_out = sc[~dedans]
    if np.isfinite(sc_out).any():
        r = pd.Series(sc_out).rank(pct=True).fillna(0.5).values
        taille_out = 7 + 13 * r
    else:
        taille_out = np.full(int((~dedans).sum()), 9.0)

    def _hover(idx):
        r = d.iloc[idx]
        ident = " | ".join(str(r[c]) for c in ID_COLS if c in d.columns)
        t = (f"<b>{ident}</b><br>"
             f"Observe    : {r['y_obs']:,.0f}<br>"
             f"Predit     : {r['y_pred']:,.0f}<br>"
             f"Intervalle : [{r['borne_basse']:,.0f} ; {r['borne_haute']:,.0f}]<br>"
             f"Largeur    : {r['borne_haute']-r['borne_basse']:,.0f}")
        if pd.notna(r.get("score_composite", np.nan)):
            t += f"<br>Rang priorite : #{int(r['rank'])}  (score {r['score_composite']:.4g})"
        return t

    log_ok = log_axes and (obs > 0).all() and (pred > 0).all() and (lo > 0).all()

    fig = go.Figure()

    lim = [min(obs.min(), lo.min()), max(obs.max(), hi.max())]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                             line=dict(color="black", width=1.6, dash="dash"),
                             name="y = x (prediction parfaite)", hoverinfo="skip"))

    idx_in = np.where(dedans)[0]
    fig.add_trace(go.Scatter(
        x=obs[dedans], y=pred[dedans], mode="markers",
        error_y=dict(type="data", symmetric=False,
                     array=hi[dedans] - pred[dedans],
                     arrayminus=pred[dedans] - lo[dedans],
                     color="rgba(31,105,224,0.45)", thickness=1.1, width=0),
        marker=dict(size=6, color="#1769E0"),
        name=f"Observation couverte ({dedans.sum()})",
        text=[_hover(i) for i in idx_in],
        hovertemplate="%{text}<extra></extra>"))

    idx_out = np.where(~dedans)[0]
    fig.add_trace(go.Scatter(
        x=obs[~dedans], y=pred[~dedans], mode="markers",
        error_y=dict(type="data", symmetric=False,
                     array=hi[~dedans] - pred[~dedans],
                     arrayminus=pred[~dedans] - lo[~dedans],
                     color="rgba(229,57,53,0.6)", thickness=1.4, width=0),
        marker=dict(size=taille_out, color="#E53935",
                    line=dict(width=0.6, color="#8e0000")),
        name=f"Observation NON couverte ({(~dedans).sum()})",
        text=[_hover(i) for i in idx_out],
        hovertemplate="%{text}<extra></extra>"))

    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98, xanchor="left", yanchor="top",
        text=(f"<b>Couverture : {100*couverture:.1f} %</b>  (cible {100*(1-ALPHA):.0f} %)<br>"
              f"Largeur moyenne : {largeur_moy:,.0f}<br>"
              f"<sub>calcule sur {n_total:,} obs. — {len(d)} affichees</sub>"),
        showarrow=False, align="left", bgcolor="rgba(255,255,255,0.9)",
        bordercolor="black", borderwidth=1, borderpad=6, font=dict(size=11))

    fig.update_layout(
        title="Conformalized Quantile Regression — intervalles individuels"
              "<br><sup>Une barre qui touche la diagonale = observation couverte. "
              "Taille des points rouges = priorite d'investigation</sup>",
        xaxis=dict(title="Valeur observee" + ("  (log)" if log_ok else ""),
                   type="log" if log_ok else "linear"),
        yaxis=dict(title="Prediction et intervalle CQR" + ("  (log)" if log_ok else ""),
                   type="log" if log_ok else "linear"),
        template="plotly_white", height=720, hovermode="closest",
        legend=dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="black", borderwidth=1))
    fig.show()





Cellule B

# ================================================================
# VUE NORMALISEE — la seule "uniformisation" methodologiquement valide
#   z = (y_obs - centre) / demi-largeur
#   |z| <= 1 : couverte   |z| > 1 : hors intervalle
#   Aucune borne n'est modifiee : c'est un changement de repere
# ================================================================

def figure_normalisee_cqr(results_v2, anomalies_prio, n_affiche=600, random_state=42):
    d = _injecter_score(results_v2, anomalies_prio)
    d = d.dropna(subset=["y_obs", "y_pred", "borne_basse", "borne_haute"])
    if len(d) > n_affiche:
        d = d.sample(n_affiche, random_state=random_state)
    d = d.sort_values("y_pred").reset_index(drop=True)

    centre = (d["borne_haute"].values + d["borne_basse"].values) / 2
    demi = np.maximum((d["borne_haute"].values - d["borne_basse"].values) / 2, 1e-9)
    z = (d["y_obs"].values - centre) / demi
    xs = d["y_pred"].values.astype(float)
    dedans = d["dans_intervalle"].values.astype(bool)
    log_ok = (xs > 0).all()

    def _hov(i):
        r = d.iloc[i]
        ident = " | ".join(str(r[c]) for c in ID_COLS if c in d.columns)
        t = (f"<b>{ident}</b><br>z = {z[i]:.2f}<br>"
             f"Observe : {r['y_obs']:,.0f}<br>Predit : {r['y_pred']:,.0f}<br>"
             f"Intervalle : [{r['borne_basse']:,.0f} ; {r['borne_haute']:,.0f}]")
        if pd.notna(r.get("score_composite", np.nan)):
            t += f"<br>Rang priorite : #{int(r['rank'])}"
        return t

    fig = go.Figure()
    fig.add_hrect(y0=-1, y1=1, fillcolor="rgba(70,110,230,0.18)", line_width=0,
                  annotation_text=f"Zone conforme ({100*(1-ALPHA):.0f} %)",
                  annotation_position="top left")
    fig.add_hline(y=1, line=dict(color="#355CDE", width=2))
    fig.add_hline(y=-1, line=dict(color="#355CDE", width=2))
    fig.add_hline(y=0, line=dict(color="black", width=1.4))

    fig.add_trace(go.Scatter(
        x=xs[dedans], y=z[dedans], mode="markers",
        marker=dict(size=5, color="#1769E0", opacity=0.7),
        name=f"Couverte ({dedans.sum()})",
        text=[_hov(i) for i in np.where(dedans)[0]],
        hovertemplate="%{text}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=xs[~dedans], y=z[~dedans], mode="markers",
        marker=dict(size=8, color="#E53935", line=dict(width=0.6, color="#8e0000")),
        name=f"Hors intervalle ({(~dedans).sum()})",
        text=[_hov(i) for i in np.where(~dedans)[0]],
        hovertemplate="%{text}<extra></extra>"))

    fig.update_layout(
        title="Vue normalisee — tous les intervalles ramenes a [-1 ; +1]"
              "<br><sup>Changement de repere : aucune borne n'est modifiee. "
              "|z| se lit comme la severite</sup>",
        xaxis=dict(title="Prediction" + ("  (log)" if log_ok else ""),
                   type="log" if log_ok else "linear"),
        yaxis=dict(title="Position normalisee  z"),
        template="plotly_white", height=620, hovermode="closest")
    fig.show()


figure_normalisee_cqr(results_v2, anomalies_prio)





Cellule C

# ================================================================
# DIAGNOSTICS MAPIE — couverture conditionnelle et adaptativite
#   Gauche : la couverture tient-elle sur toute la plage de valeurs ?
#   Droite : les intervalles s'elargissent-ils avec le montant ?
#            (c'est l'avantage du CQR sur le split conformal)
# ================================================================

from plotly.subplots import make_subplots


def diagnostics_cqr(results_v2, n_bins=5):
    d = results_v2.dropna(subset=["y_obs", "borne_basse", "borne_haute"]).copy()
    d["largeur"] = d["borne_haute"] - d["borne_basse"]
    d["strate"] = pd.qcut(d["y_obs"], q=n_bins, duplicates="drop")

    g = d.groupby("strate", observed=True).agg(
        couverture=("dans_intervalle", "mean"),
        largeur=("largeur", "mean"),
        n=("y_obs", "size")).reset_index()
    labels = [f"[{i.left:,.0f} ; {i.right:,.0f}]" for i in g["strate"]]

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Couverture par strate de montant", "Largeur moyenne de l'intervalle"))

    couleurs = ["#2e7d32" if c >= 1 - ALPHA else "#E53935" for c in g["couverture"]]
    fig.add_trace(go.Bar(x=labels, y=g["couverture"], marker_color=couleurs,
                         text=[f"{100*c:.1f} %<br><sub>n={n}</sub>"
                               for c, n in zip(g["couverture"], g["n"])],
                         textposition="outside", showlegend=False), row=1, col=1)
    fig.add_hline(y=1 - ALPHA, line=dict(color="black", dash="dash", width=2),
                  annotation_text=f"cible {100*(1-ALPHA):.0f} %", row=1, col=1)

    fig.add_trace(go.Bar(x=labels, y=g["largeur"], marker_color="#355CDE",
                         text=[f"{w:,.0f}" for w in g["largeur"]],
                         textposition="outside", showlegend=False), row=1, col=2)

    fig.update_yaxes(title_text="Couverture", range=[0, 1.12], row=1, col=1)
    fig.update_yaxes(title_text="Largeur moyenne", row=1, col=2)
    fig.update_xaxes(tickangle=30, row=1, col=1)
    fig.update_xaxes(tickangle=30, row=1, col=2)
    fig.update_layout(
        title="Validation du CQR — couverture conditionnelle et adaptativite des intervalles"
              "<br><sup>Vert : couverture atteinte | Rouge : couverture insuffisante sur la strate</sup>",
        template="plotly_white", height=520)
    fig.show()

    print("Barres de droite croissantes  -> intervalles adaptatifs : le CQR joue son role.")
    print("Barres de droite plates       -> pas d'adaptativite : un split conformal suffirait.")
    croiss = g["largeur"].is_monotonic_increasing
    print(f"\nLargeur strictement croissante avec le montant : {croiss}")
    print(f"Rapport largeur derniere strate / premiere : "
          f"{g['largeur'].iloc[-1]/max(g['largeur'].iloc[0], 1e-9):.1f}x")


diagnostics_cqr(results_v2)


Cellule D


# ================================================================
# BLOC 0 — PREPARATION (a executer en premier)
# ================================================================

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


def preparer_exploration(results_v2, anomalies_prio):
    """DataFrame unique enrichi : z normalise, score, statut."""
    cles = [c for c in ID_COLS if c in results_v2.columns and c in anomalies_prio.columns]
    for c in ["year", "quarter"]:
        if c in results_v2.columns and c in anomalies_prio.columns:
            cles.append(c)

    d = results_v2.copy()
    if "score_composite" in anomalies_prio.columns and cles:
        d = d.merge(anomalies_prio[cles + ["score_composite", "rank",
                                           "A_ecart_borne", "B_erreur_modele"]],
                    on=cles, how="left")
    else:
        for c in ["score_composite", "rank", "A_ecart_borne", "B_erreur_modele"]:
            d[c] = np.nan

    centre = (d["borne_haute"] + d["borne_basse"]) / 2
    demi = np.maximum((d["borne_haute"] - d["borne_basse"]) / 2, 1e-9)
    d["z"] = (d["y_obs"] - centre) / demi
    d["largeur"] = d["borne_haute"] - d["borne_basse"]
    d["largeur_rel"] = d["largeur"] / np.maximum(d["y_pred"].abs(), 1e-9)
    d["est_anomalie"] = (~d["dans_intervalle"]).astype(int)
    d["statut"] = np.where(d["dans_intervalle"], "Couverte", "Hors intervalle")
    d["identite"] = d[[c for c in ID_COLS if c in d.columns]].astype(str).agg(" | ".join, axis=1)
    return d


expl = preparer_exploration(results_v2, anomalies_prio)
print(f"Base d'exploration : {len(expl):,} observations, "
      f"{expl['est_anomalie'].sum():,} anomalies")


Cellule E

# ================================================================
# FIGURE 1 — SUNBURST HIERARCHIQUE (cliquez pour explorer en profondeur)
#   Anneaux : Lob -> Partner -> Risk
#   Taille   : exposition financiere
#   Couleur  : taux d'anomalie du segment
# ================================================================

def sunburst_anomalies(d, chemin=None, valeur=None):
    chemin = chemin or [c for c in ["Lob", "Partner", "Risk", "Activity"]
                        if c in d.columns][:3]
    valeur = valeur or (GWP_COL if GWP_COL in d.columns else None)

    dd = d.dropna(subset=chemin).copy()
    for c in chemin:
        dd[c] = dd[c].astype(str)
    if valeur:
        dd = dd[dd[valeur] > 0]

    fig = px.sunburst(
        dd, path=chemin, values=valeur,
        color="est_anomalie", color_continuous_scale="RdYlGn_r",
        range_color=[0, max(0.35, dd["est_anomalie"].mean() * 3)],
        custom_data=["est_anomalie"])
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Exposition : %{value:,.0f}<br>"
                      "Taux d'anomalie : %{color:.1%}<extra></extra>",
        insidetextorientation="radial")
    fig.update_layout(
        title="Ou se concentrent les anomalies ?"
              f"<br><sup>{' -> '.join(chemin)}  |  taille = exposition, "
              "couleur = taux d'anomalie  |  cliquez pour explorer</sup>",
        template="plotly_white", height=720,
        coloraxis_colorbar=dict(title="Taux<br>anomalie", tickformat=".0%"))
    fig.show()


sunburst_anomalies(expl)



Cellule F

# ================================================================
# FIGURE 2 — COORDONNEES PARALLELES (glissez sur un axe pour filtrer)
#   Chaque ligne = une anomalie. Le filtrage est natif et instantane.
# ================================================================

def coordonnees_paralleles(anomalies_prio, top_n=400):
    d = anomalies_prio.head(top_n).copy()

    dims = []
    for lab, col, log in [("Ecart borne  A", "A_ecart_borne", False),
                          ("Erreur modele  B", "B_erreur_modele", False),
                          (f"{GWP_COL}", GWP_COL, True),
                          ("Score composite", "score_composite", True)]:
        if col not in d.columns:
            continue
        v = d[col].astype(float)
        if log and (v > 0).all():
            v = np.log10(v)
            lab += "  (log10)"
        dims.append(dict(label=lab, values=v))

    fig = go.Figure(data=go.Parcoords(
        line=dict(color=d["score_composite"].rank(pct=True),
                  colorscale="Turbo", showscale=True,
                  colorbar=dict(title="Rang<br>priorite", tickformat=".0%")),
        dimensions=dims,
        labelfont=dict(size=12), tickfont=dict(size=10)))
    fig.update_layout(
        title=f"Profil des {len(d)} anomalies prioritaires"
              "<br><sup>Glissez verticalement sur un axe pour isoler une plage. "
              "Deplacez les axes pour les reordonner</sup>",
        template="plotly_white", height=560, margin=dict(l=90, r=70, t=110, b=40))
    fig.show()


coordonnees_paralleles(anomalies_prio)




Celle EERR
# ================================================================
# FIGURE 3 — VIOLONS + POINTS par categorie
#   Distribution de la position normalisee z, avec chaque observation
#   visible. La zone conforme [-1 ; +1] est materialisee.
# ================================================================

def violons_par_categorie(d, categorie=None, n_cat=6, n_points=2000):
    categorie = categorie or next((c for c in ["Lob", "Risk", "Partner"]
                                   if c in d.columns), ID_COLS[0])
    top = d[categorie].value_counts().head(n_cat).index.tolist()
    dd = d[d[categorie].isin(top)].copy()
    if len(dd) > n_points:
        dd = dd.sample(n_points, random_state=42)
    dd[categorie] = dd[categorie].astype(str)

    fig = go.Figure()
    fig.add_hrect(y0=-1, y1=1, fillcolor="rgba(70,110,230,0.12)", line_width=0,
                  annotation_text=f"Zone conforme ({100*(1-ALPHA):.0f} %)",
                  annotation_position="top left")
    fig.add_hline(y=1, line=dict(color="#355CDE", width=1.6, dash="dash"))
    fig.add_hline(y=-1, line=dict(color="#355CDE", width=1.6, dash="dash"))

    for i, cat in enumerate(top):
        sub = dd[dd[categorie] == str(cat)]
        cov = sub["dans_intervalle"].mean()
        fig.add_trace(go.Violin(
            x=[str(cat)] * len(sub), y=sub["z"],
            name=f"{cat}  ({100*cov:.0f} %)",
            box_visible=True, meanline_visible=True,
            points="all", pointpos=0, jitter=0.35,
            marker=dict(size=3, opacity=0.45,
                        color=np.where(sub["dans_intervalle"], "#1769E0", "#E53935")),
            line_color=PALETTE[i % len(PALETTE)],
            fillcolor=PALETTE[i % len(PALETTE)], opacity=0.55,
            hovertemplate="z = %{y:.2f}<extra></extra>"))

    fig.update_layout(
        title=f"Distribution de la position normalisee par {categorie}"
              "<br><sup>Chaque point est une observation. "
              "Le pourcentage en legende est la couverture de la categorie</sup>",
        yaxis=dict(title="Position normalisee  z",
                   range=[max(-6, dd["z"].quantile(0.002)),
                          min(8, dd["z"].quantile(0.998))]),
        xaxis=dict(title=categorie),
        template="plotly_white", height=650, violingap=0.25)
    fig.show()


violons_par_categorie(expl)






Celleop P

# ================================================================
# FIGURE 4 — BULLES : severite vs incertitude du modele
#   X : largeur relative de l'intervalle (incertitude du CQR)
#   Y : position normalisee z (severite de l'ecart)
#   Taille : exposition | Couleur : categorie (cliquez la legende pour filtrer)
# ================================================================

def bulles_severite(d, categorie=None, n_cat=8, n_points=1200):
    categorie = categorie or next((c for c in ["Lob", "Risk", "Partner"]
                                   if c in d.columns), ID_COLS[0])
    top = d[categorie].value_counts().head(n_cat).index.tolist()
    dd = d[d[categorie].isin(top)].copy()
    if len(dd) > n_points:
        dd = dd.sample(n_points, random_state=42)
    dd[categorie] = dd[categorie].astype(str)

    taille = (dd[GWP_COL].rank(pct=True) * 26 + 4) if GWP_COL in dd.columns else 9

    fig = go.Figure()
    fig.add_hrect(y0=-1, y1=1, fillcolor="rgba(70,110,230,0.10)", line_width=0)
    fig.add_hline(y=1, line=dict(color="#355CDE", width=1.4, dash="dash"))
    fig.add_hline(y=-1, line=dict(color="#355CDE", width=1.4, dash="dash"))

    for i, cat in enumerate(top):
        m = dd[categorie] == str(cat)
        sub = dd[m]
        fig.add_trace(go.Scatter(
            x=sub["largeur_rel"], y=sub["z"], mode="markers",
            name=str(cat),
            marker=dict(size=taille[m] if hasattr(taille, "__len__") else taille,
                        color=PALETTE[i % len(PALETTE)], opacity=0.6,
                        line=dict(width=0.4, color="white")),
            customdata=np.column_stack([sub["identite"], sub["y_obs"],
                                        sub["y_pred"], sub["statut"]]),
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "z = %{y:.2f}  |  largeur rel. = %{x:.1%}<br>"
                          "Observe : %{customdata[1]:,.0f}<br>"
                          "Predit : %{customdata[2]:,.0f}<br>"
                          "%{customdata[3]}<extra></extra>"))

    fig.update_layout(
        title="Severite de l'ecart face a l'incertitude du modele"
              "<br><sup>Haut-gauche = le plus preoccupant : gros ecart alors que "
              "le modele etait confiant  |  taille = exposition  |  "
              "cliquez la legende pour isoler</sup>",
        xaxis=dict(title="Largeur relative de l'intervalle  (incertitude CQR)",
                   tickformat=".0%"),
        yaxis=dict(title="Position normalisee  z",
                   range=[max(-6, dd["z"].quantile(0.002)),
                          min(8, dd["z"].quantile(0.998))]),
        template="plotly_white", height=680, hovermode="closest")
    fig.show()


bulles_severite(expl)



Celleue R 
# ================================================================
# FIGURE 5 — SANKEY : d'ou viennent les anomalies ?
#   Flux : Lob -> Risk -> statut de couverture
# ================================================================

def sankey_anomalies(d, niveaux=None, n_par_niveau=6):
    niveaux = niveaux or [c for c in ["Lob", "Risk"] if c in d.columns][:2]
    if len(niveaux) < 2:
        print("Au moins deux colonnes categorielles necessaires.")
        return

    dd = d.copy()
    for c in niveaux:
        top = dd[c].value_counts().head(n_par_niveau).index
        dd[c] = np.where(dd[c].isin(top), dd[c].astype(str), "Autres")

    etapes = niveaux + ["statut"]
    labels, index = [], {}
    for et in etapes:
        for v in dd[et].astype(str).unique():
            cle = f"{et}::{v}"
            if cle not in index:
                index[cle] = len(labels)
                labels.append(str(v))

    src, tgt, val, col = [], [], [], []
    for a, b in zip(etapes[:-1], etapes[1:]):
        g = dd.groupby([a, b], observed=True).size().reset_index(name="n")
        for _, r in g.iterrows():
            src.append(index[f"{a}::{r[a]}"])
            tgt.append(index[f"{b}::{r[b]}"])
            val.append(int(r["n"]))
            col.append("rgba(229,57,53,0.40)" if str(r[b]) == "Hors intervalle"
                       else "rgba(23,105,224,0.22)")

    couleurs_noeuds = ["#E53935" if l == "Hors intervalle"
                       else "#1769E0" if l == "Couverte"
                       else PALETTE[i % len(PALETTE)]
                       for i, l in enumerate(labels)]

    fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=18, thickness=18,
                  color=couleurs_noeuds, line=dict(color="white", width=0.6)),
        link=dict(source=src, target=tgt, value=val, color=col)))
    fig.update_layout(
        title="Cheminement des observations vers le statut de couverture"
              f"<br><sup>{' -> '.join(etapes)}  |  "
              "l'epaisseur du flux est le nombre d'observations</sup>",
        template="plotly_white", height=640, font=dict(size=12))
    fig.show()


sankey_anomalies(expl)


figure_cqr_canonique(results_v2, anomalies_prio)
