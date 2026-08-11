# Redirige tous les fig.show() existants vers l'affichage HTML
import plotly.graph_objects as go
from IPython.display import display, HTML

def _show_patch(self, *args, **kwargs):
    display(HTML(self.to_html(include_plotlyjs=PLOTLYJS, full_html=False)))

go.Figure.show = _show_patch
print("Patch applique : tous les fig.show() s'affichent desormais en HTML.")




# ================================================================
# BLOC 0 — CONFIGURATION DE L'AFFICHAGE  (a executer en premier)
# ================================================================

import plotly.io as pio
from IPython.display import display, HTML
import plotly.graph_objects as go

print("Renderer actuel :", pio.renderers.default)
print("Renderers disponibles :", list(pio.renderers))

for r in ["notebook_connected", "notebook", "iframe_connected", "iframe", "colab"]:
    if r in pio.renderers:
        pio.renderers.default = r
        break
print("Renderer retenu :", pio.renderers.default)

# "cdn"  -> leger, necessite un acces internet
# True   -> embarque plotly.js (~3 Mo par figure), fonctionne hors ligne
PLOTLYJS = "cdn"


def afficher(fig):
    """Affiche une figure Plotly quel que soit l'environnement."""
    display(HTML(fig.to_html(include_plotlyjs=PLOTLYJS, full_html=False)))


# --- TEST : si vous ne voyez pas ce petit graphique, changez PLOTLYJS = True
afficher(go.Figure(go.Scatter(x=[1, 2, 3, 4], y=[2, 1, 3, 2.5],
                              mode="lines+markers", line=dict(color="#4C72B0")))
         .update_layout(title="Test d'affichage — si vous voyez ceci, tout va bien",
                        template="plotly_white", height=280))












# ================================================================
# BLOC 1 — PREPARATION DES DONNEES
# ================================================================

import numpy as np
import pandas as pd
import plotly.express as px

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











# ================================================================
# FIGURE 1 — SUNBURST HIERARCHIQUE (cliquez pour explorer en profondeur)
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
        range_color=[0, max(0.35, dd["est_anomalie"].mean() * 3)])
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
    afficher(fig)


sunburst_anomalies(expl)












# ================================================================
# FIGURE 2 — COORDONNEES PARALLELES (glissez sur un axe pour filtrer)
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
    afficher(fig)


coordonnees_paralleles(anomalies_prio)

















# ================================================================
# FIGURE 3 — VIOLONS + POINTS par categorie
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
    afficher(fig)


violons_par_categorie(expl)














# ================================================================
# FIGURE 4 — BULLES : severite vs incertitude du modele
# ================================================================

def bulles_severite(d, categorie=None, n_cat=8, n_points=1200):
    categorie = categorie or next((c for c in ["Lob", "Risk", "Partner"]
                                   if c in d.columns), ID_COLS[0])
    top = d[categorie].value_counts().head(n_cat).index.tolist()
    dd = d[d[categorie].isin(top)].copy()
    if len(dd) > n_points:
        dd = dd.sample(n_points, random_state=42)
    dd[categorie] = dd[categorie].astype(str)

    taille = (dd[GWP_COL].rank(pct=True) * 26 + 4) if GWP_COL in dd.columns else None

    fig = go.Figure()
    fig.add_hrect(y0=-1, y1=1, fillcolor="rgba(70,110,230,0.10)", line_width=0)
    fig.add_hline(y=1, line=dict(color="#355CDE", width=1.4, dash="dash"))
    fig.add_hline(y=-1, line=dict(color="#355CDE", width=1.4, dash="dash"))

    for i, cat in enumerate(top):
        m = (dd[categorie] == str(cat)).values
        sub = dd[m]
        fig.add_trace(go.Scatter(
            x=sub["largeur_rel"], y=sub["z"], mode="markers",
            name=str(cat),
            marker=dict(size=(taille[m] if taille is not None else 9),
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
    afficher(fig)


bulles_severite(expl)

















# ================================================================
# FIGURE 5 — SANKEY : d'ou viennent les anomalies ?
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
    afficher(fig)


sankey_anomalies(expl)


















# ================================================================
# VISUALISATION DES UNITES STATISTIQUES — version corrigee
#   Panneau haut : montants reels + bande conforme (echelle log)
#   Panneau bas  : severite normalisee z (echelle constante)
#   Bleu = observation dans l'intervalle | Rouge = hors intervalle
#   Zoom synchronise entre les deux panneaux
# ================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualisation_unites_stat(d, n_unites=50, tri="y_obs", log_y=True,
                              mode="echantillon", label_cols=None,
                              random_state=42):
    """
    tri   : "y_obs" (comme le code de reference) ou "y_pred"
    mode  : "echantillon" -> tirage aleatoire, taux d'anomalie reel
            "anomalies"   -> moitie d'anomalies forcees (pedagogique)
    """
    label_cols = label_cols or [c for c in ["Companies", "Lob", "Risk", "Partner"]
                                if c in d.columns][:3]

    dd = d.dropna(subset=["y_obs", "y_pred", "borne_basse", "borne_haute"]).copy()
    couverture_globale = dd["dans_intervalle"].mean()
    n_total = len(dd)

    if mode == "anomalies":
        ano = dd[~dd["dans_intervalle"]]
        norm = dd[dd["dans_intervalle"]]
        k = min(n_unites // 2, len(ano))
        sel = pd.concat([ano.sample(k, random_state=random_state),
                         norm.sample(min(n_unites - k, len(norm)),
                                     random_state=random_state)])
    else:
        sel = dd.sample(min(n_unites, len(dd)), random_state=random_state)

    sel = sel.sort_values(tri).reset_index(drop=True)

    x = np.arange(len(sel))
    labels = sel[label_cols].astype(str).agg("_".join, axis=1)
    obs = sel["y_obs"].values.astype(float)
    pred = sel["y_pred"].values.astype(float)
    lo = sel["borne_basse"].values.astype(float).copy()
    hi = sel["borne_haute"].values.astype(float)
    centre = (lo + hi) / 2
    demi = np.maximum((hi - lo) / 2, 1e-9)
    z = (obs - centre) / demi
    dedans = sel["dans_intervalle"].values.astype(bool)

    log_ok = log_y and (obs > 0).all() and (pred > 0).all()
    if log_ok:
        lo = np.maximum(lo, np.nanmin(obs[obs > 0]) * 1e-2)

    hover = [(f"<b>{l}</b><br>"
              f"Observe    : {o:,.0f}<br>"
              f"Predit     : {p:,.0f}<br>"
              f"Intervalle : [{a:,.0f} ; {b:,.0f}]<br>"
              f"Largeur    : {b-a:,.0f}<br>"
              f"z          : {zz:.2f}")
             for l, o, p, a, b, zz in zip(labels, obs, pred,
                                          sel["borne_basse"], hi, z)]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.62, 0.38],
        subplot_titles=("Montants observes et intervalle conforme",
                        "Severite normalisee — toutes les unites a la meme echelle"))

    # ---------- Panneau 1 : montants reels ----------
    fig.add_trace(go.Scatter(x=x, y=lo, mode="lines",
                             line=dict(color="#4a9d4a", width=1.2, dash="dot"),
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=hi, mode="lines",
                             line=dict(color="#4a9d4a", width=1.2, dash="dot"),
                             fill="tonexty", fillcolor="rgba(120,200,120,0.30)",
                             name=f"Intervalle conforme ({100*(1-ALPHA):.0f} %)",
                             hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=centre, mode="lines",
                             line=dict(color="#2e7d32", width=2),
                             name="Centre conforme", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=pred, mode="lines",
                             line=dict(color="#1565c0", width=1.6, dash="dash"),
                             name="Prediction du modele", hoverinfo="skip"), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x[dedans], y=obs[dedans], mode="markers",
        marker=dict(size=7, color="#1769E0", line=dict(width=0.6, color="white")),
        name=f"Dans l'intervalle ({dedans.sum()})",
        text=[h for h, k in zip(hover, dedans) if k],
        hovertemplate="%{text}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x[~dedans], y=obs[~dedans], mode="markers",
        marker=dict(size=10, color="#E53935", symbol="circle",
                    line=dict(width=0.8, color="#7b0000")),
        name=f"HORS intervalle ({(~dedans).sum()})",
        text=[h for h, k in zip(hover, dedans) if not k],
        hovertemplate="%{text}<extra></extra>"), row=1, col=1)

    # Trait de depassement
    xs_o, ys_o = [], []
    for xi, o, a, b, k in zip(x, obs, sel["borne_basse"].values, hi, dedans):
        if not k:
            xs_o += [xi, xi, None]
            ys_o += [(b if o > b else a), o, None]
    if xs_o:
        fig.add_trace(go.Scatter(x=xs_o, y=ys_o, mode="lines",
                                 line=dict(color="#E53935", width=1.2, dash="dot"),
                                 showlegend=False, hoverinfo="skip"), row=1, col=1)

    # ---------- Panneau 2 : severite normalisee ----------
    fig.add_hrect(y0=-1, y1=1, fillcolor="rgba(120,200,120,0.28)", line_width=0,
                  row=2, col=1)
    fig.add_hline(y=1, line=dict(color="#4a9d4a", width=1.4), row=2, col=1)
    fig.add_hline(y=-1, line=dict(color="#4a9d4a", width=1.4), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="#2e7d32", width=1.2, dash="dash"), row=2, col=1)

    xs_z, ys_z = [], []
    for xi, zz, k in zip(x, z, dedans):
        if not k:
            xs_z += [xi, xi, None]
            ys_z += [np.sign(zz), zz, None]
    if xs_z:
        fig.add_trace(go.Scatter(x=xs_z, y=ys_z, mode="lines",
                                 line=dict(color="#E53935", width=1.2, dash="dot"),
                                 showlegend=False, hoverinfo="skip"), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=x[dedans], y=z[dedans], mode="markers",
        marker=dict(size=7, color="#1769E0", line=dict(width=0.6, color="white")),
        showlegend=False,
        text=[h for h, k in zip(hover, dedans) if k],
        hovertemplate="%{text}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x[~dedans], y=z[~dedans], mode="markers",
        marker=dict(size=10, color="#E53935", line=dict(width=0.8, color="#7b0000")),
        showlegend=False,
        text=[h for h, k in zip(hover, dedans) if not k],
        hovertemplate="%{text}<extra></extra>"), row=2, col=1)

    for xi, zz, k in zip(x, z, dedans):
        if not k:
            fig.add_annotation(x=xi, y=zz, text=f"{abs(zz):.1f}x", showarrow=False,
                               yshift=13 if zz > 0 else -15,
                               font=dict(size=8, color="#7b0000"), row=2, col=1)

    # ---------- Mise en forme ----------
    suffixe = "echantillon aleatoire" if mode == "echantillon" else "anomalies sur-representees"
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=1.13, xanchor="left",
        text=(f"<b>Couverture globale : {100*couverture_globale:.1f} %</b> "
              f"(cible {100*(1-ALPHA):.0f} %) — calculee sur {n_total:,} observations"),
        showarrow=False, font=dict(size=11),
        bgcolor="rgba(255,255,255,0.9)", bordercolor="gray", borderwidth=1, borderpad=5)

    fig.update_yaxes(title_text=TARGET + ("  (log)" if log_ok else ""),
                     type="log" if log_ok else "linear", row=1, col=1)
    fig.update_yaxes(title_text="z  (ecart / demi-largeur)",
                     range=[max(-6, np.nanmin(z) - 0.5), min(9, np.nanmax(z) + 0.7)],
                     row=2, col=1)
    fig.update_xaxes(tickmode="array", tickvals=x, ticktext=labels.tolist(),
                     tickangle=90, tickfont=dict(size=7), row=2, col=1)
    fig.update_layout(
        title=dict(text=f"Intervalles de Conformal Prediction — {len(sel)} unites statistiques"
                        f"<br><sup>Triees par {tri} croissant | {suffixe} | "
                        "zoom synchronise entre les deux panneaux</sup>", x=0.5),
        template="plotly_white", height=900, hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
        margin=dict(t=170, b=160))

    afficher(fig)


# --- Vue 1 : echantillon representatif (taux d'anomalie reel) ---
visualisation_unites_stat(expl, n_unites=50, tri="y_obs", mode="echantillon")











# ================================================================
# RUBAN CONFORME STABLE — la bande ne vacille jamais
#
#   z = (y_obs - centre) / demi-largeur
#      |z| <= 1  -> observation couverte
#      |z| >  1  -> hors intervalle, et |z| EST la severite
#
#   Zones : verte (normale) / orange (moderee) / rouge (severe)
#   Taille des points = exposition financiere
#   Histogramme marginal = distribution complete de z
# ================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output


def ruban_stable(d, tri="z", n_max=None, categorie=None, valeur=None,
                 seuil_modere=2.0, seuil_severe=3.0, out=None):
    """
    tri : "z"      -> points tries par severite : courbe monotone elegante
          "y_pred" -> ordre par taille de portefeuille
          "y_obs"  -> ordre par montant observe
    """
    dd = d.dropna(subset=["y_obs", "y_pred", "borne_basse", "borne_haute"]).copy()
    if categorie and valeur is not None and categorie in dd.columns:
        dd = dd[dd[categorie].astype(str) == str(valeur)]
    if len(dd) < 5:
        print("Trop peu d'observations pour cette selection.")
        return
    if n_max and len(dd) > n_max:
        dd = dd.sample(n_max, random_state=42)

    centre = (dd["borne_haute"] + dd["borne_basse"]) / 2
    demi = np.maximum((dd["borne_haute"] - dd["borne_basse"]) / 2, 1e-9)
    dd["z"] = (dd["y_obs"] - centre) / demi
    dd = dd.sort_values(tri).reset_index(drop=True)

    x = np.arange(len(dd))
    z = dd["z"].values
    dedans = dd["dans_intervalle"].values.astype(bool)
    az = np.abs(z)

    # Statistiques
    cov = dedans.mean()
    n_mod = int(((az > 1) & (az <= seuil_modere)).sum())
    n_sev = int(((az > seuil_modere) & (az <= seuil_severe)).sum())
    n_ext = int((az > seuil_severe).sum())

    # Taille des points selon l'exposition
    if GWP_COL in dd.columns and dd[GWP_COL].notna().any():
        taille = 5 + 16 * dd[GWP_COL].rank(pct=True).fillna(0.5).values
    else:
        taille = np.full(len(dd), 7.0)

    id_cols = [c for c in ID_COLS if c in dd.columns]
    hover = [(f"<b>{' | '.join(str(r[c]) for c in id_cols)}</b><br>"
              f"<b>z = {r['z']:+.2f}</b>  ({'couverte' if r['dans_intervalle'] else 'HORS intervalle'})<br>"
              f"─────────────<br>"
              f"Observe    : {r['y_obs']:,.0f}<br>"
              f"Predit     : {r['y_pred']:,.0f}<br>"
              f"Intervalle : [{r['borne_basse']:,.0f} ; {r['borne_haute']:,.0f}]<br>"
              f"Largeur    : {r['borne_haute']-r['borne_basse']:,.0f}"
              + (f"<br>{GWP_COL} : {r[GWP_COL]:,.0f}" if GWP_COL in dd.columns else "")
              + (f"<br>Rang priorite : #{int(r['rank'])}"
                 if pd.notna(r.get("rank", np.nan)) else ""))
             for _, r in dd.iterrows()]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.84, 0.16],
                        shared_yaxes=True, horizontal_spacing=0.02,
                        subplot_titles=("", "Distribution"))

    # ---------- Zones de gravite : parfaitement horizontales ----------
    zones = [(-1, 1, "rgba(76,175,80,0.22)", None),
             (1, seuil_modere, "rgba(255,193,7,0.20)", None),
             (-seuil_modere, -1, "rgba(255,193,7,0.20)", None),
             (seuil_modere, seuil_severe, "rgba(255,112,67,0.20)", None),
             (-seuil_severe, -seuil_modere, "rgba(255,112,67,0.20)", None)]
    for y0, y1, c, _ in zones:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=c, line_width=0, row=1, col=1)

    for yv, coul, larg in [(1, "#2e7d32", 2.4), (-1, "#2e7d32", 2.4),
                           (seuil_modere, "#f57c00", 1.4), (-seuil_modere, "#f57c00", 1.4),
                           (seuil_severe, "#d84315", 1.4), (-seuil_severe, "#d84315", 1.4)]:
        fig.add_hline(y=yv, line=dict(color=coul, width=larg), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="#1b5e20", width=1.4, dash="dash"), row=1, col=1)

    # ---------- Observations ----------
    fig.add_trace(go.Scatter(
        x=x[dedans], y=z[dedans], mode="markers",
        marker=dict(size=taille[dedans], color="#1769E0", opacity=0.62,
                    line=dict(width=0.4, color="white")),
        name=f"Couverte — {dedans.sum():,}",
        text=[h for h, k in zip(hover, dedans) if k],
        hovertemplate="%{text}<extra></extra>"), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x[~dedans], y=z[~dedans], mode="markers",
        marker=dict(size=taille[~dedans], color=az[~dedans], colorscale="YlOrRd",
                    cmin=1, cmax=max(seuil_severe, np.percentile(az[~dedans], 95)) if (~dedans).any() else 3,
                    showscale=True, opacity=0.9, line=dict(width=0.6, color="#7b0000"),
                    colorbar=dict(title="|z|", x=0.86, len=0.75, thickness=12)),
        name=f"Hors intervalle — {(~dedans).sum():,}",
        text=[h for h, k in zip(hover, dedans) if not k],
        hovertemplate="%{text}<extra></extra>"), row=1, col=1)

    # ---------- Histogramme marginal ----------
    fig.add_trace(go.Histogram(
        y=z, nbinsy=90, orientation="h",
        marker=dict(color="#90a4ae", line=dict(width=0)),
        showlegend=False, hovertemplate="z ≈ %{y:.1f}<br>n = %{x}<extra></extra>"),
        row=1, col=2)
    for yv, coul in [(1, "#2e7d32"), (-1, "#2e7d32")]:
        fig.add_hline(y=yv, line=dict(color=coul, width=1.8), row=1, col=2)

    # ---------- Encadre de synthese ----------
    fig.add_annotation(
        xref="paper", yref="paper", x=0.005, y=0.99, xanchor="left", yanchor="top",
        text=(f"<b>Couverture : {100*cov:.1f} %</b>  (cible {100*(1-ALPHA):.0f} %)<br>"
              f"<span style='color:#f57c00'>■</span> Moderee  |z| ≤ {seuil_modere:g} : {n_mod:,}<br>"
              f"<span style='color:#d84315'>■</span> Severe   |z| ≤ {seuil_severe:g} : {n_sev:,}<br>"
              f"<span style='color:#b71c1c'>■</span> Extreme  |z| > {seuil_severe:g} : {n_ext:,}<br>"
              f"<sub>{len(dd):,} observations affichees</sub>"),
        showarrow=False, align="left", font=dict(size=11),
        bgcolor="rgba(255,255,255,0.93)", bordercolor="#455a64",
        borderwidth=1, borderpad=7)

    titre = "Ruban conforme stable — la bande ne vacille jamais"
    if categorie and valeur is not None:
        titre += f"  |  {categorie} = {valeur}"

    plage = [max(-8, np.nanmin(z) - 0.6), min(12, np.nanmax(z) + 0.8)]
    fig.update_yaxes(title_text="z  =  (observe − centre) / demi-largeur",
                     range=plage, row=1, col=1)
    fig.update_yaxes(range=plage, row=1, col=2)
    fig.update_xaxes(title_text=f"Observations triees par {tri} croissant",
                     showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text="n", row=1, col=2)
    fig.update_layout(
        title=dict(text=titre + "<br><sup>Chaque intervalle CQR est ramene a [−1 ; +1]. "
                               "|z| se lit directement comme la gravite. "
                               "Taille des points = exposition financiere</sup>", x=0.5),
        template="plotly_white", height=760, hovermode="closest", bargap=0.02,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    if out is not None:
        with out:
            clear_output(wait=True)
            afficher(fig)
    else:
        afficher(fig)


# --- Vue globale : toutes les observations ---
ruban_stable(expl, tri="z")
