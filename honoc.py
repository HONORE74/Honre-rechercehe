


Le principe en une phrase
Un funnel plot compare une mesure (ta couverture) à la précision de cette mesure (l'effectif du groupe), pour distinguer une vraie anomalie d'un simple effet de petit échantillon.

Pourquoi c'est nécessaire
Avec 10 observations, obtenir 70 % ou 100 % de couverture n'a rien d'étonnant — le hasard suffit à l'expliquer. Avec 5 000 observations, un écart de seulement 3 points est déjà suspect. La même valeur observée n'a pas le même sens selon l'effectif derrière elle. Une ligne horizontale unique à 90 % (comme dans le dot plot de ton tuteur) ne fait pas cette distinction.

import numpy as np, pandas as pd
import plotly.graph_objects as go

# Palette : vert = comportement normal, rouge = decrochage reel (hors entonnoir)
COULEUR_OK       = "#0ca30c"   # status "good"
COULEUR_ALERTE   = "#d03b3b"   # status "critical"
COULEUR_BANDE_95 = "#9ec5f4"   # sequentiel bleu, step clair
COULEUR_BANDE_99 = "#cde2fb"   # sequentiel bleu, step tres clair
COULEUR_CIBLE    = "#0b0b0b"   # encre primaire


def funnel_interactif(results, segment_col="Lob", alpha=0.10, n_min=8,
                      fichier=None):
    """Funnel plot INTERACTIF sur UNE SEULE variable.
    x = effectif du segment | y = couverture observee.
    L'entonnoir (bleu) borne la variation attendue par hasard : un point
    HORS entonnoir est un decrochage reel, pas du bruit d'echantillonnage."""
    cible = 1 - alpha

    g = (results.groupby(segment_col, observed=True)["dans_intervalle"]
         .agg(k="sum", n="size").reset_index())
    g = g[g["n"] >= n_min].copy()
    g["cov"] = g["k"] / g["n"]

    # --- Bornes de l'entonnoir (approximation normale de la proportion) ---
    n_grille = np.linspace(g["n"].min(), g["n"].max(), 300)
    b95 = 1.96 * np.sqrt(cible * (1 - cible) / n_grille)
    b99 = 2.58 * np.sqrt(cible * (1 - cible) / n_grille)

    seuil95 = 1.96 * np.sqrt(cible * (1 - cible) / g["n"])
    g["hors_entonnoir"] = (g["cov"] < cible - seuil95) | (g["cov"] > cible + seuil95)
    g["ecart"] = g["cov"] - cible

    fig = go.Figure()

    # Bande 99 % (la plus large, dessinee en premier)
    fig.add_trace(go.Scatter(
        x=np.concatenate([n_grille, n_grille[::-1]]),
        y=np.concatenate([cible + b99, (cible - b99)[::-1]]),
        fill="toself", fillcolor=COULEUR_BANDE_99, line=dict(width=0),
        name="Entonnoir 99 %", hoverinfo="skip"))

    # Bande 95 %
    fig.add_trace(go.Scatter(
        x=np.concatenate([n_grille, n_grille[::-1]]),
        y=np.concatenate([cible + b95, (cible - b95)[::-1]]),
        fill="toself", fillcolor=COULEUR_BANDE_95, line=dict(width=0),
        name="Entonnoir 95 %", hoverinfo="skip"))

    # Ligne cible
    fig.add_trace(go.Scatter(
        x=[g["n"].min(), g["n"].max()], y=[cible, cible],
        mode="lines", line=dict(color=COULEUR_CIBLE, width=2, dash="dash"),
        name=f"Cible {cible:.0%}", hoverinfo="skip"))

    # Points -- verts (normal) et rouges (decrochage reel)
    for masque, coul, label in [(~g["hors_entonnoir"], COULEUR_OK, "Dans l'entonnoir"),
                                (g["hors_entonnoir"],  COULEUR_ALERTE, "Hors entonnoir (alerte)")]:
        sub = g[masque]
        if not len(sub): continue
        fig.add_trace(go.Scatter(
            x=sub["n"], y=sub["cov"], mode="markers",
            marker=dict(size=13, color=coul, line=dict(color="white", width=1.4)),
            name=label,
            customdata=np.column_stack([sub[segment_col], sub["ecart"]]),
            hovertemplate=(f"<b>{segment_col} : %{{customdata[0]}}</b><br>"
                          "Effectif : %{x:,.0f}<br>"
                          "Couverture : %{y:.2%}<br>"
                          "Ecart a la cible : %{customdata[1]:+.2%}<extra></extra>")))

    fig.update_layout(
        title=dict(text=f"Funnel plot -- couverture par {segment_col}<br>"
                        f"<sub>{g['hors_entonnoir'].sum()} segment(s) hors entonnoir "
                        f"sur {len(g)}</sub>", font=dict(size=16)),
        xaxis=dict(title=f"Effectif du segment (n)", gridcolor="#e1e0d9"),
        yaxis=dict(title="Couverture observee", tickformat=".0%",
                  range=[0, 1.05], gridcolor="#e1e0d9"),
        template="plotly_white", height=620, hovermode="closest",
        legend=dict(orientation="h", y=1.10, x=0.5, xanchor="center"),
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb")

    fig.write_html(fichier or f"funnel_{segment_col}.html")
    fig.show()
    print(f"Sauvegarde : {fichier or f'funnel_{segment_col}.html'}")
    return g


# Un seul appel par variable -- aussi simple que ca
funnel_lob  = funnel_interactif(results_test, segment_col="Lob",  alpha=ALPHA)











import numpy as np, pandas as pd
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier

COULEUR_OK, COULEUR_ATTENTION, COULEUR_ALERTE = "#0ca30c", "#fab219", "#d03b3b"


def preparer_arbre(results, X_test, alpha=0.10, max_depth=3, min_leaf=40):
    """Entraine l'arbre de non-couverture (identique a avant), prepare
    les structures pour les deux graphiques interactifs."""
    y_miss = (~results["dans_intervalle"]).astype(int).values
    X = X_test.copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "category":
            X[c] = X[c].astype("category").cat.codes
    X = X.fillna(-999)

    arbre = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf,
                                   random_state=42)
    arbre.fit(X, y_miss)
    return arbre, X, y_miss


def arbre_interactif(arbre, X, feature_names, alpha=0.10, fichier="arbre_interactif.html"):
    """Arbre de non-couverture en ICICLE Plotly : cliquer sur une branche pour
    zoomer, survoler une case pour voir la regle complete, n et couverture."""
    t = arbre.tree_
    ids, labels, parents, values, taux_list, regles = [], [], [], [], [], {}

    def parcours(noeud, parent_id, regle_txt):
        nid = f"n{noeud}"
        n = int(t.n_node_samples[noeud])
        val = t.value[noeud][0]
        taux = val[1] / val.sum() if val.sum() else 0.0
        regles[nid] = regle_txt
        ids.append(nid); parents.append(parent_id); values.append(n); taux_list.append(taux)

        if t.children_left[noeud] == -1:
            labels.append(f"Feuille (n={n})")
        else:
            f = feature_names[t.feature[noeud]]; s = t.threshold[noeud]
            labels.append("Racine" if parent_id == "" else regle_txt.split(" ET ")[-1])
            gche = f"{f} <= {s:,.0f}"; dte = f"{f} > {s:,.0f}"
            suite = "" if regle_txt == "" else regle_txt + " ET "
            parcours(t.children_left[noeud],  nid, suite + gche)
            parcours(t.children_right[noeud], nid, suite + dte)

    parcours(0, "", "")
    couverture = [1 - v for v in taux_list]

    fig = go.Figure(go.Icicle(
        ids=ids, labels=labels, parents=parents, values=values,
        branchvalues="total",
        marker=dict(colors=taux_list, colorscale=[[0, COULEUR_OK], [0.5, COULEUR_ATTENTION],
                                                   [1, COULEUR_ALERTE]],
                   cmin=0, cmax=max(alpha*2.5, max(taux_list)),
                   colorbar=dict(title="Taux de<br>non-couverture", tickformat=".0%")),
        customdata=np.column_stack([[regles[i] if regles[i] else "Racine (aucune condition)"
                                     for i in ids], couverture]),
        hovertemplate=("<b>Regle :</b> %{customdata[0]}<br>"
                      "Effectif : %{value:,.0f}<br>"
                      "Couverture : %{customdata[1]:.1%}<br>"
                      "Non-couverture : %{color:.1%}<extra></extra>"),
        root_color="#e1e0d9"))

    fig.update_layout(
        title=dict(text="Arbre de non-couverture (icicle interactif)<br>"
                        "<sub>Clic pour zoomer sur une branche -- survol pour la regle</sub>",
                   font=dict(size=16)),
        template="plotly_white", height=560, margin=dict(t=90, l=10, r=10, b=10))
    fig.write_html(fichier); fig.show()
    print(f"Sauvegarde : {fichier}")
    return fig


def feuilles_interactif(arbre, X, y_miss, feature_names, alpha=0.10,
                        fichier="feuilles_interactif.html"):
    """Taux de non-couverture par feuille, en barres interactives.
    Le survol affiche la regle complete qui mene a chaque feuille."""
    t = arbre.tree_
    chemins = {}

    def parcours(noeud, conditions):
        if t.children_left[noeud] == -1:
            chemins[noeud] = list(conditions); return
        f = feature_names[t.feature[noeud]]; s = t.threshold[noeud]
        parcours(t.children_left[noeud],  conditions + [f"{f} <= {s:,.0f}"])
        parcours(t.children_right[noeud], conditions + [f"{f} >  {s:,.0f}"])
    parcours(0, [])

    feuille_de = arbre.apply(X)
    stats = (pd.DataFrame({"feuille": feuille_de, "manque": y_miss})
             .groupby("feuille")["manque"].agg(taux="mean", n="size")
             .sort_values("taux"))
    stats["couverture"] = 1 - stats["taux"]
    stats["regle"] = [ "<br>".join(chemins[f]) for f in stats.index ]

    coul = [COULEUR_ALERTE if v > alpha*2 else COULEUR_ATTENTION if v > alpha*1.3
            else COULEUR_OK for v in stats["taux"]]

    fig = go.Figure(go.Bar(
        y=[f"Feuille {f} (n={n})" for f, n in zip(stats.index, stats["n"])],
        x=stats["taux"], orientation="h", marker=dict(color=coul),
        customdata=np.column_stack([stats["couverture"], stats["regle"]]),
        hovertemplate=("Non-couverture : %{x:.1%}<br>"
                      "Couverture : %{customdata[0]:.1%}<br>"
                      "<b>Regle :</b><br>%{customdata[1]}<extra></extra>")))

    fig.add_vline(x=alpha, line_dash="dash", line_color="#0b0b0b",
                 annotation_text=f"Taux attendu {alpha:.0%}")
    fig.update_layout(
        title=dict(text="Taux de non-couverture par feuille (interactif)<br>"
                        f"<sub>Amplitude : {stats['taux'].max()-stats['taux'].min():.1%} -- "
                        "survol pour la regle complete</sub>", font=dict(size=16)),
        xaxis=dict(title="Taux de non-couverture", tickformat=".0%", gridcolor="#e1e0d9"),
        yaxis=dict(title=""), template="plotly_white", height=520,
        margin=dict(l=10, r=10, t=90, b=10))
    fig.write_html(fichier); fig.show()
    print(f"Sauvegarde : {fichier}")
    return stats


# ---------- Execution ----------
arbre, X_encode, y_miss = preparer_arbre(results_test, X_test, alpha=ALPHA)
fig_arbre    = arbre_interactif(arbre, X_encode, list(X_test.columns), alpha=ALPHA)
stats_feuil  = feuilles_interactif(arbre, X_encode, y_miss, list(X_test.columns), alpha=ALPHA)



Le principe en une phrase
L'arbre pose une suite de questions (« est-ce que cette variable dépasse ce seuil ? ») pour séparer les observations en groupes de plus en plus homogènes vis-à-vis de la non-couverture.

La structure, nœud par nœud
Élément	Signification
Racine (le nœud du haut)	Toutes les observations, avant toute question
Nœud interne	Une question : Variable ≤ seuil ?
Branche gauche	Les observations qui répondent OUI
Branche droite	Les observations qui répondent NON
Feuille (bout de branche)	Un groupe final, après avoir répondu à toutes les questions du chemin
Comment lire une feuille précise
Sur ton arbre, pour arriver à une feuille donnée, tu suis le chemin depuis la racine en accumulant les conditions :

ATR_Split_MG_FI > 58 235   ET
Technical_losses > 233 962   ET
IBNR_best_estimate_eop > 75 229
    -> FEUILLE : n = 1 203, couverture = 78,1 %
Cette feuille décrit donc : « les contrats de grande magnitude, avec de fortes pertes techniques ET une forte provision IBNR ». C'est cette phrase (pas le numéro de la feuille) qu'on met dans le mémoire.

Ce que la couleur encode dans ton icicle interactif
Vert : la feuille a un taux de non-couverture proche de l'attendu (10 %) — comportement normal
Rouge : la feuille manque beaucoup plus que prévu — c'est une région où le modèle échoue systématiquement
Le raisonnement qui rend l'outil utile
Si la couverture était parfaitement conditionnelle, l'arbre ne trouverait aucune feuille différente des autres — toutes afficheraient ~10 %. Le fait qu'il ait trouvé une feuille à 0 % et une autre à 22 % prouve que la non-couverture est prévisible à partir des variables, donc que la couverture conditionnelle est violée.














import numpy as np, pandas as pd

def calibration_conforme_par_largeur(df_calib, y_calib, y_lo_calib, y_hi_calib,
                                     df_test, y_lo_test, y_hi_test,
                                     n_bins=10, alpha=0.10, n_min_bin=30):
    """Mondrian conditionne sur la largeur BRUTE de l'intervalle (q95-q05),
    suivant exactement les 5 etapes du tuteur."""

    # ---------- ETAPE 1 : largeur individuelle, SANS conformalisation ----------
    largeur_calib = y_hi_calib - y_lo_calib
    largeur_test  = y_hi_test  - y_lo_test

    # ---------- ETAPE 2 : bins definis sur la CALIBRATION uniquement ----------
    bornes = np.unique(np.quantile(largeur_calib, np.linspace(0, 1, n_bins + 1)))
    bornes[0], bornes[-1] = -np.inf, np.inf     # capture toute valeur extreme du test
    n_bins_reel = len(bornes) - 1

    groupe_calib = pd.cut(largeur_calib, bins=bornes, labels=False, include_lowest=True)
    groupe_test  = pd.cut(largeur_test,  bins=bornes, labels=False, include_lowest=True)

    # ---------- ETAPE 3 : Qhat par groupe ----------
    scores_calib = np.maximum(y_lo_calib - y_calib, y_calib - y_hi_calib)
    q_global = float(np.quantile(scores_calib,
                     min(np.ceil((len(scores_calib)+1)*(1-alpha))/len(scores_calib), 1.0),
                     method="higher"))

    lignes = []
    for g in range(n_bins_reel):
        m = groupe_calib == g
        n_g = int(m.sum())
        if n_g < n_min_bin:
            q_g, statut = q_global, "repli global (n trop faible)"
        else:
            niveau = min(np.ceil((n_g + 1) * (1 - alpha)) / n_g, 1.0)
            q_g = float(np.quantile(scores_calib[m], niveau, method="higher"))
            statut = "propre"
        lignes.append({"groupe": g, "largeur_min": bornes[g], "largeur_max": bornes[g+1],
                       "n_calib": n_g, "Qhat": q_g, "statut": statut})
    table_qhat = pd.DataFrame(lignes)

    print("=" * 78); print("  TABLE DES Qhat PAR GROUPE DE LARGEUR (etape 3)"); print("=" * 78)
    print(table_qhat.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    monotone = table_qhat["Qhat"].is_monotonic_increasing
    print(f"\n  Qhat croissant avec la largeur : {'OUI (coherent)' if monotone else 'NON -> a verifier'}")

    # ---------- ETAPE 4 : jointure sur le tableau individuel ----------
    map_qhat = table_qhat.set_index("groupe")["Qhat"]
    qhat_calib_ind = map_qhat.reindex(groupe_calib).values
    qhat_test_ind  = map_qhat.reindex(groupe_test).values

    # ---------- ETAPE 5 : bornes finales ----------
    lower_calib = np.clip(y_lo_calib - qhat_calib_ind, 0, None)
    upper_calib = y_hi_calib + qhat_calib_ind
    lower_test  = np.clip(y_lo_test  - qhat_test_ind,  0, None)
    upper_test  = y_hi_test  + qhat_test_ind

    return {"table_qhat": table_qhat, "bornes": bornes,
            "test": {"lower": lower_test, "upper": upper_test, "groupe": groupe_test},
            "calib": {"lower": lower_calib, "upper": upper_calib, "groupe": groupe_calib}}


res = calibration_conforme_par_largeur(
    df_calib, y_calib.values, y_lo_calib, y_hi_calib,
    df_test, y_lo_test, y_hi_test, n_bins=10, alpha=ALPHA)










    results_v2 = results_test.copy()
results_v2["borne_basse"]  = res["test"]["lower"]
results_v2["borne_haute"]  = res["test"]["upper"]
results_v2["largeur_intervalle"] = results_v2["borne_haute"] - results_v2["borne_basse"]
results_v2["groupe_largeur"] = res["test"]["groupe"]
results_v2["dans_intervalle"] = ((results_v2["y_obs"] >= results_v2["borne_basse"]) &
                                 (results_v2["y_obs"] <= results_v2["borne_haute"]))

print(f"Couverture globale : {results_v2['dans_intervalle'].mean():.2%}  (cible {1-ALPHA:.0%})")










verif = (results_v2.groupby("groupe_largeur")["dans_intervalle"]
         .agg(n="size", couverture="mean").reset_index())
verif["ecart_cible"] = verif["couverture"] - (1 - ALPHA)
print(verif.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
print(f"\nAmplitude de couverture entre groupes : "
      f"{verif['couverture'].max() - verif['couverture'].min():.2%}")
print("-> Compare a l'amplitude de 22 points trouvee par l'arbre : si elle a")
print("   nettement diminue, l'approche du tuteur corrige bien le probleme.")

















import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 6))
x = verif["groupe_largeur"]
coul = ["#d03b3b" if abs(e) > 0.05 else "#0ca30c" for e in verif["ecart_cible"]]
ax.bar(x, verif["couverture"], color=coul)
ax.axhline(1 - ALPHA, color="black", ls="--", lw=2, label=f"Cible {1-ALPHA:.0%}")
ax.set_xlabel("Groupe de largeur (0 = intervalles etroits, 9 = intervalles larges)")
ax.set_ylabel("Couverture observee")
ax.set_title("Couverture par groupe apres Mondrian sur la largeur brute")
ax.legend()
for i, (c, n) in enumerate(zip(verif["couverture"], verif["n"])):
    ax.text(i, c, f"{c:.0%}\nn={n}", ha="center", va="bottom", fontsize=8)
plt.tight_layout(); plt.savefig("mondrian_largeur.png", dpi=200); plt.show()
