# =========================================================================
# CELLULE 9 : SCORE DE PRIORISATION
# =========================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GWP_COL = "GWP"


def calculer_score_priorisation(anomalies_df, gwp_col=GWP_COL):
    """
    Calcule le score composite de priorisation sur le DataFrame `anomalies`
    deja produit par ta cellule d'extraction.

    Prerequis : `anomalies_df` doit contenir gwp_col. Si ce n'est pas le cas,
    ajoute gwp_col a la liste `anomaly_cols` dans ta cellule d'extraction :
        anomaly_cols = existing_id_cols + [..., "dans_intervalle", GWP_COL]
    """
    df = anomalies_df.copy()

    if gwp_col not in df.columns:
        raise KeyError(
            f"'{gwp_col}' absent de anomalies_df. "
            f"Ajoute-le dans anomaly_cols avant l'extraction. Colonnes dispo : {list(df.columns)}"
        )

    # --- A_i : ecart relatif a la borne EFFECTIVEMENT franchie ---
    borne_franchie = np.where(df["y_obs"] > df["borne_haute"], df["borne_haute"], df["borne_basse"])
    df["A_ecart_borne"] = np.abs(df["ecart_intervalle"]) / np.maximum(np.abs(borne_franchie), 1e-6)

    # --- B_i : erreur relative du modele ---
    df["B_erreur_modele"] = np.abs(df["y_obs"] - df["y_pred"]) / np.maximum(np.abs(df["y_pred"]), 1e-6)

    # --- Score composite : GWP brute, telle quelle ---
    df["score_composite"] = df["A_ecart_borne"] * df["B_erreur_modele"] * df[gwp_col]

    df = df.sort_values("score_composite", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    return df


anomalies_prio = calculer_score_priorisation(anomalies, GWP_COL)

print(f"\nAnomalies priorisees : {len(anomalies_prio)}")
print(f"\nTop 15 par score de priorisation :")
cols_affichage = [c for c in ID_COLS if c in anomalies_prio.columns] + [
    "year", "quarter", "y_obs", "y_pred", "A_ecart_borne", "B_erreur_modele",
    GWP_COL, "score_composite", "sens",
]
print(anomalies_prio[cols_affichage].head(15).to_string(index=False))


# =========================================================================
# CELLULE 10 : RAPPORT POUR LE TUTEUR
# =========================================================================

def creer_rapport_priorisation(df_prio, n_top=15, output_file=None):
    """Genere un rapport texte lisible pour presentation au tuteur."""
    r = []
    r.append("=" * 80)
    r.append("RAPPORT DE PRIORISATION DES ANOMALIES - RBNS_eop (IFRS 17)")
    r.append(f"Methode : Conformalized Quantile Regression group-conditionnelle (alpha={ALPHA})")
    r.append("Score_i = A_i (ecart borne) x B_i (erreur modele) x GWP_i (brute)")
    r.append("=" * 80)
    r.append("")
    r.append(f"Total anomalies : {len(df_prio)}")
    r.append("")
    r.append(f"TOP {n_top} ANOMALIES PAR SCORE DECROISSANT")
    r.append("-" * 80)
    for _, row in df_prio.head(n_top).iterrows():
        ident = ", ".join(f"{c}={row[c]}" for c in ID_COLS if c in row.index)
        r.append(f"\nRang #{row['rank']}  score={row['score_composite']:.4f}")
        r.append(f"  {ident}")
        r.append(f"  Periode       : {row['year']}-T{row['quarter']}  ({row['sens']})")
        r.append(f"  y_obs         : {row['y_obs']:>14,.0f}")
        r.append(f"  y_pred        : {row['y_pred']:>14,.0f}")
        r.append(f"  Intervalle    : [{row['borne_basse']:>12,.0f} ; {row['borne_haute']:>12,.0f}]")
        r.append(f"  Ecart borne (A): {row['A_ecart_borne']:.3f}   Erreur modele (B): {row['B_erreur_modele']:.3f}")
        r.append(f"  {GWP_COL:<8}     : {row[GWP_COL]:>14,.0f}")
    r.append("\n" + "=" * 80)
    texte = "\n".join(r)
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(texte)
        print(f"Rapport sauvegarde : {output_file}")
    return texte


rapport = creer_rapport_priorisation(anomalies_prio, n_top=15, output_file="rapport_priorisation.txt")
print(rapport)
anomalies_prio.to_csv("anomalies_priorisees.csv", index=False)


# =========================================================================
# CELLULE 11 : VISUALISATIONS
# =========================================================================

def graphiques_priorisation(results_v2, anomalies_prio, id_cols=ID_COLS, gwp_col=GWP_COL, top_n=15):
    fig, ax = plt.subplots(2, 3, figsize=(20, 12))

    # 1. y_pred vs y_obs : contexte complet (normal en gris) + anomalies colorees par score_composite
    normal = results_v2[results_v2["dans_intervalle"]]
    ax[0, 0].scatter(normal["y_pred"], normal["y_obs"], s=6, alpha=0.15, color="gray", label="Normal")
    sc = ax[0, 0].scatter(anomalies_prio["y_pred"], anomalies_prio["y_obs"], s=25,
                           c=anomalies_prio["score_composite"], cmap="Reds", alpha=0.85)
    plt.colorbar(sc, ax=ax[0, 0], label="score_composite")
    lims = [results_v2["y_obs"].min(), results_v2["y_obs"].max()]
    ax[0, 0].plot(lims, lims, "k--", lw=1)
    ax[0, 0].set_xlabel("y_pred"); ax[0, 0].set_ylabel("y_obs")
    ax[0, 0].set_title("1. Predit vs Reel - anomalies colorees par score")

    # 2. Top N anomalies par score (barh)
    top = anomalies_prio.head(top_n).iloc[::-1]
    labels = top.apply(lambda r: " | ".join(str(r[c]) for c in id_cols if c in r.index)[:35], axis=1)
    ax[0, 1].barh(labels, top["score_composite"], color="firebrick")
    ax[0, 1].set_title(f"2. Top {top_n} anomalies (score composite)")
    ax[0, 1].tick_params(axis="y", labelsize=7)

    # 3. Distribution des scores
    ax[0, 2].hist(anomalies_prio["score_composite"], bins=40, color="steelblue", alpha=0.7)
    ax[0, 2].set_title("3. Distribution des scores")
    ax[0, 2].set_xlabel("score_composite")

    # 4. Score composite vs GWP (couleur = A_i, taille = B_i)
    sc2 = ax[1, 0].scatter(anomalies_prio[gwp_col], anomalies_prio["score_composite"],
                            c=anomalies_prio["A_ecart_borne"], s=anomalies_prio["B_erreur_modele"]*200,
                            cmap="Oranges", alpha=0.7)
    plt.colorbar(sc2, ax=ax[1, 0], label="A_ecart_borne")
    ax[1, 0].set_xlabel(gwp_col); ax[1, 0].set_ylabel("Score composite")
    ax[1, 0].set_title("4. Score vs GWP (taille = B_i, couleur = A_i)")

    # 5. Anomalies par groupe (par sens de depassement)
    if "groupe_largeur" in anomalies_prio.columns:
        pivot = anomalies_prio.groupby(["groupe_largeur", "sens"]).size().unstack(fill_value=0)
        pivot.plot(kind="bar", stacked=True, ax=ax[1, 1])
        ax[1, 1].set_title("5. Anomalies par groupe (empile par sens)")
        ax[1, 1].tick_params(axis="x", rotation=45, labelsize=7)

    # 6. Pareto : concentration du score
    scores_sorted = anomalies_prio["score_composite"].sort_values(ascending=False).values
    cum_pct = np.cumsum(scores_sorted) / scores_sorted.sum() * 100
    n_pct = np.arange(1, len(scores_sorted) + 1) / len(scores_sorted) * 100
    ax[1, 2].plot(n_pct, cum_pct, color="darkred")
    ax[1, 2].axhline(80, color="gray", ls="--", lw=1)
    idx_80 = np.searchsorted(cum_pct, 80)
    if idx_80 < len(n_pct):
        ax[1, 2].axvline(n_pct[idx_80], color="gray", ls="--", lw=1)
        ax[1, 2].text(n_pct[idx_80] + 1, 40, f"{n_pct[idx_80]:.0f}% des anomalies\n= 80% du score total", fontsize=8)
    ax[1, 2].set_xlabel("% des anomalies (triees par score)")
    ax[1, 2].set_ylabel("% cumule du score total")
    ax[1, 2].set_title("6. Courbe de Pareto")

    plt.tight_layout()
    plt.savefig("visualisations_priorisation.png", dpi=150, bbox_inches="tight")
    plt.show()

















pip install plotly ipywidgets


# =========================================================================
# CELLULE 12 : VISUALISATIONS INTERACTIVES (plotly + ipywidgets)
# pip install plotly ipywidgets   (si pas deja installes)
# =========================================================================

import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output


# --- 12a. Scatter interactif Predit vs Reel (survol = detail complet) ---

def _hover_text(row, id_cols):
    ident = " | ".join(f"{c}={row[c]}" for c in id_cols if c in row.index)
    return (f"{ident}<br>"
            f"Periode: {row['year']}-T{row['quarter']}<br>"
            f"y_obs: {row['y_obs']:,.0f}<br>"
            f"y_pred: {row['y_pred']:,.0f}<br>"
            f"Score: {row['score_composite']:.4f}<br>"
            f"A_i: {row['A_ecart_borne']:.3f}  B_i: {row['B_erreur_modele']:.3f}<br>"
            f"{GWP_COL}: {row[GWP_COL]:,.0f}")


normal = results_v2[results_v2["dans_intervalle"]]

fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(
    x=normal["y_pred"], y=normal["y_obs"], mode="markers",
    marker=dict(size=4, color="lightgray", opacity=0.3),
    name="Normal", hoverinfo="skip"))
fig_scatter.add_trace(go.Scatter(
    x=anomalies_prio["y_pred"], y=anomalies_prio["y_obs"], mode="markers",
    marker=dict(size=9, color=anomalies_prio["score_composite"], colorscale="Reds",
                showscale=True, colorbar=dict(title="Score")),
    name="Anomalie",
    text=[_hover_text(row, ID_COLS) for _, row in anomalies_prio.iterrows()],
    hovertemplate="%{text}<extra></extra>"))
lims = [results_v2["y_obs"].min(), results_v2["y_obs"].max()]
fig_scatter.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                                  line=dict(dash="dash", color="black"),
                                  name="y=x", hoverinfo="skip"))
fig_scatter.update_layout(title="Predit vs Reel - anomalies interactives",
                           xaxis_title="y_pred", yaxis_title="y_obs",
                           template="plotly_white", height=600)
fig_scatter.show()


# --- 12b. Trajectoire temporelle avec bande d'intervalle conforme ---

results_v2["year_quarter"] = results_v2["year"].astype(str) + "-T" + results_v2["quarter"].astype(str)
results_v2["entity_key"] = results_v2[ID_COLS].astype(str).agg(" | ".join, axis=1)
anomalies_prio["entity_key"] = anomalies_prio[ID_COLS].astype(str).agg(" | ".join, axis=1)

# Priorite aux entites qui ont au moins une anomalie, triees par pire score
entites_avec_anomalie = (anomalies_prio.groupby("entity_key")["score_composite"]
                          .max().sort_values(ascending=False))
options_dropdown = list(entites_avec_anomalie.index)

if not options_dropdown:
    print("Aucune entite avec anomalie a afficher.")
else:
    out = widgets.Output()
    dropdown = widgets.Dropdown(
        options=options_dropdown,
        description="Entite :",
        layout=widgets.Layout(width="700px"))

    def tracer_entite(entity_key):
        sous = results_v2[results_v2["entity_key"] == entity_key].sort_values("time_idx")
        with out:
            clear_output(wait=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sous["year_quarter"], y=sous["borne_haute"], mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=sous["year_quarter"], y=sous["borne_basse"], mode="lines",
                fill="tonexty", fillcolor="rgba(52,152,219,0.2)",
                line=dict(width=0), name="Intervalle conforme", hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=sous["year_quarter"], y=sous["y_obs"], mode="lines+markers",
                line=dict(color="black"), name="y_obs",
                marker=dict(
                    size=9,
                    color=np.where(sous["dans_intervalle"], "black", "red"),
                    symbol=np.where(sous["dans_intervalle"], "circle", "x"))))
            fig.update_layout(
                title=f"Trajectoire : {entity_key}",
                xaxis_title="Periode", yaxis_title=TARGET,
                template="plotly_white", height=500)
            fig.show()

    dropdown.observe(lambda change: tracer_entite(change["new"]) if change["name"] == "value" else None,
                      names="value")
    display(dropdown, out)
    tracer_entite(options_dropdown[0])



graphiques_priorisation(results_v2, anomalies_prio)





























# =========================================================================
# CELLULE 13 : BANDE CONFORME (style publication CP)
#   vert = observation dans l'intervalle / rouge = hors intervalle
#   tri par prediction croissante -> la bande monte regulierement
# =========================================================================

DF_VIZ = results_v2   # remplacer par results_test si tu pars de la version non group-conditionnelle

# Cle lisible en abscisse (concatenation vectorisee, bien plus rapide que .apply)
DF_VIZ["Unit_Stat_Key"] = (DF_VIZ["Companies"].astype(str) + "_"
                           + DF_VIZ["Lob"].astype(str) + "_"
                           + DF_VIZ["Risk"].astype(str))


def plot_bande_conforme(df, n=40, mode="echantillon", log_y=True,
                        random_state=42, output="bande_conforme.png"):
    """
    Reproduit le style des figures de reference en Conformal Prediction.

    mode :
      "echantillon" -> tirage aleatoire : montre le taux d'anomalie REEL (honnete)
      "anomalies"   -> 50% de pires anomalies + 50% de normales : pedagogique,
                       mais ne reflete pas la proportion reelle -> a preciser en legende
    """
    d = df.copy()

    if mode == "anomalies":
        ano = d[~d["dans_intervalle"]].copy()
        ecart = np.where(ano["y_obs"] > ano["borne_haute"],
                         ano["y_obs"] - ano["borne_haute"],
                         ano["borne_basse"] - ano["y_obs"])
        ano = ano.assign(_e=ecart).nlargest(min(n // 2, len(ano)), "_e").drop(columns="_e")
        dispo_norm = d[d["dans_intervalle"]]
        norm = dispo_norm.sample(min(n - len(ano), len(dispo_norm)), random_state=random_state)
        sel = pd.concat([ano, norm])
    else:
        sel = d.sample(min(n, len(d)), random_state=random_state)

    # Tri par prediction croissante : c'est ce qui fait monter la bande
    sel = sel.sort_values("y_pred").reset_index(drop=True)
    x = np.arange(len(sel))
    obs = sel["y_obs"].values
    dedans = sel["dans_intervalle"].values.astype(bool)
    lo, hi = sel["borne_basse"].values.copy(), sel["borne_haute"].values.copy()

    log_ok = log_y and (obs > 0).all() and (sel["y_pred"].values > 0).all()
    if log_ok:
        lo = np.maximum(lo, np.min(obs[obs > 0]) * 1e-3)   # evite log(<=0) sur la borne basse

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.fill_between(x, lo, hi, color="#b9c9f2", alpha=0.5,
                    label=f"Intervalle conforme ({(1-ALPHA)*100:.0f} %)")
    ax.plot(x, lo, color="#4a6fd4", lw=0.9, ls="--")
    ax.plot(x, hi, color="#4a6fd4", lw=0.9, ls="--")
    ax.plot(x, sel["y_pred"].values, color="black", lw=1.7, label="Prediction (y_pred)")
    ax.scatter(x[dedans], obs[dedans], color="green", s=45, zorder=5,
               label=f"Dans l'intervalle  (n={dedans.sum()})")
    ax.scatter(x[~dedans], obs[~dedans], color="red", s=70, zorder=6,
               edgecolor="darkred", linewidth=0.8,
               label=f"HORS intervalle - anomalie  (n={(~dedans).sum()})")

    if log_ok:
        ax.set_yscale("log")
    suffixe = " - echantillon aleatoire" if mode == "echantillon" else " - anomalies sur-representees"
    ax.set_title(f"Intervalles conformes et observations ({len(sel)} unites statistiques){suffixe}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Unites statistiques (triees par prediction croissante)", fontsize=11)
    ax.set_ylabel(TARGET, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(sel["Unit_Stat_Key"].tolist(), rotation=90, ha="right", fontsize=7)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Graphique sauvegarde : {output}")
    plt.show()


def plot_courbe_conforme(df, n=None, log_y=True, output="courbe_conforme.png"):
    """
    Version 'courbe lisse' (2e image de reference) : beaucoup d'observations,
    triees par prediction croissante, sans etiquettes en abscisse.
    Montre la forme globale de la bande et ou se situent les anomalies.
    """
    d = df if n is None else df.sample(min(n, len(df)), random_state=42)
    d = d.sort_values("y_pred").reset_index(drop=True)
    x = np.arange(len(d))
    obs = d["y_obs"].values
    dedans = d["dans_intervalle"].values.astype(bool)
    lo, hi = d["borne_basse"].values.copy(), d["borne_haute"].values

    log_ok = log_y and (obs > 0).all() and (d["y_pred"].values > 0).all()
    if log_ok:
        lo = np.maximum(lo, np.min(obs[obs > 0]) * 1e-3)

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.fill_between(x, lo, hi, color="#c7e3c7", alpha=0.55,
                    label=f"Intervalle conforme ({(1-ALPHA)*100:.0f} %)")
    ax.plot(x, d["y_pred"].values, color="darkgreen", lw=1.8, label="Prediction (CQR)")
    ax.scatter(x[dedans], obs[dedans], s=7, color="green", alpha=0.45,
               label=f"Dans l'intervalle ({100*dedans.mean():.1f} %)")
    ax.scatter(x[~dedans], obs[~dedans], s=22, color="red", alpha=0.9,
               label=f"Anomalies ({100*(~dedans).mean():.1f} %)")

    if log_ok:
        ax.set_yscale("log")
    ax.set_title(f"Couverture conforme sur {len(d):,} observations de test",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Observations triees par prediction croissante", fontsize=11)
    ax.set_ylabel(TARGET, fontsize=11)
    ax.grid(True, ls=":", alpha=0.35)
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Graphique sauvegarde : {output}")
    plt.show()


# Vue 1 : lisible, 40 unites tirees au hasard (proportion d'anomalies reelle)
plot_bande_conforme(DF_VIZ, n=40, mode="echantillon", output="bande_conforme_echantillon.png")

# Vue 2 : centree sur les anomalies (pedagogique pour le tuteur)
plot_bande_conforme(DF_VIZ, n=40, mode="anomalies", output="bande_conforme_anomalies.png")

# Vue 3 : courbe globale sur tout le test
plot_courbe_conforme(DF_VIZ, n=None, output="courbe_conforme_globale.png")
