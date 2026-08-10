# =========================================================================
# CELLULE 16 : FIGURES PRINCIPALES — TOP 15 ANOMALIES PRIORISEES
# =========================================================================
# 1. Vue absolue      : le croquis de l'issue #22, en qualite publication
# 2. Vue normalisee   : intervalles ramenes a [-1,+1] -> aucun aplatissement
# 3. Forest plot      : grammaire rigoureuse pour entites discretes
# 4. Decomposition    : QUEL facteur (A, B, GWP) explique le classement
# =========================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

TOP_N      = 15   # anomalies prioritaires affichees
N_REF      = 8    # unites normales de reference (contraste vert)
MAXLEN_LAB = 28   # troncature des etiquettes


def _labels(df, id_cols=None, maxlen=MAXLEN_LAB):
    """Etiquette lisible par unite statistique."""
    id_cols = id_cols or [c for c in ID_COLS if c in df.columns]
    lab = df[id_cols].astype(str).agg("_".join, axis=1)
    return lab.str.slice(0, maxlen)


def _preparer_donnees(anomalies_prio, results_v2, top_n=TOP_N, n_ref=N_REF):
    """
    Top N anomalies (par score decroissant) + N_REF unites normales de reference.

    Les unites de reference sont choisies parmi les PLUS GROS GWP, et non au
    hasard : les anomalies prioritaires ont mecaniquement un GWP eleve, une
    reference tiree au hasard serait a une echelle sans rapport et la
    comparaison visuelle n'aurait aucun sens.
    """
    top = anomalies_prio.head(top_n).copy()
    top["_bloc"] = "anomalie"

    normales = results_v2[results_v2["dans_intervalle"]].copy()
    if GWP_COL in normales.columns and len(normales):
        ref = normales.nlargest(min(n_ref, len(normales)), GWP_COL).copy()
    else:
        ref = normales.head(n_ref).copy()
    ref["_bloc"] = "reference"

    for c in ["A_ecart_borne", "B_erreur_modele", "score_composite", "rank"]:
        if c not in ref.columns:
            ref[c] = np.nan

    cols = [c for c in ID_COLS if c in top.columns] + [
        "y_obs", "y_pred", "borne_basse", "borne_haute", "dans_intervalle",
        "A_ecart_borne", "B_erreur_modele", "score_composite", "rank", "_bloc"]
    if GWP_COL in top.columns and GWP_COL in ref.columns:
        cols.append(GWP_COL)
    cols = [c for c in cols if c in top.columns and c in ref.columns]

    d = pd.concat([top[cols], ref[cols]], ignore_index=True)
    d["label"] = _labels(d)
    return d


# =========================================================================
# FIGURE 1 — Vue absolue (le croquis, en qualite publication)
# =========================================================================

def figure_absolue(anomalies_prio, results_v2, top_n=TOP_N, n_ref=N_REF,
                   log_y=True, output="fig1_top_anomalies_absolu.png"):
    d = _preparer_donnees(anomalies_prio, results_v2, top_n, n_ref)
    x = np.arange(len(d))
    n_ano = int((d["_bloc"] == "anomalie").sum())
    dedans = d["dans_intervalle"].values.astype(bool)

    lo = d["borne_basse"].values.astype(float).copy()
    hi = d["borne_haute"].values.astype(float)
    obs = d["y_obs"].values.astype(float)
    pred = d["y_pred"].values.astype(float)

    log_ok = log_y and (obs > 0).all() and (pred > 0).all()
    if log_ok:
        lo = np.maximum(lo, np.nanmin(obs[obs > 0]) * 1e-3)

    fig, ax = plt.subplots(figsize=(17, 8))

    # Bande de fond : rappelle le croquis, sans pretendre a une continuite reelle
    ax.fill_between(x, lo, hi, color="#a9c4ea", alpha=0.22, step="mid", zorder=1)

    # Segment vertical par unite : la representation rigoureuse
    ax.vlines(x, lo, hi, color="#3a6bbf", lw=2.6, alpha=0.85, zorder=2)
    ax.scatter(x, lo, marker="_", s=190, color="#3a6bbf", lw=2, zorder=3)
    ax.scatter(x, hi, marker="_", s=190, color="#3a6bbf", lw=2, zorder=3)

    # Prediction
    ax.scatter(x, pred, marker="D", s=42, facecolor="white",
               edgecolor="black", lw=1.5, zorder=5)

    # Observations
    ax.scatter(x[~dedans], obs[~dedans], s=135, color="#c0392b",
               edgecolor="darkred", lw=1.2, zorder=6)
    ax.scatter(x[dedans], obs[dedans], s=110, color="#27ae60",
               edgecolor="darkgreen", lw=1.2, zorder=6)

    # Trait de dépassement : matérialise l'écart A_i
    for i in np.where(~dedans)[0]:
        cible = hi[i] if obs[i] > hi[i] else lo[i]
        ax.plot([x[i], x[i]], [cible, obs[i]], color="#c0392b",
                lw=1.4, ls=":", zorder=4)

    # Séparation anomalies / référence
    if n_ano < len(d):
        ax.axvline(n_ano - 0.5, color="gray", ls="--", lw=1.6, alpha=0.8)
        ax.text(n_ano - 0.45, ax.get_ylim()[1], "  unites normales (reference)",
                va="top", ha="left", fontsize=9, color="gray", style="italic")

    # Rang de priorite au-dessus de chaque anomalie
    for i in range(n_ano):
        ax.annotate(f"#{int(d['rank'].iloc[i])}", (x[i], obs[i]),
                    textcoords="offset points", xytext=(0, 13),
                    ha="center", fontsize=8, fontweight="bold", color="#7b241c")

    if log_ok:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(d["label"], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Unites statistiques — top anomalies triees par score de priorisation decroissant",
                  fontsize=11)
    ax.set_ylabel(f"{TARGET}" + ("  (echelle log)" if log_ok else ""), fontsize=11)
    ax.set_title(f"Intervalles conformes et valeurs observees — top {n_ano} anomalies prioritaires",
                 fontsize=13.5, fontweight="bold")

    legende = [
        Patch(facecolor="#a9c4ea", edgecolor="#3a6bbf",
              label=f"Intervalle conforme ({100*(1-ALPHA):.0f} %)"),
        Line2D([], [], marker="D", ls="", mfc="white", mec="black", ms=8,
               label="Prediction du modele"),
        Line2D([], [], marker="o", ls="", color="#c0392b", ms=10,
               label="Observation HORS intervalle"),
        Line2D([], [], marker="o", ls="", color="#27ae60", ms=10,
               label="Observation dans l'intervalle"),
    ]
    ax.legend(handles=legende, loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(True, axis="y", ls=":", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Sauvegarde : {output}")
    plt.show()


# =========================================================================
# FIGURE 2 — Vue normalisee : plus aucun aplatissement possible
# =========================================================================

def figure_normalisee(anomalies_prio, results_v2, top_n=TOP_N, n_ref=N_REF,
                      output="fig2_top_anomalies_normalise.png"):
    """
    Changement de repere : z_i = (y_obs - centre_i) / demi_largeur_i

        |z| <= 1  -> dans l'intervalle
        |z| >  1  -> hors intervalle, et |z| se lit directement comme la severite

    L'echelle monetaire disparait : un portefeuille a 300 M et un a 50 k
    deviennent strictement comparables. C'est la reponse structurelle au
    probleme d'aplatissement — bien plus efficace qu'un echantillonnage.
    """
    d = _preparer_donnees(anomalies_prio, results_v2, top_n, n_ref)
    centre = (d["borne_haute"].values + d["borne_basse"].values) / 2
    demi = np.maximum((d["borne_haute"].values - d["borne_basse"].values) / 2, 1e-9)
    z_obs = (d["y_obs"].values - centre) / demi
    z_pred = (d["y_pred"].values - centre) / demi
    x = np.arange(len(d))
    dedans = d["dans_intervalle"].values.astype(bool)
    n_ano = int((d["_bloc"] == "anomalie").sum())

    fig, ax = plt.subplots(figsize=(17, 7.5))
    ax.axhspan(-1, 1, color="#a9c4ea", alpha=0.35,
               label=f"Zone conforme ({100*(1-ALPHA):.0f} %)")
    ax.axhline(1, color="#3a6bbf", lw=1.6)
    ax.axhline(-1, color="#3a6bbf", lw=1.6)
    ax.axhline(0, color="gray", lw=1, ls="--", alpha=0.7)

    ax.vlines(x[~dedans], np.sign(z_obs[~dedans]), z_obs[~dedans],
              color="#c0392b", lw=2.2, alpha=0.75, zorder=3)
    ax.scatter(x, z_pred, marker="D", s=38, facecolor="white",
               edgecolor="black", lw=1.3, zorder=5, label="Prediction")
    ax.scatter(x[~dedans], z_obs[~dedans], s=140, color="#c0392b",
               edgecolor="darkred", lw=1.2, zorder=6, label="Hors intervalle")
    ax.scatter(x[dedans], z_obs[dedans], s=110, color="#27ae60",
               edgecolor="darkgreen", lw=1.2, zorder=6, label="Dans l'intervalle")

    for i in np.where(~dedans)[0]:
        ax.annotate(f"{abs(z_obs[i]):.1f}x", (x[i], z_obs[i]),
                    textcoords="offset points",
                    xytext=(0, 12 if z_obs[i] > 0 else -18),
                    ha="center", fontsize=8, fontweight="bold", color="#7b241c")

    if n_ano < len(d):
        ax.axvline(n_ano - 0.5, color="gray", ls="--", lw=1.6, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(d["label"], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Unites statistiques — triees par score de priorisation decroissant", fontsize=11)
    ax.set_ylabel("Position normalisee  z = (obs - centre) / demi-largeur", fontsize=11)
    ax.set_title("Vue normalisee — tous les portefeuilles a la meme echelle\n"
                 "z au-dela de +-1 : l'observation sort de l'intervalle conforme",
                 fontsize=13.5, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(True, axis="y", ls=":", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Sauvegarde : {output}")
    plt.show()


# =========================================================================
# FIGURE 3 — Forest plot (grammaire rigoureuse pour entites discretes)
# =========================================================================

def figure_forest(anomalies_prio, top_n=TOP_N, log_x=True,
                  output="fig3_forest_plot.png"):
    d = anomalies_prio.head(top_n).copy()
    d["label"] = _labels(d)
    d = d.iloc[::-1].reset_index(drop=True)     # rang 1 en haut
    y = np.arange(len(d))

    lo = d["borne_basse"].values.astype(float).copy()
    hi = d["borne_haute"].values.astype(float)
    obs = d["y_obs"].values.astype(float)
    pred = d["y_pred"].values.astype(float)
    log_ok = log_x and (obs > 0).all() and (pred > 0).all()
    if log_ok:
        lo = np.maximum(lo, np.nanmin(obs[obs > 0]) * 1e-3)

    fig, ax = plt.subplots(figsize=(13, 0.52 * len(d) + 3))
    ax.hlines(y, lo, hi, color="#3a6bbf", lw=6, alpha=0.35, zorder=1)
    ax.scatter(lo, y, marker="|", s=210, color="#3a6bbf", lw=2.2, zorder=2)
    ax.scatter(hi, y, marker="|", s=210, color="#3a6bbf", lw=2.2, zorder=2)
    ax.scatter(pred, y, marker="D", s=48, facecolor="white",
               edgecolor="black", lw=1.5, zorder=4, label="Prediction")
    ax.scatter(obs, y, s=145, color="#c0392b", edgecolor="darkred",
               lw=1.2, zorder=5, label="Valeur comptabilisee")
    for i in range(len(d)):
        cible = hi[i] if obs[i] > hi[i] else lo[i]
        ax.plot([cible, obs[i]], [y[i], y[i]], color="#c0392b", lw=1.3, ls=":", zorder=3)

    if log_ok:
        ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([f"#{int(r)}  {l}" for r, l in zip(d["rank"], d["label"])],
                       fontsize=9)
    ax.set_xlabel(f"{TARGET}" + ("  (echelle log)" if log_ok else ""), fontsize=11)
    ax.set_title(f"Top {top_n} anomalies — intervalle conforme, prediction et valeur observee",
                 fontsize=13, fontweight="bold")
    handles = [Patch(facecolor="#3a6bbf", alpha=0.35,
                     label=f"Intervalle conforme ({100*(1-ALPHA):.0f} %)")] + \
              ax.get_legend_handles_labels()[0]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    ax.grid(True, axis="x", ls=":", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Sauvegarde : {output}")
    plt.show()


# =========================================================================
# FIGURE 4 — Qu'est-ce qui explique le classement ?
# =========================================================================

def figure_decomposition(anomalies_prio, top_n=TOP_N,
                         output="fig4_decomposition_score.png"):
    """
    Pour chaque anomalie du top N, rang percentile de chaque facteur parmi
    TOUTES les anomalies. Repond a : "pourquoi celle-ci est-elle n°1 ?"

    Lecture critique : si la colonne GWP est rouge partout et les colonnes
    A et B pales, le classement ne fait que trier par taille de portefeuille
    — c'est exactement le risque de la GWP brute, ici rendu visible.
    """
    facteurs = {"A\n(ecart borne)": "A_ecart_borne",
                "B\n(erreur modele)": "B_erreur_modele",
                f"{GWP_COL}\n(exposition)": GWP_COL}
    facteurs = {k: v for k, v in facteurs.items() if v in anomalies_prio.columns}

    pct = pd.DataFrame({k: anomalies_prio[v].rank(pct=True) * 100
                        for k, v in facteurs.items()})
    pct["Score\nfinal"] = anomalies_prio["score_composite"].rank(pct=True) * 100
    d = pct.head(top_n)
    labels = _labels(anomalies_prio.head(top_n))
    rangs = anomalies_prio["rank"].head(top_n).astype(int)

    fig, ax = plt.subplots(figsize=(9, 0.48 * len(d) + 3))
    im = ax.imshow(d.values, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(d.shape[1]))
    ax.set_xticklabels(d.columns, fontsize=9)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"#{r}  {l}" for r, l in zip(rangs, labels)], fontsize=8.5)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            v = d.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if (v > 70 or v < 25) else "black")
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label("Rang percentile parmi toutes les anomalies", fontsize=9)
    ax.set_title("Quel facteur explique la priorite ?\n"
                 "100 = le plus eleve de tout le lot",
                 fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Sauvegarde : {output}")
    plt.show()

    corr = anomalies_prio[["score_composite"] + list(facteurs.values())].corr(
        method="spearman")["score_composite"].drop("score_composite")
    print("\nCorrelation de Spearman entre le score final et chaque facteur :")
    print(corr.round(3).to_string())
    dom = corr.idxmax()
    print(f"\n  Facteur dominant : {dom}  (rho = {corr.max():.3f})")
    if corr.max() > 0.9:
        print("  /!\\ Correlation > 0.9 : le classement est quasi entierement porte")
        print("      par ce seul facteur. Les deux autres n'apportent presque rien.")


# --- Execution ------------------------------------------------------------
figure_absolue(anomalies_prio, results_v2)
figure_normalisee(anomalies_prio, results_v2)
figure_forest(anomalies_prio)
figure_decomposition(anomalies_prio)
