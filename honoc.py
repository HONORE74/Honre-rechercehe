def graphique_relation_reelle(results, x_col="y_pred", strates=None):
    """Reproduit le style des graphiques de reference : axe X = une VRAIE variable
    continue (pas un simple rang trie), sans aucun logarithme."""
    if strates is None:
        p50, p90, p99 = np.percentile(results["y_obs"], [50, 90, 99])
        strates = [(-np.inf, p50, "P0-50"), (p50, p90, "P50-90"),
                  (p90, p99, "P90-99"), (p99, np.inf, "P99+")]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    for ax, (lo, hi, label) in zip(axes, strates):
        sous = results[(results["y_obs"] >= lo) & (results["y_obs"] < hi)].sort_values(x_col)
        if len(sous) == 0:
            ax.set_visible(False); continue
        x = sous[x_col].values                       # 🔴 vraie variable, plus un simple rang
        ok = sous["dans_intervalle"]

        ax.fill_between(x, sous["borne_basse"], sous["borne_haute"],
                        color="mediumseagreen", alpha=0.25, label="Intervalle conforme (90%)")
        ax.scatter(x[ok],  sous.loc[ok,"y_obs"],  s=10, color="darkgreen", alpha=0.4,
                  label="Observation (dans l'intervalle)")
        ax.scatter(x[~ok], sous.loc[~ok,"y_obs"], s=28, color="red", zorder=5,
                  label=f"Anomalie (n={(~ok).sum()})")
        ax.plot(x, sous["y_pred"], color="darkgreen", lw=1.2, alpha=0.6, label="Prediction")

        ax.set_title(f"Strate {label}  (n={len(sous):,})")
        ax.set_xlabel(x_col); ax.set_ylabel(f"{TARGET} (EUR)")
        ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout(); plt.savefig("relation_reelle.png", dpi=200); plt.show()

graphique_relation_reelle(results_test, x_col="RBNS_bop")












import numpy as np
import matplotlib.pyplot as plt

def graphique_relation_reelle(results, x_col="RBNS_bop", strates=None):
    """Observations dans l'intervalle en BLEU, anomalies (hors intervalle) en ROUGE.
    Contenu centre dans chaque repere via des marges calculees sur les donnees.
    Aucun logarithme."""
    if strates is None:
        p50, p90, p99 = np.percentile(results["y_obs"], [50, 90, 99])
        strates = [(-np.inf, p50, "P0-50"), (p50, p90, "P50-90"),
                  (p90, p99, "P90-99"), (p99, np.inf, "P99+")]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    for ax, (lo, hi, label) in zip(axes, strates):
        sous = results[(results["y_obs"] >= lo) & (results["y_obs"] < hi)].sort_values(x_col)
        if len(sous) == 0:
            ax.set_visible(False); continue
        x = sous[x_col].values
        ok = sous["dans_intervalle"]

        ax.fill_between(x, sous["borne_basse"], sous["borne_haute"],
                        color="lightsteelblue", alpha=0.35, label="Intervalle conforme (90%)")
        ax.plot(x, sous["y_pred"], color="steelblue", lw=1.3, alpha=0.8, label="Prediction")

        # 🔵 observations DANS l'intervalle -> BLEU
        ax.scatter(x[ok], sous.loc[ok, "y_obs"], s=14, color="royalblue", alpha=0.55,
                  edgecolor="none", label="Observation (dans l'intervalle)")
        # 🔴 anomalies HORS intervalle -> ROUGE, au premier plan
        ax.scatter(x[~ok], sous.loc[~ok, "y_obs"], s=32, color="red", zorder=5,
                  edgecolor="darkred", linewidth=0.4, label=f"Anomalie (n={(~ok).sum()})")

        # --- Centrage du contenu dans le repere (evite le contenu ecrase dans un coin) ---
        all_y = np.concatenate([sous["y_obs"].values, sous["borne_basse"].values,
                                sous["borne_haute"].values])
        y_lo, y_hi = np.percentile(all_y, [1, 99])
        marge_y = (y_hi - y_lo) * 0.12
        ax.set_ylim(max(0, y_lo - marge_y), y_hi + marge_y)
        ax.margins(x=0.04)

        ax.set_title(f"Strate {label}  (n={len(sous):,})")
        ax.set_xlabel(x_col); ax.set_ylabel(f"{TARGET} (EUR)")
        ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout(); plt.savefig("relation_reelle.png", dpi=200); plt.show()

graphique_relation_reelle(results_test, x_col="RBNS_bop")
