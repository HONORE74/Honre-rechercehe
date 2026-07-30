import matplotlib.pyplot as plt
import numpy as np

def graphique_bande_conforme(results, log_scale=True, echantillon=None):
    d = results.sort_values("y_pred").reset_index(drop=True)     # tri = substitut de l'axe X
    if echantillon:
        d = d.iloc[np.linspace(0, len(d)-1, echantillon).astype(int)]
    x = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.fill_between(x, d["borne_basse"], d["borne_haute"],
                    color="mediumseagreen", alpha=0.25, label="Intervalle conforme (90%)")
    ax.scatter(x, d["y_obs"], s=8, color="darkgreen", alpha=0.35, label="Observation reelle")
    ax.plot(x, d["y_pred"], color="darkgreen", lw=1.8, label="Prediction (ancre ML)")

    if log_scale:
        ax.set_yscale("symlog", linthresh=1000)       # symlog car y_obs peut etre tres proche de 0
    ax.set_xlabel("Observations triees par prediction croissante")
    ax.set_ylabel(f"{TARGET} (EUR)")
    ax.set_title("Bande de prediction conforme vs observations reelles")
    ax.legend(loc="upper left")
    plt.tight_layout(); plt.savefig("bande_conforme.png", dpi=200); plt.show()

graphique_bande_conforme(results_test, log_scale=True)






def graphique_interval_anomalies(results, log_scale=True):
    d = results.sort_values("y_pred").reset_index(drop=True)
    x = np.arange(len(d))
    ok = d["dans_intervalle"]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.fill_between(x, d["borne_basse"], d["borne_haute"],
                    color="royalblue", alpha=0.20, label="Intervalle de prediction")
    ax.plot(x, d["y_pred"], color="navy", lw=1.5, label="Prediction")
    ax.scatter(x[ok],  d.loc[ok,"y_obs"],  s=12, color="black", alpha=0.6,
              label="Observation (dans l'intervalle)")
    ax.scatter(x[~ok], d.loc[~ok,"y_obs"], s=30, color="red", zorder=5,
              label=f"ANOMALIE hors intervalle (n={(~ok).sum()})")

    if log_scale:
        ax.set_yscale("symlog", linthresh=1000)
    ax.set_xlabel("Observations triees par prediction croissante")
    ax.set_ylabel(f"{TARGET} (EUR)")
    ax.set_title("Detection d'anomalies par intervalle conforme")
    ax.legend(loc="upper left")
    plt.tight_layout(); plt.savefig("interval_anomalies.png", dpi=200); plt.show()

graphique_interval_anomalies(results_test, log_scale=True)





def graphique_zoom_corps(results, percentile_max=95):
    seuil = np.percentile(results["y_obs"], percentile_max)
    sous = results[results["y_obs"] <= seuil]
    graphique_interval_anomalies(sous, log_scale=False)   # lineaire, lisible, sans les extremes
    print(f"Zoom sur les {percentile_max}% de contrats les plus courants "
          f"(< {seuil:,.0f} EUR) -- {len(sous):,} / {len(results):,} lignes")

graphique_zoom_corps(results_test, percentile_max=95)

















import matplotlib.pyplot as plt
import numpy as np

def graphique_bande_conforme(results, strates=None, echantillon=None):
    """Bande de prediction conforme, en echelle LINEAIRE, par strate de magnitude
    (remplace l'ancienne echelle log/symlog)."""
    if strates is None:
        p50, p90, p99 = np.percentile(results["y_obs"], [50, 90, 99])
        strates = [(-np.inf, p50, "P0-50"), (p50, p90, "P50-90"),
                  (p90, p99, "P90-99"), (p99, np.inf, "P99+")]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    for ax, (lo, hi, label) in zip(axes, strates):
        sous = results[(results["y_obs"] >= lo) & (results["y_obs"] < hi)].sort_values("y_pred").reset_index(drop=True)
        if echantillon and len(sous) > echantillon:
            sous = sous.iloc[np.linspace(0, len(sous)-1, echantillon).astype(int)]
        if len(sous) == 0:
            ax.set_visible(False); continue
        x = np.arange(len(sous))

        ax.fill_between(x, sous["borne_basse"], sous["borne_haute"],
                        color="mediumseagreen", alpha=0.25, label="Intervalle conforme (90%)")
        ax.scatter(x, sous["y_obs"], s=8, color="darkgreen", alpha=0.35, label="Observation reelle")
        ax.plot(x, sous["y_pred"], color="darkgreen", lw=1.6, label="Prediction (ancre ML)")

        ax.set_title(f"Strate {label}  (n={len(sous):,})")
        ax.set_xlabel("Observations triees par prediction croissante")
        ax.set_ylabel(f"{TARGET} (EUR)")
        ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout(); plt.savefig("bande_conforme.png", dpi=200); plt.show()

graphique_bande_conforme(results_test)


def graphique_interval_anomalies(results, strates=None):
    """Detection d'anomalies par intervalle conforme, en echelle LINEAIRE, par strate."""
    if strates is None:
        p50, p90, p99 = np.percentile(results["y_obs"], [50, 90, 99])
        strates = [(-np.inf, p50, "P0-50"), (p50, p90, "P50-90"),
                  (p90, p99, "P90-99"), (p99, np.inf, "P99+")]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    for ax, (lo, hi, label) in zip(axes, strates):
        sous = results[(results["y_obs"] >= lo) & (results["y_obs"] < hi)].sort_values("y_pred").reset_index(drop=True)
        if len(sous) == 0:
            ax.set_visible(False); continue
        x = np.arange(len(sous))
        ok = sous["dans_intervalle"]

        ax.fill_between(x, sous["borne_basse"], sous["borne_haute"],
                        color="royalblue", alpha=0.20, label="Intervalle de prediction")
        ax.plot(x, sous["y_pred"], color="navy", lw=1.4, label="Prediction")
        ax.scatter(x[ok],  sous.loc[ok,"y_obs"],  s=10, color="black", alpha=0.5,
                  label="Dans l'intervalle")
        ax.scatter(x[~ok], sous.loc[~ok,"y_obs"], s=28, color="red", zorder=5,
                  label=f"ANOMALIE (n={(~ok).sum()})")

        ax.set_title(f"Strate {label}  (n={len(sous):,})")
        ax.set_xlabel("Observations triees par prediction croissante")
        ax.set_ylabel(f"{TARGET} (EUR)")
        ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout(); plt.savefig("interval_anomalies.png", dpi=200); plt.show()

graphique_interval_anomalies(results_test)


def graphique_zoom_corps(results, percentile_max=95):
    """Vue synthese a UN seul panneau, sur le corps de la distribution (sans les extremes),
    100% lineaire -- ressemble le plus aux 2 exemples de reference."""
    seuil = np.percentile(results["y_obs"], percentile_max)
    sous = results[results["y_obs"] <= seuil].sort_values("y_pred").reset_index(drop=True)
    x = np.arange(len(sous))
    ok = sous["dans_intervalle"]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.fill_between(x, sous["borne_basse"], sous["borne_haute"],
                    color="royalblue", alpha=0.20, label="Intervalle de prediction")
    ax.plot(x, sous["y_pred"], color="navy", lw=1.5, label="Prediction")
    ax.scatter(x[ok],  sous.loc[ok,"y_obs"],  s=12, color="black", alpha=0.6,
              label="Observation (dans l'intervalle)")
    ax.scatter(x[~ok], sous.loc[~ok,"y_obs"], s=30, color="red", zorder=5,
              label=f"ANOMALIE hors intervalle (n={(~ok).sum()})")

    ax.set_xlabel("Observations triees par prediction croissante")
    ax.set_ylabel(f"{TARGET} (EUR)")
    ax.set_title(f"Zoom sur les {percentile_max}% de contrats les plus courants")
    ax.legend(loc="upper left")
    plt.tight_layout(); plt.savefig("zoom_corps.png", dpi=200); plt.show()

    print(f"Zoom sur les {percentile_max}% de contrats les plus courants "
          f"(< {seuil:,.0f} EUR) -- {len(sous):,} / {len(results):,} lignes")

graphique_zoom_corps(results_test, percentile_max=95)
