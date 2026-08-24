import numpy as np, pandas as pd

def preparer_exploration(results_v2, anomalies_prio):
    """Enrichit results_v2 avec le score, la position normalisee z et le statut."""
    cles = [c for c in ID_COLS
            if c in results_v2.columns and c in anomalies_prio.columns]
    for c in ["year", "quarter"]:
        if c in results_v2.columns and c in anomalies_prio.columns:
            cles.append(c)

    d = results_v2.copy()
    a_joindre = [c for c in ["score_composite", "rank", "A_ecart_borne",
                             "B_erreur_modele"] if c in anomalies_prio.columns]
    if a_joindre and cles:
        d = d.merge(anomalies_prio[cles + a_joindre], on=cles, how="left")
    for c in ["score_composite", "rank", "A_ecart_borne", "B_erreur_modele"]:
        if c not in d.columns:
            d[c] = np.nan

    centre = (d["borne_haute"] + d["borne_basse"]) / 2
    demi = np.maximum((d["borne_haute"] - d["borne_basse"]) / 2, 1e-9)
    d["z"]            = (d["y_obs"] - centre) / demi
    d["largeur"]      = d["borne_haute"] - d["borne_basse"]
    d["largeur_rel"]  = d["largeur"] / np.maximum(d["y_pred"].abs(), 1e-9)
    d["est_anomalie"] = (~d["dans_intervalle"]).astype(int)
    d["statut"]       = np.where(d["dans_intervalle"], "Couverte", "Hors intervalle")
    d["identite"]     = (d[[c for c in ID_COLS if c in d.columns]]
                         .astype(str).agg(" | ".join, axis=1))
    return d


expl = preparer_exploration(results_v2, anomalies_prio)
print(f"expl : {len(expl):,} observations | "
      f"{int(expl['est_anomalie'].sum()):,} anomalies "
      f"({100*expl['est_anomalie'].mean():.1f} %)")
print(f"Couverture : {100*expl['dans_intervalle'].mean():.1f} %")
print(f"Colonnes ID presentes : {[c for c in ID_COLS if c in expl.columns]}")
