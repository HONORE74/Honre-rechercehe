import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy.stats import beta, binom, spearmanr

os.makedirs("figures_conformal", exist_ok=True)

VERT       = "#2E7D5B"
VERT_CLAIR = "#A8D5BF"
ROUGE      = "#C0392B"
BLEU       = "#7FA6D9"
BLEU_CLAIR = "#D3E0F2"
NOIR       = "#1B2A38"
ORANGE     = "#E67E22"
GRIS       = "#8C99A6"


def montant(v, pos=None):
    a = abs(v)
    if a >= 1e9: return f"{v/1e9:,.2f} Md"
    if a >= 1e6: return f"{v/1e6:,.1f} M"
    if a >= 1e3: return f"{v/1e3:,.0f} k"
    return f"{v:,.0f}"


def sauver(fig, nom):
    fig.tight_layout()
    fig.savefig(f"figures_conformal/{nom}.png", dpi=200, bbox_inches="tight")
    plt.show()



















def fig_01_profil_cible(results, target="RBNS_eop"):
    y = np.sort(results["y_obs"].to_numpy(float))
    rang = np.linspace(0, 100, len(y))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(rang, y, color=NOIR, lw=2, label="Valeur observee (triee)")
    ax.fill_between(rang, 0, y, color=BLEU_CLAIR, alpha=0.5)
    for p in (50, 90, 99):
        ax.axvline(p, color=ORANGE, ls="--", lw=1)
        ax.text(p, np.percentile(y, 99), f" P{p} = {montant(np.percentile(y, p))}",
                rotation=90, va="top", fontsize=8, color=ORANGE)

    ax.set_ylim(0, np.percentile(y, 99) * 1.1)
    ax.set_xlim(0, 100)
    ax.set_title(f"Profil de la cible {target} : distribution a queue lourde")
    ax.set_xlabel("Rang de l'observation (percentile)")
    ax.set_ylabel(f"{target} (EUR)")
    ax.yaxis.set_major_formatter(FuncFormatter(montant))
    ax.xaxis.set_major_formatter(PercentFormatter(100))
    ax.grid(ls=":", alpha=0.5)
    ax.legend(loc="upper left")
    sauver(fig, "fig_01_profil_cible")



















def fig_02_concentration(results, target="RBNS_eop"):
    y = np.sort(results["y_obs"].to_numpy(float))
    part_obs = np.linspace(0, 1, len(y))
    part_mnt = np.cumsum(y) / y.sum()
    gini = 1 - 2 * np.trapz(part_mnt, part_obs)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(part_obs, part_mnt, color=NOIR, lw=2.5,
            label=f"Concentration observee (Gini = {gini:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=ORANGE, label="Repartition egale")
    ax.fill_between(part_obs, part_mnt, part_obs, color=BLEU_CLAIR, alpha=0.6)

    idx = int(0.90 * (len(y) - 1))
    ax.annotate(f"Les 10 % les plus eleves\nportent {1-part_mnt[idx]:.1%} du montant",
                xy=(0.90, part_mnt[idx]), xytext=(0.42, part_mnt[idx] + 0.20),
                fontsize=9.5, color=ROUGE,
                arrowprops=dict(arrowstyle="->", color=ROUGE))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Concentration du montant : ou se joue l'enjeu de controle")
    ax.set_xlabel("Part cumulee des unites statistiques")
    ax.set_ylabel(f"Part cumulee du montant de {target}")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(ls=":", alpha=0.5)
    ax.legend(loc="upper left")
    sauver(fig, "fig_02_concentration")






















def fig_03_calibration(results_pv, calib_scores, alpha=0.10):
    s_test = results_pv["score_nonconformite"].to_numpy(float)
    cal = np.sort(np.asarray(calib_scores, float))
    n_cal, n_test = len(cal), len(s_test)
    alphas = np.linspace(0.01, 0.30, 30)

    cov = []
    for a in alphas:
        niv = min(np.ceil((n_cal + 1) * (1 - a)) / n_cal, 1.0)
        cov.append(np.mean(s_test <= np.quantile(cal, niv, method="higher")))
    cov = np.array(cov)
    nominal = 1 - alphas
    demi = 1.96 * np.sqrt(nominal * (1 - nominal) / n_test)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.fill_between(nominal, nominal - demi, nominal + demi, color=GRIS,
                    alpha=0.25, label="Tolerance a 95 %")
    ax.plot(nominal, nominal, ls="--", color=ORANGE, label="Calibration parfaite")
    ax.plot(nominal, cov, "o-", color=NOIR, ms=4, label="Couverture empirique")
    ax.axvline(1 - alpha, color=ROUGE, ls=":", lw=1.8,
               label=f"Niveau retenu : {100*(1-alpha):.0f} %")

    ax.set_title(f"Validation de la calibration conforme "
                 f"(ecart max = {100*np.max(np.abs(cov-nominal)):.2f} pts)")
    ax.set_xlabel("Couverture nominale (1 - alpha)")
    ax.set_ylabel("Couverture empirique constatee")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(ls=":", alpha=0.5)
    ax.legend(loc="lower right")
    sauver(fig, "fig_03_calibration")






















def fig_04_histogramme_pvalues(results_pv, n_bins=20, lam=0.5):
    p = results_pv["p_value"].to_numpy(float)
    p = p[np.isfinite(p)]
    m = len(p)
    pi0 = min(np.sum(p > lam) / (m * (1 - lam)), 1.0)

    fig, ax = plt.subplots(figsize=(11, 6))
    effectifs, bords, patches = ax.hist(p, bins=n_bins, range=(0, 1),
                                        color=BLEU, edgecolor="white")
    patches[0].set_facecolor(ROUGE)
    ax.axhline(m / n_bins, color=ORANGE, ls="--", lw=2,
               label="Effectif attendu si aucune anomalie")

    ax.text(0.97, 0.94, f"Observations saines estimees : {pi0:.3f}\n"
                        f"Vraies anomalies estimees : {m*(1-pi0):,.0f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="#FFFBE6", ec=ORANGE))

    ax.set_xlim(0, 1)
    ax.set_title("Distribution des p-values conformes : y a-t-il un signal ?")
    ax.set_xlabel("p-value conforme")
    ax.set_ylabel("Nombre d'observations")
    ax.grid(ls=":", alpha=0.5)
    ax.legend(loc="upper center")
    sauver(fig, "fig_04_pvalues_hist")


























def fig_05_qqplot_pvalues(results_pv):
    p = np.sort(results_pv["p_value"].to_numpy(float))
    p = p[np.isfinite(p)]
    m = len(p)
    theorique = (np.arange(1, m + 1) - 0.5) / m
    i = np.arange(1, m + 1)
    bas = beta.ppf(0.025, i, m - i + 1)
    haut = beta.ppf(0.975, i, m - i + 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.fill_between(theorique, bas, haut, color=GRIS, alpha=0.3,
                    label="Enveloppe a 95 % (loi uniforme)")
    ax.plot([0, 1], [0, 1], ls="--", color=ORANGE, label="Aucune anomalie")
    ax.plot(theorique, p, lw=2.2, color=NOIR, label="p-values observees")

    sous = p < bas
    if sous.any():
        ax.scatter(theorique[sous], p[sous], s=10, color=ROUGE, zorder=5,
                   label=f"Sous l'enveloppe (n={sous.sum():,})")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("QQ-plot des p-values contre la loi uniforme")
    ax.set_xlabel("Quantiles theoriques")
    ax.set_ylabel("p-values triees")
    ax.grid(ls=":", alpha=0.5)
    ax.legend(loc="lower right")
    sauver(fig, "fig_05_qqplot_pvalues")























def controle_fdr_bh(pvalues, q=0.05):
    p = np.asarray(pvalues, float)
    m = len(p)
    ordre = np.argsort(p)
    p_tri = p[ordre]
    passe = p_tri <= q * np.arange(1, m + 1) / m
    k = int(np.max(np.where(passe)[0]) + 1) if passe.any() else 0
    retenu = np.zeros(m, dtype=bool)
    if k:
        retenu[ordre[:k]] = True
    return retenu, k, (p_tri[k - 1] if k else 0.0)


def fig_06_benjamini_hochberg(results_pv, q=0.05, k_max=400):
    p = results_pv["p_value"].to_numpy(float)
    p = p[np.isfinite(p)]
    m = len(p)
    p_tri = np.sort(p)
    _, n_bh, seuil = controle_fdr_bh(p, q)

    k = int(min(m, max(k_max, 2 * max(n_bh, 1))))
    rangs = np.arange(1, k + 1)
    droite = q * rangs / m

    fig, ax = plt.subplots(figsize=(11, 6))
    couleurs = np.where(p_tri[:k] <= droite, ROUGE, GRIS)
    ax.scatter(rangs, p_tri[:k], s=12, c=couleurs, label="p-values triees")
    ax.plot(rangs, droite, lw=2, color=ORANGE, label=f"Seuil BH (q = {q:.0%})")
    if n_bh:
        ax.axvline(n_bh, color=NOIR, ls="--", lw=1.8,
                   label=f"{n_bh:,} anomalies retenues")

    ax.set_xlim(0, k)
    ax.set_ylim(0, max(p_tri[:k].max(), droite.max()) * 1.05)
    ax.set_title(f"Controle du taux de fausses decouvertes "
                 f"(seuil effectif = {seuil:.2e})")
    ax.set_xlabel("Rang de la p-value")
    ax.set_ylabel("p-value conforme")
    ax.grid(ls=":", alpha=0.5)
    ax.legend(loc="upper left")
    sauver(fig, "fig_06_benjamini_hochberg")


















def fig_07_couverture_par_largeur(results, alpha=0.10, n_deciles=10):
    d = results.copy()
    d["largeur"] = d["borne_haute"] - d["borne_basse"]
    d["bin"] = pd.qcut(d["largeur"], n_deciles, labels=False, duplicates="drop")
    agg = d.groupby("bin").agg(couverture=("dans_intervalle", "mean"),
                               effectif=("dans_intervalle", "size"),
                               larg_med=("largeur", "median")).reset_index()

    cible = 1 - alpha
    err = 1.96 * np.sqrt(agg["couverture"] * (1 - agg["couverture"]) / agg["effectif"])
    couleurs = np.where(np.abs(agg["couverture"] - cible) > 0.05, ROUGE, VERT)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(agg["bin"], agg["couverture"], color=couleurs, edgecolor="white")
    ax.errorbar(agg["bin"], agg["couverture"], yerr=err, fmt="none",
                ecolor=NOIR, capsize=4)
    ax.axhline(cible, color=ORANGE, ls="--", lw=2, label=f"Cible {cible:.0%}")
    for _, r in agg.iterrows():
        ax.text(r["bin"], r["couverture"] - 0.055, f"{r['couverture']:.0%}",
                ha="center", color="white", fontweight="bold", fontsize=9)

    ax.set_ylim(0, 1.05)
    ax.set_xticks(agg["bin"])
    ax.set_xticklabels([f"D{int(b)+1}\n{montant(l)}"
                        for b, l in zip(agg["bin"], agg["larg_med"])], fontsize=8)
    ax.set_title(f"Couverture par decile de largeur "
                 f"(violation max = {100*np.abs(agg['couverture']-cible).max():.1f} pts)")
    ax.set_xlabel("Decile de largeur (etroit -> large)")
    ax.set_ylabel("Couverture empirique")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend(loc="lower right")
    sauver(fig, "fig_07_ssc")





















def fig_08_largeurs(results):
    larg = np.sort((results["borne_haute"] - results["borne_basse"]).to_numpy(float))
    rang = np.linspace(0, 100, len(larg))
    ratio = np.percentile(larg, 90) / max(np.percentile(larg, 10), 1e-9)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(rang, larg, lw=2.2, color=BLEU, label="Largeur d'intervalle (triee)")
    ax.fill_between(rang, 0, larg, color=BLEU_CLAIR, alpha=0.6)
    ax.axhline(np.median(larg), color=ORANGE, ls="--",
               label=f"Largeur mediane : {montant(np.median(larg))}")

    ax.text(0.03, 0.90, f"Rapport P90 / P10 : {ratio:,.1f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec=GRIS))

    ax.set_ylim(0, np.percentile(larg, 99) * 1.1)
    ax.set_xlim(0, 100)
    ax.set_title("Adaptativite des intervalles : distribution des largeurs")
    ax.set_xlabel("Rang de l'unite statistique (percentile)")
    ax.set_ylabel("Largeur de l'intervalle conforme (EUR)")
    ax.yaxis.set_major_formatter(FuncFormatter(montant))
    ax.xaxis.set_major_formatter(PercentFormatter(100))
    ax.grid(ls=":", alpha=0.5)
    ax.legend(loc="upper left")
    sauver(fig, "fig_08_largeurs")
















def fig_09_bande_normalisee(results, alpha=0.10, z_max=6.0):
    d = results.sort_values("y_pred").reset_index(drop=True)
    centre = (d["borne_haute"] + d["borne_basse"]) / 2
    demi = np.maximum((d["borne_haute"] - d["borne_basse"]) / 2, 1e-9)
    z = ((d["y_obs"] - centre) / demi).to_numpy(float)
    dedans = d["dans_intervalle"].to_numpy(bool)
    x = np.arange(len(d))

    z_bas = float(np.clip(min(-1.3, np.percentile(z, 0.2)), -z_max, -1.3))
    z_haut = float(np.clip(max(1.3, np.percentile(z, 99.8)), 1.3, z_max))
    z_aff = np.clip(z, z_bas, z_haut)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axhspan(-1, 1, color=VERT_CLAIR, alpha=0.55,
               label=f"Zone conforme {100*(1-alpha):.0f} % (|z| <= 1)")
    ax.axhline(0, color=NOIR, ls="--", lw=1.2)
    ax.scatter(x[dedans], z_aff[dedans], s=7, color=VERT, alpha=0.45,
               label=f"Dans l'intervalle : {100*dedans.mean():.1f} %")
    ax.scatter(x[~dedans], z_aff[~dedans], s=24, color=ROUGE, alpha=0.9,
               label=f"Hors intervalle : {100*(~dedans).mean():.1f} % "
                     f"(n={(~dedans).sum():,})")

    marge = 0.10 * (z_haut - z_bas)
    ax.set_ylim(z_bas - marge, z_haut + 2.2 * marge)
    ax.set_xlim(0, len(z))
    ax.set_title(f"Bande conforme normalisee (couverture = {100*dedans.mean():.2f} %)")
    ax.set_xlabel("Unites statistiques, triees par prediction croissante")
    ax.set_ylabel("z = (observation - centre) / demi-largeur")
    ax.grid(ls=":", alpha=0.5)
    ax.legend(loc="lower right")
    sauver(fig, "fig_09_bande_normalisee")





























def fig_10_bande_strate(results, strate="P50-90", alpha=0.10,
                        target="RBNS_eop", max_points=1200):
    p50, p90, p99 = np.percentile(results["y_pred"], [50, 90, 99])
    bornes = {"P0-50": (-np.inf, p50), "P50-90": (p50, p90),
              "P90-99": (p90, p99), "P99+": (p99, np.inf)}
    lo_s, hi_s = bornes[strate]

    d = results[(results["y_pred"] >= lo_s) & (results["y_pred"] < hi_s)]
    if len(d) > max_points:
        d = d.sample(max_points, random_state=42)
    d = d.sort_values("y_pred").reset_index(drop=True)

    x = np.arange(len(d))
    obs = d["y_obs"].to_numpy(float)
    dedans = d["dans_intervalle"].to_numpy(bool)
    plafond = np.percentile(np.concatenate([d["borne_haute"], obs]), 99)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.fill_between(x, d["borne_basse"], d["borne_haute"], color=BLEU_CLAIR,
                    alpha=0.85, label=f"Intervalle conforme {100*(1-alpha):.0f} %")
    ax.plot(x, d["y_pred"], color=NOIR, lw=1.8, label="Prediction du modele")
    ax.scatter(x[dedans], obs[dedans], s=12, color=VERT, alpha=0.6,
               label=f"Dans l'intervalle (n={dedans.sum():,})")
    ax.scatter(x[~dedans], np.minimum(obs[~dedans], plafond), s=40, color=ROUGE,
               edgecolor="white", lw=0.5, zorder=5,
               label=f"Anomalie (n={(~dedans).sum():,})")

    ax.set_ylim(0, plafond * 1.08)
    ax.set_xlim(0, max(len(d) - 1, 1))
    ax.set_title(f"Bande conforme - strate {strate} "
                 f"(taux d'anomalie = {100*(~dedans).mean():.2f} %, "
                 f"attendu {100*alpha:.0f} %)")
    ax.set_xlabel("Unites de la strate, triees par prediction croissante")
    ax.set_ylabel(f"{target} (EUR)")
    ax.yaxis.set_major_formatter(FuncFormatter(montant))
    ax.grid(ls=":", alpha=0.5)
    ax.legend(loc="upper left")
    sauver(fig, f"fig_10_bande_strate_{strate.replace('+', 'plus')}")































def fig_11_bande_detaillee(results, id_cols, n=30, mode="echantillon",
                           alpha=0.10, target="RBNS_eop"):
    d = results.copy()
    d["_cle"] = d[list(id_cols)].astype(str).agg(" | ".join, axis=1).str.slice(0, 30)

    if mode == "anomalies":
        ano = d[~d["dans_intervalle"]].copy()
        ecart = np.where(ano["y_obs"] > ano["borne_haute"],
                         ano["y_obs"] - ano["borne_haute"],
                         ano["borne_basse"] - ano["y_obs"])
        ano = ano.assign(_e=ecart).nlargest(min(n // 2, len(ano)), "_e").drop(columns="_e")
        norm = d[d["dans_intervalle"]]
        norm = norm.sample(min(n - len(ano), len(norm)), random_state=42)
        sel = pd.concat([ano, norm])
        suffixe = " (anomalies sur-representees)"
    else:
        sel = d.sample(min(n, len(d)), random_state=42)
        suffixe = " (echantillon aleatoire)"

    sel = sel.sort_values("y_pred").reset_index(drop=True)
    x = np.arange(len(sel))
    obs = sel["y_obs"].to_numpy(float)
    lo = sel["borne_basse"].to_numpy(float)
    hi = sel["borne_haute"].to_numpy(float)
    dedans = sel["dans_intervalle"].to_numpy(bool)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.vlines(x, lo, hi, color=BLEU, lw=7, alpha=0.45,
              label=f"Intervalle conforme {100*(1-alpha):.0f} %")
    for xi, o, l, h, dd in zip(x, obs, lo, hi, dedans):
        if not dd:
            ax.plot([xi, xi], [h if o > h else l, o], color=ROUGE, lw=1.5, ls=":")

    ax.scatter(x, sel["y_pred"], marker="D", s=40, facecolor="white",
               edgecolor=NOIR, lw=1.4, zorder=5, label="Prediction")
    ax.scatter(x[dedans], obs[dedans], s=80, color=VERT, edgecolor="white",
               zorder=6, label=f"Dans l'intervalle (n={dedans.sum()})")
    ax.scatter(x[~dedans], obs[~dedans], s=110, color=ROUGE, edgecolor="white",
               zorder=7, label=f"Hors intervalle (n={(~dedans).sum()})")

    ax.set_ylim(0, np.percentile(np.concatenate([hi, obs]), 99) * 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(sel["_cle"], rotation=90, fontsize=7.5)
    ax.set_title(f"Intervalles conformes et valeurs comptabilisees - "
                 f"{len(sel)} unites{suffixe}")
    ax.set_xlabel("Unites statistiques, triees par prediction croissante")
    ax.set_ylabel(f"{target} (EUR)")
    ax.yaxis.set_major_formatter(FuncFormatter(montant))
    ax.grid(ls=":", alpha=0.5)
    ax.legend(loc="upper left")
    sauver(fig, f"fig_11_bande_detaillee_{mode}")




































def fig_12_predit_vs_reel_rangs(results, echantillon=20000):
    d = results.sample(min(echantillon, len(results)), random_state=42)
    r_obs = d["y_obs"].rank(pct=True).to_numpy(float) * 100
    r_pred = d["y_pred"].rank(pct=True).to_numpy(float) * 100
    dedans = d["dans_intervalle"].to_numpy(bool)
    rho = spearmanr(d["y_obs"], d["y_pred"]).statistic

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.fill_between([0, 100], [-10, 90], [10, 110], color=VERT_CLAIR, alpha=0.3,
                    label="Ecart de classement < 10 points")
    ax.plot([0, 100], [0, 100], ls="--", color=ORANGE, label="Classement parfait")
    ax.scatter(r_pred[dedans], r_obs[dedans], s=5, alpha=0.2, color=GRIS,
               label="Dans l'intervalle")
    ax.scatter(r_pred[~dedans], r_obs[~dedans], s=14, alpha=0.75, color=ROUGE,
               label=f"Hors intervalle (n={(~dedans).sum():,})")

    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.set_title(f"Rang predit contre rang observe (Spearman = {rho:.4f})")
    ax.set_xlabel("Rang de la valeur predite (percentile)")
    ax.set_ylabel("Rang de la valeur comptabilisee (percentile)")
    ax.grid(ls=":", alpha=0.5)
    ax.legend(loc="lower right")
    sauver(fig, "fig_12_predit_vs_reel_rangs")

































def fig_13_top_anomalies(anomalies_prio, id_cols, top_n=15):
    d = anomalies_prio.head(top_n).copy()
    d["_cle"] = d[list(id_cols)].astype(str).agg(" | ".join, axis=1).str.slice(0, 40)
    d = d.iloc[::-1].reset_index(drop=True)

    y = np.arange(len(d))
    scores = d["score_composite"].to_numpy(float)
    total = anomalies_prio["score_composite"].sum()

    fig, ax = plt.subplots(figsize=(11, max(6, 0.42 * len(d) + 2.4)))
    ax.barh(y, scores, color=plt.cm.Reds(0.35 + 0.55 * scores / scores.max()),
            edgecolor="white")
    for yi, s in zip(y, scores):
        ax.text(s * 1.01, yi, f"{montant(s)}  ({s/total:.1%})", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(d["_cle"], fontsize=9)
    ax.set_xlim(0, scores.max() * 1.30)
    ax.set_title(f"Top {top_n} anomalies prioritaires "
                 f"({scores.sum()/total:.1%} du score total, "
                 f"sur {len(anomalies_prio):,} anomalies)")
    ax.set_xlabel("Score composite de priorisation")
    ax.xaxis.set_major_formatter(FuncFormatter(montant))
    ax.grid(axis="x", ls=":", alpha=0.5)
    sauver(fig, "fig_13_top_anomalies")





















def fig_14_pareto(anomalies_prio):
    s = np.sort(anomalies_prio["score_composite"].to_numpy(float))[::-1]
    cum = np.cumsum(s) / s.sum()
    part = np.arange(1, len(s) + 1) / len(s)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.plot(part, cum, lw=2.6, color=NOIR, label="Part cumulee du score")
    ax.fill_between(part, 0, cum, color=BLEU_CLAIR, alpha=0.5)
    ax.plot([0, 1], [0, 1], ls="--", color=GRIS, label="Si toutes egales")

    for seuil, coul in zip((0.50, 0.80), (VERT, ROUGE)):
        idx = int(np.searchsorted(cum, seuil))
        if idx >= len(part):
            continue
        ax.plot([0, part[idx]], [seuil, seuil], ls=":", color=coul)
        ax.plot([part[idx], part[idx]], [0, seuil], ls=":", color=coul)
        ax.scatter([part[idx]], [seuil], s=60, color=coul, zorder=6)
        ax.annotate(f"{part[idx]:.1%} des anomalies ({idx+1:,} lignes)\n"
                    f"= {seuil:.0%} du score",
                    xy=(part[idx], seuil), xytext=(part[idx] + 0.08, seuil - 0.15),
                    fontsize=9, color=coul,
                    arrowprops=dict(arrowstyle="->", color=coul))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title("Concentration du risque : combien de lignes instruire ?")
    ax.set_xlabel("Part des anomalies traitees (triees par score)")
    ax.set_ylabel("Part cumulee du score total")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(ls=":", alpha=0.5)
    ax.legend(loc="lower right")
    sauver(fig, "fig_14_pareto")























def fig_15_decomposition_score(anomalies_prio, id_cols, gwp_col="GWP", top_n=15):
    facteurs = {"A : ecart borne": "A_ecart_borne",
                "B : erreur modele": "B_erreur_modele",
                f"{gwp_col} : exposition": gwp_col}
    facteurs = {k: v for k, v in facteurs.items() if v in anomalies_prio.columns}

    pct = pd.DataFrame({k: anomalies_prio[v].rank(pct=True) * 100
                        for k, v in facteurs.items()})
    pct["Score final"] = anomalies_prio["score_composite"].rank(pct=True) * 100
    d = pct.head(top_n).iloc[::-1]

    cles = (anomalies_prio[list(id_cols)].astype(str).agg(" | ".join, axis=1)
            .str.slice(0, 30).head(top_n).iloc[::-1])

    fig, ax = plt.subplots(figsize=(10, max(6, 0.42 * len(d) + 2.6)))
    im = ax.imshow(d.to_numpy(float), aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            v = d.iat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9.5,
                    color="white" if (v > 72 or v < 22) else NOIR, fontweight="bold")

    ax.set_xticks(range(d.shape[1]))
    ax.set_xticklabels(d.columns, fontsize=10)
    ax.set_yticks(range(d.shape[0]))
    ax.set_yticklabels(cles, fontsize=9)
    ax.set_title("Pourquoi ces lignes sont-elles prioritaires ?")
    fig.colorbar(im, ax=ax, pad=0.02, label="Rang percentile")
    sauver(fig, "fig_15_decomposition")




























def fig_16_fiche_anomalie(anomalie_row, calib_scores, id_cols, rang=1, alpha=0.10):
    cal = np.asarray(calib_scores, float)
    cal = cal[np.isfinite(cal)]
    s = float(anomalie_row["score_nonconformite"])
    borne_x = max(np.percentile(cal, 99.5), s * 1.05)
    q = np.quantile(cal, 1 - alpha)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(cal, bins=70, range=(cal.min(), borne_x), color=VERT_CLAIR,
            edgecolor="white", label=f"Ecarts sur donnees saines (n={len(cal):,})")
    ax.axvspan(q, borne_x, color=ROUGE, alpha=0.06)
    ax.axvline(q, color=ORANGE, ls="--", lw=2,
               label=f"Seuil de normalite ({100*(1-alpha):.0f}e percentile)")
    ax.axvline(s, color=ROUGE, lw=3, label="Cette observation")

    lignes = [f"Valeur comptabilisee : {montant(anomalie_row['y_obs'])}",
              f"Valeur attendue      : {montant(anomalie_row['y_pred'])}",
              f"Zone normale         : [{montant(anomalie_row['borne_basse'])} ; "
              f"{montant(anomalie_row['borne_haute'])}]"]
    if "p_value" in anomalie_row.index:
        lignes.append(f"p-value conforme     : {anomalie_row['p_value']:.2e}")
    ax.text(0.98, 0.70, "\n".join(lignes), transform=ax.transAxes, ha="right",
            va="top", ma="left", fontsize=10, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec=GRIS))

    ident = " | ".join(str(anomalie_row[c]) for c in id_cols if c in anomalie_row.index)
    ax.set_xlim(cal.min(), borne_x)
    ax.set_title(f"Anomalie #{rang} - {ident}")
    ax.set_xlabel("Score de non-conformite (EUR)")
    ax.set_ylabel("Nombre d'observations de calibration")
    ax.xaxis.set_major_formatter(FuncFormatter(montant))
    ax.grid(ls=":", alpha=0.5)
    ax.legend(loc="upper center")
    sauver(fig, f"fig_16_fiche_anomalie_{rang}")





























def fig_17_trajectoire(results, id_cols, entite=None, alpha=0.10, target="RBNS_eop"):
    d = results.copy()
    d["_cle"] = d[list(id_cols)].astype(str).agg(" | ".join, axis=1)

    if entite is None:
        entite = (d.assign(hors=~d["dans_intervalle"])
                  .groupby("_cle")["hors"].sum().idxmax())

    sous = d[d["_cle"] == entite].copy()
    ecart = np.maximum(sous["borne_basse"] - sous["y_obs"],
                       sous["y_obs"] - sous["borne_haute"])
    sous = (sous.assign(_e=ecart).sort_values("_e", ascending=False)
            .groupby("time_idx", as_index=False).head(1)
            .sort_values("time_idx"))

    x = np.arange(len(sous))
    obs = sous["y_obs"].to_numpy(float)
    dedans = sous["dans_intervalle"].to_numpy(bool)
    n_hors, n_tot = int((~dedans).sum()), len(sous)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.fill_between(x, sous["borne_basse"], sous["borne_haute"], color=BLEU_CLAIR,
                    alpha=0.85, label=f"Intervalle conforme {100*(1-alpha):.0f} %")
    ax.plot(x, sous["y_pred"], ls="--", lw=1.8, color=NOIR, marker="D", ms=6,
            markerfacecolor="white", label="Valeur attendue")
    ax.plot(x, obs, lw=2, color=GRIS, zorder=4)
    ax.scatter(x[dedans], obs[dedans], s=90, color=VERT, edgecolor="white",
               zorder=6, label="Trimestre conforme")
    ax.scatter(x[~dedans], obs[~dedans], s=130, marker="X", color=ROUGE,
               edgecolor="white", zorder=7, label="Trimestre hors intervalle")

    p_pers = binom.sf(n_hors - 1, n_tot, alpha) if n_tot >= 3 else np.nan
    titre = f"Trajectoire - {entite}  ({n_hors}/{n_tot} trimestres hors intervalle"
    titre += f", p = {p_pers:.2e})" if np.isfinite(p_pers) else ")"

    bas, haut = ax.get_ylim()
    ax.set_ylim(bas, haut + 0.25 * (haut - bas))
    ax.set_xticks(x)
    ax.set_xticklabels(sous["time_idx"], rotation=45)
    ax.set_title(titre)
    ax.set_xlabel("Periode")
    ax.set_ylabel(f"{target} (EUR)")
    ax.yaxis.set_major_formatter(FuncFormatter(montant))
    ax.grid(ls=":", alpha=0.5)
    ax.legend(loc="upper right")
    sauver(fig, "fig_17_trajectoire")








































def fig_18_persistance(results, id_cols, alpha=0.10, top_n=15):
    d = results.copy()
    d["_cle"] = d[list(id_cols)].astype(str).agg(" | ".join, axis=1)
    g = (d.assign(hors=~d["dans_intervalle"]).groupby("_cle")
         .agg(n_periodes=("hors", "size"), n_hors=("hors", "sum")).reset_index())
    g = g[(g["n_periodes"] >= 2) & (g["n_hors"] > 0)]
    g["p_pers"] = binom.sf(g["n_hors"] - 1, g["n_periodes"], alpha)
    g["taux"] = g["n_hors"] / g["n_periodes"]
    g = g.sort_values("p_pers").head(top_n).iloc[::-1].reset_index(drop=True)

    y = np.arange(len(g))
    couleurs = np.where(g["p_pers"] < 0.01, ROUGE,
                        np.where(g["p_pers"] < 0.05, ORANGE, GRIS))

    fig, ax = plt.subplots(figsize=(11, max(6, 0.42 * len(g) + 2.4)))
    ax.barh(y, g["taux"], color=couleurs, edgecolor="white")
    ax.axvline(alpha, color=NOIR, ls="--", lw=2,
               label=f"Taux attendu si normale : {alpha:.0%}")
    for yi, r in g.iterrows():
        ax.text(r["taux"] + 0.01, yi,
                f"{int(r['n_hors'])}/{int(r['n_periodes'])}   p = {r['p_pers']:.1e}",
                va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(g["_cle"].str.slice(0, 40), fontsize=9)
    ax.set_xlim(0, min(1.0, g["taux"].max() * 1.45))
    ax.set_title(f"Entites recidivistes "
                 f"({int((g['p_pers'] < 0.01).sum())} inexplicables au seuil de 1 %)")
    ax.set_xlabel("Part des trimestres hors intervalle")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.legend(loc="lower right")
    sauver(fig, "fig_18_persistance")




























ID_COLS = ["Companies", "Lob", "Risk"]

fig_01_profil_cible(results_v2)
fig_02_concentration(results_v2)
fig_03_calibration(results_pv, calib_scores)
fig_04_histogramme_pvalues(results_pv)
fig_05_qqplot_pvalues(results_pv)
fig_06_benjamini_hochberg(results_pv)
fig_07_couverture_par_largeur(results_v2)
fig_08_largeurs(results_v2)
fig_09_bande_normalisee(results_v2)
for s in ["P0-50", "P50-90", "P90-99", "P99+"]:
    fig_10_bande_strate(results_v2, s)
fig_11_bande_detaillee(results_v2, ID_COLS, mode="echantillon")
fig_11_bande_detaillee(results_v2, ID_COLS, mode="anomalies")
fig_12_predit_vs_reel_rangs(results_v2)
fig_13_top_anomalies(anomalies_prio, ID_COLS)
fig_14_pareto(anomalies_prio)
fig_15_decomposition_score(anomalies_prio, ID_COLS)
fig_16_fiche_anomalie(anomalies_prio.iloc[0], calib_scores, ID_COLS, rang=1)
fig_17_trajectoire(results_v2, ID_COLS)
fig_18_persistance(results_v2, ID_COLS)


