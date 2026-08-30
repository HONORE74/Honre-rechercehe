# ================================================================
# BLOC 0 — STYLE MEMOIRE : palette LaTeX, police serif, export PDF
# ================================================================
# ┌─────────────────── PARAMETRES A MODIFIER ───────────────────┐
DOSSIER_FIG = "figures_memoire"
FORMATS     = ["pdf", "png"]     # pdf pour LaTeX, png pour relecture
DPI         = 300
# └──────────────────────────────────────────────────────────────┘

import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
from pathlib import Path

RACINE_FIG = Path(DOSSIER_FIG); RACINE_FIG.mkdir(parents=True, exist_ok=True)

# Palette identique aux \definecolor de votre preambule
BLEU, VERT, ORANGE = "#0055A4", "#00783C", "#CC5500"
VIOLET, ROUGE      = "#502878", "#A01E1E"
GRISFOND, BLEUINTRO = "#F5F6FA", "#0C2850"
GRIS, GRIS_CLAIR   = "#6B7280", "#D1D5DB"
CYCLE = [BLEU, ORANGE, VERT, VIOLET, ROUGE]

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#4B5563", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRIS_CLAIR,
    "grid.linewidth": 0.5, "grid.alpha": 0.7,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.prop_cycle": mpl.cycler(color=CYCLE)})

EUR = FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " "))
PCT = PercentFormatter(xmax=1, decimals=0)


def sauver(fig, nom):
    """Enregistre la figure dans tous les formats demandes."""
    for ext in FORMATS:
        fig.savefig(RACINE_FIG / f"{nom}.{ext}", dpi=DPI)
    print(f"  {nom}  ->  {', '.join(FORMATS)}")
    plt.close(fig)


print(f"Figures : {RACINE_FIG.resolve()}")















3.1


# Proportion cumulee en abscisse, valeur en ordonnee, echelle LINEAIRE.
# La concavite de fin de courbe est la signature de la queue lourde.

y = np.sort(df[TARGET].dropna().values.astype(float))
p = np.arange(1, len(y) + 1) / len(y)

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(p, y, color=BLEU, lw=1.8)
ax.fill_between(p, 0, y, color=BLEU, alpha=.08)

for q, coul in [(.50, VERT), (.90, ORANGE), (.99, ROUGE)]:
    v = np.quantile(y, q)
    ax.axhline(v, color=coul, ls="--", lw=1, alpha=.9)
    ax.annotate(f"P{int(100*q)} = {v:,.0f} €".replace(",", " "),
                xy=(.02, v), xytext=(.02, v), va="bottom", ha="left",
                fontsize=8.5, color=coul)

ax.set_xlabel("Proportion cumulée des observations")
ax.set_ylabel(f"{TARGET} (€)")
ax.set_title("Fonction de répartition empirique de la variable cible")
ax.yaxis.set_major_formatter(EUR); ax.xaxis.set_major_formatter(PCT)
ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
sauver(fig, "fig_3_1_repartition_cible")

print(f"Rapport P99/P50 : {np.quantile(y,.99)/max(np.quantile(y,.50),1):,.0f}x")
print(f"Part du top 1 % dans la somme : {y[int(.99*len(y)):].sum()/y.sum():.1%}")


















3.2


# ┌─── PARAMETRES ───┐
GROUPE_MV = ID_COLS[0]     # dimension de groupement
N_MIN_MV  = 20             # effectif minimal par groupe
# └──────────────────┘
# Les axes portent log(moyenne) et log(variance) : ce sont des VARIABLES
# transformees sur des axes LINEAIRES, non des echelles logarithmiques.
# La pente de l'ajustement estime directement p de Var = phi * E[Y]^p.

from scipy import stats

g = (df[df[TARGET].notna()].groupby(GROUPE_MV, observed=True)[TARGET]
       .agg(mu="mean", var="var", n="size").reset_index())
g = g[(g.n >= N_MIN_MV) & (g.mu > 0) & (g["var"] > 0)]
lm, lv = np.log(g.mu.values), np.log(g["var"].values)
pente, ord0, r, pv, err = stats.linregress(lm, lv)

fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.scatter(lm, lv, s=12 + 60*g.n/g.n.max(), color=BLEU, alpha=.55,
           edgecolor="white", lw=.6, label=f"Groupes ({len(g)})")
xl = np.linspace(lm.min(), lm.max(), 100)
ax.plot(xl, ord0 + pente*xl, color=ROUGE, lw=2,
        label=f"Ajustement — pente $p$ = {pente:.3f} ± {err:.3f}")
for pv_ref, nom, coul in [(1, "Poisson", VERT), (2, "Gamma", VIOLET)]:
    ax.plot(xl, np.mean(lv - pv_ref*lm) + pv_ref*xl, ls=":", lw=1.2,
            color=coul, label=f"$p$ = {pv_ref} ({nom})")

ax.set_xlabel(r"$\log$(moyenne du groupe)")
ax.set_ylabel(r"$\log$(variance du groupe)")
ax.set_title(f"Relation moyenne-variance par {GROUPE_MV}"
             f"\n$R^2$ = {r**2:.3f}")
ax.legend(frameon=False, loc="upper left")
sauver(fig, "fig_3_2_moyenne_variance")

print(f"p empirique = {pente:.4f}  (IC 95 % : {pente-1.96*err:.3f} – {pente+1.96*err:.3f})")
print(f"-> {'Poisson-Gamma composee (1<p<2)' if 1 < pente < 2 else 'HORS intervalle usuel'}")




























3.3

# ┌─── PARAMETRES ───┐
N_INIT_SCHEMA = 8          # trimestres d'apprentissage initiaux
# └──────────────────┘

periodes = sorted(df[TIME_COL].unique())
libelles = [f"T{i+1}" for i in range(len(periodes))]
iterations = list(range(N_INIT_SCHEMA, len(periodes)))

fig, ax = plt.subplots(figsize=(7.2, .42*len(iterations) + 1.6))
for k, fin in enumerate(iterations):
    ax.barh(k, fin, left=0, height=.62, color=BLEU, alpha=.75,
            edgecolor="white", lw=.8)
    ax.barh(k, 1, left=fin, height=.62, color=ORANGE,
            edgecolor="white", lw=.8)
    ax.text(fin + .5, k, "prédit", ha="center", va="center",
            fontsize=7.5, color="white", fontweight="bold")

ax.set_yticks(range(len(iterations)))
ax.set_yticklabels([f"Itération {i+1}" for i in range(len(iterations))])
ax.set_xticks(np.arange(len(periodes)) + .5)
ax.set_xticklabels(libelles, fontsize=8)
ax.set_xlim(0, len(periodes)); ax.invert_yaxis()
ax.set_xlabel("Trimestres")
ax.set_title("Protocole d'évaluation à fenêtre extensible")
ax.grid(axis="y", visible=False)
ax.legend(handles=[mpl.patches.Patch(color=BLEU, alpha=.75, label="Apprentissage"),
                   mpl.patches.Patch(color=ORANGE, label="Période prédite")],
          frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.14))
sauver(fig, "fig_3_3_protocole_fenetre")

























vals = [t.value for t in etude.trials if t.value is not None]
best = np.minimum.accumulate(vals)

fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.scatter(range(1, len(vals)+1), vals, s=14, color=GRIS, alpha=.55,
           edgecolor="none", label="Essais")
ax.plot(range(1, len(best)+1), best, color=ROUGE, lw=2,
        label="Meilleure valeur atteinte")
i_best = int(np.argmin(vals))
ax.scatter([i_best+1], [vals[i_best]], s=80, marker="*", color=VERT,
           zorder=5, label=f"Optimum — essai {i_best+1}")
ax.annotate(f"MAE = {vals[i_best]:,.0f}".replace(",", " "),
            xy=(i_best+1, vals[i_best]), xytext=(10, 14),
            textcoords="offset points", fontsize=8.5, color=VERT)

ax.set_xlabel("Numéro d'essai"); ax.set_ylabel("MAE en validation croisée (€)")
ax.set_title("Convergence de l'optimisation bayésienne")
ax.yaxis.set_major_formatter(EUR)
ax.legend(frameon=False)
sauver(fig, "fig_3_4_convergence_optuna")

print(f"{len(vals)} essais | MAE initiale {vals[0]:,.0f} -> optimale {min(vals):,.0f}"
      f"  ({100*(1-min(vals)/vals[0]):.1f} % de gain)")































import optuna
imp = optuna.importance.get_param_importances(etude)
noms, vals_i = list(imp.keys())[::-1], list(imp.values())[::-1]

fig, ax = plt.subplots(figsize=(6.4, .34*len(noms) + 1.4))
ax.barh(noms, vals_i, color=[BLEU if v >= .1 else GRIS_CLAIR for v in vals_i],
        edgecolor="white", lw=.8)
for i, v in enumerate(vals_i):
    ax.text(v + .008, i, f"{v:.1%}", va="center", fontsize=8.5, color=GRIS)

ax.set_xlabel("Contribution à la variance de la performance")
ax.set_title("Importance relative des hyperparamètres")
ax.xaxis.set_major_formatter(PCT)
ax.grid(axis="y", visible=False)
sauver(fig, "fig_3_5_importance_hyperparametres")

dom = [n for n, v in zip(noms[::-1], vals_i[::-1]) if v >= .1]
print(f"Hyperparamètres déterminants (>= 10 %) : {dom}")





























# ┌─── PARAMETRES ───┐
MODELES_LORENZ = {"Référence naïve (médiane)": np.full(len(y_te), np.median(y_te)),
                  "Avant tuning":  pred_avant["Test"],
                  "Après tuning":  pred_apres["Test"]}
# └──────────────────┘

yt = np.asarray(y_te, float)

def _gini(yv, yp):
    o = np.argsort(yp); c = np.cumsum(yv[o])/yv.sum()
    gm = 1 - 2*np.trapezoid(c, dx=1/len(c))
    o2 = np.argsort(yv); c2 = np.cumsum(yv[o2])/yv.sum()
    return gm / (1 - 2*np.trapezoid(c2, dx=1/len(c2)))

fig, ax = plt.subplots(figsize=(6.0, 5.2))
ax.plot([0, 1], [0, 1], color=GRIS, ls="--", lw=1, label="Absence de discrimination")
for (nom, pr), coul in zip(MODELES_LORENZ.items(), [GRIS, ORANGE, BLEU]):
    o = np.argsort(pr); c = np.cumsum(yt[o])/yt.sum()
    ax.plot(np.linspace(0, 1, len(c)), c, color=coul, lw=2,
            label=f"{nom} — Gini = {_gini(yt, pr):.3f}")
o = np.argsort(yt); c = np.cumsum(yt[o])/yt.sum()
ax.plot(np.linspace(0, 1, len(c)), c, color=VERT, lw=1.4, ls=":",
        label="Tri parfait")

ax.set_xlabel("Proportion cumulée des observations (triées par prédiction)")
ax.set_ylabel("Proportion cumulée de la provision observée")
ax.set_title("Courbe de concentration des modèles comparés")
ax.xaxis.set_major_formatter(PCT); ax.yaxis.set_major_formatter(PCT)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(frameon=False, loc="upper left", fontsize=8.5)
sauver(fig, "fig_3_6_concentration_modeles")





















# ┌─── PARAMETRES ───┐
N_VARS_FIG = 15
# └──────────────────┘

imp_v = pd.Series(modele_apres.booster_.feature_importance("gain"),
                  index=modele_apres.feature_name_)
imp_v = (100*imp_v/imp_v.sum()).sort_values(ascending=False).head(N_VARS_FIG).iloc[::-1]

fig, ax = plt.subplots(figsize=(6.4, .32*len(imp_v) + 1.3))
ax.barh(imp_v.index, imp_v.values, color=BLEU, alpha=.85,
        edgecolor="white", lw=.8)
for i, v in enumerate(imp_v.values):
    ax.text(v + .4, i, f"{v:.1f}", va="center", fontsize=8.5, color=GRIS)

ax.set_xlabel("Part du gain total (%)")
ax.set_title("Contribution des variables explicatives")
ax.grid(axis="y", visible=False)
sauver(fig, "fig_3_7_importance_variables")

print(f"Les {min(3,len(imp_v))} premières concentrent "
      f"{imp_v.iloc[::-1].head(3).sum():.1f} % du gain")

























# ┌─── PARAMETRES ───┐
N_VARS_FIG = 15
# └──────────────────┘

imp_v = pd.Series(modele_apres.booster_.feature_importance("gain"),
                  index=modele_apres.feature_name_)
imp_v = (100*imp_v/imp_v.sum()).sort_values(ascending=False).head(N_VARS_FIG).iloc[::-1]

fig, ax = plt.subplots(figsize=(6.4, .32*len(imp_v) + 1.3))
ax.barh(imp_v.index, imp_v.values, color=BLEU, alpha=.85,
        edgecolor="white", lw=.8)
for i, v in enumerate(imp_v.values):
    ax.text(v + .4, i, f"{v:.1f}", va="center", fontsize=8.5, color=GRIS)

ax.set_xlabel("Part du gain total (%)")
ax.set_title("Contribution des variables explicatives")
ax.grid(axis="y", visible=False)
sauver(fig, "fig_3_7_importance_variables")

print(f"Les {min(3,len(imp_v))} premières concentrent "
      f"{imp_v.iloc[::-1].head(3).sum():.1f} % du gain")



























# FIGURE CHARNIERE DU MEMOIRE : etablit empiriquement l'heteroscedasticite
# qui justifie le recours a des intervalles de largeur variable.

res = pd.DataFrame({"pred": pred_apres["Test"], "res": yt - pred_apres["Test"]})
res["dec"] = pd.qcut(res.pred.rank(method="first"), 10, labels=range(1, 11))
g = res.groupby("dec", observed=True).agg(
    sd=("res", "std"),
    q10=("res", lambda s: s.quantile(.10)),
    q90=("res", lambda s: s.quantile(.90))).reset_index()
g["inter"] = g.q90 - g.q10

fig, ax = plt.subplots(figsize=(6.6, 4.3))
ax.bar(g.dec.astype(int), g.sd, color=BLEU, alpha=.80,
       edgecolor="white", lw=.8, label="Écart-type des résidus")
ax.set_xlabel("Décile de valeur prédite")
ax.set_ylabel("Écart-type des résidus (€)", color=BLEU)
ax.tick_params(axis="y", colors=BLEU)
ax.yaxis.set_major_formatter(EUR)

ax2 = ax.twinx()
ax2.plot(g.dec.astype(int), g.inter, color=ORANGE, lw=2.2,
         marker="o", ms=6, mfc="white", mew=1.8, label="Étendue interdécile")
ax2.set_ylabel("Étendue interdécile des résidus (€)", color=ORANGE)
ax2.tick_params(axis="y", colors=ORANGE)
ax2.yaxis.set_major_formatter(EUR)
ax2.grid(False)

rap = g.sd.iloc[-1] / max(g.sd.iloc[0], 1e-9)
ax.set_title("Dispersion des résidus par décile de prédiction"
             f"\nRapport dernier / premier décile : {rap:.0f}×")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, frameon=False, loc="upper left", fontsize=8.5)
sauver(fig, "fig_3_8_dispersion_residus")

print(f"Écart-type décile 1  : {g.sd.iloc[0]:,.0f} €")
print(f"Écart-type décile 10 : {g.sd.iloc[-1]:,.0f} €")
print(f"RAPPORT : {rap:.1f}x   <- valeur a reporter dans le texte du memoire")




























per = sorted(df[TIME_COL].unique())
n_calib, n_test = N_CALIB, N_TEST
bornes = [0, len(per)-n_calib-n_test, len(per)-n_test, len(per)]
noms   = ["Apprentissage", "Calibration", "Test"]
coul   = [BLEU, VIOLET, ORANGE]
eff    = [int((df[TIME_COL] >= per[bornes[i]]).sum() -
              (df[TIME_COL] >= per[min(bornes[i+1], len(per)-1)]).sum()
              if i < 2 else (df[TIME_COL] >= per[bornes[i]]).sum())
          for i in range(3)]

fig, ax = plt.subplots(figsize=(7.2, 2.5))
for i, (d, f, nom, c) in enumerate(zip(bornes[:-1], bornes[1:], noms, coul)):
    ax.barh(0, f-d, left=d, height=.55, color=c, alpha=.85,
            edgecolor="white", lw=1.2)
    ax.text((d+f)/2, 0, nom, ha="center", va="center",
            color="white", fontweight="bold", fontsize=10)
    ax.text((d+f)/2, -.46, f"T{d+1} – T{f}\n{f-d} trimestre(s)",
            ha="center", va="top", fontsize=8, color=c)

ax.set_xlim(0, len(per)); ax.set_ylim(-1.1, .5)
ax.set_yticks([]); ax.set_xticks(np.arange(len(per)) + .5)
ax.set_xticklabels([f"T{i+1}" for i in range(len(per))], fontsize=8)
ax.set_xlabel("Trimestres, par ordre chronologique")
ax.set_title("Organisation temporelle des trois ensembles")
ax.grid(visible=False)
for s in ax.spines.values():
    s.set_visible(False)
sauver(fig, "fig_4_1_decoupage_temporel")


























s = np.sort(np.asarray(calib_scores, float))
p = np.arange(1, len(s)+1)/len(s)

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(p, s, color=BLEU, lw=1.8)
ax.fill_between(p, 0, s, color=BLEU, alpha=.08)

for niv, coul in [(.90, VERT), (.95, ORANGE), (.99, ROUGE)]:
    n = len(s)
    ordre = min(np.ceil((n+1)*niv)/n, 1.0)
    q = np.quantile(s, ordre, method="higher")
    ax.axhline(q, color=coul, ls="--", lw=1.1)
    ax.annotate(f"{int(100*niv)} % → $\\hat{{q}}$ = {q:,.0f} €".replace(",", " "),
                xy=(.02, q), va="bottom", ha="left", fontsize=8.5, color=coul)

ax.set_xlabel("Proportion cumulée des observations de calibration")
ax.set_ylabel("Score de non-conformité (€)")
ax.set_title(f"Distribution des scores de non-conformité (n = {len(s):,})"
             .replace(",", " "))
ax.xaxis.set_major_formatter(PCT); ax.yaxis.set_major_formatter(EUR)
ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
sauver(fig, "fig_4_2_scores_nonconformite")


















# FIGURE CHARNIERE DU CHAPITRE : vue NORMALISEE, donc comparable entre
# regimes sans recours a une echelle logarithmique.
# ┌─── PARAMETRES ───┐
N_UNITES_43 = 22
# └──────────────────┘

d = results_v2.dropna(subset=["y_obs", "y_pred", "borne_basse", "borne_haute"]).copy()
d["dec"] = pd.qcut(d.y_pred.rank(method="first"), 10, labels=False)
centre = (d.borne_haute + d.borne_basse)/2
demi   = np.maximum((d.borne_haute - d.borne_basse)/2, 1e-9)
d["z"] = (d.y_obs - centre)/demi
if "Q_hat" in globals():
    d["z_split"] = (d.y_obs - d.y_pred)/max(float(Q_hat), 1e-9)
else:
    d["z_split"] = (d.y_obs - d.y_pred)/max(np.quantile(np.abs(d.y_obs-d.y_pred), .90), 1e-9)

fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.4), sharey=True)
for ax, dec, titre in zip(axes, [0, 9],
                          ["Premier décile de prédiction",
                           "Dernier décile de prédiction"]):
    sub = d[d.dec == dec].head(N_UNITES_43).reset_index(drop=True)
    x = np.arange(len(sub))
    ax.axhspan(-1, 1, color=BLEU, alpha=.13, label="Enveloppe Split CP")
    ax.axhline(1, color=BLEU, lw=1.2); ax.axhline(-1, color=BLEU, lw=1.2)
    ax.axhline(0, color=GRIS, lw=.9, ls="--")
    dedans = sub.dans_intervalle.values.astype(bool)
    ax.vlines(x[~dedans], np.sign(sub.z_split.values[~dedans]),
              sub.z_split.values[~dedans], color=ROUGE, lw=1.2, alpha=.6)
    ax.scatter(x[dedans], sub.z_split.values[dedans], s=34, color=VERT,
               edgecolor="white", lw=.8, zorder=4, label="Couverte")
    ax.scatter(x[~dedans], sub.z_split.values[~dedans], s=44, color=ROUGE,
               edgecolor="white", lw=.8, zorder=5, label="Hors intervalle")
    ax.set_title(titre, fontsize=10)
    ax.set_xlabel("Sous-portefeuilles")
    ax.set_xticks([])

axes[0].set_ylabel("Écart rapporté à la demi-largeur de l'enveloppe")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("Enveloppe de largeur constante appliquée à deux régimes",
             fontsize=11, y=1.0)
sauver(fig, "fig_4_3_enveloppe_deux_regimes")






















larg_cqr = (results_v2.borne_haute - results_v2.borne_basse).dropna().values
larg_split = 2*float(Q_hat) if "Q_hat" in globals() else np.median(larg_cqr)

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.hist(larg_cqr, bins=60, color=BLEU, alpha=.72, edgecolor="white", lw=.4,
        label=f"CQR — largeur variable (médiane {np.median(larg_cqr):,.0f} €)"
              .replace(",", " "))
ax.axvline(larg_split, color=ROUGE, lw=2.4,
           label=f"Split CP — largeur constante ({larg_split:,.0f} €)"
                 .replace(",", " "))
ax.axvline(np.median(larg_cqr), color=BLEU, ls="--", lw=1.4)

ax.set_xlabel("Largeur de l'intervalle (€)")
ax.set_ylabel("Nombre de sous-portefeuilles")
ax.set_title("Distribution des largeurs d'intervalle selon la construction")
ax.xaxis.set_major_formatter(EUR)
ax.legend(frameon=False, fontsize=8.5)
sauver(fig, "fig_4_4_distribution_largeurs")

print(f"CQR   : min {larg_cqr.min():,.0f} | médiane {np.median(larg_cqr):,.0f} "
      f"| max {larg_cqr.max():,.0f}")
print(f"Split : {larg_split:,.0f} (constante)")
print(f"Rapport max/min CQR : {larg_cqr.max()/max(larg_cqr.min(),1):.0f}x")

























sc = anomalies_prio.score_composite.sort_values(ascending=False).values
cum = np.cumsum(sc)/sc.sum()
n_pct = np.arange(1, len(sc)+1)/len(sc)

fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.plot(n_pct, cum, color=ROUGE, lw=2.2)
ax.fill_between(n_pct, 0, cum, color=ROUGE, alpha=.10)
ax.plot([0, 1], [0, 1], color=GRIS, ls="--", lw=1,
        label="Répartition uniforme")

i80 = int(np.searchsorted(cum, .80))
if i80 < len(n_pct):
    x80 = n_pct[i80]
    ax.axhline(.80, color=BLEU, ls=":", lw=1.2)
    ax.axvline(x80, color=BLEU, ls=":", lw=1.2)
    ax.scatter([x80], [.80], s=70, color=BLEU, zorder=5)
    ax.annotate(f"{100*x80:.0f} % des signaux\nportent 80 % du score",
                xy=(x80, .80), xytext=(x80+.08, .52),
                fontsize=9, color=BLEU,
                arrowprops=dict(arrowstyle="->", color=BLEU, lw=1))

ax.set_xlabel("Proportion des signaux (triés par score décroissant)")
ax.set_ylabel("Proportion cumulée du score total")
ax.set_title(f"Concentration du score de priorisation ({len(sc)} signaux)")
ax.xaxis.set_major_formatter(PCT); ax.yaxis.set_major_formatter(PCT)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.legend(frameon=False, loc="lower right")
sauver(fig, "fig_4_5_pareto_priorisation")

print(f"80 % du score sur {100*n_pct[i80]:.1f} % des signaux "
      f"({i80+1} sous-portefeuilles)")












# ┌─── PARAMETRES ───┐
DIM_COUV = ID_COLS[0]
N_MIN_COUV, N_MAX_COUV = 20, 12
# └──────────────────┘

d = results_v2.copy(); d[DIM_COUV] = d[DIM_COUV].astype(str)
g = (d.groupby(DIM_COUV, observed=True)
       .agg(n=("dans_intervalle", "size"), couv=("dans_intervalle", "mean"))
       .reset_index())
g = g[g.n >= N_MIN_COUV].nlargest(N_MAX_COUV, "n").sort_values("couv")

fig, ax = plt.subplots(figsize=(6.6, .34*len(g) + 1.6))
coul = [VERT if c >= 1-ALPHA else ROUGE for c in g.couv]
ax.barh(g[DIM_COUV], g.couv, color=coul, alpha=.85, edgecolor="white", lw=.8)
ax.axvline(1-ALPHA, color="black", ls="--", lw=1.6,
           label=f"Niveau visé ({100*(1-ALPHA):.0f} %)")
for i, (c, n) in enumerate(zip(g.couv, g.n)):
    ax.text(c + .008, i, f"{100*c:.1f} %  (n={n})", va="center",
            fontsize=8, color=GRIS)

ax.set_xlabel("Couverture empirique")
ax.set_title(f"Couverture empirique par {DIM_COUV}")
ax.xaxis.set_major_formatter(PCT)
ax.set_xlim(0, 1.12)
ax.grid(axis="y", visible=False)
ax.legend(frameon=False, loc="lower right")
sauver(fig, "fig_5_1_couverture_sousgroupe")

sous = g[g.couv < 1-ALPHA]
print(f"Couverture globale : {100*d.dans_intervalle.mean():.2f} %")
print(f"Groupes sous le niveau visé : {len(sous)} / {len(g)}")
if len(sous):
    print(sous[[DIM_COUV, "n", "couv"]].to_string(index=False))


























d = results_v2.dropna(subset=["y_pred", "borne_basse", "borne_haute"]).copy()
d["larg"] = d.borne_haute - d.borne_basse
d["dec"]  = pd.qcut(d.y_pred.rank(method="first"), 10, labels=range(1, 11))
g = d.groupby("dec", observed=True).larg.mean().reset_index()
larg_split = 2*float(Q_hat) if "Q_hat" in globals() else d.larg.median()

fig, ax = plt.subplots(figsize=(6.6, 4.3))
ax.axhline(larg_split, color=ROUGE, lw=2.2,
           label=f"Split CP — {larg_split:,.0f} € (constante)".replace(",", " "))
ax.plot(g.dec.astype(int), g.larg, color=BLEU, lw=2.4, marker="o", ms=7,
        mfc="white", mew=1.8, label="CQR — largeur adaptative")
ax.fill_between(g.dec.astype(int), larg_split, g.larg,
                where=g.larg <= larg_split, color=VERT, alpha=.16,
                label="CQR plus étroite")
ax.fill_between(g.dec.astype(int), larg_split, g.larg,
                where=g.larg > larg_split, color=ORANGE, alpha=.16,
                label="CQR plus large")

ax.set_xlabel("Décile de valeur prédite")
ax.set_ylabel("Largeur moyenne de l'intervalle (€)")
ax.set_title("Adaptativité de la largeur selon le contexte")
ax.yaxis.set_major_formatter(EUR)
ax.set_xticks(range(1, 11))
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
sauver(fig, "fig_5_2_largeur_par_decile")

print(f"CQR décile 1 : {g.larg.iloc[0]:,.0f} € "
      f"({100*g.larg.iloc[0]/larg_split:.0f} % de Split CP)")
print(f"CQR décile 10 : {g.larg.iloc[-1]:,.0f} € "
      f"({100*g.larg.iloc[-1]/larg_split:.0f} % de Split CP)")






















d = results_v2.dropna(subset=["y_pred", "borne_basse", "borne_haute"]).copy()
d["larg"] = d.borne_haute - d.borne_basse
d["dec"]  = pd.qcut(d.y_pred.rank(method="first"), 10, labels=range(1, 11))
g = d.groupby("dec", observed=True).larg.mean().reset_index()
larg_split = 2*float(Q_hat) if "Q_hat" in globals() else d.larg.median()

fig, ax = plt.subplots(figsize=(6.6, 4.3))
ax.axhline(larg_split, color=ROUGE, lw=2.2,
           label=f"Split CP — {larg_split:,.0f} € (constante)".replace(",", " "))
ax.plot(g.dec.astype(int), g.larg, color=BLEU, lw=2.4, marker="o", ms=7,
        mfc="white", mew=1.8, label="CQR — largeur adaptative")
ax.fill_between(g.dec.astype(int), larg_split, g.larg,
                where=g.larg <= larg_split, color=VERT, alpha=.16,
                label="CQR plus étroite")
ax.fill_between(g.dec.astype(int), larg_split, g.larg,
                where=g.larg > larg_split, color=ORANGE, alpha=.16,
                label="CQR plus large")

ax.set_xlabel("Décile de valeur prédite")
ax.set_ylabel("Largeur moyenne de l'intervalle (€)")
ax.set_title("Adaptativité de la largeur selon le contexte")
ax.yaxis.set_major_formatter(EUR)
ax.set_xticks(range(1, 11))
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
sauver(fig, "fig_5_2_largeur_par_decile")

print(f"CQR décile 1 : {g.larg.iloc[0]:,.0f} € "
      f"({100*g.larg.iloc[0]/larg_split:.0f} % de Split CP)")
print(f"CQR décile 10 : {g.larg.iloc[-1]:,.0f} € "
      f"({100*g.larg.iloc[-1]/larg_split:.0f} % de Split CP)")




























# ┌─── PARAMETRES A COMPLETER ───┐
# Renseignez vos mesures reelles pour les six configurations.
CONFIGS = [
    # (construction, niveau, couverture empirique, largeur moyenne)
    ("Split CP", 0.90, 0.905, 2*float(Q_hat) if "Q_hat" in globals() else 1e5),
    ("Split CP", 0.95, 0.952, 2.6*float(Q_hat) if "Q_hat" in globals() else 1.3e5),
    ("Split CP", 0.99, 0.991, 4.1*float(Q_hat) if "Q_hat" in globals() else 2.1e5),
    ("CQR",      0.90, float(results_v2.dans_intervalle.mean()),
                 float((results_v2.borne_haute-results_v2.borne_basse).mean())),
    ("CQR",      0.95, np.nan, np.nan),   # a completer
    ("CQR",      0.99, np.nan, np.nan),   # a completer
]
# └──────────────────────────────┘

cfg = pd.DataFrame(CONFIGS, columns=["constr", "niveau", "couv", "larg"]).dropna()

fig, ax = plt.subplots(figsize=(6.4, 4.6))
for constr, coul, mk in [("Split CP", ROUGE, "s"), ("CQR", BLEU, "o")]:
    s = cfg[cfg.constr == constr]
    ax.plot(s.larg, s.couv, color=coul, lw=1.4, ls="--", alpha=.6)
    ax.scatter(s.larg, s.couv, s=110, color=coul, marker=mk,
               edgecolor="white", lw=1.4, zorder=5, label=constr)
    for _, r in s.iterrows():
        ax.annotate(f"{100*r.niveau:.0f} %", xy=(r.larg, r.couv),
                    xytext=(7, 6), textcoords="offset points",
                    fontsize=8.5, color=coul)

for niv, coul in [(.90, VERT), (.95, ORANGE), (.99, VIOLET)]:
    ax.axhline(niv, color=coul, ls=":", lw=1, alpha=.7)
    ax.annotate(f"visé {int(100*niv)} %", xy=(ax.get_xlim()[1], niv),
                xytext=(-4, 3), textcoords="offset points",
                ha="right", fontsize=7.5, color=coul)

ax.set_xlabel("Largeur moyenne de l'intervalle (€)")
ax.set_ylabel("Couverture empirique")
ax.set_title("Compromis entre couverture et précision"
             "\nUne configuration en haut à gauche domine")
ax.xaxis.set_major_formatter(EUR); ax.yaxis.set_major_formatter(PCT)
ax.legend(frameon=False, loc="lower right")
sauver(fig, "fig_5_3_compromis_couverture_largeur")



























# ================================================================
# BLOC C1 — GLM vs XGBoost vs LightGBM vs reference naive
#   -> alimente tab:comparaison-modeles et la decomposition par strate
# ================================================================
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

p_tw = float(params_finaux.get("tweedie_variance_power", 1.5))
cat = [c for c in X_tr.columns if str(X_tr[c].dtype) == "category"]
num = [c for c in X_tr.columns if c not in cat]
predictions, temps = {}, {}

# --- 1. Reference naive : mediane du train ---
predictions["Référence naïve"] = np.full(len(X_te), float(np.median(y_tr)))
temps["Référence naïve"] = 0.0

# --- 2. GLM Tweedie ---
try:
    t0 = time.time()
    prep = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=0.01, sparse_output=False), cat)])
    glm = Pipeline([("prep", prep),
                    ("mdl", TweedieRegressor(power=p_tw, alpha=1e-3,
                                             link="log", max_iter=2000))])
    glm.fit(X_tr.assign(**{c: X_tr[c].astype(str) for c in cat}), y_tr)
    predictions["GLM Tweedie"] = np.clip(
        glm.predict(X_te.assign(**{c: X_te[c].astype(str) for c in cat})), 0, None)
    temps["GLM Tweedie"] = time.time() - t0
except Exception as e:
    print(f"GLM ecarte : {str(e)[:70]}")

# --- 3. XGBoost Tweedie ---
try:
    import xgboost as xgb
    t0 = time.time()
    xgm = xgb.XGBRegressor(objective="reg:tweedie", tweedie_variance_power=p_tw,
                           n_estimators=1500, learning_rate=0.03, max_depth=6,
                           subsample=0.9, colsample_bytree=0.85, reg_lambda=1.0,
                           enable_categorical=True, tree_method="hist",
                           random_state=SEED, early_stopping_rounds=100)
    xgm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    predictions["XGBoost"] = np.clip(xgm.predict(X_te), 0, None)
    temps["XGBoost"] = time.time() - t0
except Exception as e:
    print(f"XGBoost ecarte : {str(e)[:70]}")

# --- 4. LightGBM retenu ---
predictions["LightGBM"] = pred_apres["Test"]
temps["LightGBM"] = np.nan

# --- Tableau de comparaison ---
mae_naif = mean_absolute_error(yt, predictions["Référence naïve"])
lignes = []
for nom, pr in predictions.items():
    pr = np.asarray(pr, float)
    lignes.append({
        "Modèle": nom,
        "MAE": mean_absolute_error(yt, pr),
        "Gain / naïf (%)": 100*(1 - mean_absolute_error(yt, pr)/mae_naif),
        "RMSE": float(np.sqrt(mean_squared_error(yt, pr))),
        "Bilan de masse": pr.sum()/yt.sum(),
        "Gini": _gini(yt, pr),
        "Temps (s)": temps.get(nom, np.nan)})
comparaison_modeles = pd.DataFrame(lignes)

print("\n" + "="*100)
print("TABLEAU 3.x — COMPARAISON DES MODELES SUR L'ENSEMBLE DE TEST")
print("="*100)
print(comparaison_modeles.to_string(index=False,
      float_format=lambda v: f"{v:,.4f}"))

meilleur = comparaison_modeles.loc[comparaison_modeles.MAE.idxmin(), "Modèle"]
meilleur_gini = comparaison_modeles.loc[comparaison_modeles.Gini.idxmax(), "Modèle"]
print(f"\nMeilleure MAE  : {meilleur}")
print(f"Meilleur Gini  : {meilleur_gini}")
print(f"-> {'Un seul modele domine sur les deux criteres.' if meilleur == meilleur_gini else 'DOMINANCE PARTIELLE : a discuter dans le memoire.'}")

# --- Decomposition par strate ---
qs = [0, .50, .90, .99, 1.0]
b = [np.quantile(yt, q) for q in qs]; b[0], b[-1] = -np.inf, np.inf
lib = ["P0-50", "P50-90", "P90-99", "P99+"]
st = pd.cut(yt, bins=b, labels=lib, duplicates="drop")

lignes = []
for nom, pr in predictions.items():
    pr = np.asarray(pr, float)
    for lab in lib:
        m = (st == lab).to_numpy()
        if not m.sum(): continue
        lignes.append({"Modèle": nom, "Strate": lab, "n": int(m.sum()),
                       "MAE": mean_absolute_error(yt[m], pr[m]),
                       "Biais (%)": 100*(pr[m].sum()-yt[m].sum())/max(yt[m].sum(),1e-9),
                       "Bilan": pr[m].sum()/max(yt[m].sum(),1e-9),
                       "Part du total (%)": 100*yt[m].sum()/yt.sum()})
strates_modeles = pd.DataFrame(lignes)

print("\n" + "="*100)
print("TABLEAU 3.x — DECOMPOSITION PAR STRATE DE MONTANT")
print("="*100)
print(strates_modeles.pivot_table(index="Strate", columns="Modèle",
      values="MAE", observed=True).reindex(lib)
      .to_string(float_format=lambda v: f"{v:,.0f}"))

comparaison_modeles.to_csv(RACINE_FIG/"tab_3_comparaison_modeles.csv", index=False)
strates_modeles.to_csv(RACINE_FIG/"tab_3_strates_modeles.csv", index=False)
























# ================================================================
# BLOC C2 — SPLIT CP : score = residu absolu, largeur constante
#   -> alimente le tableau des quantiles conformes (chapitre 4)
# ================================================================

pred_cal  = np.clip(modele_apres.predict(X_cal), 0, None)
pred_test = np.clip(modele_apres.predict(X_te), 0, None)
scores_split = np.abs(np.asarray(y_cal, float) - pred_cal)

split_cp, lignes = {}, []
for a in [1-n for n in NIVEAUX]:
    q = _quantile_conforme(scores_split, a)
    lo = np.clip(pred_test - q, 0, None)          # contrainte de positivite
    hi = pred_test + q
    dedans = (yt >= lo) & (yt <= hi)
    split_cp[round(1-a, 2)] = dict(q=q, lo=lo, hi=hi, dedans=dedans,
                                   larg=hi-lo, pred=pred_test)
    lignes.append({"Niveau": f"{100*(1-a):.0f} %",
                   "Ordre du quantile": min(np.ceil((len(scores_split)+1)*(1-a))
                                            / len(scores_split), 1.0),
                   "Quantile conforme (€)": q,
                   "Largeur (€)": 2*q,
                   "Bornes tronquees a 0 (%)": 100*np.mean(pred_test - q < 0),
                   "Couverture empirique": dedans.mean()})

table_split = pd.DataFrame(lignes)
print("="*100)
print("TABLEAU 4.x — SPLIT CONFORMAL PREDICTION")
print("="*100)
print(table_split.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
table_split.to_csv(RACINE_FIG/"tab_4_split_cp.csv", index=False)




























Nouveau Bloc





# ================================================================
# BLOC G1 — GLM TWEEDIE : ajustement, deviance, pseudo-R2
# ================================================================
# ┌─────────────────── PARAMETRES A MODIFIER ───────────────────┐
MIN_FREQ_MODALITE = 0.005     # modalites sous ce seuil regroupees en "Autres"
ALPHA_RIDGE       = 1e-4      # regularisation L2 du GLM
MAX_ITER_GLM      = 3000
# └──────────────────────────────────────────────────────────────┘

import numpy as np, pandas as pd
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

P_TW = float(params_finaux.get("tweedie_variance_power", 1.5))
cat = [c for c in X_tr.columns if str(X_tr[c].dtype) == "category"]
num = [c for c in X_tr.columns if c not in cat]


def _en_str(X):
    return X.assign(**{c: X[c].astype(str) for c in cat})


prep = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                          min_frequency=MIN_FREQ_MODALITE,
                          sparse_output=False), cat)])

glm = Pipeline([("prep", prep),
                ("mdl", TweedieRegressor(power=P_TW, alpha=ALPHA_RIDGE,
                                         link="log", max_iter=MAX_ITER_GLM))])
glm.fit(_en_str(X_tr), y_tr)

pred_glm = {"Entrainement": np.clip(glm.predict(_en_str(X_tr)), 0, None),
            "Validation":   np.clip(glm.predict(_en_str(X_va)), 0, None),
            "Test":         np.clip(glm.predict(_en_str(X_te)), 0, None)}
pred_lgb = pred_apres


def deviance_tweedie(y, mu, p=P_TW):
    """Deviance de Tweedie, critere naturel d'ajustement pour cette famille."""
    y, mu = np.asarray(y, float), np.maximum(np.asarray(mu, float), 1e-10)
    ys = np.maximum(y, 0)
    d = 2*(np.power(ys, 2-p)/((1-p)*(2-p))
           - ys*np.power(mu, 1-p)/(1-p)
           + np.power(mu, 2-p)/(2-p))
    return float(np.sum(np.maximum(d, 0)))


yt = np.asarray(y_te, float)
mu_nul = np.full_like(yt, float(np.mean(y_tr)))
d_nul  = deviance_tweedie(yt, mu_nul)
n_par  = len(glm.named_steps["mdl"].coef_) + 1

lignes = []
for nom, pr in [("Modèle nul (moyenne)", mu_nul),
                ("GLM Tweedie", pred_glm["Test"]),
                ("LightGBM", pred_lgb["Test"])]:
    d = deviance_tweedie(yt, pr)
    lignes.append({"Modèle": nom, "Déviance": d,
                   "Déviance expliquée (%)": 100*(1 - d/d_nul),
                   "Déviance / observation": d/len(yt),
                   "MAE": mean_absolute_error(yt, pr),
                   "Bilan de masse": np.asarray(pr).sum()/yt.sum()})
table_deviance = pd.DataFrame(lignes)

print("="*100)
print(f"TABLEAU 3.x — AJUSTEMENT PAR LA DEVIANCE DE TWEEDIE  (p = {P_TW:.3f})")
print("="*100)
print(table_deviance.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
print(f"\nParametres estimes par le GLM : {n_par}")
print(f"Modalites conservees apres regroupement : "
      f"{len(glm.named_steps['prep'].get_feature_names_out())}")

table_deviance.to_csv(RACINE_FIG/"tab_3_deviance.csv", index=False)

























# ================================================================
# BLOC G2 — RELATIVITES : exp(beta), effet multiplicatif de chaque modalite
#   C'est l'argument central de l'interpretabilite du GLM.
# ================================================================
# ┌─── PARAMETRES ───┐
VAR_RELATIVITE = ID_COLS[0]      # dimension a detailler
N_MODALITES    = 15
# └──────────────────┘

noms = glm.named_steps["prep"].get_feature_names_out()
coefs = glm.named_steps["mdl"].coef_
rel = pd.DataFrame({"terme": noms, "beta": coefs})
rel["relativite"] = np.exp(rel.beta)          # lien log -> effet multiplicatif
rel["effet_%"] = 100*(rel.relativite - 1)
rel["variable"] = rel.terme.str.split("__").str[1].str.split("_").str[0]
rel["modalite"] = rel.terme.str.split("__").str[1]

print("="*100)
print("TABLEAU 3.x — RELATIVITES DU GLM  (effet multiplicatif sur la provision)")
print("="*100)
print(f"Intercept (log) : {glm.named_steps['mdl'].intercept_:.4f}"
      f"   ->  niveau de base {np.exp(glm.named_steps['mdl'].intercept_):,.0f} €")
print(f"\nDix effets les plus forts a la hausse :")
print(rel.nlargest(10, "relativite")[["modalite", "beta", "relativite", "effet_%"]]
      .to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
print(f"\nDix effets les plus forts a la baisse :")
print(rel.nsmallest(10, "relativite")[["modalite", "beta", "relativite", "effet_%"]]
      .to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

# --- Figure : relativites d'une dimension ---
sub = rel[rel.modalite.str.startswith(VAR_RELATIVITE)].copy()
if len(sub) == 0:
    sub = rel.reindex(rel.relativite.sub(1).abs().sort_values(ascending=False).index)
sub = sub.reindex(sub.relativite.sub(1).abs().sort_values(ascending=False).index)
sub = sub.head(N_MODALITES).sort_values("relativite")
sub["etiq"] = sub.modalite.str.replace(f"{VAR_RELATIVITE}_", "", regex=False).str.slice(0, 26)

fig, ax = plt.subplots(figsize=(6.6, .34*len(sub) + 1.5))
coul = [ROUGE if r > 1 else BLEU for r in sub.relativite]
ax.barh(sub.etiq, sub.relativite - 1, left=1, color=coul, alpha=.85,
        edgecolor="white", lw=.8)
ax.axvline(1, color="black", lw=1.6)
for i, r in enumerate(sub.relativite):
    ax.text(r + (.03 if r > 1 else -.03), i, f"{r:.2f}", va="center",
            ha="left" if r > 1 else "right", fontsize=8.5, color=GRIS)

ax.set_xlabel("Relativité  $\\exp(\\beta)$   —   1 = niveau de référence")
ax.set_title(f"Effets multiplicatifs estimés par le GLM — {VAR_RELATIVITE}")
ax.grid(axis="y", visible=False)
sauver(fig, "fig_3_9_relativites_glm")

rel.to_csv(RACINE_FIG/"tab_3_relativites_glm.csv", index=False)
print(f"\nCoefficients non nuls : {int((np.abs(coefs) > 1e-8).sum())} / {len(coefs)}")



























# ================================================================
# BLOC G3 — LA ROBUSTESSE DU GLM SUR LES PETITS EFFECTIFS
#   Verifie empiriquement l'argument avance au chapitre 3.
# ================================================================
# ┌─── PARAMETRES ───┐
DIM_EFFECTIF = ID_COLS[0]
BORNES_EFF   = [0, 10, 30, 100, 300, np.inf]
# └──────────────────┘

eff_train = df.loc[X_tr.index, DIM_EFFECTIF].astype(str).value_counts()
grp_te = df.loc[X_te.index, DIM_EFFECTIF].astype(str).values
n_app = np.array([eff_train.get(g, 0) for g in grp_te])

lib = [f"{int(BORNES_EFF[i])}–{int(BORNES_EFF[i+1]) if np.isfinite(BORNES_EFF[i+1]) else '+'}"
       for i in range(len(BORNES_EFF)-1)]
classe = pd.cut(n_app, bins=BORNES_EFF, labels=lib, right=False)

lignes = []
for lab in lib:
    m = (classe == lab).to_numpy()
    if not m.sum(): continue
    mae_g = mean_absolute_error(yt[m], pred_glm["Test"][m])
    mae_l = mean_absolute_error(yt[m], pred_lgb["Test"][m])
    lignes.append({"Effectif d'apprentissage": lab, "n test": int(m.sum()),
                   "MAE GLM": mae_g, "MAE LightGBM": mae_l,
                   "Gain LightGBM (%)": 100*(1 - mae_l/max(mae_g, 1e-9)),
                   "Vainqueur": "LightGBM" if mae_l < mae_g else "GLM"})
table_effectif = pd.DataFrame(lignes)

print("="*110)
print(f"TABLEAU 3.x — PERFORMANCE SELON L'EFFECTIF D'APPRENTISSAGE ({DIM_EFFECTIF})")
print("="*110)
print(table_effectif.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

fig, ax = plt.subplots(figsize=(6.6, 4.2))
x = np.arange(len(table_effectif)); w = .38
ax.bar(x - w/2, table_effectif["MAE GLM"], w, color=VIOLET, alpha=.85,
       edgecolor="white", lw=.8, label="GLM Tweedie")
ax.bar(x + w/2, table_effectif["MAE LightGBM"], w, color=BLEU, alpha=.85,
       edgecolor="white", lw=.8, label="LightGBM")
for i, (g, l) in enumerate(zip(table_effectif["MAE GLM"],
                               table_effectif["MAE LightGBM"])):
    gagnant = VIOLET if g < l else BLEU
    ax.text(i, max(g, l)*1.04, "▼", ha="center", fontsize=9, color=gagnant)

ax.set_xticks(x); ax.set_xticklabels(table_effectif["Effectif d'apprentissage"])
ax.set_xlabel(f"Effectif du {DIM_EFFECTIF} dans l'ensemble d'apprentissage")
ax.set_ylabel("MAE sur l'ensemble de test (€)")
ax.set_title("Robustesse comparée selon la représentation du segment"
             "\n(▼ désigne le modèle le plus précis)")
ax.yaxis.set_major_formatter(EUR)
ax.legend(frameon=False)
sauver(fig, "fig_3_10_glm_vs_lgbm_effectif")

n_glm = int((table_effectif.Vainqueur == "GLM").sum())
print(f"\nLe GLM devance LightGBM sur {n_glm} / {len(table_effectif)} classes d'effectif")
print("-> " + ("l'argument de robustesse sur les petits effectifs est CONFIRME"
      if n_glm and table_effectif.Vainqueur.iloc[0] == "GLM"
      else "l'argument de robustesse n'est PAS verifie sur ces donnees"))

table_effectif.to_csv(RACINE_FIG/"tab_3_glm_effectif.csv", index=False)



















# ================================================================
# BLOC G4 — DETECTION DES INTERACTIONS MANQUEES PAR LE GLM
#   Le GBM les capte nativement, le GLM les ignore sauf specification.
# ================================================================
# ┌─── PARAMETRES ───┐
DIM_INTER = ID_COLS[:2]      # les deux dimensions croisees
N_CELL    = 12               # cellules affichees
# └──────────────────┘

d = pd.DataFrame({"y": yt, "glm": pred_glm["Test"], "lgb": pred_lgb["Test"]})
for c in DIM_INTER:
    d[c] = df.loc[X_te.index, c].astype(str).values
d["ecart_glm"] = d.glm - d.y
d["ecart_lgb"] = d.lgb - d.y

g = (d.groupby(DIM_INTER, observed=True)
       .agg(n=("y", "size"), obs=("y", "mean"),
            biais_glm=("ecart_glm", "mean"), biais_lgb=("ecart_lgb", "mean"))
       .reset_index())
g = g[g.n >= 10].copy()
g["biais_rel_glm"] = 100*g.biais_glm/np.maximum(g.obs, 1e-9)
g["biais_rel_lgb"] = 100*g.biais_lgb/np.maximum(g.obs, 1e-9)
g["gain_inter"] = g.biais_rel_glm.abs() - g.biais_rel_lgb.abs()
g["cellule"] = g[DIM_INTER].astype(str).agg(" × ".join, axis=1).str.slice(0, 30)

print("="*110)
print(f"TABLEAU 3.x — BIAIS PAR CELLULE {' × '.join(DIM_INTER)}")
print("="*110)
print("Cellules ou LightGBM corrige le plus fortement le biais du GLM :")
print(g.nlargest(N_CELL, "gain_inter")[["cellule", "n", "biais_rel_glm",
                                        "biais_rel_lgb", "gain_inter"]]
      .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

sub = g.nlargest(N_CELL, "gain_inter").sort_values("gain_inter")
fig, ax = plt.subplots(figsize=(6.8, .36*len(sub) + 1.6))
y_pos = np.arange(len(sub))
ax.hlines(y_pos, sub.biais_rel_lgb, sub.biais_rel_glm,
          color=GRIS_CLAIR, lw=2.4, zorder=1)
ax.scatter(sub.biais_rel_glm, y_pos, s=70, color=VIOLET, zorder=3,
           edgecolor="white", lw=1.2, label="GLM Tweedie")
ax.scatter(sub.biais_rel_lgb, y_pos, s=70, color=BLEU, zorder=3,
           edgecolor="white", lw=1.2, label="LightGBM")
ax.axvline(0, color="black", lw=1.4)

ax.set_yticks(y_pos); ax.set_yticklabels(sub.cellule, fontsize=8)
ax.set_xlabel("Biais relatif de la cellule (%)   —   0 = prédiction non biaisée")
ax.set_title(f"Interactions {' × '.join(DIM_INTER)} non captées par le GLM"
             "\nLa longueur du trait mesure la correction apportée par LightGBM")
ax.grid(axis="y", visible=False)
ax.legend(frameon=False, loc="lower right")
sauver(fig, "fig_3_11_interactions_manquees")

biais_moy_glm = g.biais_rel_glm.abs().mean()
biais_moy_lgb = g.biais_rel_lgb.abs().mean()
print(f"\nBiais absolu moyen par cellule :")
print(f"  GLM      : {biais_moy_glm:.2f} %")
print(f"  LightGBM : {biais_moy_lgb:.2f} %")
print(f"  Reduction : {100*(1-biais_moy_lgb/max(biais_moy_glm,1e-9)):.1f} %")
print(f"\nCellules ou le GLM est significativement biaise (> 20 %) : "
      f"{int((g.biais_rel_glm.abs() > 20).sum())} / {len(g)}")

g.to_csv(RACINE_FIG/"tab_3_interactions.csv", index=False)




























# ================================================================
# BLOC G5 — SYNTHESE VISUELLE DES TROIS FAMILLES
# ================================================================

modeles_syn = {"Référence naïve": np.full(len(yt), float(np.median(y_tr))),
               "GLM Tweedie":     pred_glm["Test"],
               "LightGBM":        pred_lgb["Test"]}
if "XGBoost" in predictions:
    modeles_syn["XGBoost"] = predictions["XGBoost"]
coul_syn = dict(zip(modeles_syn, [GRIS, VIOLET, BLEU, ORANGE]))

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))

# 1. Concentration
axes[0].plot([0, 1], [0, 1], color=GRIS_CLAIR, ls="--", lw=1)
for nom, pr in modeles_syn.items():
    o = np.argsort(np.asarray(pr, float))
    c = np.cumsum(yt[o])/yt.sum()
    axes[0].plot(np.linspace(0, 1, len(c)), c, color=coul_syn[nom], lw=1.9,
                 label=f"{nom} ({_gini(yt, pr):.3f})")
axes[0].set_xlabel("Proportion cumulée des observations")
axes[0].set_ylabel("Proportion cumulée de la provision")
axes[0].set_title("Pouvoir discriminant (Gini)", fontsize=10)
axes[0].xaxis.set_major_formatter(PCT); axes[0].yaxis.set_major_formatter(PCT)
axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")

# 2. Calibration par decile
for nom, pr in modeles_syn.items():
    if nom == "Référence naïve": continue
    t = pd.DataFrame({"y": yt, "p": np.asarray(pr, float)})
    t["d"] = pd.qcut(t.p.rank(method="first"), 10, labels=range(1, 11))
    gg = t.groupby("d", observed=True).agg(o=("y", "mean"), p=("p", "mean")).reset_index()
    axes[1].plot(gg.d.astype(int), gg.p/gg.o, color=coul_syn[nom], lw=1.9,
                 marker="o", ms=5, mfc="white", label=nom)
axes[1].axhline(1, color="black", lw=1.4)
axes[1].axhspan(.9, 1.1, color=VERT, alpha=.10)
axes[1].set_xlabel("Décile de prédiction"); axes[1].set_ylabel("Prédit / Observé")
axes[1].set_title("Calibration par décile", fontsize=10)
axes[1].set_xticks(range(1, 11)); axes[1].legend(frameon=False, fontsize=7.5)

# 3. Biais par strate
qs = [0, .5, .9, .99, 1.0]; b = [np.quantile(yt, q) for q in qs]
b[0], b[-1] = -np.inf, np.inf
lib = ["P0-50", "P50-90", "P90-99", "P99+"]
st = pd.cut(yt, bins=b, labels=lib, duplicates="drop")
x = np.arange(len(lib)); w = .8/max(len(modeles_syn)-1, 1)
for i, (nom, pr) in enumerate([(k, v) for k, v in modeles_syn.items()
                               if k != "Référence naïve"]):
    pr = np.asarray(pr, float)
    biais = [100*(pr[(st == l).to_numpy()].sum()-yt[(st == l).to_numpy()].sum())
             / max(yt[(st == l).to_numpy()].sum(), 1e-9) for l in lib]
    axes[2].bar(x + i*w - .4 + w/2, biais, w, color=coul_syn[nom], alpha=.85,
                edgecolor="white", lw=.8, label=nom)
axes[2].axhline(0, color="black", lw=1.4)
axes[2].set_xticks(x); axes[2].set_xticklabels(lib, fontsize=8)
axes[2].set_xlabel("Strate de montant"); axes[2].set_ylabel("Biais (%)")
axes[2].set_title("Biais par strate", fontsize=10)
axes[2].legend(frameon=False, fontsize=7.5)

fig.suptitle("Comparaison des familles de modèles sur l'ensemble de test",
             fontsize=11.5, y=1.02)
sauver(fig, "fig_3_12_synthese_modeles")














