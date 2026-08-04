Verification si juste avant de faire tous les merges  :: 

# ---- VERIFICATION DU MERGE ----
print("=" * 60)
print(f"Lignes dans results_test   : {len(results_test):,}")
print(f"Longueur de lower_bound    : {len(lower_bound):,}")
print(f"Longueurs identiques ?     : {len(results_test) == len(lower_bound)}")

n_nan = results_test['y_pred'].isna().sum()
print(f"\ny_pred manquants (NaN)     : {n_nan:,}  ({n_nan/len(results_test):.2%})")

print(f"\nCle ID_COLS unique dans df_final ? : {df_final[ID_COLS].duplicated().sum() == 0}")
print(f"Cle ID_COLS unique dans df_test  ? : {df_test[ID_COLS].duplicated().sum() == 0}")
print("=" * 60)

if n_nan > 0:
    print("\n/!\\ PROBLEME : des lignes de test n'ont pas de prediction associee.")
    print("Exemples de lignes sans y_pred :")
    print(results_test[results_test['y_pred'].isna()][ID_COLS].head(10).to_string())
else:
    print("\nOK : chaque ligne de test a bien une prediction.")


Mision 3 : SUppresion de ceu xuiq sont present dans le test masi âs dans le trin ni calib

import numpy as np, pandas as pd

def filtrer_groupes_non_vus(df_train, df_calib, df_test, group_cols, verbose=True):
    """Supprime du TEST les groupes absents a la fois du train ET de la calibration.
    Justification : l'exchangeabilite conforme ne tient pas pour un groupe jamais vu."""
    to_tuple = lambda d: set(map(tuple, d[group_cols].astype(str).drop_duplicates().values))
    g_train, g_calib, g_test = to_tuple(df_train), to_tuple(df_calib), to_tuple(df_test)
    vus     = g_train | g_calib
    non_vus = g_test - vus

    cle_test = df_test[group_cols].astype(str).apply(tuple, axis=1)
    masque   = ~cle_test.isin(non_vus)
    df_test_filtre = df_test[masque].copy()

    if verbose:
        print("=" * 72)
        print(f"  Granularite du filtre : {group_cols}")
        print(f"  Groupes en test              : {len(g_test):>5}")
        print(f"  dont vus en train            : {len(g_test & g_train):>5}")
        print(f"  dont vus en calibration seule: {len(g_test & g_calib - g_train):>5}")
        print(f"  dont JAMAIS vus  -> SUPPRIMES: {len(non_vus):>5}")
        print(f"\n  Lignes test avant : {len(df_test):>6,}")
        print(f"  Lignes test apres : {len(df_test_filtre):>6,}  "
              f"(-{(1-len(df_test_filtre)/len(df_test)):.2%})")
        if non_vus:
            print(f"\n  Exemples de groupes supprimes (max 10) :")
            for g in list(non_vus)[:10]:
                print(f"    {dict(zip(group_cols, g))}")
        print("=" * 72)
    return df_test_filtre, non_vus

# Granularite : "Partner" seul (demande du sup) -- ou ID_COLS complet si tu preferes
GROUP_FILTRE = ["Partner"]
df_test_clean, groupes_supprimes = filtrer_groupes_non_vus(
    df_train, df_calib, df_test, GROUP_FILTRE)

# Comparaison des deux granularites, pour arbitrer
print("\n--- Comparaison des granularites de filtrage ---")
for cols in [["Partner"], ["Partner","Companies"], ID_COLS]:
    _, nv = filtrer_groupes_non_vus(df_train, df_calib, df_test, cols, verbose=False)
    cle  = df_test[cols].astype(str).apply(tuple, axis=1)
    perdu = cle.isin(nv).mean()
    print(f"  {str(cols):<55} -> {len(nv):>4} groupes, {perdu:.2%} des lignes perdues")



Mission  1 : Couvertures marginale

import matplotlib.pyplot as plt

def couverture_marginale(results, alpha, segments=("Lob","Risk","Activity","Partner"),
                         n_min_affiche=20):
    """Couverture MARGINALE : un seul Q_hat global, mesuree globalement puis par segment."""
    cible = 1 - alpha
    globale = results["dans_intervalle"].mean()

    print("=" * 78)
    print(f"  COUVERTURE MARGINALE GLOBALE : {globale:.2%}   (cible {cible:.0%})")
    print(f"  Ecart a la cible             : {globale - cible:+.2%}")
    print(f"  Largeur moyenne              : {results['largeur_intervalle'].mean():,.0f}")
    print(f"  Largeur mediane              : {results['largeur_intervalle'].median():,.0f}")
    print("=" * 78)

    tables = {}
    for seg in segments:
        if seg not in results.columns:
            continue
        t = (results.groupby(seg, observed=True)
             .agg(n=("dans_intervalle","size"),
                  couverture=("dans_intervalle","mean"),
                  largeur_moy=("largeur_intervalle","mean"))
             .query(f"n >= {n_min_affiche}")
             .sort_values("couverture"))
        t["ecart_cible"] = t["couverture"] - cible
        tables[seg] = t
        print(f"\n--- Segment : {seg}  ({len(t)} modalites avec n >= {n_min_affiche}) ---")
        print(t.to_string(float_format=lambda v: f"{v:,.3f}"))
        sous = t[t["couverture"] < cible - 0.05]
        if len(sous):
            print(f"  /!\\ {len(sous)} modalite(s) SOUS-COUVERTE(S) de plus de 5 pts : "
                  f"{list(sous.index)}")
    return tables

def graphe_couverture(tables, alpha, max_modalites=15):
    cible = 1 - alpha
    segs = list(tables.keys())
    fig, axes = plt.subplots(1, len(segs), figsize=(5.5*len(segs), 6))
    axes = np.atleast_1d(axes)
    for ax, seg in zip(axes, segs):
        t = tables[seg].head(max_modalites)
        couleurs = ["firebrick" if c < cible - 0.05 else
                    "darkorange" if c < cible else "seagreen" for c in t["couverture"]]
        ax.barh(range(len(t)), t["couverture"], color=couleurs)
        ax.set_yticks(range(len(t))); ax.set_yticklabels(t.index, fontsize=8)
        ax.axvline(cible, color="black", ls="--", lw=1.5, label=f"Cible {cible:.0%}")
        ax.set_xlim(0, 1.02); ax.set_xlabel("Couverture empirique")
        ax.set_title(f"Couverture marginale par {seg}")
        ax.legend(fontsize=8)
        for i, (c, n) in enumerate(zip(t["couverture"], t["n"])):
            ax.text(c + 0.01, i, f"{c:.0%} (n={n})", va="center", fontsize=7)
    plt.tight_layout(); plt.savefig("couverture_marginale.png", dpi=200); plt.show()

tables_marg = couverture_marginale(results_test, ALPHA)
graphe_couverture(tables_marg, ALPHA)

Mission 2 : Couvertures Conditinnnelle 

def conformal_mondrian(df_calib, y_calib, y_lo_calib, y_hi_calib,
                       df_test, y_lo_test, y_hi_test,
                       segment_col, alpha, n_min=30):
    """Conformal MONDRIAN : un Q_hat calcule SEPAREMENT dans chaque segment.
    Garantit la couverture 1-alpha a l'interieur de chaque groupe."""
    scores = np.maximum(y_lo_calib - y_calib, y_calib - y_hi_calib)

    n_all   = len(scores)
    q_all   = min(np.ceil((n_all + 1) * (1 - alpha)) / n_all, 1.0)
    Q_global = np.quantile(scores, q_all, method="higher")

    seg_calib = df_calib[segment_col].astype(str).values
    table_Q = {}
    for s in np.unique(seg_calib):
        m = seg_calib == s
        n = int(m.sum())
        if n < n_min:
            table_Q[s] = (Q_global, n, "repli global")
        else:
            q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
            table_Q[s] = (float(np.quantile(scores[m], q, method="higher")), n, "propre")

    seg_test = df_test[segment_col].astype(str).values
    Q_test = np.array([table_Q.get(s, (Q_global, 0, "absent calib"))[0] for s in seg_test])

    lower = np.clip(y_lo_test - Q_test, 0, None)
    upper = y_hi_test + Q_test

    print("=" * 78)
    print(f"  CONFORMAL MONDRIAN sur '{segment_col}'   (Q_hat global = {Q_global:,.0f})")
    print(f"\n  {'segment':<28}{'n_calib':>9}{'Q_hat':>16}   statut")
    for s, (q, n, st) in sorted(table_Q.items(), key=lambda kv: -kv[1][0]):
        print(f"  {s:<28}{n:>9}{q:>16,.0f}   {st}")
    print("=" * 78)
    return lower, upper, table_Q, Q_global


SEGMENT = "Lob"     # segment retenu pour la couverture conditionnelle

lower_mond, upper_mond, table_Q, Q_global = conformal_mondrian(
    df_calib, y_calib.values, y_lo_calib, y_hi_calib,
    df_test_clean, y_lo_test, y_hi_test,
    segment_col=SEGMENT, alpha=ALPHA)

results_mond = results_test.copy()
results_mond["borne_basse"] = lower_mond
results_mond["borne_haute"] = upper_mond
results_mond["largeur_intervalle"] = upper_mond - lower_mond
results_mond["dans_intervalle"] = ((results_mond["y_obs"] >= results_mond["borne_basse"]) &
                                   (results_mond["y_obs"] <= results_mond["borne_haute"]))

print("\n--- Couverture CONDITIONNELLE apres Mondrian ---")
tables_cond = couverture_marginale(results_mond, ALPHA, segments=(SEGMENT,))


Mission 2 : Couverture conditionnelle (modrain)

def conformal_mondrian(df_calib, y_calib, y_lo_calib, y_hi_calib,
                       df_test, y_lo_test, y_hi_test,
                       segment_col, alpha, n_min=30):
    """Conformal MONDRIAN : un Q_hat calcule SEPAREMENT dans chaque segment.
    Garantit la couverture 1-alpha a l'interieur de chaque groupe."""
    scores = np.maximum(y_lo_calib - y_calib, y_calib - y_hi_calib)

    n_all   = len(scores)
    q_all   = min(np.ceil((n_all + 1) * (1 - alpha)) / n_all, 1.0)
    Q_global = np.quantile(scores, q_all, method="higher")

    seg_calib = df_calib[segment_col].astype(str).values
    table_Q = {}
    for s in np.unique(seg_calib):
        m = seg_calib == s
        n = int(m.sum())
        if n < n_min:
            table_Q[s] = (Q_global, n, "repli global")
        else:
            q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
            table_Q[s] = (float(np.quantile(scores[m], q, method="higher")), n, "propre")

    seg_test = df_test[segment_col].astype(str).values
    Q_test = np.array([table_Q.get(s, (Q_global, 0, "absent calib"))[0] for s in seg_test])

    lower = np.clip(y_lo_test - Q_test, 0, None)
    upper = y_hi_test + Q_test

    print("=" * 78)
    print(f"  CONFORMAL MONDRIAN sur '{segment_col}'   (Q_hat global = {Q_global:,.0f})")
    print(f"\n  {'segment':<28}{'n_calib':>9}{'Q_hat':>16}   statut")
    for s, (q, n, st) in sorted(table_Q.items(), key=lambda kv: -kv[1][0]):
        print(f"  {s:<28}{n:>9}{q:>16,.0f}   {st}")
    print("=" * 78)
    return lower, upper, table_Q, Q_global


SEGMENT = "Lob"     # segment retenu pour la couverture conditionnelle

lower_mond, upper_mond, table_Q, Q_global = conformal_mondrian(
    df_calib, y_calib.values, y_lo_calib, y_hi_calib,
    df_test_clean, y_lo_test, y_hi_test,
    segment_col=SEGMENT, alpha=ALPHA)

results_mond = results_test.copy()
results_mond["borne_basse"] = lower_mond
results_mond["borne_haute"] = upper_mond
results_mond["largeur_intervalle"] = upper_mond - lower_mond
results_mond["dans_intervalle"] = ((results_mond["y_obs"] >= results_mond["borne_basse"]) &
                                   (results_mond["y_obs"] <= results_mond["borne_haute"]))

print("\n--- Couverture CONDITIONNELLE apres Mondrian ---")
tables_cond = couverture_marginale(results_mond, ALPHA, segments=(SEGMENT,))


Comparaison visuelle conditionnelle / mondrian

def comparer_marginal_mondrian(res_marg, res_mond, segment_col, alpha, n_min=20):
    cible = 1 - alpha
    a = (res_marg.groupby(segment_col, observed=True)
         .agg(n=("dans_intervalle","size"), cov=("dans_intervalle","mean"),
              larg=("largeur_intervalle","mean")).query(f"n >= {n_min}"))
    b = (res_mond.groupby(segment_col, observed=True)
         .agg(cov=("dans_intervalle","mean"), larg=("largeur_intervalle","mean")))
    comp = a.join(b, lsuffix="_marg", rsuffix="_mond").sort_values("cov_marg")

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    y = np.arange(len(comp)); h = 0.38
    ax[0].barh(y - h/2, comp["cov_marg"], h, color="darkorange", label="Marginal (Q global)")
    ax[0].barh(y + h/2, comp["cov_mond"], h, color="seagreen",   label="Mondrian (Q par segment)")
    ax[0].axvline(cible, color="black", ls="--", lw=1.5, label=f"Cible {cible:.0%}")
    ax[0].set_yticks(y); ax[0].set_yticklabels(comp.index, fontsize=8)
    ax[0].set_xlim(0, 1.02); ax[0].set_xlabel("Couverture"); ax[0].legend(fontsize=8)
    ax[0].set_title("Couverture par segment")

    ax[1].barh(y - h/2, comp["larg_marg"], h, color="darkorange", label="Marginal")
    ax[1].barh(y + h/2, comp["larg_mond"], h, color="seagreen",   label="Mondrian")
    ax[1].set_yticks(y); ax[1].set_yticklabels(comp.index, fontsize=8)
    ax[1].set_xlabel("Largeur moyenne (EUR)"); ax[1].legend(fontsize=8)
    ax[1].set_title("Prix a payer : largeur des intervalles")

    plt.tight_layout(); plt.savefig("comparaison_mondrian.png", dpi=200); plt.show()

    ecart_m = (comp["cov_marg"] - cible).abs().mean()
    ecart_c = (comp["cov_mond"] - cible).abs().mean()
    print(f"\n  Ecart absolu moyen a la cible :")
    print(f"    Marginal : {ecart_m:.2%}")
    print(f"    Mondrian : {ecart_c:.2%}   ({(1-ecart_c/ecart_m)*100:+.0f} % d'amelioration)")
    print(f"  Largeur moyenne  Marginal : {comp['larg_marg'].mean():,.0f}")
    print(f"                   Mondrian : {comp['larg_mond'].mean():,.0f}")
    return comp

comparaison = comparer_marginal_mondrian(results_test, results_mond, SEGMENT, ALPHA)





Visualisation :: 

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats

def valider_conformal(results, alpha, segment_cols=("Lob","Risk"), n_bins_ssc=5):
    """Suite de validation complete d'un modele de Conformal Prediction.
    results doit contenir : y_obs, y_pred, borne_basse, borne_haute,
    largeur_intervalle, dans_intervalle."""
    y   = results["y_obs"].values
    lo  = results["borne_basse"].values
    hi  = results["borne_haute"].values
    w   = results["largeur_intervalle"].values
    ok  = results["dans_intervalle"].values
    n   = len(results)
    cible = 1 - alpha
    verdict = {}

    # ================= AXE 1 : VALIDITE =================
    cov = ok.mean()
    se  = np.sqrt(cible * alpha / n)
    ic_bas, ic_haut = cov - 1.96*np.sqrt(cov*(1-cov)/n), cov + 1.96*np.sqrt(cov*(1-cov)/n)
    test = stats.binomtest(int(ok.sum()), n, cible)
    valide = test.pvalue > 0.05

    print("=" * 80)
    print("  AXE 1 -- VALIDITE (couverture marginale)")
    print("=" * 80)
    print(f"  Couverture observee   : {cov:.4f}   (cible {cible:.2f})")
    print(f"  IC 95%                : [{ic_bas:.4f} ; {ic_haut:.4f}]")
    print(f"  Ecart                 : {cov - cible:+.4f}  ({(cov-cible)/se:+.2f} ecarts-types)")
    print(f"  Test binomial p-value : {test.pvalue:.4f}")
    print(f"  --> {'VALIDE' if valide else 'NON VALIDE'} : "
          f"{'couverture compatible avec la cible' if valide else 'ecart significatif a la cible'}")
    verdict["validite"] = valide

    # ================= AXE 2 : EFFICACITE =================
    larg_rel = w / np.maximum(np.abs(results['y_pred'].values), 1e-6)
    print("\n" + "=" * 80)
    print("  AXE 2 -- EFFICACITE (finesse des intervalles)")
    print("=" * 80)
    print(f"  Largeur moyenne       : {w.mean():>16,.0f} EUR")
    print(f"  Largeur mediane       : {np.median(w):>16,.0f} EUR")
    print(f"  Largeur relative med. : {np.median(larg_rel):>16.2f}  (largeur / prediction)")
    print(f"  Intervalles > 200% de la prediction : {(larg_rel > 2).mean():.1%}")

    # Score de Winkler (regle de score propre pour intervalles -- plus bas = mieux)
    winkler = (hi - lo) + (2/alpha)*(lo - y)*(y < lo) + (2/alpha)*(y - hi)*(y > hi)
    print(f"  Score de Winkler moyen: {winkler.mean():>16,.0f}   (plus bas = mieux)")
    verdict["largeur_mediane"] = float(np.median(w))
    verdict["winkler"] = float(winkler.mean())

    # ================= AXE 3 : ADAPTATIVITE =================
    err = np.abs(y - results["y_pred"].values)
    rho, p_rho = stats.spearmanr(w, err)
    print("\n" + "=" * 80)
    print("  AXE 3 -- ADAPTATIVITE (l'intervalle s'elargit-il quand l'erreur augmente ?)")
    print("=" * 80)
    print(f"  Correlation Spearman largeur/erreur : {rho:+.3f}  (p={p_rho:.2e})")
    print(f"  --> {'BONNE adaptativite' if rho > 0.3 else 'adaptativite FAIBLE'}")
    verdict["adaptativite"] = float(rho)

    # SSC : Size-Stratified Coverage -- couverture par tranche de LARGEUR
    q = np.quantile(w, np.linspace(0, 1, n_bins_ssc + 1))
    q[-1] += 1e-9
    bins = pd.cut(w, bins=np.unique(q), include_lowest=True)
    ssc = pd.DataFrame({"largeur_bin": bins, "couvert": ok}).groupby(
        "largeur_bin", observed=True)["couvert"].agg(["size","mean"])
    ssc.columns = ["n", "couverture"]
    print(f"\n  SSC -- couverture par tranche de largeur d'intervalle :")
    print(ssc.to_string(float_format=lambda v: f"{v:,.3f}"))
    ssc_min = ssc["couverture"].min()
    print(f"  --> SSC min = {ssc_min:.3f}  "
          f"({'OK' if ssc_min > cible - 0.10 else 'ALERTE : une tranche est mal couverte'})")
    verdict["ssc_min"] = float(ssc_min)

    # ================= AXE 4 : COUVERTURE CONDITIONNELLE =================
    print("\n" + "=" * 80)
    print("  AXE 4 -- COUVERTURE CONDITIONNELLE (par segment metier)")
    print("=" * 80)
    ecarts_max = {}
    for seg in segment_cols:
        if seg not in results.columns: continue
        t = (results.groupby(seg, observed=True)
             .agg(n=("dans_intervalle","size"), cov=("dans_intervalle","mean"))
             .query("n >= 20").sort_values("cov"))
        if not len(t): continue
        ec = (t["cov"] - cible).abs().max()
        ecarts_max[seg] = float(ec)
        print(f"\n  {seg} : {len(t)} modalites | ecart max a la cible = {ec:.3f}")
        print(f"    pire  : {t.index[0]} -> {t['cov'].iloc[0]:.3f} (n={t['n'].iloc[0]})")
        print(f"    meilleur : {t.index[-1]} -> {t['cov'].iloc[-1]:.3f} (n={t['n'].iloc[-1]})")
    verdict["ecart_conditionnel_max"] = ecarts_max

    # ================= VERDICT =================
    print("\n" + "=" * 80)
    print("  VERDICT DE VALIDATION")
    print("=" * 80)
    checks = [
        ("Couverture marginale valide (test binomial)", valide),
        ("Adaptativite suffisante (rho > 0.3)", rho > 0.3),
        ("SSC min acceptable (> cible - 10 pts)", ssc_min > cible - 0.10),
        ("Largeur relative mediane < 200%", np.median(larg_rel) < 2.0),
        ("Ecart conditionnel max < 10 pts",
         all(v < 0.10 for v in ecarts_max.values()) if ecarts_max else False),
    ]
    for label, res in checks:
        print(f"  [{'OK ' if res else 'KO '}] {label}")
    score = sum(r for _, r in checks)
    print(f"\n  SCORE GLOBAL : {score}/{len(checks)}")
    print("=" * 80)
    return verdict, ssc


def graphes_validation(results, alpha, ssc):
    y = results["y_obs"].values; w = results["largeur_intervalle"].values
    err = np.abs(y - results["y_pred"].values); ok = results["dans_intervalle"].values
    cible = 1 - alpha
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))

    # 1. Couverture cumulee (stabilite de la garantie)
    cum = np.cumsum(ok) / np.arange(1, len(ok) + 1)
    ax[0,0].plot(cum, color="steelblue", lw=1.2)
    ax[0,0].axhline(cible, color="red", ls="--", label=f"Cible {cible:.0%}")
    ax[0,0].set_ylim(0.5, 1.02); ax[0,0].set_xlabel("Nombre d'observations")
    ax[0,0].set_ylabel("Couverture cumulee")
    ax[0,0].set_title("Axe 1 : convergence de la couverture"); ax[0,0].legend()

    # 2. SSC
    ax[0,1].bar(range(len(ssc)), ssc["couverture"], color="seagreen")
    ax[0,1].axhline(cible, color="red", ls="--", label=f"Cible {cible:.0%}")
    ax[0,1].set_xticks(range(len(ssc)))
    ax[0,1].set_xticklabels([f"Q{i+1}" for i in range(len(ssc))])
    ax[0,1].set_xlabel("Tranche de largeur d'intervalle (croissante)")
    ax[0,1].set_ylabel("Couverture")
    ax[0,1].set_title("Axe 3 : Size-Stratified Coverage"); ax[0,1].legend()

    # 3. Adaptativite : largeur vs erreur
    ax[1,0].scatter(w[ok], err[ok], s=8, alpha=.3, color="royalblue", label="Couvert")
    ax[1,0].scatter(w[~ok], err[~ok], s=20, alpha=.7, color="red", label="Anomalie")
    ax[1,0].set_xlabel("Largeur de l'intervalle"); ax[1,0].set_ylabel("Erreur absolue")
    ax[1,0].set_title("Axe 3 : adaptativite (largeur vs erreur)"); ax[1,0].legend()

    # 4. Distribution des largeurs
    ax[1,1].hist(w, bins=50, color="slateblue", alpha=.75)
    ax[1,1].axvline(np.median(w), color="red", ls="--",
                    label=f"Mediane = {np.median(w):,.0f}")
    ax[1,1].set_xlabel("Largeur (EUR)"); ax[1,1].set_ylabel("Frequence")
    ax[1,1].set_title("Axe 2 : distribution des largeurs"); ax[1,1].legend()

    plt.tight_layout(); plt.savefig("validation_conformal.png", dpi=200); plt.show()


verdict, ssc = valider_conformal(results_test, ALPHA)
graphes_validation(results_test, ALPHA, ssc)

