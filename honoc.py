TARGET = "RBNS_eop"
ID_COLS = [c for c in ["Partner","Companies","Lob","Activity","Periodicity","Risk"]
           if c in df.columns]

EXCLUDE_COLS = [c for c in [TARGET, "time_idx", "year", "quarter", "annee", "Time"]
                if c in df.columns]
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE_COLS]

for c in ID_COLS:
    if c in FEATURE_COLS:
        df[c] = df[c].astype("category")

print(f"Features : {len(FEATURE_COLS)} | Categorielles incluses : "
      f"{[c for c in ID_COLS if c in FEATURE_COLS]}")

ALPHA   = 0.10
N_CALIB = 1     # consigne du responsable
N_TEST  = 1     # consigne du responsable



import lightgbm as lgb

def model_factory():
    return lgb.LGBMRegressor(
        # --- Parametres optimises par Optuna ---
        objective="tweedie",
        tweedie_variance_power=1.2154502817516997,
        learning_rate=0.031124174985263424,
        num_leaves=62,
        min_child_samples=32,
        colsample_bytree=0.9223551556702686,
        subsample=0.5714157915306912,      # <- verifie ce chiffre dans ton notebook
        reg_alpha=8.644759633929173,
        reg_lambda=0.001896001409587627,
        max_bin=410,
        # --- Parametres fixes / techniques ---
        n_estimators=2000,
        max_depth=-1,
        subsample_freq=1,
        random_state=42, n_jobs=-1, verbose=-1)







df_train, df_calib, df_test, periods_dict = chronological_split_train_calib_test(
    df, time_col="time_idx", n_calib_periods=N_CALIB, n_test_periods=N_TEST)








X_train, y_train = df_train[FEATURE_COLS], df_train[TARGET].astype(float)
X_calib, y_calib = df_calib[FEATURE_COLS], df_calib[TARGET].astype(float)
X_test,  y_test  = df_test[FEATURE_COLS],  df_test[TARGET].astype(float)
anchor_pipeline = Pipeline([("preprocess", TableVectorizer()), ("model", model_factory())])
anchor_pipeline.fit(X_train, y_train)
anchor_pred_train = np.clip(anchor_pipeline.predict(X_train), 0, None)
anchor_pred_calib = np.clip(anchor_pipeline.predict(X_calib), 0, None)
anchor_pred_test  = np.clip(anchor_pipeline.predict(X_test), 0, None)
print(f"Ancre entrainee. MAE calibration : "
      f"{np.mean(np.abs(y_calib.values - anchor_pred_calib)):,.0f}")









def quantile_resid_factory(alpha_level: float):
    return lgb.LGBMRegressor(
        objective="quantile", alpha=alpha_level,
        n_estimators=1500, learning_rate=0.05,
        num_leaves=63, max_depth=-1,
        min_child_samples=50, reg_alpha=0.1, reg_lambda=5.0,
        colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
        random_state=42, n_jobs=-1, verbose=-1)

q_lo, q_hi = ALPHA/2, 1 - ALPHA/2

resid_train = y_train.values - anchor_pred_train      # cible = ce que l'ancre a rate

pipeline_lo = Pipeline([("preprocess", TableVectorizer()), ("model", quantile_resid_factory(q_lo))])
pipeline_hi = Pipeline([("preprocess", TableVectorizer()), ("model", quantile_resid_factory(q_hi))])
pipeline_lo.fit(X_train, resid_train)
pipeline_hi.fit(X_train, resid_train)
print("Modeles de quantile des RESIDUS entraines.")








def compute_conformal_quantile(resid_calib, resid_lo_calib, resid_hi_calib, alpha: float):
    scores = np.maximum(resid_lo_calib - resid_calib, resid_calib - resid_hi_calib)
    n = len(scores)
    if n < 30:
        print(f"ATTENTION : seulement {n} observations en calibration "
              f"(N_CALIB=1) -> intervalle instable, a surveiller trimestre par trimestre.")
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    Q_hat = np.quantile(scores, q_level, method="higher")
    return Q_hat, scores

resid_calib    = y_calib.values - anchor_pred_calib
resid_lo_calib = pipeline_lo.predict(X_calib)
resid_hi_calib = pipeline_hi.predict(X_calib)

Q_hat, calib_scores = compute_conformal_quantile(resid_calib, resid_lo_calib, resid_hi_calib, ALPHA)
print(f"Q_hat = {Q_hat:.2f}")





resid_lo_test = pipeline_lo.predict(X_test)
resid_hi_test = pipeline_hi.predict(X_test)

lower_bound = anchor_pred_test + resid_lo_test - Q_hat
upper_bound = anchor_pred_test + resid_hi_test + Q_hat

lower_bound = np.clip(lower_bound, 0, None)                     # pas de provision negative

# securite : l'ancre doit toujours etre dans son propre intervalle
lower_bound = np.minimum(lower_bound, anchor_pred_test)
upper_bound = np.maximum(upper_bound, anchor_pred_test)








results_test = df_test[ID_COLS + ["year","quarter","time_idx"]].copy()
results_test["y_obs"]  = y_test.values
results_test["y_pred"] = anchor_pred_test                       # l'ancre = LE modele officiel
results_test["borne_basse"] = lower_bound
results_test["borne_haute"] = upper_bound
results_test["largeur_intervalle"] = upper_bound - lower_bound
results_test["dans_intervalle"] = ((results_test["y_obs"] >= results_test["borne_basse"]) &
                                   (results_test["y_obs"] <= results_test["borne_haute"]))
results_test["pred_dans_intervalle"] = ((results_test["y_pred"] >= results_test["borne_basse"]) &
                                        (results_test["y_pred"] <= results_test["borne_haute"]))

print(f"Coherence y_pred dans son intervalle : {results_test['pred_dans_intervalle'].mean():.1%}  "
      f"(doit etre 100% par construction)")










import matplotlib.pyplot as plt

def dashboard_conformal(results, calib_scores, Q_hat, alpha):
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))

    lo_p, hi_p = np.percentile(calib_scores, [1, 99])
    clipped = calib_scores[(calib_scores >= lo_p) & (calib_scores <= hi_p)]
    ax[0,0].hist(clipped, bins=50, color="steelblue", alpha=0.75, edgecolor="white")
    ax[0,0].axvline(Q_hat, color="red", ls="--", lw=2, label=f"Q_hat = {Q_hat:,.0f}")
    ax[0,0].set_title("Scores conformes des residus (1er-99e percentile)"); ax[0,0].legend()

    w = results["largeur_intervalle"]
    ax[0,1].hist(w[w>0], bins=50, color="seagreen", alpha=0.75)
    ax[0,1].set_xscale("log")
    ax[0,1].axvline(w.mean(), color="red", ls="--", label=f"Moyenne={w.mean():,.0f}")
    ax[0,1].set_title("Largeur des intervalles (echelle log)"); ax[0,1].legend()

    m = results["dans_intervalle"]
    ax[1,0].scatter(results.loc[m,"y_obs"], results.loc[m,"y_pred"], s=15, alpha=.4,
                    color="steelblue", label="Dans l'intervalle")
    ax[1,0].scatter(results.loc[~m,"y_obs"], results.loc[~m,"y_pred"], s=25, alpha=.8,
                    color="red", label="ANOMALIE (hors intervalle)")
    lims = [results["y_obs"].min(), results["y_obs"].max()]
    ax[1,0].plot(lims, lims, "k--", lw=1)
    ax[1,0].set_xscale("symlog"); ax[1,0].set_yscale("symlog")
    ax[1,0].set_xlabel("Reel"); ax[1,0].set_ylabel("Predit (ancre)"); ax[1,0].legend()
    ax[1,0].set_title("Predit vs Reel")

    if "Lob" in results.columns:
        cov = results.groupby("Lob")["dans_intervalle"].mean().sort_values()
        cov.plot.barh(ax=ax[1,1], color="slateblue")
        ax[1,1].axvline(1-alpha, color="red", ls="--", label=f"Cible {1-alpha:.0%}")
        ax[1,1].set_title("Couverture par Lob"); ax[1,1].legend()

    plt.tight_layout(); plt.savefig("dashboard_conformal.png", dpi=200); plt.show()
    print(f"\nCouverture globale : {results['dans_intervalle'].mean():.1%} (cible {1-alpha:.0%})")
    print(f"Largeur mediane    : {w.median():,.0f}")
    print(f"Anomalies : {(~m).sum()} / {len(results)}")

dashboard_conformal(results_test, calib_scores, Q_hat, ALPHA)








def prioriser(results, capacite=50, poids=(0.40, 0.35, 0.25), facteur_sous_estim=1.35):
    """Priorisation sur la sortie conforme corrigee (ancre + residus).
    results doit contenir : y_obs, y_pred, borne_basse, borne_haute,
    largeur_intervalle, dans_intervalle (+ ID_COLS, year, quarter)."""
    d = results.copy()

    # --- 1. Ecart HORS intervalle (0 si dedans) ---
    d["ecart_hors"] = np.maximum.reduce([d["borne_basse"] - d["y_obs"],
                                         d["y_obs"] - d["borne_haute"],
                                         np.zeros(len(d))])

    # --- 2. Severite relative : ecart normalise par la largeur de l'intervalle ---
    larg = d["largeur_intervalle"].clip(lower=1e-6)
    d["severite"] = d["ecart_hors"] / larg

    # --- 3. Materialite : impact en euros ---
    d["ecart_euros"] = (d["y_obs"] - d["y_pred"]).abs()

    # --- 4. Confiance du modele : intervalle etroit = modele sur de lui ---
    d["ratio_largeur"] = larg / d["y_pred"].abs().clip(lower=1e-6)
    d["confiance"] = 1 / (1 + d["ratio_largeur"])

    # --- 5. Sens : sous-provisionner est plus grave (impact bilan IFRS 17) ---
    d["sens"] = np.where(d["y_obs"] > d["borne_haute"], "sous-provisionne",
                np.where(d["y_obs"] < d["borne_basse"], "sur-provisionne", "conforme"))

    # --- 6. Score composite (uniquement sur les anomalies) ---
    ano = ~d["dans_intervalle"]
    w_s, w_m, w_c = poids
    r = lambda s: s.rank(pct=True)
    d["score"] = 0.0
    if ano.sum():
        sub = d[ano]
        d.loc[ano, "score"] = (w_s * r(sub["severite"])
                             + w_m * r(sub["ecart_euros"])
                             + w_c * r(sub["confiance"]))
        d.loc[ano & (d["sens"] == "sous-provisionne"), "score"] *= facteur_sous_estim

    # --- 7. Triage 3 niveaux ---
    d["priorite"] = "conforme"
    if ano.sum():
        q = d.loc[ano, "score"].quantile([0.50, 0.80])
        d.loc[ano, "priorite"] = np.select(
            [d.loc[ano, "score"] >= q[0.80], d.loc[ano, "score"] >= q[0.50]],
            ["HAUTE", "MOYENNE"], default="BASSE")

    # --- 8. Rapport ---
    print("=" * 74)
    print(f"  Observations          : {len(d):,}")
    print(f"  Couverture empirique  : {d['dans_intervalle'].mean():.2%}")
    print(f"  Anomalies detectees   : {ano.sum():,} ({ano.mean():.2%})")
    print(f"\n  {'niveau':<10}{'n':>7}{'impact EUR':>18}{'severite med':>15}")
    for niv in ["HAUTE", "MOYENNE", "BASSE"]:
        m = d["priorite"] == niv
        if m.sum():
            print(f"  {niv:<10}{m.sum():>7}{d.loc[m,'ecart_euros'].sum():>18,.0f}"
                  f"{d.loc[m,'severite'].median():>15.2f}")
    print(f"\n  Sens des anomalies :")
    print(d.loc[ano, "sens"].value_counts().to_string() if ano.sum() else "  aucune")
    print("=" * 74)

    # --- 9. File de travail bornee par la capacite de l'equipe ---
    cols = [c for c in (list(ID_COLS) + ["year","quarter"]) if c in d.columns]
    file = (d[ano].nlargest(capacite, "score")
            [cols + ["y_obs","y_pred","borne_basse","borne_haute",
                     "ecart_euros","severite","sens","priorite","score"]])
    print(f"\n  TOP {min(capacite, len(file))} A INVESTIGUER :")
    print(file.head(12).to_string(index=False))
    return d, file


resultats_finaux, file_travail = prioriser(results_test, capacite=50)
file_travail.to_csv("anomalies_prioritaires.csv", index=False)













def graphique_priorisation(resultats):
    ano = resultats[resultats["priorite"] != "conforme"]
    if not len(ano):
        print("Aucune anomalie a visualiser."); return

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ordre = ["BASSE", "MOYENNE", "HAUTE"]
    couleurs = {"BASSE":"gold", "MOYENNE":"darkorange", "HAUTE":"firebrick"}

    counts = ano["priorite"].value_counts().reindex(ordre).fillna(0)
    ax[0].bar(ordre, counts.values, color=[couleurs[o] for o in ordre])
    ax[0].set_title("Nombre d'anomalies par priorite")
    for i, v in enumerate(counts.values):
        ax[0].text(i, v, f"{int(v)}", ha="center", va="bottom")

    for niv in ordre:
        sub = ano[ano["priorite"] == niv]
        ax[1].scatter(sub["severite"], sub["ecart_euros"], s=40, alpha=.7,
                      color=couleurs[niv], label=niv)
    ax[1].set_xlabel("Severite (ecart / largeur intervalle)")
    ax[1].set_ylabel("Impact (EUR)"); ax[1].set_yscale("log")
    ax[1].set_title("Severite vs Materialite"); ax[1].legend()

    plt.tight_layout(); plt.savefig("priorisation_anomalies.png", dpi=200); plt.show()

graphique_priorisation(resultats_finaux)
