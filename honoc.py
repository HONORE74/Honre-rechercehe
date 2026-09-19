modele_apres = entrainer(params_finaux, X_tr, y_tr, X_va, y_va)

perf_avant_cv = calculer_perfs_splittings(modele_avant, X_tune, y_tune, n_splits=N_SPLITS_CV)
perf_apres_cv = calculer_perfs_splittings(modele_apres, X_tune, y_tune, n_splits=N_SPLITS_CV)
comparaison_cv = perf_avant_cv.merge(perf_apres_cv, on="Split", suffixes=("_avant", "_apres"))

pred_apres = {"Entrainement": predire(modele_apres, X_tr),
             "Validation": predire(modele_apres, X_va),
             "Test": predire(modele_apres, X_te)}
perf_apres_split = pd.DataFrame([metriques(k, vrais[k], pred_apres[k]) for k in vrais])

print(comparaison_cv)
print(perf_apres_split)
