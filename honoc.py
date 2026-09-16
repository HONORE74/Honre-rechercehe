resultats_comparaison = []

for t_mode in ["cumul", "dec"]:
    for f_mode in ["A", "B", "C"]:
        TARGET_MODE = t_mode
        FEATURE_MODE = f_mode

        # -- (garde ta reconstruction existante de FEATURES_MODEL, MODEL_TARGET,
        #     X_tr, y_tr, X_va, y_va, X_te, y_te, inchangée)

        modele_combo = lgb.LGBMRegressor(
            objective=OBJECTIVE,
            metric=METRIC,
            n_estimators=4000,
            learning_rate=0.02,
            num_leaves=40,
            max_depth=8,
            min_child_samples=100,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.5,
            reg_lambda=1.0,
            **LGBM_DETERMINISM,   # random_state=SEED, n_jobs=4, deterministic=True, force_row_wise=True
            verbose=-1,
        )
        modele_combo.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)],
        )

        pred_test = modele_combo.predict(X_te)
        idx_test = X_te.index

        # -- (garde ta logique de reconstruction "si dec" et le calcul des métriques inchangés,
        #     juste remplacer modele.predict(...) par modele_combo.predict(...) partout)
