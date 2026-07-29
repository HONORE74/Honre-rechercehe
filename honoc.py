# Verification prealable OBLIGATOIRE
print(f"min    : {df[TARGET].min()}")
print(f"zeros  : {(df[TARGET] == 0).sum()}")     # Gamma plante si > 0
print(f"negatifs: {(df[TARGET] < 0).sum()}")

CANDIDATS = {
    "L1 (actuel)"   : {"objective": "regression_l1"},
    "L2"            : {"objective": "regression"},
    "Gamma"         : {"objective": "gamma"},
    "Tweedie p=1.5" : {"objective": "tweedie", "tweedie_variance_power": 1.5},
    "Tweedie p=1.8" : {"objective": "tweedie", "tweedie_variance_power": 1.8},
    "Poisson"       : {"objective": "poisson"},
}

res = []
for nom, par in CANDIDATS.items():
    def mf(p=par):
        return lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=63,
                                 max_depth=-1, min_child_samples=50, reg_lambda=5.0,
                                 colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                                 random_state=42, n_jobs=-1, verbose=-1, **p)
    try:
        ph, _, _ = rolling_forecasting_procedure(
            df, TARGET, FEATURE_COLS, ID_COLS, INITIAL_TRAIN_PERIODS, mf, verbose=False)
        yt = ph["Valeur_reelle"].values.astype(float)
        yp = np.clip(ph["Valeur_predite"].values.astype(float), 0, None)
        f = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[f], yp[f]
        s99 = yt >= np.percentile(yt, 99)
        res.append({"objectif": nom,
                    "MAE": mean_absolute_error(yt, yp),
                    "wMAPE_%": np.abs(yt-yp).sum()/yt.sum()*100,
                    "MAE_P99+": mean_absolute_error(yt[s99], yp[s99]),
                    "bilan": yp.sum()/yt.sum()})
        print(f"  {nom:<15} MAE={res[-1]['MAE']:>12,.0f}  bilan={res[-1]['bilan']:.4f}")
    except Exception as e:
        print(f"  {nom:<15} ECHEC : {str(e)[:60]}")

print("\n" + pd.DataFrame(res).sort_values("MAE").to_string(index=False))
