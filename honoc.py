X_tune = pd.concat([X_tr, X_va], axis=0)
y_tune = pd.concat([y_tr, y_va], axis=0)
y_tune.name = 'target'

df_tune = pd.concat([X_tune, y_tune], axis=1)
df_tune["_tri"] = df.loc[df_tune.index, TIME_COL].values
df_tune = df_tune.sort_values("_tri").reset_index(drop=True)
temps_tune = df_tune["_tri"].copy()

X_tune = df_tune.drop(columns=['target', '_tri'])
y_tune = df_tune['target']



NOM_ETUDE = f"rbns_{OBJECTIVE}_v2"






def objectif(trial):
    p = dict(
        objective=OBJECTIVE,
        tweedie_variance_power=trial.suggest_float("tweedie_variance_power", 1.1, 1.95),
        n_estimators=8000,
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 400, log=True),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        max_bin=trial.suggest_int("max_bin", 63, 511),
        random_state=SEED,
        n_jobs=-1,
        verbose=-1
    )

    if OBJECTIVE != "tweedie":
        p.pop("tweedie_variance_power", None)
