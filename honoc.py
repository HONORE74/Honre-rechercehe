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



Code sur la partie optimisation:::


import optuna, numpy as np, lightgbm as lgb, time
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
optuna.logging.set_verbosity(optuna.logging.WARNING)

BUDGET_SECONDES = 1500          # 25 min : Optuna s'arrete tout seul

# ---------- 1. Donnees triees CHRONOLOGIQUEMENT ----------
last_period = df["time_idx"].max()
d_tune = df[df["time_idx"] < last_period].sort_values("time_idx").reset_index(drop=True)
X_raw, y_tune = d_tune[FEATURE_COLS], d_tune[TARGET].astype(float)

# Transformation UNE SEULE FOIS (gros gain de temps)
vectorizer = TableVectorizer()
X_tune = vectorizer.fit_transform(X_raw)
print(f"Donnees pretes : {X_tune.shape[0]:,} lignes x {X_tune.shape[1]} colonnes")

cv = TimeSeriesSplit(n_splits=3)         # PAS de KFold(shuffle=True) -> fuite temporelle

# ---------- 2. Objectif ----------
def objective(trial):
    params = {
        "objective": "tweedie",
        "tweedie_variance_power": trial.suggest_float(
            "tweedie_variance_power", 1.2, 1.8),          # resserre autour du gagnant (1.5)
        "n_estimators": 3000,                             # FIXE, early stopping decide
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
        "max_depth":         -1,                          # pilote par num_leaves
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 300, log=True),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq":    1,                           # sinon subsample ignore
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        "max_bin":           trial.suggest_int("max_bin", 127, 511),
        "random_state": 42, "n_jobs": -1, "verbose": -1,
    }
    scores = []
    for k, (i_tr, i_va) in enumerate(cv.split(X_tune)):
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tune[i_tr], y_tune.iloc[i_tr],
              eval_set=[(X_tune[i_va], y_tune.iloc[i_va])], eval_metric="mae",
              callbacks=[lgb.early_stopping(80, verbose=False)])
        pred = np.clip(m.predict(X_tune[i_va]), 0, None)
        scores.append(mean_absolute_error(y_tune.iloc[i_va], pred))
        trial.report(float(np.mean(scores)), k)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(scores))

# ---------- 3. Optimisation ----------
t0 = time.time()
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(multivariate=True, seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))

study.optimize(objective, n_trials=200, timeout=BUDGET_SECONDES,   # arret garanti a 25 min
               show_progress_bar=True)

print(f"\nDuree : {(time.time()-t0)/60:.1f} min | Essais : {len(study.trials)} "
      f"(elagues : {sum(t.state==optuna.trial.TrialState.PRUNED for t in study.trials)})")
print("="*60)
for k, v in study.best_params.items():
    print(f"  {k:<26} = {v}")
print(f"\n  MAE CV = {study.best_value:,.0f}")
print("="*60)

# ---------- 4. Injection dans le rolling ----------
BEST = study.best_params

def model_factory():
    return lgb.LGBMRegressor(
        objective="tweedie", n_estimators=2000, max_depth=-1,
        subsample_freq=1, random_state=42, n_jobs=-1, verbose=-1, **BEST)

# Relance rolling_forecasting_procedure avec ce model_factory pour la MAE finale
prediction_history_tuned, model_history_tuned, final_model_tuned = rolling_forecasting_procedure(
    df_raw=df, target=TARGET, feature_cols=FEATURE_COLS, id_cols=ID_COLS,
    initial_train_periods=INITIAL_TRAIN_PERIODS, model_factory=model_factory, verbose=True)
