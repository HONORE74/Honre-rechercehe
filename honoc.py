if OBJECTIVE in ("tweedie", "gamma"):
    df_opt = df.loc[df[MODEL_TARGET].notna() & (df[MODEL_TARGET] >= 0)].copy()
else:
    df_opt = df.loc[df[MODEL_TARGET].notna()].copy()




assert OBJECTIVE in ("regression", "tweedie", "gamma"), (
    f"OBJECTIVE invalide ou périmé : {OBJECTIVE!r} — relance la cellule de sélection de loi "
    f"avant de lancer le tuning."
)






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
    n_jobs=4,              # fixe, pas -1
    deterministic=True,
    force_row_wise=True,
    verbose=-1,
)
if OBJECTIVE != "tweedie":
    p.pop("tweedie_variance_power", None)





EARLY_STOP = 150   # une seule définition, en haut du notebook

# dans objectif(trial) :
callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)]








RESET_STUDY = True   # True pour un tuning propre et comparable ; False pour continuer un tuning existant

if RESET_STUDY:
    CHEMIN_DB.unlink(missing_ok=True)

etude = optuna.create_study(
    study_name=NOM_ETUDE,
    storage=f"sqlite:///{CHEMIN_DB.as_posix()}",
    load_if_exists=not RESET_STUDY,
    direction="minimize",
    sampler=optuna.samplers.TPESampler(multivariate=True, seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
)








faits = len([t for t in etude.trials if t.state == optuna.trial.TrialState.COMPLETE])
n_a_lancer = max(0, N_TRIALS_MAX - faits)
print(f"Essais déjà réalisés : {faits} | Essais à lancer maintenant : {n_a_lancer}")

if RELANCER_TUNING:
    if n_a_lancer == 0:
        print("⚠️ Aucun nouvel essai ne sera lancé — augmente N_TRIALS_MAX ou repasse RESET_STUDY=True.")
    else:
        t0 = time.time()
        etude.optimize(objectif, n_trials=n_a_lancer, timeout=BUDGET_SECONDES, show_progress_bar=True)
        print(f"Durée : {(time.time()-t0)/60:.1f} min")




# Avant
callbacks=[
    lgb.early_stopping(100, verbose=False),
    lgb.log_evaluation(0),
    lgb.record_evaluation(suivi)])

# Après
callbacks=[
    lgb.early_stopping(EARLY_STOP, verbose=False),
    lgb.log_evaluation(0),
    lgb.record_evaluation(suivi)])








# Avant
callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)]

# Après
callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)]



