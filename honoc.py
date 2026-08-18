import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_absolute_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

def splits_temporels(df_source, col_temps, n_splits=4, horizon=2, min_train=8):
    """
    Fenêtre extensible : le train grandit, le test avance dans le temps.
    Aucun trimestre de test n'est antérieur à un trimestre d'entraînement.
    """
    periodes = np.sort(df_source[col_temps].unique())
    splits = []
    for k in range(n_splits):
        fin_train = min_train + k * horizon
        if fin_train + horizon > len(periodes):
            break
        p_train = periodes[:fin_train]
        p_test  = periodes[fin_train:fin_train + horizon]
        splits.append((
            df_source.index[df_source[col_temps].isin(p_train)],
            df_source.index[df_source[col_temps].isin(p_test)]
        ))
    return splits

df_opt = df.loc[df[TARGET].notna() & (df[TARGET] >= 0)].copy()
folds = splits_temporels(df_opt, TIME_COL, n_splits=4, horizon=2, min_train=8)

for i, (tr, te) in enumerate(folds, 1):
    pt = df_opt.loc[tr, TIME_COL]
    pe = df_opt.loc[te, TIME_COL]
    print(f"Fold {i} | train : {pt.min()}-{pt.max()} ({len(tr):,} obs) "
          f"| test : {pe.min()}-{pe.max()} ({len(te):,} obs)")















def espace_recherche(trial):
    max_depth = trial.suggest_int("max_depth", 4, 14)

    params = {
        # --- Objectif, avec p traité comme hyperparamètre à part entière ---
        "objective": "tweedie",
        "tweedie_variance_power": trial.suggest_float(
            "tweedie_variance_power", 1.10, 1.90, step=0.05),

        # --- Capacité du modèle ---
        "max_depth": max_depth,
        "num_leaves": trial.suggest_int(
            "num_leaves", 16, min(256, 2**max_depth - 1), log=True),

        # --- Vitesse d'apprentissage ---
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),

        # --- Anti-surapprentissage (essentiel avec vos valeurs extrêmes) ---
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 300, log=True),
        "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),

        # --- Sous-échantillonnage ---
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq":   trial.suggest_int("subsample_freq", 1, 7),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),

        # --- Régularisation ---
        "reg_alpha":  trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),

        # --- Discrétisation ---
        "max_bin": trial.suggest_int("max_bin", 63, 511, log=True),

        # --- Fixes ---
        "n_estimators": 4000,
        "random_state": 42, "n_jobs": -1, "verbose": -1,
    }
    return params



















def espace_recherche(trial):
    max_depth = trial.suggest_int("max_depth", 4, 14)

    params = {
        # --- Objectif, avec p traité comme hyperparamètre à part entière ---
        "objective": "tweedie",
        "tweedie_variance_power": trial.suggest_float(
            "tweedie_variance_power", 1.10, 1.90, step=0.05),

        # --- Capacité du modèle ---
        "max_depth": max_depth,
        "num_leaves": trial.suggest_int(
            "num_leaves", 16, min(256, 2**max_depth - 1), log=True),

        # --- Vitesse d'apprentissage ---
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),

        # --- Anti-surapprentissage (essentiel avec vos valeurs extrêmes) ---
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 300, log=True),
        "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),

        # --- Sous-échantillonnage ---
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq":   trial.suggest_int("subsample_freq", 1, 7),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),

        # --- Régularisation ---
        "reg_alpha":  trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),

        # --- Discrétisation ---
        "max_bin": trial.suggest_int("max_bin", 63, 511, log=True),

        # --- Fixes ---
        "n_estimators": 4000,
        "random_state": 42, "n_jobs": -1, "verbose": -1,
    }
    return params
















def objectif(trial):
    params = espace_recherche(trial)
    scores = []

    for i, (idx_tr, idx_te) in enumerate(folds):
        X_tr_f = df_opt.loc[idx_tr, FEATURES]
        y_tr_f = df_opt.loc[idx_tr, TARGET]
        X_te_f = df_opt.loc[idx_te, FEATURES]
        y_te_f = df_opt.loc[idx_te, TARGET]

        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr_f, y_tr_f,
              eval_set=[(X_te_f, y_te_f)],
              eval_metric="mae",
              callbacks=[lgb.early_stopping(150, verbose=False),
                         lgb.log_evaluation(0)])

        pred = np.clip(m.predict(X_te_f), 0, None)
        scores.append(mean_absolute_error(y_te_f, pred))

        # Élagage : abandon des essais manifestement mauvais
        trial.report(np.mean(scores), step=i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


etude = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(multivariate=True, seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=1),
    study_name="lgbm_ibnr"
)

etude.optimize(objectif, n_trials=150, show_progress_bar=True)

print(f"\nEssais terminés : {len(etude.trials)}")
print(f"Essais élagués  : {sum(t.state == optuna.trial.TrialState.PRUNED for t in etude.trials)}")
print(f"Meilleur MAE    : {etude.best_value:,.0f} €\n")
for k, v in etude.best_params.items():
    print(f"  {k:28s} : {v}")


























fig = optuna.visualization.plot_optimization_history(etude)
fig.update_layout(
    title="Convergence de l'optimisation bayésienne",
    xaxis_title="Numéro d'essai", yaxis_title="MAE moyen en validation temporelle (€)",
    template="plotly_white", height=520
)
fig















fig = optuna.visualization.plot_param_importances(etude)
fig.update_layout(
    title="Contribution de chaque hyperparamètre à la variance de la performance",
    template="plotly_white", height=560
)
fig













cles = list(etude.best_params.keys())[:6]
fig = optuna.visualization.plot_slice(etude, params=cles)
fig.update_layout(
    title="Performance obtenue selon la valeur de chaque hyperparamètre",
    template="plotly_white", height=560
)
fig













donnees_p = [(t.params["tweedie_variance_power"], t.value)
             for t in etude.trials
             if t.value is not None and "tweedie_variance_power" in t.params]
p_vals, scores_p = zip(*donnees_p)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=p_vals, y=scores_p, mode="markers",
    marker=dict(size=7, color=scores_p, colorscale="Viridis_r",
                colorbar=dict(title="MAE")),
    name="Essais"))
fig.add_vline(x=etude.best_params["tweedie_variance_power"],
              line_dash="dash", line_color="#c62828",
              annotation_text=f"Optuna : p = {etude.best_params['tweedie_variance_power']:.2f}")
fig.add_vline(x=p_optimal, line_dash="dot", line_color="#2e7d32",
              annotation_text=f"Déviance : p = {p_optimal:.2f}")
fig.update_layout(
    title=("Paramètre de puissance de Tweedie : convergence des deux méthodes d'estimation<br>"
           "<sub>Un écart faible entre les deux traits valide le choix de la loi</sub>"),
    xaxis_title="tweedie_variance_power", yaxis_title="MAE en validation (€)",
    template="plotly_white", height=540
)
fig






















params_finaux = {**etude.best_params,
                 "objective": "tweedie", "n_estimators": 4000,
                 "random_state": 42, "n_jobs": -1, "verbose": -1}

modele_opt = lgb.LGBMRegressor(**params_finaux)
modele_opt.fit(X_tr, y_tr,
               eval_set=[(X_va, y_va)], eval_metric="mae",
               callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])

pred_avant = np.clip(modele.predict(X_te), 0, None)
pred_apres = np.clip(modele_opt.predict(X_te), 0, None)

comp = pd.DataFrame([
    evaluer("Avant optimisation", y_te, pred_avant),
    evaluer("Après optimisation", y_te, pred_apres),
]).rename(columns={"Ensemble": "Modèle"})
print(comp.to_string(index=False))

metriques = ["RMSE", "MAE", "Gini"]
fig = go.Figure()
for i, m in enumerate(metriques):
    v_av, v_ap = comp.loc[0, m], comp.loc[1, m]
    ref = max(abs(v_av), 1e-9)
    fig.add_trace(go.Bar(x=[m], y=[100*v_av/ref], name="Avant",
                         marker_color=PALETTE["ref"], showlegend=(i == 0)))
    fig.add_trace(go.Bar(x=[m], y=[100*v_ap/ref], name="Après",
                         marker_color=PALETTE["train"], showlegend=(i == 0)))

fig.update_layout(
    title="Effet de l'optimisation des hyperparamètres, base 100 avant optimisation",
    yaxis_title="Indice (avant = 100)", barmode="group",
    template="plotly_white", height=520
)
fig
