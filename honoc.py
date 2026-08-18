import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

TARGET   = "IBNR_best_estimate_eop"
TIME_COL = "time_idx"
SEUIL_RARETE = 30

ID_COLS = [c for c in ["Partner", "Companies", "Lob", "Activity",
                       "Periodicity", "Risk", "Reinsurance"] if c in df.columns]
EXCLURE = [TARGET, TIME_COL, "year", "quarter", "annee", "pays"] + ID_COLS
FEATURES = [c for c in df.columns if c not in EXCLURE]

PALETTE = {"train": "#1f4e79", "valid": "#e07b39",
           "test": "#2e7d32", "ref": "#b0b0b0"}

# --- 1. Identification des colonnes texte -----------------------------------
CAT_COLS = [c for c in FEATURES
            if df[c].dtype == object or str(df[c].dtype) == "category"]
print(f"Variables catégorielles détectées ({len(CAT_COLS)}) : {CAT_COLS}\n")

# --- 2. Regroupement des modalités rares ------------------------------------
rapport_rarete = []
for c in CAT_COLS:
    df[c] = df[c].astype(str)
    effectifs = df[c].value_counts()
    rares = effectifs[effectifs < SEUIL_RARETE].index

    rapport_rarete.append({
        "Variable": c,
        "Modalités avant": len(effectifs),
        "Modalités rares": len(rares),
        "Obs. regroupées": int(effectifs[rares].sum()),
        "Exemples": ", ".join(map(str, rares[:4]))
    })
    df.loc[df[c].isin(rares), c] = "AUTRE"

print(pd.DataFrame(rapport_rarete).to_string(index=False))

# --- 3. Encodage catégoriel UNIQUE et GLOBAL --------------------------------
for c in CAT_COLS:
    df[c] = df[c].astype("category")

print(f"\nModalités après regroupement :")
for c in CAT_COLS:
    print(f"  {c:25s} : {df[c].nunique()} modalités")














def splits_temporels(df_source, col_temps, n_splits=4, horizon=2, min_train=8):
    """Fenêtre extensible. Aucun trimestre de test n'est antérieur au train."""
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

print(f"Observations retenues : {len(df_opt):,} sur {len(df):,}\n")

controle = []
for i, (tr, te) in enumerate(folds, 1):
    pt, pe = df_opt.loc[tr, TIME_COL], df_opt.loc[te, TIME_COL]
    fuite = pt.max() >= pe.min()
    controle.append({
        "Fold": i,
        "Train": f"{pt.min()} → {pt.max()}",
        "Obs. train": len(tr),
        "Test": f"{pe.min()} → {pe.max()}",
        "Obs. test": len(te),
        "Fuite": "OUI — ANOMALIE" if fuite else "non"
    })
print(pd.DataFrame(controle).to_string(index=False))

# --- Contrôle de couverture catégorielle par fold ---------------------------
print("\nModalités présentes en test mais absentes du train :")
for i, (tr, te) in enumerate(folds, 1):
    manquantes = {}
    for c in CAT_COLS:
        mod_tr = set(df_opt.loc[tr, c].dropna().unique())
        mod_te = set(df_opt.loc[te, c].dropna().unique())
        absentes = mod_te - mod_tr
        if absentes:
            manquantes[c] = list(absentes)
    print(f"  Fold {i} : {manquantes if manquantes else 'aucune'}")























def espace_recherche(trial):
    max_depth = trial.suggest_int("max_depth", 4, 14)

    return {
        # --- Loi de la cible ---
        "objective": "tweedie",
        "tweedie_variance_power": trial.suggest_float(
            "tweedie_variance_power", 1.10, 1.90, step=0.05),

        # --- Capacité ---
        "max_depth": max_depth,
        "num_leaves": trial.suggest_int(
            "num_leaves", 16, min(256, 2**max_depth - 1), log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),

        # --- Anti-surapprentissage ---
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 300, log=True),
        "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),

        # --- Sous-échantillonnage ---
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq":   trial.suggest_int("subsample_freq", 1, 7),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),

        # --- Régularisation ---
        "reg_alpha":  trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "max_bin":    trial.suggest_int("max_bin", 63, 511, log=True),

        # --- Stabilité des variables catégorielles ---
        "min_data_per_group": trial.suggest_int("min_data_per_group", 20, 200, log=True),
        "cat_smooth":         trial.suggest_float("cat_smooth", 1.0, 100.0, log=True),
        "cat_l2":             trial.suggest_float("cat_l2", 1.0, 50.0, log=True),

        # --- Fixes ---
        "n_estimators": 4000,
        "random_state": 42, "n_jobs": -1, "verbose": -1,
    }






























def objectif(trial):
    params = espace_recherche(trial)
    scores = []

    for i, (idx_tr, idx_te) in enumerate(folds):
        X_tr_f, y_tr_f = df_opt.loc[idx_tr, FEATURES], df_opt.loc[idx_tr, TARGET]
        X_te_f, y_te_f = df_opt.loc[idx_te, FEATURES], df_opt.loc[idx_te, TARGET]

        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr_f, y_tr_f,
              eval_set=[(X_te_f, y_te_f)],
              eval_metric="mae",
              categorical_feature=CAT_COLS,
              callbacks=[lgb.early_stopping(150, verbose=False),
                         lgb.log_evaluation(0)])

        pred = np.clip(m.predict(X_te_f), 0, None)
        scores.append(mean_absolute_error(y_te_f, pred))

        trial.report(float(np.mean(scores)), step=i)
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

n_elag = sum(t.state == optuna.trial.TrialState.PRUNED for t in etude.trials)
print(f"\nEssais terminés : {len(etude.trials)}  (dont {n_elag} élagués)")
print(f"Meilleur MAE moyen sur les {len(folds)} folds : {etude.best_value:,.0f} €\n")
for k, v in etude.best_params.items():
    print(f"  {k:28s} : {v}")






























fig = optuna.visualization.plot_optimization_history(etude)
fig.update_layout(
    title="Convergence de l'optimisation bayésienne",
    xaxis_title="Numéro d'essai",
    yaxis_title="MAE moyen en validation temporelle (€)",
    template="plotly_white", height=520
)
fig
























fig = optuna.visualization.plot_param_importances(etude)
fig.update_layout(
    title="Contribution de chaque hyperparamètre à la variance de la performance",
    template="plotly_white", height=600
)
fig


















cles = sorted(etude.best_params.keys())[:6]
fig = optuna.visualization.plot_slice(etude, params=cles)
fig.update_layout(
    title="Performance obtenue selon la valeur de chaque hyperparamètre",
    template="plotly_white", height=560
)
fig













cles = sorted(etude.best_params.keys())[:6]
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
                colorbar=dict(title="MAE")), name="Essais"))
fig.add_vline(x=etude.best_params["tweedie_variance_power"],
              line_dash="dash", line_color="#c62828",
              annotation_text=f"Optuna : p = {etude.best_params['tweedie_variance_power']:.2f}")

try:
    fig.add_vline(x=p_optimal, line_dash="dot", line_color="#2e7d32",
                  annotation_text=f"Déviance : p = {p_optimal:.2f}")
except NameError:
    pass

fig.update_layout(
    title=("Paramètre de puissance de Tweedie : convergence des méthodes d'estimation<br>"
           "<sub>Un écart faible entre les deux traits valide le choix de la loi</sub>"),
    xaxis_title="tweedie_variance_power", yaxis_title="MAE en validation (€)",
    template="plotly_white", height=540
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
                colorbar=dict(title="MAE")), name="Essais"))
fig.add_vline(x=etude.best_params["tweedie_variance_power"],
              line_dash="dash", line_color="#c62828",
              annotation_text=f"Optuna : p = {etude.best_params['tweedie_variance_power']:.2f}")

try:
    fig.add_vline(x=p_optimal, line_dash="dot", line_color="#2e7d32",
                  annotation_text=f"Déviance : p = {p_optimal:.2f}")
except NameError:
    pass

fig.update_layout(
    title=("Paramètre de puissance de Tweedie : convergence des méthodes d'estimation<br>"
           "<sub>Un écart faible entre les deux traits valide le choix de la loi</sub>"),
    xaxis_title="tweedie_variance_power", yaxis_title="MAE en validation (€)",
    template="plotly_white", height=540
)
fig




















# --- Découpage final, reconstruit sur le df nettoyé -------------------------
periodes = np.sort(df[TIME_COL].unique())
cut_train = periodes[int(0.70 * len(periodes))]
cut_valid = periodes[int(0.85 * len(periodes))]

masque = lambda d: d[TARGET].notna() & (d[TARGET] >= 0)

df_train = df[df[TIME_COL] <= cut_train]
df_valid = df[(df[TIME_COL] > cut_train) & (df[TIME_COL] <= cut_valid)]
df_test  = df[df[TIME_COL] > cut_valid]

X_tr, y_tr = df_train.loc[masque(df_train), FEATURES], df_train.loc[masque(df_train), TARGET]
X_va, y_va = df_valid.loc[masque(df_valid), FEATURES], df_valid.loc[masque(df_valid), TARGET]
X_te, y_te = df_test.loc[masque(df_test),  FEATURES], df_test.loc[masque(df_test),  TARGET]

print(f"Train {len(X_tr):,} | Valid {len(X_va):,} | Test {len(X_te):,}")

# --- Modèle de référence, paramètres par défaut -----------------------------
params_base = dict(objective="tweedie", tweedie_variance_power=1.5,
                   n_estimators=4000, learning_rate=0.05, num_leaves=31,
                   random_state=42, n_jobs=-1, verbose=-1)

modele_base = lgb.LGBMRegressor(**params_base)
modele_base.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="mae",
                categorical_feature=CAT_COLS,
                callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])

# --- Modèle optimisé --------------------------------------------------------
params_finaux = {**etude.best_params, "objective": "tweedie",
                 "n_estimators": 4000, "random_state": 42,
                 "n_jobs": -1, "verbose": -1}

modele_opt = lgb.LGBMRegressor(**params_finaux)
modele_opt.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="mae",
               categorical_feature=CAT_COLS,
               callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])

print(f"Itérations retenues — base : {modele_base.best_iteration_}, "
      f"optimisé : {modele_opt.best_iteration_}")

# --- Évaluation -------------------------------------------------------------
def indice_gini(y_vrai, y_pred):
    ordre = np.argsort(y_pred)
    y_ord = np.asarray(y_vrai)[ordre]
    g_mod = 1 - 2*np.trapz(np.cumsum(y_ord)/y_ord.sum(), dx=1/len(y_ord))
    y_par = np.sort(np.asarray(y_vrai))
    g_par = 1 - 2*np.trapz(np.cumsum(y_par)/y_par.sum(), dx=1/len(y_par))
    return g_mod/g_par if g_par else np.nan

def evaluer(nom, y_vrai, y_pred):
    y_vrai, y_pred = np.asarray(y_vrai), np.asarray(y_pred)
    return {"Modèle": nom,
            "RMSE": np.sqrt(mean_squared_error(y_vrai, y_pred)),
            "MAE": mean_absolute_error(y_vrai, y_pred),
            "R2": r2_score(y_vrai, y_pred),
            "Gini": indice_gini(y_vrai, y_pred),
            "Biais (%)": 100*(y_pred.sum()-y_vrai.sum())/y_vrai.sum()}

pred_base = np.clip(modele_base.predict(X_te), 0, None)
pred_opt  = np.clip(modele_opt.predict(X_te),  0, None)

comp = pd.DataFrame([evaluer("Paramètres par défaut", y_te, pred_base),
                     evaluer("Après optimisation",    y_te, pred_opt)])
print("\n" + comp.to_string(index=False))

gain_mae  = 100*(1 - comp.loc[1,"MAE"]/comp.loc[0,"MAE"])
gain_gini = 100*(comp.loc[1,"Gini"]/comp.loc[0,"Gini"] - 1)
print(f"\nGain MAE : {gain_mae:+.1f} %   |   Gain Gini : {gain_gini:+.1f} %")

fig = go.Figure()
for m in ["RMSE", "MAE", "Gini"]:
    ref = max(abs(comp.loc[0, m]), 1e-9)
    fig.add_trace(go.Bar(x=[m], y=[100*comp.loc[0,m]/ref], name="Par défaut",
                         marker_color=PALETTE["ref"], showlegend=(m == "RMSE")))
    fig.add_trace(go.Bar(x=[m], y=[100*comp.loc[1,m]/ref], name="Optimisé",
                         marker_color=PALETTE["train"], showlegend=(m == "RMSE")))

fig.update_layout(
    title=(f"Effet de l'optimisation des hyperparamètres, base 100<br>"
           f"<sub>Gain MAE : {gain_mae:+.1f} %  ·  Gain Gini : {gain_gini:+.1f} %</sub>"),
    yaxis_title="Indice (défaut = 100)", barmode="group",
    template="plotly_white", height=520
)
fig






















# --- Découpage final, reconstruit sur le df nettoyé -------------------------
periodes = np.sort(df[TIME_COL].unique())
cut_train = periodes[int(0.70 * len(periodes))]
cut_valid = periodes[int(0.85 * len(periodes))]

masque = lambda d: d[TARGET].notna() & (d[TARGET] >= 0)

df_train = df[df[TIME_COL] <= cut_train]
df_valid = df[(df[TIME_COL] > cut_train) & (df[TIME_COL] <= cut_valid)]
df_test  = df[df[TIME_COL] > cut_valid]

X_tr, y_tr = df_train.loc[masque(df_train), FEATURES], df_train.loc[masque(df_train), TARGET]
X_va, y_va = df_valid.loc[masque(df_valid), FEATURES], df_valid.loc[masque(df_valid), TARGET]
X_te, y_te = df_test.loc[masque(df_test),  FEATURES], df_test.loc[masque(df_test),  TARGET]

print(f"Train {len(X_tr):,} | Valid {len(X_va):,} | Test {len(X_te):,}")

# --- Modèle de référence, paramètres par défaut -----------------------------
params_base = dict(objective="tweedie", tweedie_variance_power=1.5,
                   n_estimators=4000, learning_rate=0.05, num_leaves=31,
                   random_state=42, n_jobs=-1, verbose=-1)

modele_base = lgb.LGBMRegressor(**params_base)
modele_base.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="mae",
                categorical_feature=CAT_COLS,
                callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])

# --- Modèle optimisé --------------------------------------------------------
params_finaux = {**etude.best_params, "objective": "tweedie",
                 "n_estimators": 4000, "random_state": 42,
                 "n_jobs": -1, "verbose": -1}

modele_opt = lgb.LGBMRegressor(**params_finaux)
modele_opt.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="mae",
               categorical_feature=CAT_COLS,
               callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])

print(f"Itérations retenues — base : {modele_base.best_iteration_}, "
      f"optimisé : {modele_opt.best_iteration_}")

# --- Évaluation -------------------------------------------------------------
def indice_gini(y_vrai, y_pred):
    ordre = np.argsort(y_pred)
    y_ord = np.asarray(y_vrai)[ordre]
    g_mod = 1 - 2*np.trapz(np.cumsum(y_ord)/y_ord.sum(), dx=1/len(y_ord))
    y_par = np.sort(np.asarray(y_vrai))
    g_par = 1 - 2*np.trapz(np.cumsum(y_par)/y_par.sum(), dx=1/len(y_par))
    return g_mod/g_par if g_par else np.nan

def evaluer(nom, y_vrai, y_pred):
    y_vrai, y_pred = np.asarray(y_vrai), np.asarray(y_pred)
    return {"Modèle": nom,
            "RMSE": np.sqrt(mean_squared_error(y_vrai, y_pred)),
            "MAE": mean_absolute_error(y_vrai, y_pred),
            "R2": r2_score(y_vrai, y_pred),
            "Gini": indice_gini(y_vrai, y_pred),
            "Biais (%)": 100*(y_pred.sum()-y_vrai.sum())/y_vrai.sum()}

pred_base = np.clip(modele_base.predict(X_te), 0, None)
pred_opt  = np.clip(modele_opt.predict(X_te),  0, None)

comp = pd.DataFrame([evaluer("Paramètres par défaut", y_te, pred_base),
                     evaluer("Après optimisation",    y_te, pred_opt)])
print("\n" + comp.to_string(index=False))

gain_mae  = 100*(1 - comp.loc[1,"MAE"]/comp.loc[0,"MAE"])
gain_gini = 100*(comp.loc[1,"Gini"]/comp.loc[0,"Gini"] - 1)
print(f"\nGain MAE : {gain_mae:+.1f} %   |   Gain Gini : {gain_gini:+.1f} %")

fig = go.Figure()
for m in ["RMSE", "MAE", "Gini"]:
    ref = max(abs(comp.loc[0, m]), 1e-9)
    fig.add_trace(go.Bar(x=[m], y=[100*comp.loc[0,m]/ref], name="Par défaut",
                         marker_color=PALETTE["ref"], showlegend=(m == "RMSE")))
    fig.add_trace(go.Bar(x=[m], y=[100*comp.loc[1,m]/ref], name="Optimisé",
                         marker_color=PALETTE["train"], showlegend=(m == "RMSE")))

fig.update_layout(
    title=(f"Effet de l'optimisation des hyperparamètres, base 100<br>"
           f"<sub>Gain MAE : {gain_mae:+.1f} %  ·  Gain Gini : {gain_gini:+.1f} %</sub>"),
    yaxis_title="Indice (défaut = 100)", barmode="group",
    template="plotly_white", height=520
)
fig
