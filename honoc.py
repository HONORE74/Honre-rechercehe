# ┌─── PARAMETRES ───┐
DOSSIER_ARTEFACTS = "artefacts_modele"
NOM_ETUDE         = "rbns_tweedie_v1"
BUDGET_SECONDES   = 5400
N_TRIALS_MAX      = 200
SEED              = 42
UTILISER_OFFSET   = False
COL_EXPOSITION    = "GWP"
# └──────────────────┘

import os, json, time, warnings
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd, lightgbm as lgb, joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
RACINE = Path(DOSSIER_ARTEFACTS); RACINE.mkdir(parents=True, exist_ok=True)
CHEMIN_DB        = RACINE / f"{NOM_ETUDE}_optuna.db"
CHEMIN_PARAMS    = RACINE / f"{NOM_ETUDE}_best_params.json"
CHEMIN_MODELE    = RACINE / f"{NOM_ETUDE}_modele.joblib"
CHEMIN_METRIQUES = RACINE / f"{NOM_ETUDE}_metriques.csv"

print(f"Artefacts : {RACINE.resolve()}")
for f in [CHEMIN_DB, CHEMIN_PARAMS, CHEMIN_MODELE]:
    print(f"  {'[existe]' if f.exists() else '[absent]'} {f.name}")

















QUANTILES_STRATES = [0, 0.50, 0.90, 0.99, 1.0]

def indice_gini(y_vrai, y_pred):
    y_vrai, y_pred = np.asarray(y_vrai, float), np.asarray(y_pred, float)
    if y_vrai.sum() <= 0: return np.nan
    cum = np.cumsum(y_vrai[np.argsort(y_pred)]) / y_vrai.sum()
    g_m = 1 - 2*np.trapezoid(cum, dx=1/len(cum))
    cum_p = np.cumsum(y_vrai[np.argsort(y_vrai)]) / y_vrai.sum()
    g_p = 1 - 2*np.trapezoid(cum_p, dx=1/len(cum_p))
    return g_m/g_p if g_p else np.nan

def metriques(nom, y_vrai, y_pred):
    y_vrai, y_pred = np.asarray(y_vrai, float), np.asarray(y_pred, float)
    f = np.isfinite(y_vrai) & np.isfinite(y_pred); y_vrai, y_pred = y_vrai[f], y_pred[f]
    nn = y_vrai > 1
    return {"Ensemble": nom, "n": len(y_vrai),
            "MAE": mean_absolute_error(y_vrai, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_vrai, y_pred))),
            "wMAPE_%": 100*np.abs(y_vrai-y_pred).sum()/max(y_vrai.sum(),1e-9),
            "MAPE_median_%": 100*np.median(np.abs(y_vrai[nn]-y_pred[nn])/y_vrai[nn]) if nn.any() else np.nan,
            "R2": r2_score(y_vrai, y_pred), "Gini": indice_gini(y_vrai, y_pred),
            "Biais_global_%": 100*(y_pred.sum()-y_vrai.sum())/max(y_vrai.sum(),1e-9)}

def metriques_par_strate(y_vrai, y_pred, quantiles=QUANTILES_STRATES):
    y_vrai, y_pred = np.asarray(y_vrai, float), np.asarray(y_pred, float)
    b = [np.quantile(y_vrai, q) for q in quantiles]; b[0], b[-1] = -np.inf, np.inf
    lib = [f"P{int(100*quantiles[i])}-P{int(100*quantiles[i+1])}" for i in range(len(quantiles)-1)]
    st = pd.cut(y_vrai, bins=b, labels=lib, duplicates="drop")
    out = []
    for lab in lib:
        m = (st == lab).to_numpy()
        if not m.sum(): continue
        yv, yp = y_vrai[m], y_pred[m]
        out.append({"Strate": lab, "n": int(m.sum()), "Montant_moyen": yv.mean(),
                    "MAE": mean_absolute_error(yv, yp),
                    "wMAPE_%": 100*np.abs(yv-yp).sum()/max(yv.sum(),1e-9),
                    "Biais_%": 100*(yp.sum()-yv.sum())/max(yv.sum(),1e-9),
                    "Part_du_total_%": 100*yv.sum()/max(y_vrai.sum(),1e-9)})
    return pd.DataFrame(out)
















# ┌─── PARAMETRES ───┐
PARAMS_BASELINE = dict(objective="tweedie", tweedie_variance_power=1.5, metric="tweedie",
    n_estimators=4000, learning_rate=0.02, num_leaves=40, max_depth=8,
    min_child_samples=100, subsample=0.9, subsample_freq=1, colsample_bytree=0.85,
    reg_alpha=0.5, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
EARLY_STOP = 100
# └──────────────────┘

if "p_safe" in globals():
    PARAMS_BASELINE["tweedie_variance_power"] = float(p_safe)
print(f"Puissance Tweedie : {PARAMS_BASELINE['tweedie_variance_power']:.4f}")

def _init_score(X):
    if not UTILISER_OFFSET or COL_EXPOSITION not in X.columns: return None
    return np.log(X[COL_EXPOSITION].astype(float).clip(lower=1e-6)).values

def entrainer(params, X_tr, y_tr, X_va, y_va, es=EARLY_STOP):
    m = lgb.LGBMRegressor(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)],
          eval_names=["Entrainement", "Validation"], init_score=_init_score(X_tr),
          eval_init_score=[_init_score(X_tr), _init_score(X_va)],
          callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)])
    return m

def predire(m, X):
    p = m.predict(X)
    o = _init_score(X)
    if o is not None: p = p * np.exp(o)
    return np.clip(p, 0, None)

t0 = time.time()
modele_avant = entrainer(PARAMS_BASELINE, X_tr, y_tr, X_va, y_va)
print(f"Entraine en {time.time()-t0:.1f}s | iteration {modele_avant.best_iteration_}")

pred_avant = {"Entrainement": predire(modele_avant, X_tr),
              "Validation": predire(modele_avant, X_va),
              "Test": predire(modele_avant, X_te)}
vrais = {"Entrainement": y_tr, "Validation": y_va, "Test": y_te}
perf_avant = pd.DataFrame([metriques(k, vrais[k], pred_avant[k]) for k in vrais])
strates_avant = metriques_par_strate(y_te, pred_avant["Test"])
print(perf_avant.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

















# ┌─── PARAMETRES ───┐
PARAMS_BASELINE = dict(objective="tweedie", tweedie_variance_power=1.5, metric="tweedie",
    n_estimators=4000, learning_rate=0.02, num_leaves=40, max_depth=8,
    min_child_samples=100, subsample=0.9, subsample_freq=1, colsample_bytree=0.85,
    reg_alpha=0.5, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
EARLY_STOP = 100
# └──────────────────┘

if "p_safe" in globals():
    PARAMS_BASELINE["tweedie_variance_power"] = float(p_safe)
print(f"Puissance Tweedie : {PARAMS_BASELINE['tweedie_variance_power']:.4f}")

def _init_score(X):
    if not UTILISER_OFFSET or COL_EXPOSITION not in X.columns: return None
    return np.log(X[COL_EXPOSITION].astype(float).clip(lower=1e-6)).values

def entrainer(params, X_tr, y_tr, X_va, y_va, es=EARLY_STOP):
    m = lgb.LGBMRegressor(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)],
          eval_names=["Entrainement", "Validation"], init_score=_init_score(X_tr),
          eval_init_score=[_init_score(X_tr), _init_score(X_va)],
          callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)])
    return m

def predire(m, X):
    p = m.predict(X)
    o = _init_score(X)
    if o is not None: p = p * np.exp(o)
    return np.clip(p, 0, None)

t0 = time.time()
modele_avant = entrainer(PARAMS_BASELINE, X_tr, y_tr, X_va, y_va)
print(f"Entraine en {time.time()-t0:.1f}s | iteration {modele_avant.best_iteration_}")

pred_avant = {"Entrainement": predire(modele_avant, X_tr),
              "Validation": predire(modele_avant, X_va),
              "Test": predire(modele_avant, X_te)}
vrais = {"Entrainement": y_tr, "Validation": y_va, "Test": y_te}
perf_avant = pd.DataFrame([metriques(k, vrais[k], pred_avant[k]) for k in vrais])
strates_avant = metriques_par_strate(y_te, pred_avant["Test"])
print(perf_avant.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))






















# ┌─── PARAMETRES ───┐
RELANCER_TUNING = True      # False = charge les params deja trouves sans retuner
N_SPLITS_CV     = 3
# └──────────────────┘

import optuna
from sklearn.model_selection import TimeSeriesSplit
optuna.logging.set_verbosity(optuna.logging.WARNING)

X_tune = pd.concat([X_tr, X_va], axis=0); y_tune = pd.concat([y_tr, y_va], axis=0)
print(f"Tuning sur {len(X_tune):,} lignes")

def objectif(trial):
    p = dict(objective="tweedie",
        tweedie_variance_power=trial.suggest_float("tweedie_variance_power", 1.1, 1.95),
        metric="mae", n_estimators=3000,
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 400, log=True),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        max_bin=trial.suggest_int("max_bin", 63, 511),
        random_state=SEED, n_jobs=-1, verbose=-1)
    sc, na = [], []
    for k, (i_tr, i_va) in enumerate(TimeSeriesSplit(n_splits=N_SPLITS_CV).split(X_tune)):
        m = lgb.LGBMRegressor(**p)
        m.fit(X_tune.iloc[i_tr], y_tune.iloc[i_tr],
              eval_set=[(X_tune.iloc[i_va], y_tune.iloc[i_va])], eval_metric="mae",
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        na.append(m.best_iteration_ or p["n_estimators"])
        sc.append(mean_absolute_error(y_tune.iloc[i_va],
                                      np.clip(m.predict(X_tune.iloc[i_va]), 0, None)))
        trial.report(float(np.mean(sc)), k)
        if trial.should_prune(): raise optuna.TrialPruned()
    trial.set_user_attr("n_estimators", int(np.mean(na)))
    return float(np.mean(sc))

etude = optuna.create_study(study_name=NOM_ETUDE,
    storage=f"sqlite:///{CHEMIN_DB.as_posix()}", load_if_exists=True, direction="minimize",
    sampler=optuna.samplers.TPESampler(multivariate=True, seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))

faits = len([t for t in etude.trials if t.state == optuna.trial.TrialState.COMPLETE])
print(f"Essais deja realises : {faits}")

if RELANCER_TUNING:
    t0 = time.time()
    etude.optimize(objectif, n_trials=max(0, N_TRIALS_MAX-faits),
                   timeout=BUDGET_SECONDES, show_progress_bar=True)
    print(f"Duree : {(time.time()-t0)/60:.1f} min")

best = dict(etude.best_params)
best["n_estimators"] = int(etude.best_trial.user_attrs.get("n_estimators", 2000))
best.update(objective="tweedie", metric="tweedie", subsample_freq=1,
            random_state=SEED, n_jobs=-1, verbose=-1)
with open(CHEMIN_PARAMS, "w", encoding="utf-8") as f:
    json.dump({"params": best, "mae_cv": float(etude.best_value),
               "n_trials": len(etude.trials),
               "date": datetime.now().isoformat(timespec="seconds")}, f, indent=2)
for k, v in best.items(): print(f"  {k:<26} = {v}")
print(f"\n  MAE CV = {etude.best_value:,.2f}  ->  {CHEMIN_PARAMS.name}")



















with open(CHEMIN_PARAMS, encoding="utf-8") as f: sv = json.load(f)
params_finaux = dict(sv["params"])
print(f"Tuning du {sv['date']} | {sv['n_trials']} essais | MAE CV {sv['mae_cv']:,.2f}")
params_finaux["n_estimators"] = max(int(params_finaux.get("n_estimators", 2000))*2, 500)

t0 = time.time()
modele_apres = entrainer(params_finaux, X_tr, y_tr, X_va, y_va)
print(f"Entraine en {time.time()-t0:.1f}s | iteration {modele_apres.best_iteration_}")

pred_apres = {"Entrainement": predire(modele_apres, X_tr),
              "Validation": predire(modele_apres, X_va),
              "Test": predire(modele_apres, X_te)}
perf_apres = pd.DataFrame([metriques(k, vrais[k], pred_apres[k]) for k in vrais])
strates_apres = metriques_par_strate(y_te, pred_apres["Test"])
print(perf_apres.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
























comparaison = perf_avant.merge(perf_apres, on="Ensemble", suffixes=("_avant","_apres"))
for col in ["MAE","RMSE","wMAPE_%","Gini"]:
    s = 1 if col == "Gini" else -1
    comparaison[f"Gain_{col}_%"] = s*100*(comparaison[f"{col}_apres"]-comparaison[f"{col}_avant"]) \
                                   / comparaison[f"{col}_avant"].abs()
print(comparaison[["Ensemble","MAE_avant","MAE_apres","Gain_MAE_%",
                   "Gini_avant","Gini_apres","Gain_Gini_%"]].to_string(
      index=False, float_format=lambda v: f"{v:,.3f}"))
comparaison.to_csv(CHEMIN_METRIQUES, index=False)























comparaison = perf_avant.merge(perf_apres, on="Ensemble", suffixes=("_avant","_apres"))
for col in ["MAE","RMSE","wMAPE_%","Gini"]:
    s = 1 if col == "Gini" else -1
    comparaison[f"Gain_{col}_%"] = s*100*(comparaison[f"{col}_apres"]-comparaison[f"{col}_avant"]) \
                                   / comparaison[f"{col}_avant"].abs()
print(comparaison[["Ensemble","MAE_avant","MAE_apres","Gain_MAE_%",
                   "Gini_avant","Gini_apres","Gain_Gini_%"]].to_string(
      index=False, float_format=lambda v: f"{v:,.3f}"))
comparaison.to_csv(CHEMIN_METRIQUES, index=False)

















class ModeleRBNS:
    def __init__(s, modele, features, categorielles, categories, params,
                 metriques_test=None, offset=False, col_offset=None):
        s.modele, s.features, s.categorielles = modele, list(features), list(categorielles)
        s.categories, s.params, s.metriques_test = categories, params, metriques_test
        s.offset, s.col_offset = offset, col_offset
        s.date = datetime.now().isoformat(timespec="seconds")

    def _preparer(s, X):
        manq = [c for c in s.features if c not in X.columns]
        if manq: raise KeyError(f"Colonnes absentes : {manq}")
        Xp = X[s.features].copy()
        for c in s.categorielles:
            Xp[c] = pd.Categorical(Xp[c].astype(str), categories=s.categories[c])
        return Xp

    def predict(s, X):
        p = s.modele.predict(s._preparer(X))
        if s.offset and s.col_offset in X.columns:
            p = p * np.exp(np.log(X[s.col_offset].astype(float).clip(lower=1e-6)))
        return np.clip(p, 0, None)

    def fit(s, X, y, **kw):
        s.modele = lgb.LGBMRegressor(**s.params); s.modele.fit(s._preparer(X), y, **kw)
        s.date = datetime.now().isoformat(timespec="seconds"); return s

    def importance(s, n=20):
        v = pd.Series(s.modele.booster_.feature_importance("gain"), index=s.modele.feature_name_)
        return (100*v/v.sum()).sort_values(ascending=False).head(n)

    def sauver(s, chemin): joblib.dump(s, chemin); return chemin

    @staticmethod
    def charger(chemin): return joblib.load(chemin)

    def __repr__(s):
        mae = f"{s.metriques_test['MAE']:,.0f}" if s.metriques_test else "n/a"
        return f"<ModeleRBNS | {len(s.features)} variables | MAE test {mae} | {s.date}>"

cats = {c: list(X_tr[c].cat.categories) for c in X_tr.columns if str(X_tr[c].dtype)=="category"}
paquet = ModeleRBNS(modele_apres, list(X_tr.columns), list(cats.keys()), cats,
                    params_finaux, metriques("Test", y_te, pred_apres["Test"]),
                    UTILISER_OFFSET, COL_EXPOSITION)
paquet.sauver(CHEMIN_MODELE)
print(paquet, f"\n{CHEMIN_MODELE.resolve()} ({CHEMIN_MODELE.stat().st_size/1e6:.2f} Mo)")
ecart = np.abs(ModeleRBNS.charger(CHEMIN_MODELE).predict(X_te) - pred_apres["Test"]).max()
print(f"Ecart apres rechargement : {ecart:.10f}  {'OK' if ecart < 1e-6 else '/!\\'}")





















import joblib, numpy as np, pandas as pd
# Copier la classe ModeleRBNS (BLOC 6) ici, ou : from modele_rbns import ModeleRBNS

modele = joblib.load("artefacts_modele/rbns_tweedie_v1_modele.joblib")
print(modele)
print(modele.importance(10).to_string())

y_pred = modele.predict(df_nouveau)
print(f"{len(y_pred):,} predictions | median {np.median(y_pred):,.0f} | max {y_pred.max():,.0f}")

# Reentrainement eventuel :
# modele.fit(df_nouveau[modele.features], df_nouveau[TARGET])
# modele.sauver("artefacts_modele/rbns_tweedie_v2_modele.joblib")
















import numpy as np, pandas as pd, plotly.graph_objects as go
from plotly.subplots import make_subplots

C = dict(bleu="#2E5EAA", corail="#F26B5B", teal="#17A2A2", ambre="#F5A623",
         violet="#7B5EA7", ardoise="#4A5568", vert="#2BB673", rose="#E85D9F",
         gris="#B8C2CC")
SEQ = [[0, "#2E5EAA"], [0.5, "#F5A623"], [1, "#F26B5B"]]
TPL = "plotly_white"

Y  = np.asarray(y_te, float)          # observé test
P  = np.asarray(pred_apres["Test"], float)   # prédit après tuning
P0 = np.asarray(pred_avant["Test"], float)   # prédit avant tuning
M, X = modele_apres, X_te

print(f"{len(Y):,} obs | obs médiane {np.median(Y):,.0f} | max {Y.max():,.0f}")
















n = 10
d = pd.DataFrame({"y": Y, "p": P})
d["q"] = pd.qcut(d["p"].rank(method="first"), n, labels=range(1, n+1))
g = d.groupby("q", observed=True).agg(obs=("y","mean"), pred=("p","mean"),
                                      eff=("y","size")).reset_index()

fig = go.Figure()
fig.add_bar(x=g["q"].astype(str), y=g["obs"], name="Observé",
            marker=dict(color=C["bleu"], opacity=.85),
            hovertemplate="Décile %{x}<br>Observé %{y:,.0f}<extra></extra>")
fig.add_scatter(x=g["q"].astype(str), y=g["pred"], name="Prédit (LightGBM)",
                mode="lines+markers", line=dict(color=C["corail"], width=3.5),
                marker=dict(size=11, symbol="diamond"),
                hovertemplate="Décile %{x}<br>Prédit %{y:,.0f}<extra></extra>")
fig.add_hline(y=Y.mean(), line=dict(color=C["ardoise"], dash="dot"),
              annotation_text=f"moyenne globale {Y.mean():,.0f}")
fig.update_layout(title="<b>Lift chart</b> — le modèle sépare-t-il bien les risques ?",
                  xaxis_title="Décile de prédiction (croissant)",
                  yaxis=dict(title="Montant moyen", type="log"),
                  template=TPL, height=520, bargap=.25,
                  legend=dict(orientation="h", y=1.06, x=.5, xanchor="center"))
fig.show()
print(f"Lift extrême : décile 10 / décile 1 = {g['obs'].iloc[-1]/max(g['obs'].iloc[0],1e-9):,.1f}x")

















d = pd.DataFrame({"y": Y, "a": P0, "b": P})
d["r"] = d["b"] / np.maximum(d["a"], 1e-9)
d["q"] = pd.qcut(d["r"].rank(method="first"), 10, labels=range(1, 11))
g = d.groupby("q", observed=True).agg(obs=("y","mean"), a=("a","mean"), b=("b","mean")).reset_index()
base = d["y"].mean()

fig = go.Figure()
for col, nom, coul, w, dash in [("obs","Observé",C["ardoise"],4,None),
                                ("a","Avant tuning",C["gris"],2.5,"dash"),
                                ("b","Après tuning",C["corail"],3,"dot")]:
    fig.add_scatter(x=g["q"].astype(str), y=g[col]/base, name=nom, mode="lines+markers",
                    line=dict(color=coul, width=w, dash=dash), marker=dict(size=9))
fig.update_layout(title="<b>Double lift chart</b> — la courbe la plus proche du noir gagne",
                  xaxis_title="Décile du rapport après/avant", yaxis_title="Rapport à la moyenne",
                  template=TPL, height=500,
                  legend=dict(orientation="h", y=1.06, x=.5, xanchor="center"))
fig.show()
e_a, e_b = np.abs(g["a"]-g["obs"]).mean(), np.abs(g["b"]-g["obs"]).mean()
print(f"Écart moyen à l'observé — avant {e_a:,.0f} | après {e_b:,.0f} "
      f"→ {'APRÈS gagne' if e_b < e_a else 'AVANT gagne'}")















d = pd.DataFrame({"y": Y, "p": P})
d["q"] = pd.qcut(d["p"].rank(method="first"), 10, labels=range(1, 11))
g = d.groupby("q", observed=True).agg(obs=("y","mean"), pred=("p","mean")).reset_index()
g["ratio"] = g["pred"] / g["obs"].replace(0, np.nan)

fig = go.Figure()
fig.add_bar(x=g["q"].astype(str), y=g["ratio"],
            marker=dict(color=g["ratio"], colorscale=SEQ, cmid=1,
                        line=dict(color="white", width=1.5)),
            text=g["ratio"].round(2), textposition="outside",
            hovertemplate="Décile %{x}<br>Prédit/Observé %{y:.3f}<extra></extra>")
fig.add_hline(y=1, line=dict(color=C["ardoise"], width=2.5))
fig.add_hrect(y0=.9, y1=1.1, fillcolor=C["vert"], opacity=.12, line_width=0,
              annotation_text="±10 % — zone acceptable")
fig.update_layout(title="<b>Calibration</b> — ratio prédit / observé par décile",
                  xaxis_title="Décile de prédiction", yaxis_title="Prédit / Observé",
                  template=TPL, height=480, showlegend=False)
fig.show()













fig = go.Figure()
for pred, nom, coul in [(P0,"Avant",C["gris"]), (P,"Après",C["corail"])]:
    o = np.argsort(pred); cum = np.cumsum(Y[o])/Y.sum(); x = np.linspace(0,1,len(cum))
    fig.add_scatter(x=x, y=cum, mode="lines", name=f"{nom} (Gini {indice_gini(Y,pred):.3f})",
                    line=dict(color=coul, width=3), fill="tonexty" if nom=="Après" else None,
                    fillcolor="rgba(242,107,91,.12)")
o = np.argsort(Y); fig.add_scatter(x=np.linspace(0,1,len(Y)), y=np.cumsum(Y[o])/Y.sum(),
                                   mode="lines", name="Tri parfait",
                                   line=dict(color=C["teal"], width=2, dash="dot"))
fig.add_scatter(x=[0,1], y=[0,1], mode="lines", name="Aléatoire",
                line=dict(color=C["gris"], dash="dash"))
fig.update_layout(title="<b>Courbes de Lorenz</b> — pouvoir de discrimination",
                  xaxis_title="% cumulé des observations (triées par prédiction)",
                  yaxis_title="% cumulé du montant total", template=TPL, height=560,
                  legend=dict(x=.02, y=.98, bgcolor="rgba(255,255,255,.85)"))
fig.show()














idx = np.random.default_rng(42).choice(len(Y), min(5000, len(Y)), replace=False)
m = (Y[idx] > 0) & (P[idx] > 0)
err = np.abs(np.log10(P[idx][m]) - np.log10(Y[idx][m]))

fig = go.Figure()
fig.add_scatter(x=Y[idx][m], y=P[idx][m], mode="markers",
                marker=dict(size=5, color=err, colorscale=SEQ, opacity=.55,
                            colorbar=dict(title="Erreur<br>log10", thickness=14, len=.7)),
                hovertemplate="Observé %{x:,.0f}<br>Prédit %{y:,.0f}<extra></extra>",
                name="Observations")
lim = [max(Y[Y>0].min(),1), Y.max()]
for f, dash, nom in [(1,"solid","y = x"), (2,"dash","×2 / ÷2"), (.5,"dash",None)]:
    fig.add_scatter(x=lim, y=[l*f for l in lim], mode="lines", name=nom,
                    showlegend=nom is not None,
                    line=dict(color=C["ardoise"] if f==1 else C["gris"],
                              width=2 if f==1 else 1.2, dash=dash))
fig.update_layout(title="<b>Prédit vs Observé</b> — couleur = amplitude de l'erreur",
                  xaxis=dict(title="Observé", type="log"),
                  yaxis=dict(title="Prédit", type="log"),
                  template=TPL, height=620)
fig.show()
dans2 = np.mean((P[m2 := (Y>0)&(P>0)]/Y[m2] < 2) & (P[m2]/Y[m2] > .5))
print(f"Prédictions dans un facteur 2 de l'observé : {100*dans2:.1f} %")




















m = P > 0
res_rel = (Y[m] - P[m]) / P[m]
d = pd.DataFrame({"p": P[m], "r": res_rel})
d["q"] = pd.qcut(d["p"].rank(method="first"), 20, labels=False)
g = d.groupby("q").agg(p=("p","median"), med=("r","median"),
                       q25=("r",lambda s: s.quantile(.25)),
                       q75=("r",lambda s: s.quantile(.75))).reset_index()

fig = go.Figure()
fig.add_scatter(x=g["p"], y=g["q75"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
fig.add_scatter(x=g["p"], y=g["q25"], mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor="rgba(46,94,170,.18)", name="Écart interquartile", hoverinfo="skip")
fig.add_scatter(x=g["p"], y=g["med"], mode="lines+markers", name="Résidu médian",
                line=dict(color=C["bleu"], width=3), marker=dict(size=8))
fig.add_hline(y=0, line=dict(color=C["corail"], width=2.5, dash="dash"))
fig.update_layout(title="<b>Résidus relatifs</b> — la médiane doit rester sur zéro",
                  xaxis=dict(title="Prédiction", type="log"),
                  yaxis_title="(Observé − Prédit) / Prédit",
                  template=TPL, height=500,
                  legend=dict(orientation="h", y=1.06, x=.5, xanchor="center"))
fig.show()



















imp = pd.Series(M.booster_.feature_importance("gain"), index=M.feature_name_)
imp = (100*imp/imp.sum()).sort_values(ascending=False).head(20).iloc[::-1]

fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                       marker=dict(color=imp.values, colorscale=SEQ,
                                   line=dict(color="white", width=1.2)),
                       text=[f"{v:.1f} %" for v in imp.values], textposition="outside",
                       hovertemplate="<b>%{y}</b><br>%{x:.2f} % du gain<extra></extra>"))
fig.update_layout(title="<b>Importance des variables</b> — part du gain total",
                  xaxis_title="% du gain", template=TPL, height=620, showlegend=False)
fig.show()
top3 = imp.iloc[::-1].head(3)
print(f"Top 3 = {top3.sum():.1f} % du gain : {list(top3.index)}")





















VAR_PDP = imp.index[-1]        # variable la plus importante — à changer librement
N_GRILLE, N_ECH = 30, 3000

ech = X.sample(min(N_ECH, len(X)), random_state=42)
col = ech[VAR_PDP]
if str(col.dtype) in ("category","object"):
    grille = col.value_counts().head(15).index.tolist()
else:
    grille = np.linspace(col.quantile(.01), col.quantile(.99), N_GRILLE)

vals = []
for v in grille:
    tmp = ech.copy(); tmp[VAR_PDP] = v
    vals.append(np.clip(M.predict(tmp), 0, None).mean())

fig = go.Figure()
fig.add_scatter(x=[str(v) for v in grille] if isinstance(grille, list) else grille,
                y=vals, mode="lines+markers", name="Effet moyen",
                line=dict(color=C["violet"], width=3.5), marker=dict(size=8),
                fill="tozeroy", fillcolor="rgba(123,94,167,.15)")
fig.add_hline(y=np.mean(vals), line=dict(color=C["gris"], dash="dot"))
fig.update_layout(title=f"<b>Dépendance partielle</b> — effet de <b>{VAR_PDP}</b> "
                        "toutes choses égales par ailleurs",
                  xaxis_title=VAR_PDP, yaxis_title="Prédiction moyenne",
                  template=TPL, height=480, showlegend=False)
fig.show()
print(f"Amplitude de l'effet : {max(vals)/max(min(vals),1e-9):.2f}x")






















# pip install shap
import shap
N_SHAP = 2000

Xs = X.sample(min(N_SHAP, len(X)), random_state=42)
sv = shap.TreeExplainer(M).shap_values(Xs)
sv = sv[0] if isinstance(sv, list) else sv
ordre = np.argsort(np.abs(sv).mean(0))[-15:]

fig = go.Figure()
rng = np.random.default_rng(0)
for i, j in enumerate(ordre):
    v = Xs.iloc[:, j]
    v = v.cat.codes if str(v.dtype)=="category" else v
    v = pd.to_numeric(v, errors="coerce").astype(float)
    coul = (v - np.nanpercentile(v,5)) / max(np.nanpercentile(v,95)-np.nanpercentile(v,5), 1e-9)
    fig.add_scatter(x=sv[:, j], y=i + rng.uniform(-.28,.28,len(sv)), mode="markers",
                    marker=dict(size=4.5, color=np.clip(coul,0,1), colorscale=SEQ,
                                opacity=.65, showscale=(i==len(ordre)-1),
                                colorbar=dict(title="Valeur<br>de la variable",
                                              tickvals=[0,1], ticktext=["faible","élevée"],
                                              thickness=14, len=.6)),
                    name=Xs.columns[j], showlegend=False,
                    hovertemplate=f"<b>{Xs.columns[j]}</b><br>SHAP %{{x:,.0f}}<extra></extra>")
fig.add_vline(x=0, line=dict(color=C["ardoise"], dash="dash"))
fig.update_layout(title="<b>SHAP beeswarm</b> — sens et amplitude de chaque variable",
                  xaxis_title="Contribution SHAP à la prédiction",
                  yaxis=dict(tickmode="array", tickvals=list(range(len(ordre))),
                             ticktext=[Xs.columns[j] for j in ordre]),
                  template=TPL, height=620)
fig.show()




















I_OBS = int(np.argmax(np.abs(Y - P)))   # la pire erreur du test — modifiable

Xw = X.iloc[[I_OBS]]
expl = shap.TreeExplainer(M)
svw = expl.shap_values(Xw)
svw = (svw[0] if isinstance(svw, list) else svw).ravel()
s = pd.Series(svw, index=X.columns)
s = s.reindex(s.abs().sort_values().index).tail(14)

fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                       marker=dict(color=np.where(s.values>0, C["corail"], C["bleu"]),
                                   line=dict(color="white", width=1.2)),
                       text=[f"{v:+,.0f}" for v in s.values], textposition="outside",
                       hovertemplate="<b>%{y}</b><br>%{x:+,.0f}<extra></extra>"))
fig.add_vline(x=0, line=dict(color=C["ardoise"], width=2))
fig.update_layout(title=f"<b>Décomposition SHAP</b> — observation #{I_OBS}<br>"
                        f"<sup>Observé {Y[I_OBS]:,.0f} · Prédit {P[I_OBS]:,.0f} · "
                        f"base {expl.expected_value:,.0f}</sup>",
                  xaxis_title="Contribution", template=TPL, height=560, showlegend=False)
fig.show()



















VAR_HET = imp.index[-1]        # variable à examiner

v = X[VAR_HET]
v = v.cat.codes if str(v.dtype)=="category" else pd.to_numeric(v, errors="coerce")
d = pd.DataFrame({"v": v.values, "r": Y - P}).dropna()
d["q"] = pd.qcut(d["v"].rank(method="first"), 15, labels=False, duplicates="drop")
g = d.groupby("q").agg(v=("v","median"), sd=("r","std"), n=("r","size")).reset_index()

fig = go.Figure(go.Scatter(x=g["v"], y=g["sd"], mode="lines+markers",
                           line=dict(color=C["teal"], width=3),
                           marker=dict(size=10+12*g["n"]/g["n"].max(), color=C["teal"]),
                           fill="tozeroy", fillcolor="rgba(23,162,162,.15)",
                           hovertemplate="%{x:,.2f}<br>Écart-type %{y:,.0f}<extra></extra>"))
fig.update_layout(title=f"<b>Hétéroscédasticité</b> — dispersion des résidus selon {VAR_HET}"
                        "<br><sup>Une courbe croissante confirme le choix de Tweedie</sup>",
                  xaxis_title=VAR_HET, yaxis_title="Écart-type des résidus",
                  template=TPL, height=460, showlegend=False)
fig.show()















st = metriques_par_strate(Y, P)

fig = make_subplots(rows=1, cols=2, subplot_titles=("Biais relatif", "Part du montant total"))
fig.add_bar(x=st["Strate"], y=st["Biais_%"], row=1, col=1, showlegend=False,
            marker=dict(color=np.where(st["Biais_%"]>0, C["corail"], C["bleu"]),
                        line=dict(color="white", width=1.5)),
            text=st["Biais_%"].round(1), textposition="outside")
fig.add_bar(x=st["Strate"], y=st["Part_du_total_%"], row=1, col=2, showlegend=False,
            marker=dict(color=C["violet"], line=dict(color="white", width=1.5)),
            text=st["Part_du_total_%"].round(1), textposition="outside")
fig.add_hline(y=0, line=dict(color=C["ardoise"], width=2), row=1, col=1)
fig.update_yaxes(title_text="Biais %", row=1, col=1)
fig.update_yaxes(title_text="% du total", row=1, col=2)
fig.update_layout(title="<b>Où le modèle se trompe-t-il, et sur quels enjeux ?</b>",
                  template=TPL, height=460)
fig.show()
print(st.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))


















yo, pp = Y[Y>0], P[P>0]

fig = go.Figure()
fig.add_histogram(x=np.log10(yo), nbinsx=70, name="Observé", opacity=.65,
                  marker_color=C["bleu"], histnorm="probability density")
fig.add_histogram(x=np.log10(pp), nbinsx=70, name="Prédit", opacity=.65,
                  marker_color=C["corail"], histnorm="probability density")
for v, nom, coul in [(np.median(yo),"médiane obs",C["bleu"]), (np.median(pp),"médiane préd",C["corail"])]:
    fig.add_vline(x=np.log10(v), line=dict(color=coul, dash="dash"),
                  annotation_text=f"{nom} {v:,.0f}")
fig.update_layout(title="<b>Le modèle reproduit-il la forme de la distribution ?</b>"
                        "<br><sup>Une prédiction trop concentrée signale un excès de lissage</sup>",
                  xaxis_title="log10(montant)", yaxis_title="Densité",
                  barmode="overlay", template=TPL, height=500,
                  legend=dict(orientation="h", y=1.06, x=.5, xanchor="center"))
fig.show()
print(f"Écart-type log10 — observé {np.std(np.log10(yo)):.3f} | prédit {np.std(np.log10(pp)):.3f}")




















q = np.linspace(.01, .99, 99)
qo, qp = np.quantile(Y, q), np.quantile(P, q)

fig = go.Figure()
fig.add_scatter(x=qo, y=qp, mode="markers+lines", name="Quantiles",
                marker=dict(size=7, color=q, colorscale=SEQ,
                            colorbar=dict(title="Quantile", thickness=14, len=.7)),
                line=dict(color=C["gris"], width=1),
                hovertemplate="q=%{marker.color:.2f}<br>Obs %{x:,.0f}<br>Préd %{y:,.0f}<extra></extra>")
lim = [max(qo.min(),1), qo.max()]
fig.add_scatter(x=lim, y=lim, mode="lines", name="Adéquation parfaite",
                line=dict(color=C["ardoise"], width=2, dash="dash"))
fig.update_layout(title="<b>QQ plot</b> — les quantiles prédits collent-ils aux observés ?"
                        "<br><sup>Sous la diagonale à droite = queue sous-estimée</sup>",
                  xaxis=dict(title="Quantiles observés", type="log"),
                  yaxis=dict(title="Quantiles prédits", type="log"),
                  template=TPL, height=560)
fig.show()
print(f"Ratio au quantile 99 % : {np.quantile(P,.99)/np.quantile(Y,.99):.3f}")


















res = M.evals_result_
met = list(res["Validation"].keys())[0]

fig = go.Figure()
for nom, coul in [("Entrainement", C["bleu"]), ("Validation", C["corail"])]:
    if nom in res:
        fig.add_scatter(y=res[nom][met], mode="lines", name=nom,
                        line=dict(color=coul, width=2.5))
fig.add_vline(x=M.best_iteration_, line=dict(color=C["vert"], width=2.5, dash="dash"),
              annotation_text=f"arrêt anticipé — {M.best_iteration_} arbres")
fig.update_layout(title="<b>Convergence</b> — l'écart entre les deux courbes mesure le surapprentissage",
                  xaxis_title="Itération", yaxis_title=met, template=TPL, height=460,
                  legend=dict(orientation="h", y=1.06, x=.5, xanchor="center"))
fig.show()
if "Entrainement" in res:
    e, v = res["Entrainement"][met][M.best_iteration_-1], res["Validation"][met][M.best_iteration_-1]
    print(f"Écart train/valid à l'arrêt : {100*(v-e)/max(abs(e),1e-9):+.1f} %")




















o = np.argsort(P)[::-1]
cum_y = np.cumsum(Y[o]) / Y.sum()
pct = np.arange(1, len(Y)+1) / len(Y)

fig = go.Figure()
fig.add_scatter(x=pct, y=cum_y, mode="lines", name="Modèle",
                line=dict(color=C["corail"], width=3.5),
                fill="tozeroy", fillcolor="rgba(242,107,91,.15)")
fig.add_scatter(x=[0,1], y=[0,1], mode="lines", name="Aléatoire",
                line=dict(color=C["gris"], dash="dash"))
for seuil, coul in [(.10, C["bleu"]), (.20, C["violet"])]:
    k = int(seuil*len(Y))
    fig.add_annotation(x=seuil, y=cum_y[k], ax=45, ay=-45, arrowhead=2, arrowcolor=coul,
                       text=f"<b>{100*seuil:.0f} %</b> des unités<br>= {100*cum_y[k]:.0f} % du montant",
                       font=dict(color=coul, size=11), bgcolor="rgba(255,255,255,.9)",
                       bordercolor=coul, borderwidth=1)
fig.update_layout(title="<b>Concentration du risque</b> — quel effort de contrôle pour quel montant couvert ?",
                  xaxis=dict(title="% des unités (triées par prédiction décroissante)", tickformat=".0%"),
                  yaxis=dict(title="% cumulé du montant réel", tickformat=".0%"),
                  template=TPL, height=520,
                  legend=dict(x=.6, y=.15, bgcolor="rgba(255,255,255,.85)"))
fig.show()



















AAAAAAAAAAAAAAAAAAAA



# ================================================================
# TABLEAU DE BORD — version revisee + granularite d'affichage
#   ① Couches du cercle          (empilement des anneaux)
#   ② Maille + valeur            (FILTRE : quelles anomalies retenir)
#   ③ Maille souhaitee           (AFFICHAGE : granularite des unites en bas)
#   Panneaux : cartes -> bar plot -> forest plot
# ================================================================
# ┌─────────────────── PARAMETRES A MODIFIER ───────────────────┐
TOP_N_PANNEAUX = 12       # Unites affichees dans le bar plot ET le forest plot
ECHELLE        = "Bluered"
LOG_FOREST     = True     # Echelle log sur l'axe des montants du forest plot
COL_BARPLOT    = "score_total"   # Grandeur des barres apres agregation.
                          # "score_total" = somme du groupe (defaut)
                          # "score_moyen" = gravite moyenne du groupe
N_SEGMENTS     = 4        # Nombre de selecteurs de granularite
# └──────────────────────────────────────────────────────────────┘

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

VUE_GENERALE = ""


# ------------------------------------------------------------- helpers

def _hierarchie(dd, chemin):
    lignes = []
    for prof in range(1, len(chemin) + 1):
        cols = chemin[:prof]
        agg = {"score_total": ("score_composite", "sum"),
               "score_moyen": ("score_composite", "mean"),
               "score_max":   ("score_composite", "max"),
               "n":           ("score_composite", "size")}
        if GWP_COL in dd.columns:
            agg["gwp"] = (GWP_COL, "sum")
        g = dd.groupby(cols, observed=True).agg(**agg).reset_index()
        for _, r in g.iterrows():
            vals = [str(r[c]) for c in cols]
            lignes.append({"id": "/".join(vals), "label": vals[-1],
                           "parent": "/".join(vals[:-1]) if prof > 1 else "",
                           "profondeur": prof, "score_total": r["score_total"],
                           "score_moyen": r["score_moyen"], "score_max": r["score_max"],
                           "n": int(r["n"]), "gwp": r.get("gwp", np.nan)})
    return pd.DataFrame(lignes)


def _filtrer_maille(df, colonne, valeur):
    if not valeur:
        return df, f"{colonne} — vue generale"
    return df[df[colonne].astype(str) == str(valeur)], f"{colonne} = {valeur}"


def _agreger(sub, gran, maxlen=38):
    """
    Agrege les anomalies a la granularite choisie.

    Pour chaque groupe on retient :
      - score_total / score_moyen / n  (statistiques du groupe)
      - la PIRE anomalie du groupe et son intervalle CQR REEL
        -> aucune moyenne de bornes, aucune deformation
    Si gran == toutes les ID_COLS, chaque groupe contient une seule
    anomalie : le comportement est identique a l'affichage individuel.
    """
    if sub.empty:
        return sub.assign(_label="", score_total=np.nan,
                          score_moyen=np.nan, n_groupe=0)

    stats = (sub.groupby(gran, observed=True)
               .agg(score_total=("score_composite", "sum"),
                    score_moyen=("score_composite", "mean"),
                    n_groupe=("score_composite", "size"))
               .reset_index())

    # CONDITION : ligne representative = anomalie de plus fort score du groupe
    idx_pires = sub.groupby(gran, observed=True)["score_composite"].idxmax()
    pires = sub.loc[idx_pires].copy()

    out = pires.merge(stats, on=gran, how="left")
    out["_label"] = out[gran].astype(str).agg(" | ".join, axis=1).str.slice(0, maxlen)
    return out


def _hover_groupe(r, gran):
    ident = " | ".join(f"{c}={r[c]}" for c in gran)
    t = (f"<b>{ident}</b><br>"
         f"Anomalies du groupe : {int(r['n_groupe'])}<br>"
         f"Score cumule : {r['score_total']:.4g}<br>"
         f"Gravite moyenne : {r['score_moyen']:.4g}<br>"
         "─────────────<br>"
         f"<i>Pire anomalie du groupe :</i><br>"
         f"Observe    : {r['y_obs']:,.0f}<br>"
         f"Predit     : {r['y_pred']:,.0f}<br>"
         f"Intervalle : [{r['borne_basse']:,.0f} ; {r['borne_haute']:,.0f}]")
    if "rank" in r.index and pd.notna(r["rank"]):
        t += f"<br>Rang priorite : #{int(r['rank'])}"
    if GWP_COL in r.index and pd.notna(r[GWP_COL]):
        t += f"<br>{GWP_COL} : {r[GWP_COL]:,.0f}"
    return t


def _cartes(sub, dd_global, titre, sub_expl=None, gran=None):
    part = sub["score_composite"].sum() / max(dd_global["score_composite"].sum(), 1e-12)
    gwp = (f"{sub[GWP_COL].sum():,.0f}"
           if GWP_COL in sub.columns and sub[GWP_COL].notna().any() else "n/a")
    grav = f"{sub['score_composite'].mean():,.4g}" if len(sub) else "—"
    n_grp = (sub.groupby(gran, observed=True).ngroups if gran and len(sub) else len(sub))
    pire = "—"
    if len(sub) and "rank" in sub.columns:
        rk = sub.loc[sub["score_composite"].idxmax(), "rank"]
        if pd.notna(rk):
            pire = f"#{int(rk)}"
    couv = (f"{100*sub_expl['dans_intervalle'].mean():.1f} %"
            if sub_expl is not None and len(sub_expl) else "n/a")

    cartes = [("Anomalies", f"{len(sub):,}", "#37474f"),
              ("Groupes affiches", f"{n_grp:,}", "#455a64"),
              ("Gravite moyenne", grav, "#c62828"),
              ("Score cumule", f"{sub['score_composite'].sum():,.3g}", "#1565c0"),
              ("Part du score global", f"{100*part:.1f} %", "#ad1457"),
              (f"{GWP_COL} total", gwp, "#2e7d32"),
              ("Couverture CQR", couv, "#00838f"),
              ("Pire anomalie", pire, "#6a1b9a")]
    blocs = "".join(
        f"<div style='flex:1;min-width:128px;background:#fff;border:1px solid #e0e0e0;"
        f"border-left:5px solid {c};border-radius:7px;padding:11px 13px;"
        f"box-shadow:0 1px 3px rgba(0,0,0,.07)'>"
        f"<div style='font-size:10.5px;color:#78909c;text-transform:uppercase;"
        f"letter-spacing:.6px'>{t}</div>"
        f"<div style='font-size:19px;font-weight:600;color:{c};margin-top:4px'>{v}</div>"
        f"</div>" for t, v, c in cartes)
    return HTML(f"<div style='font-family:system-ui,sans-serif;margin:6px 0 14px 0'>"
                f"<div style='font-size:15px;font-weight:600;color:#263238;"
                f"margin-bottom:10px'>📍 {titre}</div>"
                f"<div style='display:flex;gap:9px;flex-wrap:wrap'>{blocs}</div></div>")


# ------------------- panneau 1 : bar plot

def _creer_fw_bar():
    fig = go.Figure(go.Bar(x=[], y=[], orientation="h", showlegend=False,
                           marker=dict(colorscale=ECHELLE, cmin=0, cmax=1,
                                       line=dict(width=0.5, color="white"))))
    fig.update_layout(template="plotly_white", height=520,
                      yaxis=dict(tickfont=dict(size=9)),
                      margin=dict(l=10, r=40, t=95, b=45))
    return go.FigureWidget(fig)


def _maj_fw_bar(fw, sub, titre, gran, top_n=TOP_N_PANNEAUX, col=COL_BARPLOT):
    if len(sub) == 0:
        with fw.batch_update():
            fw.data[0].x, fw.data[0].y = [], []
            fw.layout.title = dict(text=f"{titre}<br><sup>Aucune anomalie</sup>",
                                   font=dict(size=14))
            fw.layout.height = 260
        return

    agg = _agreger(sub, gran)
    cle = col if col in agg.columns else "score_total"
    top = agg.nlargest(min(top_n, len(agg)), cle).iloc[::-1]

    with fw.batch_update():
        fw.data[0].x = top[cle].tolist()
        fw.data[0].y = top["_label"].tolist()
        fw.data[0].marker.color = top[cle].rank(pct=True).tolist()
        fw.data[0].text = [_hover_groupe(r, gran) for _, r in top.iterrows()]
        fw.data[0].hovertemplate = "%{text}<extra></extra>"
        fw.layout.xaxis.title.text = ("Score cumule du groupe" if cle == "score_total"
                                      else "Gravite moyenne du groupe")
        fw.layout.title = dict(
            text=f"Top {len(top)} — {titre}"
                 f"<br><sup>Granularite : {' | '.join(gran)}</sup>",
            font=dict(size=14))
        fw.layout.height = max(360, 34 * len(top) + 160)


# ------------------- panneau 2 : forest plot

def _creer_fw_forest():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                             line=dict(color="#3a6bbf", width=10), opacity=0.3,
                             name=f"Intervalle conforme ({100*(1-ALPHA):.0f} %)",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                             line=dict(color="#c0392b", width=2, dash="dot"),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name="Prediction",
                             marker=dict(symbol="diamond", size=10, color="white",
                                         line=dict(color="black", width=1.5))))
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name="Valeur comptabilisee",
                             marker=dict(size=13, color="#c0392b",
                                         line=dict(color="#7b241c", width=1.3))))
    fig.update_layout(template="plotly_white", height=520,
                      xaxis=dict(title=TARGET),
                      legend=dict(orientation="h", yanchor="bottom", y=1.03,
                                  xanchor="center", x=0.5),
                      margin=dict(l=10, r=40, t=115, b=50))
    return go.FigureWidget(fig)


def _maj_fw_forest(fw, sub, titre, gran, top_n=TOP_N_PANNEAUX,
                   log_x=LOG_FOREST, col=COL_BARPLOT):
    if len(sub) == 0:
        with fw.batch_update():
            for t in fw.data:
                t.x, t.y = [], []
            fw.layout.title = dict(text=f"{titre}<br><sup>Aucune anomalie</sup>",
                                   font=dict(size=14))
            fw.layout.height = 260
        return

    agg = _agreger(sub, gran)
    cle = col if col in agg.columns else "score_total"
    # CONDITION : memes unites que le bar plot, meme ordre
    d = agg.nlargest(min(top_n, len(agg)), cle).iloc[::-1].reset_index(drop=True)

    y = list(range(len(d)))
    lo = d["borne_basse"].values.astype(float)
    hi = d["borne_haute"].values.astype(float)
    obs = d["y_obs"].values.astype(float)
    pred = d["y_pred"].values.astype(float)
    log_ok = bool(log_x and (obs > 0).all() and (pred > 0).all() and (lo > 0).all())

    xs_band, ys_band = [], []
    for yi, l, h in zip(y, lo, hi):
        xs_band += [l, h, None]
        ys_band += [yi, yi, None]

    xs_over, ys_over = [], []
    for yi, o, l, h in zip(y, obs, lo, hi):
        cible = h if o > h else l
        xs_over += [cible, o, None]
        ys_over += [yi, yi, None]

    textes = [_hover_groupe(r, gran) for _, r in d.iterrows()]
    ticks = [f"{lab}" + (f"  ({int(n)})" if n > 1 else "")
             for lab, n in zip(d["_label"], d["n_groupe"])]

    multi = (d["n_groupe"] > 1).any()
    sous_titre = (f"Granularite : {' | '.join(gran)}"
                  + ("  —  chaque ligne montre la PIRE anomalie du groupe "
                     "(effectif entre parentheses)" if multi else ""))

    with fw.batch_update():
        fw.data[0].x, fw.data[0].y = xs_band, ys_band
        fw.data[1].x, fw.data[1].y = xs_over, ys_over
        fw.data[2].x, fw.data[2].y = pred, y
        fw.data[2].text = textes
        fw.data[2].hovertemplate = "%{text}<extra></extra>"
        fw.data[3].x, fw.data[3].y = obs, y
        fw.data[3].text = textes
        fw.data[3].hovertemplate = "%{text}<extra></extra>"
        fw.layout.xaxis.type = "log" if log_ok else "linear"
        fw.layout.xaxis.title.text = TARGET + ("  (log)" if log_ok else "")
        fw.layout.yaxis = dict(tickmode="array", tickvals=y, ticktext=ticks,
                               tickfont=dict(size=9))
        fw.layout.title = dict(
            text=f"Intervalle conforme, prediction et valeur observee — {titre}"
                 f"<br><sup>{sous_titre}</sup>", font=dict(size=14))
        fw.layout.height = max(400, 40 * len(d) + 175)


# ------------------------------------------------------------- dashboard

def dashboard_complet(anomalies_prio, expl, top_n=TOP_N_PANNEAUX):
    cols = [c for c in ID_COLS if c in anomalies_prio.columns and c in expl.columns]
    if not cols:
        print("Aucune colonne d'identification commune.")
        return

    dd = anomalies_prio.dropna(subset=["score_composite"]).copy()
    ex = expl.copy()
    for c in cols:
        dd[c] = dd[c].astype(str)
        ex[c] = ex[c].astype(str)
    score_global = dd["score_composite"].sum()
    verrou = {"actif": False}

    # ---- ① Couches du cercle ------------------------------------------
    prefs = [c for c in ["Lob", "Partner", "Companies", "Risk"] if c in cols]
    def_cercle = [prefs[0] if prefs else cols[0], None, None, None]
    niveaux = [widgets.Dropdown(
        options=[("— aucun —", None)] + [(c, c) for c in cols],
        value=def_cercle[i], description=f"Couche {i+1} :",
        layout=widgets.Layout(width="255px"),
        style={"description_width": "72px"}) for i in range(4)]

    fw_cercle = go.FigureWidget(data=[go.Sunburst(ids=[], labels=[], parents=[],
                                                  values=[], branchvalues="total")])
    fw_cercle.update_layout(template="plotly_white", height=620, margin=dict(t=110, b=20))

    # ---- ② Filtre : maille puis valeur ---------------------------------
    maille_defaut = prefs[0] if prefs else cols[0]
    sel_maille = widgets.Dropdown(
        options=[(c, c) for c in cols], value=maille_defaut,
        description="1 · Maille :", layout=widgets.Layout(width="330px"),
        style={"description_width": "90px"})
    sel_valeur = widgets.Dropdown(
        options=[("— vue generale —", VUE_GENERALE)], value=VUE_GENERALE,
        description="2 · Valeur :", layout=widgets.Layout(width="480px"),
        style={"description_width": "90px"})

    # ---- ③ Granularite d'affichage des unites --------------------------
    # CONDITION : par defaut, les 3 premieres colonnes disponibles.
    # Segment 1 est toujours renseigne -> la granularite n'est jamais vide.
    ordre_def = [c for c in ["Partner", "Companies", "Lob", "Activity", "Risk"]
                 if c in cols] or cols
    def_seg = [ordre_def[i] if i < min(3, len(ordre_def)) else None
               for i in range(N_SEGMENTS)]
    segments = [widgets.Dropdown(
        options=([(c, c) for c in cols] if i == 0
                 else [("— aucun —", None)] + [(c, c) for c in cols]),
        value=def_seg[i] or (cols[0] if i == 0 else None),
        description=f"Segment {i+1} :", layout=widgets.Layout(width="255px"),
        style={"description_width": "80px"}) for i in range(N_SEGMENTS)]

    z_cartes = widgets.Output()
    fw_bar = _creer_fw_bar()
    fw_forest = _creer_fw_forest()

    def _chemin():
        vus, out = set(), []
        for w in niveaux:
            if w.value and w.value not in vus:
                out.append(w.value)
                vus.add(w.value)
        return out

    def _granularite():
        vus, out = set(), []
        for w in segments:
            if w.value and w.value not in vus:
                out.append(w.value)
                vus.add(w.value)
        return out or [cols[0]]      # CONDITION : jamais vide

    def _maj_cercle(*_):
        chemin = _chemin()
        if not chemin:
            with fw_cercle.batch_update():
                t = fw_cercle.data[0]
                t.ids, t.labels, t.parents, t.values = [], [], [], []
                fw_cercle.layout.title = dict(text="Activez au moins une couche.",
                                              font=dict(size=15))
            return
        h = _hierarchie(dd, chemin)
        cmax = float(np.nanpercentile(h["score_moyen"], 95)) or float(h["score_moyen"].max())
        survol = [f"<b>{r['label']}</b><br>"
                  f"Gravite moyenne : {r['score_moyen']:.4g}<br>"
                  f"─────────────<br>Anomalies : {r['n']}<br>"
                  f"Score cumule : {r['score_total']:.4g} "
                  f"({100*r['score_total']/score_global:.1f} % du total)<br>"
                  f"Pire anomalie : {r['score_max']:.4g}"
                  + (f"<br>{GWP_COL} : {r['gwp']:,.0f}" if pd.notna(r["gwp"]) else "")
                  for _, r in h.iterrows()]
        with fw_cercle.batch_update():
            t = fw_cercle.data[0]
            t.ids, t.labels = h["id"].tolist(), h["label"].tolist()
            t.parents, t.values = h["parent"].tolist(), h["score_total"].tolist()
            t.text = [f"{100*v/score_global:.0f} %" for v in h["score_total"]]
            t.texttemplate = "%{label}<br>%{text}"
            t.hovertext, t.hoverinfo = survol, "text"
            t.insidetextorientation = "radial"
            t.maxdepth = len(chemin)
            t.marker = dict(colors=h["score_moyen"].tolist(), colorscale=ECHELLE,
                            cmin=0, cmax=cmax, line=dict(color="white", width=1.6),
                            colorbar=dict(title="Gravite<br>moyenne", thickness=16,
                                          len=0.7, tickformat=".2g"))
            fw_cercle.layout.title = dict(
                text=f"Repartition des {len(dd)} anomalies — {len(chemin)} couche(s) : "
                     f"{' › '.join(chemin)}"
                     "<br><sup>Taille = score cumule | Couleur = gravite moyenne "
                     "(bleu faible, rouge elevee)</sup>", font=dict(size=15))

    def _maj_panneaux(*_):
        colonne, valeur = sel_maille.value, sel_valeur.value
        gran = _granularite()
        sub, titre = _filtrer_maille(dd, colonne, valeur)
        sub_ex, _ = _filtrer_maille(ex, colonne, valeur)
        with z_cartes:
            clear_output(wait=True)
            display(_cartes(sub, dd, titre, sub_ex, gran))
        _maj_fw_bar(fw_bar, sub, titre, gran, top_n=top_n)
        _maj_fw_forest(fw_forest, sub, titre, gran, top_n=top_n)

    def _options_valeurs(colonne):
        g = (dd.groupby(colonne, observed=True)["score_composite"]
               .agg(["size", "sum"]).reset_index().sort_values("sum", ascending=False))
        return [("— vue generale —", VUE_GENERALE)] + [
            (f"{r[colonne]}  ({int(r['size'])} anomalies)", str(r[colonne]))
            for _, r in g.iterrows()]

    def _maj_valeurs(*_):
        if verrou["actif"]:
            return
        verrou["actif"] = True
        try:
            sel_valeur.options = _options_valeurs(sel_maille.value)
            sel_valeur.value = VUE_GENERALE
        finally:
            verrou["actif"] = False
        _maj_panneaux()

    def _au_clic(trace, points, state):
        if not points.point_inds:
            return
        parts = trace.ids[points.point_inds[0]].split("/")
        chemin = _chemin()
        if not parts or len(parts) > len(chemin):
            return
        colonne, valeur = chemin[len(parts) - 1], parts[-1]
        verrou["actif"] = True
        try:
            sel_maille.value = colonne
            sel_valeur.options = _options_valeurs(colonne)
            dispo = [v for _, v in sel_valeur.options]
            sel_valeur.value = valeur if valeur in dispo else VUE_GENERALE
        finally:
            verrou["actif"] = False
        _maj_panneaux()

    try:
        fw_cercle.data[0].on_click(_au_clic)
        clic = True
    except Exception:
        clic = False

    for w in niveaux:
        w.observe(lambda c: _maj_cercle() if c["name"] == "value" else None, names="value")
    sel_maille.observe(lambda c: _maj_valeurs() if c["name"] == "value" else None,
                       names="value")
    sel_valeur.observe(lambda c: (_maj_panneaux() if not verrou["actif"] else None)
                       if c["name"] == "value" else None, names="value")
    for w in segments:
        w.observe(lambda c: _maj_panneaux() if c["name"] == "value" else None,
                  names="value")

    def _bandeau(txt, fond="#eceff1", coul="#37474f"):
        return HTML(f"<div style='font-family:system-ui,sans-serif;font-size:12.5px;"
                    f"color:{coul};background:{fond};padding:9px 13px;border-radius:6px;"
                    f"margin:14px 0 8px 0'>{txt}</div>")

    display(_bandeau(
        "<b>① COUCHES DU CERCLE</b> — chaque couche activee ajoute un anneau. "
        "Couleur = gravite moyenne des anomalies du segment."))
    display(widgets.HBox(niveaux[:2]), widgets.HBox(niveaux[2:]))
    display(fw_cercle)

    display(_bandeau(
        "<b>② FILTRE</b> — quelles anomalies retenir. Maille (ligne 1) puis valeur "
        f"(ligne 2). Maille active par defaut : <b>{maille_defaut}</b>."
        + ("<br>Le clic sur un segment du cercle synchronise ces deux commandes."
           if clic else ""),
        fond="#e3f2fd", coul="#0d47a1"))
    display(sel_maille)
    display(sel_valeur)

    display(_bandeau(
        "<b>③ SELECTIONNEZ LA MAILLE SOUHAITEE</b> — granularite des unites affichees "
        "en bas des graphiques. Segment 1 est toujours actif ; ajoutez les segments 2 a 4 "
        "pour affiner. Une seule colonne (ex. Partner) regroupe toutes les anomalies "
        "de ce partenaire en une ligne.",
        fond="#f1f8e9", coul="#33691e"))
    display(widgets.HBox(segments[:2]), widgets.HBox(segments[2:]))

    display(z_cartes, fw_bar, fw_forest)

    _maj_cercle()
    _maj_valeurs()
    return {"cercle": fw_cercle, "bar": fw_bar, "forest": fw_forest,
            "maille": sel_maille, "valeur": sel_valeur, "segments": segments}


controles = dashboard_complet(anomalies_prio, expl)
















bbbbbbbbbbbbbbbbbbbbbbbb


# ================================================================
# TABLEAU DE BORD — version revisee + granularite d'affichage
#   ① Couches du cercle          (empilement des anneaux)
#   ② Maille + valeur            (FILTRE : quelles anomalies retenir)
#   ③ Maille souhaitee           (AFFICHAGE : granularite des unites en bas)
#   Panneaux : cartes -> bar plot -> forest plot
# ================================================================
# ┌─────────────────── PARAMETRES A MODIFIER ───────────────────┐
TOP_N_PANNEAUX    = 12       # Unites affichees dans le bar plot ET le forest plot
ECHELLE           = "Bluered"
LOG_FOREST        = True     # Echelle log sur l'axe des montants du forest plot
COL_BARPLOT       = "score_total"   # "score_total" = somme du groupe
                             # "score_moyen" = gravite moyenne du groupe
N_SEGMENTS        = 4        # Nombre de selecteurs de granularite
TEXTE_DANS_CERCLE = False    # False = AUCUN nom ecrit dans les segments du cercle
                             #         (tout reste lisible au survol)
                             # True  = libelle + pourcentage ecrits dedans
LABELS_AXE_BAS    = True     # True = libelles des unites sur l'axe des graphiques
                             #        du bas. False = axe nu, survol seul.
# └──────────────────────────────────────────────────────────────┘

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

VUE_GENERALE = ""


# ------------------------------------------------------------- helpers

def _hierarchie(dd, chemin):
    lignes = []
    for prof in range(1, len(chemin) + 1):
        cols = chemin[:prof]
        agg = {"score_total": ("score_composite", "sum"),
               "score_moyen": ("score_composite", "mean"),
               "score_max":   ("score_composite", "max"),
               "n":           ("score_composite", "size")}
        if GWP_COL in dd.columns:
            agg["gwp"] = (GWP_COL, "sum")
        g = dd.groupby(cols, observed=True).agg(**agg).reset_index()
        for _, r in g.iterrows():
            vals = [str(r[c]) for c in cols]
            lignes.append({"id": "/".join(vals), "label": vals[-1],
                           "parent": "/".join(vals[:-1]) if prof > 1 else "",
                           "profondeur": prof, "score_total": r["score_total"],
                           "score_moyen": r["score_moyen"], "score_max": r["score_max"],
                           "n": int(r["n"]), "gwp": r.get("gwp", np.nan)})
    return pd.DataFrame(lignes)


def _filtrer_maille(df, colonne, valeur):
    if not valeur:
        return df, f"{colonne} — vue generale"
    return df[df[colonne].astype(str) == str(valeur)], f"{colonne} = {valeur}"


def _agreger(sub, gran, maxlen=38):
    """
    Agrege les anomalies a la granularite choisie.
    Pour chaque groupe : statistiques + la PIRE anomalie et son intervalle REEL
    (aucune moyenne de bornes, aucune deformation).
    """
    if sub.empty:
        return sub.assign(_label="", score_total=np.nan,
                          score_moyen=np.nan, n_groupe=0)

    stats = (sub.groupby(gran, observed=True)
               .agg(score_total=("score_composite", "sum"),
                    score_moyen=("score_composite", "mean"),
                    n_groupe=("score_composite", "size"))
               .reset_index())

    # CONDITION : ligne representative = anomalie de plus fort score du groupe
    idx_pires = sub.groupby(gran, observed=True)["score_composite"].idxmax()
    pires = sub.loc[idx_pires].copy()

    out = pires.merge(stats, on=gran, how="left")
    out["_label"] = out[gran].astype(str).agg(" | ".join, axis=1).str.slice(0, maxlen)
    return out


def _hover_groupe(r, gran):
    ident = " | ".join(f"{c}={r[c]}" for c in gran)
    t = (f"<b>{ident}</b><br>"
         f"Anomalies du groupe : {int(r['n_groupe'])}<br>"
         f"Score cumule : {r['score_total']:.4g}<br>"
         f"Gravite moyenne : {r['score_moyen']:.4g}<br>"
         "─────────────<br>"
         f"<i>Pire anomalie du groupe :</i><br>"
         f"Observe    : {r['y_obs']:,.0f}<br>"
         f"Predit     : {r['y_pred']:,.0f}<br>"
         f"Intervalle : [{r['borne_basse']:,.0f} ; {r['borne_haute']:,.0f}]")
    if "rank" in r.index and pd.notna(r["rank"]):
        t += f"<br>Rang priorite : #{int(r['rank'])}"
    if GWP_COL in r.index and pd.notna(r[GWP_COL]):
        t += f"<br>{GWP_COL} : {r[GWP_COL]:,.0f}"
    return t


def _cartes(sub, dd_global, titre, sub_expl=None, gran=None):
    part = sub["score_composite"].sum() / max(dd_global["score_composite"].sum(), 1e-12)
    gwp = (f"{sub[GWP_COL].sum():,.0f}"
           if GWP_COL in sub.columns and sub[GWP_COL].notna().any() else "n/a")
    grav = f"{sub['score_composite'].mean():,.4g}" if len(sub) else "—"
    n_grp = (sub.groupby(gran, observed=True).ngroups if gran and len(sub) else len(sub))
    pire = "—"
    if len(sub) and "rank" in sub.columns:
        rk = sub.loc[sub["score_composite"].idxmax(), "rank"]
        if pd.notna(rk):
            pire = f"#{int(rk)}"
    couv = (f"{100*sub_expl['dans_intervalle'].mean():.1f} %"
            if sub_expl is not None and len(sub_expl) else "n/a")

    cartes = [("Anomalies", f"{len(sub):,}", "#37474f"),
              ("Groupes affiches", f"{n_grp:,}", "#455a64"),
              ("Gravite moyenne", grav, "#c62828"),
              ("Score cumule", f"{sub['score_composite'].sum():,.3g}", "#1565c0"),
              ("Part du score global", f"{100*part:.1f} %", "#ad1457"),
              (f"{GWP_COL} total", gwp, "#2e7d32"),
              ("Couverture CQR", couv, "#00838f"),
              ("Pire anomalie", pire, "#6a1b9a")]
    blocs = "".join(
        f"<div style='flex:1;min-width:128px;background:#fff;border:1px solid #e0e0e0;"
        f"border-left:5px solid {c};border-radius:7px;padding:11px 13px;"
        f"box-shadow:0 1px 3px rgba(0,0,0,.07)'>"
        f"<div style='font-size:10.5px;color:#78909c;text-transform:uppercase;"
        f"letter-spacing:.6px'>{t}</div>"
        f"<div style='font-size:19px;font-weight:600;color:{c};margin-top:4px'>{v}</div>"
        f"</div>" for t, v, c in cartes)
    return HTML(f"<div style='font-family:system-ui,sans-serif;margin:6px 0 14px 0'>"
                f"<div style='font-size:15px;font-weight:600;color:#263238;"
                f"margin-bottom:10px'>📍 {titre}</div>"
                f"<div style='display:flex;gap:9px;flex-wrap:wrap'>{blocs}</div></div>")


# ------------------- panneau 1 : bar plot

def _creer_fw_bar():
    fig = go.Figure(go.Bar(x=[], y=[], orientation="h", showlegend=False,
                           marker=dict(colorscale=ECHELLE, cmin=0, cmax=1,
                                       line=dict(width=0.5, color="white"))))
    fig.update_layout(template="plotly_white", height=520,
                      yaxis=dict(tickfont=dict(size=9)),
                      margin=dict(l=10, r=40, t=95, b=45))
    return go.FigureWidget(fig)


def _maj_fw_bar(fw, sub, titre, gran, top_n=TOP_N_PANNEAUX, col=COL_BARPLOT):
    if len(sub) == 0:
        with fw.batch_update():
            fw.data[0].x, fw.data[0].y = [], []
            fw.layout.title = dict(text=f"{titre}<br><sup>Aucune anomalie</sup>",
                                   font=dict(size=14))
            fw.layout.height = 260
        return

    agg = _agreger(sub, gran)
    cle = col if col in agg.columns else "score_total"
    top = agg.nlargest(min(top_n, len(agg)), cle).iloc[::-1]

    with fw.batch_update():
        fw.data[0].x = top[cle].tolist()
        fw.data[0].y = top["_label"].tolist()
        fw.data[0].marker.color = top[cle].rank(pct=True).tolist()
        fw.data[0].text = [_hover_groupe(r, gran) for _, r in top.iterrows()]
        fw.data[0].hovertemplate = "%{text}<extra></extra>"
        # CONDITION : aucun texte ecrit sur les barres elles-memes
        fw.data[0].textposition = "none"
        fw.layout.xaxis.title.text = ("Score cumule du groupe" if cle == "score_total"
                                      else "Gravite moyenne du groupe")
        fw.layout.yaxis.showticklabels = LABELS_AXE_BAS
        fw.layout.title = dict(
            text=f"Top {len(top)} — {titre}"
                 f"<br><sup>Granularite : {' | '.join(gran)}</sup>",
            font=dict(size=14))
        fw.layout.height = max(360, 34 * len(top) + 160)


# ------------------- panneau 2 : forest plot

def _creer_fw_forest():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                             line=dict(color="#3a6bbf", width=10), opacity=0.3,
                             name=f"Intervalle conforme ({100*(1-ALPHA):.0f} %)",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                             line=dict(color="#c0392b", width=2, dash="dot"),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name="Prediction",
                             marker=dict(symbol="diamond", size=10, color="white",
                                         line=dict(color="black", width=1.5))))
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name="Valeur comptabilisee",
                             marker=dict(size=13, color="#c0392b",
                                         line=dict(color="#7b241c", width=1.3))))
    fig.update_layout(template="plotly_white", height=520,
                      xaxis=dict(title=TARGET),
                      legend=dict(orientation="h", yanchor="bottom", y=1.03,
                                  xanchor="center", x=0.5),
                      margin=dict(l=10, r=40, t=115, b=50))
    return go.FigureWidget(fig)


def _maj_fw_forest(fw, sub, titre, gran, top_n=TOP_N_PANNEAUX,
                   log_x=LOG_FOREST, col=COL_BARPLOT):
    if len(sub) == 0:
        with fw.batch_update():
            for t in fw.data:
                t.x, t.y = [], []
            fw.layout.title = dict(text=f"{titre}<br><sup>Aucune anomalie</sup>",
                                   font=dict(size=14))
            fw.layout.height = 260
        return

    agg = _agreger(sub, gran)
    cle = col if col in agg.columns else "score_total"
    d = agg.nlargest(min(top_n, len(agg)), cle).iloc[::-1].reset_index(drop=True)

    y = list(range(len(d)))
    lo = d["borne_basse"].values.astype(float)
    hi = d["borne_haute"].values.astype(float)
    obs = d["y_obs"].values.astype(float)
    pred = d["y_pred"].values.astype(float)
    log_ok = bool(log_x and (obs > 0).all() and (pred > 0).all() and (lo > 0).all())

    xs_band, ys_band = [], []
    for yi, l, h in zip(y, lo, hi):
        xs_band += [l, h, None]
        ys_band += [yi, yi, None]

    xs_over, ys_over = [], []
    for yi, o, l, h in zip(y, obs, lo, hi):
        cible = h if o > h else l
        xs_over += [cible, o, None]
        ys_over += [yi, yi, None]

    textes = [_hover_groupe(r, gran) for _, r in d.iterrows()]
    ticks = [f"{lab}" + (f"  ({int(n)})" if n > 1 else "")
             for lab, n in zip(d["_label"], d["n_groupe"])]

    multi = (d["n_groupe"] > 1).any()
    sous_titre = (f"Granularite : {' | '.join(gran)}"
                  + ("  —  chaque ligne montre la PIRE anomalie du groupe "
                     "(effectif entre parentheses)" if multi else ""))

    with fw.batch_update():
        fw.data[0].x, fw.data[0].y = xs_band, ys_band
        fw.data[1].x, fw.data[1].y = xs_over, ys_over
        fw.data[2].x, fw.data[2].y = pred, y
        fw.data[2].text = textes
        fw.data[2].hovertemplate = "%{text}<extra></extra>"
        fw.data[3].x, fw.data[3].y = obs, y
        fw.data[3].text = textes
        fw.data[3].hovertemplate = "%{text}<extra></extra>"
        fw.layout.xaxis.type = "log" if log_ok else "linear"
        fw.layout.xaxis.title.text = TARGET + ("  (log)" if log_ok else "")
        fw.layout.yaxis = dict(tickmode="array", tickvals=y, ticktext=ticks,
                               tickfont=dict(size=9),
                               showticklabels=LABELS_AXE_BAS)
        fw.layout.title = dict(
            text=f"Intervalle conforme, prediction et valeur observee — {titre}"
                 f"<br><sup>{sous_titre}</sup>", font=dict(size=14))
        fw.layout.height = max(400, 40 * len(d) + 175)


# ------------------------------------------------------------- dashboard

def dashboard_complet(anomalies_prio, expl, top_n=TOP_N_PANNEAUX):
    cols = [c for c in ID_COLS if c in anomalies_prio.columns and c in expl.columns]
    if not cols:
        print("Aucune colonne d'identification commune.")
        return

    dd = anomalies_prio.dropna(subset=["score_composite"]).copy()
    ex = expl.copy()
    for c in cols:
        dd[c] = dd[c].astype(str)
        ex[c] = ex[c].astype(str)
    score_global = dd["score_composite"].sum()
    verrou = {"actif": False}

    # ---- ① Couches du cercle ------------------------------------------
    prefs = [c for c in ["Lob", "Partner", "Companies", "Risk"] if c in cols]
    def_cercle = [prefs[0] if prefs else cols[0], None, None, None]
    niveaux = [widgets.Dropdown(
        options=[("— aucun —", None)] + [(c, c) for c in cols],
        value=def_cercle[i], description=f"Couche {i+1} :",
        layout=widgets.Layout(width="255px"),
        style={"description_width": "72px"}) for i in range(4)]

    fw_cercle = go.FigureWidget(data=[go.Sunburst(ids=[], labels=[], parents=[],
                                                  values=[], branchvalues="total")])
    fw_cercle.update_layout(template="plotly_white", height=620, margin=dict(t=110, b=20))

    # ---- ② Filtre : maille puis valeur ---------------------------------
    maille_defaut = prefs[0] if prefs else cols[0]
    sel_maille = widgets.Dropdown(
        options=[(c, c) for c in cols], value=maille_defaut,
        description="1 · Maille :", layout=widgets.Layout(width="330px"),
        style={"description_width": "90px"})
    sel_valeur = widgets.Dropdown(
        options=[("— vue generale —", VUE_GENERALE)], value=VUE_GENERALE,
        description="2 · Valeur :", layout=widgets.Layout(width="480px"),
        style={"description_width": "90px"})

    # ---- ③ Granularite d'affichage des unites --------------------------
    ordre_def = [c for c in ["Partner", "Companies", "Lob", "Activity", "Risk"]
                 if c in cols] or cols
    def_seg = [ordre_def[i] if i < min(3, len(ordre_def)) else None
               for i in range(N_SEGMENTS)]
    segments = [widgets.Dropdown(
        options=([(c, c) for c in cols] if i == 0
                 else [("— aucun —", None)] + [(c, c) for c in cols]),
        value=def_seg[i] or (cols[0] if i == 0 else None),
        description=f"Segment {i+1} :", layout=widgets.Layout(width="255px"),
        style={"description_width": "80px"}) for i in range(N_SEGMENTS)]

    z_cartes = widgets.Output()
    fw_bar = _creer_fw_bar()
    fw_forest = _creer_fw_forest()

    def _chemin():
        vus, out = set(), []
        for w in niveaux:
            if w.value and w.value not in vus:
                out.append(w.value)
                vus.add(w.value)
        return out

    def _granularite():
        vus, out = set(), []
        for w in segments:
            if w.value and w.value not in vus:
                out.append(w.value)
                vus.add(w.value)
        return out or [cols[0]]      # CONDITION : jamais vide

    def _maj_cercle(*_):
        chemin = _chemin()
        if not chemin:
            with fw_cercle.batch_update():
                t = fw_cercle.data[0]
                t.ids, t.labels, t.parents, t.values = [], [], [], []
                fw_cercle.layout.title = dict(text="Activez au moins une couche.",
                                              font=dict(size=15))
            return
        h = _hierarchie(dd, chemin)
        cmax = float(np.nanpercentile(h["score_moyen"], 95)) or float(h["score_moyen"].max())
        survol = [f"<b>{r['label']}</b><br>"
                  f"Gravite moyenne : {r['score_moyen']:.4g}<br>"
                  f"─────────────<br>Anomalies : {r['n']}<br>"
                  f"Score cumule : {r['score_total']:.4g} "
                  f"({100*r['score_total']/score_global:.1f} % du total)<br>"
                  f"Pire anomalie : {r['score_max']:.4g}"
                  + (f"<br>{GWP_COL} : {r['gwp']:,.0f}" if pd.notna(r["gwp"]) else "")
                  for _, r in h.iterrows()]
        with fw_cercle.batch_update():
            t = fw_cercle.data[0]
            t.ids, t.labels = h["id"].tolist(), h["label"].tolist()
            t.parents, t.values = h["parent"].tolist(), h["score_total"].tolist()
            t.hovertext, t.hoverinfo = survol, "text"
            t.maxdepth = len(chemin)
            # CONDITION : texte dans les segments seulement si demande
            if TEXTE_DANS_CERCLE:
                t.text = [f"{100*v/score_global:.0f} %" for v in h["score_total"]]
                t.texttemplate = "%{label}<br>%{text}"
                t.textinfo = None
                t.insidetextorientation = "radial"
            else:
                t.text = None
                t.texttemplate = None
                t.textinfo = "none"
            t.marker = dict(colors=h["score_moyen"].tolist(), colorscale=ECHELLE,
                            cmin=0, cmax=cmax, line=dict(color="white", width=1.6),
                            colorbar=dict(title="Gravite<br>moyenne", thickness=16,
                                          len=0.7, tickformat=".2g"))
            fw_cercle.layout.title = dict(
                text=f"Repartition des {len(dd)} anomalies — {len(chemin)} couche(s) : "
                     f"{' › '.join(chemin)}"
                     "<br><sup>Taille = score cumule | Couleur = gravite moyenne "
                     "(bleu faible, rouge elevee) | survolez pour le detail</sup>",
                font=dict(size=15))

    def _maj_panneaux(*_):
        colonne, valeur = sel_maille.value, sel_valeur.value
        gran = _granularite()
        sub, titre = _filtrer_maille(dd, colonne, valeur)
        sub_ex, _ = _filtrer_maille(ex, colonne, valeur)
        with z_cartes:
            clear_output(wait=True)
            display(_cartes(sub, dd, titre, sub_ex, gran))
        _maj_fw_bar(fw_bar, sub, titre, gran, top_n=top_n)
        _maj_fw_forest(fw_forest, sub, titre, gran, top_n=top_n)

    def _options_valeurs(colonne):
        g = (dd.groupby(colonne, observed=True)["score_composite"]
               .agg(["size", "sum"]).reset_index().sort_values("sum", ascending=False))
        return [("— vue generale —", VUE_GENERALE)] + [
            (f"{r[colonne]}  ({int(r['size'])} anomalies)", str(r[colonne]))
            for _, r in g.iterrows()]

    def _maj_valeurs(*_):
        if verrou["actif"]:
            return
        verrou["actif"] = True
        try:
            sel_valeur.options = _options_valeurs(sel_maille.value)
            sel_valeur.value = VUE_GENERALE
        finally:
            verrou["actif"] = False
        _maj_panneaux()

    def _au_clic(trace, points, state):
        if not points.point_inds:
            return
        parts = trace.ids[points.point_inds[0]].split("/")
        chemin = _chemin()
        if not parts or len(parts) > len(chemin):
            return
        colonne, valeur = chemin[len(parts) - 1], parts[-1]
        verrou["actif"] = True
        try:
            sel_maille.value = colonne
            sel_valeur.options = _options_valeurs(colonne)
            dispo = [v for _, v in sel_valeur.options]
            sel_valeur.value = valeur if valeur in dispo else VUE_GENERALE
        finally:
            verrou["actif"] = False
        _maj_panneaux()

    try:
        fw_cercle.data[0].on_click(_au_clic)
        clic = True
    except Exception:
        clic = False

    for w in niveaux:
        w.observe(lambda c: _maj_cercle() if c["name"] == "value" else None, names="value")
    sel_maille.observe(lambda c: _maj_valeurs() if c["name"] == "value" else None,
                       names="value")
    sel_valeur.observe(lambda c: (_maj_panneaux() if not verrou["actif"] else None)
                       if c["name"] == "value" else None, names="value")
    for w in segments:
        w.observe(lambda c: _maj_panneaux() if c["name"] == "value" else None,
                  names="value")

    def _bandeau(txt, fond="#eceff1", coul="#37474f"):
        return HTML(f"<div style='font-family:system-ui,sans-serif;font-size:12.5px;"
                    f"color:{coul};background:{fond};padding:9px 13px;border-radius:6px;"
                    f"margin:14px 0 8px 0'>{txt}</div>")

    display(_bandeau(
        "<b>① COUCHES DU CERCLE</b> — chaque couche activee ajoute un anneau. "
        "Couleur = gravite moyenne des anomalies du segment. "
        "Survolez un segment pour son identite et ses statistiques."))
    display(widgets.HBox(niveaux[:2]), widgets.HBox(niveaux[2:]))
    display(fw_cercle)

    display(_bandeau(
        "<b>② FILTRE</b> — quelles anomalies retenir. Maille (ligne 1) puis valeur "
        f"(ligne 2). Maille active par defaut : <b>{maille_defaut}</b>."
        + ("<br>Le clic sur un segment du cercle synchronise ces deux commandes."
           if clic else ""),
        fond="#e3f2fd", coul="#0d47a1"))
    display(sel_maille)
    display(sel_valeur)

    display(_bandeau(
        "<b>③ SELECTIONNEZ LA MAILLE SOUHAITEE</b> — granularite des unites affichees "
        "en bas des graphiques. Segment 1 est toujours actif ; ajoutez les segments 2 a 4 "
        "pour affiner. Une seule colonne (ex. Partner) regroupe toutes les anomalies "
        "de ce partenaire en une ligne.",
        fond="#f1f8e9", coul="#33691e"))
    display(widgets.HBox(segments[:2]), widgets.HBox(segments[2:]))

    display(z_cartes, fw_bar, fw_forest)

    _maj_cercle()
    _maj_valeurs()
    return {"cercle": fw_cercle, "bar": fw_bar, "forest": fw_forest,
            "maille": sel_maille, "valeur": sel_valeur, "segments": segments}


controles = dashboard_complet(anomalies_prio, expl)












