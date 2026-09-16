Bloc 0


# Avant (manque le nom)
resultats = metriques(y_true_final, y_pred_final)

# Après
resultats = metriques(f"{TARGET_MODE}_{FEATURE_MODE}", y_true_final, y_pred_final)





# Avant
mae_test_val = float(metriques(y_true_final, y_pred_final)['MAE'])

# Après
mae_test_val = float(metriques("test", y_true_final, y_pred_final)['MAE'])




# ── Switchs de configuration ──────────────────────────────────────────
TARGET_MODE  = "cumul"   # "cumul" ou "dec"
FEATURE_MODE = "C"       # "A", "B" ou "C"

# Nombre de lags souhaités
N_LAGS_TARGET  = 4   # lags sur la target (et dec_target)
N_LAGS_VARS    = 2   # lags sur les variables explicatives










Bmoc 1

# ── Bloc 1 : Préparation des features selon FEATURE_MODE ─────────────

# Variables explicatives importantes (remplace par les vrais noms)
VAR_EXPLIQUE_1 = "nom_variable_1"
VAR_EXPLIQUE_2 = "nom_variable_2"
VAR_EXPLIQUE_3 = "nom_variable_3"
VAR_EXPLIQUE_4 = "nom_variable_4"
VAR_EXPLIQUE_5 = "nom_variable_5"

vars_expliques = [VAR_EXPLIQUE_1, VAR_EXPLIQUE_2, VAR_EXPLIQUE_3,
                  VAR_EXPLIQUE_4, VAR_EXPLIQUE_5]

# Target du modèle selon TARGET_MODE
if TARGET_MODE == "cumul":
    MODEL_TARGET = TARGET
elif TARGET_MODE == "dec":
    MODEL_TARGET = "dec_" + TARGET

# Lags de la target selon TARGET_MODE et N_LAGS_TARGET
cols_lag_target = [TARGET, "dec_" + TARGET] if TARGET_MODE == "cumul" \
                  else ["dec_" + TARGET]

lags_target = [f"{col}_lag_{i}"
               for col in cols_lag_target
               for i in range(1, N_LAGS_TARGET + 1)
               if f"{col}_lag_{i}" in df.columns]

# Lags des variables explicatives selon N_LAGS_VARS
lags_vars = [f"{v}_lag_{i}"
             for v in vars_expliques
             for i in range(1, N_LAGS_VARS + 1)
             if f"{v}_lag_{i}" in df.columns]

tous_les_lags = lags_target + lags_vars

# Features courantes (sans lags)
features_courantes = [c for c in FEATURES if "_lag_" not in c]

# Construction selon FEATURE_MODE
if FEATURE_MODE == "A":
    FEATURES_MODEL = features_courantes
elif FEATURE_MODE == "B":
    FEATURES_MODEL = tous_les_lags
elif FEATURE_MODE == "C":
    FEATURES_MODEL = features_courantes + tous_les_lags

print(f"TARGET_MODE  : {TARGET_MODE}  → MODEL_TARGET = {MODEL_TARGET}")
print(f"FEATURE_MODE : {FEATURE_MODE}")
print(f"  features courantes : {len(features_courantes)}")
print(f"  lags target        : {len(lags_target)}")
print(f"  lags var_expliques : {len(lags_vars)}")
print(f"  FEATURES_MODEL     : {len(FEATURES_MODEL)} features au total")






Bloc 2 — Entraînement

X_tr, y_tr = df_train[FEATURES_MODEL], df_train[MODEL_TARGET]
X_va, y_va = df_valid[FEATURES_MODEL], df_valid[MODEL_TARGET]
X_te, y_te = df_test[FEATURES_MODEL],  df_test[MODEL_TARGET]





Bloc 3 


# ── Bloc 3 : Reconstruction du cumulé si TARGET_MODE = "dec" ─────────

if TARGET_MODE == "dec":
    pred_test_cumulé = compute_cumulative_prediction(
        df_test,
        id_vars=ID_COLS,
        target_col=TARGET,
        pred_col="dec_" + TARGET,
        year_col='year',
        time_col=TIME_COL
    )
    y_pred_final = pred_test_cumulé
    y_true_final = df_test.loc[idx_test, TARGET].values

elif TARGET_MODE == "cumul":
    y_pred_final = pred_test
    y_true_final = y_te.values

print(f"TARGET_MODE  : {TARGET_MODE}")
print(f"Prédictions  : {len(y_pred_final)} observations")










# ── Bloc 4 : Évaluation ───────────────────────────────────────────────

resultats = metriques(y_true_final, y_pred_final)
resultats["TARGET_MODE"]  = TARGET_MODE
resultats["FEATURE_MODE"] = FEATURE_MODE

print(f"=== Résultats  TARGET={TARGET_MODE}  FEATURES={FEATURE_MODE} ===")
for k, v in resultats.items():
    print(f"  {k} : {v}")









# ── Bloc 5 : Comparaison des 6 combinaisons ──────────────────────────
import pandas as pd

resultats_comparaison = []

for t_mode in ["cumul", "dec"]:
    for f_mode in ["A", "B", "C"]:

        # -- Mise à jour des switchs
        TARGET_MODE  = t_mode
        FEATURE_MODE = f_mode

        # -- Reconstruction FEATURES_MODEL et MODEL_TARGET
        if TARGET_MODE == "cumul":
            MODEL_TARGET = TARGET
            cols_lag_target = [TARGET, "dec_" + TARGET]
        else:
            MODEL_TARGET = "dec_" + TARGET
            cols_lag_target = ["dec_" + TARGET]

        lags_target = [f"{col}_lag_{i}"
                       for col in cols_lag_target
                       for i in range(1, N_LAGS_TARGET + 1)
                       if f"{col}_lag_{i}" in df.columns]
        lags_vars   = [f"{v}_lag_{i}"
                       for v in vars_expliques
                       for i in range(1, N_LAGS_VARS + 1)
                       if f"{v}_lag_{i}" in df.columns]
        tous_les_lags     = lags_target + lags_vars
        features_courantes = [c for c in FEATURES if "_lag_" not in c]

        if FEATURE_MODE == "A":
            FEATURES_MODEL = features_courantes
        elif FEATURE_MODE == "B":
            FEATURES_MODEL = tous_les_lags
        elif FEATURE_MODE == "C":
            FEATURES_MODEL = features_courantes + tous_les_lags

        # -- Entraînement
        X_tr = df_train[FEATURES_MODEL]; y_tr = df_train[MODEL_TARGET]
        X_va = df_valid[FEATURES_MODEL]; y_va = df_valid[MODEL_TARGET]
        X_te = df_test[FEATURES_MODEL];  y_te = df_test[MODEL_TARGET]

        modele.entrainer(X_tr, y_tr, X_va, y_va)
        pred_test = modele.predict(X_te)
        idx_test  = X_te.index

        # -- Reconstruction si dec
        if TARGET_MODE == "dec":
            y_pred_final = compute_cumulative_prediction(
                df_test, id_vars=ID_COLS, target_col=TARGET,
                pred_col="dec_" + TARGET, year_col='year',
                time_col=TIME_COL)
            y_true_final = df_test.loc[idx_test, TARGET].values
        else:
            y_pred_final = pred_test
            y_true_final = y_te.values

        # -- Métriques
        res = metriques(y_true_final, y_pred_final)
        res["TARGET_MODE"]  = t_mode
        res["FEATURE_MODE"] = f_mode
        resultats_comparaison.append(res)
        print(f"  {t_mode} / {f_mode} → MAE={res['MAE']:,.1f}")

# -- Tableau final
df_comparaison = pd.DataFrame(resultats_comparaison)
df_comparaison = df_comparaison.sort_values("MAE").reset_index(drop=True)
print("\n=== Classement des 6 combinaisons (MAE croissante) ===")
print(df_comparaison[["TARGET_MODE","FEATURE_MODE",
                       "MAE","RMSE","wMAPE"]].to_string(index=False))











# ── Bloc 5 : Comparaison des 6 combinaisons ──────────────────────────
import pandas as pd

resultats_comparaison = []

for t_mode in ["cumul", "dec"]:
    for f_mode in ["A", "B", "C"]:

        # -- Mise à jour des switchs
        TARGET_MODE  = t_mode
        FEATURE_MODE = f_mode

        # -- Reconstruction FEATURES_MODEL et MODEL_TARGET
        if TARGET_MODE == "cumul":
            MODEL_TARGET = TARGET
            cols_lag_target = [TARGET, "dec_" + TARGET]
        else:
            MODEL_TARGET = "dec_" + TARGET
            cols_lag_target = ["dec_" + TARGET]

        lags_target = [f"{col}_lag_{i}"
                       for col in cols_lag_target
                       for i in range(1, N_LAGS_TARGET + 1)
                       if f"{col}_lag_{i}" in df.columns]
        lags_vars   = [f"{v}_lag_{i}"
                       for v in vars_expliques
                       for i in range(1, N_LAGS_VARS + 1)
                       if f"{v}_lag_{i}" in df.columns]
        tous_les_lags     = lags_target + lags_vars
        features_courantes = [c for c in FEATURES if "_lag_" not in c]

        if FEATURE_MODE == "A":
            FEATURES_MODEL = features_courantes
        elif FEATURE_MODE == "B":
            FEATURES_MODEL = tous_les_lags
        elif FEATURE_MODE == "C":
            FEATURES_MODEL = features_courantes + tous_les_lags

        # -- Entraînement
        X_tr = df_train[FEATURES_MODEL]; y_tr = df_train[MODEL_TARGET]
        X_va = df_valid[FEATURES_MODEL]; y_va = df_valid[MODEL_TARGET]
        X_te = df_test[FEATURES_MODEL];  y_te = df_test[MODEL_TARGET]

        modele.entrainer(X_tr, y_tr, X_va, y_va)
        pred_test = modele.predict(X_te)
        idx_test  = X_te.index

        # -- Reconstruction si dec
        if TARGET_MODE == "dec":
            y_pred_final = compute_cumulative_prediction(
                df_test, id_vars=ID_COLS, target_col=TARGET,
                pred_col="dec_" + TARGET, year_col='year',
                time_col=TIME_COL)
            y_true_final = df_test.loc[idx_test, TARGET].values
        else:
            y_pred_final = pred_test
            y_true_final = y_te.values

        # -- Métriques
        res = metriques(y_true_final, y_pred_final)
        res["TARGET_MODE"]  = t_mode
        res["FEATURE_MODE"] = f_mode
        resultats_comparaison.append(res)
        print(f"  {t_mode} / {f_mode} → MAE={res['MAE']:,.1f}")

# -- Tableau final
df_comparaison = pd.DataFrame(resultats_comparaison)
df_comparaison = df_comparaison.sort_values("MAE").reset_index(drop=True)
print("\n=== Classement des 6 combinaisons (MAE croissante) ===")
print(df_comparaison[["TARGET_MODE","FEATURE_MODE",
                       "MAE","RMSE","wMAPE"]].to_string(index=False))










# ── Bloc 6 : Barres comparatives des métriques ───────────────────────
couleurs = px.colors.qualitative.Bold

fig1 = make_subplots(rows=1, cols=3,
                     subplot_titles=["MAE","RMSE","wMAPE"])

for idx, metrique in enumerate(["MAE","RMSE","wMAPE"]):
    fig1.add_trace(
        go.Bar(
            x=df_comparaison["Combinaison"],
            y=df_comparaison[metrique],
            marker_color=couleurs[:6],
            showlegend=False
        ),
        row=1, col=idx+1
    )

fig1.update_layout(
    title="Comparaison des 6 combinaisons — Métriques",
    height=450,
    plot_bgcolor="#F8F9FA",
    paper_bgcolor="#F8F9FA"
)
fig1.show()













# ── Bloc 7 : Radar — performance normalisée ───────────────────────────
from sklearn.preprocessing import MinMaxScaler
import numpy as np

metriques_cols = ["MAE","RMSE","wMAPE"]
df_radar      = df_comparaison[metriques_cols].copy()
scaler        = MinMaxScaler()
df_radar_norm = pd.DataFrame(
    scaler.fit_transform(df_radar),
    columns=metriques_cols
)

fig2 = go.Figure()
for i, row in df_comparaison.iterrows():
    valeurs  = df_radar_norm.iloc[i].tolist()
    valeurs += [valeurs[0]]
    fig2.add_trace(go.Scatterpolar(
        r=valeurs,
        theta=metriques_cols + [metriques_cols[0]],
        name=row["Combinaison"],
        line=dict(color=couleurs[i], width=2),
        fill='toself',
        opacity=0.4
    ))

fig2.update_layout(
    title="Radar — Performance normalisée (plus petit = meilleur)",
    polar=dict(radialaxis=dict(visible=True, range=[0,1])),
    height=500,
    paper_bgcolor="#F8F9FA"
)
fig2.show()














# ── Bloc 8 : Réel vs Prédit — meilleure combinaison ──────────────────
meilleur = df_comparaison.iloc[0]
print(f"Meilleure combinaison : {meilleur['Combinaison']}")

TARGET_MODE  = meilleur["TARGET_MODE"]
FEATURE_MODE = meilleur["FEATURE_MODE"]

if TARGET_MODE == "cumul":
    MODEL_TARGET    = TARGET
    cols_lag_target = [TARGET, "dec_" + TARGET]
else:
    MODEL_TARGET    = "dec_" + TARGET
    cols_lag_target = ["dec_" + TARGET]

lags_target = [f"{col}_lag_{i}"
               for col in cols_lag_target
               for i in range(1, N_LAGS_TARGET + 1)
               if f"{col}_lag_{i}" in df.columns]
lags_vars   = [f"{v}_lag_{i}"
               for v in vars_expliques
               for i in range(1, N_LAGS_VARS + 1)
               if f"{v}_lag_{i}" in df.columns]
tous_les_lags      = lags_target + lags_vars
features_courantes = [c for c in FEATURES if "_lag_" not in c]

if FEATURE_MODE == "A":
    FEATURES_MODEL = features_courantes
elif FEATURE_MODE == "B":
    FEATURES_MODEL = tous_les_lags
elif FEATURE_MODE == "C":
    FEATURES_MODEL = features_courantes + tous_les_lags

X_te = df_test[FEATURES_MODEL]
y_te = df_test[MODEL_TARGET]
modele.entrainer(
    df_train[FEATURES_MODEL], df_train[MODEL_TARGET],
    df_valid[FEATURES_MODEL], df_valid[MODEL_TARGET]
)
pred_test = modele.predict(X_te)
idx_test  = X_te.index

if TARGET_MODE == "dec":
    y_pred_final = compute_cumulative_prediction(
        df_test, id_vars=ID_COLS, target_col=TARGET,
        pred_col="dec_" + TARGET, year_col='year',
        time_col=TIME_COL)
    y_true_final = df_test.loc[idx_test, TARGET].values
else:
    y_pred_final = pred_test
    y_true_final = y_te.values

indices = list(range(len(y_true_final)))
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=indices, y=y_true_final,
    mode='markers', name='Réel',
    marker=dict(color='#2E86AB', size=5, opacity=0.7)
))
fig3.add_trace(go.Scatter(
    x=indices, y=y_pred_final,
    mode='markers', name='Prédit',
    marker=dict(color='#E84855', size=5, opacity=0.7)
))
fig3.update_layout(
    title=f"Réel vs Prédit — {meilleur['Combinaison']}",
    xaxis_title="Observations",
    yaxis_title="Claims_incurred",
    height=450,
    plot_bgcolor="#F8F9FA",
    paper_bgcolor="#F8F9FA"
)
fig3.show()












Enregistrement du modèle


# Actuellement (bug)
return np.clip(p, 0, None)

# Correct (correction C4 qu'on avait appliquée)
return p if CLIP_MIN is None else np.clip(p, CLIP_MIN, None)







mae_test_val = float(metriques(y_true_final, y_pred_final)['MAE'])












paquet = ModeleRBNS.charger("chemin/vers/claim_incurred.joblib")
predictions = paquet.predict(X_nouveau)






Voci la parie CQR et tous 



# Actuellement (bug)
return np.clip(p, 0, None)

# Correct
return p if CLIP_MIN is None else np.clip(p, CLIP_MIN, None)






predict_test = paquet.predict(paquet.X_test)






import joblib, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
import lightgbm as lgb

class ModeleRBNS:
    def __init__(self, modele, features, categorielles, categories, params,
                 clip_min=None, metriques_test=None,
                 X_test=None, y_test=None, infos_test=None):
        self.modele        = modele
        self.features      = list(features)
        self.categorielles = list(categorielles)
        self.categories    = categories
        self.params        = params
        self.clip_min      = clip_min          # stocké dans le paquet
        self.metriques_test = metriques_test
        self.date          = datetime.now().isoformat(timespec="seconds")
        self.X_test        = X_test
        self.y_test        = y_test
        self.infos_test    = infos_test

    def _preparer(self, X):
        manq = [c for c in self.features if c not in X.columns]
        if manq: raise KeyError(f"Colonnes absentes : {manq}")
        Xp = X[self.features].copy()
        for c in self.categorielles:
            Xp[c] = pd.Categorical(Xp[c].astype(str), categories=self.categories[c])
        return Xp

    def predict(self, X):
        p = self.modele.predict(self._preparer(X))
        return p if self.clip_min is None else np.clip(p, self.clip_min, None)

    def fit(self, X, y, **kw):
        self.modele = lgb.LGBMRegressor(**self.params)
        self.modele.fit(self._preparer(X), y, **kw)
        self.date = datetime.now().isoformat(timespec="seconds")
        return self

    def importance(self, n=20):
        v = pd.Series(self.modele.booster_.feature_importance("gain"),
                      index=self.modele.feature_name_)
        return (100*v/v.sum()).sort_values(ascending=False).head(n)

    def sauver(self, chemin):
        joblib.dump(self, chemin)
        return chemin

    @staticmethod
    def charger(chemin):
        return joblib.load(chemin)

    def __repr__(self):
        mae_str = "n/a"
        if self.metriques_test and 'MAE' in self.metriques_test:
            mae_val = self.metriques_test['MAE']
            mae_str = f"{mae_val:,.0f}" if isinstance(mae_val, (int, float)) \
                      else str(mae_val)
        return f"<ModeleRBNS | {len(self.features)} variables | " \
               f"MAE test {mae_str} | {self.date}>"









# AVANT
mae_test_val = float(metriques("Test", y_te, pred_apres["Test"])['MAE'])
# APRÈS
mae_test_val = float(metriques(y_true_final, y_pred_final)['MAE'])

# AVANT
paquet = ModeleRBNS(modele=modele_apres, features=list(X_tr.columns), ...
# APRÈS
paquet = ModeleRBNS(modele=modele_apres, features=list(FEATURES_MODEL),
                    clip_min=CLIP_MIN, ...



