import numpy as np
import pandas as pd
import lightgbm as lgb
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

TARGET   = "IBNR_best_estimate_eop"
ID_COLS  = [c for c in ["Partner", "Companies", "Lob", "Activity",
                        "Periodicity", "Risk", "Reinsurance"] if c in df.columns]
TIME_COL = "time_idx"

PALETTE  = {"train": "#1f4e79", "valid": "#e07b39",
            "test":  "#2e7d32", "ref": "#b0b0b0"}

print(f"Lignes : {len(df):,}")
print(f"Colonnes : {df.shape[1]}")
print(f"Identifiants métier : {ID_COLS}")













y_all = df[TARGET]

n_neg   = (y_all < 0).sum()
n_zero  = (y_all == 0).sum()
n_pos   = (y_all > 0).sum()
n_nan   = y_all.isna().sum()

diagnostic = pd.DataFrame({
    "Catégorie": ["Négatives", "Nulles", "Strictement positives", "Manquantes"],
    "Effectif" : [n_neg, n_zero, n_pos, n_nan],
    "Part (%)" : [100*n_neg/len(df), 100*n_zero/len(df),
                  100*n_pos/len(df), 100*n_nan/len(df)]
})
print(diagnostic.to_string(index=False))

print(f"\nMinimum   : {y_all.min():,.2f}")
print(f"Médiane   : {y_all.median():,.2f}")
print(f"Moyenne   : {y_all.mean():,.2f}")
print(f"Maximum   : {y_all.max():,.2f}")
print(f"Ratio max/médiane : {y_all.max()/max(y_all.median(),1):,.0f}")

if n_neg > 0:
    print(f"\n[ALERTE] {n_neg:,} valeurs négatives détectées.")
    print("Les objectifs 'tweedie', 'gamma' et 'poisson' de LightGBM les refuseront.")
    print("Voir BLOC 2bis pour les stratégies de traitement.")y_all = df[TARGET]

n_neg   = (y_all < 0).sum()
n_zero  = (y_all == 0).sum()
n_pos   = (y_all > 0).sum()
n_nan   = y_all.isna().sum()

diagnostic = pd.DataFrame({
    "Catégorie": ["Négatives", "Nulles", "Strictement positives", "Manquantes"],
    "Effectif" : [n_neg, n_zero, n_pos, n_nan],
    "Part (%)" : [100*n_neg/len(df), 100*n_zero/len(df),
                  100*n_pos/len(df), 100*n_nan/len(df)]
})
print(diagnostic.to_string(index=False))

print(f"\nMinimum   : {y_all.min():,.2f}")
print(f"Médiane   : {y_all.median():,.2f}")
print(f"Moyenne   : {y_all.mean():,.2f}")
print(f"Maximum   : {y_all.max():,.2f}")
print(f"Ratio max/médiane : {y_all.max()/max(y_all.median(),1):,.0f}")

if n_neg > 0:
    print(f"\n[ALERTE] {n_neg:,} valeurs négatives détectées.")
    print("Les objectifs 'tweedie', 'gamma' et 'poisson' de LightGBM les refuseront.")
    print("Voir BLOC 2bis pour les stratégies de traitement.")
























y_pos = y_all[y_all > 0]

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=np.log10(y_pos),
    nbinsx=80,
    marker_color=PALETTE["train"],
    name="log10(IBNR)"
))
fig.update_layout(
    title="Distribution de l'IBNR strictement positif, échelle logarithmique",
    xaxis_title="log10(IBNR en euros)",
    yaxis_title="Effectif",
    template="plotly_white",
    height=500,
    showlegend=False
)
fig.add_annotation(
    x=np.log10(y_pos.median()), y=0, ay=-60,
    text=f"Médiane<br>{y_pos.median():,.0f} €",
    showarrow=True, arrowhead=2
)
fig



















GROUPE = ID_COLS[0] if ID_COLS else None

stats_grp = (df[df[TARGET].notna()]
             .groupby(GROUPE)[TARGET]
             .agg(mu="mean", var="var", n="size")
             .query("n >= 20 and mu > 0 and var > 0"))

log_mu, log_var = np.log(stats_grp["mu"]), np.log(stats_grp["var"])
pente, ordonnee, r_val, p_val, err = stats.linregress(log_mu, log_var)

x_line = np.linspace(log_mu.min(), log_mu.max(), 100)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=log_mu, y=log_var, mode="markers",
    marker=dict(size=stats_grp["n"]**0.4, color=PALETTE["train"], opacity=0.6),
    text=stats_grp.index, name="Groupes"
))
fig.add_trace(go.Scatter(
    x=x_line, y=ordonnee + pente*x_line, mode="lines",
    line=dict(color=PALETTE["orange"] if "orange" in PALETTE else "#e07b39", width=3),
    name=f"Pente = {pente:.3f}"
))
fig.update_layout(
    title=(f"Relation moyenne-variance par {GROUPE}<br>"
           f"<sub>p empirique = {pente:.3f} (R² = {r_val**2:.3f}, "
           f"erreur-type = {err:.3f})</sub>"),
    xaxis_title="log(moyenne)",
    yaxis_title="log(variance)",
    template="plotly_white", height=550
)
fig



















from scipy.optimize import minimize_scalar

def tweedie_deviance(y, mu, p):
    """Déviance de Tweedie, proxy de la log-vraisemblance à dispersion fixée."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    if p == 0:
        d = (y - mu)**2
    elif p == 1:
        d = 2*(np.where(y > 0, y*np.log(y/mu), 0) - (y - mu))
    elif p == 2:
        d = 2*(np.log(mu/y) + y/mu - 1)
    else:
        d = 2*(np.power(y, 2-p)/((1-p)*(2-p))
               - y*np.power(mu, 1-p)/(1-p)
               + np.power(mu, 2-p)/(2-p))
    return np.sum(d)

y_fit = df.loc[df[TARGET].notna() & (df[TARGET] >= 0), TARGET].values
mu_naif = np.full_like(y_fit, y_fit.mean())

grille_p = np.arange(1.05, 1.96, 0.05)
deviances = [tweedie_deviance(y_fit, mu_naif, p) for p in grille_p]
p_optimal = grille_p[int(np.argmin(deviances))]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=grille_p, y=deviances, mode="lines+markers",
    line=dict(color=PALETTE["train"], width=3),
    marker=dict(size=8), name="Déviance"
))
fig.add_vline(x=p_optimal, line_dash="dash", line_color="#c62828",
              annotation_text=f"p optimal = {p_optimal:.2f}")
fig.add_vline(x=pente, line_dash="dot", line_color="#2e7d32",
              annotation_text=f"p empirique = {pente:.2f}")
fig.update_layout(
    title="Sélection du paramètre de puissance de Tweedie par minimisation de la déviance",
    xaxis_title="Paramètre de puissance p",
    yaxis_title="Déviance de Tweedie",
    template="plotly_white", height=520
)
fig























lois = {
    "Log-normale": stats.lognorm,
    "Gamma"      : stats.gamma,
    "Weibull"    : stats.weibull_min,
    "Pareto"     : stats.pareto,
}

echantillon = y_pos.sample(min(20000, len(y_pos)), random_state=42).values
resultats = []

for nom, loi in lois.items():
    try:
        params = loi.fit(echantillon, floc=0)
        ll = np.sum(loi.logpdf(echantillon, *params))
        k = len(params)
        resultats.append({
            "Loi": nom,
            "Log-vraisemblance": ll,
            "AIC": 2*k - 2*ll,
            "BIC": k*np.log(len(echantillon)) - 2*ll
        })
    except Exception as e:
        print(f"{nom} : échec de l'ajustement ({e})")

comparaison = pd.DataFrame(resultats).sort_values("AIC")

fig = go.Figure()
fig.add_trace(go.Bar(
    x=comparaison["Loi"], y=comparaison["AIC"],
    marker_color=PALETTE["train"],
    text=comparaison["AIC"].round(0), textposition="outside"
))
fig.update_layout(
    title="Comparaison des lois candidates par critère AIC (valeur faible = meilleur ajustement)",
    xaxis_title="Loi", yaxis_title="AIC",
    template="plotly_white", height=500
)
fig













periodes = np.sort(df[TIME_COL].unique())
n_p = len(periodes)

cut_train = periodes[int(0.70*n_p)]
cut_valid = periodes[int(0.85*n_p)]

df_train = df[df[TIME_COL] <= cut_train]
df_valid = df[(df[TIME_COL] > cut_train) & (df[TIME_COL] <= cut_valid)]
df_test  = df[df[TIME_COL] > cut_valid]

repartition = pd.DataFrame({
    "Ensemble": ["Entraînement", "Validation", "Test"],
    "Périodes": [df_train[TIME_COL].nunique(),
                 df_valid[TIME_COL].nunique(),
                 df_test[TIME_COL].nunique()],
    "Observations": [len(df_train), len(df_valid), len(df_test)],
    "Part (%)": [100*len(df_train)/len(df),
                 100*len(df_valid)/len(df),
                 100*len(df_test)/len(df)]
})
print(repartition.to_string(index=False))

fig = go.Figure()
for nom, sous_df, coul in [("Entraînement", df_train, PALETTE["train"]),
                           ("Validation",   df_valid, PALETTE["valid"]),
                           ("Test",         df_test,  PALETTE["test"])]:
    serie = sous_df.groupby(TIME_COL).size()
    fig.add_trace(go.Bar(x=serie.index, y=serie.values,
                         name=nom, marker_color=coul))

fig.update_layout(
    title="Découpage temporel des ensembles, volumétrie par trimestre",
    xaxis_title="Indice temporel", yaxis_title="Nombre d'observations",
    barmode="stack", template="plotly_white", height=500
)
fig















EXCLURE = [TARGET, TIME_COL, "year", "quarter", "annee", "pays"] + ID_COLS
FEATURES = [c for c in df.columns if c not in EXCLURE]

for sd in (df_train, df_valid, df_test):
    for c in FEATURES:
        if sd[c].dtype == object:
            sd[c] = sd[c].astype("category")

masque = lambda d: d[TARGET].notna() & (d[TARGET] >= 0)

X_tr, y_tr = df_train.loc[masque(df_train), FEATURES], df_train.loc[masque(df_train), TARGET]
X_va, y_va = df_valid.loc[masque(df_valid), FEATURES], df_valid.loc[masque(df_valid), TARGET]
X_te, y_te = df_test.loc[masque(df_test),  FEATURES], df_test.loc[masque(df_test),  TARGET]

params = dict(
    objective="tweedie",
    tweedie_variance_power=float(p_optimal),
    metric="tweedie",
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=64,
    max_depth=-1,
    min_child_samples=30,
    subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbose=-1
)

modele = lgb.LGBMRegressor(**params)
suivi = {}

modele.fit(
    X_tr, y_tr,
    eval_set=[(X_tr, y_tr), (X_va, y_va)],
    eval_names=["Entraînement", "Validation"],
    callbacks=[
        lgb.early_stopping(200, verbose=False),
        lgb.log_evaluation(0),
        lgb.record_evaluation(suivi)
    ]
)
print(f"Meilleure itération : {modele.best_iteration_}")















metrique = list(suivi["Entraînement"].keys())[0]

fig = go.Figure()
fig.add_trace(go.Scatter(
    y=suivi["Entraînement"][metrique], mode="lines",
    line=dict(color=PALETTE["train"], width=2), name="Entraînement"))
fig.add_trace(go.Scatter(
    y=suivi["Validation"][metrique], mode="lines",
    line=dict(color=PALETTE["valid"], width=2), name="Validation"))
fig.add_vline(x=modele.best_iteration_, line_dash="dash", line_color="#c62828",
              annotation_text=f"Arrêt anticipé : {modele.best_iteration_}")
fig.update_layout(
    title="Convergence du modèle et détection du surapprentissage",
    xaxis_title="Itération", yaxis_title=f"Métrique ({metrique})",
    template="plotly_white", height=520
)
fig














def indice_gini(y_vrai, y_pred):
    """Gini normalisé fondé sur la courbe de concentration."""
    ordre = np.argsort(y_pred)
    y_ord = np.asarray(y_vrai)[ordre]
    cum = np.cumsum(y_ord) / y_ord.sum()
    gini_modele = 1 - 2*np.trapz(cum, dx=1/len(cum))
    ordre_parfait = np.argsort(y_vrai)
    cum_p = np.cumsum(np.asarray(y_vrai)[ordre_parfait]) / y_vrai.sum()
    gini_parfait = 1 - 2*np.trapz(cum_p, dx=1/len(cum_p))
    return gini_modele / gini_parfait if gini_parfait else np.nan

def evaluer(nom, y_vrai, y_pred):
    y_vrai, y_pred = np.asarray(y_vrai), np.asarray(y_pred)
    non_nul = y_vrai > 1
    return {
        "Ensemble": nom,
        "RMSE": np.sqrt(mean_squared_error(y_vrai, y_pred)),
        "MAE": mean_absolute_error(y_vrai, y_pred),
        "R2": r2_score(y_vrai, y_pred),
        "MAPE médian (%)": 100*np.median(
            np.abs(y_vrai[non_nul]-y_pred[non_nul])/y_vrai[non_nul]),
        "Gini": indice_gini(y_vrai, y_pred),
        "Biais global (%)": 100*(y_pred.sum()-y_vrai.sum())/y_vrai.sum()
    }

perf = pd.DataFrame([
    evaluer("Entraînement", y_tr, modele.predict(X_tr)),
    evaluer("Validation",   y_va, modele.predict(X_va)),
    evaluer("Test",         y_te, modele.predict(X_te)),
])
print(perf.to_string(index=False))


















y_pred_te = modele.predict(X_te)

def courbe_lorenz(y_vrai, y_pred):
    ordre = np.argsort(y_pred)
    y_ord = np.asarray(y_vrai)[ordre]
    return (np.arange(1, len(y_ord)+1)/len(y_ord),
            np.cumsum(y_ord)/y_ord.sum())

x_m, y_m = courbe_lorenz(y_te, y_pred_te)
x_o, y_o = courbe_lorenz(y_te, y_te)

fig = go.Figure()
fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
    line=dict(color=PALETTE["ref"], dash="dash"), name="Aucun pouvoir discriminant"))
fig.add_trace(go.Scatter(x=x_o, y=y_o, mode="lines",
    line=dict(color=PALETTE["test"], width=2, dash="dot"), name="Modèle parfait"))
fig.add_trace(go.Scatter(x=x_m, y=y_m, mode="lines",
    line=dict(color=PALETTE["train"], width=3), name="Modèle LightGBM",
    fill="tonexty", fillcolor="rgba(31,78,121,0.12)"))
fig.update_layout(
    title=(f"Courbe de Lorenz sur l'ensemble de test<br>"
           f"<sub>Gini normalisé = {indice_gini(y_te, y_pred_te):.4f}</sub>"),
    xaxis_title="Proportion cumulée des observations, triées par prédiction croissante",
    yaxis_title="Proportion cumulée de l'IBNR observé",
    template="plotly_white", height=580
)
fig





