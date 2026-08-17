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













ech = np.random.RandomState(42).choice(len(y_te), min(6000, len(y_te)), replace=False)
yv, yp = np.asarray(y_te)[ech], y_pred_te[ech]
pos = (yv > 0) & (yp > 0)

fig = go.Figure()
fig.add_trace(go.Scattergl(
    x=yv[pos], y=yp[pos], mode="markers",
    marker=dict(size=4, color=PALETTE["train"], opacity=0.35),
    name="Observations"))
lim = [max(yv[pos].min(), 1), yv[pos].max()]
fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
    line=dict(color="#c62828", dash="dash", width=2), name="Prédiction parfaite"))
fig.update_layout(
    title="Valeurs prédites contre valeurs observées, échelle logarithmique",
    xaxis_title="IBNR observé (€)", yaxis_title="IBNR prédit (€)",
    xaxis_type="log", yaxis_type="log",
    template="plotly_white", height=600
)
fig














residus = np.asarray(y_te) - y_pred_te
pred_pos = y_pred_te > 0

fig = go.Figure()
fig.add_trace(go.Scattergl(
    x=y_pred_te[pred_pos], y=residus[pred_pos], mode="markers",
    marker=dict(size=4, color=PALETTE["train"], opacity=0.3), name="Résidus"))
fig.add_hline(y=0, line_color="#c62828", line_dash="dash")

bins = pd.qcut(y_pred_te[pred_pos], 20, duplicates="drop")
ec_type = pd.Series(residus[pred_pos]).groupby(bins).std()
centres = pd.Series(y_pred_te[pred_pos]).groupby(bins).median()
fig.add_trace(go.Scatter(x=centres, y=ec_type, mode="lines+markers",
    line=dict(color=PALETTE["valid"], width=3), name="Écart-type par décile"))

fig.update_layout(
    title=("Résidus contre prédictions<br><sub>La croissance de l'écart-type "
           "avec le niveau prédit établit l'hétéroscédasticité</sub>"),
    xaxis_title="IBNR prédit (€)", yaxis_title="Résidu (€)",
    xaxis_type="log", template="plotly_white", height=580
)
fig















cal = pd.DataFrame({"vrai": np.asarray(y_te), "pred": y_pred_te})
cal["decile"] = pd.qcut(cal["pred"], 10, labels=False, duplicates="drop")
agg = cal.groupby("decile").agg(
    moyenne_predite=("pred","mean"), moyenne_observee=("vrai","mean"),
    effectif=("vrai","size")).reset_index()

fig = go.Figure()
fig.add_trace(go.Bar(x=agg["decile"]+1, y=agg["moyenne_observee"],
    name="Observé", marker_color=PALETTE["train"]))
fig.add_trace(go.Scatter(x=agg["decile"]+1, y=agg["moyenne_predite"],
    mode="lines+markers", name="Prédit",
    line=dict(color="#c62828", width=3), marker=dict(size=10)))
fig.update_layout(
    title="Calibration du modèle par décile de prédiction",
    xaxis_title="Décile de prédiction", yaxis_title="IBNR moyen (€)",
    template="plotly_white", height=520
)
fig















imp = (pd.DataFrame({"variable": FEATURES,
                     "gain": modele.booster_.feature_importance("gain")})
       .sort_values("gain", ascending=True).tail(25))

fig = go.Figure(go.Bar(
    x=imp["gain"], y=imp["variable"], orientation="h",
    marker=dict(color=imp["gain"], colorscale="Blues")))
fig.update_layout(
    title="Importance des variables, mesurée par le gain cumulé",
    xaxis_title="Gain total", yaxis_title="",
    template="plotly_white", height=750
)
fig














objectifs = {
    "Tweedie":       dict(objective="tweedie", tweedie_variance_power=float(p_optimal)),
    "Gamma":         dict(objective="gamma"),
    "Poisson":       dict(objective="poisson"),
    "L2 (quadratique)": dict(objective="regression"),
    "L1 (absolue)":  dict(objective="regression_l1"),
    "Huber":         dict(objective="huber"),
}

comp = []
for nom, obj in objectifs.items():
    try:
        p = {**params, **obj, "n_estimators": 800, "metric": "rmse"}
        p.pop("tweedie_variance_power", None) if "tweedie" not in obj.get("objective","") else None
        m = lgb.LGBMRegressor(**p)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        comp.append(evaluer(nom, y_te, m.predict(X_te)))
    except Exception as e:
        print(f"{nom} : {e}")

tableau = pd.DataFrame(comp).rename(columns={"Ensemble":"Objectif"})
print(tableau.to_string(index=False))

fig = go.Figure()
fig.add_trace(go.Bar(x=tableau["Objectif"], y=tableau["Gini"],
                     marker_color=PALETTE["train"], name="Gini",
                     text=tableau["Gini"].round(4), textposition="outside"))
fig.update_layout(
    title="Pouvoir discriminant comparé des objectifs LightGBM (indice de Gini sur le test)",
    xaxis_title="Objectif", yaxis_title="Gini normalisé",
    template="plotly_white", height=520
)
fig



















GRP = ID_COLS[0]
res = df_test.loc[masque(df_test), [GRP, TARGET]].copy()
res["pred"] = y_pred_te

par_groupe = (res.groupby(GRP)
              .apply(lambda g: pd.Series({
                  "n": len(g),
                  "RMSE": np.sqrt(mean_squared_error(g[TARGET], g["pred"])),
                  "Biais (%)": 100*(g["pred"].sum()-g[TARGET].sum())/max(g[TARGET].sum(),1),
                  "IBNR moyen": g[TARGET].mean()}))
              .query("n >= 20").sort_values("Biais (%)"))

fig = go.Figure(go.Bar(
    x=par_groupe["Biais (%)"], y=par_groupe.index, orientation="h",
    marker=dict(color=par_groupe["Biais (%)"], colorscale="RdBu",
                cmid=0, colorbar=dict(title="Biais %")),
    text=par_groupe["n"], textposition="outside"))
fig.add_vline(x=0, line_color="#333", line_width=2)
fig.update_layout(
    title=f"Biais relatif par {GRP} sur l'ensemble de test (effectif indiqué)",
    xaxis_title="Biais relatif (%)", yaxis_title="",
    template="plotly_white", height=max(500, 22*len(par_groupe))
)
fig
















GRP = ID_COLS[0]
res = df_test.loc[masque(df_test), [GRP, TARGET]].copy()
res["pred"] = y_pred_te

par_groupe = (res.groupby(GRP)
              .apply(lambda g: pd.Series({
                  "n": len(g),
                  "RMSE": np.sqrt(mean_squared_error(g[TARGET], g["pred"])),
                  "Biais (%)": 100*(g["pred"].sum()-g[TARGET].sum())/max(g[TARGET].sum(),1),
                  "IBNR moyen": g[TARGET].mean()}))
              .query("n >= 20").sort_values("Biais (%)"))

fig = go.Figure(go.Bar(
    x=par_groupe["Biais (%)"], y=par_groupe.index, orientation="h",
    marker=dict(color=par_groupe["Biais (%)"], colorscale="RdBu",
                cmid=0, colorbar=dict(title="Biais %")),
    text=par_groupe["n"], textposition="outside"))
fig.add_vline(x=0, line_color="#333", line_width=2)
fig.update_layout(
    title=f"Biais relatif par {GRP} sur l'ensemble de test (effectif indiqué)",
    xaxis_title="Biais relatif (%)", yaxis_title="",
    template="plotly_white", height=max(500, 22*len(par_groupe))
)
fig




















fractions = [0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
courbe = []
for f in fractions:
    idx = np.random.RandomState(42).choice(len(X_tr), int(f*len(X_tr)), replace=False)
    m = lgb.LGBMRegressor(**{**params, "n_estimators": 600})
    m.fit(X_tr.iloc[idx], y_tr.iloc[idx],
          eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    courbe.append({
        "n": len(idx),
        "RMSE entraînement": np.sqrt(mean_squared_error(y_tr.iloc[idx], m.predict(X_tr.iloc[idx]))),
        "RMSE validation": np.sqrt(mean_squared_error(y_va, m.predict(X_va)))
    })

lc = pd.DataFrame(courbe)

fig = go.Figure()
fig.add_trace(go.Scatter(x=lc["n"], y=lc["RMSE entraînement"], mode="lines+markers",
    line=dict(color=PALETTE["train"], width=3), name="Entraînement"))
fig.add_trace(go.Scatter(x=lc["n"], y=lc["RMSE validation"], mode="lines+markers",
    line=dict(color=PALETTE["valid"], width=3), name="Validation"))
fig.update_layout(
    title="Courbe d'apprentissage : le volume de données est-il le facteur limitant ?",
    xaxis_title="Nombre d'observations d'entraînement", yaxis_title="RMSE",
    template="plotly_white", height=520
)
fig




# pip install shap
import shap

ech_shap = X_te.sample(min(3000, len(X_te)), random_state=42)
explainer = shap.TreeExplainer(modele)
valeurs_shap = explainer.shap_values(ech_shap)

imp_shap = (pd.DataFrame({
        "variable": ech_shap.columns,
        "impact_moyen": np.abs(valeurs_shap).mean(axis=0)})
    .sort_values("impact_moyen", ascending=True).tail(20))

fig = go.Figure(go.Bar(
    x=imp_shap["impact_moyen"], y=imp_shap["variable"], orientation="h",
    marker=dict(color=imp_shap["impact_moyen"], colorscale="Blues"),
    text=imp_shap["impact_moyen"].round(0), textposition="outside"))
fig.update_layout(
    title=("Contribution moyenne de chaque variable à la prédiction "
           "(valeurs de Shapley)<br><sub>Exprimée en euros d'IBNR, "
           "interprétable directement</sub>"),
    xaxis_title="Impact absolu moyen sur la prédiction (€)",
    yaxis_title="", template="plotly_white", height=700
)
fig











res_temp = df_test.loc[masque(df_test), [TIME_COL, TARGET]].copy()
res_temp["pred"] = y_pred_te
res_temp["residu"] = res_temp[TARGET] - res_temp["pred"]

par_periode = (res_temp.groupby(TIME_COL)
    .apply(lambda g: pd.Series({
        "n": len(g),
        "RMSE": np.sqrt(mean_squared_error(g[TARGET], g["pred"])),
        "Biais (%)": 100*(g["pred"].sum()-g[TARGET].sum())/max(g[TARGET].sum(),1),
        "Résidu médian": g["residu"].median()}))
    .reset_index())

fig = go.Figure()
fig.add_trace(go.Bar(
    x=par_periode[TIME_COL], y=par_periode["Biais (%)"],
    marker=dict(color=par_periode["Biais (%)"], colorscale="RdBu", cmid=0),
    name="Biais relatif"))
fig.add_hline(y=0, line_color="#333", line_width=2)

pente_t, _, r_t, pval_t, _ = stats.linregress(
    par_periode[TIME_COL], par_periode["Biais (%)"])

fig.update_layout(
    title=(f"Stabilité du biais dans le temps sur l'ensemble de test<br>"
           f"<sub>Tendance = {pente_t:+.2f} point par trimestre "
           f"(p = {pval_t:.3f}). Une tendance significative remet en cause "
           f"l'hypothèse d'échangeabilité</sub>"),
    xaxis_title="Trimestre", yaxis_title="Biais relatif (%)",
    template="plotly_white", height=540
)
fig













prio = pd.DataFrame({"vrai": np.asarray(y_te), "pred": y_pred_te})
prio["ecart_abs"] = (prio["vrai"] - prio["pred"]).abs()
prio = prio.sort_values("ecart_abs", ascending=False).reset_index(drop=True)

prio["part_controles"] = (prio.index + 1) / len(prio)
prio["part_ecart"] = prio["ecart_abs"].cumsum() / prio["ecart_abs"].sum()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=prio["part_controles"]*100, y=prio["part_ecart"]*100,
    mode="lines", line=dict(color=PALETTE["train"], width=3),
    fill="tozeroy", fillcolor="rgba(31,78,121,0.12)",
    name="Priorisation par le modèle"))
fig.add_trace(go.Scatter(
    x=[0,100], y=[0,100], mode="lines",
    line=dict(color=PALETTE["ref"], dash="dash"),
    name="Contrôle aléatoire"))

for seuil in [5, 10, 20]:
    couvert = prio.loc[prio["part_controles"] <= seuil/100, "part_ecart"].max()*100
    fig.add_annotation(x=seuil, y=couvert,
        text=f"{seuil} % des contrôles<br>couvrent {couvert:.0f} % de l'écart",
        showarrow=True, arrowhead=2, ax=50, ay=-40)

fig.update_layout(
    title="Capacité de priorisation : part de l'écart total couverte selon l'effort de contrôle",
    xaxis_title="Part des observations contrôlées (%)",
    yaxis_title="Part de l'écart total couverte (%)",
    template="plotly_white", height=580
)
fig











Pour emplcer le Boloc 

Blc 2

y_pos = y_all[y_all > 0].sort_values()
rang = np.arange(1, len(y_pos)+1) / len(y_pos)

fig = go.Figure()
fig.add_trace(go.Scattergl(
    x=rang, y=y_pos.values, mode="lines",
    line=dict(color=PALETTE["train"], width=2), name="IBNR observé"))

for q in [0.50, 0.90, 0.99]:
    v = y_pos.quantile(q)
    fig.add_hline(y=v, line_dash="dot", line_color="#999",
                  annotation_text=f"Q{int(q*100)} = {v:,.0f} €",
                  annotation_position="right")

fig.update_layout(
    title=("Fonction de répartition empirique de l'IBNR<br>"
           "<sub>La forte concavité en fin de courbe traduit la queue lourde "
           "de la distribution</sub>"),
    xaxis_title="Proportion cumulée des observations",
    yaxis_title="IBNR (€)",
    template="plotly_white", height=550, showlegend=False
)
fig





















Blc11

comp_rang = pd.DataFrame({"vrai": np.asarray(y_te), "pred": y_pred_te})
comp_rang["centile"] = pd.qcut(comp_rang["vrai"].rank(method="first"),
                               100, labels=False, duplicates="drop")
agg_c = comp_rang.groupby("centile").agg(
    observe=("vrai","mean"), predit=("pred","mean")).reset_index()

fig = go.Figure()
fig.add_trace(go.Scatter(x=agg_c["centile"]+1, y=agg_c["observe"],
    mode="lines+markers", name="Observé",
    line=dict(color=PALETTE["train"], width=3)))
fig.add_trace(go.Scatter(x=agg_c["centile"]+1, y=agg_c["predit"],
    mode="lines+markers", name="Prédit",
    line=dict(color="#c62828", width=3, dash="dash")))
fig.update_layout(
    title="Valeurs moyennes observées et prédites par centile d'IBNR observé",
    xaxis_title="Centile d'IBNR observé", yaxis_title="IBNR moyen (€)",
    template="plotly_white", height=560
)
fig
























Blc 12 


diag = pd.DataFrame({"pred": y_pred_te, "residu": np.asarray(y_te) - y_pred_te})
diag["decile"] = pd.qcut(diag["pred"].rank(method="first"), 10,
                         labels=False, duplicates="drop")

par_dec = diag.groupby("decile").agg(
    pred_moy=("pred","mean"),
    ecart_type=("residu","std"),
    q10=("residu", lambda s: s.quantile(0.10)),
    q90=("residu", lambda s: s.quantile(0.90))).reset_index()

fig = go.Figure()
fig.add_trace(go.Bar(
    x=par_dec["decile"]+1, y=par_dec["ecart_type"],
    marker_color=PALETTE["train"], name="Écart-type des résidus",
    text=par_dec["ecart_type"].round(0), textposition="outside"))
fig.add_trace(go.Scatter(
    x=par_dec["decile"]+1, y=par_dec["q90"]-par_dec["q10"],
    mode="lines+markers", name="Étendue interdécile des résidus",
    line=dict(color=PALETTE["valid"], width=3), yaxis="y2"))

ratio = par_dec["ecart_type"].iloc[-1] / max(par_dec["ecart_type"].iloc[0], 1)
fig.update_layout(
    title=(f"Dispersion des résidus par décile de prédiction<br>"
           f"<sub>Rapport dernier décile sur premier décile = {ratio:,.0f}. "
           f"Une enveloppe de largeur constante serait inadaptée</sub>"),
    xaxis_title="Décile de prédiction",
    yaxis=dict(title="Écart-type des résidus (€)"),
    yaxis2=dict(title="Étendue interdécile (€)", overlaying="y", side="right"),
    template="plotly_white", height=560
)
fig









































