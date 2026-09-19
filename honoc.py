perf_avant_split = pd.DataFrame([metriques(k, vrais[k], pred_avant[k]) for k in vrais])
print(perf_avant_split.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))





pred_apres = {
    "Entrainement": predire(modele_apres, X_tr),
    "Validation": predire(modele_apres, X_va),
    "Test": predire(modele_apres, X_te),
}
perf_apres_split = pd.DataFrame([metriques(k, vrais[k], pred_apres[k]) for k in vrais])
print(perf_apres_split.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))









if 'time_idx' in X_tune.columns:
    ordre = X_tune['time_idx'].sort_values().index
else:
    ordre = X_tune.sort_index().index
X_tune = X_tune.loc[ordre].reset_index(drop=True)
y_tune = y_tune.loc[ordre].reset_index(drop=True)

for col in ['Time', 'period']:
    if col in X_tune.columns:
        X_tune[col] = pd.to_numeric(X_tune[col], errors='coerce').fillna(0)









perf_avant_cv = calculer_perfs_splittings(modele_avant, X_tune, y_tune, n_splits=N_SPLITS_CV)
perf_apres_cv = calculer_perfs_splittings(modele_apres, X_tune, y_tune, n_splits=N_SPLITS_CV)
comparaison_cv = perf_avant_cv.merge(perf_apres_cv, on="Split", suffixes=("_avant", "_apres"))









comparaison_test = perf_avant_split.merge(perf_apres_split, on="Split", suffixes=("_avant", "_apres"))
data_test = comparaison_test[comparaison_test["Split"] == "Test"].iloc[0]







fig.update_layout(title=f"<b>Double lift chart ({n_groups} groupes)</b>", ...)













def graphique_parite(y_obs, y_pred, titre):
    y_obs, y_pred = np.asarray(y_obs, float), np.asarray(y_pred, float)
    lim = [min(y_obs.min(), y_pred.min()), max(y_obs.max(), y_pred.max())]
    fig = go.Figure()
    fig.add_scatter(x=lim, y=lim, mode="lines", line=dict(color="#c62828", dash="dash"),
                    name="y = x", hoverinfo="skip")
    fig.add_scatter(x=y_obs, y=y_pred, mode="markers",
                    marker=dict(size=5, color="#2E5EAA", opacity=.45), name="Observations")
    fig.update_layout(title=titre, xaxis_title="Valeur observee", yaxis_title="Valeur predite",
                      template="plotly_white", height=480, showlegend=False)
    return fig

graphique_parite(y_te, pred_avant["Test"], "Parite - avant tuning (Test)").show()
graphique_parite(y_te, pred_apres["Test"], "Parite - apres tuning (Test)").show()

def courbe_temporelle(df_source, col_temps, y_obs, pred_dict, titre):
    d = df_source[[col_temps]].copy()
    d["Observe"] = np.asarray(y_obs, float)
    for nom, p in pred_dict.items():
        d[nom] = np.asarray(p, float)
    g = d.groupby(col_temps, observed=True).mean(numeric_only=True).reset_index()
    fig = go.Figure()
    fig.add_scatter(x=g[col_temps].astype(str), y=g["Observe"], mode="lines+markers",
                    name="Observe", line=dict(color="#141B34", width=3))
    for nom, coul in zip(pred_dict, ["#95A5A6", "#1F3A93"]):
        fig.add_scatter(x=g[col_temps].astype(str), y=g[nom], mode="lines+markers",
                        name=nom, line=dict(color=coul, width=2, dash="dot"))
    fig.update_layout(title=titre, xaxis_title=col_temps, yaxis_title="Moyenne",
                      template="plotly_white", height=480,
                      legend=dict(orientation="h", y=1.08, x=.5, xanchor="center"))
    return fig

courbe_temporelle(df_test, TIME_COL, y_te,
                  {"Avant tuning": pred_avant["Test"], "Apres tuning": pred_apres["Test"]},
                  "Observe vs predit par periode (Test)").show()



