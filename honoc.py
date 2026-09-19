# Remplace :
y_pred_te = modele.predict(X_te)
# Par :
y_pred_te = modele_apres.predict(X_te)








# Remplace :
last_model = modele_final
# Par :
last_model = modele_apres









# Remplace :
model_lgbm = modele_final
# Par :
model_lgbm = modele_apres












# Remplace :
last_model = modele_final
# Par :
last_model = modele_apres










# Remplace :
y_true = df_perf["Valeur_Reelle"].values.astype(float)
y_pred = df_perf["Valeur_Predite"].values.astype(float)
# Par :
y_true = df_perf["y_obs"].values.astype(float)
y_pred = df_perf["y_pred"].values.astype(float)












# Remplace :
model=modele_final
# Par :
model=modele_apres









