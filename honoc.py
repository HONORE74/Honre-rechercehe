print([c for c in df.columns if "_lag_" in c])


# Après entraînement :
importance = pd.Series(
    modele_apres.feature_importances_,
    index=modele_apres.feature_name_
).sort_values(ascending=False)
print(importance.head(10))
