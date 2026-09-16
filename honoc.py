# y_pred_final_avant = tes prédictions reconstruites du modèle AVANT tuning
# (même logique de reconstruction "si dec" que pour y_pred_final, mais avec modele_avant / pred_avant["Test"])
# Si TARGET_MODE == "cumul", c'est simplement pred_avant["Test"]

comparaison_residus = pd.DataFrame({
    "y_vrai": np.asarray(y_true_final),
    "erreur_avant": np.abs(np.asarray(y_pred_final_avant) - np.asarray(y_true_final)),
    "erreur_apres": np.abs(np.asarray(y_pred_final)       - np.asarray(y_true_final)),
})
comparaison_residus["delta"] = comparaison_residus["erreur_apres"] - comparaison_residus["erreur_avant"]

pct_degrade  = (comparaison_residus["delta"] > 0).mean() * 100
pct_ameliore = (comparaison_residus["delta"] < 0).mean() * 100
print(f"% de lignes où le tuning DÉGRADE la prédiction  : {pct_degrade:.1f}%")
print(f"% de lignes où le tuning AMÉLIORE la prédiction : {pct_ameliore:.1f}%")

degradation_totale = comparaison_residus.loc[comparaison_residus["delta"] > 0, "delta"].sum()
pires = comparaison_residus.sort_values("delta", ascending=False).head(10)
part_top10_degrad = pires["delta"].sum() / degradation_totale * 100

print(f"\nLes 10 pires dégradations représentent {part_top10_degrad:.1f}% de la dégradation TOTALE")
print(pires.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
