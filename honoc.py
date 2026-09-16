# Si TARGET_MODE == "cumul" (ton mode actuel), c'est simplement :
y_pred_final_avant = pred_avant["Test"]

# Si jamais tu es en TARGET_MODE == "dec", il faudrait la même reconstruction
# que pour y_pred_final mais à partir de modele_avant — dis-le-moi si c'est le cas.

comparaison_residus = pd.DataFrame({
    "y_vrai": y_true_final,
    "erreur_avant": np.abs(np.asarray(y_pred_final_avant) - np.asarray(y_true_final)),
    "erreur_apres": np.abs(np.asarray(y_pred_final)       - np.asarray(y_true_final)),
})
comparaison_residus = pd.concat(
    [df_test.loc[idx_test, COLS_ID].reset_index(drop=True), comparaison_residus.reset_index(drop=True)],
    axis=1,
)
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
