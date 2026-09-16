residus = pd.DataFrame({
    "y_vrai": y_true_final,
    "y_pred": y_pred_final,
    "erreur_abs": np.abs(np.asarray(y_pred_final) - np.asarray(y_true_final)),
})
residus = pd.concat([df_test.loc[idx_test, COLS_ID].reset_index(drop=True), residus.reset_index(drop=True)], axis=1)

top_erreurs = residus.sort_values("erreur_abs", ascending=False).head(15)
print(top_erreurs.to_string(index=False))

part_top10 = top_erreurs["erreur_abs"].head(10).sum() / residus["erreur_abs"].sum() * 100
print(f"\nLes 10 plus grosses erreurs représentent {part_top10:.1f}% de la somme totale des erreurs absolues")
