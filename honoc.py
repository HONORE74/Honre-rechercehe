degrade_rows  = comparaison_residus[comparaison_residus["delta"] > 0]
ameliore_rows = comparaison_residus[comparaison_residus["delta"] < 0]

delta_moyen_degrad  = degrade_rows["delta"].mean()
delta_moyen_amelior = ameliore_rows["delta"].abs().mean()
ratio = delta_moyen_degrad / delta_moyen_amelior

print(f"Ampleur moyenne d'une dégradation  : {delta_moyen_degrad:,.0f}")
print(f"Ampleur moyenne d'une amélioration : {delta_moyen_amelior:,.0f}")
print(f"Ratio (dégradation / amélioration) : {ratio:.2f}x")

# Est-ce que le tuning se trompe plus sur les GROS dossiers ?
corr = comparaison_residus["y_vrai"].corr(comparaison_residus["delta"])
print(f"\nCorrélation entre la taille du dossier (y_vrai) et la dégradation (delta) : {corr:.2f}")
