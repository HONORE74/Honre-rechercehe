Pour la partie exclusion de celule 

EXCLUDE = [c for c in {TARGET, TIME_COL, "year", "quarter", "annee",
                       *REMOVED, *CONSTANTES,
                       *Exclude_feature.get(TARGET, [])}
           if c in df_model.columns]

Pour verfirer si cela a bien marcher

_fuite = [c for c in Exclude_feature.get(TARGET, []) if c in df_model.columns]
print(f"Colonnes de fuite retirees : {len(_fuite)}  ->  {_fuite}")


Selection de la loi de la TARGET après le scoping Target

target_values = df[TARGET].astype(float)
nombre_negatifs = int((target_values < 0).sum())
nombre_zeros = int((target_values == 0).sum())

if nombre_negatifs > 0:
    LOI       = "gaussienne"
    OBJECTIVE = "regression"
    METRIC    = "l2"
    CLIP_MIN  = None
elif nombre_zeros > 0:
    LOI       = "Tweedie"
    OBJECTIVE = "tweedie"
    METRIC    = "tweedie"
    CLIP_MIN  = 0.0
else:
    LOI       = "Gamma"
    OBJECTIVE = "gamma"
    METRIC    = "gamma"
    CLIP_MIN  = 0.0

print(f"negatives : {nombre_negatifs:,}   zeros : {nombre_zeros:,}   sur {len(target_values):,}")
print(f"loi retenue : {LOI}   ->   objective = '{OBJECTIVE}'")








Pour le Bloc 3 toujours dans cette logiques 

objective=OBJECTIVE,
metric=METRIC,



if OBJECTIVE != "tweedie":
    params.pop("tweedie_variance_power", None)


objective=OBJECTIVE,
metric=METRIC,


objective=OBJECTIVE,
eval_metric=METRIC,


objective=OBJECTIVE,
metric=METRIC,





TO DO 

df = df_with_lag.copy()
lags_ajoutes = [c for c in df.columns if "_lag_" in c and c not in FEATURES]
FEATURES = FEATURES + lags_ajoutes
NUMERIQUES = [c for c in FEATURES if c not in CATEGORIELLES]

print(f"{len(lags_ajoutes)} lags ajoutes   ->   {len(FEATURES)} features au total")
