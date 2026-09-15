preds_clipped = preds if CLIP_MIN is None else np.clip(preds, CLIP_MIN, None)



for col in ["MAE", "RMSE", "wMAPE"]:
    s = -1
    comparaison[f"Gain_{col}_%"] = s*100*(comparaison[f"{col}_apres"]-comparaison[f"{col}_avant"]) \
                                   / comparaison[f"{col}_avant"].abs()


print(comparaison[["Split", "MAE_avant", "MAE_apres", "Gain_MAE_%",
                   "RMSE_avant", "RMSE_apres", "Gain_RMSE_%",
                   "wMAPE_avant", "wMAPE_apres", "Gain_wMAPE_%"]].to_string(
      index=False, float_format=lambda v: f"{v:,.3f}"))
