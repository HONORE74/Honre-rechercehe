preds_clipped = preds if CLIP_MIN is None else np.clip(preds, CLIP_MIN, None)


    return p if CLIP_MIN is None else np.clip(p, CLIP_MIN, None)




print(comparaison[["Split", "MAE_avant", "MAE_apres", "Gain_MAE_%",
                   "RMSE_avant", "RMSE_apres", "Gain_RMSE_%",
                   "wMAPE_avant", "wMAPE_apres", "Gain_wMAPE_%"]].to_string(
      index=False, float_format=lambda v: f"{v:,.3f}"))
