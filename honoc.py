preds_clipped = preds if CLIP_MIN is None else np.clip(preds, CLIP_MIN, None)


    return p if CLIP_MIN is None else np.clip(p, CLIP_MIN, None)



for col in ["MAE", "RMSE", "wMAPE"]:
    s = -1
    comparaison[f"Gain_{col}_%"] = s*100*(comparaison[f"{col}_apres"]-comparaison[f"{col}_avant"]) \
                                   / comparaison[f"{col}_avant"].abs()
