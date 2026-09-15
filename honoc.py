if OBJECTIVE != "tweedie":
    PARAMS_BASELINE.pop("tweedie_variance_power", None)




def predire(m, X):
    p = m.predict(X)
    return p if CLIP_MIN is None else np.clip(p, CLIP_MIN, None)
