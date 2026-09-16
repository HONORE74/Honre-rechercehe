print(X_tr.shape, X_va.shape, X_te.shape)
lag_cols = [c for c in X_tr.columns if c not in X_te.columns]
print(lag_cols)
