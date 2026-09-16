# Code validate

# 0. Correspondance colonne -> nombre de lags (2 groupes désormais : target et vars_expliques)
lags_par_colonne = {
    TARGET: N_LAGS_TARGET,
    'dec_' + TARGET: N_LAGS_TARGET,
    **{v: N_LAGS_VARS for v in vars_expliques},
}

# 2. Check columns exist
expected_cols = [f'{c}_lag_{i}' for c, n in lags_par_colonne.items() for i in range(1, n + 1)]
assert all(c in df_with_lag.columns for c in expected_cols), "Missing lag columns"
print("✅ All lag columns present")

# 3. Verify lag logic & Display Examples
sort_cols = id_vars + [year_col, time_col]
df_sorted = df_with_lag.sort_values(sort_cols).reset_index(drop=True)

first_id_row = df_sorted[id_vars].sample(n=1, random_state=SEED).iloc[0]
mask = (df_sorted[id_vars] == first_id_row).all(axis=1)
g = df_sorted[mask].reset_index(drop=True)

display_cols = id_vars + [year_col, time_col] + list(lags_par_colonne.keys()) + expected_cols
print("Example Rows (First Group):")
display(g[display_cols].sort_values([year_col, time_col], ascending=False))

# Automated check for sample group on all lagged columns
for col, n in lags_par_colonne.items():
    for i in range(len(g)):
        for lag in range(1, n + 1):
            lc = f'{col}_lag_{lag}'
            val = g[lc].iloc[i]
            exp = g[col].iloc[i - lag] if i - lag >= 0 else np.nan
            assert (pd.isna(val) and pd.isna(exp)) or (val == exp), f"Lag mismatch at row {i}, lag {lag}"
    print(f"✅ Lag values correct for first group for col : {col}")

# 4. Check NaNs in first row of each group
first_rows_idx = df_sorted.groupby(id_vars, observed=False).apply(lambda x: x.index[0], include_groups=False)
assert df_sorted.loc[first_rows_idx, expected_cols].isna().all().all(), "First rows of groups should have NaN lags"
print("✅ NaNs correct in first row of each group")
