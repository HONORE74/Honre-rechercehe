N_LAGS_TARGET = 4
N_LAGS_VARS   = 2

VAR_EXPLIQUE_1 = "Paid_claims"
VAR_EXPLIQUE_2 = "Earned_Risk_Premium"
VAR_EXPLIQUE_3 = "Claim_Result"
VAR_EXPLIQUE_4 = "PLR"
VAR_EXPLIQUE_5 = "LR"
vars_expliques = [VAR_EXPLIQUE_1, VAR_EXPLIQUE_2, VAR_EXPLIQUE_3, VAR_EXPLIQUE_4, VAR_EXPLIQUE_5]






id_vars  = ID_COLS
year_col = 'year'
time_col = TIME_COL

df_with_lag = add_lags(
    tmp_df, id_vars=id_vars,
    cols_to_lag=[TARGET, 'dec_' + TARGET],
    n_lags=N_LAGS_TARGET, year_col=year_col, time_col=time_col,
)
df_with_lag = add_lags(
    df_with_lag, id_vars=id_vars,
    cols_to_lag=vars_expliques,
    n_lags=N_LAGS_VARS, year_col=year_col, time_col=time_col,
)







np.random.seed(42)

if TARGET_MODE == "cumul":
    MODEL_TARGET = TARGET
elif TARGET_MODE == "dec":
    MODEL_TARGET = "dec_" + TARGET

cols_lag_target = [TARGET, "dec_" + TARGET] if TARGET_MODE == "cumul" else ["dec_" + TARGET]
lags_target = [f"{col}_lag_{i}" for col in cols_lag_target for i in range(1, N_LAGS_TARGET + 1)]
