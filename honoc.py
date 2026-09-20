import os
chemin_db = f"artefacts_modele/{NOM_ETUDE}.db"
if os.path.exists(chemin_db):
    os.remove(chemin_db)
    print(f"Base supprimée : {chemin_db}")







d = pd.DataFrame({"y": Y, "a": P0, "b": P})
d["r"] = d["b"] / np.maximum(d["a"], 1e-9)
n_groups = 15
d["q"] = pd.qcut(d["r"].rank(method="first"), n_groups, labels=range(1, n_groups + 1))
g = d.groupby("q", observed=True).agg(obs=("y", "mean"), a=("a", "mean"), b=("b", "mean")).reset_index()

fig = go.Figure()
for col, nom, coul, w, dash in [("obs", "Observé", C["ardoise"], 4, None),
                                 ("a", "Avant tuning", C["gris"], 2.5, "dash"),
                                 ("b", "Après tuning", C["corail"], 3, "dot")]:
    fig.add_scatter(x=g["q"].astype(str), y=g[col], name=nom, mode="lines+markers",
                    line=dict(color=coul, width=w, dash=dash), marker=dict(size=9))
fig.update_layout(title=f"<b>Observe vs predit ({n_groups} groupes)</b>",
                  xaxis_title="Groupe (trie par rapport de prediction croissant)",
                  yaxis_title="Valeur moyenne du groupe (IBNR)",
                  template=TPL, height=500,
                  legend=dict(orientation="h", y=1.06, x=.5, xanchor="center"))
fig.show()





















# =============================================================================
#  CELLULES A INSERER DANS 04_Model_V1_50.ipynb
#  Objectif : ajouter lags target + lags variables explicatives + categoriques
# =============================================================================
#
#  ORDRE D'INSERTION :
#    Cellule A  →  apres le nettoyage des donnees (df existe, time_idx existe)
#    Cellule B  →  juste apres la Cellule A
#    Cellule C  →  juste apres la Cellule B
#    Cellule D  →  remplace la definition actuelle de FEATURES_MODEL
#    Cellule E  →  MODIFIER la cellule du modele baseline (pas de tuning)
#                  pour ajouter categorical_feature
#
#  RIEN D'AUTRE NE CHANGE : le split, le tuning, les graphiques restent identiques.
# =============================================================================


# %% =========================================================================
#  CELLULE A — Configuration des lags
#  Inserer APRES : Time Indexing and Data Preparation
#  Inserer AVANT : le split train / validation / test
# =============================================================================

N_LAGS_TARGET = 4      # nombre de lags sur Claims_incurred
N_LAGS_VARS   = 5      # nombre de lags sur les variables explicatives

TARGET = "Claims_incurred"
ID_COLS = ["Partner", "Companies", "Lob", "Activity", "Periodicity", "Risk"]

# Variables explicatives importantes (issues du SHAP ou de la connaissance metier)
vars_expliques = ["Paid_claims", "Earned_Risk_Premium", "Claim_Result"]


# %% =========================================================================
#  CELLULE B — Fonction add_lags
#  Inserer juste apres la Cellule A
# =============================================================================

def add_lags(df, id_vars, cols_to_lag, n_lags=4,
             year_col='year', time_col='time_idx'):
    """Ajoute des features laguees par groupe (id_vars) et par ordre temporel.

    Chaque groupe (ex: un couple Partner x Companies x Lob x ...) est traite
    independamment. Le shift(i) decale de i periodes vers le passe, donc
    la valeur au lag 1 est la valeur de la periode precedente.

    Les premieres lignes de chaque groupe auront des NaN (pas de passe) :
    LightGBM gere les NaN nativement, donc on ne les remplit PAS.
    """
    sort_cols = id_vars + [year_col, time_col]
    df_res = df.sort_values(by=sort_cols).copy()

    groupers = df_res.groupby(id_vars, observed=False)

    for col in cols_to_lag:
        grouped_col = groupers[col]
        for i in range(1, n_lags + 1):
            df_res[f'{col}_lag_{i}'] = grouped_col.shift(i)

    return df_res


# %% =========================================================================
#  CELLULE C — Appliquer les lags + convertir les categoriques
#  Inserer juste apres la Cellule B
# =============================================================================

# --- 1. Lags sur la target ---------------------------------------------------
df = add_lags(
    df,
    id_vars=ID_COLS,
    cols_to_lag=[TARGET],
    n_lags=N_LAGS_TARGET,
    year_col='year',
    time_col='time_idx',
)

# --- 2. Lags sur les variables explicatives ----------------------------------
df = add_lags(
    df,
    id_vars=ID_COLS,
    cols_to_lag=vars_expliques,
    n_lags=N_LAGS_VARS,
    year_col='year',
    time_col='time_idx',
)

# --- 3. Convertir les colonnes ID en type category ---------------------------
#     LightGBM les traitera nativement (splits par modalite, pas one-hot)
for col in ID_COLS:
    if col in df.columns:
        df[col] = df[col].astype("category")

# --- 4. Lister les nouvelles colonnes creees ---------------------------------
lag_cols_target = [f"{TARGET}_lag_{i}" for i in range(1, N_LAGS_TARGET + 1)]
lag_cols_vars   = [f"{v}_lag_{i}" for v in vars_expliques
                   for i in range(1, N_LAGS_VARS + 1)]

print(f"Lags target  : {len(lag_cols_target)} colonnes  → {lag_cols_target}")
print(f"Lags vars    : {len(lag_cols_vars)} colonnes  → {lag_cols_vars[:6]}...")
print(f"Categoriques : {ID_COLS}")
print(f"NaN introduits par les lags : {df[lag_cols_target + lag_cols_vars].isna().sum().sum():,}")
print(f"Lignes totales : {len(df):,}")


# %% =========================================================================
#  CELLULE D — Mise a jour de FEATURES_MODEL
#  REMPLACER la definition actuelle de FEATURES_MODEL par ceci
# =============================================================================

# --- Features "courantes" (celles que tu utilisais deja, SANS les lags) ------
# IMPORTANT : adapte cette liste a tes features actuelles.
# Si tu as deja une variable `features_courantes` ou `FEATURES_MODEL`,
# pars de celle-la et ajoute les nouvelles colonnes :

# Option 1 : si tu as deja une liste `features_courantes`
# FEATURES_MODEL = features_courantes + lag_cols_target + lag_cols_vars + ID_COLS

# Option 2 : construire automatiquement depuis df
#   On prend toutes les colonnes numeriques + les categoriques,
#   SAUF la target et les colonnes de temps
cols_exclure = {TARGET, 'year', 'quarter', 'time_idx', 'Time', 'annee', 'period'}
features_numeriques = [c for c in df.select_dtypes(include='number').columns
                       if c not in cols_exclure]
features_categoriques = [c for c in df.select_dtypes(include='category').columns]

FEATURES_MODEL = sorted(set(features_numeriques + features_categoriques))

print(f"\nFEATURES_MODEL : {len(FEATURES_MODEL)} features au total")
print(f"  dont {len([c for c in FEATURES_MODEL if 'lag' in c])} lags")
print(f"  dont {len([c for c in FEATURES_MODEL if c in ID_COLS])} categoriques")

# Verification : toutes les colonnes existent dans df
manquantes = [c for c in FEATURES_MODEL if c not in df.columns]
assert not manquantes, f"Colonnes manquantes dans df : {manquantes}"


# %% =========================================================================
#  CELLULE E — MODIFIER le modele baseline (celui SANS tuning)
#  Dans la cellule ou tu entraines le modele avec PARAMS_BASELINE :
#  AJOUTER categorical_feature dans le .fit()
# =============================================================================

# Si tu utilises deja la fonction entrainer() qui detecte automatiquement
# les colonnes category, tu n'as RIEN A CHANGER — elle le fait deja :
#
#   def entrainer(params, X_tr, y_tr, X_va, y_va, es=EARLY_STOP):
#       categorical_feature = [col for col in X_tr.columns
#                              if X_tr[col].dtype.name == 'category']
#       ...
#       m.fit(X_tr, y_tr, ..., categorical_feature=categorical_feature, ...)
#
# MAIS si tu entraines directement avec modele.fit() sans passer par
# entrainer(), alors ajoute cette ligne AVANT le .fit() :

# categorical_feature = [col for col in X_tr.columns
#                        if X_tr[col].dtype.name == 'category']
#
# Et dans le .fit(), ajoute : categorical_feature=categorical_feature


# %% =========================================================================
#  CELLULE F — Verification rapide (optionnelle mais recommandee)
#  A executer apres le split train/val/test pour verifier que tout est ok
# =============================================================================

def verifier_lags(df_check, id_vars, target, n_lags):
    """Verifie que les lags sont corrects sur un groupe aleatoire."""
    import random
    random.seed(42)

    # Prendre un groupe au hasard
    groupes = df_check.groupby(id_vars, observed=False)
    groupe_ids = [name for name, g in groupes if len(g) >= n_lags + 1]
    if not groupe_ids:
        print("Pas assez de groupes avec suffisamment de periodes pour verifier")
        return

    choisi = random.choice(groupe_ids)
    g = groupes.get_group(choisi).sort_values('time_idx').reset_index(drop=True)

    print(f"Verification sur le groupe : {choisi}")
    print(f"{'time_idx':>10} {'target':>15} {'lag_1':>15} {'lag_2':>15}")
    print("-" * 58)
    for i in range(min(6, len(g))):
        row = g.iloc[i]
        t = f"{row['time_idx']:.0f}"
        v = f"{row[target]:,.0f}"
        l1 = f"{row[f'{target}_lag_1']:,.0f}" if pd.notna(row[f'{target}_lag_1']) else "NaN"
        l2 = f"{row[f'{target}_lag_2']:,.0f}" if pd.notna(row[f'{target}_lag_2']) else "NaN"
        print(f"{t:>10} {v:>15} {l1:>15} {l2:>15}")

    # Verification automatique
    erreurs = 0
    for i in range(len(g)):
        for lag in range(1, n_lags + 1):
            val = g.iloc[i][f'{target}_lag_{lag}']
            if i - lag >= 0:
                attendu = g.iloc[i - lag][target]
                if pd.notna(val) and val != attendu:
                    erreurs += 1
            else:
                if pd.notna(val):
                    erreurs += 1
    print(f"\nErreurs detectees : {erreurs}")

# Decommente pour executer :
# verifier_lags(df, ID_COLS, TARGET, N_LAGS_TARGET)

