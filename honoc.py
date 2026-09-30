# =============================================================================
#  ANONYMISATION + FILTRE DES SINISTRES
#  A placer JUSTE APRES le chargement de df_model (read_parquet), AVANT la
#  creation des lags et des variables. Meme bloc pour V1, V2, V3.
# =============================================================================
import os
import numpy as np
import pandas as pd

# ---- Reglages ---------------------------------------------------------------
ANONYMISER = {"Partner": "Part", "Companies": "Comp", "Risk": "Risque"}
# Pour anonymiser d'autres colonnes, les ajouter ici, par exemple :
# ANONYMISER.update({"Lob": "Lob", "Activity": "Activ", "Periodicity": "Perio"})

COL_SINISTRES = "Claims_incurred"
SINISTRE_MIN, SINISTRE_MAX = 100, 5_000_000      # bornes incluses

# "sous_portefeuille" : retire les sous-portefeuilles ayant AU MOINS un
#     trimestre hors bornes. Les series restent completes : les lags restent justes.
# "ligne" : retire seulement les trimestres hors bornes. A n'utiliser qu'APRES
#     add_lags(), sinon lag_1 pointe vers un trimestre plus ancien.
MODE_FILTRE = "sous_portefeuille"
ID_COLS_SP = ["Partner", "Companies", "Lob", "Activity", "Periodicity", "Risk"]

# Table de correspondance vrai nom <-> nom anonyme : la MEME pour V1, V2, V3 et
# pour toutes les tables (results_v2, anomalies_prio...). A GARDER PRIVEE : elle
# contient les vrais noms (ne pas la mettre dans le memoire ni dans git).
CHEMIN_CORRESPONDANCE = "correspondance_anonymisation.csv"
GRAINE_ANONYMISATION = 42


def anonymiser(df, colonnes=None, chemin=None, graine=GRAINE_ANONYMISATION,
               verifier_fuites=True):
    """Remplace les vrais noms par Part 1, Part 2... Comp 1... Risque 1...
    La numerotation est tiree au hasard (graine fixe), puis enregistree dans
    `chemin` : a chaque execution et dans chaque notebook, un meme vrai nom
    recoit toujours le meme nom anonyme. Un nom jamais vu recoit le numero
    suivant. Relancer la fonction sur une table deja anonymisee ne change rien.
    Les valeurs manquantes restent manquantes ; une colonne categorielle le reste."""
    colonnes = ANONYMISER if colonnes is None else colonnes
    chemin = CHEMIN_CORRESPONDANCE if chemin is None else chemin
    if os.path.exists(chemin):
        corr = pd.read_csv(chemin, dtype=str)
    else:
        corr = pd.DataFrame(columns=["colonne", "original", "anonyme"], dtype=str)
    df = df.copy()
    originaux_par_col = {}
    for col, prefixe in colonnes.items():
        if col not in df.columns:
            print(f"[anonymisation] colonne absente, ignoree : {col}")
            continue
        etait_categorielle = isinstance(df[col].dtype, pd.CategoricalDtype)
        valeurs = df[col].astype("string").str.strip()
        connues = corr[corr["colonne"] == col]
        table = dict(zip(connues["original"], connues["anonyme"]))
        deja_anonymes = set(table.values())
        nouvelles = sorted(set(valeurs.dropna()) - set(table) - deja_anonymes)
        if nouvelles:
            np.random.default_rng(graine + len(table)).shuffle(nouvelles)
            for i, v in enumerate(nouvelles, start=len(table) + 1):
                table[v] = f"{prefixe} {i}"
            corr = pd.concat([corr, pd.DataFrame({
                "colonne": col, "original": nouvelles,
                "anonyme": [table[v] for v in nouvelles]})], ignore_index=True)
        table.update({a: a for a in deja_anonymes})     # deja anonymise : inchange
        nouvelle_col = valeurs.map(table).astype(object)
        nouvelle_col[valeurs.isna()] = np.nan
        df[col] = nouvelle_col.astype("category") if etait_categorielle else nouvelle_col
        originaux_par_col[col] = set(connues["original"]) | set(nouvelles)
        print(f"[anonymisation] {col} : {valeurs.nunique()} valeurs -> "
              f"{prefixe} 1 ... {prefixe} {len(set(table.values()))}"
              + (f" ({len(nouvelles)} nouvelle(s))" if nouvelles else ""))
    corr.to_csv(chemin, index=False)

    if verifier_fuites:   # un vrai nom survit-il dans une AUTRE colonne texte ?
        tous = {v for s in originaux_par_col.values() for v in s if len(v) >= 3}
        for c in df.columns:
            if c in colonnes or not (df[c].dtype == object
                                     or isinstance(df[c].dtype, pd.CategoricalDtype)
                                     or pd.api.types.is_string_dtype(df[c])):
                continue
            uniques = pd.Series(df[c].dropna().astype(str).unique())
            if len(uniques) > 50_000:
                continue
            fuites = [v for v in tous if uniques.str.contains(v, regex=False).any()]
            if fuites:
                print(f"[anonymisation] ATTENTION : la colonne '{c}' contient encore "
                      f"des vrais noms (ex. {fuites[:3]}). A anonymiser ou supprimer.")
    return df


def filtrer_sinistres(df, col=None, bas=None, haut=None, mode=None, id_cols=None):
    """Garde les sinistres compris entre `bas` et `haut` (bornes incluses).
    Les montants manquants ne sont ni gardes ni exclus par eux-memes."""
    col = COL_SINISTRES if col is None else col
    bas = SINISTRE_MIN if bas is None else bas
    haut = SINISTRE_MAX if haut is None else haut
    mode = MODE_FILTRE if mode is None else mode
    id_cols = [c for c in (ID_COLS_SP if id_cols is None else id_cols) if c in df.columns]
    y = pd.to_numeric(df[col], errors="coerce")
    hors = y.notna() & ~y.between(bas, haut)

    if mode == "ligne":
        garde = ~hors
    elif mode == "sous_portefeuille":
        if not id_cols:
            raise ValueError("Aucune colonne d'identification trouvee pour le mode "
                             "'sous_portefeuille' : renseigner ID_COLS_SP.")
        sp_hors = hors.groupby([df[c] for c in id_cols], observed=True,
                               dropna=False).transform("any")
        garde = ~sp_hors.to_numpy()
    else:
        raise ValueError("MODE_FILTRE doit valoir 'sous_portefeuille' ou 'ligne'.")

    res = df[garde].copy()
    n_sp = lambda d: len(d[id_cols].drop_duplicates()) if id_cols else float("nan")
    y_res = pd.to_numeric(res[col], errors="coerce")
    print(f"[filtre] {col} entre {bas:,.0f} et {haut:,.0f} (mode {mode})".replace(",", " "))
    print(f"[filtre]   lignes : {len(df):,} -> {len(res):,} "
          f"({100 * len(res) / max(len(df), 1):.1f} % conservees)".replace(",", " "))
    if id_cols:
        print(f"[filtre]   sous-portefeuilles : {n_sp(df)} -> {n_sp(res)}")
    print(f"[filtre]   dont lignes hors bornes : {int(hors.sum())} "
          f"(< {bas:,.0f} : {int((y < bas).sum())} ; > {haut:,.0f} : {int((y > haut).sum())})"
          .replace(",", " "))
    if y.isna().any():
        print(f"[filtre]   montants manquants conserves : {int(y_res.isna().sum())}")
    if len(res):
        print(f"[filtre]   {col} apres filtre : min {y_res.min():,.0f} ; "
              f"max {y_res.max():,.0f}".replace(",", " "))
    return res


# ---- Application ------------------------------------------------------------
df_model = anonymiser(df_model)
df_model = filtrer_sinistres(df_model)
# df = df_model.copy()        <- la suite de ton code, inchangee




















suspects = {"", "NA", "N/A", "n/a", "NaN", "nan", "None", "null", "NULL", "#N/A", "<NA>"}
for c in ["Partner", "Companies", "Risk"]:
    v = df_model[c].astype("string").str.strip()
    print(c, "->", v[v.isin(suspects)].value_counts().to_dict())

















cols = ["Partner", "Companies", "Risk"]
avant = {c: df_model[c].astype("string").str.strip().nunique() for c in cols}
df_model = anonymiser(df_model)
apres = {c: df_model[c].nunique() for c in cols}
print(avant)
print(apres)
