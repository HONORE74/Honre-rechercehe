# ================================================================
# ANONYMISATION + FILTRAGE + EXPORT
# ================================================================
# ┌─────────────────── PARAMETRES A MODIFIER ───────────────────┐
COLS_ANONYMES = {
    "Companies":   "COMP",
    "Partner":     "PART",
    "Risk":        "RISK",
    "Activity":    "ACT",
    "Periodicity": "PERIOD",
    "pays":        "PAY",
}

CIBLE      = "RBNS_eop"
SEUIL_BAS  = 5
SEUIL_HAUT = 8_000_000

ORDRE = "aleatoire"          # "aleatoire" | "frequence" | "alpha"
SEED  = 42

CHEMIN_EXPORT = r"oooop/rrrr/rr/r/colokr/base_anonymisee.csv"   # <-- votre chemin
# └──────────────────────────────────────────────────────────────┘

import pandas as pd, numpy as np
from pathlib import Path

manquantes = [c for c in list(COLS_ANONYMES) + [CIBLE] if c not in df.columns]
if manquantes:
    raise KeyError(f"Colonnes absentes : {manquantes}")

df_anon = df.copy()
CORRESPONDANCES = {}
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- 1. anonymisation
for col, prefixe in COLS_ANONYMES.items():
    serie = df_anon[col].astype("object")
    valeurs = serie.dropna().astype(str)

    if ORDRE == "frequence":
        modalites = valeurs.value_counts().index.tolist()
    elif ORDRE == "alpha":
        modalites = sorted(valeurs.unique())
    else:
        modalites = list(rng.permutation(sorted(valeurs.unique())))

    largeur = max(2, len(str(len(modalites))))
    mapping = {v: f"{prefixe}_{i:0{largeur}d}" for i, v in enumerate(modalites, 1)}
    CORRESPONDANCES[col] = mapping

    nouvelle = serie.astype(str).map(mapping)
    df_anon[col] = (nouvelle.astype("category")
                    if str(df[col].dtype) == "category" else nouvelle)
    print(f"  {col:<14} -> {prefixe}_XX   {len(mapping):>5} modalites")

# ---------------------------------------------------------------- 2. suppression des manquants
n0 = len(df_anon)
df_anon = df_anon.dropna(subset=list(COLS_ANONYMES) + [CIBLE])
n_na = n0 - len(df_anon)

# ---------------------------------------------------------------- 3. filtrage
cible  = df_anon[CIBLE]
m_bas  = cible < SEUIL_BAS
m_haut = cible > SEUIL_HAUT
df_final = df_anon.loc[(cible >= SEUIL_BAS) & (cible <= SEUIL_HAUT)].copy()

print("\n" + "=" * 70)
print(f"FILTRAGE '{CIBLE}'  —  intervalle [{SEUIL_BAS:,} ; {SEUIL_HAUT:,}]")
print("=" * 70)
print(f"  Lignes initiales           : {n0:>12,}")
print(f"  Supprimees (manquants)     : {n_na:>12,}   ({100*n_na/n0:>5.2f} %)")
print(f"  Supprimees < {SEUIL_BAS:<13,} : {int(m_bas.sum()):>12,}   ({100*m_bas.sum()/n0:>5.2f} %)")
print(f"  Supprimees > {SEUIL_HAUT:<13,} : {int(m_haut.sum()):>12,}   ({100*m_haut.sum()/n0:>5.2f} %)")
print("-" * 70)
print(f"  Lignes conservees          : {len(df_final):>12,}   ({100*len(df_final)/n0:>5.2f} %)")
print(f"  Total supprime             : {n0-len(df_final):>12,}   "
      f"({100*(n0-len(df_final))/n0:>5.2f} %)")
print("=" * 70)

c = df_final[CIBLE]
print(f"\nCible : min {c.min():,.2f} | mediane {c.median():,.2f} | max {c.max():,.2f}")
print(f"\n{df_final[list(COLS_ANONYMES)].head(8).to_string(index=False)}")

# ---------------------------------------------------------------- 4. export
chemin = Path(CHEMIN_EXPORT)
chemin.parent.mkdir(parents=True, exist_ok=True)
df_final.to_csv(chemin, index=False)
print(f"\nExporte : {chemin.resolve()}")
print(f"{len(df_final):,} lignes x {len(df_final.columns)} colonnes | "
      f"{chemin.stat().st_size/1e6:.2f} Mo")

# Cle de correspondance conservee en memoire dans CORRESPONDANCES (non exportee)
