
import numpy as np, pandas as pd

class EncodeurTabulaire:
    """Remplacement direct de skrub.TableVectorizer, meme API
    (fit / transform / fit_transform / get_feature_names_out).

    Les categorielles deviennent des 'category' pandas avec modalites FIGEES
    a l'ajustement. LightGBM les traite nativement : pas de colonnes creuses,
    et une modalite inconnue au test devient simplement une valeur manquante.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.colonnes_ = list(X.columns)
        self.cat_cols_ = [c for c in X.columns
                          if X[c].dtype == object
                          or str(X[c].dtype) in ("category", "string")]
        self.categories_ = {}
        for c in self.cat_cols_:
            v = X[c].astype("string")
            self.categories_[c] = pd.Index(sorted(v.dropna().unique()))
        return self

    def transform(self, X):
        X = pd.DataFrame(X)[self.colonnes_].copy()
        for c in self.colonnes_:
            if c in self.cat_cols_:
                X[c] = pd.Categorical(X[c].astype("string"),
                                      categories=self.categories_[c])
            else:
                X[c] = pd.to_numeric(X[c], errors="coerce")
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.colonnes_, dtype=object)


















def compute_conformal_quantile(y_calib, y_lo_calib, y_hi_calib, alpha: float):
    y  = np.asarray(y_calib,    dtype=float)
    lo = np.asarray(y_lo_calib, dtype=float)
    hi = np.asarray(y_hi_calib, dtype=float)
    scores = np.maximum(lo - y, y - hi)

    # LA LIGNE QUI MANQUAIT
    finis = np.isfinite(scores)
    if (~finis).any():
        print(f"  {(~finis).sum():,} / {len(scores):,} scores non finis ecartes "
              f"(cible manquante en calibration).")
    scores = scores[finis]

    n = len(scores)
    if n == 0:
        raise ValueError("Calibration vide apres retrait des NaN : la cible est "
                         "absente sur TOUTE la periode de calibration. "
                         "Augmenter N_CALIB ou reculer le decoupage.")
    if n < 30:
        print(f"ATTENTION : seulement {n} observations en calibration.")

    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    Q_hat = np.quantile(scores, q_level, method="higher")
    return Q_hat, scores














print("calib :", len(df_calib), "lignes |", df_calib[TARGET].isna().sum(), "cibles NaN",
      f"({df_calib[TARGET].isna().mean():.1%})")
print("periodes calib :", sorted(df_calib["time_idx"].unique()))
print("y_lo NaN :", np.isnan(pipeline_lo.predict(df_calib[FEATURE_COLS])).sum())








# -*- coding: utf-8 -*-
# =============================================================================
#  CQR (Conformalized Quantile Regression) - NOTEBOOK ORDONNE DE BOUT EN BOUT
#  Memoire : detection et priorisation d'observations atypiques (IFRS 17 / S2)
#
#  Chaque bloc "# %%" est une cellule Jupyter. A executer DANS L'ORDRE.
#  Pre-requis : la variable df_model doit exister dans la session.
#
#  Contrainte projet respectee : AUCUN logarithme, nulle part (ni donnees,
#  ni axes). Toute mise a l'echelle passe par des divisions ou des rangs.
# =============================================================================


# %% ==========================================================================
#  CELLULE 0 - IMPORTS ET CONFIGURATION
# =============================================================================
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

# --- Cible et fuites ---------------------------------------------------------
TARGET = "RBNS_eop"
LEAKS  = ["Conso", "Currencies", "period", "Reinsurance"]

# --- Parametres conformes ----------------------------------------------------
ALPHA   = 0.10   # couverture visee = 1 - ALPHA = 90 %
N_CALIB = 2      # trimestres reserves a la calibration (1 est trop peu, cf. §)
N_TEST  = 1      # trimestres reserves au test

# --- Periodes a ecarter (trimestres non clotures ou corrompus) ---------------
# Present dans pipeline_conformal_complet.py:88 et absent du notebook : c'est
# une source de NaN independante du filtre sur la cible.
EXCLURE_PERIODES = [(2024, 3)]

# --- Encodage des categorielles ----------------------------------------------
#   "natif"    -> LightGBM gere les categorielles directement (RECOMMANDE)
#   "pipeline" -> ColumnTransformer scikit-learn (impute + standardise + one-hot)
MODE_ENCODAGE = "natif"

MIN_FREQ_MODALITE = 20   # utilise seulement en mode "pipeline"

# --- Artefact du modele LightGBM entraine dans l'autre notebook --------------
CHEMIN_ARTEFACT = "artefacts_modele/modele_final.joblib"

QUARTER_MAP  = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
RANDOM_STATE = 42

ID_COLS_BRUTS = ["Partner", "Companies", "Lob", "Activity", "Periodicity", "Risk"]

print(f"Configuration : ALPHA={ALPHA} | N_CALIB={N_CALIB} | N_TEST={N_TEST} "
      f"| encodage={MODE_ENCODAGE}")


# %% ==========================================================================
#  CELLULE 1 - PREPARATION TEMPORELLE
#
#  L'index temporel est construit AVANT toute autre chose. Dans la version
#  precedente il etait cree apres la cellule qui fait df = df_model.copy(),
#  donc re-executer cette cellule effacait time_idx et cassait le decoupage.
# =============================================================================
def construire_time_idx(frame: pd.DataFrame, annee_ref=None) -> pd.DataFrame:
    """Ajoute year / quarter / time_idx a un DataFrame.

    annee_ref DOIT etre la meme pour tous les DataFrames que l'on compte
    joindre ensuite. Si chacun utilise son propre min(), les time_idx sont
    decales l'un par rapport a l'autre et la jointure echoue silencieusement.
    """
    f = frame.copy()

    if "year" in f.columns:
        f["year"] = f["year"].astype(int)
    elif "annee" in f.columns:
        f["year"] = f["annee"].astype(int)
    else:
        raise KeyError("Ni 'year' ni 'annee' : periode impossible a reconstruire.")

    if "quarter" in f.columns:
        f["quarter"] = f["quarter"].astype(int)
    elif "Time" in f.columns:
        q = f["Time"].astype(str).str.strip().str.upper().map(QUARTER_MAP)
        if q.isna().any():
            raise ValueError(
                f"Valeurs de 'Time' non reconnues : "
                f"{f.loc[q.isna(), 'Time'].unique()[:5].tolist()}")
        f["quarter"] = q.astype(int)
    else:
        raise KeyError("Ni 'quarter' ni 'Time' : periode impossible a reconstruire.")

    ref = f["year"].min() if annee_ref is None else annee_ref
    f["time_idx"] = (f["year"] - ref) * 4 + f["quarter"]
    return f


df = construire_time_idx(df_model.copy())
ANNEE_REF = int(df["year"].min())          # reference partagee, a reutiliser tel quel

n_avant = len(df)
for an, tr in EXCLURE_PERIODES:
    df = df[~((df["year"] == an) & (df["quarter"] == tr))]
print(f"Periodes ecartees {EXCLURE_PERIODES} : {n_avant - len(df):,} lignes retirees")

df = df.sort_values("time_idx").reset_index(drop=True)

ID_COLS = [c for c in ID_COLS_BRUTS if c in df.columns]

apercu = (df.groupby(["year", "quarter", "time_idx"], as_index=False)
            .agg(n_lignes=(TARGET, "size"), n_cible_nan=(TARGET, lambda s: s.isna().sum()))
            .sort_values("time_idx"))
apercu["pct_nan"] = (apercu["n_cible_nan"] / apercu["n_lignes"]).map("{:.1%}".format)
print("\nPeriodes disponibles (surveiller la colonne pct_nan) :")
print(apercu.to_string(index=False))


# %% ==========================================================================
#  CELLULE 2 - IMPORT DES PREDICTIONS LightGBM DE L'AUTRE NOTEBOOK
#
#  ORDRE CRITIQUE : la jointure se fait ICI, sur les identifiants D'ORIGINE,
#  AVANT l'anonymisation de la cellule 3. Joindre apres coup ne peut pas
#  marcher : df porterait "PART_01" et l'artefact le libelle reel.
# =============================================================================
paquet = joblib.load(CHEMIN_ARTEFACT)
print(f"Artefact du {paquet.get('date', '?')} | MAE test enregistree : "
      f"{paquet.get('mae_test', float('nan')):,.0f}")

# --- Reconstruction des predictions, a l'identique de votre cellule d'origine
X_art = paquet["X_test"][paquet["features"]].copy()
for c in paquet["categorielles"]:
    X_art[c] = pd.Categorical(X_art[c].astype(str),
                              categories=[str(v) for v in paquet["categories"][c]])

df_final = paquet["infos_test"].reset_index(drop=True).copy()
df_final["y_pred"]     = np.clip(paquet["modele"].predict(X_art), 0, None)
df_final["y_obs_art"]  = np.asarray(paquet["y_test"], dtype=float)

mae_recalc = (df_final["y_pred"] - df_final["y_obs_art"]).abs().mean()
print(f"MAE recalculee : {mae_recalc:,.0f}  ({len(df_final):,} predictions)")

# --- Meme referentiel temporel que df ---------------------------------------
df_final = construire_time_idx(df_final, annee_ref=ANNEE_REF)

cles_manquantes = [c for c in ID_COLS if c not in df_final.columns]
if cles_manquantes:
    raise KeyError(
        f"L'artefact ne contient pas {cles_manquantes}. La jointure ne peut pas "
        f"etre faite au bon niveau de granularite. Colonnes disponibles dans "
        f"infos_test : {sorted(df_final.columns.tolist())}")

CLE = ID_COLS + ["time_idx"]

# --- Controle d'unicite AVANT jointure (evite l'explosion many-to-many) ------
n_dup = df_final.duplicated(subset=CLE).sum()
if n_dup:
    print(f"ATTENTION : {n_dup:,} doublons sur la cle dans l'artefact, "
          f"agregation par moyenne.")
    externe = (df_final.groupby(CLE, as_index=False)
                       .agg(y_pred=("y_pred", "mean"), y_obs_art=("y_obs_art", "mean")))
else:
    externe = df_final[CLE + ["y_pred", "y_obs_art"]]

# Granularite de df elle-meme : si un couple (identifiants, periode) apparait
# plusieurs fois, y_pred serait duplique a l'identique sur des lignes qui ne
# designent pas le meme objet. Il manque alors une colonne a la cle.
n_dup_df = df.duplicated(subset=CLE).sum()
if n_dup_df:
    print(f"/!\\ {n_dup_df:,} doublons sur {CLE} dans df : la cle metier est "
          f"incomplete, y_pred sera duplique. Ajouter la dimension manquante.")

df = df.merge(externe, on=CLE, how="left", validate="m:1")

# --- Rapport de jointure : ne jamais avancer sans le lire --------------------
periodes_art = sorted(df_final["time_idx"].unique())
print(f"\nPeriodes couvertes par l'artefact : {periodes_art}")
print(f"Taux d'appariement global : {df['y_pred'].notna().mean():.1%} "
      f"({df['y_pred'].notna().sum():,} / {len(df):,} lignes)")
print("\nAppariement par periode :")
print(df.groupby("time_idx")["y_pred"]
        .agg(n="size", apparie=lambda s: s.notna().sum())
        .assign(taux=lambda d: (d["apparie"] / d["n"]).map("{:.1%}".format))
        .to_string())

# --- Coherence des cibles sur les lignes appariees ---------------------------
comm = df["y_pred"].notna() & df[TARGET].notna()
if comm.any():
    ecart = (df.loc[comm, TARGET].astype(float) - df.loc[comm, "y_obs_art"]).abs()
    print(f"\nCoherence des cibles sur {comm.sum():,} lignes appariees : "
          f"ecart max = {ecart.max():,.4f}")
    if ecart.max() > 1e-6:
        print("  /!\\ Les cibles divergent : l'artefact et df_model ne portent pas "
              "sur la meme base. Verifier avant d'exploiter y_pred.")
df = df.drop(columns=["y_obs_art"])


# %% ==========================================================================
#  CELLULE 3 - ANONYMISATION DES IDENTIFIANTS ET TYPAGE
#
#  Le dictionnaire de correspondance est CONSERVE : sans lui, impossible de
#  redonner au metier le nom du sous-portefeuille signale comme atypique.
# =============================================================================
prefix_map = {"Partner": "PART", "Companies": "COMP", "Lob": "LOB",
              "Activity": "ACT", "Periodicity": "PER", "Risk": "RISK"}

mappings_anonymisation = {}
for col in ID_COLS:
    prefixe = prefix_map.get(col, col.upper()[:4])
    valeurs = sorted(df[col].dropna().unique())
    mapping = {v: f"{prefixe}_{i:02d}" for i, v in enumerate(valeurs, start=1)}
    mappings_anonymisation[col] = mapping
    df[col] = df[col].map(mapping)
    print(f"{col:14s} -> {len(mapping):3d} modalites anonymisees")

joblib.dump(mappings_anonymisation, "mapping_anonymisation.joblib")

# --- Selection des features --------------------------------------------------
# 'Time' est ecarte : il encode la periode. En "Q1".."Q4" il donne la
# saisonnalite en clair ; en "2024Q3" chaque periode est une modalite jamais
# vue au test, et LightGBM enverrait tout le test dans une branche arbitraire.
EXCLUDE_COLS = ([TARGET, "time_idx", "year", "quarter", "annee", "Time", "y_pred"]
                + LEAKS)

FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE_COLS]

CATEGORIELLES = [c for c in FEATURE_COLS
                 if df[c].dtype == object or str(df[c].dtype) == "category"]
for c in CATEGORIELLES:
    df[c] = df[c].astype("category")

print(f"\n{len(FEATURE_COLS)} features, dont {len(CATEGORIELLES)} categorielles")
print(f"Categorielles : {CATEGORIELLES}")
absentes = [c for c in EXCLUDE_COLS if c not in df.columns]
if absentes:
    print(f"Note : colonnes listees en exclusion mais absentes de df : {absentes}")


# %% ==========================================================================
#  CELLULE 4 - DECOUPAGE TEMPOREL train / calibration / test
#
#  La calibration doit etre disjointe de l'entrainement ET anterieure au test.
#  Calibrer sur des donnees vues a l'entrainement donne des scores de
#  non-conformite trop petits, donc des intervalles trop etroits et une
#  couverture reelle inferieure au niveau annonce.
# =============================================================================
def decoupage_chronologique(df_raw, time_col="time_idx",
                            n_calib=N_CALIB, n_test=N_TEST):
    periodes = sorted(df_raw[time_col].unique())
    if len(periodes) <= n_calib + n_test:
        raise ValueError(
            f"{len(periodes)} periodes disponibles, il en faut au moins "
            f"{n_calib + n_test + 1} pour reserver {n_calib} de calibration "
            f"et {n_test} de test.")

    p_test  = periodes[-n_test:]
    p_calib = periodes[-(n_test + n_calib):-n_test]
    p_train = periodes[:-(n_test + n_calib)]

    blocs = {}
    for nom, per in [("train", p_train), ("calib", p_calib), ("test", p_test)]:
        bloc = df_raw[df_raw[time_col].isin(per)].copy().reset_index(drop=True)
        y = bloc[TARGET].astype(float)
        n_util = int(np.isfinite(y).sum())
        print(f"{nom:6s} periodes {per[0]}-{per[-1]} | {len(bloc):7,} lignes | "
              f"{n_util:7,} cibles exploitables ({y.isna().mean():5.1%} NaN)")
        if n_util == 0:
            raise RuntimeError(
                f"Le bloc '{nom}' n'a AUCUNE cible exploitable. Les periodes "
                f"{per} ne sont pas cloturees : les ajouter a EXCLURE_PERIODES "
                f"ou augmenter N_CALIB / N_TEST pour reculer le decoupage.")
        blocs[nom] = bloc
    return blocs["train"], blocs["calib"], blocs["test"], \
           {"train": p_train, "calib": p_calib, "test": p_test}


df_train, df_calib, df_test, periodes_dict = decoupage_chronologique(df)


# %% ==========================================================================
#  CELLULE 5 - ENCODEURS (les deux voies, utilisees correctement)
# =============================================================================
def _noms_propres(noms):
    """LightGBM refuse les caracteres JSON speciaux dans les noms de features.
    Un libelle metier contenant [ ] < > fait echouer .fit() sans rapport
    apparent avec le probleme. On nettoie et on deduplique."""
    vus, sortie = {}, []
    for n in (re.sub(r"[^\w]+", "_", str(x)).strip("_") for x in noms):
        k = vus.get(n, 0)
        vus[n] = k + 1
        sortie.append(n if k == 0 else f"{n}__{k}")
    return sortie


class EncodeurNatif:
    """LightGBM traite les categorielles nativement.

    Sur des identifiants a forte cardinalite (Partner, Companies), le
    regroupement par gradient de LightGBM est superieur au one-hot : il ne
    cree pas des centaines de colonnes creuses et expose cat_smooth / cat_l2.
    C'est aussi l'encodage du modele sauvegarde dans l'artefact, donc le seul
    qui rende les deux modeles comparables.
    """

    def fit(self, X):
        self.colonnes_ = list(X.columns)
        self.cat_cols_ = [c for c in X.columns
                          if str(X[c].dtype) in ("category", "object")]
        # Les colonnes ont ete typees "category" sur le df COMPLET en cellule 3 :
        # la liste des modalites est donc celle de tout le jeu, train inclus.
        # Ce n'est pas une fuite, aucune information sur la cible n'intervient,
        # seulement la liste des modalites existantes. C'est meme necessaire :
        # sans elle, les codes categoriels de calib et test ne correspondraient
        # plus a ceux du train et les predictions seraient fausses SANS erreur.
        # Une modalite absente du train ne recoit simplement aucun split.
        self.categories_ = {c: [str(v) for v in X[c].astype("category").cat.categories]
                            for c in self.cat_cols_}
        return self

    def transform(self, X):
        X = X[self.colonnes_].copy()
        for c in self.cat_cols_:
            X[c] = pd.Categorical(X[c].astype(str), categories=self.categories_[c])
        return X

    @property
    def cat_features(self):
        return self.cat_cols_ if self.cat_cols_ else "auto"


class EncodeurPipeline:
    """ColumnTransformer scikit-learn : impute + standardise + one-hot.

    min_frequency regroupe les modalites rares dans une categorie
    'infrequent'. Sans ce garde-fou, un identifiant a 400 modalites produit
    400 colonnes creuses et les hyperparametres regles pour l'espace natif
    (min_child_samples=139, colsample_bytree=0.74) deviennent absurdes.
    """

    def fit(self, X):
        self.colonnes_ = list(X.columns)
        num = X.select_dtypes(include=["number"]).columns.tolist()
        cat = X.select_dtypes(include=["object", "category"]).columns.tolist()

        try:
            ohe = OneHotEncoder(handle_unknown="infrequent_if_exist",
                                sparse_output=False,
                                min_frequency=MIN_FREQ_MODALITE)
        except TypeError:                      # scikit-learn < 1.2
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        self.ct_ = ColumnTransformer(
            [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                               ("sc",  StandardScaler())]), num),
             ("cat", Pipeline([("imp", SimpleImputer(strategy="constant",
                                                     fill_value="__manquant__")),
                               ("oh",  ohe)]), cat)],
            remainder="drop")
        self.ct_.fit(X)
        self.noms_ = _noms_propres(self.ct_.get_feature_names_out())
        print(f"  one-hot : {len(self.colonnes_)} colonnes -> {len(self.noms_)} features")
        return self

    def transform(self, X):
        M = self.ct_.transform(X[self.colonnes_])
        return pd.DataFrame(M, columns=self.noms_, index=X.index)

    @property
    def cat_features(self):
        return "auto"          # tout est numerique apres one-hot


def construire_encodeur(mode):
    if mode == "natif":
        return EncodeurNatif()
    if mode == "pipeline":
        return EncodeurPipeline()
    raise ValueError("MODE_ENCODAGE doit valoir 'natif' ou 'pipeline'.")


# %% ==========================================================================
#  CELLULE 6 - ENTRAINEMENT DES DEUX MODELES QUANTILES
# =============================================================================
def quantile_model_factory(alpha_level: float):
    return lgb.LGBMRegressor(
        objective="quantile", alpha=alpha_level,
        learning_rate=0.03239, num_leaves=84, min_child_samples=139,
        colsample_bytree=0.7365, subsample=0.982, subsample_freq=1,
        reg_alpha=8.6629, reg_lambda=19.963, max_bin=451,
        n_estimators=2000, max_depth=7,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)


q_lo_level, q_hi_level = ALPHA / 2, 1 - ALPHA / 2

X_train_brut = df_train[FEATURE_COLS]
y_train      = df_train[TARGET].astype(float)

valides = np.isfinite(y_train)
X_train_brut, y_train = X_train_brut.loc[valides], y_train.loc[valides]
print(f"Train : {len(y_train):,} lignes exploitables "
      f"({(~valides).sum():,} ecartees pour cible manquante)")

# L'encodeur est ajuste sur le TRAIN SEUL, jamais sur calib ni test.
encodeur = construire_encodeur(MODE_ENCODAGE).fit(X_train_brut)
X_train  = encodeur.transform(X_train_brut)

modele_lo = quantile_model_factory(q_lo_level)
modele_hi = quantile_model_factory(q_hi_level)
modele_lo.fit(X_train, y_train, categorical_feature=encodeur.cat_features)
modele_hi.fit(X_train, y_train, categorical_feature=encodeur.cat_features)

print(f"Modeles quantiles {q_lo_level:.3f} et {q_hi_level:.3f} entraines "
      f"(encodage {MODE_ENCODAGE}, {X_train.shape[1]} features).")


# %% ==========================================================================
#  CELLULE 7 - QUANTILE CONFORME SUR LA CALIBRATION
#
#  C'est l'etape qui transforme un intervalle quantile SANS garantie en un
#  intervalle a couverture garantie a distance finie.
# =============================================================================
def quantile_conforme_cqr(y_calib, y_lo_calib, y_hi_calib, alpha: float):
    """Score CQR de Romano, Patterson & Candes (2019) :
        E_i = max( q_lo(X_i) - Y_i ,  Y_i - q_hi(X_i) )
    negatif quand l'observation tombe dans l'intervalle nominal, positif sinon.

    Le quantile est pris au rang ceil((n+1)(1-alpha))/n et NON au quantile
    (1-alpha) classique : cette correction d'echantillon fini est ce qui rend
    la garantie valide pour tout n. L'oublier donne une sous-couverture.
    """
    y  = np.asarray(y_calib,     dtype=float)
    lo = np.asarray(y_lo_calib,  dtype=float)
    hi = np.asarray(y_hi_calib,  dtype=float)
    scores = np.maximum(lo - y, y - hi)

    finis = np.isfinite(scores)
    if (~finis).any():
        print(f"  {(~finis).sum():,} / {len(scores):,} scores non finis ecartes "
              f"(cible manquante en calibration).")
    scores = scores[finis]
    n = len(scores)

    if n == 0:
        raise ValueError(
            "Calibration vide apres retrait des NaN : la cible est absente sur "
            "toute la periode de calibration. Augmenter N_CALIB ou ajouter la "
            "periode fautive a EXCLURE_PERIODES.")

    niveau = np.ceil((n + 1) * (1 - alpha)) / n
    if niveau > 1.0:
        raise ValueError(
            f"n={n} trop petit pour alpha={alpha} : il faut n >= "
            f"{int(np.ceil(1 / alpha)) - 1}. Ecreter le niveau a 1 renverrait "
            f"le score maximal en laissant croire a une garantie inexistante.")

    if n < 100:
        print(f"  n={n} : la garantie reste valide, mais la couverture realisee "
              f"fluctuera d'environ {np.sqrt(alpha * (1 - alpha) / n):.1%}.")

    return float(np.quantile(scores, niveau, method="higher")), scores


X_calib = encodeur.transform(df_calib[FEATURE_COLS])
y_calib = df_calib[TARGET].astype(float)

y_lo_calib = modele_lo.predict(X_calib)
y_hi_calib = modele_hi.predict(X_calib)

n_croise = int((y_lo_calib > y_hi_calib).sum())
if n_croise:
    print(f"{n_croise:,} quantiles croises (lo > hi). Normal en regression "
          f"quantile independante ; la conformalisation le corrige.")

Q_hat, calib_scores = quantile_conforme_cqr(
    y_calib.values, y_lo_calib, y_hi_calib, ALPHA)

print(f"\nQ_hat (marge conforme) = {Q_hat:,.2f}   sur n = {len(calib_scores):,}")
print(f"Part des scores negatifs (deja couverts sans marge) : "
      f"{(calib_scores < 0).mean():.1%}")


# %% ==========================================================================
#  CELLULE 8 - APPLICATION AU TEST
# =============================================================================
X_test_brut = df_test[FEATURE_COLS]
X_test      = encodeur.transform(X_test_brut)
y_test      = df_test[TARGET].astype(float)

y_lo_test = modele_lo.predict(X_test)
y_hi_test = modele_hi.predict(X_test)

colonnes_sortie = ID_COLS + ["year", "quarter", "time_idx"]
if "y_pred" in df_test.columns:
    colonnes_sortie.append("y_pred")          # prediction LightGBM importee

results_test = df_test[colonnes_sortie].copy().reset_index(drop=True)
results_test["y_obs"]       = y_test.values
results_test["borne_basse"] = y_lo_test - Q_hat
results_test["borne_haute"] = y_hi_test + Q_hat
results_test["centre_cqr"]  = (results_test["borne_basse"] + results_test["borne_haute"]) / 2
results_test["largeur"]     = results_test["borne_haute"] - results_test["borne_basse"]

results_test["dans_intervalle"] = results_test["y_obs"].between(
    results_test["borne_basse"], results_test["borne_haute"])

# Depassement signe : > 0 hors intervalle, <= 0 dedans.
results_test["depassement"] = np.maximum(
    results_test["borne_basse"] - results_test["y_obs"],
    results_test["y_obs"] - results_test["borne_haute"])

# Severite normalisee par la largeur locale : c'est ce qui rend comparables
# un sous-portefeuille a 0,7 EUR et un autre a 317 MEUR, sans logarithme.
results_test["severite"] = results_test["depassement"] / results_test["largeur"].replace(0, np.nan)

evaluables = results_test["y_obs"].notna()
couverture = results_test.loc[evaluables, "dans_intervalle"].mean()
largeur_moy = results_test["largeur"].mean()

print(f"Couverture empirique : {couverture:.1%}   (visee {1 - ALPHA:.0%})")
print(f"  sur {evaluables.sum():,} observations evaluables / {len(results_test):,}")
print(f"Largeur mediane : {results_test['largeur'].median():,.0f}")
print(f"Coefficient de variation de la largeur : "
      f"{results_test['largeur'].std() / largeur_moy:.2f}")
print("  (proche de 0 = intervalle constant, donc equivalent a un seuil fixe ;")
print("   nettement > 0 = l'intervalle s'adapte au sous-portefeuille -> H2)")


# %% ==========================================================================
#  CELLULE 9 - COUVERTURE CONDITIONNELLE (hypothese H2)
#
#  La garantie conforme est MARGINALE : elle porte sur la moyenne, pas sur
#  chaque strate. Barber et al. (2021) montrent qu'aucune garantie
#  conditionnelle non triviale n'est atteignable sans hypothese
#  supplementaire. Mesurer l'ecart par strate est donc le seul moyen honnete
#  de savoir si l'intervalle tient sur les gros sous-portefeuilles.
#
#  Les strates sont definies sur la PREDICTION, jamais sur l'observe :
#  decouper sur y_obs revient a conditionner sur le resultat teste.
# =============================================================================
utilise_y_pred = ("y_pred" in results_test.columns
                  and results_test["y_pred"].notna().any())
base_strate = results_test["y_pred"] if utilise_y_pred else results_test["centre_cqr"]
nom_base = "y_pred (LightGBM importe)" if utilise_y_pred else "centre_cqr (repli)"

# Sur une cible Tweedie a forte masse en zero, plusieurs percentiles peuvent
# coincider. pd.cut refuse alors des bornes non strictement croissantes : on
# deduplique et on renomme les strates en consequence plutot que de planter.
seuils = base_strate.quantile([0.50, 0.90, 0.99]).values
noms   = ["P50", "P90", "P99"]
bornes, etiquettes, precedent = [-np.inf], [], "P0"
for seuil, nom in zip(seuils, noms):
    if seuil > bornes[-1]:
        bornes.append(float(seuil))
        etiquettes.append(f"{precedent}-{nom}")
        precedent = nom
bornes.append(np.inf)
etiquettes.append(f"{precedent}+")

if len(etiquettes) < 4:
    print(f"NOTE : percentiles confondus sur {nom_base}, "
          f"{len(etiquettes)} strates au lieu de 4 (masse importante en zero).")

results_test["strate"] = pd.cut(base_strate, bins=bornes, labels=etiquettes)
print(f"Strates construites sur {nom_base}\n")

recap = (results_test[evaluables]
         .groupby("strate", observed=True)
         .agg(n=("y_obs", "size"),
              couverture=("dans_intervalle", "mean"),
              largeur_mediane=("largeur", "median"),
              severite_max=("severite", "max")))
recap["couverture"] = recap["couverture"].map("{:.1%}".format)
print(recap.to_string())

print("\nCouverture par dimension metier :")
for dim in ID_COLS:
    par_dim = (results_test[evaluables].groupby(dim, observed=True)["dans_intervalle"]
               .agg(n="size", couv="mean"))
    par_dim = par_dim[par_dim["n"] >= 30]
    if len(par_dim):
        print(f"  {dim:14s} min={par_dim['couv'].min():.1%}  "
              f"max={par_dim['couv'].max():.1%}  sur {len(par_dim)} modalites (n>=30)")


# %% ==========================================================================
#  CELLULE 10 - PRIORISATION DES ANOMALIES (hypothese H3)
# =============================================================================
anomalies = (results_test[evaluables & (~results_test["dans_intervalle"])]
             .sort_values("severite", ascending=False)
             .reset_index(drop=True))

print(f"{len(anomalies):,} observations hors intervalle "
      f"({len(anomalies) / max(evaluables.sum(), 1):.1%} du test)\n")

cols_aff = ID_COLS + ["year", "quarter", "y_obs", "borne_basse", "borne_haute", "severite"]
if "y_pred" in anomalies.columns:
    cols_aff.insert(-1, "y_pred")

print("Top 20 des sous-portefeuilles a investiguer :")
print(anomalies.head(20)[cols_aff].to_string(
    index=False, float_format=lambda v: f"{v:,.2f}"))

print("\nConcentration des signaux par axe metier :")
for dim in ID_COLS:
    top = anomalies[dim].value_counts().head(3)
    if len(top):
        print(f"  {dim:14s} " + " | ".join(f"{k}={v}" for k, v in top.items()))


# %% ==========================================================================
#  CELLULE 11 - SAUVEGARDE
# =============================================================================
results_test.to_csv("cqr_resultats_test.csv", index=False)
anomalies.to_csv("cqr_anomalies_priorisees.csv", index=False)
pd.DataFrame({"score": calib_scores}).to_csv("cqr_scores_calibration.csv", index=False)

joblib.dump({
    "encodeur": encodeur, "mode_encodage": MODE_ENCODAGE,
    "modele_lo": modele_lo, "modele_hi": modele_hi,
    "Q_hat": Q_hat, "alpha": ALPHA,
    "features": FEATURE_COLS, "id_cols": ID_COLS,
    "periodes": periodes_dict, "annee_ref": ANNEE_REF,
    "n_calib": len(calib_scores), "couverture_test": float(couverture),
}, "artefacts_modele/cqr_final.joblib")

print("Sauvegarde :")
print("  cqr_resultats_test.csv          - toutes les observations du test")
print("  cqr_anomalies_priorisees.csv    - les hors-intervalle, tries par severite")
print("  cqr_scores_calibration.csv      - scores de non-conformite")
print("  artefacts_modele/cqr_final.joblib")
print("  mapping_anonymisation.joblib    - pour redonner les vrais libelles")
