# =============================================================================
#  PIPELINE FINAL - COMPARAISON DE MODELES + SPLIT CONFORMAL PREDICTION
#  Validation des indicateurs IFRS 17 / Solvabilite II
#
#  DECOUPAGE TEMPOREL FIXE (pas de rolling) :
#      TRAIN        2020Q1 -> 2024Q2   (apprentissage des 3 modeles)
#      CALIBRATION  2024Q3             (scores de non-conformite)
#      TEST         2024Q4             (evaluation finale, jamais vu)
#
#  ---------------------------------------------------------------------------
#  POURQUOI PAS DE ROLLING FORECAST ? (justification a reprendre au memoire)
#  ---------------------------------------------------------------------------
#  Le rolling forecast multiplie les points d'evaluation et teste la robustesse
#  temporelle, mais il presente trois inconvenients pour ce travail :
#
#   1. GARANTIE CONFORME DIFFICILE A ENONCER. La prediction conforme garantit
#      P(Y in C(X)) >= 1-alpha pour UN jeu de calibration donne. En rolling, le
#      jeu de calibration change a chaque pas : la couverture observee agrege
#      des garanties distinctes, ce qui rend l'enonce theorique flou.
#      Avec un decoupage fixe, la garantie s'enonce une fois, proprement.
#
#   2. ECART AVEC LA REALITE OPERATIONNELLE. En validation de cloture, on ne
#      reestime pas un modele a chaque trimestre passe : on dispose d'un
#      historique et l'on controle la cloture courante. Le decoupage fixe
#      reproduit exactement cette situation.
#
#   3. COUT ET LISIBILITE. Le rolling reentraine n fois trois modeles, sans
#      apporter de reponse supplementaire a la question de recherche, qui
#      porte sur la capacite a signaler les atypismes d'une cloture donnee.
#
#  LIMITE ASSUMEE : une seule periode de test signifie que les resultats
#  dependent des specificites de 2024Q4. Ce point doit figurer dans les
#  limites de l'etude.
#  ---------------------------------------------------------------------------
#
#  PLAN :
#    BLOC 0   Configuration (A RENSEIGNER)
#    BLOC 1   Chargement et preparation temporelle
#    BLOC 2   Decoupage train / calibration / test
#    BLOC 3   Preprocessing partage (identique pour les 3 modeles)
#    BLOC 4   Entrainement LightGBM / XGBoost / GLM
#    BLOC 5   Evaluation comparative
#    BLOC 6   Visualisations comparatives
#    BLOC 7   Split Conformal Prediction (3 variantes)
#    BLOC 8   Validite, efficacite et couverture conditionnelle
#    BLOC 9   Priorisation des sous-portefeuilles atypiques
#    BLOC 10  Visualisations conformes
#    BLOC 11  Sauvegarde
# =============================================================================

# =============================================================================
# BLOC 0 - CONFIGURATION  (les seules lignes a adapter)
# =============================================================================
import warnings, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
import statsmodels.api as sm

from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 80)

# --- Source des donnees ------------------------------------------------------
#  Le script utilise `df_model` s'il existe deja en memoire (cas normal).
#  CHEMIN_DONNEES ne sert que de repli si `df_model` n'est pas defini.
CHEMIN_DONNEES = "base_indicateurs.csv"   # <-- repli uniquement
TARGET         = "montant"                 # <-- nom de la colonne cible

# --- Colonnes de structure ---------------------------------------------------
COL_ANNEE   = "annee"       # colonne annee (entier ou texte)
COL_TIME    = "Time"        # colonne trimestre au format "Q1".."Q4"
ID_COLS     = ["Partner", "Companies", "Lob", "Activity", "Periodicity", "Risk"]

# --- Decoupage temporel ------------------------------------------------------
TRAIN_DEBUT = (2020, 1)     # (annee, trimestre)
TRAIN_FIN   = (2024, 2)
CALIBRATION = (2024, 3)
TEST        = (2024, 4)

# --- Colonnes a exclure (fuite averee) ---------------------------------------
LEAK_COLS = []              # <-- a completer apres audit de fuite

# --- Parametres de modelisation ---------------------------------------------
RANDOM_STATE     = 42
TWEEDIE_POWER    = 1.7      # identique pour les 3 modeles : comparaison equitable
MIN_MODALITE     = 50       # modalite vue < 50 fois -> regroupee dans "AUTRE"
GLM_MAX_FEATURES = 60       # garde-fou : un GLM sur 300 colonnes ne converge pas

# --- Parametres conformes ----------------------------------------------------
ALPHA           = 0.10                              # couverture visee 90 %
ALPHAS_BALAYAGE = [0.20, 0.15, 0.10, 0.05, 0.01]

QUARTER_MAP = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


# =============================================================================
# BLOC 1 - CHARGEMENT ET PREPARATION TEMPORELLE
# =============================================================================
print("=" * 78)
print("PIPELINE FINAL - 3 MODELES + SPLIT CONFORMAL PREDICTION")
print("=" * 78)

try:
    df = df_model.copy()
    print(f"\nBase reprise depuis `df_model` : {len(df):,} lignes | "
          f"{df.shape[1]} colonnes")
except NameError:
    df = pd.read_csv(CHEMIN_DONNEES)
    print(f"\nBase chargee depuis {CHEMIN_DONNEES} : {len(df):,} lignes | "
          f"{df.shape[1]} colonnes")

# --- construction de l'index temporel ---------------------------------------
df["year"] = pd.to_numeric(df[COL_ANNEE], errors="coerce").astype("Int64")

if "quarter" not in df.columns:
    df["quarter"] = df[COL_TIME].astype(str).str.strip().str.upper().map(QUARTER_MAP)

df = df.dropna(subset=["year", "quarter"]).copy()
df["year"]    = df["year"].astype(int)
df["quarter"] = df["quarter"].astype(int)

# time_idx : entier croissant continu, 1 unite = 1 trimestre
df["time_idx"] = df["year"] * 4 + df["quarter"]

def idx_de(annee_trim):
    """Convertit un couple (annee, trimestre) en time_idx."""
    a, q = annee_trim
    return a * 4 + q

df = df.sort_values("time_idx").reset_index(drop=True)

# --- profil de la cible ------------------------------------------------------
y_all = pd.to_numeric(df[TARGET], errors="coerce")
n_neg = int((y_all < 0).sum())
n_nan = int(y_all.isna().sum())

print("\n" + "-" * 78)
print("PROFIL DE LA CIBLE")
print(f"  n             : {len(y_all):,}")
print(f"  NaN           : {n_nan:,}")
print(f"  valeurs < 0   : {n_neg:,}  ({n_neg / len(y_all):.2%})")
print(f"  valeurs = 0   : {int((y_all == 0).sum()):,}")
print(f"  min / max     : {y_all.min():,.2f}  /  {y_all.max():,.2f}")
for q in (25, 50, 75, 90, 99):
    print(f"  P{q:<12} : {np.nanpercentile(y_all.dropna(), q):,.2f}")
print(f"  asymetrie     : {y_all.skew():,.2f}")

# La loi Tweedie (1 < p < 2) est definie sur [0, +inf[ : les valeurs negatives
# sont ecretees. LightGBM le fait implicitement, on l'explicite ici.
if n_neg:
    print(f"\n  !! {n_neg:,} valeurs negatives ecretees a 0 (contrainte du support Tweedie).")
    print("     A documenter dans les limites du memoire.")
df[TARGET] = y_all.clip(lower=0)
df = df.dropna(subset=[TARGET]).reset_index(drop=True)

print("\n  Periodes presentes dans la base :")
periodes = (df.groupby(["year", "quarter"]).size()
            .reset_index(name="n").sort_values(["year", "quarter"]))
print(periodes.to_string(index=False))


# =============================================================================
# BLOC 2 - DECOUPAGE TRAIN / CALIBRATION / TEST
# -----------------------------------------------------------------------------
#  Le jeu de CALIBRATION doit etre disjoint du jeu d'entrainement. S'il ne
#  l'est pas, les residus de calibration sont artificiellement petits, le
#  quantile conforme est sous-estime, et la garantie de couverture tombe
#  silencieusement : le modele parait fiable alors qu'il ne l'est pas.
# =============================================================================
m_train = (df["time_idx"] >= idx_de(TRAIN_DEBUT)) & (df["time_idx"] <= idx_de(TRAIN_FIN))
m_cal   = df["time_idx"] == idx_de(CALIBRATION)
m_test  = df["time_idx"] == idx_de(TEST)

df_train, df_cal, df_test = df[m_train].copy(), df[m_cal].copy(), df[m_test].copy()

print("\n" + "=" * 78)
print("DECOUPAGE TEMPOREL")
print("=" * 78)
print(f"  TRAIN        {TRAIN_DEBUT[0]}Q{TRAIN_DEBUT[1]} -> {TRAIN_FIN[0]}Q{TRAIN_FIN[1]}"
      f"   : {len(df_train):>7,} lignes  ({df_train['time_idx'].nunique()} trimestres)")
print(f"  CALIBRATION  {CALIBRATION[0]}Q{CALIBRATION[1]}"
      f"                : {len(df_cal):>7,} lignes")
print(f"  TEST         {TEST[0]}Q{TEST[1]}"
      f"                : {len(df_test):>7,} lignes")

for nom, d in (("TRAIN", df_train), ("CALIBRATION", df_cal), ("TEST", df_test)):
    if d.empty:
        raise ValueError(f"Le jeu {nom} est vide : verifier les bornes du decoupage.")

if len(df_cal) < 100:
    print(f"\n  !! Calibration de seulement {len(df_cal)} lignes.")
    print("     Le quantile conforme sera instable. Envisager d'elargir la calibration.")

# Verification que la cible reste comparable entre les trois jeux : une derive
# forte remet en cause l'hypothese d'echangeabilite sur laquelle repose la
# garantie conforme.
print("\n  Distribution de la cible par jeu (test d'echangeabilite informel) :")
comp_jeux = pd.DataFrame({
    nom: [d[TARGET].mean(), d[TARGET].median(), d[TARGET].std(),
          d[TARGET].quantile(0.90)]
    for nom, d in (("TRAIN", df_train), ("CALIBRATION", df_cal), ("TEST", df_test))
}, index=["moyenne", "mediane", "ecart-type", "P90"])
print(comp_jeux.round(2).to_string())
print("\n  Un ecart marque entre CALIBRATION et TEST annonce une degradation")
print("  de la couverture : l'echangeabilite est alors mise en defaut.")


# =============================================================================
# BLOC 3 - PREPROCESSING PARTAGE
# -----------------------------------------------------------------------------
#  Les trois modeles recoivent EXACTEMENT la meme matrice de design. C'est la
#  condition pour que la comparaison porte sur l'algorithme et non sur des
#  encodages differents. Sans cela, un ecart de performance est ininterpretable.
#
#  Tout est appris sur le TRAIN seul, puis applique tel quel a CALIBRATION
#  et TEST. Apprendre l'encodage sur l'ensemble des donnees injecterait de
#  l'information future.
# =============================================================================
cat_cols = [c for c in ID_COLS if c in df.columns]
exclure  = set([TARGET, "time_idx", "year", "quarter", COL_ANNEE, COL_TIME]
               + cat_cols + LEAK_COLS)
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclure]

print("\n" + "=" * 78)
print("PREPROCESSING")
print("=" * 78)
print(f"  Variables numeriques   : {len(num_cols)}")
print(f"  Variables categorielles: {cat_cols}")


def log_signe(x):
    """Transformation log preservant le signe : comprime les queues lourdes."""
    x = np.asarray(x, float)
    return np.sign(x) * np.log1p(np.abs(x))


def ajuster_preprocessing(df_tr, num_cols, cat_cols, min_count=MIN_MODALITE):
    """Apprend medianes d'imputation et modalites conservees SUR LE TRAIN."""
    medianes = df_tr[num_cols].replace([np.inf, -np.inf], np.nan).median() \
        if num_cols else pd.Series(dtype=float)
    modalites = {}
    for c in cat_cols:
        vc = df_tr[c].astype(str).value_counts()
        modalites[c] = set(vc[vc >= min_count].index)
    return dict(medianes=medianes, modalites=modalites,
                num_cols=num_cols, cat_cols=cat_cols)


def appliquer_preprocessing(df_x, prep, colonnes_ref=None):
    """Construit la matrice de design a partir des parametres appris."""
    blocs = []

    if prep["num_cols"]:
        X_num = df_x[prep["num_cols"]].replace([np.inf, -np.inf], np.nan)
        X_num = X_num.fillna(prep["medianes"]).fillna(0.0)
        X_num = pd.DataFrame(log_signe(X_num.to_numpy()),
                             columns=[f"log_{c}" for c in prep["num_cols"]],
                             index=df_x.index)
        blocs.append(X_num)

    for c in prep["cat_cols"]:
        s = df_x[c].astype(str)
        s = s.where(s.isin(prep["modalites"][c]), "AUTRE")
        blocs.append(pd.get_dummies(s, prefix=c, drop_first=True, dtype=float))

    X = pd.concat(blocs, axis=1) if blocs else pd.DataFrame(index=df_x.index)

    # alignement strict : le test doit avoir les memes colonnes que le train
    if colonnes_ref is not None:
        X = X.reindex(columns=colonnes_ref, fill_value=0.0)
    return X


prep = ajuster_preprocessing(df_train, num_cols, cat_cols)

X_train = appliquer_preprocessing(df_train, prep)
# colonnes constantes : inutiles pour les arbres, fatales pour le GLM
X_train = X_train.loc[:, X_train.std() > 1e-12]
COLONNES = X_train.columns

X_cal  = appliquer_preprocessing(df_cal,  prep, colonnes_ref=COLONNES)
X_test = appliquer_preprocessing(df_test, prep, colonnes_ref=COLONNES)

y_train = df_train[TARGET].to_numpy(float)
y_cal   = df_cal[TARGET].to_numpy(float)
y_test  = df_test[TARGET].to_numpy(float)

print(f"  Matrice de design      : {X_train.shape[1]} colonnes")
print(f"    train {X_train.shape} | calibration {X_cal.shape} | test {X_test.shape}")


# =============================================================================
# BLOC 4 - ENTRAINEMENT DES TROIS MODELES
# -----------------------------------------------------------------------------
#  Meme famille de loi (Tweedie), meme puissance de variance, memes donnees.
#  Seul l'algorithme change :
#    GLM       : forme multiplicative imposee a priori, coefficients lisibles
#    LightGBM  : arbres, croissance feuille par feuille (leaf-wise)
#    XGBoost   : arbres, croissance niveau par niveau (level-wise)
# =============================================================================
print("\n" + "=" * 78)
print("ENTRAINEMENT DES MODELES")
print("=" * 78)

modeles, predictions = {}, {}

# --- 1. LightGBM -------------------------------------------------------------
print("\n[1/3] LightGBM Tweedie ...")
m_lgb = lgb.LGBMRegressor(
    objective="tweedie", tweedie_variance_power=TWEEDIE_POWER,
    n_estimators=800, learning_rate=0.03, num_leaves=31,
    min_child_samples=100, colsample_bytree=0.8,
    subsample=0.8, subsample_freq=1,
    reg_alpha=0.1, reg_lambda=5.0,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
m_lgb.fit(X_train, y_train)
modeles["LightGBM"] = m_lgb
predictions["LightGBM"] = np.clip(m_lgb.predict(X_test), 0, None)
print(f"      termine ({m_lgb.n_estimators_} arbres)")

# --- 2. XGBoost --------------------------------------------------------------
print("[2/3] XGBoost Tweedie ...")
m_xgb = xgb.XGBRegressor(
    objective="reg:tweedie", tweedie_variance_power=TWEEDIE_POWER,
    n_estimators=800, learning_rate=0.03, max_depth=6,
    min_child_weight=10, colsample_bytree=0.8, subsample=0.8,
    reg_alpha=0.1, reg_lambda=5.0, tree_method="hist",
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
m_xgb.fit(X_train, y_train)
modeles["XGBoost"] = m_xgb
predictions["XGBoost"] = np.clip(m_xgb.predict(X_test), 0, None)
print("      termine")

# --- 3. GLM Tweedie ----------------------------------------------------------
#  Garde-fou : au-dela de GLM_MAX_FEATURES colonnes, l'estimation par maximum
#  de vraisemblance devient instable (colinearite, non-convergence). On
#  restreint alors au sous-ensemble le plus correle a la cible. Ce choix est
#  un parti pris methodologique a assumer dans le memoire.
print("[3/3] GLM Tweedie (statsmodels) ...")

cols_glm = COLONNES
if len(COLONNES) > GLM_MAX_FEATURES:
    corr = (X_train.corrwith(pd.Series(y_train, index=X_train.index)).abs()
            .replace([np.inf, -np.inf], np.nan).dropna()
            .sort_values(ascending=False))
    cols_glm = corr.head(GLM_MAX_FEATURES).index
    print(f"      design reduit de {len(COLONNES)} a {len(cols_glm)} colonnes "
          f"(garde-fou de convergence)")

Xg_train = sm.add_constant(X_train[cols_glm], has_constant="add")
Xg_test  = sm.add_constant(X_test[cols_glm],  has_constant="add")
Xg_cal   = sm.add_constant(X_cal[cols_glm],   has_constant="add")

famille = sm.families.Tweedie(link=sm.families.links.Log(),
                              var_power=TWEEDIE_POWER)
res_glm = sm.GLM(y_train, Xg_train, family=famille).fit(maxiter=100, tol=1e-8)

modeles["GLM"] = res_glm
predictions["GLM"] = np.clip(res_glm.predict(Xg_test).to_numpy(float), 0, None)

print(f"      convergence : {res_glm.converged}")
print(f"      deviance/ddl: {res_glm.deviance / res_glm.df_resid:,.3f}"
      f"   (>> 1 = surdispersion residuelle)")
print(f"      AIC         : {res_glm.aic:,.1f}")
if not res_glm.converged:
    print("      !! NON CONVERGE : baisser GLM_MAX_FEATURES ou monter MIN_MODALITE.")

# --- table des coefficients (l'argument d'interpretabilite) ------------------
coefs_glm = pd.DataFrame({
    "coef": res_glm.params, "std_err": res_glm.bse,
    "z": res_glm.tvalues, "p_value": res_glm.pvalues,
})
coefs_glm["effet_multiplicatif"] = np.exp(coefs_glm["coef"])
coefs_glm = coefs_glm[coefs_glm.index != "const"].sort_values("p_value")

print("\n  GLM - 10 variables les plus significatives")
print("  " + "-" * 74)
apercu = coefs_glm.head(10).copy()
apercu["p_value"] = apercu["p_value"].map(lambda v: f"{v:.2e}")
print(apercu[["coef", "std_err", "p_value", "effet_multiplicatif"]]
      .round(4).to_string())
n_signif = int((coefs_glm["p_value"] < 0.05).sum())
print(f"\n  Significatives a 5 % : {n_signif} / {len(coefs_glm)}")


# =============================================================================
# BLOC 5 - EVALUATION COMPARATIVE
# -----------------------------------------------------------------------------
#  La MAE seule est trompeuse sur une cible a queue lourde : elle agrege des
#  regimes tres differents. On y adjoint donc :
#    - un baseline trivial (mediane) que le modele doit battre nettement
#    - le bilan somme(predit)/somme(reel), qui revele un biais systematique
#    - une decomposition par strate, qui localise l'echec
#    - le Gini, qui mesure le pouvoir de TRI (essentiel pour prioriser)
# =============================================================================
def gini(y_true, y_pred):
    o = np.argsort(y_pred)
    cum = np.cumsum(y_true[o]) / y_true.sum()
    return 1 - 2 * np.trapz(cum, np.linspace(0, 1, len(cum)))


def evaluer(y_true, y_pred):
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[ok], y_pred[ok]
    base = np.full_like(yt, np.median(yt))
    return dict(
        MAE=mean_absolute_error(yt, yp),
        RMSE=float(np.sqrt(np.mean((yt - yp) ** 2))),
        MAE_baseline=mean_absolute_error(yt, base),
        gain_vs_baseline=1 - mean_absolute_error(yt, yp) / mean_absolute_error(yt, base),
        bilan_somme=yp.sum() / yt.sum(),
        Gini=gini(yt, yp),
    )


resultats = pd.DataFrame({n: evaluer(y_test, p) for n, p in predictions.items()}).T
resultats = resultats.sort_values("MAE")

print("\n" + "=" * 78)
print(f"EVALUATION SUR LE TEST ({TEST[0]}Q{TEST[1]}, {len(df_test):,} observations)")
print("=" * 78)
print(resultats.round(4).to_string())

meilleur = resultats.index[0]
print(f"\n  Meilleure MAE : {meilleur}")
for autre in resultats.index[1:]:
    ecart = (resultats.loc[autre, "MAE"] - resultats.loc[meilleur, "MAE"]) \
            / resultats.loc[autre, "MAE"]
    print(f"    gain sur {autre:<10} : {ecart:>6.1%}")

# --- decomposition par strate ------------------------------------------------
edges  = [-np.inf] + [np.percentile(y_test, q) for q in (50, 90, 99)] + [np.inf]
labels = ["P0-50", "P50-90", "P90-99", "P99+"]
strate = pd.cut(y_test, bins=edges, labels=labels)

print("\n  MAE PAR STRATE DE CIBLE")
print("  " + "-" * 62)
mae_strate = pd.DataFrame(
    {n: [mean_absolute_error(y_test[(strate == l).to_numpy()],
                             p[(strate == l).to_numpy()])
         if (strate == l).sum() else np.nan for l in labels]
     for n, p in predictions.items()}, index=labels)
mae_strate["n"] = [int((strate == l).sum()) for l in labels]
print(mae_strate.round(0).to_string())
print("\n  Un modele qui gagne globalement mais perd sur P99+ est un mauvais")
print("  choix pour la validation comptable : c'est la que se logent les")
print("  anomalies les plus couteuses.")


# =============================================================================
# BLOC 6 - VISUALISATIONS COMPARATIVES  (4 graphiques, tous exploitables)
# =============================================================================
COULEURS = {"LightGBM": "darkgreen", "XGBoost": "darkorange", "GLM": "steelblue"}

fig, ax = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle(f"Comparaison des modeles - test {TEST[0]}Q{TEST[1]}",
             fontsize=15, fontweight="bold")

# 1 - predit vs reel
for nom, p in predictions.items():
    m = (y_test > 0) & (p > 0)
    ax[0, 0].scatter(y_test[m], p[m], s=6, alpha=.3,
                     color=COULEURS.get(nom, "gray"), label=nom)
mm = (y_test > 0)
lim = [y_test[mm].min(), y_test[mm].max()]
ax[0, 0].plot(lim, lim, "r--", lw=1.5, label="y = x")
ax[0, 0].set_xscale("log"); ax[0, 0].set_yscale("log")
ax[0, 0].set_xlabel("valeur reelle"); ax[0, 0].set_ylabel("valeur predite")
ax[0, 0].set_title("1. Predit vs Reel")
ax[0, 0].legend(markerscale=2)

# 2 - MAE par strate
x = np.arange(len(labels)); w = 0.8 / len(predictions)
for i, nom in enumerate(predictions):
    ax[0, 1].bar(x + i * w, mae_strate[nom], w, label=nom,
                 color=COULEURS.get(nom, "gray"))
ax[0, 1].set_xticks(x + w * (len(predictions) - 1) / 2)
ax[0, 1].set_xticklabels(labels)
ax[0, 1].set_yscale("log")
ax[0, 1].set_ylabel("MAE")
ax[0, 1].set_title("2. MAE par strate de cible")
ax[0, 1].legend()

# 3 - courbes de Lorenz : pouvoir de tri
for nom, p in predictions.items():
    o = np.argsort(p)
    cum = np.cumsum(y_test[o]) / y_test.sum()
    xs = np.linspace(0, 1, len(cum))
    ax[1, 0].plot(xs, cum, lw=2, color=COULEURS.get(nom, "gray"),
                  label=f"{nom} (Gini={gini(y_test, p):.3f})")
ax[1, 0].plot([0, 1], [0, 1], "r--", lw=1.2, label="aleatoire")
ax[1, 0].set_xlabel("part des observations (triees par prediction)")
ax[1, 0].set_ylabel("part cumulee de la cible")
ax[1, 0].set_title("3. Courbe de Lorenz - pouvoir de tri")
ax[1, 0].legend()

# 4 - importance des variables (LightGBM, top 15)
imp = (pd.DataFrame({"feature": X_train.columns,
                     "gain": m_lgb.booster_.feature_importance("gain")})
       .sort_values("gain", ascending=False).head(15).iloc[::-1])
imp["gain_%"] = 100 * imp["gain"] / m_lgb.booster_.feature_importance("gain").sum()
ax[1, 1].barh(imp["feature"], imp["gain_%"], color="darkgreen")
ax[1, 1].set_xlabel("part du gain total (%)")
ax[1, 1].set_title("4. Variables les plus explicatives (LightGBM)")
ax[1, 1].tick_params(axis="y", labelsize=8)

plt.tight_layout()
plt.show()


# =============================================================================
# BLOC 7 - SPLIT CONFORMAL PREDICTION
# -----------------------------------------------------------------------------
#  Principe : le modele de base fournit une prediction ponctuelle ; la
#  calibration fournit la loi empirique des ecarts ; le quantile de cette loi
#  donne la marge a ajouter pour garantir la couverture.
#
#  GARANTIE (Vovk et al. 2005) : sous echangeabilite entre calibration et test,
#      P( Y_test dans C(X_test) ) >= 1 - alpha
#  Cette garantie est MARGINALE. Rien n'assure qu'elle tienne sur un
#  sous-groupe donne (Barber et al. 2021) : d'ou le BLOC 8.
#
#  TROIS VARIANTES, et la premiere est indispensable a la demonstration :
#    A - absolu     : largeur CONSTANTE -> reproduit la limite du seuil fixe
#    B - normalise  : largeur proportionnelle a la difficulte locale sigma(x)
#    C - CQR        : largeur issue de deux regressions quantiles
# =============================================================================
MODELE_BASE = meilleur       # le meilleur des trois sert de modele de base

print("\n" + "=" * 78)
print(f"SPLIT CONFORMAL PREDICTION - modele de base : {MODELE_BASE}")
print(f"Calibration {CALIBRATION[0]}Q{CALIBRATION[1]} ({len(df_cal):,} obs) "
      f"| Test {TEST[0]}Q{TEST[1]} ({len(df_test):,} obs) | alpha = {ALPHA}")
print("=" * 78)


def predire(nom, X, X_glm=None):
    """Prediction du modele designe, ecretee a 0."""
    if nom == "GLM":
        return np.clip(modeles["GLM"].predict(X_glm).to_numpy(float), 0, None)
    return np.clip(modeles[nom].predict(X), 0, None)


def quantile_conforme(scores, alpha):
    """Quantile conforme avec CORRECTION D'ECHANTILLON FINI.

    Le rang est ceil((n+1)(1-alpha))/n et non le quantile (1-alpha) classique.
    C'est cette correction qui rend la garantie valable pour tout n fini.
    L'omettre produit une sous-couverture systematique et silencieuse.

    Retourne +inf si n est trop petit pour garantir le niveau demande : un
    intervalle non informatif est preferable a une fausse assurance.
    """
    s = np.asarray(scores, float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n == 0:
        return np.inf
    niveau = np.ceil((n + 1) * (1 - alpha)) / n
    if niveau > 1:
        print(f"    !! n={n} trop petit pour garantir 1-alpha={1 - alpha:.2f}")
        return np.inf
    return float(np.quantile(s, niveau, method="higher"))


def construire_intervalles(alpha, verbose=True):
    """Construit les intervalles des trois variantes conformes."""
    pred_cal  = predire(MODELE_BASE, X_cal,  Xg_cal)
    pred_test = predire(MODELE_BASE, X_test, Xg_test)
    pred_tr   = predire(MODELE_BASE, X_train, Xg_train)
    out = {}

    # ---------- A : split conformal absolu (largeur constante) --------------
    s_abs = np.abs(y_cal - pred_cal)
    q_abs = quantile_conforme(s_abs, alpha)
    out["A_absolu"] = (np.clip(pred_test - q_abs, 0, None), pred_test + q_abs)

    # ---------- B : split conformal normalise (largeur variable) ------------
    #  sigma(x) est estime sur le TRAIN en modelisant log1p(|residu|).
    #  C'est lui qui rend l'intervalle adaptatif : un sous-portefeuille
    #  historiquement volatil recoit une marge plus large, un sous-portefeuille
    #  stable une marge plus etroite. C'est la reponse directe a la critique
    #  du seuil fixe formulee au chapitre 1.
    m_sigma = lgb.LGBMRegressor(
        objective="regression", n_estimators=400, learning_rate=0.05,
        num_leaves=31, min_child_samples=100,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    m_sigma.fit(X_train, np.log1p(np.abs(y_train - pred_tr)))

    sigma_cal  = np.clip(np.expm1(m_sigma.predict(X_cal)),  1e-6, None)
    sigma_test = np.clip(np.expm1(m_sigma.predict(X_test)), 1e-6, None)

    q_norm = quantile_conforme(np.abs(y_cal - pred_cal) / sigma_cal, alpha)
    out["B_normalise"] = (np.clip(pred_test - q_norm * sigma_test, 0, None),
                          pred_test + q_norm * sigma_test)

    # ---------- C : CQR (Romano, Patterson, Candes 2019) --------------------
    par_q = dict(objective="quantile", n_estimators=400, learning_rate=0.05,
                 num_leaves=31, min_child_samples=100,
                 random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    q_lo = lgb.LGBMRegressor(**par_q, alpha=alpha / 2).fit(X_train, y_train)
    q_hi = lgb.LGBMRegressor(**par_q, alpha=1 - alpha / 2).fit(X_train, y_train)

    lo_cal,  hi_cal  = q_lo.predict(X_cal),  q_hi.predict(X_cal)
    lo_test, hi_test = q_lo.predict(X_test), q_hi.predict(X_test)

    # score CQR : depassement du cote le plus contraignant
    s_cqr = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    q_cqr = quantile_conforme(s_cqr, alpha)
    out["C_cqr"] = (np.clip(lo_test - q_cqr, 0, None), hi_test + q_cqr)

    if verbose:
        print(f"\n  Quantiles conformes (n_cal = {len(y_cal):,})")
        print(f"    A absolu    : {q_abs:>16,.2f}   (marge fixe, identique pour tous)")
        print(f"    B normalise : {q_norm:>16,.4f}   (multiplicateur de sigma)")
        print(f"    C cqr       : {q_cqr:>16,.2f}   (correction des quantiles)")
        print(f"    sigma test  : mediane {np.median(sigma_test):,.2f} | "
              f"P10 {np.percentile(sigma_test, 10):,.2f} | "
              f"P90 {np.percentile(sigma_test, 90):,.2f}")

    return out, pred_test, sigma_test


intervalles, pred_test_base, sigma_test = construire_intervalles(ALPHA)

# --- assemblage du tableau de resultats -------------------------------------
conf = df_test[[c for c in cat_cols if c in df_test.columns]].reset_index(drop=True)
conf["year_quarter"]   = f"{TEST[0]}Q{TEST[1]}"
conf["Valeur_reelle"]  = y_test
conf["Valeur_predite"] = pred_test_base
conf["sigma"]          = sigma_test
conf["strate"]         = strate

METHODES = [("A_absolu",    "A - absolu (largeur constante)"),
            ("B_normalise", "B - normalise (largeur variable)"),
            ("C_cqr",       "C - CQR (largeur variable)")]

for cle, _ in METHODES:
    lo, hi = intervalles[cle]
    conf[f"lo_{cle}"]      = lo
    conf[f"hi_{cle}"]      = hi
    conf[f"largeur_{cle}"] = hi - lo
    conf[f"couvert_{cle}"] = ((y_test >= lo) & (y_test <= hi)).astype(int)


# =============================================================================
# BLOC 8 - VALIDITE, EFFICACITE ET COUVERTURE CONDITIONNELLE
# -----------------------------------------------------------------------------
#  Deux criteres indissociables :
#    VALIDITE   la couverture empirique atteint-elle 1 - alpha ?
#    EFFICACITE l'intervalle est-il assez etroit pour etre exploitable ?
#  Un intervalle [0, +inf[ couvre a 100 % et ne sert a rien.
# =============================================================================
lignes = []
for cle, nom in METHODES:
    larg = conf[f"largeur_{cle}"]
    couv = conf[f"couvert_{cle}"].mean()
    lignes.append({
        "methode": nom,
        "couverture": couv,
        "ecart_cible": couv - (1 - ALPHA),
        "largeur_moyenne": larg.mean(),
        "largeur_mediane": larg.median(),
        "cv_largeur": larg.std() / larg.mean() if larg.mean() else 0.0,
        "n_signales": int((1 - conf[f"couvert_{cle}"]).sum()),
    })
resume_conf = pd.DataFrame(lignes)

print("\n" + "=" * 78)
print(f"VALIDITE ET EFFICACITE - couverture visee {1 - ALPHA:.0%}")
print("=" * 78)
for _, r in resume_conf.iterrows():
    statut = "OK" if r["ecart_cible"] >= -0.02 else "SOUS-COUVERTURE"
    print(f"\n  {r['methode']}")
    print(f"    couverture empirique : {r['couverture']:>9.2%}   ({statut})")
    print(f"    ecart a la cible     : {r['ecart_cible']:>+9.2%}")
    print(f"    largeur mediane      : {r['largeur_mediane']:>13,.0f}")
    print(f"    coef. de variation   : {r['cv_largeur']:>9.3f}")
    print(f"    observations hors IC : {r['n_signales']:>13,}")

print("\n" + "-" * 78)
print("  LECTURE DU COEFFICIENT DE VARIATION DE LA LARGEUR")
print("  Proche de 0  -> l'intervalle est identique pour tous les")
print("                  sous-portefeuilles : c'est exactement la limite du")
print("                  seuil fixe denoncee au chapitre 1.")
print("  Nettement > 0 -> l'intervalle s'adapte au contexte de chaque")
print("                  sous-portefeuille : la critique est levee.")
print("-" * 78)

# --- couverture conditionnelle : le point challengeable en soutenance -------
def couverture_par(colonne, min_n=20, top=10):
    res = []
    for mod, g in conf.groupby(colonne, observed=True):
        if len(g) < min_n:
            continue
        d = {colonne: str(mod), "n": len(g)}
        for cle, _ in METHODES:
            d[f"couv_{cle}"] = g[f"couvert_{cle}"].mean()
        res.append(d)
    if not res:
        return pd.DataFrame()
    return (pd.DataFrame(res).sort_values("n", ascending=False)
            .head(top).reset_index(drop=True))


print("\n" + "=" * 78)
print("COUVERTURE CONDITIONNELLE (garantie marginale, Barber et al. 2021)")
print("=" * 78)

cc_strate = couverture_par("strate")
print("\n  Par strate de cible :")
print(cc_strate.round(4).to_string(index=False))
print("\n  Une couverture qui s'effondre sur P99+ signifie que les valeurs")
print("  extremes ne sont pas protegees - la ou la validation en a le plus besoin.")

cc_metier = {}
for col in cat_cols:
    cc = couverture_par(col)
    if not cc.empty:
        cc_metier[col] = cc
        print(f"\n  Par {col} :")
        print(cc.round(4).to_string(index=False))


# =============================================================================
# BLOC 9 - PRIORISATION DES SOUS-PORTEFEUILLES ATYPIQUES
# -----------------------------------------------------------------------------
#  Sortir de l'intervalle est un signal binaire. Les equipes de validation ont
#  un temps fini : il faut ORDONNER les signaux.
#
#      severite = depassement / largeur de l'intervalle
#
#  Rapporter le depassement a la largeur rend les signaux comparables entre
#  sous-portefeuilles de tailles tres differentes : un ecart de 50 kEUR sur un
#  perimetre stable pese plus qu'un ecart de 500 kEUR sur un perimetre
#  historiquement volatil. C'est la reponse operationnelle a l'hypothese H3.
# =============================================================================
CLE_RETENUE = "C_cqr"

lo = conf[f"lo_{CLE_RETENUE}"].to_numpy(float)
hi = conf[f"hi_{CLE_RETENUE}"].to_numpy(float)
largeur = np.maximum(hi - lo, 1e-9)

conf["depassement"] = np.where(y_test > hi, y_test - hi,
                        np.where(y_test < lo, lo - y_test, 0.0))
conf["severite"]    = conf["depassement"] / largeur
conf["sens_ecart"]  = np.where(y_test > hi, "au-dessus de l'attendu",
                        np.where(y_test < lo, "en-dessous de l'attendu", "conforme"))

signaux = (conf[conf["severite"] > 0]
           .sort_values("severite", ascending=False).reset_index(drop=True))

print("\n" + "=" * 78)
print(f"PRIORISATION DES SIGNAUX - variante {CLE_RETENUE}")
print("=" * 78)
print(f"  Observations testees : {len(conf):,}")
print(f"  Signaux (hors IC)    : {len(signaux):,}  ({len(signaux) / len(conf):.2%})")
print(f"  Attendu theorique    : {ALPHA:.2%}")

if len(signaux):
    cols_aff = ([c for c in cat_cols if c in signaux.columns]
                + ["Valeur_reelle", "Valeur_predite",
                   f"lo_{CLE_RETENUE}", f"hi_{CLE_RETENUE}",
                   "depassement", "severite", "sens_ecart"])
    print("\n  TOP 15 A INVESTIGUER EN PRIORITE")
    print("  " + "-" * 74)
    print(signaux[cols_aff].head(15).round(2).to_string(index=False))

    if cat_cols:
        axe = cat_cols[0]
        conc = (signaux.groupby(axe)
                .agg(n_signaux=("severite", "size"),
                     severite_moy=("severite", "mean"),
                     severite_max=("severite", "max"))
                .sort_values("n_signaux", ascending=False).head(10))
        print(f"\n  CONCENTRATION DES SIGNAUX PAR {axe.upper()}")
        print("  " + "-" * 74)
        print(conc.round(3).to_string())


# =============================================================================
# BLOC 10 - VISUALISATIONS CONFORMES  (6 graphiques, tous exploitables)
# =============================================================================
C_METH = {"A_absolu": "indianred", "B_normalise": "steelblue", "C_cqr": "darkgreen"}

fig, ax = plt.subplots(2, 3, figsize=(19, 11))
fig.suptitle(f"Split Conformal Prediction - test {TEST[0]}Q{TEST[1]} "
             f"(alpha = {ALPHA})", fontsize=15, fontweight="bold")

# 1 - validite : couverture atteinte
noms  = [c.split("_")[0] for c, _ in METHODES]
couvs = [conf[f"couvert_{c}"].mean() for c, _ in METHODES]
b = ax[0, 0].bar(noms, couvs, color=[C_METH[c] for c, _ in METHODES])
ax[0, 0].axhline(1 - ALPHA, color="k", ls="--", lw=2, label=f"cible {1 - ALPHA:.0%}")
ax[0, 0].set_ylim(min(couvs) - 0.1, 1.03)
ax[0, 0].set_ylabel("couverture empirique")
ax[0, 0].set_title("1. Validite")
ax[0, 0].legend()
for bb, v in zip(b, couvs):
    ax[0, 0].text(bb.get_x() + bb.get_width() / 2, v + 0.005,
                  f"{v:.1%}", ha="center", fontsize=10)

# 2 - efficacite : distribution des largeurs (LE graphique du memoire)
for cle, _ in METHODES:
    l = conf[f"largeur_{cle}"]
    l = l[l > 0]
    ax[0, 1].hist(l, bins=60, alpha=.55, color=C_METH[cle], label=cle.split("_")[0])
ax[0, 1].set_xscale("log")
ax[0, 1].set_xlabel("largeur de l'intervalle")
ax[0, 1].set_title("2. Efficacite : distribution des largeurs")
ax[0, 1].legend()

# 3 - adaptativite : largeur vs prediction
for cle, _ in METHODES:
    m = conf["Valeur_predite"] > 0
    ax[0, 2].scatter(conf.loc[m, "Valeur_predite"], conf.loc[m, f"largeur_{cle}"],
                     s=5, alpha=.25, color=C_METH[cle], label=cle.split("_")[0])
ax[0, 2].set_xscale("log"); ax[0, 2].set_yscale("log")
ax[0, 2].set_xlabel("valeur predite"); ax[0, 2].set_ylabel("largeur")
ax[0, 2].set_title("3. Adaptativite au contexte")
ax[0, 2].legend(markerscale=3)

# 4 - couverture conditionnelle par strate
if not cc_strate.empty:
    x = np.arange(len(cc_strate)); w = 0.26
    for i, (cle, _) in enumerate(METHODES):
        ax[1, 0].bar(x + i * w, cc_strate[f"couv_{cle}"], w,
                     label=cle.split("_")[0], color=C_METH[cle])
    ax[1, 0].axhline(1 - ALPHA, color="k", ls="--", lw=2)
    ax[1, 0].set_xticks(x + w)
    ax[1, 0].set_xticklabels(cc_strate["strate"])
    ax[1, 0].set_ylabel("couverture")
    ax[1, 0].set_title("4. Couverture conditionnelle par strate")
    ax[1, 0].legend(fontsize=8)

# 5 - intervalles sur un echantillon trie, signaux en rouge
ech = conf.sample(min(250, len(conf)), random_state=RANDOM_STATE)
ech = ech.sort_values("Valeur_predite").reset_index(drop=True)
xs = np.arange(len(ech))
ax[1, 1].fill_between(xs, ech[f"lo_{CLE_RETENUE}"], ech[f"hi_{CLE_RETENUE}"],
                      alpha=.3, color="darkgreen", label="intervalle conforme")
ax[1, 1].plot(xs, ech["Valeur_predite"], lw=1.4, color="black", label="predit")
hors = (ech[f"couvert_{CLE_RETENUE}"] == 0).to_numpy()
ax[1, 1].scatter(xs[~hors], ech.loc[~hors, "Valeur_reelle"], s=7,
                 color="steelblue", label="reel couvert")
ax[1, 1].scatter(xs[hors], ech.loc[hors, "Valeur_reelle"], s=30,
                 color="red", marker="x", label="SIGNAL")
ax[1, 1].set_yscale("symlog")
ax[1, 1].set_title(f"5. Intervalles {CLE_RETENUE} (echantillon trie)")
ax[1, 1].legend(fontsize=8)

# 6 - severite des signaux
sev = conf[conf["severite"] > 0]["severite"]
if len(sev):
    ax[1, 2].hist(sev, bins=40, color="darkorange")
    ax[1, 2].set_yscale("log")
    ax[1, 2].set_xlabel("severite (depassement / largeur)")
    ax[1, 2].set_ylabel("nombre de signaux")
    ax[1, 2].set_title(f"6. Severite des {len(sev):,} signaux")
else:
    ax[1, 2].axis("off")

plt.tight_layout()
plt.show()


# --- Arbitrage couverture / largeur selon alpha ------------------------------
#  Repond a la question "quel niveau de confiance retenir ?" en montrant le
#  cout en largeur de chaque gain de couverture.
print("\n" + "=" * 78)
print("ARBITRAGE COUVERTURE / LARGEUR")
print("=" * 78)

bal = []
for a in ALPHAS_BALAYAGE:
    iv, _, _ = construire_intervalles(a, verbose=False)
    for cle, _ in METHODES:
        lo_a, hi_a = iv[cle]
        bal.append(dict(alpha=a, cible=1 - a, methode=cle,
                        couverture=float(((y_test >= lo_a) & (y_test <= hi_a)).mean()),
                        largeur_mediane=float(np.median(hi_a - lo_a))))
bal = pd.DataFrame(bal)
print(bal.pivot(index="cible", columns="methode",
                values="couverture").round(4).to_string())

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Arbitrage couverture / largeur", fontsize=14, fontweight="bold")
for cle, _ in METHODES:
    s = bal[bal["methode"] == cle].sort_values("cible")
    ax[0].plot(s["cible"], s["couverture"], "o-", lw=2,
               color=C_METH[cle], label=cle.split("_")[0])
    ax[1].plot(s["cible"], s["largeur_mediane"], "o-", lw=2,
               color=C_METH[cle], label=cle.split("_")[0])
ax[0].plot([0.78, 1.0], [0.78, 1.0], "k--", lw=1.5, label="calibration parfaite")
ax[0].set_xlabel("couverture visee"); ax[0].set_ylabel("couverture obtenue")
ax[0].set_title("Validite selon le niveau"); ax[0].legend()
ax[1].set_xlabel("couverture visee"); ax[1].set_ylabel("largeur mediane")
ax[1].set_yscale("log")
ax[1].set_title("Cout en largeur"); ax[1].legend()
plt.tight_layout(); plt.show()


# =============================================================================
# BLOC 11 - SAUVEGARDE
# =============================================================================
resultats.to_csv("resultats_comparaison_modeles.csv")
mae_strate.to_csv("resultats_mae_par_strate.csv")
coefs_glm.to_csv("glm_coefficients.csv")
conf.to_csv("predictions_conformal.csv", index=False)
resume_conf.to_csv("conformal_validite_efficacite.csv", index=False)
cc_strate.to_csv("conformal_couverture_strate.csv", index=False)
bal.to_csv("conformal_arbitrage_alpha.csv", index=False)
if len(signaux):
    signaux.to_csv("conformal_signaux_prioritaires.csv", index=False)
for col, cc in cc_metier.items():
    cc.to_csv(f"conformal_couverture_{col}.csv", index=False)
joblib.dump({"modeles": modeles, "prep": prep, "colonnes": list(COLONNES),
             "colonnes_glm": list(cols_glm)}, "modeles_et_preprocessing.pkl")

print("\n" + "=" * 78)
print("ARTEFACTS SAUVEGARDES")
print("=" * 78)
for f in ["resultats_comparaison_modeles.csv", "resultats_mae_par_strate.csv",
          "glm_coefficients.csv", "predictions_conformal.csv",
          "conformal_validite_efficacite.csv", "conformal_couverture_strate.csv",
          "conformal_arbitrage_alpha.csv", "conformal_signaux_prioritaires.csv",
          "modeles_et_preprocessing.pkl"]:
    print(f"   {f}")
print("=" * 78)
