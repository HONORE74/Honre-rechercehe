# =============================================================================
#  PIPELINE COMPLET  -  GLM Tweedie (reference actuarielle)
#  Concu pour etre COMPARABLE au pipeline LightGBM :
#    meme base, meme protocole de rolling forecast, meme evaluation.
#
#  Pourquoi un GLM ici ?
#    Le GLM est la reference du metier : coefficients lisibles, effets
#    quantifiables, cadre statistique explicite (deviance, significativite).
#    Il constitue le baseline face auquel LightGBM doit demontrer un gain.
#    Sans lui, "le ML fait mieux" est une affirmation invalidee.
#
#  ORDRE D'EXECUTION :
#    BLOC 0  Imports & configuration
#    BLOC 1  Preparation temporelle (identique au pipeline LightGBM)
#    BLOC 2  Construction de la matrice de design GLM
#    BLOC 3  Ajustement GLM Tweedie + diagnostics statistiques
#    BLOC 4  Rolling forecasting GLM (meme protocole que LightGBM)
#    BLOC 5  Evaluation (meme fonction que LightGBM)
#    BLOC 6  Graphiques de diagnostic GLM
#    BLOC 7  Comparaison GLM vs LightGBM
#    BLOC 8  Sauvegarde des artefacts
#
#  PREREQUIS : df_model et TARGET definis en amont (comme pour LightGBM).
# =============================================================================

# =============================================================================
# BLOC 0 - IMPORTS & CONFIGURATION
# =============================================================================
import warnings, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.metrics import mean_absolute_error
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

RANDOM_STATE = 42
QUARTER_MAP  = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

# --- Parametres de modelisation GLM -----------------------------------------
TWEEDIE_POWER   = 1.7    # meme valeur que l'objectif LightGBM : comparaison equitable
MIN_MODALITE    = 50     # une modalite vue < 50 fois est regroupee dans "AUTRE"
TOP_NUM_FEATURES = 25    # nb de variables numeriques retenues (par correlation)
INITIAL_TRAIN_PERIODS = 8   # identique au pipeline LightGBM


# =============================================================================
# BLOC 1 - PREPARATION TEMPORELLE  (strictement identique au pipeline LightGBM)
# =============================================================================
df = df_model.copy()
df["year"] = df["annee"].astype(int)

if "quarter" not in df.columns:
    df["quarter"] = df["Time"].str.strip().map(QUARTER_MAP).astype(int)

df["time_idx"] = (df["year"] - df["year"].min()) * 4 + df["quarter"]

# Exclusion 2024Q3 (meme exclusion que LightGBM, sinon comparaison biaisee)
df = df[~((df["year"] == 2024) & (df["quarter"] == 3))].reset_index(drop=True)
df = df.sort_values("time_idx").reset_index(drop=True)

print("=" * 70)
print("PIPELINE GLM TWEEDIE - reference actuarielle")
print("=" * 70)
print(f"BASE : {len(df):,} lignes  |  {df.shape[1]} colonnes")
print(f"Periodes : {df['time_idx'].nunique()}")

# --- Contrainte Tweedie : la cible doit etre >= 0 ---------------------------
#  Pour 1 < p < 2, la loi Tweedie est definie sur [0, +inf[ avec une masse en 0.
#  Les valeurs negatives sont donc ecretees a 0, comme le fait implicitement
#  LightGBM avec objective="tweedie". A documenter dans le memoire.
y_brut = df[TARGET].astype(float)
n_neg  = int((y_brut < 0).sum())
if n_neg:
    print(f"\n!! {n_neg:,} valeurs negatives ecretees a 0 "
          f"({n_neg / len(y_brut):.2%} des lignes) - contrainte du support Tweedie.")
df[TARGET] = y_brut.clip(lower=0)


# =============================================================================
# BLOC 2 - CONSTRUCTION DE LA MATRICE DE DESIGN GLM
# -----------------------------------------------------------------------------
#  Un GLM n'ingere pas des colonnes brutes comme un arbre :
#    - les categorielles doivent etre encodees en indicatrices
#    - les modalites rares doivent etre regroupees (sinon coefficients instables)
#    - les numeriques tres asymetriques doivent etre transformees (lien log)
#
#  POINT METHODOLOGIQUE CRUCIAL : l'encodage est appris sur le TRAIN seul,
#  puis applique au TEST. Sinon on injecte de l'information future -> fuite.
# =============================================================================
ID_COLS = [c for c in ["Partner", "Companies", "Lob", "Activity",
                       "Periodicity", "Risk"] if c in df.columns]

LEAK_COLS = []   # a remplir si l'audit du pipeline LightGBM (BLOC 2) a revele une fuite

EXCLUDE_COLS = [TARGET, "time_idx", "year", "quarter", "annee", "Time"] + LEAK_COLS

NUM_COLS_ALL = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in EXCLUDE_COLS and c not in ID_COLS]

print(f"\nIdentifiants metier    : {ID_COLS}")
print(f"Variables numeriques   : {len(NUM_COLS_ALL)} disponibles")


def selectionner_numeriques(df_tr, target, cols, k=TOP_NUM_FEATURES):
    """Retient les k numeriques les plus correlees a la cible SUR LE TRAIN.

    Un GLM avec 200 regresseurs sur donnees colineaires ne converge pas
    proprement. On restreint donc le design a un sous-ensemble stable.
    Ce choix est un parti pris methodologique, a assumer dans le memoire.
    """
    if not cols:
        return []
    corr = (df_tr[cols].corrwith(df_tr[target]).abs()
            .replace([np.inf, -np.inf], np.nan).dropna()
            .sort_values(ascending=False))
    return list(corr.head(k).index)


def log_signe(x):
    """Transformation log preservant le signe : attenue les queues lourdes.

    log1p classique echoue sur les valeurs negatives ; cette version
    symetrique gere les deux cotes de la distribution.
    """
    x = np.asarray(x, float)
    return np.sign(x) * np.log1p(np.abs(x))


def ajuster_encodage(df_tr, cat_cols, min_count=MIN_MODALITE):
    """Apprend, pour chaque categorielle, la liste des modalites conservees.

    Une modalite vue moins de `min_count` fois produit un coefficient
    estime sur trop peu d'observations : variance enorme, non interpretable.
    Elle est donc fondue dans une modalite "AUTRE".
    """
    mapping = {}
    for c in cat_cols:
        vc = df_tr[c].astype(str).value_counts()
        mapping[c] = set(vc[vc >= min_count].index)
    return mapping


def construire_design(df_x, num_cols, cat_cols, mapping, colonnes_ref=None):
    """Construit la matrice de design : numeriques transformees + indicatrices.

    `colonnes_ref` garantit que le TEST possede exactement les memes colonnes
    que le TRAIN (les modalites absentes sont creees a 0). Sans cela, la
    prediction echoue ou, pire, decale silencieusement les coefficients.
    """
    blocs = []

    # --- numeriques : transformation log signee + imputation mediane ---
    if num_cols:
        X_num = df_x[num_cols].replace([np.inf, -np.inf], np.nan)
        X_num = X_num.fillna(X_num.median(numeric_only=True)).fillna(0.0)
        X_num = pd.DataFrame(log_signe(X_num.to_numpy()),
                             columns=[f"log_{c}" for c in num_cols],
                             index=df_x.index)
        blocs.append(X_num)

    # --- categorielles : regroupement des rares puis indicatrices ---
    for c in cat_cols:
        serie = df_x[c].astype(str)
        serie = serie.where(serie.isin(mapping[c]), "AUTRE")
        dummies = pd.get_dummies(serie, prefix=c, drop_first=True, dtype=float)
        blocs.append(dummies)

    X = pd.concat(blocs, axis=1) if blocs else pd.DataFrame(index=df_x.index)

    # --- alignement train/test ---
    if colonnes_ref is not None:
        X = X.reindex(columns=colonnes_ref, fill_value=0.0)

    return X


def nettoyer_colineaires(X, seuil=1e-10):
    """Retire les colonnes constantes ou parfaitement colineaires.

    Une matrice de design non inversible fait echouer l'estimation par
    maximum de vraisemblance, ou produit des coefficients aberrants.
    """
    garder = X.columns[X.std(axis=0) > seuil].tolist()
    X = X[garder]
    if X.shape[1] > 1:
        corr = X.corr().abs().to_numpy()
        np.fill_diagonal(corr, 0.0)
        a_retirer = set()
        cols = list(X.columns)
        for i in range(len(cols)):
            if cols[i] in a_retirer:
                continue
            for j in range(i + 1, len(cols)):
                if corr[i, j] > 0.999:
                    a_retirer.add(cols[j])
        X = X.drop(columns=list(a_retirer))
    return X


# =============================================================================
# BLOC 3 - AJUSTEMENT GLM TWEEDIE + DIAGNOSTICS STATISTIQUES
# -----------------------------------------------------------------------------
#  Famille Tweedie, lien log :
#      E[Y | X] = exp(X . beta)          Var(Y) = phi * E[Y]^p
#  Avec 1 < p < 2 la loi melange une masse en zero et une partie continue
#  positive : exactement le profil d'un indicateur d'assurance agrege.
# =============================================================================
def ajuster_glm(X, y, power=TWEEDIE_POWER, maxiter=100):
    """Estime un GLM Tweedie a lien log par maximum de vraisemblance (IRLS)."""
    Xc = sm.add_constant(X, has_constant="add")
    famille = sm.families.Tweedie(link=sm.families.links.Log(),
                                  var_power=power)
    modele = sm.GLM(y, Xc, family=famille)
    return modele.fit(maxiter=maxiter, tol=1e-8)


def diagnostics_glm(res, titre="DIAGNOSTICS GLM"):
    """Affiche les indicateurs d'ajustement attendus dans un memoire actuariel."""
    print("\n" + "=" * 70)
    print(titre)
    print("=" * 70)
    print(f"  Observations          : {int(res.nobs):,}")
    print(f"  Parametres estimes    : {len(res.params):,}")
    print(f"  Deviance              : {res.deviance:,.1f}")
    print(f"  Deviance / ddl        : {res.deviance / res.df_resid:,.3f}"
          f"   (>> 1 = surdispersion residuelle)")
    print(f"  Pearson chi2 / ddl    : {res.pearson_chi2 / res.df_resid:,.3f}")
    print(f"  Log-vraisemblance     : {res.llf:,.1f}")
    print(f"  AIC                   : {res.aic:,.1f}")
    print(f"  BIC                   : {res.bic_llf:,.1f}")
    print(f"  Convergence           : {res.converged}")
    if not res.converged:
        print("  !! Le modele n'a PAS converge : resultats non exploitables.")
        print("     Reduire TOP_NUM_FEATURES ou augmenter MIN_MODALITE.")

    # --- coefficients les plus significatifs ---
    tab = pd.DataFrame({
        "coef":    res.params,
        "std_err": res.bse,
        "z":       res.tvalues,
        "p_value": res.pvalues,
    })
    tab["effet_multiplicatif"] = np.exp(tab["coef"])   # lien log : lecture directe
    tab = tab[tab.index != "const"]
    tab = tab.reindex(tab["p_value"].sort_values().index)

    print("\n  VARIABLES LES PLUS SIGNIFICATIVES (top 20)")
    print("  " + "-" * 66)
    apercu = tab.head(20).copy()
    apercu["p_value"] = apercu["p_value"].map(lambda v: f"{v:.2e}")
    apercu["effet_multiplicatif"] = apercu["effet_multiplicatif"].round(4)
    print(apercu[["coef", "std_err", "p_value", "effet_multiplicatif"]]
          .round(4).to_string())

    n_signif = int((res.pvalues.drop("const", errors="ignore") < 0.05).sum())
    print(f"\n  Variables significatives a 5% : {n_signif} / {len(tab)}")
    print("=" * 70)
    return tab


# =============================================================================
# BLOC 4 - ROLLING FORECASTING GLM
# -----------------------------------------------------------------------------
#  Protocole IDENTIQUE au pipeline LightGBM : fenetre expansive, on entraine
#  sur tout le passe disponible et on predit la periode t+1.
#  Toute difference de protocole invaliderait la comparaison.
# =============================================================================
def rolling_forecasting_glm(df_raw, target, num_cols_all, cat_cols,
                            initial_train_periods, power=TWEEDIE_POWER,
                            verbose=True):

    all_periods = sorted(df_raw["time_idx"].unique())
    if initial_train_periods >= len(all_periods):
        raise ValueError("initial_train_periods doit etre < nombre de periodes.")

    train_periods      = list(all_periods[:initial_train_periods])
    periods_to_predict = all_periods[initial_train_periods:]
    records, modeles, skipped = [], {}, []
    dernier_res = dernier_contexte = None

    for step, period in enumerate(periods_to_predict, start=1):
        mask_tr = df_raw["time_idx"].isin(train_periods)
        mask_te = df_raw["time_idx"] == period

        df_tr = df_raw.loc[mask_tr]
        df_te = df_raw.loc[mask_te]

        if df_tr.empty or df_te.empty:
            if verbose:
                print(f"  Etape {step} ignoree : pas de donnees.")
            continue

        try:
            # --- tout l'encodage est appris sur le TRAIN uniquement ---
            num_cols = selectionner_numeriques(df_tr, target, num_cols_all)
            mapping  = ajuster_encodage(df_tr, cat_cols)

            X_tr = construire_design(df_tr, num_cols, cat_cols, mapping)
            X_tr = nettoyer_colineaires(X_tr)
            X_te = construire_design(df_te, num_cols, cat_cols, mapping,
                                     colonnes_ref=X_tr.columns)

            y_tr = df_tr[target].to_numpy(float)
            y_te = df_te[target].to_numpy(float)

            res = ajuster_glm(X_tr, y_tr, power=power)

            X_te_c = sm.add_constant(X_te, has_constant="add")
            X_te_c = X_te_c.reindex(columns=res.params.index, fill_value=0.0)
            y_pred = np.clip(res.predict(X_te_c).to_numpy(float), 0, None)

            dernier_res = res
            dernier_contexte = dict(num_cols=num_cols, cat_cols=cat_cols,
                                    mapping=mapping, colonnes=X_tr.columns,
                                    X_tr=X_tr, y_tr=y_tr)

        except Exception as err:
            print(f"  Erreur etape {step} : {err}")
            skipped.append(step)
            continue

        idx  = df_te.index
        info = df_raw.loc[idx, ["year", "quarter"]].reset_index(drop=True)
        ids  = (df_raw.loc[idx, cat_cols].reset_index(drop=True)
                if cat_cols else pd.DataFrame())

        batch = pd.concat([ids, info], axis=1)
        batch["Valeur_reelle"]   = y_te
        batch["Valeur_predite"]  = y_pred
        batch["step"]            = step
        batch["n_train_obs"]     = len(df_tr)
        batch["n_parametres"]    = len(res.params)
        records.append(batch)
        modeles[period] = res

        if verbose:
            ok = np.isfinite(y_te) & np.isfinite(y_pred)
            mae = mean_absolute_error(y_te[ok], y_pred[ok]) if ok.any() else np.nan
            cvg = "ok" if res.converged else "NON CONVERGE"
            print(f"Etape {step:>2} | train {len(train_periods):>2} periodes "
                  f"({len(df_tr):>6,} obs) | test {len(df_te):>5,} | "
                  f"MAE = {mae:>14,.0f} | {len(res.params):>3} params | {cvg}")

        train_periods.append(period)

    if skipped:
        print(f"\n!! Etapes sautees : {skipped}")
    if not records:
        return pd.DataFrame(), modeles, None, None

    return (pd.concat(records, ignore_index=True), modeles,
            dernier_res, dernier_contexte)


print("\n" + "=" * 70)
print(f"ROLLING FORECAST GLM - {INITIAL_TRAIN_PERIODS} trimestres initiaux")
print("=" * 70)

pred_glm, modeles_glm, res_final_glm, contexte_glm = rolling_forecasting_glm(
    df_raw=df, target=TARGET, num_cols_all=NUM_COLS_ALL, cat_cols=ID_COLS,
    initial_train_periods=INITIAL_TRAIN_PERIODS, verbose=True)

pred_glm["year_quarter"] = (pred_glm["year"].astype(str)
                            + "Q" + pred_glm["quarter"].astype(str))
print(f"\nPredictions GLM generees : {len(pred_glm):,}")

# Diagnostics sur le dernier modele ajuste (le plus riche en historique)
tableau_coefs = diagnostics_glm(res_final_glm,
                                "DIAGNOSTICS GLM - dernier modele du rolling")


# =============================================================================
# BLOC 5 - EVALUATION  (fonction identique au pipeline LightGBM)
# =============================================================================
def evaluer(y_true, y_pred, titre="EVALUATION"):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]

    base = np.full_like(y_true, np.median(y_true))
    mae_m = mean_absolute_error(y_true, y_pred)
    mae_b = mean_absolute_error(y_true, base)

    print("\n" + "=" * 70)
    print(titre)
    print("=" * 70)
    print(f"  MAE modele     : {mae_m:>16,.0f}")
    print(f"  MAE baseline   : {mae_b:>16,.0f}   (predire la mediane)")
    print(f"  Gain vs base   : {(1 - mae_m / mae_b):>15.1%}"
          f"   {'<-- ALERTE : pire que trivial' if mae_m >= mae_b else ''}")
    print(f"  Bilan somme    : {y_pred.sum() / y_true.sum():>16.3f}   (cible ~1.000)")
    print(f"  Biais moyen    : {np.mean(y_pred - y_true):>+16,.0f}")

    edges  = [-np.inf] + [np.percentile(y_true, q) for q in (50, 90, 99)] + [np.inf]
    labels = ["P0-50", "P50-90", "P90-99", "P99+"]
    st = pd.cut(y_true, bins=edges, labels=labels)
    print(f"\n  {'strate':<8} {'n':>7} {'MAE':>16} {'biais moyen':>16} {'bilan':>8}")
    print("  " + "-" * 58)
    for lab in labels:
        m = (st == lab).to_numpy()
        if m.sum():
            print(f"  {lab:<8} {m.sum():>7,} "
                  f"{mean_absolute_error(y_true[m], y_pred[m]):>16,.0f} "
                  f"{np.mean(y_pred[m] - y_true[m]):>+16,.0f} "
                  f"{y_pred[m].sum() / y_true[m].sum():>8.2f}")
    print("=" * 70)
    return dict(mae=mae_m, mae_baseline=mae_b,
                bilan=y_pred.sum() / y_true.sum())


res_eval_glm = evaluer(pred_glm["Valeur_reelle"], pred_glm["Valeur_predite"],
                       "ROLLING FORECAST GLM - performance out-of-sample")


# =============================================================================
# BLOC 6 - GRAPHIQUES DE DIAGNOSTIC GLM
# -----------------------------------------------------------------------------
#  Six graphiques directement exploitables dans le memoire :
#    1. Residus de deviance vs predit   -> validite de la specification
#    2. QQ-plot des residus             -> adequation de la loi choisie
#    3. Coefficients + IC 95%           -> l'argument d'interpretabilite
#    4. Calibration par decile          -> le modele est-il juste en moyenne ?
#    5. Predit vs reel (log-log)        -> comparable au graphique LightGBM
#    6. Somme trimestrielle             -> lecture metier de la derive
# =============================================================================
def graphiques_glm(res, X_tr, y_tr, hist, tab_coefs,
                   titre="Diagnostic du GLM Tweedie"):
    fig, ax = plt.subplots(2, 3, figsize=(19, 11))
    fig.suptitle(titre, fontsize=15, fontweight="bold")

    mu       = np.asarray(res.fittedvalues, float)
    res_dev  = np.asarray(res.resid_deviance, float)
    ok       = np.isfinite(mu) & np.isfinite(res_dev)

    # 1 - residus de deviance vs valeurs predites
    ax[0, 0].scatter(mu[ok], res_dev[ok], s=5, alpha=.25, color="indianred")
    ax[0, 0].axhline(0, color="k", ls="--", lw=1.2)
    ax[0, 0].set_xscale("symlog")
    ax[0, 0].set_xlabel("valeur predite (mu)")
    ax[0, 0].set_ylabel("residu de deviance")
    ax[0, 0].set_title("1. Residus de deviance vs predit")

    # 2 - QQ-plot des residus de deviance
    r = np.sort(res_dev[ok])
    q_theo = stats.norm.ppf(np.linspace(0.001, 0.999, len(r)))
    ax[0, 1].scatter(q_theo, r, s=5, alpha=.3, color="darkslategray")
    lim = [min(q_theo.min(), r.min()), max(q_theo.max(), r.max())]
    ax[0, 1].plot(lim, lim, "r--", lw=1.5)
    ax[0, 1].set_xlabel("quantiles theoriques")
    ax[0, 1].set_ylabel("quantiles observes")
    ax[0, 1].set_title("2. QQ-plot des residus de deviance")

    # 3 - coefficients significatifs avec intervalle de confiance a 95%
    top = tab_coefs.head(15).iloc[::-1]
    ax[0, 2].errorbar(top["coef"], range(len(top)),
                      xerr=1.96 * top["std_err"], fmt="o",
                      color="steelblue", ecolor="lightsteelblue",
                      capsize=3, markersize=5)
    ax[0, 2].axvline(0, color="k", ls="--", lw=1.2)
    ax[0, 2].set_yticks(range(len(top)))
    ax[0, 2].set_yticklabels([s[:32] for s in top.index], fontsize=8)
    ax[0, 2].set_xlabel("coefficient (echelle log)")
    ax[0, 2].set_title("3. Coefficients + IC 95%")

    # 4 - calibration par decile de prediction
    y_true = hist["Valeur_reelle"].to_numpy(float)
    y_pred = hist["Valeur_predite"].to_numpy(float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    dec = pd.qcut(pd.Series(y_pred[m]).rank(method="first"), 10, labels=False)
    calib = (pd.DataFrame({"reel": y_true[m], "pred": y_pred[m], "d": dec})
             .groupby("d")[["reel", "pred"]].mean())
    ax[1, 0].plot(calib.index, calib["reel"], "o-", lw=2,
                  color="darkgreen", label="moyenne reelle")
    ax[1, 0].plot(calib.index, calib["pred"], "s--", lw=2,
                  color="darkorange", label="moyenne predite")
    ax[1, 0].set_xlabel("decile de prediction")
    ax[1, 0].set_yscale("symlog")
    ax[1, 0].set_title("4. Calibration par decile")
    ax[1, 0].legend()

    # 5 - predit vs reel en log-log
    p = m & (y_true > 0) & (y_pred > 0)
    ax[1, 1].scatter(y_true[p], y_pred[p], s=5, alpha=.25, color="darkslategray")
    lim = [min(y_true[p].min(), y_pred[p].min()),
           max(y_true[p].max(), y_pred[p].max())]
    ax[1, 1].plot(lim, lim, "r--", lw=1.5, label="y = x")
    ax[1, 1].set_xscale("log"); ax[1, 1].set_yscale("log")
    ax[1, 1].set_xlabel("reel"); ax[1, 1].set_ylabel("predit")
    ax[1, 1].set_title("5. Predit vs Reel (GLM)")
    ax[1, 1].legend()

    # 6 - somme par trimestre
    if "year_quarter" in hist.columns:
        g = (hist.groupby("year_quarter")[["Valeur_reelle", "Valeur_predite"]]
             .sum().sort_index())
        g.plot(ax=ax[1, 2], marker="o", lw=2)
        ax[1, 2].set_title("6. Somme par trimestre : reel vs predit")
        ax[1, 2].tick_params(axis="x", rotation=45)
    else:
        ax[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


graphiques_glm(res_final_glm, contexte_glm["X_tr"], contexte_glm["y_tr"],
               pred_glm, tableau_coefs,
               "Diagnostic - GLM Tweedie (rolling forecast)")


# =============================================================================
# BLOC 7 - COMPARAISON GLM vs LightGBM
# -----------------------------------------------------------------------------
#  C'est ce bloc qui produit le tableau comparatif du memoire.
#  Il ne s'execute que si le pipeline LightGBM a tourne dans la meme session
#  (variable `prediction_history_opt` disponible).
# =============================================================================
def comparer_modeles(pred_a, pred_b, nom_a="GLM Tweedie", nom_b="LightGBM"):
    """Compare deux jeux de predictions sur le meme perimetre d'evaluation."""
    lignes = []
    for nom, p in ((nom_a, pred_a), (nom_b, pred_b)):
        yt = p["Valeur_reelle"].to_numpy(float)
        yp = p["Valeur_predite"].to_numpy(float)
        ok = np.isfinite(yt) & np.isfinite(yp)
        yt, yp = yt[ok], yp[ok]

        o    = np.argsort(yp)
        cum  = np.cumsum(yt[o]) / yt.sum()
        gini = 1 - 2 * np.trapz(cum, np.linspace(0, 1, len(cum)))

        lignes.append({
            "modele": nom,
            "MAE": mean_absolute_error(yt, yp),
            "RMSE": float(np.sqrt(np.mean((yt - yp) ** 2))),
            "bilan_somme": yp.sum() / yt.sum(),
            "Gini": gini,
            "n": len(yt),
        })

    comp = pd.DataFrame(lignes)
    print("\n" + "=" * 70)
    print("COMPARAISON DES MODELES - meme protocole, meme perimetre")
    print("=" * 70)
    print(comp.round(4).to_string(index=False))

    mae_a, mae_b = comp.loc[0, "MAE"], comp.loc[1, "MAE"]
    ecart = (mae_a - mae_b) / mae_a
    print(f"\n  Gain de {nom_b} sur {nom_a} (MAE) : {ecart:>7.1%}")
    if ecart < 0.05:
        print("  => Gain faible : le GLM reste competitif. Son interpretabilite")
        print("     devient alors un argument decisif en contexte de validation.")
    print("=" * 70)

    # --- graphique comparatif : Lorenz + MAE par strate ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"{nom_a} vs {nom_b}", fontsize=14, fontweight="bold")

    for nom, p, coul in ((nom_a, pred_a, "darkorange"),
                         (nom_b, pred_b, "darkgreen")):
        yt = p["Valeur_reelle"].to_numpy(float)
        yp = p["Valeur_predite"].to_numpy(float)
        ok = np.isfinite(yt) & np.isfinite(yp)
        yt, yp = yt[ok], yp[ok]
        o   = np.argsort(yp)
        cum = np.cumsum(yt[o]) / yt.sum()
        xs  = np.linspace(0, 1, len(cum))
        g   = 1 - 2 * np.trapz(cum, xs)
        ax[0].plot(xs, cum, lw=2, color=coul, label=f"{nom} (Gini={g:.3f})")
    ax[0].plot([0, 1], [0, 1], "r--", lw=1.2, label="aleatoire")
    ax[0].set_title("Courbe de Lorenz - pouvoir de tri")
    ax[0].legend()

    labels = ["P0-50", "P50-90", "P90-99", "P99+"]
    largeur, positions = 0.38, np.arange(len(labels))
    for i, (nom, p, coul) in enumerate(((nom_a, pred_a, "darkorange"),
                                        (nom_b, pred_b, "darkgreen"))):
        yt = p["Valeur_reelle"].to_numpy(float)
        yp = p["Valeur_predite"].to_numpy(float)
        ok = np.isfinite(yt) & np.isfinite(yp)
        yt, yp = yt[ok], yp[ok]
        edges = [-np.inf] + [np.percentile(yt, q) for q in (50, 90, 99)] + [np.inf]
        st = pd.cut(yt, bins=edges, labels=labels)
        maes = [mean_absolute_error(yt[(st == l).to_numpy()],
                                    yp[(st == l).to_numpy()])
                if (st == l).sum() else 0 for l in labels]
        ax[1].bar(positions + i * largeur, maes, largeur, label=nom, color=coul)
    ax[1].set_xticks(positions + largeur / 2)
    ax[1].set_xticklabels(labels)
    ax[1].set_yscale("log")
    ax[1].set_title("MAE par strate de cible")
    ax[1].legend()

    plt.tight_layout()
    plt.show()
    return comp


try:
    comparaison = comparer_modeles(pred_glm, prediction_history_opt)
    comparaison.to_csv("comparaison_glm_lgbm.csv", index=False)
except NameError:
    print("\n(Comparaison ignoree : lancer d'abord le pipeline LightGBM "
          "dans la meme session pour disposer de `prediction_history_opt`.)")
    comparaison = None


# =============================================================================
# BLOC 8 - SAUVEGARDE DES ARTEFACTS
# =============================================================================
joblib.dump(res_final_glm,  "glm_final_model.pkl")
joblib.dump(contexte_glm,   "glm_contexte_encodage.pkl")
pred_glm.to_csv("predictions_rolling_glm.csv", index=False)
tableau_coefs.to_csv("glm_coefficients.csv")

print("\nArtefacts sauvegardes :")
print("   glm_final_model.pkl        - modele statsmodels ajuste")
print("   glm_contexte_encodage.pkl  - encodage (indispensable pour rejouer)")
print("   predictions_rolling_glm.csv")
print("   glm_coefficients.csv       - table des coefficients pour le memoire")
if comparaison is not None:
    print("   comparaison_glm_lgbm.csv   - tableau comparatif")
