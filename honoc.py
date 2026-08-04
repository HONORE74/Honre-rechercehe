import numpy as np, pandas as pd, lightgbm as lgb, joblib, warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
warnings.filterwarnings("ignore")

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "grid.linestyle": ":", "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})

RANDOM_STATE = 42
TARGET       = "RBNS_eop"
QUARTER_MAP  = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

# Decoupage chronologique : test = 2 derniers trimestres, validation = 2 avant
N_TEST_PERIODS  = 2
N_VALID_PERIODS = 2

FIGDIR = "./figures"
import os; os.makedirs(FIGDIR, exist_ok=True)
def sauver(nom):
    plt.savefig(f"{FIGDIR}/{nom}.png", dpi=200, bbox_inches="tight")








chemin = "/domino/datasets/local/conformal_pred_actuariat/DataSet/Dossier_concatener/data_model_f20.parquet"
df_model = pd.read_parquet(chemin)

df = df_model.copy()
df["year"] = df["annee"].astype(int)
if "quarter" not in df.columns:
    df["quarter"] = df["Time"].str.strip().map(QUARTER_MAP).astype(int)
df["time_idx"] = (df["year"] - df["year"].min()) * 4 + df["quarter"]
df = df.sort_values("time_idx").reset_index(drop=True)

periodes = (df[["year","quarter","time_idx"]].drop_duplicates()
            .sort_values("time_idx").reset_index(drop=True))
print(f"Lignes : {len(df):,} | Colonnes : {df.shape[1]} | Periodes : {len(periodes)}")
print(periodes.to_string(index=False))

# --- TEST 1 : integrite des donnees ---
assert df[TARGET].notna().all(),        "La cible contient des NaN"
assert df["time_idx"].notna().all(),    "time_idx contient des NaN"
assert len(periodes) > N_TEST_PERIODS + N_VALID_PERIODS + 4, "Pas assez de periodes"
print("\nTEST 1 -- integrite : OK")










y_all = df[TARGET].astype(float)

print("=" * 74)
print(f"  PROFIL DE LA CIBLE : {TARGET}")
print("=" * 74)
print(f"  n            : {len(y_all):>18,}")
print(f"  min / max    : {y_all.min():>18,.2f} / {y_all.max():,.0f}")
print(f"  moyenne      : {y_all.mean():>18,.0f}")
print(f"  mediane      : {y_all.median():>18,.0f}")
print(f"  ecart-type   : {y_all.std():>18,.0f}")
print(f"  asymetrie    : {y_all.skew():>18.2f}")
print(f"  kurtosis     : {y_all.kurtosis():>18.2f}")
print(f"  zeros        : {(y_all == 0).sum():>18,}")
print(f"  negatifs     : {(y_all < 0).sum():>18,}")
for q in (25, 50, 75, 90, 95, 99, 99.9):
    print(f"  P{q:<10} : {np.percentile(y_all, q):>18,.0f}")
top1 = y_all.nlargest(max(1, int(len(y_all)*0.01))).sum() / y_all.sum()
print(f"\n  Part du top 1% dans la somme totale : {top1:.1%}")
print("=" * 74)


def graphiques_cible(y):
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    # (1) Corps de la distribution -- sous le P95, echelle lineaire
    p95 = np.percentile(y, 95)
    ax[0,0].hist(y[y <= p95], bins=60, color="steelblue", edgecolor="white")
    ax[0,0].axvline(y.median(), color="red", ls="--", label=f"Mediane {y.median():,.0f}")
    ax[0,0].set_title("1. Corps de la distribution (jusqu'au P95)")
    ax[0,0].set_xlabel("EUR"); ax[0,0].legend(fontsize=8)

    # (2) Queue -- au-dela du P95
    ax[0,1].hist(y[y > p95], bins=50, color="firebrick", edgecolor="white")
    ax[0,1].set_title(f"2. Queue de la distribution (au-dela de {p95:,.0f})")
    ax[0,1].set_xlabel("EUR")

    # (3) Boxplot par strate (lineaire dans chaque strate)
    edges  = [-np.inf, *[np.percentile(y, q) for q in (50, 90, 99)], np.inf]
    labels = ["P0-50","P50-90","P90-99","P99+"]
    strate = pd.cut(y, bins=edges, labels=labels)
    data   = [y[strate == l].values for l in labels]
    bp = ax[0,2].boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for p, c in zip(bp["boxes"], ["#B5D4F4","#85B7EB","#378ADD","#185FA5"]):
        p.set_facecolor(c)
    ax[0,2].set_title("3. Dispersion par strate de magnitude")
    ax[0,2].set_ylabel("EUR")

    # (4) Concentration : % du total porte par % des observations
    tri = np.sort(y.values)[::-1]
    cum = np.cumsum(tri) / tri.sum()
    pct = np.arange(1, len(tri)+1) / len(tri) * 100
    ax[1,0].plot(pct, cum*100, color="darkgreen", lw=2)
    for seuil in (1, 5, 10, 25):
        v = cum[int(len(cum)*seuil/100)] * 100
        ax[1,0].plot([seuil, seuil], [0, v], "r:", lw=1)
        ax[1,0].annotate(f"{seuil}% -> {v:.0f}%", (seuil, v), fontsize=8,
                         xytext=(4, -8), textcoords="offset points")
    ax[1,0].set_xlabel("% des observations (triees decroissant)")
    ax[1,0].set_ylabel("% du montant total")
    ax[1,0].set_title("4. Concentration du montant")

    # (5) Effectifs par strate
    ax[1,1].bar(labels, [len(d) for d in data], color="slateblue")
    for i, d in enumerate(data):
        ax[1,1].text(i, len(d), f"{len(d):,}", ha="center", va="bottom", fontsize=9)
    ax[1,1].set_title("5. Effectif par strate")

    # (6) Evolution temporelle
    g = df.groupby("time_idx")[TARGET].agg(["sum","median","size"])
    ax[1,2].plot(g.index, g["sum"], marker="o", color="navy", label="Somme")
    ax[1,2].set_xlabel("time_idx"); ax[1,2].set_ylabel("Somme (EUR)")
    a2 = ax[1,2].twinx()
    a2.plot(g.index, g["median"], marker="s", color="darkorange", label="Mediane")
    a2.set_ylabel("Mediane (EUR)"); a2.grid(False)
    ax[1,2].set_title("6. Evolution temporelle")
    ax[1,2].legend(loc="upper left", fontsize=8); a2.legend(loc="lower right", fontsize=8)

    plt.tight_layout(); sauver("02_distribution_cible"); plt.show()

graphiques_cible(y_all)










LEAKS        = ["Conso", "Currencies", "period", "Reinsurance"]
CANDIDATS_ID = ["Partner","Companies","Lob","Activity","Periodicity","Risk"]

introuvables = [c for c in LEAKS if c not in df.columns]
if introuvables:
    print(f"/!\\ LEAKS introuvables (donc NON exclus) : {introuvables}")
    print(f"    Colonnes proches : "
          f"{[c for c in df.columns if any(k in c.lower() for k in ['conso','curr','period','reins'])]}")

# Constantes (une seule modalite) -> inutiles par construction
CONSTANTES = [c for c in df.columns if c != TARGET and df[c].nunique(dropna=False) <= 1]

EXCLUDE = [c for c in ([TARGET, "time_idx", "year", "quarter", "annee", "Time"]
                       + LEAKS + CONSTANTES) if c in df.columns]
FEATURES = [c for c in df.columns if c not in EXCLUDE]

CATEGORIELLES = [c for c in FEATURES
                 if df[c].dtype == object or str(df[c].dtype) == "category"]
for c in CATEGORIELLES:
    df[c] = df[c].astype("category")
NUMERIQUES = [c for c in FEATURES if c not in CATEGORIELLES]

print(f"\nFeatures retenues : {len(FEATURES)}  "
      f"({len(NUMERIQUES)} numeriques, {len(CATEGORIELLES)} categorielles)")
print(f"Categorielles : {CATEGORIELLES}")
print(f"Constantes exclues : {CONSTANTES}")
print(f"Total exclu : {EXCLUDE}")


def graphiques_features(df, features, numeriques, categorielles, target):
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))

    # (1) Taux de valeurs manquantes
    nan = (df[features].isna().mean() * 100).sort_values(ascending=False).head(20)
    nan = nan[nan > 0]
    if len(nan):
        ax[0,0].barh(range(len(nan)), nan.values, color="coral")
        ax[0,0].set_yticks(range(len(nan))); ax[0,0].set_yticklabels(nan.index, fontsize=8)
        ax[0,0].set_xlabel("% de NaN")
    else:
        ax[0,0].text(.5,.5,"Aucune valeur manquante", ha="center", transform=ax[0,0].transAxes)
    ax[0,0].set_title("1. Valeurs manquantes (top 20)")

    # (2) Cardinalite des categorielles
    if categorielles:
        card = pd.Series({c: df[c].nunique() for c in categorielles}).sort_values()
        coul = ["firebrick" if v/len(df) > 0.05 else "seagreen" for v in card]
        ax[0,1].barh(range(len(card)), card.values, color=coul)
        ax[0,1].set_yticks(range(len(card))); ax[0,1].set_yticklabels(card.index, fontsize=9)
        ax[0,1].set_xlabel("Nombre de modalites")
        for i, v in enumerate(card.values):
            ax[0,1].text(v, i, f" {v} ({v/len(df):.1%})", va="center", fontsize=7.5)
    ax[0,1].set_title("2. Cardinalite -- rouge = > 5% des lignes")

    # (3) Correlation avec la cible (Spearman : robuste, sans hypothese de linearite)
    cors = {}
    for c in numeriques:
        s = df[[c, target]].dropna()
        if len(s) > 100 and s[c].nunique() > 1:
            cors[c] = stats.spearmanr(s[c], s[target]).statistic
    cs = pd.Series(cors).abs().sort_values(ascending=False).head(20)
    coul = ["firebrick" if v > 0.95 else "darkorange" if v > 0.80 else "steelblue"
            for v in cs.values]
    ax[1,0].barh(range(len(cs)), cs.values, color=coul)
    ax[1,0].set_yticks(range(len(cs))); ax[1,0].set_yticklabels(cs.index, fontsize=8)
    ax[1,0].axvline(0.95, color="red", ls="--", lw=1, label="Seuil de fuite (0.95)")
    ax[1,0].set_xlabel("|Spearman| avec la cible"); ax[1,0].legend(fontsize=8)
    ax[1,0].set_title("3. Correlation avec la cible -- rouge = fuite suspecte")

    # (4) Repartition des lignes par periode
    g = df.groupby("time_idx").size()
    ax[1,1].bar(g.index, g.values, color="steelblue")
    ax[1,1].axhline(g.mean(), color="red", ls="--", label=f"Moyenne {g.mean():,.0f}")
    ax[1,1].set_xlabel("time_idx"); ax[1,1].set_ylabel("Nombre de lignes")
    ax[1,1].set_title("4. Volume par periode"); ax[1,1].legend(fontsize=8)

    plt.tight_layout(); sauver("03_audit_features"); plt.show()
    return cs

correlations = graphiques_features(df, FEATURES, NUMERIQUES, CATEGORIELLES, TARGET)

# --- TEST 2 : detection de fuite ---
suspects = correlations[correlations > 0.95]
if len(suspects):
    print(f"\n/!\\ TEST 2 -- {len(suspects)} feature(s) a |correlation| > 0.95 :")
    print(suspects.to_string())
    print("    -> A auditer : sont-elles legitimement connues avant la periode predite ?")
else:
    print("\nTEST 2 -- aucune correlation suspecte : OK")













toutes = sorted(df["time_idx"].unique())
p_test  = toutes[-N_TEST_PERIODS:]
p_valid = toutes[-(N_TEST_PERIODS + N_VALID_PERIODS):-N_TEST_PERIODS]
p_train = toutes[:-(N_TEST_PERIODS + N_VALID_PERIODS)]

m_train = df["time_idx"].isin(p_train)
m_valid = df["time_idx"].isin(p_valid)
m_test  = df["time_idx"].isin(p_test)

X_train, y_train = df.loc[m_train, FEATURES], df.loc[m_train, TARGET].astype(float)
X_valid, y_valid = df.loc[m_valid, FEATURES], df.loc[m_valid, TARGET].astype(float)
X_test,  y_test  = df.loc[m_test,  FEATURES], df.loc[m_test,  TARGET].astype(float)

print("=" * 74)
print(f"  TRAIN      : periodes {p_train[0]}-{p_train[-1]}   {len(X_train):>7,} lignes "
      f"({len(X_train)/len(df):.1%})")
print(f"  VALIDATION : periodes {p_valid[0]}-{p_valid[-1]}   {len(X_valid):>7,} lignes "
      f"({len(X_valid)/len(df):.1%})   [early stopping]")
print(f"  TEST       : periodes {p_test[0]}-{p_test[-1]}   {len(X_test):>7,} lignes "
      f"({len(X_test)/len(df):.1%})   [jamais vu]")
print("=" * 74)

# --- TEST 3 : etancheite du decoupage ---
assert not (set(p_train) & set(p_valid)),      "Chevauchement train/validation"
assert not (set(p_valid) & set(p_test)),       "Chevauchement validation/test"
assert max(p_train) < min(p_valid) < min(p_test), "Ordre chronologique non respecte"
print("TEST 3 -- etancheite et ordre chronologique : OK")


def graphiques_split(df, m_train, m_valid, m_test, target):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    # (1) Frise du decoupage
    for m, coul, lab in [(m_train,"steelblue","Train"), (m_valid,"darkorange","Validation"),
                         (m_test,"firebrick","Test")]:
        g = df.loc[m].groupby("time_idx").size()
        ax[0].bar(g.index, g.values, color=coul, label=lab)
    ax[0].set_xlabel("time_idx"); ax[0].set_ylabel("Lignes")
    ax[0].set_title("1. Decoupage chronologique"); ax[0].legend(fontsize=9)

    # (2) Distribution de la cible par bloc (corps, sans les extremes)
    p95 = np.percentile(df[target], 95)
    for m, coul, lab in [(m_train,"steelblue","Train"), (m_valid,"darkorange","Validation"),
                         (m_test,"firebrick","Test")]:
        v = df.loc[m, target]
        ax[1].hist(v[v <= p95], bins=45, alpha=0.5, color=coul, density=True, label=lab)
    ax[1].set_xlabel(f"{target} (jusqu'au P95)"); ax[1].set_ylabel("Densite")
    ax[1].set_title("2. Comparabilite des distributions"); ax[1].legend(fontsize=9)

    # (3) Statistiques de la cible par periode, avec les blocs colores
    g = df.groupby("time_idx")[target].agg(["median","mean"])
    ax[2].plot(g.index, g["median"], marker="o", color="navy", label="Mediane")
    ax[2].axvspan(min(p_valid)-0.5, max(p_valid)+0.5, color="darkorange", alpha=0.15)
    ax[2].axvspan(min(p_test)-0.5,  max(p_test)+0.5,  color="firebrick",  alpha=0.15)
    ax[2].set_xlabel("time_idx"); ax[2].set_ylabel("Mediane (EUR)")
    ax[2].set_title("3. Stabilite temporelle (zones = valid/test)"); ax[2].legend(fontsize=9)

    plt.tight_layout(); sauver("04_split_chronologique"); plt.show()

graphiques_split(df, m_train, m_valid, m_test, TARGET)

# --- TEST 4 : derive de distribution train vs test ---
print("\nTEST 4 -- derive (Kolmogorov-Smirnov train vs test)")
derives = []
for c in NUMERIQUES:
    a, b = X_train[c].dropna(), X_test[c].dropna()
    if len(a) > 50 and len(b) > 50:
        ks, p = stats.ks_2samp(a, b)
        derives.append({"feature": c, "KS": ks, "p_value": p})
d = pd.DataFrame(derives).sort_values("KS", ascending=False)
print(f"  Features avec derive significative (p < 0.01) : "
      f"{(d['p_value'] < 0.01).sum()} / {len(d)}")
print(d.head(8).to_string(index=False))

ks_cible, p_cible = stats.ks_2samp(y_train, y_test)
print(f"\n  Derive de la CIBLE : KS={ks_cible:.4f}, p={p_cible:.2e}")
print(f"  -> {'DERIVE SIGNIFICATIVE de la cible' if p_cible < 0.01 else 'cible stable'}")









toutes = sorted(df["time_idx"].unique())
p_test  = toutes[-N_TEST_PERIODS:]
p_valid = toutes[-(N_TEST_PERIODS + N_VALID_PERIODS):-N_TEST_PERIODS]
p_train = toutes[:-(N_TEST_PERIODS + N_VALID_PERIODS)]

m_train = df["time_idx"].isin(p_train)
m_valid = df["time_idx"].isin(p_valid)
m_test  = df["time_idx"].isin(p_test)

X_train, y_train = df.loc[m_train, FEATURES], df.loc[m_train, TARGET].astype(float)
X_valid, y_valid = df.loc[m_valid, FEATURES], df.loc[m_valid, TARGET].astype(float)
X_test,  y_test  = df.loc[m_test,  FEATURES], df.loc[m_test,  TARGET].astype(float)

print("=" * 74)
print(f"  TRAIN      : periodes {p_train[0]}-{p_train[-1]}   {len(X_train):>7,} lignes "
      f"({len(X_train)/len(df):.1%})")
print(f"  VALIDATION : periodes {p_valid[0]}-{p_valid[-1]}   {len(X_valid):>7,} lignes "
      f"({len(X_valid)/len(df):.1%})   [early stopping]")
print(f"  TEST       : periodes {p_test[0]}-{p_test[-1]}   {len(X_test):>7,} lignes "
      f"({len(X_test)/len(df):.1%})   [jamais vu]")
print("=" * 74)

# --- TEST 3 : etancheite du decoupage ---
assert not (set(p_train) & set(p_valid)),      "Chevauchement train/validation"
assert not (set(p_valid) & set(p_test)),       "Chevauchement validation/test"
assert max(p_train) < min(p_valid) < min(p_test), "Ordre chronologique non respecte"
print("TEST 3 -- etancheite et ordre chronologique : OK")


def graphiques_split(df, m_train, m_valid, m_test, target):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    # (1) Frise du decoupage
    for m, coul, lab in [(m_train,"steelblue","Train"), (m_valid,"darkorange","Validation"),
                         (m_test,"firebrick","Test")]:
        g = df.loc[m].groupby("time_idx").size()
        ax[0].bar(g.index, g.values, color=coul, label=lab)
    ax[0].set_xlabel("time_idx"); ax[0].set_ylabel("Lignes")
    ax[0].set_title("1. Decoupage chronologique"); ax[0].legend(fontsize=9)

    # (2) Distribution de la cible par bloc (corps, sans les extremes)
    p95 = np.percentile(df[target], 95)
    for m, coul, lab in [(m_train,"steelblue","Train"), (m_valid,"darkorange","Validation"),
                         (m_test,"firebrick","Test")]:
        v = df.loc[m, target]
        ax[1].hist(v[v <= p95], bins=45, alpha=0.5, color=coul, density=True, label=lab)
    ax[1].set_xlabel(f"{target} (jusqu'au P95)"); ax[1].set_ylabel("Densite")
    ax[1].set_title("2. Comparabilite des distributions"); ax[1].legend(fontsize=9)

    # (3) Statistiques de la cible par periode, avec les blocs colores
    g = df.groupby("time_idx")[target].agg(["median","mean"])
    ax[2].plot(g.index, g["median"], marker="o", color="navy", label="Mediane")
    ax[2].axvspan(min(p_valid)-0.5, max(p_valid)+0.5, color="darkorange", alpha=0.15)
    ax[2].axvspan(min(p_test)-0.5,  max(p_test)+0.5,  color="firebrick",  alpha=0.15)
    ax[2].set_xlabel("time_idx"); ax[2].set_ylabel("Mediane (EUR)")
    ax[2].set_title("3. Stabilite temporelle (zones = valid/test)"); ax[2].legend(fontsize=9)

    plt.tight_layout(); sauver("04_split_chronologique"); plt.show()

graphiques_split(df, m_train, m_valid, m_test, TARGET)

# --- TEST 4 : derive de distribution train vs test ---
print("\nTEST 4 -- derive (Kolmogorov-Smirnov train vs test)")
derives = []
for c in NUMERIQUES:
    a, b = X_train[c].dropna(), X_test[c].dropna()
    if len(a) > 50 and len(b) > 50:
        ks, p = stats.ks_2samp(a, b)
        derives.append({"feature": c, "KS": ks, "p_value": p})
d = pd.DataFrame(derives).sort_values("KS", ascending=False)
print(f"  Features avec derive significative (p < 0.01) : "
      f"{(d['p_value'] < 0.01).sum()} / {len(d)}")
print(d.head(8).to_string(index=False))

ks_cible, p_cible = stats.ks_2samp(y_train, y_test)
print(f"\n  Derive de la CIBLE : KS={ks_cible:.4f}, p={p_cible:.2e}")
print(f"  -> {'DERIVE SIGNIFICATIVE de la cible' if p_cible < 0.01 else 'cible stable'}")








def evaluer(y_true, y_pred, nom=""):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    f = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[f], y_pred[f]
    return {"modele": nom,
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "wMAPE_%": np.abs(y_true-y_pred).sum()/y_true.sum()*100,
            "bilan": y_pred.sum()/y_true.sum()}

baselines = [
    evaluer(y_test, np.full(len(y_test), y_train.median()), "Mediane du train"),
    evaluer(y_test, np.full(len(y_test), y_train.mean()),   "Moyenne du train"),
]
if "RBNS_bop" in df.columns:
    baselines.append(evaluer(y_test, df.loc[m_test, "RBNS_bop"].values,
                             "Persistance (RBNS_bop)"))

tab_base = pd.DataFrame(baselines)
print("=" * 74); print("  BASELINES A BATTRE"); print("=" * 74)
print(tab_base.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
SEUIL_MAE = tab_base["MAE"].min()
print(f"\n  >>> Le modele doit obtenir une MAE inferieure a {SEUIL_MAE:,.0f}")









PARAMS = dict(
    objective="tweedie",
    tweedie_variance_power=1.2154502817516997,
    learning_rate=0.031124174985263424,
    num_leaves=62,
    min_child_samples=32,
    colsample_bytree=0.9223551556702686,
    subsample=0.5714157915306912,
    reg_alpha=8.644759633929173,
    reg_lambda=0.001896001409587627,
    max_bin=410,
    n_estimators=4000,
    max_depth=-1,
    subsample_freq=1,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)

modele = lgb.LGBMRegressor(**PARAMS)
evals = {}
modele.fit(X_train, y_train,
           eval_set=[(X_train, y_train), (X_valid, y_valid)],
           eval_names=["train", "validation"],
           eval_metric="mae",
           callbacks=[lgb.early_stopping(150, verbose=False),
                      lgb.log_evaluation(0),
                      lgb.record_evaluation(evals)])

best_it = modele.best_iteration_ or PARAMS["n_estimators"]
print(f"Arbres retenus par early stopping : {best_it} / {PARAMS['n_estimators']}")

pred_train = np.clip(modele.predict(X_train, num_iteration=best_it), 0, None)
pred_valid = np.clip(modele.predict(X_valid, num_iteration=best_it), 0, None)
pred_test  = np.clip(modele.predict(X_test,  num_iteration=best_it), 0, None)


def graphique_apprentissage(evals, best_it):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for nom, coul in [("train","steelblue"), ("validation","firebrick")]:
        v = evals[nom]["l1"]
        ax[0].plot(v, color=coul, lw=1.6, label=nom)
    ax[0].axvline(best_it, color="green", ls="--", label=f"Early stopping ({best_it})")
    ax[0].set_xlabel("Iteration"); ax[0].set_ylabel("MAE")
    ax[0].set_title("Courbe d'apprentissage"); ax[0].legend(fontsize=9)

    ecart = np.array(evals["validation"]["l1"]) - np.array(evals["train"]["l1"])
    ax[1].plot(ecart, color="darkviolet", lw=1.6)
    ax[1].axvline(best_it, color="green", ls="--")
    ax[1].axhline(0, color="black", lw=0.8)
    ax[1].set_xlabel("Iteration"); ax[1].set_ylabel("MAE validation - MAE train")
    ax[1].set_title("Ecart de generalisation (surapprentissage si croissant)")
    plt.tight_layout(); sauver("06_apprentissage"); plt.show()

graphique_apprentissage(evals, best_it)

# --- TEST 5 : surapprentissage ---
mae_tr, mae_te = mean_absolute_error(y_train, pred_train), mean_absolute_error(y_test, pred_test)
ratio = mae_te / mae_tr
print(f"\nTEST 5 -- surapprentissage")
print(f"  MAE train : {mae_tr:>14,.0f}")
print(f"  MAE test  : {mae_te:>14,.0f}")
print(f"  Ratio     : {ratio:>14.2f}")
print(f"  -> {'SURAPPRENTISSAGE marque' if ratio > 2.5 else 'ecart acceptable'}")








res = pd.DataFrame(baselines + [
    evaluer(y_train, pred_train, "LightGBM (train)"),
    evaluer(y_valid, pred_valid, "LightGBM (validation)"),
    evaluer(y_test,  pred_test,  "LightGBM (TEST)")])
print("=" * 86); print("  COMPARAISON GLOBALE"); print("=" * 86)
print(res.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

gain = (1 - mae_te / SEUIL_MAE) * 100
print(f"\n  Gain du modele vs meilleure baseline : {gain:+.1f} %")
print(f"  -> {'Le modele apporte de la valeur' if gain > 0 else 'ATTENTION : pas mieux que le trivial'}")

# --- Evaluation stratifiee ---
def stratifie(y_true, y_pred, qs=(50, 90, 99)):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    edges  = [-np.inf, *[np.percentile(y_true, q) for q in qs], np.inf]
    labels = ["P0-50","P50-90","P90-99","P99+"]
    st = pd.cut(y_true, bins=edges, labels=labels)
    out = []
    for l in labels:
        m = (st == l).to_numpy()
        if m.sum():
            out.append({"strate": l, "n": int(m.sum()),
                        "MAE": mean_absolute_error(y_true[m], y_pred[m]),
                        "wMAPE_%": np.abs(y_true[m]-y_pred[m]).sum()/y_true[m].sum()*100,
                        "bilan": y_pred[m].sum()/y_true[m].sum(),
                        "biais_moyen": np.mean(y_pred[m]-y_true[m])})
    return pd.DataFrame(out)



tab_strat = stratifie(y_test, pred_test)
print("\n--- Performance par strate (TEST) ---")
print(tab_strat.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))










def graphiques_diagnostic(y_true, y_pred, tab_strat, df_test_meta):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    # (1) Predit vs reel -- cadre sur le P99 pour rester lisible
    p99 = np.percentile(y_true, 99)
    m = y_true <= p99
    ax[0,0].scatter(y_true[m], y_pred[m], s=7, alpha=.25, color="steelblue")
    lim = [0, p99]
    ax[0,0].plot(lim, lim, "r--", lw=1.5, label="Diagonale parfaite")
    ax[0,0].set_xlabel("Reel"); ax[0,0].set_ylabel("Predit")
    ax[0,0].set_xlim(lim); ax[0,0].set_ylim(lim)
    ax[0,0].set_title("1. Predit vs Reel (jusqu'au P99)"); ax[0,0].legend(fontsize=8)

    # (2) Residus vs predit -> detecte le biais systematique
    ax[0,1].scatter(y_pred[m], (y_true-y_pred)[m], s=7, alpha=.25, color="darkorange")
    ax[0,1].axhline(0, color="red", ls="--", lw=1.5)
    ax[0,1].set_xlabel("Predit"); ax[0,1].set_ylabel("Reel - Predit")
    ax[0,1].set_title("2. Residus (un nuage centre = pas de biais)")

    # (3) MAE par strate
    ax[0,2].bar(tab_strat["strate"], tab_strat["MAE"], color="indianred")
    for i, v in enumerate(tab_strat["MAE"]):
        ax[0,2].text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    ax[0,2].set_ylabel("MAE (EUR)"); ax[0,2].set_title("3. MAE par strate")

    # (4) Bilan predit/reel par strate
    ax[1,0].bar(tab_strat["strate"], tab_strat["bilan"], color="seagreen")
    ax[1,0].axhline(1, color="red", ls="--", lw=1.5, label="Equilibre parfait")
    ax[1,0].set_ylabel("Somme predite / somme reelle")
    ax[1,0].set_title("4. Bilan technique par strate"); ax[1,0].legend(fontsize=8)
    for i, v in enumerate(tab_strat["bilan"]):
        ax[1,0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # (5) Calibration par decile de prediction
    dec = pd.qcut(y_pred, 10, labels=False, duplicates="drop")
    cal = pd.DataFrame({"dec": dec, "reel": y_true, "pred": y_pred}).groupby("dec").mean()
    ax[1,1].plot(cal["pred"], cal["reel"], marker="o", color="navy", label="Observe")
    lm = [cal.min().min(), cal.max().max()]
    ax[1,1].plot(lm, lm, "r--", lw=1.5, label="Calibration parfaite")
    ax[1,1].set_xlabel("Moyenne predite par decile"); ax[1,1].set_ylabel("Moyenne reelle")
    ax[1,1].set_title("5. Calibration par decile"); ax[1,1].legend(fontsize=8)

    # (6) Courbe de Lorenz -- qualite de tri des risques
    o  = np.argsort(y_pred)[::-1]; c  = np.cumsum(y_true[o]) / y_true.sum()
    o2 = np.argsort(y_true)[::-1]; c2 = np.cumsum(y_true[o2]) / y_true.sum()
    xx = np.linspace(0, 1, len(c))
    ax[1,2].plot(xx, c,  color="navy",     lw=2, label="Modele")
    ax[1,2].plot(xx, c2, color="darkgreen", lw=1.5, ls="--", label="Tri parfait")
    ax[1,2].plot([0,1], [0,1], "r:", lw=1, label="Aleatoire")
    g  = (c.sum() - (len(c)+1)/2) / len(c)
    g2 = (c2.sum() - (len(c2)+1)/2) / len(c2)
    ax[1,2].set_title(f"6. Lorenz -- Gini normalise = {g/g2:.4f}")
    ax[1,2].legend(fontsize=8)

    plt.tight_layout(); sauver("08_diagnostic"); plt.show()
    return g/g2

gini = graphiques_diagnostic(y_test, pred_test, tab_strat, df.loc[m_test])
print(f"\nGini normalise (qualite de tri) : {gini:.4f}")










def graphiques_importance(modele, features, X_test, top=20):
    gain  = pd.Series(modele.booster_.feature_importance("gain"),  index=features)
    split = pd.Series(modele.booster_.feature_importance("split"), index=features)
    gp    = (gain / gain.sum() * 100).sort_values(ascending=False)

    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    t = gp.head(top).sort_values()
    coul = ["firebrick" if v > 40 else "darkorange" if v > 15 else "steelblue" for v in t]
    ax[0].barh(range(len(t)), t.values, color=coul)
    ax[0].set_yticks(range(len(t))); ax[0].set_yticklabels(t.index, fontsize=9)
    ax[0].set_xlabel("% du gain total")
    ax[0].set_title(f"Importance par GAIN (top {top}) -- rouge = > 40%")
    for i, v in enumerate(t.values):
        ax[0].text(v, i, f" {v:.1f}%", va="center", fontsize=8)

    cum = gp.cumsum()
    ax[1].plot(range(1, len(cum)+1), cum.values, color="darkgreen", lw=2)
    for s in (80, 90, 95):
        k = int((cum <= s).sum()) + 1
        ax[1].plot([k, k], [0, s], "r:", lw=1)
        ax[1].annotate(f"{k} features -> {s}%", (k, s), fontsize=8,
                       xytext=(5, -10), textcoords="offset points")
    ax[1].set_xlabel("Nombre de features (par importance decroissante)")
    ax[1].set_ylabel("% cumule du gain")
    ax[1].set_title("Concentration de l'information")

    plt.tight_layout(); sauver("09_importance"); plt.show()

    print("\n--- Top 15 features (% du gain) ---")
    print(gp.head(15).round(2).to_string())
    inutiles = gp[gp < 0.01]
    print(f"\nFeatures a importance quasi nulle (< 0.01%) : {len(inutiles)}")
    if len(inutiles):
        print(f"  Candidates a la suppression : {list(inutiles.index)[:15]}")
    dom = gp[gp > 40]
    if len(dom):
        print(f"\n/!\\ {list(dom.index)} concentre(nt) > 40% du gain -> verifier la fuite")
    return gp

importances = graphiques_importance(modele, FEATURES, X_test)

# --- SHAP (optionnel) ---
try:
    import shap
    ech = X_test.sample(min(2000, len(X_test)), random_state=RANDOM_STATE)
    expl = shap.TreeExplainer(modele)
    vals = expl.shap_values(ech)
    plt.figure(figsize=(11, 8))
    shap.summary_plot(vals, ech, max_display=20, show=False)
    plt.title("Contribution SHAP par feature", fontsize=12, fontweight="bold")
    plt.tight_layout(); sauver("09b_shap"); plt.show()
except ImportError:
    print("\nSHAP non installe (pip install shap) -- etape ignoree.")















top_feat = importances.index[0]
print(f"TEST 6 -- ablation de la feature dominante : '{top_feat}' "
      f"({importances.iloc[0]:.1f}% du gain)")

feats_sans = [c for c in FEATURES if c != top_feat]
m_sans = lgb.LGBMRegressor(**{**PARAMS, "n_estimators": best_it})
m_sans.fit(X_train[feats_sans], y_train)
mae_sans = mean_absolute_error(y_test, np.clip(m_sans.predict(X_test[feats_sans]), 0, None))

print(f"  MAE avec '{top_feat}' : {mae_te:>14,.0f}")
print(f"  MAE sans '{top_feat}' : {mae_sans:>14,.0f}   ({(mae_sans/mae_te-1)*100:+.1f} %)")
if mae_sans / mae_te > 4:
    print("  -> Cette feature porte presque toute l'information : FUITE PROBABLE.")
elif mae_sans / mae_te > 2:
    print("  -> Feature tres dominante : verifier qu'elle est connue avant la prediction.")
else:
    print("  -> Contribution normale, pas de signe de fuite.")













top_feat = importances.index[0]
print(f"TEST 6 -- ablation de la feature dominante : '{top_feat}' "
      f"({importances.iloc[0]:.1f}% du gain)")

feats_sans = [c for c in FEATURES if c != top_feat]
m_sans = lgb.LGBMRegressor(**{**PARAMS, "n_estimators": best_it})
m_sans.fit(X_train[feats_sans], y_train)
mae_sans = mean_absolute_error(y_test, np.clip(m_sans.predict(X_test[feats_sans]), 0, None))

print(f"  MAE avec '{top_feat}' : {mae_te:>14,.0f}")
print(f"  MAE sans '{top_feat}' : {mae_sans:>14,.0f}   ({(mae_sans/mae_te-1)*100:+.1f} %)")
if mae_sans / mae_te > 4:
    print("  -> Cette feature porte presque toute l'information : FUITE PROBABLE.")
elif mae_sans / mae_te > 2:
    print("  -> Feature tres dominante : verifier qu'elle est connue avant la prediction.")
else:
    print("  -> Contribution normale, pas de signe de fuite.")










resume = {
    "n_lignes": len(df), "n_features": len(FEATURES),
    "periodes_train": f"{p_train[0]}-{p_train[-1]}",
    "periodes_test":  f"{p_test[0]}-{p_test[-1]}",
    "arbres_retenus": int(best_it),
    "MAE_test": float(mae_te), "MAE_baseline": float(SEUIL_MAE),
    "gain_vs_baseline_%": float(gain),
    "ratio_surapprentissage": float(ratio),
    "gini_normalise": float(gini),
    "bilan_global": float(pred_test.sum()/y_test.sum()),
}
print("=" * 74); print("  SYNTHESE"); print("=" * 74)
for k, v in resume.items():
    print(f"  {k:<26} : {v:,.4f}" if isinstance(v, float) else f"  {k:<26} : {v}")

checks = [
    ("Le modele bat la meilleure baseline", gain > 0),
    ("Pas de surapprentissage marque (ratio < 2.5)", ratio < 2.5),
    ("Bilan global entre 0.95 et 1.05", 0.95 <= resume["bilan_global"] <= 1.05),
    ("Gini > 0.70 (tri des risques)", gini > 0.70),
    ("Aucune feature > 40% du gain", (importances > 40).sum() == 0),
    ("Cible stable entre train et test (p > 0.01)", p_cible > 0.01),
]
print("\n  VALIDATION FINALE")
for lab, ok in checks:
    print(f"  [{'OK ' if ok else 'KO '}] {lab}")
print(f"\n  SCORE : {sum(o for _, o in checks)}/{len(checks)}")
print("=" * 74)

joblib.dump({"modele": modele, "features": FEATURES, "categorielles": CATEGORIELLES,
             "params": PARAMS, "best_iteration": int(best_it), "resume": resume},
            "lgbm_rbns_final.pkl")

sortie = df.loc[m_test, [c for c in CANDIDATS_ID if c in df.columns]
                        + ["year","quarter","time_idx"]].copy()
sortie["y_obs"], sortie["y_pred"] = y_test.values, pred_test
sortie["residu"] = sortie["y_obs"] - sortie["y_pred"]
sortie.to_csv("predictions_test.csv", index=False)
print("\nSauvegarde : lgbm_rbns_final.pkl + predictions_test.csv")
print(f"Figures dans : {FIGDIR}/")









from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def benchmark_glm(X_train, y_train, X_test, y_test, numeriques, categorielles,
                  power=1.5, max_modalites=30):
    """GLM Tweedie : la reference actuarielle. Sans cette comparaison, impossible
    de justifier l'apport du ML devant un jury."""
    cats_ok = [c for c in categorielles if X_train[c].nunique() <= max_modalites]
    exclues = set(categorielles) - set(cats_ok)
    if exclues:
        print(f"  Categorielles exclues du GLM (cardinalite > {max_modalites}) : {exclues}")

    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc",  StandardScaler())]), numeriques),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01,
                              sparse_output=False), cats_ok)])

    glm = Pipeline([("prep", prep),
                    ("glm", TweedieRegressor(power=power, alpha=1e-3, link="log",
                                             max_iter=3000, tol=1e-5))])
    glm.fit(X_train, y_train)
    p_glm = np.clip(glm.predict(X_test), 0, None)

    print(f"\n  GLM Tweedie (power={power}) : {glm['prep'].transform(X_train.head(2)).shape[1]} "
          f"colonnes apres encodage")
    return glm, p_glm

glm, pred_glm = benchmark_glm(X_train, y_train, X_test, y_test, NUMERIQUES, CATEGORIELLES)

comp = pd.DataFrame([
    evaluer(y_test, np.full(len(y_test), y_train.median()), "Baseline mediane"),
    evaluer(y_test, pred_glm,  "GLM Tweedie"),
    evaluer(y_test, pred_test, "LightGBM Tweedie")])
comp["gain_vs_GLM_%"] = (1 - comp["MAE"] / comp.loc[1, "MAE"]) * 100

print("=" * 90); print("  ML CONTRE GLM -- LA COMPARAISON QUE LE JURY ATTEND"); print("=" * 90)
print(comp.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

def graphique_ml_vs_glm(y_true, p_glm, p_lgb):
    y_true = np.asarray(y_true, float)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))

    edges  = [-np.inf, *[np.percentile(y_true, q) for q in (50,90,99)], np.inf]
    labels = ["P0-50","P50-90","P90-99","P99+"]
    st = pd.cut(y_true, bins=edges, labels=labels)
    m_glm = [mean_absolute_error(y_true[(st==l).to_numpy()], p_glm[(st==l).to_numpy()]) for l in labels]
    m_lgb = [mean_absolute_error(y_true[(st==l).to_numpy()], p_lgb[(st==l).to_numpy()]) for l in labels]
    x = np.arange(len(labels)); w = 0.38
    ax[0].bar(x-w/2, m_glm, w, color="#DAA520", label="GLM Tweedie")
    ax[0].bar(x+w/2, m_lgb, w, color="#2E8B57", label="LightGBM")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("MAE (EUR)"); ax[0].set_title("MAE par strate")
    ax[0].legend(fontsize=9)

    for p, coul, lab in [(p_glm,"#DAA520","GLM"), (p_lgb,"#2E8B57","LightGBM")]:
        o = np.argsort(p)[::-1]; c = np.cumsum(y_true[o])/y_true.sum()
        ax[1].plot(np.linspace(0,1,len(c)), c, color=coul, lw=2, label=lab)
    ax[1].plot([0,1],[0,1],"r:",lw=1, label="Aleatoire")
    ax[1].set_title("Pouvoir de tri (Lorenz)"); ax[1].legend(fontsize=9)

    p99 = np.percentile(y_true, 99); m = y_true <= p99
    ax[2].scatter(p_glm[m], p_lgb[m], s=7, alpha=.25, color="slateblue")
    lim = [0, max(p_glm[m].max(), p_lgb[m].max())]
    ax[2].plot(lim, lim, "r--", lw=1.5)
    ax[2].set_xlabel("Prediction GLM"); ax[2].set_ylabel("Prediction LightGBM")
    ax[2].set_title("Accord entre les deux modeles")

    plt.tight_layout(); sauver("12_ml_vs_glm"); plt.show()

graphique_ml_vs_glm(y_test, pred_glm, pred_test)












def bootstrap_metriques(y_true, preds_dict, n_boot=2000, seed=RANDOM_STATE):
    """IC bootstrap : la MAE est une ESTIMATION, elle a une incertitude."""
    y_true = np.asarray(y_true, float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    res = {}
    for nom, p in preds_dict.items():
        p = np.asarray(p, float)
        maes = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            maes[b] = np.abs(y_true[idx] - p[idx]).mean()
        res[nom] = {"MAE": np.abs(y_true - p).mean(),
                    "ic_bas": np.percentile(maes, 2.5),
                    "ic_haut": np.percentile(maes, 97.5),
                    "ecart_type": maes.std(), "_boot": maes}
    return res

preds = {"GLM Tweedie": pred_glm, "LightGBM": pred_test,
         "Baseline mediane": np.full(len(y_test), y_train.median())}
boot = bootstrap_metriques(y_test, preds)

print("=" * 84); print("  MAE AVEC INTERVALLES DE CONFIANCE A 95 %"); print("=" * 84)
for nom, r in sorted(boot.items(), key=lambda kv: kv[1]["MAE"]):
    print(f"  {nom:<22} MAE = {r['MAE']:>12,.0f}   "
          f"IC95 = [{r['ic_bas']:>12,.0f} ; {r['ic_haut']:>12,.0f}]")

# --- Test apparie : la difference LightGBM vs GLM est-elle significative ? ---
d = np.abs(y_test.values - pred_glm) - np.abs(y_test.values - pred_test)
t_stat, p_val = stats.ttest_1samp(d, 0)
w_stat, p_w   = stats.wilcoxon(d)
print(f"\n  TEST APPARIE  LightGBM contre GLM")
print(f"    Gain moyen de MAE : {d.mean():>12,.0f} EUR par observation")
print(f"    t-test  : p = {p_val:.3e}")
print(f"    Wilcoxon: p = {p_w:.3e}   (robuste, sans hypothese de normalite)")
print(f"    -> {'Difference SIGNIFICATIVE' if p_w < 0.05 else 'Difference NON significative'}")

def graphique_bootstrap(boot):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    noms = sorted(boot, key=lambda k: boot[k]["MAE"])
    coul = ["#2E8B57","#DAA520","#B22222"][:len(noms)]
    for i, (nom, c) in enumerate(zip(noms, coul)):
        ax[0].hist(boot[nom]["_boot"], bins=60, alpha=.55, color=c, label=nom, density=True)
    ax[0].set_xlabel("MAE bootstrap"); ax[0].set_ylabel("Densite")
    ax[0].set_title("Distribution bootstrap de la MAE"); ax[0].legend(fontsize=9)

    y = np.arange(len(noms))
    for i, (nom, c) in enumerate(zip(noms, coul)):
        r = boot[nom]
        ax[1].plot([r["ic_bas"], r["ic_haut"]], [i, i], color=c, lw=3.5)
        ax[1].scatter([r["MAE"]], [i], s=140, color=c, zorder=5, edgecolor="white", lw=1.5)
    ax[1].set_yticks(y); ax[1].set_yticklabels(noms, fontsize=10)
    ax[1].set_xlabel("MAE (IC 95 %)")
    ax[1].set_title("Les IC se chevauchent-ils ? Si non, ecart significatif")
    plt.tight_layout(); sauver("13_bootstrap"); plt.show()

graphique_bootstrap(boot)








def stabilite_seeds(X_train, y_train, X_valid, y_valid, X_test, y_test,
                    params, best_it, features, n_seeds=8):
    """Le resultat depend-il de la chance ? On rejoue avec plusieurs graines."""
    maes, bilans, imps = [], [], []
    for s in range(n_seeds):
        m = lgb.LGBMRegressor(**{**params, "n_estimators": best_it, "random_state": s})
        m.fit(X_train, y_train)
        p = np.clip(m.predict(X_test), 0, None)
        maes.append(mean_absolute_error(y_test, p))
        bilans.append(p.sum() / y_test.sum())
        imps.append(pd.Series(m.booster_.feature_importance("gain"), index=features))
    maes, bilans = np.array(maes), np.array(bilans)
    imp_df = pd.concat(imps, axis=1)

    print("=" * 78); print(f"  STABILITE SUR {n_seeds} GRAINES"); print("=" * 78)
    print(f"  MAE   : moyenne {maes.mean():,.0f}  ecart-type {maes.std():,.0f}  "
          f"CV {maes.std()/maes.mean():.2%}")
    print(f"  Bilan : moyenne {bilans.mean():.4f}  ecart-type {bilans.std():.4f}")
    print(f"  -> {'INSTABLE (CV > 5%)' if maes.std()/maes.mean() > 0.05 else 'Stable'}")
    return maes, bilans, imp_df


def validation_temporelle_multiple(df, features, target, params, n_folds=4,
                                   n_test=1, n_valid=1):
    """Plusieurs fenetres de test successives : la performance tient-elle dans le temps ?
    Validation de robustesse -- PAS une methode de production."""
    toutes = sorted(df["time_idx"].unique())
    res = []
    for k in range(n_folds):
        fin_test   = len(toutes) - k
        p_te = toutes[fin_test - n_test:fin_test]
        p_va = toutes[fin_test - n_test - n_valid:fin_test - n_test]
        p_tr = toutes[:fin_test - n_test - n_valid]
        if len(p_tr) < 8:
            break
        mtr = df["time_idx"].isin(p_tr); mva = df["time_idx"].isin(p_va)
        mte = df["time_idx"].isin(p_te)

        m = lgb.LGBMRegressor(**params)
        m.fit(df.loc[mtr, features], df.loc[mtr, target].astype(float),
              eval_set=[(df.loc[mva, features], df.loc[mva, target].astype(float))],
              eval_metric="mae",
              callbacks=[lgb.early_stopping(150, verbose=False)])
        bi = m.best_iteration_ or params["n_estimators"]
        yt = df.loc[mte, target].astype(float).values
        p  = np.clip(m.predict(df.loc[mte, features], num_iteration=bi), 0, None)
        res.append({"fold": k+1, "test": f"{p_te[0]}-{p_te[-1]}",
                    "n_train": int(mtr.sum()), "n_test": int(mte.sum()),
                    "arbres": int(bi), "MAE": mean_absolute_error(yt, p),
                    "wMAPE_%": np.abs(yt-p).sum()/yt.sum()*100,
                    "bilan": p.sum()/yt.sum()})
    t = pd.DataFrame(res)
    print("\n" + "=" * 84); print("  VALIDATION TEMPORELLE MULTIPLE"); print("=" * 84)
    print(t.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print(f"\n  MAE : moyenne {t['MAE'].mean():,.0f}  ecart-type {t['MAE'].std():,.0f}  "
          f"CV {t['MAE'].std()/t['MAE'].mean():.1%}")
    print(f"  -> {'Performance INSTABLE dans le temps' if t['MAE'].std()/t['MAE'].mean() > 0.25 else 'Performance stable dans le temps'}")
    return t

maes_s, bilans_s, imp_seeds = stabilite_seeds(
    X_train, y_train, X_valid, y_valid, X_test, y_test, PARAMS, best_it, FEATURES)
tab_temp = validation_temporelle_multiple(df, FEATURES, TARGET, PARAMS)


def graphique_robustesse(maes_s, bilans_s, imp_seeds, tab_temp, mae_ref):
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    ax[0,0].hist(maes_s, bins=12, color="steelblue", edgecolor="white")
    ax[0,0].axvline(mae_ref, color="red", ls="--", lw=2, label=f"Seed 42 : {mae_ref:,.0f}")
    ax[0,0].set_xlabel("MAE"); ax[0,0].set_title("1. Dispersion de la MAE selon la graine")
    ax[0,0].legend(fontsize=8)

    ax[0,1].hist(bilans_s, bins=12, color="seagreen", edgecolor="white")
    ax[0,1].axvline(1.0, color="red", ls="--", lw=2, label="Equilibre parfait")
    ax[0,1].set_xlabel("Bilan predit/reel"); ax[0,1].set_title("2. Stabilite du bilan technique")
    ax[0,1].legend(fontsize=8)

    # Stabilite du CLASSEMENT des features
    rangs = imp_seeds.rank(ascending=False)
    top = imp_seeds.mean(axis=1).nlargest(12).index
    ax[1,0].boxplot([rangs.loc[f].values for f in top], labels=top, vert=False,
                    patch_artist=True)
    ax[1,0].set_xlabel("Rang d'importance (1 = plus important)")
    ax[1,0].set_title("3. Stabilite du classement des features")
    ax[1,0].tick_params(labelsize=8)

    ax[1,1].errorbar(tab_temp["fold"], tab_temp["MAE"], fmt="o-", color="navy",
                     lw=2, markersize=9, label="MAE par fold")
    ax[1,1].axhline(tab_temp["MAE"].mean(), color="red", ls="--",
                    label=f"Moyenne {tab_temp['MAE'].mean():,.0f}")
    ax[1,1].fill_between(tab_temp["fold"],
                         tab_temp["MAE"].mean() - tab_temp["MAE"].std(),
                         tab_temp["MAE"].mean() + tab_temp["MAE"].std(),
                         color="red", alpha=0.12, label="+/- 1 ecart-type")
    ax[1,1].set_xlabel("Fold temporel"); ax[1,1].set_ylabel("MAE")
    ax[1,1].set_title("4. Performance dans le temps"); ax[1,1].legend(fontsize=8)

    plt.tight_layout(); sauver("14_robustesse"); plt.show()

graphique_robustesse(maes_s, bilans_s, imp_seeds, tab_temp, mae_te)













from sklearn.isotonic import IsotonicRegression

def recalibrer(y_valid, pred_valid, pred_test, y_test):
    """Trois recalibrations apprises sur la VALIDATION, appliquees au TEST."""
    y_valid, pred_valid = np.asarray(y_valid, float), np.asarray(pred_valid, float)
    resultats = {}

    # (1) Facteur multiplicatif global : le plus simple, tres utilise en actuariat
    facteur = y_valid.sum() / pred_valid.sum()
    resultats["Facteur global"] = (np.clip(pred_test * facteur, 0, None),
                                   f"facteur = {facteur:.4f}")

    # (2) Facteur par decile de prediction : corrige les biais locaux
    dec_v = pd.qcut(pred_valid, 10, labels=False, duplicates="drop")
    fac_d = (pd.DataFrame({"d": dec_v, "y": y_valid, "p": pred_valid})
             .groupby("d").apply(lambda g: g["y"].sum()/g["p"].sum()))
    bornes = np.quantile(pred_valid, np.linspace(0, 1, len(fac_d)+1)); bornes[-1] = np.inf
    dec_t = np.clip(np.searchsorted(bornes[1:-1], pred_test), 0, len(fac_d)-1)
    resultats["Facteur par decile"] = (np.clip(pred_test * fac_d.values[dec_t], 0, None),
                                       f"{len(fac_d)} facteurs")

    # (3) Regression isotone : monotone, non parametrique
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(pred_valid, y_valid)
    resultats["Isotone"] = (np.clip(iso.predict(pred_test), 0, None), "monotone")

    print("=" * 88); print("  RECALIBRATION POST-HOC (apprise sur la validation)"); print("=" * 88)
    lignes = [evaluer(y_test, pred_test, "Sans recalibration")]
    for nom, (p, info) in resultats.items():
        r = evaluer(y_test, p, nom); r["detail"] = info
        lignes.append(r)
    t = pd.DataFrame(lignes)
    print(t.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print(f"\n  Objectif : bilan le plus proche de 1.000 SANS degrader la MAE")
    return resultats, t

recals, tab_recal = recalibrer(y_valid, pred_valid, pred_test, y_test)

def graphique_recalibration(y_test, pred_test, recals):
    y = np.asarray(y_test, float)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))

    noms = ["Sans recalibration"] + list(recals)
    tous = [pred_test] + [p for p, _ in recals.values()]
    bil  = [p.sum()/y.sum() for p in tous]
    mae  = [mean_absolute_error(y, p) for p in tous]

    coul = ["#B22222" if abs(b-1) > 0.02 else "#2E8B57" for b in bil]
    ax[0].bar(range(len(noms)), bil, color=coul)
    ax[0].axhline(1, color="black", ls="--", lw=2, label="Equilibre parfait")
    ax[0].axhspan(0.98, 1.02, color="green", alpha=0.10, label="Tolerance +/- 2 %")
    ax[0].set_xticks(range(len(noms)))
    ax[0].set_xticklabels(noms, rotation=25, ha="right", fontsize=9)
    ax[0].set_ylabel("Bilan predit/reel"); ax[0].set_title("Bilan technique")
    ax[0].legend(fontsize=8)
    for i, b in enumerate(bil):
        ax[0].text(i, b, f"{b:.4f}", ha="center", va="bottom", fontsize=8)

    ax[1].bar(range(len(noms)), mae, color="steelblue")
    ax[1].axhline(mae[0], color="red", ls="--", lw=1.5, label="Reference sans recalibration")
    ax[1].set_xticks(range(len(noms)))
    ax[1].set_xticklabels(noms, rotation=25, ha="right", fontsize=9)
    ax[1].set_ylabel("MAE"); ax[1].set_title("Cout en precision")
    ax[1].legend(fontsize=8)

    plt.tight_layout(); sauver("15_recalibration"); plt.show()

graphique_recalibration(y_test, pred_test, recals)















from sklearn.inspection import PartialDependenceDisplay

def graphiques_pdp(modele, X_test, importances, n_feat=6):
    """PDP : comment la prediction reagit-elle a CHAQUE variable ?"""
    top = [f for f in importances.index[:n_feat*2]
           if f in NUMERIQUES and X_test[f].nunique() > 10][:n_feat]
    if not top:
        print("Pas assez de features numeriques pour les PDP."); return
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    PartialDependenceDisplay.from_estimator(
        modele, X_test, top, ax=ax.ravel()[:len(top)],
        grid_resolution=40, percentiles=(0.02, 0.98))
    for a in ax.ravel()[len(top):]:
        a.set_visible(False)
    plt.suptitle("Dependances partielles -- effet marginal de chaque variable",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); sauver("16_pdp"); plt.show()
    return top

top_pdp = graphiques_pdp(modele, X_test, importances)


def modele_monotone(X_train, y_train, X_valid, y_valid, X_test, y_test,
                    features, params, croissantes=()):
    """Contraintes de monotonie : exigence courante de gouvernance actuarielle.
    Le modele ne peut plus produire d'effet contre-intuitif."""
    if not croissantes:
        print("Aucune contrainte specifiee -- renseigne 'croissantes'."); return None, None
    mono = [1 if f in croissantes else 0 for f in features]
    print(f"  Contraintes croissantes sur : {[f for f in features if f in croissantes]}")

    m = lgb.LGBMRegressor(**{**params, "monotone_constraints": mono,
                             "monotone_constraints_method": "advanced"})
    m.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="mae",
          callbacks=[lgb.early_stopping(150, verbose=False)])
    bi = m.best_iteration_ or params["n_estimators"]
    p = np.clip(m.predict(X_test, num_iteration=bi), 0, None)

    t = pd.DataFrame([evaluer(y_test, pred_test, "Sans contrainte"),
                      evaluer(y_test, p, "Avec monotonie")])
    print("\n  Cout des contraintes de monotonie :")
    print(t.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    return m, p

# 🔴 A ADAPTER : liste les variables dont l'effet doit etre croissant par construction
VARS_CROISSANTES = [f for f in ["RBNS_bop", "Claims_incurred", "exposition"]
                    if f in FEATURES]
m_mono, pred_mono = modele_monotone(X_train, y_train, X_valid, y_valid, X_test, y_test,
                                    FEATURES, PARAMS, VARS_CROISSANTES)













def analyse_erreur(df_test_meta, y_true, y_pred, id_cols, top=20):
    """Quels segments portent l'essentiel de l'erreur totale ? Une erreur de 2%
    sur un gros segment coute plus cher qu'une erreur de 40% sur un petit."""
    d = df_test_meta.copy()
    d["y_obs"], d["y_pred"] = np.asarray(y_true, float), np.asarray(y_pred, float)
    d["err_abs"] = (d["y_obs"] - d["y_pred"]).abs()
    d["err_signee"] = d["y_pred"] - d["y_obs"]
    total = d["err_abs"].sum()

    print("=" * 90); print("  CONTRIBUTION A L'ERREUR TOTALE"); print("=" * 90)
    print(f"  Erreur absolue totale : {total:,.0f} EUR\n")

    tables = {}
    for col in [c for c in id_cols if c in d.columns]:
        t = (d.groupby(col, observed=True)
             .agg(n=("err_abs","size"), err_totale=("err_abs","sum"),
                  err_moy=("err_abs","mean"), biais=("err_signee","sum"),
                  vol_reel=("y_obs","sum")).reset_index())
        t["part_erreur_%"] = t["err_totale"] / total * 100
        t["err_rel_%"] = t["err_totale"] / t["vol_reel"] * 100
        t = t.sort_values("part_erreur_%", ascending=False)
        tables[col] = t
        print(f"--- {col} : top 6 contributeurs a l'erreur ---")
        print(t.head(6)[[col,"n","part_erreur_%","err_rel_%","biais"]]
              .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
        print()

    print(f"--- Top {top} pires observations individuelles ---")
    cols_aff = [c for c in id_cols if c in d.columns] + ["y_obs","y_pred","err_abs"]
    print(d.nlargest(top, "err_abs")[cols_aff]
          .to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
    return d, tables


def graphique_analyse_erreur(tables, d, col_principal="Lob"):
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    t = tables[col_principal].head(12)

    ax[0,0].barh(range(len(t)), t["part_erreur_%"], color="indianred")
    ax[0,0].set_yticks(range(len(t)))
    ax[0,0].set_yticklabels(t[col_principal].astype(str), fontsize=8)
    ax[0,0].set_xlabel("% de l'erreur totale")
    ax[0,0].set_title(f"1. Contribution a l'erreur totale ({col_principal})")

    cum = t["part_erreur_%"].cumsum()
    ax[0,1].plot(range(1, len(cum)+1), cum, marker="o", color="darkgreen", lw=2)
    ax[0,1].axhline(80, color="red", ls="--", label="80 % de l'erreur")
    ax[0,1].set_xlabel("Nombre de segments"); ax[0,1].set_ylabel("% cumule")
    ax[0,1].set_title("2. Concentration -- combien de segments a revoir ?")
    ax[0,1].legend(fontsize=8)

    coul = ["#B22222" if b > 0 else "#4169E1" for b in t["biais"]]
    ax[1,0].barh(range(len(t)), t["biais"], color=coul)
    ax[1,0].axvline(0, color="black", lw=1)
    ax[1,0].set_yticks(range(len(t)))
    ax[1,0].set_yticklabels(t[col_principal].astype(str), fontsize=8)
    ax[1,0].set_xlabel("Biais total (predit - reel), EUR")
    ax[1,0].set_title("3. Sens du biais -- rouge = sur-provisionne")

    tt = tables[col_principal]
    tailles = 60 + 500*(tt["n"] - tt["n"].min())/max(tt["n"].max()-tt["n"].min(), 1)
    ax[1,1].scatter(tt["err_rel_%"], tt["part_erreur_%"], s=tailles,
                    c="slateblue", alpha=.65, edgecolor="white", lw=1.4)
    for r in tt.head(8).itertuples():
        ax[1,1].annotate(str(getattr(r, col_principal))[:14],
                         (r.err_rel_, r.part_erreur_), fontsize=7.5)
    ax[1,1].set_xlabel("Erreur relative du segment (%)")
    ax[1,1].set_ylabel("% de l'erreur totale portee")
    ax[1,1].set_title("4. Priorisation : haut-droite = a traiter en premier")

    plt.tight_layout(); sauver("17_analyse_erreur"); plt.show()

d_err, tables_err = analyse_erreur(df.loc[m_test], y_test, pred_test, CANDIDATS_ID)
graphique_analyse_erreur(tables_err, d_err, "Lob")
