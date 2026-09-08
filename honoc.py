Enregistrer , sauvegarder

from pathlib import Path
Path("Dossier_concatener").mkdir(exist_ok=True)

results_v1.to_pickle("Dossier_concatener/results_v1.pkl")
anomalies_prio.to_pickle("Dossier_concatener/anomalies_prio.pkl")
print("Enregistré.")


Apeler

import pandas as pd

results_v1     = pd.read_pickle("Dossier_concatener/results_v1.pkl")
anomalies_prio = pd.read_pickle("Dossier_concatener/anomalies_prio.pkl")
print(results_v1.shape, anomalies_prio.shape)




Shap

# -*- coding: utf-8 -*-
# =============================================================================
#  MISSION 2 - TIME EVOLUTION OF THE FIRST IMPORTANT VARIABLES
#
#  Enonce :
#    - SHAP local (contribution / decomposition) pour identifier les 5 variables
#      les plus determinantes dans la prediction de la target ;
#    - visualiser leur evolution sur les 10 derniers trimestres et sur la
#      periode a valider, comme pour la target.
#
#  A EXECUTER JUSTE APRES la cellule de la mission 1 : reutilise UNITE, h, per,
#  ctx et nom, deja construits pour le sous-portefeuille choisi.
#  Prerequis : pip install shap
# =============================================================================

import joblib
import numpy as np
import pandas as pd
import shap
from plotly.subplots import make_subplots

# ┌──────────────────────────── PARAMETRES ────────────────────────────┐
N_VARS  = 5                                    # enonce : 5 variables
CHEMIN  = "artefacts_modele/modele_final.joblib"
#  MODELE et X_MODEL sont repris de la session s'ils y existent deja. Pour les
#  fixer, definissez-les dans une cellule AVANT celle-ci (MODELE = mon_modele),
#  plutot que d'editer ces deux lignes.
MODELE  = globals().get("MODELE")    # None -> charge l'artefact CHEMIN
X_MODEL = globals().get("X_MODEL")   # features exactes de la prediction, indexees
                                     # comme df ; None -> reconstruites depuis h
# └─────────────────────────────────────────────────────────────────────┘

for _v in ("h", "per", "nom"):
    if _v not in globals():
        raise NameError(f"`{_v}` absent : executez d'abord la cellule mission 1.")

if "fmt" not in globals():
    def fmt(v):
        return f"{v:,.0f}".replace(",", " ")

ENCRE  = globals().get("ENCRE", "#141B34")
GRIS   = globals().get("GRIS", "#8A93A5")
GRILLE = globals().get("GRILLE", "#EDF1F7")
ACCENT = globals().get("ACCENT", "#FF5A5F")
DOUX   = globals().get("DOUX", ["#6C8EBF", "#82B366", "#C08552",
                                "#9673A6", "#5F9EA0"])


# %% -------------------------------------------------------------------------
#  1. SHAP LOCAL : decomposition de LA prediction du sous-portefeuille
# -----------------------------------------------------------------------------
if MODELE is None:
    _art = joblib.load(CHEMIN)
    MODELE = (next(v for v in _art.values() if hasattr(v, "predict"))
              if isinstance(_art, dict) else _art)
mdl = MODELE.named_steps["model"] if hasattr(MODELE, "named_steps") else MODELE
FEATS = (list(mdl.feature_name_) if hasattr(mdl, "feature_name_")
         else list(mdl.feature_names_in_))

ligne = h.iloc[[-1]]                       # dernier trimestre = periode a valider
if X_MODEL is not None:
    x = X_MODEL.loc[ligne.index, FEATS]
else:
    absentes = [f for f in FEATS if f not in ligne.columns]
    if absentes:
        raise KeyError(f"{len(absentes)} variable(s) du modele absente(s) de df "
                       f"(ex. {absentes[:3]}). Renseignez X_MODEL.")
    x = ligne[FEATS].copy()
    for c in x.columns:                    # LightGBM natif attend des category
        if x[c].dtype == object:
            x[c] = x[c].astype("category")

explainer = shap.TreeExplainer(mdl)
sv = np.asarray(explainer.shap_values(x)).ravel()
base = float(np.ravel(explainer.expected_value)[0])
contrib = pd.Series(sv, index=FEATS)

#  Controle de decomposition : base + somme des contributions doit redonner la
#  prediction du modele. C'est ce qui rend l'explication opposable en soutenance.
pred_mdl = float(mdl.predict(x)[0])
ecart = abs(base + contrib.sum() - pred_mdl)

print(f"{nom}  |  periode expliquee {per[-1]}")
print(f"Valeur de base (moyenne du modele) : {fmt(base)}")
print(f"Somme des contributions SHAP       : {fmt(contrib.sum())}")
print(f"Prediction reconstituee            : {fmt(base + contrib.sum())}")
print(f"Prediction du modele               : {fmt(pred_mdl)}")
print(f"Ecart de reconstitution            : {ecart:.6g}"
      f"   {'(decomposition exacte)' if ecart < 1e-6 * max(abs(pred_mdl), 1) else '(A VERIFIER)'}")

top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(N_VARS)
part = 100 * top.abs() / contrib.abs().sum()

print(f"\nLes {N_VARS} variables les plus determinantes pour cette prediction :")
recap = pd.DataFrame({
    "valeur": [ligne[v].iloc[0] if v in ligne.columns else np.nan for v in top.index],
    "contribution": top.values,
    "sens": np.where(top.values >= 0, "pousse a la hausse", "pousse a la baisse"),
    "part_abs_%": part.round(1).values}, index=top.index)
print(recap.to_string(float_format=lambda v: f"{v:,.2f}".replace(",", " ")))


# %% -------------------------------------------------------------------------
#  2. EVOLUTION DE CES 5 VARIABLES, meme principe que la target
# -----------------------------------------------------------------------------
tracables = [v for v in top.index
             if v in h.columns and pd.api.types.is_numeric_dtype(h[v])]
ecartees = [v for v in top.index if v not in tracables]
if ecartees:
    print(f"\nNon tracables (categorielles ou absentes de l'historique) : {ecartees}")
if not tracables:
    raise ValueError("Aucune des variables retenues n'est numerique et suivie "
                     "dans l'historique : rien a tracer.")

k_valid = per.index(ctx["per"]) if globals().get("ctx") and ctx["per"] in per else len(per) - 1

fig2 = make_subplots(rows=len(tracables), cols=1, shared_xaxes=True,
                     vertical_spacing=.05)

for i, v in enumerate(tracables, start=1):
    c = DOUX[(i - 1) % len(DOUX)]
    vv = h[v].values.astype(float)
    fig2.add_vrect(x0=k_valid - .5, x1=k_valid + .5,
                   fillcolor="rgba(99,110,250,0.055)", line_width=0,
                   layer="below", row=i, col=1)
    fig2.add_scatter(x=per, y=vv, mode="lines+markers",
                     line=dict(color=c, width=2.4, shape="spline", smoothing=.55),
                     marker=dict(size=7, color="white",
                                 line=dict(color=c, width=1.9)),
                     name=v, showlegend=False,
                     hovertemplate=f"<b>%{{x}}</b><br>{v} : <b>%{{y:,.4g}}</b>"
                                   "<extra></extra>", row=i, col=1)
    fig2.add_scatter(x=[per[k_valid]], y=[vv[k_valid]], mode="markers",
                     marker=dict(size=13, color=c,
                                 line=dict(color="white", width=2.4)),
                     showlegend=False, hoverinfo="skip", row=i, col=1)
    signe = "+" if top[v] >= 0 else "-"
    fig2.add_annotation(xref="paper", x=1.008, y=vv[k_valid], xanchor="left",
                        text=f"<b>{v}</b><br>"
                             f"<span style='font-size:10px;color:{GRIS}'>"
                             f"SHAP {signe}{fmt(abs(top[v]))} "
                             f"({part[v]:.0f} %)</span>",
                        showarrow=False, align="left",
                        font=dict(size=11, color=c), row=i, col=1)

fig2.update_xaxes(showgrid=False, showticklabels=False, linecolor=GRILLE)
fig2.update_xaxes(title_text="<b>Trimestre</b>", title_font=dict(size=12),
                  showticklabels=True, tickfont=dict(size=11),
                  row=len(tracables), col=1)
fig2.update_yaxes(gridcolor=GRILLE, zeroline=False, tickformat=",.4g",
                  tickfont=dict(size=10))

fig2.update_layout(
    title=dict(text=f"<b style='font-size:19px;color:{ENCRE}'>{nom}</b>"
                    f"<br><span style='font-size:12px;color:{GRIS}'>"
                    f"{len(tracables)} variables les plus determinantes "
                    f"(SHAP local) · {len(h)} trimestres · {per[0]} → {per[-1]} · "
                    f"periode a valider <b style='color:{ACCENT}'>{per[k_valid]}"
                    f"</b> surlignee</span>", x=.015, xanchor="left", y=.97),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="white", bordercolor=GRILLE, align="left",
                    font=dict(size=12.5, family="Inter, system-ui, sans-serif",
                              color=ENCRE)),
    template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color=GRIS),
    height=150 * len(tracables) + 150,
    margin=dict(l=80, r=150, t=125, b=60), separators=", ")

fig2.show()













Aujourdhui

# -*- coding: utf-8 -*-
# =============================================================================
#  LE POIDS DU PORTEFEUILLE DANS LE SCORE DE PRIORISATION
#
#  Question posee : le score suppose que Earned_Premium mesure le poids du
#  portefeuille. Deux reserves.
#    1. Certains portefeuilles ne collectent plus de prime (GWP = 0). Est-ce
#       aussi le cas de Earned_Premium ?
#    2. Une commission importante versee au partenaire reduit la prime nette.
#       Un poids brut surpondere alors des portefeuilles peu rentables, et en
#       masque d'autres qu'on ne regarde pas assez.
#
#  POURQUOI C'EST STRUCTURANT, ET PAS UN DETAIL DE CALIBRAGE
#  ---------------------------------------------------------
#  Le score s'ecrit    score = A_ecart_borne * B_erreur_modele * poids
#  Le poids entre en FACTEUR. Un poids nul ne donne donc pas un score faible,
#  il donne un score exactement nul : le sous-portefeuille se retrouve dernier
#  du classement quelle que soit la gravite de son anomalie, et ne remonte
#  jamais a la validation. C'est un angle mort, pas une sous-ponderation.
#
#  Trois blocs, a executer dans l'ordre. Prerequis : df (ou la base de
#  travail) et, pour le bloc 3, anomalies (ou anomalies_prio).
#
#  Contrainte projet respectee : aucun logarithme. Tout passe par des rangs,
#  des quantiles et des rapports.
# =============================================================================

import re

import numpy as np
import pandas as pd

# ┌──────────────────────────── PARAMETRES ────────────────────────────┐
BASE          = None          # None -> df de la session
COL_POIDS     = "Earned_Premium"
COL_GWP       = "GWP"
TABLE_ANOMALIES = None        # None -> anomalies_prio, sinon anomalies
TOP_N         = 20            # taille du haut de classement compare au bloc 3
# └─────────────────────────────────────────────────────────────────────┘

#  Deux familles distinctes, et la distinction compte. Une colonne de
#  commission n'est pas un poids de portefeuille, c'est le moyen d'en
#  construire un si aucune colonne nette n'existe deja dans la base.
MOTIFS_POIDS = [
    r"prime", r"premium", r"risque", r"risk", r"acquis", r"earned",
    r"loading", r"net", r"brut", r"gross",
]
MOTIFS_COMMISSION = [r"commission", r"\bcom\b", r"charg", r"frais", r"\bfee"]

#  Une prime nette de commission reste inferieure ou egale a la prime brute,
#  sans changer d'ordre de grandeur. En dehors de cette plage, la colonne
#  mesure autre chose (la commission elle-meme, un taux, un cumul).
RATIO_MIN, RATIO_MAX = .20, 1.05


def _fmt(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:,.0f}".replace(",", " ")


def _chercher(nom):
    """Retrouve une variable de la session, que ce fichier soit colle dans le
    notebook ou importe. Un simple globals() ne verrait que ce module."""
    if nom in globals():
        return globals()[nom]
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None and nom in ip.user_ns:
            return ip.user_ns[nom]
    except Exception:
        pass
    import __main__
    return getattr(__main__, nom, None)


def _base():
    b = BASE if BASE is not None else _chercher("df")
    if b is None:
        raise NameError("Aucune base : renseignez BASE ou definissez df.")
    return b


# %% ==========================================================================
#  BLOC 1 - DISTRIBUTION DU POIDS, ET SURTOUT LES ZEROS
# =============================================================================
def profil_colonne(d, col):
    """Ce qu'il faut savoir d'une colonne avant de s'en servir comme poids."""
    if col not in d.columns:
        return None
    v = pd.to_numeric(d[col], errors="coerce")
    n = len(v)
    q = v.dropna().quantile([0, .01, .25, .5, .75, .99, 1]).to_dict()
    return dict(colonne=col, n=n,
                nan=int(v.isna().sum()), pct_nan=100 * v.isna().mean(),
                zeros=int((v == 0).sum()), pct_zeros=100 * (v == 0).mean(),
                negatifs=int((v < 0).sum()), pct_negatifs=100 * (v < 0).mean(),
                min=q.get(0), p01=q.get(.01), p25=q.get(.25), median=q.get(.5),
                p75=q.get(.75), p99=q.get(.99), max=q.get(1),
                somme=float(v.sum(skipna=True)))


def afficher_profil(p):
    if p is None:
        print("  (colonne absente)")
        return
    print(f"  {p['colonne']}  —  {p['n']:,} lignes".replace(",", " "))
    print(f"    manquants : {p['nan']:>8,}  ({p['pct_nan']:.2f} %)"
          .replace(",", " "))
    print(f"    zeros     : {p['zeros']:>8,}  ({p['pct_zeros']:.2f} %)"
          .replace(",", " "))
    print(f"    negatifs  : {p['negatifs']:>8,}  ({p['pct_negatifs']:.2f} %)"
          .replace(",", " "))
    print(f"    quantiles : min {_fmt(p['min'])} | p25 {_fmt(p['p25'])} | "
          f"med {_fmt(p['median'])} | p75 {_fmt(p['p75'])} | "
          f"p99 {_fmt(p['p99'])} | max {_fmt(p['max'])}")


def croiser_zeros(d, col_a, col_b):
    """Les zeros de col_a et col_b se recouvrent-ils ? C'est la question 1."""
    if col_a not in d.columns or col_b not in d.columns:
        print(f"  Croisement impossible : {col_a} ou {col_b} absent.")
        return None
    a = pd.to_numeric(d[col_a], errors="coerce").fillna(0) == 0
    b = pd.to_numeric(d[col_b], errors="coerce").fillna(0) == 0
    t = pd.crosstab(a.rename(f"{col_a} = 0"), b.rename(f"{col_b} = 0"))
    print(f"\n  Croisement des zeros ({col_a} x {col_b}) :")
    print("   " + t.to_string().replace("\n", "\n   "))
    n_a_seul = int((a & ~b).sum())
    n_b_seul = int((~a & b).sum())
    n_deux = int((a & b).sum())
    print(f"\n    {col_b} = 0 mais {col_a} > 0 : {n_b_seul:,}".replace(",", " ")
          + "   (collecte arretee, prime encore acquise)")
    print(f"    {col_a} = 0 mais {col_b} > 0 : {n_a_seul:,}".replace(",", " ")
          + "   (cas inverse, a expliquer)")
    print(f"    les deux a 0                : {n_deux:,}".replace(",", " ")
          + "   (portefeuille reellement eteint)")
    return t


print("=" * 78)
print("BLOC 1 - LE POIDS ACTUEL EST-IL UTILISABLE ?")
print("=" * 78)
_d = _base()
_p_poids = profil_colonne(_d, COL_POIDS)
_p_gwp = profil_colonne(_d, COL_GWP)
afficher_profil(_p_poids)
print()
afficher_profil(_p_gwp)
croiser_zeros(_d, COL_POIDS, COL_GWP)

if _p_poids and _p_poids["zeros"]:
    print(f"\n  REPONSE A LA QUESTION 1 : oui, {COL_POIDS} contient "
          f"{_p_poids['zeros']:,} zeros ({_p_poids['pct_zeros']:.2f} %)."
          .replace(",", " "))
    print("  Comme le poids est un facteur du score, ces lignes ont un score")
    print("  exactement nul et ne peuvent jamais remonter dans le classement.")
elif _p_poids:
    print(f"\n  REPONSE A LA QUESTION 1 : {COL_POIDS} ne contient aucun zero.")
    print("  Le risque d'annulation du score ne se materialise pas ici, mais")
    print("  la reserve reste valable si la base evolue.")


# %% ==========================================================================
#  BLOC 2 - CHERCHER UN INDICATEUR DE PRIME NETTE DE COMMISSION
# =============================================================================
def chercher_candidats(d=None, motifs=None, ref=None, n_min_remplissage=50.,
                       colonnes_sup=None):
    """Colonnes numeriques dont le nom evoque une prime, et leur aptitude.

    Le rapport median a la reference se lit comme un taux de retention : 0,72
    signifie qu'il reste 72 % de la prime de reference apres commission et
    chargements. Un rapport proche de 1 signale une colonne quasi identique a
    la reference, donc sans apport.
    """
    d = d if d is not None else _base()
    motifs = motifs or MOTIFS_POIDS
    ref = ref or COL_POIDS
    rx = re.compile("|".join(motifs), re.I)
    #  Les colonnes construites exprès (prime nette reconstituee) entrent sans
    #  passer par les motifs : leur nom contient "commission" et le filtre
    #  ci-dessous les rejetterait alors qu'elles sont precisement l'objectif.
    sup = set()
    if colonnes_sup is not None:
        colonnes_sup = pd.DataFrame(colonnes_sup)
        sup = set(colonnes_sup.columns)
        d = pd.concat([d, colonnes_sup], axis=1)

    ref_v = (pd.to_numeric(d[ref], errors="coerce")
             if ref in d.columns else None)
    lignes = []
    rx_com = re.compile("|".join(MOTIFS_COMMISSION), re.I)
    for c in d.columns:
        if not pd.api.types.is_numeric_dtype(d[c]):
            continue
        if c not in sup:
            if not rx.search(str(c)):
                continue
            if rx_com.search(str(c)) and c != ref:
                continue                  # une commission n'est pas un poids
        p = profil_colonne(d, c)
        v = pd.to_numeric(d[c], errors="coerce")
        rho = ratio = np.nan
        if ref_v is not None and c != ref:
            ok = v.notna() & ref_v.notna()
            if ok.sum() > 30:
                rho = float(v[ok].corr(ref_v[ok], method="spearman"))
                den = ref_v[ok].replace(0, np.nan)
                ratio = float((v[ok] / den).median())
        lignes.append({
            "colonne": c, "pct_remplissage": 100 - p["pct_nan"],
            "pct_zeros": p["pct_zeros"], "pct_negatifs": p["pct_negatifs"],
            "median": p["median"], "somme": p["somme"],
            "spearman_vs_ref": rho, "ratio_median_vs_ref": ratio})

    t = pd.DataFrame(lignes)
    if not len(t):
        return t

    #  Aptitude comme poids : bien rempli, pas PLUS de zeros que la reference
    #  (les colonnes derivees en heritent mecaniquement, ce n'est pas un
    #  defaut), ordonne comme la reference donc mesurant bien un volume, et
    #  d'un ordre de grandeur compatible avec une prime nette.
    zeros_ref = (_p_poids or {}).get("pct_zeros", 100.)
    t["apte"] = (
        (t["colonne"] != ref)
        & (t["pct_remplissage"] >= n_min_remplissage)
        & (t["pct_zeros"] <= zeros_ref + .5)
        & (t["pct_negatifs"] < 5)
        & (t["spearman_vs_ref"].fillna(0).abs() >= .5)
        & t["ratio_median_vs_ref"].between(RATIO_MIN, RATIO_MAX))
    t["ecart_a_la_ref"] = (1 - t["ratio_median_vs_ref"]).abs()
    #  Parmi les aptes, on met en tete celle qui s'ecarte le plus de la
    #  reference : c'est celle qui changerait reellement le classement.
    return t.sort_values(["apte", "ecart_a_la_ref"],
                         ascending=[False, False]).reset_index(drop=True)


def chercher_commissions(d=None):
    """Colonnes de commission ou de chargement, pour CONSTRUIRE une prime nette."""
    d = d if d is not None else _base()
    rx = re.compile("|".join(MOTIFS_COMMISSION), re.I)
    return [c for c in d.columns
            if rx.search(str(c)) and pd.api.types.is_numeric_dtype(d[c])]


def construire_prime_nette(d=None, ref=None, col_commission=None, nom=None):
    """Prime nette reconstruite, quand aucune colonne nette n'existe deja."""
    d = d if d is not None else _base()
    ref = ref or COL_POIDS
    brut = pd.to_numeric(d[ref], errors="coerce")
    com = pd.to_numeric(d[col_commission], errors="coerce").fillna(0)
    nette = (brut - com).clip(lower=0)
    nette.name = nom or f"{ref}_net_de_{col_commission}"
    return nette


print()
print("=" * 78)
print("BLOC 2 - CANDIDATS AU POIDS NET DE COMMISSION")
print("=" * 78)
_commissions = chercher_commissions()
_synthetiques = None
if _commissions:
    print(f"  Colonnes de commission ou de chargement reperees : {_commissions}")
    print("  Elles ne sont pas des poids, mais permettent d'en construire un.")
    _synthetiques = pd.concat(
        [construire_prime_nette(col_commission=c) for c in _commissions], axis=1)
    print(f"  Candidats construits : {list(_synthetiques.columns)}\n")

_cand = chercher_candidats(colonnes_sup=_synthetiques)
if not len(_cand):
    print(f"  Aucune colonne dont le nom evoque une prime. Motifs cherches : "
          f"{MOTIFS_POIDS}")
    print("  Elargissez MOTIFS_POIDS, ou listez les colonnes a la main.")
else:
    _aff = _cand[["colonne", "pct_remplissage", "pct_zeros", "median",
                  "spearman_vs_ref", "ratio_median_vs_ref", "apte"]]
    print(_aff.to_string(index=False, float_format=lambda v: f"{v:,.3f}"
                         .replace(",", " ")))
    _retenus = _cand.loc[_cand["apte"], "colonne"].tolist()
    print(f"\n  Candidats retenus : {_retenus or 'aucun'}")
    if _retenus:
        print("  Lecture du rapport median : 0,72 signifie qu'il reste 72 % de")
        print(f"  {COL_POIDS} apres commission et chargements. Un rapport tres")
        print("  proche de 1 signale une colonne redondante avec la reference.")


# %% ==========================================================================
#  BLOC 3 - CE QUE CHANGE LE POIDS SUR LE CLASSEMENT
# =============================================================================
def _severite_intrinseque(a):
    """A * B : la gravite de l'anomalie, poids exclu."""
    d = a.copy()
    if "A_ecart_borne" not in d.columns:
        borne = np.where(d["y_obs"] > d["borne_haute"],
                         d["borne_haute"], d["borne_basse"])
        ecart = np.where(d["y_obs"] > d["borne_haute"],
                         d["y_obs"] - d["borne_haute"],
                         d["borne_basse"] - d["y_obs"])
        d["A_ecart_borne"] = np.abs(ecart) / np.maximum(np.abs(borne), 1e-6)
    if "B_erreur_modele" not in d.columns:
        d["B_erreur_modele"] = (np.abs(d["y_obs"] - d["y_pred"])
                                / np.maximum(np.abs(d["y_pred"]), 1e-6))
    return d["A_ecart_borne"] * d["B_erreur_modele"], d


def anomalies_invisibilisees(a, col_poids=None, n=10):
    """Les anomalies graves que le poids nul fait disparaitre du classement."""
    col_poids = col_poids or COL_POIDS
    sev, d = _severite_intrinseque(a)
    d["severite_sans_poids"] = sev
    if col_poids not in d.columns:
        print(f"  {col_poids} absent de la table d'anomalies.")
        return None
    nul = pd.to_numeric(d[col_poids], errors="coerce").fillna(0) == 0
    if not nul.any():
        print(f"  Aucune anomalie avec {col_poids} = 0. Rien n'est masque.")
        return d.head(0)
    part = 100 * nul.mean()
    rang_median = int(d.loc[nul, "severite_sans_poids"]
                      .rank(ascending=False).median())
    print(f"  {int(nul.sum()):,} anomalies sur {len(d):,} ({part:.1f} %) ont "
          f"{col_poids} = 0".replace(",", " "))
    print(f"  et donc un score composite exactement nul, quelle que soit leur "
          f"gravite.")
    cols = [c for c in ("rank", "y_obs", "y_pred", "borne_haute",
                        "severite_sans_poids", col_poids) if c in d.columns]
    top = d[nul].nlargest(n, "severite_sans_poids")[cols]
    print(f"\n  Les {len(top)} plus graves parmi elles (rang median de gravite "
          f"intrinseque : {rang_median}) :")
    print("   " + top.to_string(
        index=False, float_format=lambda v: f"{v:,.2f}".replace(",", " ")
    ).replace("\n", "\n   "))
    return top


def comparer_poids(a, colonnes, top_n=None):
    """Le classement change-t-il vraiment si l'on change de poids ?"""
    top_n = top_n or TOP_N
    sev, d = _severite_intrinseque(a)
    d["severite_sans_poids"] = sev
    colonnes = [c for c in colonnes if c in d.columns]
    if not colonnes:
        print("  Aucune des colonnes de poids proposees n'est dans la table.")
        return None

    classements, lignes = {}, []
    for c in colonnes:
        w = pd.to_numeric(d[c], errors="coerce").fillna(0)
        s = d["severite_sans_poids"] * w
        classements[c] = s.rank(ascending=False, method="first")
        lignes.append({"poids": c, "scores_nuls": int((s == 0).sum()),
                       "pct_scores_nuls": 100 * float((s == 0).mean())})
    t = pd.DataFrame(lignes)

    ref = colonnes[0]
    for c in colonnes[1:]:
        rho = float(classements[ref].corr(classements[c], method="spearman"))
        t_ref = set(classements[ref].nsmallest(top_n).index)
        t_c = set(classements[c].nsmallest(top_n).index)
        t.loc[t["poids"] == c, "spearman_vs_" + ref] = rho
        t.loc[t["poids"] == c, f"communs_top{top_n}"] = len(t_ref & t_c)
        t.loc[t["poids"] == c, f"entrants_top{top_n}"] = len(t_c - t_ref)
    print(t.to_string(index=False,
                      float_format=lambda v: f"{v:,.3f}".replace(",", " ")))

    if len(colonnes) > 1:
        c = colonnes[1]
        t_ref = set(classements[ref].nsmallest(top_n).index)
        t_c = set(classements[c].nsmallest(top_n).index)
        entrants = sorted(t_c - t_ref)
        if entrants:
            cols = [x for x in (list(_chercher("ID_COLS") or [])
                                + ["y_obs", "severite_sans_poids", ref, c])
                    if x in d.columns]
            print(f"\n  Portefeuilles qui ENTRENT dans le top {top_n} en "
                  f"passant de {ref} a {c} :")
            print("   " + d.loc[entrants, cols].to_string(
                index=False, float_format=lambda v: f"{v:,.2f}".replace(",", " ")
            ).replace("\n", "\n   "))
            print("\n  Ce sont eux, les portefeuilles que l'on ne regardait pas.")
    return t


print()
print("=" * 78)
print("BLOC 3 - IMPACT SUR LE CLASSEMENT")
print("=" * 78)
_a = TABLE_ANOMALIES if TABLE_ANOMALIES is not None else \
    (_chercher("anomalies_prio") if _chercher("anomalies_prio") is not None
     else _chercher("anomalies"))
if _a is None:
    print("  Ni anomalies_prio ni anomalies dans la session : bloc 3 ignore.")
else:
    anomalies_invisibilisees(_a)
    #  Les primes nettes reconstruites doivent aussi exister cote anomalies,
    #  sinon la comparaison ne porterait que sur les colonnes deja extraites.
    _a = _a.copy()
    for _c in (_commissions or []):
        if _c in _a.columns and COL_POIDS in _a.columns:
            _s = construire_prime_nette(_a, COL_POIDS, _c)
            _a[_s.name] = _s
    _poids = [COL_POIDS] + (list(_cand.loc[_cand["apte"], "colonne"])
                            if len(_cand) else [])
    _poids = list(dict.fromkeys([c for c in _poids if c in _a.columns]))
    if len(_poids) > 1:
        print()
        comparer_poids(_a, _poids)
    else:
        print(f"\n  Un seul poids disponible dans la table d'anomalies "
              f"({_poids}). Ajoutez les colonnes candidates a `anomaly_cols` "
              f"lors de l'extraction pour pouvoir comparer.")
