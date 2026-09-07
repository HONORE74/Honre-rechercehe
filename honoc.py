

from cibles_et_modeles import *
ruban = selecteur_cible()


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













Fin ici


# -*- coding: utf-8 -*-
# =============================================================================
#  SELECTION DE LA CIBLE ET GESTION DES MODELES ENREGISTRES
#
#  Trois problemes traites, dans l'ordre :
#
#    1. Ne plus retaper le nom de l'indicateur. Un ruban de selection en haut
#       du notebook pose TARGET, NOM_ETUDE et CHEMIN_MODELE d'un seul clic.
#    2. Ne plus ecraser le meme fichier. Chaque indicateur a son propre
#       artefact, nomme d'apres lui, plus un registre de ce qui existe.
#    3. Ne plus refaire le protocole dans le second notebook. Une ligne
#       recharge le modele voulu, avec ses donnees de test et ses metriques.
#
#  USAGE
#  -----
#  Notebook d'entrainement, tout en haut :
#      ruban = selecteur_cible()                 # clic -> TARGET est pose
#
#  Notebook d'entrainement, a l'enregistrement :
#      chemin = sauver_modele(modele_apres, X_tr, params_finaux,
#                             metriques_test=metriques("Test", y_te, pred_te),
#                             X_test=X_te, y_test=y_te, infos_test=infos_te)
#
#  Second notebook, en entier :
#      paquet   = charger_modele("RBNS_eop")     # ou selecteur_modele()
#      df_final = paquet.tableau()               # y_obs / y_pred prets
#
#  POURQUOI UN DICTIONNAIRE ET NON VOTRE CLASSE ModeleRBNS
#  --------------------------------------------------------
#  Une instance de classe enregistree par joblib ne se recharge QUE si la
#  classe est definie dans la session qui la relit. Dans un second notebook ou
#  ModeleRBNS n'existe pas, joblib.load leve
#      AttributeError: Can't get attribute 'ModeleRBNS' on module __main__
#  et il n'y a aucun moyen de recuperer le contenu sans recopier la classe.
#  L'artefact est donc un dictionnaire, qui se relit partout avec joblib seul.
#  Le confort de la classe est rendu au chargement, par un objet construit a
#  la volee, pas par un objet deserialise.
# =============================================================================

import json
import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import ipywidgets as widgets
    from IPython.display import display, HTML
    _WIDGETS = True
except ImportError:                                    # utilisable hors notebook
    _WIDGETS = False

# ┌──────────────────────────── A PARAMETRER ──────────────────────────┐
DOSSIER_ARTEFACTS = Path("artefacts_modele")

#  Les indicateurs proposes dans le ruban. Cle = nom exact de la colonne,
#  valeur = libelle lisible affiche sur le bouton.
INDICATEURS = {
    "RBNS_eop":               "RBNS",
    "IBNR_best_estimate_eop": "IBNR best estimate",
    "Risk_margin_eop":        "Marge de risque",
    "BEL_eop":                "Best estimate liabilities",
    "CSM_eop":                "CSM",
    "LRC_eop":                "LRC",
    "LIC_eop":                "LIC",
    "technical_losses":       "Pertes techniques",
    "PLR":                    "PLR",
    "GWP":                    "Primes emises",
}
CIBLE_PAR_DEFAUT = "RBNS_eop"
# └─────────────────────────────────────────────────────────────────────┘

FICHIER_REGISTRE = "registre_modeles.json"
_ENCRE, _OK, _ACCENT, _BLEU = "#141B34", "#2e7d32", "#c62828", "#0d47a1"


# =============================================================================
#  1. CIBLE ACTIVE
# =============================================================================
def _slug(nom):
    """Nom de fichier sur, derive du nom de la colonne."""
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(nom)).strip("_") or "cible"


def chemin_modele(cible, dossier=None):
    dossier = Path(dossier or DOSSIER_ARTEFACTS)
    return dossier / f"{_slug(cible)}_modele.joblib"


def _injecter(**kv):
    """Publie les variables dans le notebook, pas seulement dans ce module.

    Sans cela, un `from cibles_et_modeles import *` poserait TARGET dans le
    module et non dans la session, et vos cellules suivantes liraient l'ancienne
    valeur sans qu'aucune erreur ne le signale.
    """
    ns = None
    try:
        from IPython import get_ipython
        ip = get_ipython()
        ns = ip.user_ns if ip is not None else None
    except Exception:
        pass
    for k, v in kv.items():
        globals()[k] = v
        if ns is not None:
            ns[k] = v


def definir_cible(cible, dossier=None, silencieux=False):
    """Pose TARGET, CIBLE, NOM_ETUDE et CHEMIN_MODELE partout.

    Equivalent programmatique du ruban. A utiliser dans un run automatise, ou
    l'etat d'un widget ne serait pas reproductible.
    """
    ch = chemin_modele(cible, dossier)
    _injecter(TARGET=cible, CIBLE=cible, NOM_ETUDE=_slug(cible),
              CHEMIN_MODELE=ch, CIBLE_ACTIVE=cible)
    if not silencieux:
        etat = "deja enregistre" if ch.exists() else "pas encore enregistre"
        print(f"TARGET = {cible!r}   |   fichier : {ch}   ({etat})")
    return cible


def _libelles(indicateurs, df=None):
    """(libelle, colonne) pour le ruban, en signalant les colonnes absentes."""
    items = (indicateurs.items() if isinstance(indicateurs, dict)
             else [(c, c) for c in indicateurs])
    options = []
    for col, lib in items:
        if df is not None and col not in df.columns:
            options.append((f"{lib}  (absent)", col))
        else:
            options.append((lib, col))
    return options


def selecteur_cible(indicateurs=None, df=None, defaut=None, dossier=None):
    """Ruban de selection de l'indicateur. Un clic pose TARGET.

    Retourne le widget. La cible par defaut est posee immediatement, pour
    qu'un `Run All` sans clic parte quand meme d'un etat defini.
    """
    indicateurs = indicateurs or INDICATEURS
    if df is None:
        df = globals().get("df")
    options = _libelles(indicateurs, df)
    valeurs = [v for _, v in options]
    defaut = defaut or (CIBLE_PAR_DEFAUT if CIBLE_PAR_DEFAUT in valeurs
                        else valeurs[0])

    definir_cible(defaut, dossier, silencieux=True)

    if not _WIDGETS:
        print("ipywidgets absent : cible posee par defaut.")
        definir_cible(defaut, dossier)
        return None

    ruban = widgets.ToggleButtons(
        options=options, value=defaut, description="",
        layout=widgets.Layout(display="flex", flex_flow="row wrap",
                              width="100%"),
        style={"button_width": "auto"})
    bandeau = widgets.HTML()

    def _rendu(cible):
        ch = chemin_modele(cible, dossier)
        if ch.exists():
            taille = ch.stat().st_size / 1e6
            etat = (f"<b style='color:{_OK}'>modele enregistre</b> "
                    f"({taille:.1f} Mo)")
        else:
            etat = f"<span style='color:{_ACCENT}'>pas encore entraine</span>"
        absente = (df is not None and cible not in df.columns)
        alerte = (f"<br><b style='color:{_ACCENT}'>Attention : la colonne "
                  f"{cible} est absente de df.</b>" if absente else "")
        bandeau.value = (
            f"<div style='font-family:system-ui,sans-serif;font-size:13px;"
            f"background:#eef3fb;color:{_BLEU};padding:10px 14px;"
            f"border-radius:6px;margin:8px 0'>"
            f"Cible active : <b style='font-size:15px'>{cible}</b>"
            f" &nbsp;·&nbsp; fichier <code>{ch}</code> &nbsp;·&nbsp; {etat}"
            f"{alerte}</div>")

    def _au_clic(c):
        if c["name"] == "value":
            definir_cible(c["new"], dossier, silencieux=True)
            _rendu(c["new"])

    ruban.observe(_au_clic, names="value")
    _rendu(defaut)
    display(widgets.VBox([ruban, bandeau]))
    return ruban


# =============================================================================
#  2. ENREGISTREMENT
# =============================================================================
def _categories(X):
    return {c: list(X[c].cat.categories) for c in X.columns
            if str(X[c].dtype) == "category"}


def preparer(X, paquet):
    """Remet les colonnes du modele dans l'ordre et le dtype attendus."""
    feats = paquet["features"]
    manquantes = [c for c in feats if c not in X.columns]
    if manquantes:
        raise KeyError(f"Colonnes absentes de X : {manquantes[:5]}"
                       f"{' ...' if len(manquantes) > 5 else ''}")
    Xp = X[feats].copy()
    for c in paquet.get("categorielles", []):
        Xp[c] = pd.Categorical(Xp[c].astype(str),
                               categories=paquet["categories"][c])
    return Xp


def predire(paquet, X, positif=True):
    p = paquet["modele"].predict(preparer(X, paquet))
    return np.clip(p, 0, None) if positif else p


def sauver_modele(modele, X_tr, params=None, *, cible=None, metriques_test=None,
                  X_test=None, y_test=None, infos_test=None, dossier=None,
                  verifier=True, extras=None):
    """Enregistre le modele dans un fichier propre a la cible active.

    Ce qui est ecrit est un dictionnaire, relisable partout avec joblib seul.
    X_test / y_test / infos_test sont facultatifs mais fortement conseilles :
    ce sont eux qui permettent au second notebook de ne rien recalculer.
    """
    cible = cible or globals().get("CIBLE_ACTIVE") or globals().get("TARGET")
    if not cible:
        raise ValueError("Aucune cible active. Appelez selecteur_cible() ou "
                         "definir_cible('NOM_COLONNE') d'abord.")
    dossier = Path(dossier or DOSSIER_ARTEFACTS)
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = chemin_modele(cible, dossier)

    cats = _categories(X_tr)
    paquet = {
        "cible": cible,
        "modele": modele,
        "features": list(X_tr.columns),
        "categorielles": list(cats.keys()),
        "categories": cats,
        "params": params,
        "metriques_test": metriques_test,
        "date": datetime.now().isoformat(timespec="seconds"),
        "n_train": int(len(X_tr)),
        "X_test": X_test,
        "y_test": None if y_test is None else np.asarray(y_test),
        "infos_test": infos_test,
        "format": "dict-v1",
    }
    if extras:
        paquet.update(extras)

    joblib.dump(paquet, chemin)
    taille = chemin.stat().st_size / 1e6

    #  Verification de rechargement. Un modele qui ne redonne pas exactement
    #  les memes predictions apres un aller-retour disque est inutilisable, et
    #  le probleme ne se verrait qu'au moment ou l'on s'en sert.
    ecart, statut = None, "non verifie"
    if verifier and X_test is not None:
        relu = joblib.load(chemin)
        ecart = float(np.abs(predire(relu, X_test)
                             - predire(paquet, X_test)).max())
        statut = "identique" if ecart < 1e-9 else "ECART, A VERIFIER"

    _maj_registre(dossier, cible, paquet, chemin, taille)

    mae = None
    if isinstance(metriques_test, dict):
        mae = metriques_test.get("MAE")
    print(f"Cible      : {cible}")
    print(f"Fichier    : {chemin.resolve()}  ({taille:.2f} Mo)")
    print(f"Variables  : {len(paquet['features'])}"
          + (f"   |   MAE test : {mae:,.0f}".replace(",", " ")
             if isinstance(mae, (int, float)) else ""))
    if ecart is not None:
        print(f"Rechargement : ecart max {ecart:.3g}   ->   {statut}")
    return chemin


def _maj_registre(dossier, cible, paquet, chemin, taille):
    """Un index lisible de ce qui existe, pour ne pas avoir a lister le disque."""
    fichier = Path(dossier) / FICHIER_REGISTRE
    reg = {}
    if fichier.exists():
        try:
            reg = json.loads(fichier.read_text(encoding="utf-8"))
        except Exception:
            reg = {}
    mae = (paquet["metriques_test"] or {}).get("MAE") \
        if isinstance(paquet["metriques_test"], dict) else None
    reg[cible] = {"fichier": chemin.name, "date": paquet["date"],
                  "n_features": len(paquet["features"]),
                  "n_train": paquet["n_train"],
                  "mae_test": float(mae) if isinstance(mae, (int, float)) else None,
                  "taille_mo": round(taille, 2)}
    fichier.write_text(json.dumps(reg, indent=2, ensure_ascii=False),
                       encoding="utf-8")


# =============================================================================
#  3. CHARGEMENT (second notebook)
# =============================================================================
_ALIAS = {"y_test": ("y_test", "y_te"), "X_test": ("X_test", "X_te"),
          "infos_test": ("infos_test", "infos_te"), "cible": ("cible", "target")}


def _normaliser(brut):
    """Accepte le format dict de ce fichier ET vos artefacts precedents."""
    if isinstance(brut, dict):
        p = dict(brut)
    else:                                   # ancienne instance de classe
        p = {k: v for k, v in vars(brut).items() if not k.startswith("_")}
    for cle, sources in _ALIAS.items():
        if cle not in p:
            for s in sources:
                if s in p:
                    p[cle] = p[s]
                    break
    if "mae_test" in p and not isinstance(p.get("metriques_test"), dict):
        p["metriques_test"] = {"MAE": p["mae_test"]}
    p.setdefault("cible", "cible inconnue")
    p.setdefault("categorielles", [])
    p.setdefault("categories", {})
    for oblig in ("modele", "features"):
        if oblig not in p:
            raise KeyError(f"Artefact incomplet : cle '{oblig}' absente.")
    return p


class Modele:
    """Confort de la classe, construit au chargement et non deserialise."""

    def __init__(self, paquet, chemin=None):
        self.p, self.chemin = paquet, chemin
        for k in ("cible", "features", "categorielles", "categories",
                  "params", "metriques_test", "date"):
            setattr(self, k, paquet.get(k))
        self.modele = paquet["modele"]

    def __repr__(self):
        mae = (self.metriques_test or {}).get("MAE") \
            if isinstance(self.metriques_test, dict) else None
        m = f" | MAE test {mae:,.0f}".replace(",", " ") \
            if isinstance(mae, (int, float)) else ""
        return (f"<Modele {self.cible} | {len(self.features)} variables{m} "
                f"| {self.date}>")

    def predire(self, X=None, positif=True):
        X = self.p["X_test"] if X is None else X
        if X is None:
            raise ValueError("Aucun X fourni et aucun X_test dans l'artefact.")
        return predire(self.p, X, positif)

    def importance(self, n=20):
        b = getattr(self.modele, "booster_", None)
        noms = getattr(self.modele, "feature_name_", self.features)
        if b is None:
            raise AttributeError("Le modele n'expose pas d'importance de gain.")
        v = pd.Series(b.feature_importance("gain"), index=noms)
        return (100 * v / v.sum()).sort_values(ascending=False).head(n)

    def tableau(self, trier=True):
        """df_final pret a l'emploi : infos metier + y_obs + y_pred + ecart."""
        X, y = self.p.get("X_test"), self.p.get("y_test")
        if X is None or y is None:
            raise ValueError("L'artefact ne contient pas X_test / y_test. "
                             "Relancez sauver_modele() en les passant.")
        infos = self.p.get("infos_test")
        d = (infos.reset_index(drop=True).copy() if infos is not None
             else pd.DataFrame(index=range(len(X))))
        d[self.cible] = np.asarray(y)
        d["y_obs"] = np.asarray(y)
        d["y_pred"] = self.predire(X)
        d["ecart"] = d["y_pred"] - d["y_obs"]
        if trier:
            d = d.sort_values("y_obs", ascending=False)
        return d.reset_index(drop=True)

    def resume(self, n=20):
        d = self.tableau()
        mae = d["ecart"].abs().mean()
        enreg = (self.metriques_test or {}).get("MAE") \
            if isinstance(self.metriques_test, dict) else None
        print(f"Cible {self.cible}  |  modele du {self.date}  |  "
              f"{len(d):,} predictions".replace(",", " "))
        ligne = f"MAE recalculee : {mae:,.0f}".replace(",", " ")
        if isinstance(enreg, (int, float)):
            coherent = abs(mae - enreg) < max(1e-6 * max(enreg, 1), 1e-6)
            ligne += (f"   (enregistree : {enreg:,.0f})".replace(",", " ")
                      + ("   coherent" if coherent else "   ECART A VERIFIER"))
        print(ligne)
        print(d.head(n).to_string(
            index=False, float_format=lambda v: f"{v:,.2f}".replace(",", " ")))
        return d


def lister_modeles(dossier=None):
    """Ce qui est disponible sur le disque, sans rien charger en memoire."""
    dossier = Path(dossier or DOSSIER_ARTEFACTS)
    if not dossier.exists():
        return pd.DataFrame(columns=["cible", "date", "n_features", "mae_test",
                                     "taille_mo", "fichier"])
    reg = {}
    f = dossier / FICHIER_REGISTRE
    if f.exists():
        try:
            reg = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            reg = {}
    lignes = []
    for cible, info in reg.items():
        if (dossier / info["fichier"]).exists():
            lignes.append({"cible": cible, **info})
    connus = {l["fichier"] for l in lignes}
    for ch in sorted(dossier.glob("*_modele.joblib")):     # artefacts hors registre
        if ch.name not in connus:
            lignes.append({"cible": ch.stem.replace("_modele", ""),
                           "fichier": ch.name, "date": None, "n_features": None,
                           "n_train": None, "mae_test": None,
                           "taille_mo": round(ch.stat().st_size / 1e6, 2)})
    return pd.DataFrame(lignes).sort_values("cible").reset_index(drop=True)


def charger_modele(cible=None, dossier=None, chemin=None):
    """Recharge un modele. Sans argument, prend la cible active du ruban."""
    if chemin is not None:
        ch = Path(chemin)
    else:
        cible = cible or globals().get("CIBLE_ACTIVE") or globals().get("TARGET")
        dossier = Path(dossier or DOSSIER_ARTEFACTS)
        if cible is None:
            dispo = lister_modeles(dossier)
            if len(dispo) == 1:
                cible = dispo["cible"].iloc[0]
            else:
                raise ValueError(
                    "Precisez la cible. Disponibles : "
                    + ", ".join(dispo["cible"]) if len(dispo)
                    else f"Aucun modele dans {dossier}.")
        ch = chemin_modele(cible, dossier)
    if not ch.exists():
        dispo = lister_modeles(ch.parent)
        raise FileNotFoundError(
            f"{ch} introuvable. Disponibles : "
            + (", ".join(dispo["cible"]) if len(dispo) else "aucun."))
    return Modele(_normaliser(joblib.load(ch)), ch)


def selecteur_modele(dossier=None, au_chargement=None):
    """Ruban des modeles disponibles, dans le second notebook.

    Retourne un dict dont la cle 'modele' contient le dernier charge, pour que
    la cellule suivante puisse s'en servir sans reappeler charger_modele().
    """
    dossier = Path(dossier or DOSSIER_ARTEFACTS)
    dispo = lister_modeles(dossier)
    etat = {"modele": None}
    if not len(dispo):
        print(f"Aucun modele enregistre dans {dossier.resolve()}.")
        return etat
    if not _WIDGETS:
        etat["modele"] = charger_modele(dispo["cible"].iloc[0], dossier)
        print(etat["modele"])
        return etat

    options = []
    for _, r in dispo.iterrows():
        mae = (f"  MAE {r['mae_test']:,.0f}".replace(",", " ")
               if pd.notna(r.get("mae_test")) else "")
        options.append((f"{r['cible']}{mae}", r["cible"]))
    ruban = widgets.ToggleButtons(
        options=options, value=options[0][1],
        layout=widgets.Layout(display="flex", flex_flow="row wrap", width="100%"),
        style={"button_width": "auto"})
    sortie = widgets.Output()

    def _charger(cible):
        with sortie:
            from IPython.display import clear_output
            clear_output(wait=True)
            try:
                m = charger_modele(cible, dossier)
                etat["modele"] = m
                _injecter(MODELE=m, CIBLE_CHARGEE=cible)
                print(m)
                if au_chargement is not None:
                    au_chargement(m)
            except Exception as e:
                print(f"Chargement impossible : {type(e).__name__} : {e}")

    ruban.observe(lambda c: _charger(c["new"]) if c["name"] == "value" else None,
                  names="value")
    display(widgets.VBox([ruban, sortie]))
    _charger(options[0][1])
    return etat














La fin est icic 














# -*- coding: utf-8 -*-
# =============================================================================
#  MISSION 1 - EVOLUTION DE LA TARGET POUR UN SOUS-PORTEFEUILLE
#
#  Votre code, avec trois modifications integrees :
#    1. format des nombres en francais (1 249 441 et non 1,249,441), cote Python
#    2. format des nombres en francais cote Plotly (axes et infobulles)
#    3. zone decorative sous la courbe passee en gris neutre, pour ne plus
#       etre confondue avec la bande bleue de l'intervalle conforme
#
#  Prerequis dans la session : df, expl, anomalies_prio, ID_COLS, TARGET
# =============================================================================

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

# ┌─────────────────────────── PARAMETRES ────────────────────────────┐
RANG          = 1        # 1 = l'anomalie la plus grave  (*)
N_trimestre   = 8        # nombre de trimestres d'historique affiches
VARS_SECOND   = 0        # variables secondaires sous la courbe (0 = aucune)
AFFICHER_VAL  = True     # afficher la valeur au-dessus de chaque point
# └────────────────────────────────────────────────────────────────────┘
#  (*) cette ligne etait coupee sur votre capture, remettez votre valeur.

ENCRE, ACCENT, OK  = "#141B34", "#FF5A5F", "#3D5A9E"
BLEU, GRILLE, GRIS = "#636EFA", "#EDF1F7", "#8A93A5"
DOUX = ["#6C8EBF", "#82B366", "#C08552", "#9673A6", "#5F9EA0", "#B85C7E"]

# --- MODIFICATION 1 ----------------------------------------------------------
#  Format francais des nombres cote Python. Remplace tous les f"{v:,.0f}", qui
#  produisaient le format anglo-saxon a virgules (1,249,441).
# -----------------------------------------------------------------------------
def fmt(v):
    """1249441.0 -> '1 249 441'  (separateur de milliers francais)."""
    return f"{v:,.0f}".replace(",", " ")


# ---------------------------------------------------- selection du sous-portefeuille
_c = [c for c in ID_COLS if c in expl.columns and c in df.columns]
UNITE = tuple(str(anomalies_prio.iloc[RANG - 1][x]) for x in _c)
nom = " · ".join(UNITE)

_m = np.logical_and.reduce([df[c].astype(str).values == v
                            for c, v in zip(_c, UNITE)])
h = df[_m].sort_values("time_idx").tail(N_trimestre).copy()
if len(h) == 0:
    raise ValueError("Aucun historique pour cette unite.")
if len(h) < N_trimestre:
    print(f"ATTENTION : seulement {len(h)} trimestre(s) disponible(s) pour "
          f"{nom} (demande : {N_trimestre}). Ce sous-portefeuille est "
          f"probablement recent ou incomplet dans la base.")

per = (h["year"].astype(int).astype(str) + "-T"
       + h["quarter"].astype(int).astype(str)).tolist()
val = h[TARGET].values.astype(float)

# ------------------------------------------- contexte conforme (periode validee)
_mt = np.logical_and.reduce([expl[c].astype(str).values == v
                             for c, v in zip(_c, UNITE)])
_t = expl[_mt]
ctx = None
if len(_t):
    r = _t.iloc[0]
    ctx = dict(per=f"{int(r['year'])}-T{int(r['quarter'])}",
               pred=float(r["y_pred"]),
               lo=float(r["borne_basse"]), hi=float(r["borne_haute"]),
               obs=float(r["y_obs"]), couvert=bool(r["dans_intervalle"]))

# ------------------------------------------------------- variables secondaires
secondaires = []
if VARS_SECOND > 0:
    num = [c for c in h.columns
           if c != TARGET and pd.api.types.is_numeric_dtype(h[c])
           and h[c].notna().all() and h[c].nunique() > 1
           and c not in ("time_idx", "year", "quarter")]
    if "MODELE_TE" in globals() and MODELE_TE is not None:
        try:
            mdl = (MODELE_TE.named_steps["model"]
                   if hasattr(MODELE_TE, "named_steps") else MODELE_TE)
            imp = pd.Series(mdl.booster_.feature_importance("gain"),
                            index=mdl.feature_name_)
            num = [v for v in imp.sort_values(ascending=False).index
                   if v in num] or num
        except Exception:
            pass
    secondaires = num[:VARS_SECOND]

# ------------------------------------------------------------------- figure
n_rows = 1 + len(secondaires)
fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=.06,
                    row_heights=[.58] + [.42 / max(len(secondaires), 1)]
                                * len(secondaires) if secondaires else [1.0])

if ctx and ctx["per"] in per:
    k = per.index(ctx["per"])
    for rr in range(1, n_rows + 1):
        fig.add_vrect(x0=k - .5, x1=k + .5, fillcolor="rgba(99,110,250,0.055)",
                      line_width=0, layer="below", row=rr, col=1)

# --- MODIFICATION 3 ----------------------------------------------------------
#  La zone sous la courbe etait bleue, donc de la meme famille que la bande de
#  l'intervalle conforme. Un lecteur pouvait croire que TOUT l'historique etait
#  couvert par un intervalle, alors qu'il n'y en a qu'un, sur la periode validee.
#  Elle passe en gris neutre. Pour la supprimer completement plutot que la
#  neutraliser, commentez le bloc try/except ci-dessous.
# -----------------------------------------------------------------------------
aire = dict(x=per, y=val, mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
            fill="tozeroy", showlegend=False, hoverinfo="skip")
try:
    fig.add_scatter(**aire, fillgradient=dict(type="vertical", colorscale=[
        (0, "rgba(140,147,165,0.02)"), (1, "rgba(140,147,165,0.14)")]),
        row=1, col=1)
except Exception:
    fig.add_scatter(**aire, fillcolor="rgba(140,147,165,0.10)", row=1, col=1)

# ----------------------------------------------------------------------------
#  BLOC RECONSTRUIT : coupe entre vos 2e et 3e captures. Reconstitue d'apres la
#  legende de votre graphique (Intervalle CP 90 %, Prediction) et la bande
#  verticale bleue visible sur 2024-T4. A verifier contre votre original.
# ----------------------------------------------------------------------------
if ctx and ctx["per"] in per:
    fig.add_scatter(x=[ctx["per"], ctx["per"]], y=[ctx["lo"], ctx["hi"]],
                    mode="lines", line=dict(color=BLEU, width=15), opacity=.26,
                    name="Intervalle CP 90 %",
                    hovertemplate=f"Intervalle<br>[{fmt(ctx['lo'])} ; "
                                  f"{fmt(ctx['hi'])}]<extra></extra>",
                    row=1, col=1)
    fig.add_scatter(x=[ctx["per"]], y=[ctx["pred"]], mode="markers",
                    marker=dict(size=10, symbol="diamond", color="white",
                                line=dict(color=ENCRE, width=1.8)),
                    name="Prediction",
                    hovertemplate=f"Prediction<br>{fmt(ctx['pred'])}"
                                  "<extra></extra>",
                    row=1, col=1)
# ------------------------------------------------- fin du bloc reconstruit ---

fig.add_scatter(x=per, y=val,
                mode="lines+markers+text" if AFFICHER_VAL else "lines+markers",
                line=dict(color=ENCRE, width=2.8, shape="spline", smoothing=.55),
                marker=dict(size=9, color="white",
                            line=dict(color=ENCRE, width=2.2)),
                text=[fmt(v) for v in val] if AFFICHER_VAL else None,
                textposition="top center", textfont=dict(size=9.5, color=GRIS),
                name=TARGET,
                hovertemplate="<b>%{x}</b><br>" + TARGET +
                              " : <b>%{y:,.0f}</b><extra></extra>",
                row=1, col=1)

if ctx and ctx["per"] in per:
    coul = ACCENT if not ctx["couvert"] else OK
    halo = ("rgba(255,90,95,0.20)" if not ctx["couvert"]
            else "rgba(61,90,158,0.18)")
    for taille, c in ((38, halo), (24, halo)):
        fig.add_scatter(x=[ctx["per"]], y=[ctx["obs"]], mode="markers",
                        marker=dict(size=taille, color=c), showlegend=False,
                        hoverinfo="skip", row=1, col=1)
    fig.add_scatter(x=[ctx["per"]], y=[ctx["obs"]], mode="markers",
                    marker=dict(size=13, color=coul,
                                line=dict(color="white", width=2.4)),
                    name="Hors intervalle" if not ctx["couvert"] else "Couvert",
                    hovertemplate=f"<b>{ctx['per']}</b><br>"
                                  f"Observe : {fmt(ctx['obs'])}<br>"
                                  + ("HORS intervalle" if not ctx["couvert"]
                                     else "Couvert")
                                  + "<extra></extra>", row=1, col=1)

for i, v in enumerate(secondaires, start=2):
    c = DOUX[(i - 2) % len(DOUX)]
    vv = h[v].values.astype(float)
    fig.add_scatter(x=per, y=vv, mode="lines+markers",
                    line=dict(color=c, width=2.2, shape="spline", smoothing=.55),
                    marker=dict(size=6, color="white",
                                line=dict(color=c, width=1.8)),
                    name=v, showlegend=False,
                    # ligne reconstruite : fin coupee sur votre 3e capture
                    hovertemplate=f"<b>%{{x}}</b><br>{v} : <b>%{{y:,.4g}}</b>"
                                  "<extra></extra>",
                    row=i, col=1)
    fig.add_annotation(xref="paper", x=1.005, y=vv[-1], xanchor="left",
                       text=f"<b>{v}</b>", showarrow=False,
                       font=dict(size=10, color=c), row=i, col=1)

fig.update_xaxes(showgrid=False, showspikes=True, spikemode="across",
                 spikethickness=1.2, spikedash="dot", spikecolor=GRIS,
                 tickfont=dict(size=11), linecolor=GRILLE)
fig.update_yaxes(gridcolor=GRILLE, zeroline=False, showspikes=True,
                 spikemode="across", spikethickness=1.2, spikedash="dot",
                 spikecolor=GRIS, tickformat=",.0f", tickfont=dict(size=11))
fig.update_yaxes(title_text=f"<b>{TARGET}</b>", title_font=dict(size=12),
                 row=1, col=1)
fig.update_xaxes(title_text="<b>Trimestre</b>", title_font=dict(size=12),
                 row=n_rows, col=1)

delta = 100 * (val[-1] - val[0]) / abs(val[0]) if val[0] else np.nan
fleche = "▲" if delta >= 0 else "▼"

fig.update_layout(
    title=dict(text=f"<b style='font-size:19px;color:{ENCRE}'>{nom}</b>"
                    f"<br><span style='font-size:12px;color:{GRIS}'>"
                    f"{TARGET} · {len(h)} trimestres · "
                    f"{per[0]} → {per[-1]} · "
                    f"<span style='color:{ACCENT if delta < 0 else OK}'>{fleche} "
                    f"{abs(delta):.1f} %</span> sur la periode</span>",
               x=.015, xanchor="left", y=.96),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="white", bordercolor=GRILLE,
                    font=dict(size=12.5, family="Inter, system-ui, sans-serif",
                              color=ENCRE), align="left"),
    template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color=GRIS),
    height=430 + 130 * len(secondaires),
    legend=dict(orientation="h", y=1.04, x=1, xanchor="right",
                bgcolor="rgba(255,255,255,0)", font=dict(size=11)),
    margin=dict(l=80, r=110, t=120, b=60),
    # --- MODIFICATION 2 ------------------------------------------------------
    #  Format francais cote Plotly. Les %{y:,.0f} des hovertemplate et le
    #  tickformat des axes sont calcules par le navigateur, pas par Python :
    #  fmt() ne les touche pas. Cette ligne les regle tous d'un coup, virgule
    #  pour les decimales et espace pour les milliers.
    # -------------------------------------------------------------------------
    separators=", ")

fig.show()

print(f"Rang #{RANG} · {nom}")
print(f"Min {fmt(val.min())} | Median {fmt(np.median(val))} | "
      f"Max {fmt(val.max())}")
if ctx:
    print(f"Periode validee {ctx['per']} : observe {fmt(ctx['obs'])} | "
          f"predit {fmt(ctx['pred'])} | "
          f"intervalle [{fmt(ctx['lo'])} ; {fmt(ctx['hi'])}]"
          f" -> {'COUVERT' if ctx['couvert'] else 'HORS INTERVALLE'}")








Mission 2





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
