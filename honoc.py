# =============================================================================
#  GRAPHIQUES CONFORMAL PREDICTION  -  VERSION "UNE FIGURE PAR BLOC"
#  Memoire : detection et priorisation des observations atypiques (IFRS 17 / S2)
# -----------------------------------------------------------------------------
#  REGLES ABSOLUES DE CE FICHIER
#    1. UNE fonction = UNE figure = UNE slide. Jamais de plt.subplots(2, 3).
#    2. AUCUN logarithme. Ni sur les donnees (np.log, log1p), ni sur les axes
#       (set_yscale("log"), "symlog", plotly type="log"). Le BLOC 1 le verifie
#       automatiquement et leve une erreur si une occurrence reapparait.
#    3. Chaque figure est autoportante : titre, sous-titre de lecture, unites,
#       note de bas de figure. Elle doit se comprendre sans commentaire oral.
#
#  ORDRE D'EXECUTION
#    PARTIE 0  Socle          BLOC 0   Imports & charte graphique
#                             BLOC 1   Outils communs + garde-fou "zero log"
#                             BLOC 2   Contrat de donnees (verification)
#    PARTIE A  Contexte       FIG 01   Profil de la cible (echelle de rang)
#                             FIG 02   Concentration de la cible (Lorenz)
#    PARTIE B  Validite       FIG 03   Couverture empirique vs nominale
#                             FIG 04   Histogramme des p-values conformes
#                             FIG 05   QQ-plot des p-values
#                             FIG 06   Procedure de Benjamini-Hochberg
#                             FIG 07   Couverture par decile de largeur (SSC)
#                             FIG 08   Distribution des largeurs (rang)
#    PARTIE C  Bande conforme FIG 09   Bande normalisee (population entiere)
#                             FIG 10   Bande absolue, UNE strate par appel
#                             FIG 11   Bande detaillee, N unites etiquetees
#                             FIG 12   Predit vs reel en echelle de rang
#    PARTIE D  Priorisation   FIG 13   Top N par score composite
#                             FIG 14   Pareto de concentration du score
#                             FIG 15   Decomposition du score (percentiles)
#                             FIG 16   Fiche visuelle d'une anomalie
#                             FIG 17   Trajectoire temporelle d'une entite
#                             FIG 18   Persistance / recurrence
#    PARTIE E  Sortie         BLOC 3   Execution complete + index des figures
# =============================================================================


# =============================================================================
#  BLOC 0 - IMPORTS ET CHARTE GRAPHIQUE
# -----------------------------------------------------------------------------
#  Une charte unique garantit que les 18 figures forment un ensemble coherent :
#  memes couleurs, memes polices, meme grille. C'est ce qui distingue une
#  planche de memoire d'une serie de graphiques bricoles un par un.
# =============================================================================
import os
import inspect
import textwrap

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

# --- Dossier de sortie ------------------------------------------------------
DOSSIER_FIGURES = "figures_conformal"
os.makedirs(DOSSIER_FIGURES, exist_ok=True)

# --- Palette (daltonisme-safe, impression N&B acceptable) -------------------
COULEURS = {
    "conforme":     "#2E7D5B",   # vert   : observation dans l'intervalle
    "conforme_cl":  "#A8D5BF",   # vert clair
    "anomalie":     "#C0392B",   # rouge  : observation hors intervalle
    "anomalie_cl":  "#F0B7B0",   # rouge clair
    "prediction":   "#1B2A38",   # bleu nuit : ancre du modele
    "bande":        "#7FA6D9",   # bleu   : intervalle conforme
    "bande_cl":     "#D3E0F2",   # bleu tres clair (remplissage)
    "reference":    "#E67E22",   # orange : seuils, references theoriques
    "neutre":       "#8C99A6",   # gris   : contexte, elements secondaires
    "neutre_cl":    "#DDE2E7",
    "accent":       "#6C3483",   # violet : mise en evidence ponctuelle
}

# --- Parametres matplotlib globaux ------------------------------------------
mpl.rcParams.update({
    "figure.figsize":      (11.5, 6.4),   # ratio 16/9 approche = 1 slide
    "figure.dpi":          110,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "font.family":         "DejaVu Sans",
    "font.size":           11,
    "axes.titlesize":      13.5,
    "axes.titleweight":    "bold",
    "axes.labelsize":      11,
    "axes.edgecolor":      "#4A5560",
    "axes.linewidth":      0.9,
    "axes.grid":           True,
    "axes.axisbelow":      True,
    "grid.color":          "#C9D0D6",
    "grid.linestyle":      ":",
    "grid.linewidth":      0.7,
    "grid.alpha":          0.7,
    "legend.frameon":      True,
    "legend.framealpha":   0.92,
    "legend.edgecolor":    "#C9D0D6",
    "legend.fontsize":     9.5,
    "xtick.labelsize":     9.5,
    "ytick.labelsize":     9.5,
    "xtick.color":         "#4A5560",
    "ytick.color":         "#4A5560",
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    "figure.max_open_warning": 0,   # on produit volontairement 20+ figures
})

# --- Parametres a adapter a ton notebook ------------------------------------
# Ces noms doivent correspondre a ce que tu as deja en memoire.
TARGET     = globals().get("TARGET", "RBNS_eop")
ALPHA      = globals().get("ALPHA", 0.10)
ID_COLS    = globals().get("ID_COLS", ["Companies", "Lob", "Risk"])
GWP_COL    = globals().get("GWP_COL", "GWP")
UNITE      = "EUR"

# Percentiles de decoupage en strates de magnitude (remplacent le role du log)
STRATES_PCT = [50, 90, 99]
STRATES_NOM = ["P0-50", "P50-90", "P90-99", "P99+"]


# =============================================================================
#  BLOC 1 - OUTILS COMMUNS + GARDE-FOU "ZERO LOG"
# -----------------------------------------------------------------------------
#  Tout ce qui est partage entre les 18 figures est ici, et nulle part ailleurs.
#  Modifier une couleur ou un format de nombre se fait a UN seul endroit.
# =============================================================================

def format_montant(v, pos=None):
    """Formate un montant en euros de facon lisible, SANS aucun logarithme.

    Le choix du suffixe (k / M / Md) est un simple test de seuil sur la valeur
    absolue : c'est de la mise en forme d'etiquette, pas une transformation de
    la donnee. L'axe reste strictement lineaire.
    """
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:,.2f} Md"
    if a >= 1e6:
        return f"{v / 1e6:,.1f} M"
    if a >= 1e3:
        return f"{v / 1e3:,.0f} k"
    return f"{v:,.0f}"


def cadrer_axe_y(ax, valeurs, p_bas=0.0, p_haut=99.5, marge=0.06):
    """Cadrage robuste : borne l'axe Y sur le corps de la distribution.

    C'est l'alternative honnete au logarithme. Au lieu de comprimer l'echelle,
    on choisit une fenetre de lecture et on DECLARE combien d'observations
    sortent du cadre. Le lecteur sait exactement ce qu'il ne voit pas, ce qui
    n'est jamais le cas avec un axe logarithmique.

    Retourne (n_au_dessus, n_en_dessous).
    """
    v = np.asarray(valeurs, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0, 0

    lo = np.percentile(v, p_bas)
    hi = np.percentile(v, p_haut)
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    ax.set_ylim(lo - marge * span, hi + marge * span)

    n_sup = int(np.sum(v > hi + marge * span))
    n_inf = int(np.sum(v < lo - marge * span))

    mentions = []
    if n_sup:
        mentions.append(f"{n_sup:,} observation(s) au-dessus du cadre")
    if n_inf:
        mentions.append(f"{n_inf:,} observation(s) sous le cadre")
    if mentions:
        ax.text(0.995, 0.015, "Hors cadre : " + " ; ".join(mentions),
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, style="italic", color=COULEURS["neutre"],
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec=COULEURS["neutre_cl"], alpha=0.9))
    return n_sup, n_inf


def marquer_hors_cadre(ax, x, y, seuil_haut, couleur=None):
    """Dessine une fleche au bord superieur pour chaque point hors cadre.

    Le point n'est pas efface : sa position en X est conservee, seule sa
    hauteur est plafonnee. On voit donc OU se situent les valeurs extremes,
    sans qu'elles ecrasent l'echelle.
    """
    couleur = couleur or COULEURS["anomalie"]
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    m = y > seuil_haut
    if m.any():
        ax.scatter(x[m], np.full(m.sum(), seuil_haut), marker="^", s=55,
                   color=couleur, edgecolor="white", linewidth=0.6,
                   zorder=8, clip_on=False,
                   label=f"Valeur au-dela du cadre (n={m.sum():,})")
    return int(m.sum())


def finaliser(fig, ax, titre, lecture=None, xlabel=None, ylabel=None,
              nom_fichier=None, legende="best", note=None,
              format_y_montant=False, format_x_montant=False):
    """Applique la charte a une figure et l'enregistre.

    `lecture` est la phrase qui explique COMMENT lire le graphique. Elle est
    placee sous le titre : c'est ce qui rend chaque planche autoportante en
    soutenance, quand le jury regarde la figure avant que tu ne parles.
    """
    if lecture:
        # Le titre est place AU-DESSUS de la phrase de lecture. Sa hauteur est
        # calculee a partir du nombre de lignes reellement produites par le
        # retour a la ligne automatique : sans ce calcul, un texte de lecture
        # long passe sous le titre et les deux se chevauchent.
        texte = textwrap.fill(lecture, 105)
        n_lignes = texte.count("\n") + 1
        hauteur_ligne = 0.042          # en coordonnees d'axes
        ax.text(0.0, 1.015, texte, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9.5, color="#4A5560",
                style="italic", linespacing=1.35)
        ax.set_title(titre, loc="left", pad=0,
                     y=1.015 + n_lignes * hauteur_ligne + 0.018)
    else:
        ax.set_title(titre, loc="left", pad=12)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if format_y_montant:
        ax.yaxis.set_major_formatter(FuncFormatter(format_montant))
    if format_x_montant:
        ax.xaxis.set_major_formatter(FuncFormatter(format_montant))

    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    if handles and legende:
        # Deduplication : evite "Anomalie" repete 3 fois dans la legende
        vus, h2, l2 = set(), [], []
        for h, l in zip(handles, labels):
            if l not in vus and not l.startswith("_"):
                vus.add(l)
                h2.append(h)
                l2.append(l)
        ax.legend(h2, l2, loc=legende)

    if note:
        fig.text(0.005, -0.02, textwrap.fill(note, 135), fontsize=8.5,
                 color=COULEURS["neutre"], ha="left", va="top")

    fig.tight_layout()

    if nom_fichier:
        chemin = os.path.join(DOSSIER_FIGURES, f"{nom_fichier}.png")
        fig.savefig(chemin)
        print(f"  [figure] {chemin}")
    plt.show()
    return fig


def bornes_strate(y, pcts=STRATES_PCT):
    """Retourne la liste [(min, max, nom), ...] des strates de magnitude."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    seuils = [np.percentile(y, p) for p in pcts]
    bornes = [-np.inf] + seuils + [np.inf]
    return [(bornes[i], bornes[i + 1], STRATES_NOM[i])
            for i in range(len(STRATES_NOM))]


def verifier_absence_de_log(fonctions=None, verbose=True):
    """GARDE-FOU. Leve une AssertionError si un logarithme reapparait.

    Interdits detectes : np.log, np.log1p, np.log10, math.log,
    set_xscale/set_yscale("log"|"symlog"), plotly type="log".

    Cette fonction est une PREUVE EXECUTABLE : tu peux la lancer devant ton
    responsable, elle inspecte le code source reellement charge en memoire,
    pas un commentaire ni une promesse.
    """
    interdits = ["np.log", "numpy.log", "math.log", "log1p", "log10",
                 '"log"', "'log'", "symlog", "logspace"]

    if fonctions is None:
        fonctions = [v for k, v in sorted(globals().items())
                     if callable(v) and (k.startswith("fig_")
                                         or k in ("cadrer_axe_y", "format_montant",
                                                  "finaliser", "bornes_strate"))]
    fautifs = []
    for f in fonctions:
        try:
            src = inspect.getsource(f)
        except (OSError, TypeError):
            continue
        # On ignore les lignes de commentaire : le mot "log" y est autorise
        lignes = [l for l in src.splitlines()
                  if not l.strip().startswith("#")]
        code = "\n".join(lignes)
        for mot in interdits:
            if mot in code:
                fautifs.append((getattr(f, "__name__", str(f)), mot))

    if fautifs:
        detail = "\n".join(f"   - {n} contient {m}" for n, m in fautifs)
        raise AssertionError(
            f"/!\\ LOGARITHME DETECTE dans le code de tracage :\n{detail}")

    if verbose:
        print("=" * 78)
        print("CONTROLE 'ZERO LOGARITHME'")
        print("=" * 78)
        print(f"  Fonctions inspectees : {len(fonctions)}")
        print("  Resultat             : AUCUN logarithme, ni sur les donnees,")
        print("                         ni sur les axes.")
        print("=" * 78)
    return True


# =============================================================================
#  BLOC 2 - CONTRAT DE DONNEES
# -----------------------------------------------------------------------------
#  Avant de tracer 18 figures, on verifie une fois pour toutes que les colonnes
#  attendues existent. Cela evite de decouvrir un KeyError a la figure 14 apres
#  huit minutes de calcul.
# =============================================================================

COLONNES_RESULTATS = ["y_obs", "y_pred", "borne_basse", "borne_haute",
                      "dans_intervalle"]
COLONNES_PRIO = ["score_composite", "A_ecart_borne", "B_erreur_modele", "rank"]


def verifier_contrat(results, anomalies_prio=None, results_pv=None,
                     verbose=True):
    """Controle la presence des colonnes et la coherence des intervalles."""
    manquantes = [c for c in COLONNES_RESULTATS if c not in results.columns]
    if manquantes:
        raise KeyError(f"Colonnes absentes de `results` : {manquantes}")

    d = results
    n = len(d)
    inversees = int((d["borne_haute"] < d["borne_basse"]).sum())
    incoherentes = int((
        (d["y_obs"] >= d["borne_basse"]) & (d["y_obs"] <= d["borne_haute"])
        != d["dans_intervalle"].astype(bool)).sum())
    couverture = float(d["dans_intervalle"].mean())

    if verbose:
        print("=" * 78)
        print("CONTRAT DE DONNEES")
        print("=" * 78)
        print(f"  Observations de test          : {n:>10,}")
        print(f"  Couverture empirique          : {couverture:>10.4f}"
              f"   (cible {1 - ALPHA:.2f})")
        print(f"  Ecart a la cible              : "
              f"{100 * (couverture - (1 - ALPHA)):>+9.2f} points")
        print(f"  Bornes inversees (hi < lo)    : {inversees:>10,}")
        print(f"  Indicateurs incoherents       : {incoherentes:>10,}")
        print(f"  Largeur mediane d'intervalle  : "
              f"{format_montant(np.median(d['borne_haute'] - d['borne_basse'])):>10}")
        print(f"  Amplitude de la cible         : "
              f"{format_montant(d['y_obs'].min())} -> "
              f"{format_montant(d['y_obs'].max())}")
        if anomalies_prio is not None:
            manq_p = [c for c in COLONNES_PRIO
                      if c not in anomalies_prio.columns]
            print(f"  Anomalies priorisees          : {len(anomalies_prio):>10,}"
                  + (f"   (colonnes manquantes : {manq_p})" if manq_p else ""))
        if results_pv is not None and "p_value" in results_pv.columns:
            print(f"  p-values conformes calculees  : "
                  f"{results_pv['p_value'].notna().sum():>10,}")
        print("=" * 78)

    if inversees:
        raise ValueError(
            f"{inversees} intervalles ont borne_haute < borne_basse. "
            "Corriger en amont (croisement des quantiles CQR).")
    return dict(n=n, couverture=couverture, incoherentes=incoherentes)


# #############################################################################
#  PARTIE A - COMPRENDRE LA CIBLE
# #############################################################################

# =============================================================================
#  FIG 01 - PROFIL DE LA CIBLE EN ECHELLE DE RANG
# -----------------------------------------------------------------------------
#  Remplace l'histogramme log-log de l'ancien BLOC 6.
#  Principe : on trace la FONCTION QUANTILE (valeurs triees en fonction de leur
#  rang en pourcentage). L'axe X est un rang, l'axe Y est un montant lineaire.
#  Aucune transformation : c'est un tri, operation strictement monotone et
#  totalement transparente.
# =============================================================================
def fig_01_profil_cible(results, colonne="y_obs", p_haut=99.0,
                        nom_fichier="fig_01_profil_cible"):
    y = np.sort(np.asarray(results[colonne], dtype=float))
    y = y[np.isfinite(y)]
    rang = np.linspace(0, 100, len(y))

    fig, ax = plt.subplots()
    ax.plot(rang, y, color=COULEURS["prediction"], lw=2.0,
            label="Valeur observee (triee)")
    ax.fill_between(rang, 0, y, color=COULEURS["bande_cl"], alpha=0.55)

    # Reperes de strates
    for p, style in zip(STRATES_PCT, ["--", "--", "-"]):
        v = np.percentile(y, p)
        ax.axvline(p, color=COULEURS["reference"], ls=style, lw=1.2, alpha=0.8)
        ax.text(p, ax.get_ylim()[1], f" P{p} = {format_montant(v)}",
                rotation=90, va="top", ha="left", fontsize=8.5,
                color=COULEURS["reference"])

    n_sup, _ = cadrer_axe_y(ax, y, p_haut=p_haut)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter(100))

    part_top1 = y[y >= np.percentile(y, 99)].sum() / y.sum()
    ax.text(0.03, 0.92,
            f"Le dernier centile concentre {part_top1:.1%} du montant total",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            color=COULEURS["anomalie"],
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=COULEURS["anomalie_cl"]))

    return finaliser(
        fig, ax,
        titre=f"Profil de la cible {TARGET} : une distribution a queue lourde",
        lecture="L'abscisse est le rang en pourcentage, l'ordonnee un montant "
                "en echelle lineaire. La courbe reste plate sur 90 % de la "
                "population puis decroche : c'est cette rupture qui justifie "
                "des intervalles de prediction adaptatifs plutot qu'un seuil fixe.",
        xlabel="Rang de l'observation (percentile)",
        ylabel=f"{TARGET} ({UNITE})",
        note="Aucune transformation logarithmique. Le cadrage est borne au "
             f"percentile {p_haut} ; les observations situees au-dela sont "
             "comptees explicitement en bas a droite.",
        format_y_montant=True, legende="upper left",
        nom_fichier=nom_fichier)


# =============================================================================
#  FIG 02 - CONCENTRATION DE LA CIBLE (COURBE DE LORENZ)
# -----------------------------------------------------------------------------
#  Figure entierement SANS ECHELLE : les deux axes sont des pourcentages
#  cumules. Le probleme d'amplitude disparait par construction, sans le moindre
#  artifice. C'est l'argument le plus solide a opposer a la demande de log.
# =============================================================================
def fig_02_concentration_cible(results, colonne="y_obs",
                               nom_fichier="fig_02_concentration"):
    y = np.sort(np.asarray(results[colonne], dtype=float))
    y = y[np.isfinite(y)]
    part_obs = np.linspace(0, 1, len(y))
    part_montant = np.concatenate([[0], np.cumsum(y) / y.sum()])[:len(y)]
    gini = 1 - 2 * np.trapz(part_montant, part_obs)

    fig, ax = plt.subplots()
    ax.plot(part_obs, part_montant, color=COULEURS["prediction"], lw=2.4,
            label=f"Concentration observee (Gini = {gini:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1.4, color=COULEURS["reference"],
            label="Repartition parfaitement egale")
    ax.fill_between(part_obs, part_montant, part_obs,
                    color=COULEURS["bande_cl"], alpha=0.6)

    for seuil in (0.90, 0.99):
        idx = int(seuil * (len(y) - 1))
        reste = 1 - part_montant[idx]
        ax.plot([seuil, seuil], [part_montant[idx], seuil], color=COULEURS["neutre"],
                lw=1.0, ls=":")
        ax.annotate(f"Les {100 * (1 - seuil):.0f} % les plus eleves\n"
                    f"portent {reste:.1%} du montant",
                    xy=(seuil, part_montant[idx]),
                    xytext=(seuil - 0.42, part_montant[idx] + 0.16),
                    fontsize=9, color=COULEURS["anomalie"],
                    arrowprops=dict(arrowstyle="->", color=COULEURS["anomalie"],
                                    lw=1.1))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    return finaliser(
        fig, ax,
        titre="Concentration du montant : ou se joue reellement l'enjeu de controle",
        lecture="Plus la courbe s'ecarte de la diagonale, plus le montant total "
                "est concentre sur un petit nombre d'unites statistiques. "
                "Consequence directe pour la priorisation : controler au hasard "
                "revient a passer a cote de l'essentiel de l'exposition.",
        xlabel="Part cumulee des unites statistiques (triees par montant croissant)",
        ylabel=f"Part cumulee du montant total de {TARGET}",
        note="Les deux axes sont des pourcentages cumules : cette figure est "
             "invariante par changement d'echelle, elle ne peut structurellement "
             "pas necessiter de logarithme.",
        legende="upper left", nom_fichier=nom_fichier)


# #############################################################################
#  PARTIE B - VALIDITE DE LA METHODE CONFORME
#  Diagnostics standards de la litterature :
#    - Vovk, Gammerman & Shafer (2005), p-values conformes
#    - Bates, Candes, Lei, Romano, Sesia (2023), "Testing for Outliers with
#      Conformal p-values", Annals of Statistics
#    - Angelopoulos & Bates (2023), "Conformal Prediction: A Gentle
#      Introduction" -> Size-Stratified Coverage
#    - Benjamini & Hochberg (1995), controle du FDR
# #############################################################################

# =============================================================================
#  FIG 03 - COUVERTURE EMPIRIQUE VS COUVERTURE NOMINALE
# -----------------------------------------------------------------------------
#  LA figure de validation. Si la courbe suit la diagonale, la garantie
#  theorique de la Conformal Prediction est verifiee sur TES donnees.
# =============================================================================
def fig_03_calibration(results_pv, calib_scores, alphas=None,
                       nom_fichier="fig_03_calibration"):
    if "score_nonconformite" not in results_pv.columns:
        raise KeyError("`results_pv` doit contenir 'score_nonconformite'. "
                       "Voir la cellule 14a du pipeline.")
    s_test = np.asarray(results_pv["score_nonconformite"], dtype=float)
    cal = np.sort(np.asarray(calib_scores, dtype=float))
    n_cal = len(cal)
    alphas = np.linspace(0.01, 0.30, 30) if alphas is None else np.asarray(alphas)

    cov = []
    for a in alphas:
        niv = min(np.ceil((n_cal + 1) * (1 - a)) / n_cal, 1.0)
        q_a = np.quantile(cal, niv, method="higher")
        cov.append(float(np.mean(s_test <= q_a)))
    cov = np.array(cov)
    nominal = 1 - alphas

    # Bande de tolerance binomiale a 95 % : la couverture empirique fluctue
    n_test = len(s_test)
    demi = 1.96 * np.sqrt(nominal * (1 - nominal) / n_test)

    fig, ax = plt.subplots()
    ax.fill_between(nominal, nominal - demi, nominal + demi,
                    color=COULEURS["neutre_cl"], alpha=0.8,
                    label="Tolerance a 95 % (fluctuation d'echantillonnage)")
    ax.plot(nominal, nominal, ls="--", lw=1.5, color=COULEURS["reference"],
            label="Calibration parfaite")
    ax.plot(nominal, cov, "o-", ms=5, lw=2.0, color=COULEURS["prediction"],
            label="Couverture empirique mesuree sur le test")
    ax.axvline(1 - ALPHA, color=COULEURS["anomalie"], ls=":", lw=1.8,
               label=f"Niveau retenu : {100 * (1 - ALPHA):.0f} %")

    ecart_max = float(np.max(np.abs(cov - nominal)))
    ax.text(0.03, 0.90,
            f"Ecart maximal a la cible : {100 * ecart_max:.2f} points",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=COULEURS["neutre_cl"]))

    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    return finaliser(
        fig, ax,
        titre="Validation de la calibration conforme",
        lecture="Chaque point compare le taux de couverture promis par la theorie "
                "(abscisse) au taux reellement observe sur l'echantillon de test "
                "(ordonnee). Une courbe confondue avec la diagonale, dans la bande "
                "de tolerance, valide la garantie de couverture marginale.",
        xlabel="Couverture nominale (1 - alpha)",
        ylabel="Couverture empirique constatee",
        note=f"Calibration : {n_cal:,} observations. Test : {n_test:,} observations. "
             "La bande grise correspond a l'incertitude binomiale a 95 %, elle "
             "evite de sur-interpreter un ecart de quelques dixiemes de point.",
        legende="lower right", nom_fichier=nom_fichier)


# =============================================================================
#  FIG 04 - HISTOGRAMME DES P-VALUES CONFORMES
# -----------------------------------------------------------------------------
#  Sous l'hypothese "aucune anomalie", les p-values conformes sont uniformes
#  sur [0,1]. Un pic a gauche est donc la SIGNATURE d'anomalies reelles.
#  Aucun probleme d'echelle ici : une p-value vit dans [0,1] par construction.
# =============================================================================
def fig_04_histogramme_pvalues(results_pv, n_bins=20, lam=0.5,
                               nom_fichier="fig_04_pvalues_hist"):
    p = np.asarray(results_pv["p_value"], dtype=float)
    p = p[np.isfinite(p)]
    m = len(p)

    # Estimateur de Storey de la proportion d'observations saines
    pi0 = float(min(np.sum(p > lam) / (m * (1 - lam)), 1.0))
    n_vraies = m * (1 - pi0)

    fig, ax = plt.subplots()
    effectifs, bords, patches = ax.hist(
        p, bins=n_bins, range=(0, 1), color=COULEURS["bande"],
        edgecolor="white", linewidth=1.0, label="p-values conformes observees")

    # Le premier bin est celui qui porte le signal : on le met en evidence
    patches[0].set_facecolor(COULEURS["anomalie"])
    patches[0].set_label("Premier bin : concentration du signal")

    attendu = m / n_bins
    ax.axhline(attendu, color=COULEURS["reference"], ls="--", lw=2.0,
               label="Effectif attendu si aucune anomalie (loi uniforme)")

    exces = int(effectifs[0] - attendu)
    ax.annotate(f"Exces de {exces:,} observations\ndans le premier bin",
                xy=(bords[1] / 2, effectifs[0]),
                xytext=(0.28, effectifs[0] * 0.82),
                fontsize=9.5, color=COULEURS["anomalie"],
                arrowprops=dict(arrowstyle="->", color=COULEURS["anomalie"]))

    ax.text(0.97, 0.94,
            f"Proportion estimee d'observations saines : {pi0:.3f}\n"
            f"Nombre estime de vraies anomalies : {n_vraies:,.0f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", fc="#FFFBE6",
                      ec=COULEURS["reference"], alpha=0.95))

    ax.set_xlim(0, 1)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    return finaliser(
        fig, ax,
        titre="Distribution des p-values conformes : y a-t-il un signal ?",
        lecture="Si aucune observation n'etait atypique, les p-values seraient "
                "uniformement reparties et l'histogramme serait plat au niveau "
                "de la ligne orange. Toute masse excedentaire pres de zero "
                "correspond a des observations que le modele ne peut pas "
                "expliquer par la seule variabilite normale.",
        xlabel="p-value conforme",
        ylabel="Nombre d'observations",
        note="Estimateur de Storey (lam = %.2f) : pi0 = #{p > lam} / (m(1-lam)). "
             "Reference : Bates, Candes, Lei, Romano & Sesia, "
             "\"Testing for Outliers with Conformal p-values\"." % lam,
        legende="upper right", nom_fichier=nom_fichier)


# =============================================================================
#  FIG 05 - QQ-PLOT DES P-VALUES CONTRE LA LOI UNIFORME
# -----------------------------------------------------------------------------
#  Version "cumulative" de la figure precedente : plus sensible dans la zone
#  des petites p-values, qui est justement celle qui nous interesse.
# =============================================================================
def fig_05_qqplot_pvalues(results_pv, nom_fichier="fig_05_qqplot_pvalues"):
    p = np.sort(np.asarray(results_pv["p_value"], dtype=float))
    p = p[np.isfinite(p)]
    m = len(p)
    theorique = (np.arange(1, m + 1) - 0.5) / m

    # Enveloppe de confiance ponctuelle a 95 % via la loi Beta des statistiques
    # d'ordre de l'uniforme : ordre i ~ Beta(i, m-i+1).
    from scipy.stats import beta as loi_beta
    i = np.arange(1, m + 1)
    bas = loi_beta.ppf(0.025, i, m - i + 1)
    haut = loi_beta.ppf(0.975, i, m - i + 1)

    fig, ax = plt.subplots()
    ax.fill_between(theorique, bas, haut, color=COULEURS["neutre_cl"],
                    alpha=0.85, label="Enveloppe a 95 % sous l'hypothese uniforme")
    ax.plot([0, 1], [0, 1], ls="--", lw=1.5, color=COULEURS["reference"],
            label="Aucune anomalie (loi uniforme)")
    ax.plot(theorique, p, lw=2.2, color=COULEURS["prediction"],
            label="p-values conformes observees")

    # Mise en evidence de la zone de decision
    sous = p < bas
    if sous.any():
        ax.scatter(theorique[sous], p[sous], s=12, color=COULEURS["anomalie"],
                   zorder=6, label=f"Sous l'enveloppe (n={sous.sum():,})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return finaliser(
        fig, ax,
        titre="QQ-plot des p-values conformes contre la loi uniforme",
        lecture="Une courbe qui decroche SOUS la diagonale dans la partie gauche "
                "signifie qu'il y a davantage de tres petites p-values que ne le "
                "permettrait le hasard : c'est la preuve statistique que des "
                "observations atypiques sont presentes, et non un simple artefact "
                "du niveau alpha retenu.",
        xlabel="Quantiles theoriques de la loi uniforme",
        ylabel="p-values conformes triees",
        note="Enveloppe construite a partir des lois Beta(i, m-i+1) des "
             "statistiques d'ordre de l'uniforme. Les deux axes vivent dans "
             "[0,1] : aucune question d'echelle ne se pose.",
        legende="lower right", nom_fichier=nom_fichier)


# =============================================================================
#  FIG 06 - PROCEDURE DE BENJAMINI-HOCHBERG
# -----------------------------------------------------------------------------
#  Repond a la question que pose toujours un tuteur : "sur vos N alertes,
#  combien sont du bruit ?" Reponse : au plus q en esperance.
# =============================================================================
def controle_fdr_bh(pvalues, q=0.05):
    """Benjamini-Hochberg. Retourne (masque_retenu, nb_retenu, seuil_effectif)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool), 0, 0.0
    ordre = np.argsort(p)
    p_tri = p[ordre]
    passe = p_tri <= q * np.arange(1, m + 1) / m
    k = int(np.max(np.where(passe)[0]) + 1) if passe.any() else 0
    retenu = np.zeros(m, dtype=bool)
    if k:
        retenu[ordre[:k]] = True
    return retenu, k, (p_tri[k - 1] if k else 0.0)


def fig_06_benjamini_hochberg(results_pv, q=0.05, k_max=400,
                              nom_fichier="fig_06_benjamini_hochberg"):
    p = np.asarray(results_pv["p_value"], dtype=float)
    p = p[np.isfinite(p)]
    m = len(p)
    p_tri = np.sort(p)
    retenu, n_bh, seuil = controle_fdr_bh(p, q=q)

    k_aff = int(min(m, max(k_max, 2 * max(n_bh, 1))))
    rangs = np.arange(1, k_aff + 1)
    droite = q * rangs / m

    fig, ax = plt.subplots()
    couleurs_pts = np.where(p_tri[:k_aff] <= droite,
                            COULEURS["anomalie"], COULEURS["neutre"])
    ax.scatter(rangs, p_tri[:k_aff], s=14, c=couleurs_pts, zorder=5,
               label="p-values triees par ordre croissant")
    ax.plot(rangs, droite, lw=2.0, color=COULEURS["reference"],
            label=f"Droite de rejet Benjamini-Hochberg (q = {q:.0%})")

    if n_bh:
        ax.axvline(n_bh, color=COULEURS["prediction"], ls="--", lw=1.8)
        ax.annotate(f"{n_bh:,} anomalies retenues\n"
                    f"faux positifs attendus : {q * n_bh:.1f}",
                    xy=(n_bh, droite[min(n_bh, k_aff) - 1]),
                    xytext=(n_bh * 1.15, np.percentile(p_tri[:k_aff], 60)),
                    fontsize=9.5, color=COULEURS["prediction"],
                    arrowprops=dict(arrowstyle="->",
                                    color=COULEURS["prediction"]))
    else:
        ax.text(0.5, 0.5, "Aucune anomalie ne survit au controle du FDR",
                transform=ax.transAxes, ha="center", fontsize=12,
                color=COULEURS["anomalie"], fontweight="bold")

    ax.set_xlim(0, k_aff)
    ax.set_ylim(0, max(float(p_tri[:k_aff].max()), float(droite.max())) * 1.05)

    return finaliser(
        fig, ax,
        titre="Controle du taux de fausses decouvertes (Benjamini-Hochberg)",
        lecture="Chaque point est une observation, classee de la plus surprenante "
                "a la moins surprenante. Toutes celles situees sous la droite "
                "orange sont retenues comme anomalies : la procedure garantit "
                f"qu'au plus {q:.0%} d'entre elles sont des fausses alertes, en esperance.",
        xlabel="Rang de la p-value (ordre croissant)",
        ylabel="p-value conforme",
        note=f"Seuil de p-value effectif : {seuil:.3e}. Reference : Benjamini & "
             "Hochberg (1995). Le controle du FDR est ce qui transforme une liste "
             "d'ecarts en une liste d'alertes defendable devant un controleur.",
        legende="upper left", nom_fichier=nom_fichier)


# =============================================================================
#  FIG 07 - COUVERTURE PAR DECILE DE LARGEUR (SIZE-STRATIFIED COVERAGE)
# -----------------------------------------------------------------------------
#  Diagnostic d'ADAPTATIVITE. Une methode peut tenir sa couverture globale de
#  90 % tout en etant catastrophique sur les intervalles etroits. Cette figure
#  le revele. C'est le diagnostic que le jury connait le mieux apres MAPIE.
# =============================================================================
def fig_07_couverture_par_largeur(results, n_deciles=10,
                                  nom_fichier="fig_07_ssc"):
    d = results.copy()
    if "largeur_intervalle" in d.columns:
        larg = d["largeur_intervalle"].to_numpy(dtype=float)
    else:
        larg = (d["borne_haute"] - d["borne_basse"]).to_numpy(dtype=float)

    tab = pd.DataFrame({"largeur": larg,
                        "dedans": d["dans_intervalle"].to_numpy(dtype=bool)})
    tab["bin"] = pd.qcut(tab["largeur"], n_deciles, labels=False,
                         duplicates="drop")
    agg = tab.groupby("bin", observed=True).agg(
        couverture=("dedans", "mean"),
        effectif=("dedans", "size"),
        largeur_med=("largeur", "median")).reset_index()

    cible = 1 - ALPHA
    # Intervalle de Wald a 95 % par decile
    err = 1.96 * np.sqrt(agg["couverture"] * (1 - agg["couverture"])
                         / agg["effectif"])

    couleurs = np.where(np.abs(agg["couverture"] - cible) > 0.05,
                        COULEURS["anomalie"], COULEURS["conforme"])

    fig, ax = plt.subplots()
    ax.bar(agg["bin"], agg["couverture"], color=couleurs, edgecolor="white",
           linewidth=1.2, width=0.75, label="Couverture observee par decile")
    ax.errorbar(agg["bin"], agg["couverture"], yerr=err, fmt="none",
                ecolor=COULEURS["prediction"], elinewidth=1.2, capsize=4,
                label="Intervalle de confiance a 95 %")
    ax.axhline(cible, color=COULEURS["reference"], ls="--", lw=2.0,
               label=f"Couverture cible : {cible:.0%}")
    ax.axhspan(cible - 0.05, cible + 0.05, color=COULEURS["reference"],
               alpha=0.10, label="Tolerance de +/- 5 points")

    # Etiquette placee A L'INTERIEUR de la barre : au-dessus, elle entrerait en
    # collision avec la barre d'erreur.
    for _, r in agg.iterrows():
        ax.text(r["bin"], r["couverture"] - 0.055, f"{r['couverture']:.0%}",
                ha="center", va="center", fontsize=9.5, color="white",
                fontweight="bold")

    ecart_max = float(np.abs(agg["couverture"] - cible).max())
    ax.text(0.02, 0.06,
            f"Violation maximale : {100 * ecart_max:.1f} points",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            color=COULEURS["anomalie"] if ecart_max > 0.05 else COULEURS["conforme"],
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=COULEURS["neutre_cl"]))

    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xticks(agg["bin"])
    ax.set_xticklabels([f"D{int(b) + 1}\n{format_montant(l)}"
                        for b, l in zip(agg["bin"], agg["largeur_med"])],
                       fontsize=8)

    return finaliser(
        fig, ax,
        titre="Couverture conditionnelle par decile de largeur d'intervalle",
        lecture="La garantie conforme est marginale : elle porte sur la moyenne. "
                "Cette figure verifie qu'elle tient aussi sur chaque sous-groupe. "
                "Des barres alignees sur la cible signifient que les intervalles "
                "etroits sont aussi honnetes que les larges, donc qu'une alerte "
                "sur un petit portefeuille vaut autant qu'une alerte sur un gros.",
        xlabel="Decile de largeur d'intervalle (etroit -> large) et largeur mediane",
        ylabel="Couverture empirique",
        note="Size-Stratified Coverage, cf. Angelopoulos & Bates (2023). "
             "Une barre rouge signale un decile dont la couverture s'ecarte de "
             "plus de 5 points de la cible.",
        legende="lower right", nom_fichier=nom_fichier)


# =============================================================================
#  FIG 08 - DISTRIBUTION DES LARGEURS D'INTERVALLE, EN ECHELLE DE RANG
# -----------------------------------------------------------------------------
#  Remplace l'histogramme log de l'ancienne planche.
#  L'adaptativite se lit ici comme une pente : plus la courbe monte, plus le
#  modele module son incertitude selon le profil de l'unite statistique.
# =============================================================================
def fig_08_largeurs(results, p_haut=99.0, nom_fichier="fig_08_largeurs"):
    if "largeur_intervalle" in results.columns:
        larg = results["largeur_intervalle"].to_numpy(dtype=float)
    else:
        larg = (results["borne_haute"] - results["borne_basse"]).to_numpy(float)
    larg = np.sort(larg[np.isfinite(larg)])
    rang = np.linspace(0, 100, len(larg))

    fig, ax = plt.subplots()
    ax.plot(rang, larg, lw=2.2, color=COULEURS["bande"],
            label="Largeur d'intervalle (triee)")
    ax.fill_between(rang, 0, larg, color=COULEURS["bande_cl"], alpha=0.6)

    med = float(np.median(larg))
    ax.axhline(med, color=COULEURS["reference"], ls="--", lw=1.4,
               label=f"Largeur mediane : {format_montant(med)}")

    for p in (25, 50, 75, 90):
        ax.plot([p, p], [0, np.percentile(larg, p)], ls=":", lw=1.0,
                color=COULEURS["neutre"])
        ax.text(p, np.percentile(larg, p), f" P{p}", fontsize=8,
                va="bottom", color=COULEURS["neutre"])

    ratio = float(np.percentile(larg, 90) / max(np.percentile(larg, 10), 1e-9))
    ax.text(0.03, 0.90,
            f"Rapport P90 / P10 des largeurs : {ratio:,.1f}\n"
            "Un rapport eleve = intervalles fortement adaptatifs",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=COULEURS["neutre_cl"]))

    cadrer_axe_y(ax, larg, p_haut=p_haut)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter(100))

    return finaliser(
        fig, ax,
        titre="Adaptativite des intervalles : distribution des largeurs",
        lecture="Chaque abscisse est un rang, chaque ordonnee une largeur "
                "d'intervalle en euros. Une courbe croissante et etalee prouve "
                "que la methode ne produit pas une bande de largeur constante : "
                "elle resserre la ou elle est sure, elle elargit la ou l'unite "
                "statistique est intrinsequement volatile.",
        xlabel="Rang de l'unite statistique (percentile de largeur)",
        ylabel=f"Largeur de l'intervalle conforme ({UNITE})",
        note="Le rapport P90/P10 est un indicateur d'adaptativite : une methode "
             "non adaptative (conformal split classique sur residus absolus) "
             "donnerait un rapport egal a 1.",
        format_y_montant=True, legende="upper left", nom_fichier=nom_fichier)


# #############################################################################
#  PARTIE C - LA BANDE CONFORME
#  C'est le coeur visuel du memoire : montrer l'intervalle, la prediction et
#  l'observation, et faire ressortir en rouge ce qui sort de la bande.
# #############################################################################

# =============================================================================
#  FIG 09 - BANDE CONFORME NORMALISEE, POPULATION ENTIERE
# -----------------------------------------------------------------------------
#  LA figure qui remplace definitivement l'echelle logarithmique.
#
#  Principe :  z = (y_obs - centre_intervalle) / demi_largeur_intervalle
#  Lecture  :  |z| <= 1  -> dans l'intervalle       (vert)
#              |z| >  1  -> hors intervalle, anomalie (rouge)
#
#  Pourquoi ca resout le probleme : chaque observation est ramenee a SA propre
#  echelle d'incertitude. Une ligne a 0,7 EUR et une ligne a 317 MEUR se lisent
#  sur le meme axe, sans qu'aucune fonction non lineaire n'intervienne. C'est
#  une simple division, exactement comme un ratio de solvabilite.
# =============================================================================
def fig_09_bande_normalisee(results, z_max=6.0, echantillon=None,
                            random_state=42,
                            nom_fichier="fig_09_bande_normalisee"):
    d = results.copy()
    if echantillon and echantillon < len(d):
        d = d.sample(echantillon, random_state=random_state)

    centre = (d["borne_haute"].to_numpy(float) + d["borne_basse"].to_numpy(float)) / 2
    demi = np.maximum((d["borne_haute"].to_numpy(float)
                       - d["borne_basse"].to_numpy(float)) / 2, 1e-9)
    z = (d["y_obs"].to_numpy(float) - centre) / demi
    dedans = d["dans_intervalle"].to_numpy(bool)

    ordre = np.argsort(d["y_pred"].to_numpy(float))
    z, dedans = z[ordre], dedans[ordre]
    x = np.arange(len(z))

    # Cadrage asymetrique. Sur une cible positive, l'observation ne peut pas
    # descendre indefiniment sous l'intervalle : z est borne par le bas mais
    # pas par le haut. Un cadre symetrique gaspillerait donc la moitie basse
    # de la figure. On adapte chaque bord a la distribution reelle de z, tout
    # en garantissant que la zone conforme [-1, 1] reste visible.
    z_bas = float(np.clip(min(-1.3, np.percentile(z, 0.2)), -z_max, -1.3))
    z_haut = float(np.clip(max(1.3, np.percentile(z, 99.8)), 1.3, z_max))

    # Les valeurs au-dela du cadre sont ramenees au bord et signalees par un
    # triangle : leur position en abscisse reste lisible, aucune n'est effacee.
    z_aff = np.clip(z, z_bas, z_haut)
    satures = (z > z_haut) | (z < z_bas)

    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    ax.axhspan(-1, 1, color=COULEURS["conforme_cl"], alpha=0.55,
               label=f"Zone conforme a {100 * (1 - ALPHA):.0f} % (|z| <= 1)")
    ax.axhline(0, color=COULEURS["prediction"], ls="--", lw=1.2,
               label="Centre de l'intervalle")
    for b in (-1, 1):
        ax.axhline(b, color=COULEURS["bande"], lw=1.3)

    ax.scatter(x[dedans], z_aff[dedans], s=7, color=COULEURS["conforme"],
               alpha=0.45,
               label=f"Dans l'intervalle : {100 * dedans.mean():.1f} %")
    ax.scatter(x[~dedans & ~satures], z_aff[~dedans & ~satures], s=26,
               color=COULEURS["anomalie"], alpha=0.9, zorder=5,
               label=f"Hors intervalle : {100 * (~dedans).mean():.1f} % "
                     f"(n={(~dedans).sum():,})")
    if satures.any():
        ax.scatter(x[satures], z_aff[satures],
                   marker="^", s=60, color=COULEURS["accent"], zorder=6,
                   edgecolor="white", linewidth=0.6,
                   label=f"Ecart au-dela du cadre, ramene au bord "
                         f"(n={satures.sum():,})")

    # Marge superieure elargie : elle reserve la place de l'encadre de synthese
    # sans que celui-ci ne recouvre les points extremes.
    marge = 0.10 * (z_haut - z_bas)
    ax.set_ylim(z_bas - marge, z_haut + 2.6 * marge)
    ax.set_xlim(0, len(z))

    depassement_median = float(np.median(np.abs(z[~dedans]))) if (~dedans).any() else np.nan
    ax.text(0.015, 0.955,
            f"Couverture : {100 * dedans.mean():.2f} %  "
            f"(cible {100 * (1 - ALPHA):.0f} %)\n"
            f"Depassement median des anomalies : |z| = {depassement_median:.2f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", fc="white",
                      ec=COULEURS["neutre_cl"], alpha=0.95))

    return finaliser(
        fig, ax,
        titre="Bande conforme normalisee sur l'ensemble du portefeuille de test",
        lecture="Chaque observation est ramenee a sa propre echelle d'incertitude : "
                "z mesure l'ecart au centre de l'intervalle, exprime en demi-largeurs. "
                "La bande verte est la zone conforme. Tout point rouge est une "
                "observation que le modele n'explique pas, quel que soit son montant.",
        xlabel="Unites statistiques, triees par prediction croissante",
        ylabel="Position normalisee z = (observation - centre) / demi-largeur",
        note="Cette normalisation est une division par la demi-largeur de "
             "l'intervalle. Elle rend comparables des lignes de 0,7 EUR et de "
             "317 MEUR sans recourir a la moindre echelle logarithmique : "
             "c'est le meme principe qu'un ratio rapporte a son exigence.",
        legende="lower right", nom_fichier=nom_fichier)


# =============================================================================
#  FIG 10 - BANDE CONFORME ABSOLUE, UNE STRATE DE MAGNITUDE PAR APPEL
# -----------------------------------------------------------------------------
#  Version en euros, pour ceux qui veulent voir les montants reels.
#  UN APPEL = UNE STRATE = UNE FIGURE. Aucune grille 2x2.
#  A l'interieur d'une strate, toutes les valeurs sont du meme ordre de
#  grandeur : l'axe lineaire est naturellement lisible.
# =============================================================================
def fig_10_bande_strate(results, strate="P50-90", max_points=1200,
                        random_state=42, nom_fichier=None):
    """Trace la bande conforme en euros pour UNE strate de magnitude.

    strate : "P0-50" | "P50-90" | "P90-99" | "P99+"
    Appeler la fonction quatre fois produit quatre figures independantes,
    chacune destinee a sa propre slide.

    POINT METHODOLOGIQUE. Les strates sont definies sur la PREDICTION, pas sur
    la valeur observee. Decouper selon y_obs reviendrait a conditionner sur le
    resultat que l'on cherche a tester : les strates hautes concentreraient
    mecaniquement les depassements par le haut et les strates basses les
    depassements par le bas, ce qui ferait apparaitre des taux d'anomalie tres
    superieurs a alpha sans qu'aucune anomalie reelle ne soit en cause.
    Decouper selon y_pred utilise une information disponible AVANT l'observation
    et preserve donc l'interpretation du taux d'anomalie dans chaque strate.
    """
    bornes = {n: (lo, hi) for lo, hi, n in bornes_strate(results["y_pred"])}
    if strate not in bornes:
        raise ValueError(f"Strate inconnue : {strate}. Valeurs : {list(bornes)}")
    lo_s, hi_s = bornes[strate]

    d = results[(results["y_pred"] >= lo_s) & (results["y_pred"] < hi_s)].copy()
    if d.empty:
        raise ValueError(f"Strate {strate} vide.")
    if len(d) > max_points:
        d = d.sample(max_points, random_state=random_state)
    d = d.sort_values("y_pred").reset_index(drop=True)

    x = np.arange(len(d))
    obs = d["y_obs"].to_numpy(float)
    pred = d["y_pred"].to_numpy(float)
    lo = d["borne_basse"].to_numpy(float)
    hi = d["borne_haute"].to_numpy(float)
    dedans = d["dans_intervalle"].to_numpy(bool)

    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    ax.fill_between(x, lo, hi, color=COULEURS["bande_cl"], alpha=0.85,
                    label=f"Intervalle conforme a {100 * (1 - ALPHA):.0f} %")
    ax.plot(x, lo, color=COULEURS["bande"], lw=1.0, ls="--")
    ax.plot(x, hi, color=COULEURS["bande"], lw=1.0, ls="--")
    ax.plot(x, pred, color=COULEURS["prediction"], lw=1.8,
            label="Prediction du modele")
    ax.scatter(x[dedans], obs[dedans], s=13, color=COULEURS["conforme"],
               alpha=0.6, label=f"Dans l'intervalle (n={dedans.sum():,})")
    ax.scatter(x[~dedans], obs[~dedans], s=42, color=COULEURS["anomalie"],
               edgecolor="white", linewidth=0.6, zorder=6,
               label=f"Anomalie (n={(~dedans).sum():,})")

    # Cadrage sur l'ensemble bande + observations de la strate
    ensemble = np.concatenate([lo, hi, obs, pred])
    cadrer_axe_y(ax, ensemble, p_haut=99.0)
    marquer_hors_cadre(ax, x, obs, ax.get_ylim()[1])
    ax.set_xlim(0, max(len(d) - 1, 1))

    taux = 100 * (~dedans).mean()
    # Encadre place sous la legende (elle occupe le coin superieur gauche) et
    # a gauche de la bande, qui monte vers la droite : cette zone reste libre.
    ax.text(0.015, 0.70,
            f"Strate definie sur la prediction : "
            f"{format_montant(lo_s if np.isfinite(lo_s) else pred.min())}"
            f" -> {format_montant(hi_s if np.isfinite(hi_s) else pred.max())}\n"
            f"Taux d'anomalie dans la strate : {taux:.2f} %  "
            f"(attendu {100 * ALPHA:.0f} %)",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", fc="white",
                      ec=COULEURS["neutre_cl"], alpha=0.95))

    return finaliser(
        fig, ax,
        titre=f"Bande conforme en euros - strate de magnitude {strate}",
        lecture="Le decoupage en strates remplace l'echelle logarithmique : a "
                "l'interieur d'une strate, toutes les valeurs sont du meme ordre "
                "de grandeur, donc l'axe lineaire est directement lisible. Les "
                "observations sont triees par prediction croissante, ce qui fait "
                "monter regulierement la bande.",
        xlabel="Unites statistiques de la strate, triees par prediction croissante",
        ylabel=f"{TARGET} ({UNITE})",
        note="Aucune transformation des donnees : uniquement un filtre sur les "
             "lignes avant tracage. Les strates sont definies sur la prediction "
             "et non sur la valeur observee, afin de ne pas conditionner le "
             "decoupage sur le resultat teste.",
        format_y_montant=True, legende="upper left",
        nom_fichier=nom_fichier or f"fig_10_bande_strate_{strate.replace('+', 'plus')}")


# =============================================================================
#  FIG 11 - BANDE CONFORME DETAILLEE SUR N UNITES ETIQUETEES
# -----------------------------------------------------------------------------
#  Le style "publication CP" de tes images de reference : peu de points, une
#  etiquette lisible par unite, l'intervalle materialise par une barre.
#  Deux modes, mais UN SEUL graphique par appel.
# =============================================================================
def fig_11_bande_detaillee(results, n=30, mode="echantillon",
                           random_state=42, cle_lisible=None,
                           nom_fichier=None):
    """mode = "echantillon" : tirage aleatoire, taux d'anomalie REEL (honnete)
       mode = "anomalies"   : moitie pires anomalies, pedagogique mais
                              non representatif -> mentionne dans le titre.
    """
    d = results.copy()

    if cle_lisible is None:
        dispo = [c for c in ID_COLS if c in d.columns]
        d["_cle"] = d[dispo].astype(str).agg(" | ".join, axis=1).str.slice(0, 30)
    else:
        d["_cle"] = d[cle_lisible].astype(str).str.slice(0, 30)

    if mode == "anomalies":
        ano = d[~d["dans_intervalle"].astype(bool)].copy()
        ecart = np.where(ano["y_obs"] > ano["borne_haute"],
                         ano["y_obs"] - ano["borne_haute"],
                         ano["borne_basse"] - ano["y_obs"])
        ano = ano.assign(_e=ecart).nlargest(min(n // 2, len(ano)), "_e")
        norm = d[d["dans_intervalle"].astype(bool)]
        norm = norm.sample(min(n - len(ano), len(norm)), random_state=random_state)
        sel = pd.concat([ano.drop(columns="_e"), norm])
        suffixe = " (anomalies volontairement sur-representees)"
    else:
        sel = d.sample(min(n, len(d)), random_state=random_state)
        suffixe = " (echantillon aleatoire, proportion reelle)"

    sel = sel.sort_values("y_pred").reset_index(drop=True)
    x = np.arange(len(sel))
    obs = sel["y_obs"].to_numpy(float)
    pred = sel["y_pred"].to_numpy(float)
    lo = sel["borne_basse"].to_numpy(float)
    hi = sel["borne_haute"].to_numpy(float)
    dedans = sel["dans_intervalle"].to_numpy(bool)

    fig, ax = plt.subplots(figsize=(13.5, 7.2))

    # Intervalle materialise par une barre verticale par unite : beaucoup plus
    # lisible qu'un fill_between quand les largeurs varient fortement.
    ax.vlines(x, lo, hi, color=COULEURS["bande"], lw=7, alpha=0.45,
              label=f"Intervalle conforme a {100 * (1 - ALPHA):.0f} %")

    # Trait pointille rouge : de la borne franchie jusqu'a l'observation
    for xi, o, l, h, dd in zip(x, obs, lo, hi, dedans):
        if not dd:
            cible = h if o > h else l
            ax.plot([xi, xi], [cible, o], color=COULEURS["anomalie"],
                    lw=1.6, ls=":", zorder=4)

    ax.scatter(x, pred, marker="D", s=42, facecolor="white",
               edgecolor=COULEURS["prediction"], linewidth=1.4, zorder=5,
               label="Prediction du modele")
    ax.scatter(x[dedans], obs[dedans], s=80, color=COULEURS["conforme"],
               edgecolor="white", linewidth=0.9, zorder=6,
               label=f"Dans l'intervalle (n={dedans.sum()})")
    ax.scatter(x[~dedans], obs[~dedans], s=110, color=COULEURS["anomalie"],
               edgecolor="white", linewidth=0.9, zorder=7,
               label=f"Hors intervalle (n={(~dedans).sum()})")

    ensemble = np.concatenate([lo, hi, obs, pred])
    cadrer_axe_y(ax, ensemble, p_haut=99.0)
    ax.set_xticks(x)
    ax.set_xticklabels(sel["_cle"].tolist(), rotation=90, fontsize=7.5)
    ax.set_xlim(-0.7, len(sel) - 0.3)

    return finaliser(
        fig, ax,
        titre=f"Intervalles conformes et valeurs comptabilisees - {len(sel)} unites{suffixe}",
        lecture="Chaque colonne est une unite statistique. La barre bleue est son "
                "intervalle de prediction conforme, le losange la valeur attendue "
                "par le modele, le point la valeur reellement comptabilisee. Un "
                "point rouge relie par un pointille sort de son intervalle : "
                "c'est une observation a instruire.",
        xlabel="Unites statistiques, triees par prediction croissante",
        ylabel=f"{TARGET} ({UNITE})",
        note="Mode 'anomalies' : la proportion de rouge est volontairement "
             "amplifiee a des fins pedagogiques et ne doit pas etre lue comme un "
             "taux d'anomalie. Le taux reel figure sur la figure 09.",
        format_y_montant=True, legende="upper left",
        nom_fichier=nom_fichier or f"fig_11_bande_detaillee_{mode}")


# =============================================================================
#  FIG 12 - PREDIT VS REEL EN ECHELLE DE RANG
# -----------------------------------------------------------------------------
#  Remplace le nuage log-log de l'ancien BLOC 6.
#  On compare les RANGS et non les valeurs : l'information "le modele classe-t-il
#  correctement les unites ?" est preservee, l'ecrasement disparait, et aucune
#  transformation n'est appliquee.
# =============================================================================
def fig_12_predit_vs_reel_rangs(results, echantillon=20000, random_state=42,
                                nom_fichier="fig_12_predit_vs_reel_rangs"):
    d = results.copy()
    if echantillon and echantillon < len(d):
        d = d.sample(echantillon, random_state=random_state)

    r_obs = d["y_obs"].rank(pct=True).to_numpy(float) * 100
    r_pred = d["y_pred"].rank(pct=True).to_numpy(float) * 100
    dedans = d["dans_intervalle"].to_numpy(bool)

    from scipy.stats import spearmanr
    rho = float(spearmanr(d["y_obs"], d["y_pred"]).statistic)

    fig, ax = plt.subplots(figsize=(8.6, 8.0))
    ax.plot([0, 100], [0, 100], ls="--", lw=1.5, color=COULEURS["reference"],
            label="Classement parfait", zorder=3)
    ax.scatter(r_pred[dedans], r_obs[dedans], s=5, alpha=0.18,
               color=COULEURS["neutre"], label="Dans l'intervalle")
    ax.scatter(r_pred[~dedans], r_obs[~dedans], s=16, alpha=0.75,
               color=COULEURS["anomalie"], zorder=5,
               label=f"Hors intervalle (n={(~dedans).sum():,})")

    # Bandes de tolerance sur le rang : +/- 10 points de percentile
    ax.fill_between([0, 100], [-10, 90], [10, 110],
                    color=COULEURS["conforme_cl"], alpha=0.25, zorder=2,
                    label="Ecart de classement inferieur a 10 points")

    ax.text(0.035, 0.94,
            f"Correlation de Spearman : {rho:.4f}",
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=COULEURS["neutre_cl"]))

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(PercentFormatter(100))
    ax.yaxis.set_major_formatter(PercentFormatter(100))

    return finaliser(
        fig, ax,
        titre="Qualite du classement : rang predit contre rang observe",
        lecture="Les deux axes sont des rangs en pourcentage, pas des montants. "
                "Cette representation mesure exactement ce qui compte pour un "
                "controle : le modele place-t-il les unites dans le bon ordre ? "
                "Les points rouges eloignes de la diagonale sont ceux dont le "
                "positionnement reel contredit le plus la prediction.",
        xlabel="Rang de la valeur predite (percentile)",
        ylabel="Rang de la valeur comptabilisee (percentile)",
        note="Le passage aux rangs est une transformation monotone (un tri), pas "
             "une compression d'echelle. Elle conserve integralement l'ordre des "
             "observations, ce qui est precisement l'objet de la priorisation.",
        legende="lower right", nom_fichier=nom_fichier)


# #############################################################################
#  PARTIE D - PRIORISATION DES ANOMALIES
#  Score_i = A_i (ecart relatif a la borne franchie)
#          x B_i (erreur relative du modele)
#          x GWP_i (exposition brute)
# #############################################################################

# =============================================================================
#  FIG 13 - TOP N DES ANOMALIES PAR SCORE COMPOSITE
# -----------------------------------------------------------------------------
#  Le classement operationnel : la liste de travail de l'equipe de controle.
# =============================================================================
def fig_13_top_anomalies(anomalies_prio, top_n=15,
                         nom_fichier="fig_13_top_anomalies"):
    d = anomalies_prio.head(top_n).copy()
    dispo = [c for c in ID_COLS if c in d.columns]
    d["_cle"] = d[dispo].astype(str).agg(" | ".join, axis=1).str.slice(0, 42)
    d = d.iloc[::-1].reset_index(drop=True)

    y = np.arange(len(d))
    scores = d["score_composite"].to_numpy(float)
    part = scores / anomalies_prio["score_composite"].sum()

    # Degrade : intensite proportionnelle au rang
    couleurs = plt.cm.Reds(0.35 + 0.55 * (scores / scores.max()))

    fig, ax = plt.subplots(figsize=(11.5, max(6.0, 0.42 * len(d) + 2.4)))
    ax.barh(y, scores, color=couleurs, edgecolor="white", linewidth=1.0,
            height=0.72)

    for yi, s, p in zip(y, scores, part):
        ax.text(s * 1.012, yi, f"{format_montant(s)}   ({p:.1%} du total)",
                va="center", fontsize=9, color=COULEURS["prediction"])

    ax.set_yticks(y)
    ax.set_yticklabels([f"#{int(r)}  {c}"
                        for r, c in zip(d["rank"], d["_cle"])], fontsize=9)
    ax.set_xlim(0, scores.max() * 1.30)
    ax.xaxis.set_major_formatter(FuncFormatter(format_montant))
    ax.grid(axis="y", visible=False)

    concentration = anomalies_prio["score_composite"].head(top_n).sum() \
        / anomalies_prio["score_composite"].sum()
    ax.text(0.98, 0.04,
            f"Ces {top_n} anomalies concentrent {concentration:.1%} du score total\n"
            f"sur {len(anomalies_prio):,} anomalies detectees",
            transform=ax.transAxes, ha="right", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", fc="#FFFBE6",
                      ec=COULEURS["reference"]))

    return finaliser(
        fig, ax,
        titre=f"Classement operationnel : les {top_n} anomalies les plus prioritaires",
        lecture="Le score combine trois dimensions : l'ampleur du depassement de "
                "l'intervalle, l'erreur relative du modele, et l'exposition en "
                "primes. Une anomalie de faible montant sur un portefeuille "
                "majeur peut ainsi passer devant un ecart spectaculaire sur une "
                "ligne marginale.",
        xlabel="Score composite de priorisation (A x B x exposition)",
        ylabel=None, legende=None,
        note="Score_i = A_i x B_i x GWP_i, ou A_i est l'ecart relatif a la borne "
             "franchie, B_i l'erreur relative du modele, GWP_i la prime brute "
             "emise sans normalisation.",
        nom_fichier=nom_fichier)


# =============================================================================
#  FIG 14 - PARETO DE CONCENTRATION DU SCORE
# -----------------------------------------------------------------------------
#  Argument de gestion : combien de lignes faut-il reellement instruire pour
#  couvrir l'essentiel du risque ? Deux axes en pourcentage, donc aucun
#  probleme d'echelle.
# =============================================================================
def fig_14_pareto(anomalies_prio, seuils=(0.50, 0.80, 0.95),
                  nom_fichier="fig_14_pareto"):
    s = np.sort(anomalies_prio["score_composite"].to_numpy(float))[::-1]
    s = s[np.isfinite(s)]
    cum = np.cumsum(s) / s.sum()
    part_lignes = np.arange(1, len(s) + 1) / len(s)

    fig, ax = plt.subplots()
    ax.plot(part_lignes, cum, lw=2.6, color=COULEURS["prediction"],
            label="Part cumulee du score total")
    ax.fill_between(part_lignes, 0, cum, color=COULEURS["bande_cl"], alpha=0.5)
    ax.plot([0, 1], [0, 1], ls="--", lw=1.3, color=COULEURS["neutre"],
            label="Si toutes les anomalies pesaient pareil")

    couleurs_seuils = [COULEURS["conforme"], COULEURS["reference"],
                       COULEURS["anomalie"]]
    for seuil, coul in zip(seuils, couleurs_seuils):
        idx = int(np.searchsorted(cum, seuil))
        if idx >= len(part_lignes):
            continue
        px = part_lignes[idx]
        ax.plot([0, px], [seuil, seuil], ls=":", lw=1.2, color=coul)
        ax.plot([px, px], [0, seuil], ls=":", lw=1.2, color=coul)
        ax.scatter([px], [seuil], s=60, color=coul, zorder=6)
        ax.annotate(f"{px:.1%} des anomalies ({idx + 1:,} lignes)\n"
                    f"= {seuil:.0%} du score total",
                    xy=(px, seuil), xytext=(px + 0.08, seuil - 0.14),
                    fontsize=9, color=coul,
                    arrowprops=dict(arrowstyle="->", color=coul, lw=1.0))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    return finaliser(
        fig, ax,
        titre="Concentration du risque : combien de lignes faut-il reellement instruire ?",
        lecture="La courbe repond a une question de gestion, pas de statistique : "
                "si l'equipe de controle ne peut traiter qu'une fraction des "
                "alertes, quelle part de l'enjeu couvre-t-elle ? Plus la courbe "
                "monte vite, plus la priorisation est efficace.",
        xlabel="Part des anomalies traitees (classees par score decroissant)",
        ylabel="Part cumulee du score total",
        note="Les deux axes sont des pourcentages cumules : figure structurellement "
             "insensible a l'amplitude des montants sous-jacents.",
        legende="lower right", nom_fichier=nom_fichier)


# =============================================================================
#  FIG 15 - DECOMPOSITION DU SCORE EN RANGS PERCENTILES
# -----------------------------------------------------------------------------
#  Repond a : "pourquoi cette ligne est-elle en tete ?"
#  Les trois facteurs sont convertis en rangs percentiles, donc directement
#  comparables entre eux sans normalisation ni transformation.
# =============================================================================
def fig_15_decomposition_score(anomalies_prio, top_n=15,
                               nom_fichier="fig_15_decomposition"):
    facteurs = {"A : ecart a la borne": "A_ecart_borne",
                "B : erreur du modele": "B_erreur_modele",
                f"{GWP_COL} : exposition": GWP_COL}
    facteurs = {k: v for k, v in facteurs.items()
                if v in anomalies_prio.columns}
    if not facteurs:
        raise KeyError("Aucun facteur du score present dans `anomalies_prio`.")

    pct = pd.DataFrame({k: anomalies_prio[v].rank(pct=True) * 100
                        for k, v in facteurs.items()})
    pct["Score final"] = anomalies_prio["score_composite"].rank(pct=True) * 100

    d = pct.head(top_n).iloc[::-1]
    dispo = [c for c in ID_COLS if c in anomalies_prio.columns]
    cles = (anomalies_prio[dispo].astype(str).agg(" | ".join, axis=1)
            .str.slice(0, 30).head(top_n).iloc[::-1])
    rangs = anomalies_prio["rank"].head(top_n).astype(int).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10.5, max(6.0, 0.42 * len(d) + 2.6)))
    im = ax.imshow(d.to_numpy(float), aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=100)

    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            v = d.iat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9.5,
                    color="white" if (v > 72 or v < 22) else "#1B2A38",
                    fontweight="bold")

    ax.set_xticks(range(d.shape[1]))
    ax.set_xticklabels(d.columns, fontsize=10)
    ax.set_yticks(range(d.shape[0]))
    ax.set_yticklabels([f"#{r}  {c}" for r, c in zip(rangs, cles)], fontsize=9)
    ax.grid(visible=False)

    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Rang percentile au sein des anomalies detectees", fontsize=9.5)

    from scipy.stats import spearmanr
    correlations = {k: float(spearmanr(anomalies_prio["score_composite"],
                                       anomalies_prio[v]).statistic)
                    for k, v in facteurs.items()}
    dominant = max(correlations, key=correlations.get)
    alerte = ("  /!\\ un seul facteur pilote le classement"
              if correlations[dominant] > 0.9 else "")

    return finaliser(
        fig, ax,
        titre="Pourquoi ces lignes sont-elles prioritaires ? Decomposition du score",
        lecture="Chaque cellule donne le rang percentile de la ligne sur un des "
                "trois facteurs, 100 designant la valeur la plus elevee du lot. "
                "Une ligne rouge sur toute la largeur cumule les trois motifs "
                "d'alerte ; une ligne rouge sur une seule colonne signale un cas "
                "a lire avec prudence.",
        xlabel=None, ylabel=None, legende=None,
        note="Correlations de Spearman avec le score final : "
             + ", ".join(f"{k} = {v:.3f}" for k, v in correlations.items())
             + f". Facteur dominant : {dominant}." + alerte,
        nom_fichier=nom_fichier)


# =============================================================================
#  FIG 16 - FICHE VISUELLE D'UNE ANOMALIE
# -----------------------------------------------------------------------------
#  UN APPEL = UNE ANOMALIE = UNE FIGURE.
#  Positionne l'ecart de la ligne dans la distribution des ecarts observes en
#  calibration. C'est la figure qui convainc un interlocuteur non statisticien.
# =============================================================================
def fig_16_fiche_anomalie(anomalie_row, calib_scores, rang=None,
                          nom_fichier=None):
    cal = np.asarray(calib_scores, dtype=float)
    cal = cal[np.isfinite(cal)]
    s = float(anomalie_row["score_nonconformite"])
    p = float(anomalie_row.get("p_value", np.nan))
    rang = rang if rang is not None else int(anomalie_row.get("rank", 0))

    # Cadrage : on borne l'axe X au P99.5 de la calibration ou au score observe
    borne_x = max(np.percentile(cal, 99.5), s * 1.05)

    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    ax.hist(cal, bins=70, range=(cal.min(), borne_x),
            color=COULEURS["conforme_cl"], edgecolor="white", linewidth=0.7,
            label=f"Ecarts observes sur les donnees saines (n={len(cal):,})")

    q = float(np.quantile(cal, 1 - ALPHA))
    ax.axvline(q, color=COULEURS["reference"], ls="--", lw=2.0,
               label=f"Seuil de normalite ({100 * (1 - ALPHA):.0f}e percentile)")
    ax.axvline(s, color=COULEURS["anomalie"], lw=3.0,
               label="Cette observation")

    ax.axvspan(q, borne_x, color=COULEURS["anomalie"], alpha=0.06)
    ax.annotate("Zone d'anomalie", xy=((q + borne_x) / 2, ax.get_ylim()[1] * 0.92),
                ha="center", fontsize=10, color=COULEURS["anomalie"],
                fontweight="bold")

    depasse = "au-dessus" if anomalie_row["y_obs"] > anomalie_row["borne_haute"] \
        else "en dessous"
    lignes = [
        f"Valeur comptabilisee : {format_montant(anomalie_row['y_obs'])}",
        f"Valeur attendue      : {format_montant(anomalie_row['y_pred'])}",
        f"Zone normale         : [{format_montant(anomalie_row['borne_basse'])} ; "
        f"{format_montant(anomalie_row['borne_haute'])}]",
        f"Position             : {depasse} de la zone normale",
    ]
    if np.isfinite(p):
        lignes.append(f"p-value conforme     : {p:.2e}")
        lignes.append(f"Plus extreme que {100 * (1 - p):.2f} % des cas sains")
    # ma="left" : le bloc est ancre a droite, mais ses lignes restent alignees
    # a gauche entre elles, sinon les deux-points du tableau se desalignent.
    ax.text(0.985, 0.72, "\n".join(lignes), transform=ax.transAxes,
            ha="right", va="top", ma="left", fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.55", fc="white",
                      ec=COULEURS["neutre_cl"], alpha=0.96))

    ax.set_xlim(cal.min(), borne_x)
    ax.xaxis.set_major_formatter(FuncFormatter(format_montant))

    dispo = [c for c in ID_COLS if c in anomalie_row.index]
    ident = " | ".join(str(anomalie_row[c]) for c in dispo)

    return finaliser(
        fig, ax,
        titre=f"Anomalie #{rang} - {ident}",
        lecture="L'histogramme represente les ecarts au comportement attendu "
                "mesures sur des donnees saines. Le trait rouge situe cette "
                "observation dans cette distribution de reference. Plus il est a "
                "droite, moins l'ecart s'explique par la variabilite normale.",
        xlabel=f"Score de non-conformite ({UNITE})",
        ylabel="Nombre d'observations de calibration",
        note="Le score de non-conformite CQR vaut max(borne_basse - y, "
             "y - borne_haute) : il mesure de combien l'observation sort de son "
             "intervalle, en euros.",
        legende="upper center",
        nom_fichier=nom_fichier or f"fig_16_fiche_anomalie_{rang}")


# =============================================================================
#  FIG 17 - TRAJECTOIRE TEMPORELLE D'UNE ENTITE
# -----------------------------------------------------------------------------
#  UN APPEL = UNE ENTITE = UNE FIGURE.
#  Le passage au temps change la nature de l'argument : un ecart isole peut
#  etre du bruit, un ecart repete trimestre apres trimestre ne l'est plus.
# =============================================================================
def fig_17_trajectoire(results, entite=None, id_cols=None, nom_fichier=None):
    id_cols = id_cols or [c for c in ID_COLS if c in results.columns]
    d = results.copy()
    d["_cle"] = d[id_cols].astype(str).agg(" | ".join, axis=1)

    if entite is None:  # a defaut, l'entite la plus souvent hors intervalle
        classement = (d.assign(hors=~d["dans_intervalle"].astype(bool))
                      .groupby("_cle")["hors"].sum().sort_values(ascending=False))
        entite = classement.index[0]

    sous = d[d["_cle"] == entite].copy()
    if sous.empty:
        raise ValueError(f"Entite introuvable : {entite}")

    # Une trajectoire suppose UNE observation par periode. Si les identifiants
    # fournis ne suffisent pas a isoler une unite statistique, plusieurs lignes
    # coexistent sur le meme trimestre et la courbe devient illisible. On
    # conserve alors, pour chaque periode, la ligne la plus atypique, et on le
    # signale explicitement plutot que d'agreger en silence : additionner des
    # bornes conformes serait faux (la somme de quantiles n'est pas le quantile
    # de la somme).
    cle_periode = ("time_idx" if "time_idx" in sous.columns
                   else ["year", "quarter"])
    n_avant = len(sous)
    ecart = np.maximum(sous["borne_basse"] - sous["y_obs"],
                       sous["y_obs"] - sous["borne_haute"])
    sous = (sous.assign(_ecart=ecart)
            .sort_values("_ecart", ascending=False)
            .groupby(cle_periode, as_index=False, observed=True).head(1)
            .drop(columns="_ecart"))
    doublons = n_avant - len(sous)
    if doublons:
        print(f"  [fig 17] {doublons} ligne(s) partagent une meme periode pour "
              f"cette entite : la plus atypique de chaque trimestre est retenue.\n"
              f"           Affiner `id_cols` si une unite statistique plus fine "
              f"est attendue.")

    if "time_idx" in sous.columns:
        sous = sous.sort_values("time_idx")
    elif {"year", "quarter"}.issubset(sous.columns):
        sous = sous.sort_values(["year", "quarter"])
    etiquettes = (sous["year"].astype(str) + "-T" + sous["quarter"].astype(str)
                  if {"year", "quarter"}.issubset(sous.columns)
                  else pd.Series(range(len(sous)), index=sous.index).astype(str))

    x = np.arange(len(sous))
    obs = sous["y_obs"].to_numpy(float)
    pred = sous["y_pred"].to_numpy(float)
    lo = sous["borne_basse"].to_numpy(float)
    hi = sous["borne_haute"].to_numpy(float)
    dedans = sous["dans_intervalle"].to_numpy(bool)

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.fill_between(x, lo, hi, color=COULEURS["bande_cl"], alpha=0.85,
                    label=f"Intervalle conforme a {100 * (1 - ALPHA):.0f} %")
    ax.plot(x, pred, ls="--", lw=1.8, color=COULEURS["prediction"],
            marker="D", ms=6, markerfacecolor="white",
            label="Valeur attendue par le modele")
    ax.plot(x, obs, lw=2.0, color=COULEURS["neutre"], zorder=4)
    ax.scatter(x[dedans], obs[dedans], s=95, color=COULEURS["conforme"],
               edgecolor="white", linewidth=1.0, zorder=6,
               label="Trimestre conforme")
    ax.scatter(x[~dedans], obs[~dedans], s=140, marker="X",
               color=COULEURS["anomalie"], edgecolor="white", linewidth=1.0,
               zorder=7, label="Trimestre hors intervalle")

    for xi, o, dd in zip(x, obs, dedans):
        if not dd:
            ax.annotate(format_montant(o), xy=(xi, o), xytext=(0, 14),
                        textcoords="offset points", ha="center", fontsize=8.5,
                        color=COULEURS["anomalie"], fontweight="bold")

    n_hors = int((~dedans).sum())
    n_tot = len(sous)
    if n_tot >= 3:
        from scipy.stats import binom
        p_pers = float(binom.sf(n_hors - 1, n_tot, ALPHA))
        msg = (f"{n_hors} trimestre(s) hors intervalle sur {n_tot}\n"
               f"Probabilite sous l'hypothese du hasard : {p_pers:.2e}")
    else:
        msg = f"{n_hors} trimestre(s) hors intervalle sur {n_tot}"
    ax.text(0.015, 0.955, msg, transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", fc="white",
                      ec=COULEURS["neutre_cl"], alpha=0.95))

    # Reserve de hauteur pour la legende et l'encadre de synthese, qui sinon
    # recouvrent les pics de la trajectoire.
    cadrer_axe_y(ax, np.concatenate([lo, hi, obs, pred]), p_haut=100.0)
    bas, haut = ax.get_ylim()
    ax.set_ylim(bas, haut + 0.32 * (haut - bas))
    ax.set_xticks(x)
    ax.set_xticklabels(etiquettes.tolist(), rotation=45, ha="right")

    return finaliser(
        fig, ax,
        titre=f"Trajectoire trimestrielle - {entite}",
        lecture="La bande bleue est la zone que le modele juge normale a chaque "
                "trimestre, compte tenu du profil de l'entite. Une sortie isolee "
                "peut relever du hasard ; une sortie repetee, du meme cote, "
                "signale un probleme de processus et non une fluctuation.",
        xlabel="Periode",
        ylabel=f"{TARGET} ({UNITE})",
        note="La probabilite affichee suit une loi binomiale de parametres "
             "(nombre de trimestres, alpha) : elle mesure la chance d'observer "
             "autant de sorties si l'entite etait parfaitement normale.",
        format_y_montant=True, legende="upper right",
        nom_fichier=nom_fichier or "fig_17_trajectoire")


# =============================================================================
#  FIG 18 - PERSISTANCE : LES ENTITES QUI RECIDIVENT
# -----------------------------------------------------------------------------
#  Le passage de "une observation atypique" a "une entite a auditer".
#  C'est l'argument le plus fort du memoire : la recurrence.
# =============================================================================
def fig_18_persistance(results, top_n=15, id_cols=None,
                       nom_fichier="fig_18_persistance"):
    from scipy.stats import binom
    id_cols = id_cols or [c for c in ID_COLS if c in results.columns]

    d = results.copy()
    d["_cle"] = d[id_cols].astype(str).agg(" | ".join, axis=1)
    g = (d.assign(hors=~d["dans_intervalle"].astype(bool))
         .groupby("_cle")
         .agg(n_periodes=("hors", "size"), n_hors=("hors", "sum"))
         .reset_index())
    g = g[g["n_periodes"] >= 2]
    if g.empty:
        raise ValueError("Pas assez de periodes par entite pour la persistance.")

    g["p_persistance"] = binom.sf(g["n_hors"] - 1, g["n_periodes"], ALPHA)
    g["taux"] = g["n_hors"] / g["n_periodes"]
    g = g[g["n_hors"] > 0].sort_values("p_persistance").head(top_n)
    g = g.iloc[::-1].reset_index(drop=True)

    y = np.arange(len(g))
    couleurs = np.where(g["p_persistance"] < 0.01, COULEURS["anomalie"],
                        np.where(g["p_persistance"] < 0.05,
                                 COULEURS["reference"], COULEURS["neutre"]))

    fig, ax = plt.subplots(figsize=(11.5, max(6.0, 0.42 * len(g) + 2.4)))
    ax.barh(y, g["taux"], color=couleurs, edgecolor="white", linewidth=1.0,
            height=0.72)
    ax.axvline(ALPHA, color=COULEURS["prediction"], ls="--", lw=2.0,
               label=f"Taux attendu si l'entite etait normale : {ALPHA:.0%}")

    for yi, r in g.iterrows():
        ax.text(r["taux"] + 0.012, yi,
                f"{int(r['n_hors'])}/{int(r['n_periodes'])} trimestres   "
                f"p = {r['p_persistance']:.1e}",
                va="center", fontsize=9, color=COULEURS["prediction"])

    ax.set_yticks(y)
    ax.set_yticklabels(g["_cle"].str.slice(0, 40).tolist(), fontsize=9)
    ax.set_xlim(0, min(1.0, float(g["taux"].max()) * 1.45))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", visible=False)

    n_signif = int((g["p_persistance"] < 0.01).sum())
    ax.text(0.98, 0.04,
            f"{n_signif} entite(s) dont la recurrence n'est pas explicable\n"
            "par le hasard au seuil de 1 %",
            transform=ax.transAxes, ha="right", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", fc="#FFFBE6",
                      ec=COULEURS["reference"]))

    return finaliser(
        fig, ax,
        titre="Entites recidivistes : la recurrence comme critere d'audit",
        lecture="Une observation isolee hors intervalle est attendue dans "
                f"{ALPHA:.0%} des cas, par construction de la methode. Ce qui ne "
                "l'est pas, c'est qu'une meme entite ressorte trimestre apres "
                "trimestre. Les barres rouges designent les entites pour "
                "lesquelles cette repetition est statistiquement inexplicable.",
        xlabel="Part des trimestres passes hors intervalle",
        ylabel=None,
        note="Sous l'hypothese d'independance, le nombre de sorties suit une loi "
             "binomiale de parametres (n_trimestres, alpha). La p-value affichee "
             "est la probabilite d'observer au moins autant de sorties.",
        legende="lower right", nom_fichier=nom_fichier)


# #############################################################################
#  BLOC 3 - EXECUTION COMPLETE
#  Chaque appel produit UNE figure et UN fichier PNG. Rien n'est empile.
# #############################################################################

def produire_toutes_les_figures(results, anomalies_prio=None, results_pv=None,
                                calib_scores=None, strates=None):
    """Genere l'integralite des figures, une par une, dans l'ordre du memoire.

    Commenter une ligne suffit a retirer une figure du jeu final.
    """
    strates = strates or STRATES_NOM
    print("\n" + "#" * 78)
    print("#  PRODUCTION DES FIGURES - une figure par appel, aucun empilement")
    print("#" * 78)

    verifier_contrat(results, anomalies_prio, results_pv)
    verifier_absence_de_log()

    print("\n--- PARTIE A : comprendre la cible " + "-" * 40)
    fig_01_profil_cible(results)
    fig_02_concentration_cible(results)

    if results_pv is not None and calib_scores is not None:
        print("\n--- PARTIE B : validite de la methode " + "-" * 37)
        fig_03_calibration(results_pv, calib_scores)
        fig_04_histogramme_pvalues(results_pv)
        fig_05_qqplot_pvalues(results_pv)
        fig_06_benjamini_hochberg(results_pv)
    else:
        print("\n[PARTIE B ignoree : results_pv ou calib_scores non fournis]")
    fig_07_couverture_par_largeur(results)
    fig_08_largeurs(results)

    print("\n--- PARTIE C : la bande conforme " + "-" * 42)
    fig_09_bande_normalisee(results)
    for s in strates:
        try:
            fig_10_bande_strate(results, strate=s)
        except ValueError as err:
            print(f"  [strate {s} ignoree] {err}")
    fig_11_bande_detaillee(results, n=30, mode="echantillon")
    fig_11_bande_detaillee(results, n=30, mode="anomalies")
    fig_12_predit_vs_reel_rangs(results)

    if anomalies_prio is not None and len(anomalies_prio):
        print("\n--- PARTIE D : priorisation " + "-" * 47)
        fig_13_top_anomalies(anomalies_prio)
        fig_14_pareto(anomalies_prio)
        fig_15_decomposition_score(anomalies_prio)
        if calib_scores is not None and \
                "score_nonconformite" in anomalies_prio.columns:
            for i in range(min(3, len(anomalies_prio))):
                fig_16_fiche_anomalie(anomalies_prio.iloc[i], calib_scores,
                                      rang=i + 1)
    try:
        fig_17_trajectoire(results)
        fig_18_persistance(results)
    except ValueError as err:
        print(f"  [figures temporelles ignorees] {err}")

    print("\n" + "#" * 78)
    print(f"#  Toutes les figures sont dans : {os.path.abspath(DOSSIER_FIGURES)}")
    print("#" * 78)


# --- Lancement --------------------------------------------------------------
# Adapter le nom du DataFrame source (results_v2 ou results_test) :
#
# produire_toutes_les_figures(
#     results=results_v2,
#     anomalies_prio=anomalies_prio,
#     results_pv=results_pv,
#     calib_scores=calib_scores,
# )
#
# Ou, pour une seule figure a la fois (usage recommande en redaction) :
#
# fig_09_bande_normalisee(results_v2)
# fig_10_bande_strate(results_v2, strate="P90-99")
# fig_16_fiche_anomalie(anomalies_prio.iloc[0], calib_scores, rang=1)
