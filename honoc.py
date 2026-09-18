dash>=2.11
pandas
numpy
plotly
jupyter
ipykernel


# -*- coding: utf-8 -*-
# =============================================================================
#  TABLEAU DE BORD UNIFIE - anomalies, evolution, priorisation
#
#  UN SEUL BLOC, UNE SEULE BASE. Il suffit d'avoir en session :
#      df, ID_COLS, TARGET            (ALPHA facultatif)
#
#  DEUX REGLES QUI COMMANDENT TOUT LE FICHIER
#  -------------------------------------------
#  1. SEULES LES ANOMALIES SONT AFFICHEES. Une ligne entre dans le tableau de
#     bord si et seulement si son score_composite est different de zero. Les
#     situations normales ne polluent ni les graphiques ni les montants.
#  2. AUCUNE SOMMATION. La valeur observee, la prediction et les bornes
#     conformes affichees sont celles de la LIGNE, telles qu'elles sont dans
#     df. Rien n'est agrege, donc rien ne peut etre fausse par un cumul.
#
#  CE QUE FONT LES AXES
#  --------------------
#  Les axes commandent deux choses, et deux seulement : la profondeur du
#  cercle, et la composition des libelles. Ils ne modifient AUCUN montant.
#  C'est le changement par rapport a la version precedente, qui les faisait
#  sommer les montants et melangeait de ce fait les lignes saines aux
#  anomalies.
#
#  D'OU VIENT CHAQUE CHOSE
#  ------------------------
#    periode validee  = les lignes de df qui portent une prediction conforme,
#                       c'est-a-dire dont borne_basse, borne_haute et y_pred
#                       sont renseignees. Si plusieurs trimestres en portent,
#                       le plus recent est retenu et le code le dit.
#    anomalies        = ces lignes-la, filtrees sur score_composite != 0
#    historique       = df en entier, filtre sur la cle COMPLETE du
#                       sous-portefeuille, tous trimestres disponibles
#
#  SI LA COURBE D'EVOLUTION N'AFFICHE QU'UN POINT
#  -----------------------------------------------
#  C'est que df ne contient qu'un seul trimestre pour ce sous-portefeuille.
#  Une trace a un point n'affiche qu'un marqueur, jamais de ligne. Le titre du
#  panneau le dit explicitement, et le resume de demarrage indique combien de
#  trimestres df contient reellement.
# =============================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display

from dash import Dash, dcc, html, Input, Output, State, callback_context

# ┌──────────────────────────── PARAMETRES ────────────────────────────┐
TOP_N_PANNEAUX = 12       # barres et forest plot
N_TRIMESTRES   = 10       # historique affiche
N_UNITES_LISTE = 30       # unites proposees dans le selecteur
N_LIGNES_TABLE = 15       # lignes du tableau du bas
N_VARS_EXPLIC  = 5        # variables explicatives sous la courbe
COL_SCORE      = "score_composite"
COL_OBS        = "y_obs"          # a defaut, la cible elle-meme est utilisee
COL_PRED       = "y_pred"
COL_LO, COL_HI = "borne_basse", "borne_haute"
PERIODE_VALIDEE = "auto"  # "auto" ou un couple explicite, ex. (2024, 4)
ECHELLE        = "Bluered"
VUE_GENERALE   = ""
# └─────────────────────────────────────────────────────────────────────┘

#  Separateur interne des identifiants. U+001F ne peut pas apparaitre dans un
#  libelle metier, contrairement a "/".
SEP = "\x1f"

#  Colonnes de resultat, jamais candidates comme variable explicative.
_EXCLURE_VARS = {"time_idx", "year", "quarter", "y_obs", "y_pred",
                 "borne_basse", "borne_haute", "dans_intervalle", "largeur",
                 "score_composite", "rank", "ecart_intervalle", "severite",
                 "A_ecart_borne", "B_erreur_modele", "sigma", "step"}

_ENCRE, _ACCENT, _OK = "#141B34", "#c0392b", "#3D5A9E"
_BLEU, _GRILLE, _GRIS = "#636EFA", "#EDF1F7", "#8A93A5"
_DOUX = ["#6C8EBF", "#82B366", "#C08552", "#9673A6", "#5F9EA0"]
_VIDE = "rgba(0,0,0,0)"


def _fmt(v):
    """Format francais des milliers : 1249441.0 -> '1 249 441'."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:,.0f}".replace(",", " ")


def _fmt4(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:,.4g}".replace(",", " ")


def _session(nom):
    """Retrouve une variable de la session Jupyter, pas seulement du module."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None and nom in ip.user_ns:
            return True, ip.user_ns[nom]
    except Exception:
        pass
    if nom in globals():
        return True, globals()[nom]
    import __main__
    if hasattr(__main__, nom):
        return True, getattr(__main__, nom)
    return False, None


def _mise_en_forme(fig, hauteur, marges=None):
    fig.update_layout(
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=_GRIS),
        height=hauteur, autosize=True, separators=", ",
        hoverlabel=dict(bgcolor="white", bordercolor=_GRILLE, align="left",
                        font=dict(size=12.5, color=_ENCRE)),
        margin=marges or dict(l=10, r=40, t=95, b=45))
    return fig


def _bandeau(txt, fond="#eceff1", coul="#37474f"):
    """Bandeau HTML. Prend UNE CHAINE, jamais le retour d'un display()."""
    return HTML(f"<div style='font-family:system-ui,sans-serif;font-size:12.5px;"
                f"color:{coul};background:{fond};padding:9px 13px;"
                f"border-radius:6px;margin:14px 0 8px 0'>{txt}</div>")


# =============================================================================
#  1. SOCLE DE DONNEES - anomalies seules, valeurs brutes
# =============================================================================
class _Socle:
    """Prepare une fois ce que les mises a jour reliront des dizaines de fois."""

    def __init__(self, df, id_cols, target, alpha, verbeux=True):
        self.df, self.target, self.alpha = df, target, alpha
        self.cles = [c for c in id_cols if c in df.columns]
        if not self.cles:
            raise ValueError(f"Aucune colonne de ID_COLS n'est dans df : "
                             f"{list(id_cols)}")
        if target not in df.columns:
            raise ValueError(f"TARGET '{target}' absent de df.")
        if COL_SCORE not in df.columns:
            raise ValueError(f"Colonne de score '{COL_SCORE}' absente de df.")

        #  --- la periode validee : les lignes qui portent une prediction ---
        #  Marqueur tire de la donnee, pas d'une convention comme "le dernier
        #  trimestre" : une ligne appartient a la periode validee si et
        #  seulement si elle porte un intervalle conforme.
        cp = [c for c in (COL_LO, COL_HI, COL_PRED) if c in df.columns]
        if not cp:
            raise ValueError(
                f"df ne contient aucune des colonnes {COL_LO}, {COL_HI}, "
                f"{COL_PRED} : impossible d'identifier la periode validee.")
        porte_cp = df[cp].notna().all(axis=1)
        if not porte_cp.any():
            raise ValueError("Aucune ligne de df ne porte de prediction "
                             f"conforme ({cp} tous renseignes).")

        valide = df[porte_cp]
        self.periode = None
        if {"year", "quarter"} <= set(df.columns):
            couples = sorted(set(zip(valide["year"].astype(int),
                                     valide["quarter"].astype(int))))
            self.periode = (tuple(PERIODE_VALIDEE)
                            if PERIODE_VALIDEE != "auto" else couples[-1])
            if len(couples) > 1 and verbeux:
                print(f"Note : {len(couples)} trimestres portent une "
                      f"prediction. Le plus recent est retenu "
                      f"({self.periode[0]}-T{self.periode[1]}).")
            valide = valide[(valide["year"].astype(int) == self.periode[0])
                            & (valide["quarter"].astype(int) == self.periode[1])]

        #  --- REGLE 1 : seules les anomalies entrent dans le tableau de bord ---
        score = pd.to_numeric(valide[COL_SCORE], errors="coerce").fillna(0)
        self.ano = valide[score != 0].copy()
        self.ano[COL_SCORE] = score[score != 0].values
        if not len(self.ano):
            raise ValueError(
                f"Aucune anomalie : toutes les lignes de la periode validee "
                f"ont un {COL_SCORE} nul.")

        #  y_obs : la colonne dediee si elle existe, sinon la cible elle-meme.
        if COL_OBS not in self.ano.columns:
            self.ano[COL_OBS] = self.ano[target].values

        for c in self.cles:
            self.ano[c] = self.ano[c].astype(str)
        self.ano = self.ano.sort_values(COL_SCORE, ascending=False) \
                           .reset_index(drop=True)
        self.ano["rang"] = np.arange(1, len(self.ano) + 1)
        self.score_global = float(self.ano[COL_SCORE].sum()) or 1.0

        #  Cles textuelles de df, par axe. Construites une fois : sans elles,
        #  chaque changement de selection relirait df en entier.
        self._df_txt = {c: df[c].astype(str).values for c in self.cles}

        #  Variables explicatives candidates : numeriques, hors cible et hors
        #  colonnes de resultat.
        self.vars_explic = [
            c for c in df.columns
            if c != target and c not in _EXCLURE_VARS
            and pd.api.types.is_numeric_dtype(df[c])]

        #  Profondeur d'historique reellement disponible : c'est elle qui
        #  determine si une courbe peut etre tracee.
        self.trimestres = []
        if {"year", "quarter"} <= set(df.columns):
            self.trimestres = sorted(set(zip(df["year"].astype(int),
                                             df["quarter"].astype(int))))

        if verbeux:
            per_txt = (f"{self.periode[0]}-T{self.periode[1]}"
                       if self.periode else "non datee")
            print("=" * 74)
            print(f"SOURCE UNIQUE : df   ·   {len(df):,} lignes".replace(",", " "))
            print("=" * 74)
            if self.trimestres:
                t0, t1 = self.trimestres[0], self.trimestres[-1]
                print(f"   trimestres dans df : {t0[0]}-T{t0[1]} -> "
                      f"{t1[0]}-T{t1[1]}   ({len(self.trimestres)} au total)")
                if len(self.trimestres) < 2:
                    print("      ATTENTION : un seul trimestre dans df. La "
                          "courbe d'evolution ne pourra")
                    print("      afficher qu'un point, sans ligne : il n'y a "
                          "rien a relier.")
                elif len(self.trimestres) < 3:
                    print("      NOTE : deux trimestres seulement. Le classement "
                          "des variables")
                    print("      explicatives demande au moins trois points et "
                          "sera indisponible.")
            print(f"   periode validee    : {per_txt}   "
                  f"({int(porte_cp.sum()):,} lignes avec prediction)"
                  .replace(",", " "))
            print(f"   anomalies retenues : {len(self.ano):,} "
                  f"({COL_SCORE} != 0)".replace(",", " "))
            print(f"   axes disponibles   : {', '.join(self.cles)}")
            print(f"   variables suivies  : {len(self.vars_explic)}")
            print("=" * 74)

    # ------------------------------------------------------------ filtrage
    def filtrer(self, colonne, valeur):
        """Filtre les anomalies. Se replie sur la vue generale si la valeur ne
        correspond pas a la colonne : c'est l'etat transitoire d'un changement
        de maille, et filtrer dessus viderait tous les panneaux."""
        if not colonne or colonne not in self.ano.columns:
            return self.ano, "vue generale"
        if not valeur:
            return self.ano, f"{colonne} — vue generale"
        m = self.ano[colonne].astype(str) == str(valeur)
        if not m.any():
            return self.ano, f"{colonne} — vue generale"
        return self.ano[m], f"{colonne} = {valeur}"

    # ================================================================
    #  REGLE 2 : aucune sommation. Une ligne reste une ligne.
    # ================================================================
    def preparer(self, sub, axes):
        """Ajoute seulement un libelle lisible, tire des axes actifs.

        Aucun groupby, aucune somme : y_obs, y_pred, borne_basse et
        borne_haute restent les valeurs de la ligne, telles qu'elles sont dans
        df. Les axes ne servent ici qu'a composer le libelle affiche.
        """
        axes = [a for a in axes if a in sub.columns] or self.cles[:1]
        if not len(sub):
            return sub.assign(libelle=pd.Series(dtype=str))
        t = sub.copy()
        t["libelle"] = t[axes].astype(str).agg(" | ".join, axis=1).str.slice(0, 38)
        #  Le rang prefixe garantit un libelle unique meme si deux anomalies
        #  ne different que par un axe desactive.
        t["libelle_rang"] = ("#" + t["rang"].astype(str) + "  " + t["libelle"])
        return t

    # ------------------------------------------------------------ unites
    def unites(self, sub, axes, n=N_UNITES_LISTE):
        """Une entree par ANOMALIE, identifiee par sa cle complete."""
        if not len(sub):
            return []
        t = self.preparer(sub.head(n), axes)
        return [(f"{r['libelle_rang']}   ({_fmt(r[COL_OBS])})",
                 tuple(str(r[c]) for c in self.cles))
                for _, r in t.iterrows()]

    def ligne(self, sub, cle):
        """Retrouve l'anomalie exacte a partir de sa cle complete."""
        if not len(sub) or cle is None:
            return None
        m = np.ones(len(sub), dtype=bool)
        for c, v in zip(self.cles, cle):
            m &= (sub[c].astype(str).values == str(v))
        t = sub[m]
        return None if not len(t) else t.iloc[0]

    def contexte(self, r):
        """Valeurs BRUTES de la ligne : rien n'est somme ni recalcule."""
        if r is None:
            return None
        per = (f"{self.periode[0]}-T{self.periode[1]}" if self.periode else None)
        obs, lo, hi = float(r[COL_OBS]), float(r[COL_LO]), float(r[COL_HI])
        return dict(per=per, pred=float(r[COL_PRED]), lo=lo, hi=hi, obs=obs,
                    couvert=bool(lo <= obs <= hi), score=float(r[COL_SCORE]),
                    rang=int(r["rang"]))

    # ------------------------------------------------- masques par axes
    def _masque(self, textes, cles, valeurs):
        m = np.ones(len(next(iter(textes.values()))), dtype=bool)
        for a, v in zip(cles, valeurs):
            m &= (textes[a] == str(v))
        return m

    # -------------------------------------------------------- historique
    def historique(self, cle, n=N_TRIMESTRES):
        """Historique du sous-portefeuille, sur sa CLE COMPLETE.

        Aucune agregation entre sous-portefeuilles : on suit exactement la
        ligne selectionnee a travers le temps.
        """
        d = self.df[self._masque(self._df_txt, self.cles, cle)]
        if not len(d):
            return None, []
        colonnes = [self.target] + [v for v in self.vars_explic if v in d.columns]
        if {"year", "quarter"} <= set(d.columns):
            #  Le groupby ne sert qu'a dedoublonner si df contenait plusieurs
            #  lignes pour un meme trimestre ; avec la cle complete il n'y en a
            #  normalement qu'une, la somme est alors l'identite.
            g = d.groupby(["year", "quarter"], observed=True)[colonnes] \
                 .sum().reset_index().sort_values(["year", "quarter"]).tail(n)
            per = (g["year"].astype(int).astype(str) + "-T"
                   + g["quarter"].astype(int).astype(str)).tolist()
            return g, per
        return d[colonnes].tail(n).reset_index(drop=True), \
            [str(i) for i in range(min(n, len(d)))]

    # ------------------------------------ variables explicatives, notees
    def variables_explicatives(self, hist, n=N_VARS_EXPLIC):
        """Les n variables dont la RUPTURE au dernier trimestre est la plus
        forte, mesuree contre leur propre passe.

        Critere : (valeur du trimestre valide − mediane des trimestres
        precedents) rapportee a l'ecart inter-quartile de ces memes trimestres.
        Une variable stable qui saute remonte en tete ; une variable
        naturellement volatile ne remonte que si elle sort de son regime.
        """
        vide = pd.DataFrame(columns=["variable", "valeur", "z"])
        if hist is None or len(hist) < 3:
            return vide
        lignes = []
        for v in self.vars_explic:
            if v not in hist.columns:
                continue
            vals = hist[v].to_numpy(dtype="float64", na_value=np.nan)
            if not np.isfinite(vals).all():
                continue
            passe, courant = vals[:-1], vals[-1]
            med = float(np.median(passe))
            q1, q3 = np.percentile(passe, [25, 75])
            #  Plancher de dispersion : sans lui, une variable strictement
            #  constante donnerait une division par zero et un z infini.
            dispersion = max(float(q3 - q1), abs(med) * .01, 1e-9)
            z = (courant - med) / dispersion
            if not np.isfinite(z) or z == 0:
                continue
            lignes.append({"variable": v, "valeur": courant, "z": z})
        if not lignes:
            return vide
        t = pd.DataFrame(lignes)
        return t.reindex(t["z"].abs().sort_values(ascending=False).index) \
                .head(n).reset_index(drop=True)

    # ------------------------------------------------------- hierarchie
    def hierarchie(self, sub, chemin):
        """Noeuds du cercle. Ici l'agregation porte sur le SCORE, pas sur des
        montants : cumuler des scores de gravite est licite, c'est le propos
        meme du cercle. Les montants, eux, ne sont jamais sommes."""
        if not len(sub) or not chemin:
            return pd.DataFrame()
        prof_max = len(chemin)
        agg = {"score_total": (COL_SCORE, "sum"),
               "score_moyen": (COL_SCORE, "mean"),
               "score_max": (COL_SCORE, "max"), "n": (COL_SCORE, "size")}
        total = float(sub[COL_SCORE].sum()) or 1.0

        lignes = []
        for prof in range(1, prof_max + 1):
            cols = chemin[:prof]
            g = sub.groupby(cols, observed=True).agg(**agg).reset_index()
            for _, r in g.iterrows():
                vals = [str(r[c]) for c in cols]
                st = float(r["score_total"])
                if not np.isfinite(st):
                    continue
                lignes.append({
                    "id": SEP.join(vals), "label": vals[-1],
                    "parent": SEP.join(vals[:-1]) if prof > 1 else "",
                    "profondeur": prof, "score_total": st,
                    "valeur_secteur": st if prof == prof_max else 0.0,
                    "score_moyen": float(r["score_moyen"]),
                    "score_max": float(r["score_max"]), "n": int(r["n"]),
                    "part": 100 * st / total})
        return pd.DataFrame(lignes)


# =============================================================================
#  2. CREATION DES FIGURES (structure figee, seules les donnees changent)
# =============================================================================
def _creer_cercle():
    fig = go.Figure(go.Sunburst(ids=[], labels=[], parents=[], values=[],
                                branchvalues="remainder"))
    _mise_en_forme(fig, 620, dict(l=10, r=10, t=100, b=15))
    return fig


def _creer_barres():
    fig = go.Figure(go.Bar(x=[], y=[], orientation="h", showlegend=False,
                           marker=dict(colorscale=ECHELLE,
                                       line=dict(width=.5, color="white"),
                                       colorbar=dict(title="Montant",
                                                     thickness=14, len=.7,
                                                     tickformat="~s"))))
    fig.update_layout(xaxis=dict(tickformat="~s"),
                      yaxis=dict(tickfont=dict(size=10)))
    _mise_en_forme(fig, 520, dict(l=10, r=40, t=95, b=50))
    return fig


def _creer_forest(target):
    fig = go.Figure()
    fig.add_scatter(x=[], y=[], mode="lines", hoverinfo="skip", opacity=.35,
                    line=dict(color="#3a6bbf", width=10),
                    name="Intervalle conforme")                     # 0 bande
    fig.add_scatter(x=[], y=[], mode="lines", showlegend=False, hoverinfo="skip",
                    line=dict(color=_ACCENT, width=2, dash="dot"))  # 1 depassement
    fig.add_scatter(x=[], y=[], mode="markers", name="Prediction",
                    marker=dict(symbol="diamond", size=10, color="white",
                                line=dict(color="black", width=1.5)))  # 2 pred
    fig.add_scatter(x=[], y=[], mode="markers", name="Valeur observee",
                    marker=dict(size=13, color=_ACCENT,
                                line=dict(color="#7b241c", width=1.3)))  # 3 obs
    fig.update_xaxes(tickformat="~s", title_text=f"<b>{target}</b>")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="center", x=.5))
    _mise_en_forme(fig, 520, dict(l=10, r=50, t=115, b=55))
    return fig


def _creer_evolution(target):
    """Courbe d'evolution, avec l'intervalle conforme et la prediction."""
    fig = go.Figure()
    fig.add_scatter(x=[], y=[], mode="lines", line=dict(color=_VIDE, width=0),
                    fill="tozeroy", fillcolor="rgba(140,147,165,0.10)",
                    showlegend=False, hoverinfo="skip")                # 0 aire
    fig.add_scatter(x=[], y=[], mode="lines", line=dict(color=_BLEU, width=15),
                    opacity=.26, name="Intervalle conforme")           # 1 bande
    fig.add_scatter(x=[], y=[], mode="markers", name="Prediction",
                    marker=dict(symbol="diamond", size=11, color="white",
                                line=dict(color=_ENCRE, width=1.8)))   # 2 pred
    fig.add_scatter(x=[], y=[], mode="lines+markers+text", name=target,
                    line=dict(color=_ENCRE, width=2.8, shape="spline",
                              smoothing=.55),
                    marker=dict(size=9, color="white",
                                line=dict(color=_ENCRE, width=2.2)),
                    textposition="top center",
                    textfont=dict(size=9.5, color=_GRIS))              # 3 cible
    fig.add_scatter(x=[], y=[], mode="markers", showlegend=False,
                    hoverinfo="skip",
                    marker=dict(size=34, color="rgba(192,57,43,0.20)"))  # 4 halo
    fig.add_scatter(x=[], y=[], mode="markers", name="Statut",
                    marker=dict(size=13, color=_ACCENT,
                                line=dict(color="white", width=2.4)))  # 5 statut
    fig.update_xaxes(showgrid=False, linecolor=_GRILLE, showspikes=True,
                     spikemode="across", spikethickness=1.2, spikedash="dot",
                     spikecolor=_GRIS, title_text="<b>Trimestre</b>")
    fig.update_yaxes(gridcolor=_GRILLE, zeroline=False, tickformat="~s",
                     title_text=f"<b>{target}</b>")
    _mise_en_forme(fig, 420, dict(l=85, r=45, t=95, b=45))
    fig.update_layout(hovermode="x unified",
                      legend=dict(orientation="h", y=1.06, x=1,
                                  xanchor="right", font=dict(size=11)))
    return fig


def _creer_variables(n=N_VARS_EXPLIC):
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        vertical_spacing=.055, subplot_titles=[" "] * n)
    for i in range(n):
        c = _DOUX[i % len(_DOUX)]
        fig.add_scatter(x=[], y=[], mode="lines+markers", showlegend=False,
                        line=dict(color=c, width=2.3, shape="spline",
                                  smoothing=.55),
                        marker=dict(size=6, color="white",
                                    line=dict(color=c, width=1.8)),
                        row=i + 1, col=1)                       # 2i   courbe
        fig.add_scatter(x=[], y=[], mode="markers", showlegend=False,
                        hoverinfo="skip",
                        marker=dict(size=13, color=c,
                                    line=dict(color="white", width=2.2)),
                        row=i + 1, col=1)                       # 2i+1 trimestre
    fig.update_xaxes(showgrid=False, showticklabels=False, linecolor=_GRILLE)
    fig.update_xaxes(showticklabels=True, title_text="<b>Trimestre</b>",
                     row=n, col=1)
    fig.update_yaxes(gridcolor=_GRILLE, zeroline=False, tickformat="~s",
                     tickfont=dict(size=9))
    _mise_en_forme(fig, 150 * n + 120, dict(l=85, r=45, t=95, b=45))
    fig.update_layout(hovermode="x unified")
    for a in fig.layout.annotations:
        a.update(x=0, xanchor="left", font=dict(size=12, color=_GRIS))
    return fig



# =============================================================================
#  3. TABLEAU DE BORD DASH - EXECUTION INLINE DANS JUPYTER
# =============================================================================
#
#  Cette version remplace UNIQUEMENT la couche ipywidgets par Dash.
#  Le socle de donnees et les regles metier restent ceux de la version
#  precedente :
#      - seules les anomalies (score_composite != 0) sont affichees ;
#      - les montants de ligne ne sont jamais sommes dans les panneaux ;
#      - les scores peuvent etre agreges uniquement pour la hierarchie du
#        sunburst, puisque cette agregation concerne la gravite et non les
#        montants metier.
#
#  IMPORTANT : dans Jupyter, Dash 2.11+ supporte l'affichage inline.
#  La derniere ligne app.run(jupyter_mode="inline") affiche donc le dashboard
#  directement dans la sortie de la cellule.
# =============================================================================


def _dash_css():
    return {
        "fontFamily": "Inter, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
        "color": _ENCRE,
        "backgroundColor": "#ffffff",
        "padding": "18px 20px 35px 20px",
        "maxWidth": "1600px",
        "margin": "0 auto",
    }


def _dash_bandeau(txt, fond="#eceff1", coul="#37474f"):
    return html.Div(
        [html.Span(txt)],
        style={
            "fontSize": "12.5px", "color": coul, "backgroundColor": fond,
            "padding": "9px 13px", "borderRadius": "6px",
            "margin": "14px 0 8px 0",
        },
    )


def _dash_cartes(sub, titre, score_global):
    part = float(sub[COL_SCORE].sum()) / float(score_global or 1.0) if len(sub) else 0
    hors = int((~((sub[COL_OBS] >= sub[COL_LO]) &
                  (sub[COL_OBS] <= sub[COL_HI]))).sum()) if len(sub) else 0
    cartes = [
        ("Anomalies", f"{len(sub):,}".replace(",", " "), "#37474f"),
        ("Part du score global", f"{100 * part:.1f} %", "#ad1457"),
        ("Hors intervalle", f"{hors:,}".replace(",", " "), "#00838f"),
        ("Gravite moyenne", _fmt4(sub[COL_SCORE].mean()) if len(sub) else "—", "#5e35b1"),
        ("Pire anomalie", _fmt4(sub[COL_SCORE].max()) if len(sub) else "—", "#6a1b9a"),
    ]
    blocs = []
    for lab, val, c in cartes:
        blocs.append(html.Div([
            html.Div(lab, style={
                "fontSize": "10.5px", "color": "#78909c",
                "textTransform": "uppercase", "letterSpacing": "0.6px"
            }),
            html.Div(val, style={
                "fontSize": "19px", "fontWeight": "600",
                "color": c, "marginTop": "4px"
            }),
        ], style={
            "flex": "1", "minWidth": "130px", "backgroundColor": "#fff",
            "border": "1px solid #e0e0e0", "borderLeft": f"5px solid {c}",
            "borderRadius": "7px", "padding": "11px 13px",
            "boxShadow": "0 1px 3px rgba(0,0,0,.07)"
        }))
    return html.Div([
        html.Div(titre, style={
            "fontSize": "15px", "fontWeight": "600", "color": "#263238",
            "marginBottom": "10px"
        }),
        html.Div(blocs, style={"display": "flex", "gap": "9px", "flexWrap": "wrap"})
    ], style={"margin": "6px 0 14px 0"})


def _dash_table(sub, titre, socle, axes):
    if not len(sub):
        return html.Div(f"Aucune anomalie dans le perimetre : {titre}",
                        style={"padding": "12px", "color": _GRIS})
    d = socle.preparer(sub.head(N_LIGNES_TABLE), axes)
    columns = ["Rang", "Maille", "Y_obs", "Y_pred", "CP_bas", "CP_haut", "Couvert", "Score"]
    rows = []
    for _, r in d.iterrows():
        rows.append([
            int(r["rang"]), r["libelle"], _fmt(r[COL_OBS]), _fmt(r[COL_PRED]),
            _fmt(r[COL_LO]), _fmt(r[COL_HI]),
            "Oui" if float(r[COL_LO]) <= float(r[COL_OBS]) <= float(r[COL_HI]) else "Non",
            _fmt4(r[COL_SCORE])
        ])
    header = html.Tr([html.Th(c, style={
        "padding": "8px 9px", "backgroundColor": "#f5f7fa",
        "borderBottom": "1px solid #dfe4ea", "textAlign": "left",
        "fontSize": "11px", "color": "#546e7a", "whiteSpace": "nowrap"
    }) for c in columns])
    body = []
    for row in rows:
        body.append(html.Tr([html.Td(v, style={
            "padding": "7px 9px", "borderBottom": "1px solid #edf1f7",
            "fontSize": "11.5px", "whiteSpace": "nowrap"
        }) for v in row]))
    return html.Div([
        html.Div(f"Tableau de priorisation — {titre}", style={
            "fontSize": "14px", "fontWeight": "600", "marginBottom": "8px"
        }),
        html.Div(html.Table([html.Thead(header), html.Tbody(body)],
                            style={"width": "100%", "borderCollapse": "collapse"}),
                 style={"overflowX": "auto", "border": "1px solid #e0e0e0",
                        "borderRadius": "6px"})
    ])


def _dash_cercle(sub, axes, titre):
    fig = _creer_cercle()
    h = _Socle_hierarchie_current(sub, axes)
    if not len(h):
        fig.update_layout(title=dict(text="Aucune anomalie a representer.", font=dict(size=14), x=.015, xanchor="left"), height=300)
        return fig
    cmax = float(np.nanpercentile(h["score_moyen"], 95))
    if not np.isfinite(cmax) or cmax <= 0:
        cmax = float(h["score_moyen"].max()) or 1.0
    fig.data[0].ids = h["id"].tolist()
    fig.data[0].labels = h["label"].tolist()
    fig.data[0].parents = h["parent"].tolist()
    fig.data[0].values = h["valeur_secteur"].tolist()
    fig.data[0].branchvalues = "remainder"
    fig.data[0].text = [f"{p:.0f} %" for p in h["part"]]
    fig.data[0].texttemplate = "%{label}<br>%{text}"
    fig.data[0].hovertext = [
        f"<b>{r['label']}</b><br>Gravite moyenne : {_fmt4(r['score_moyen'])}<br>"
        f"Anomalies : {int(r['n'])}<br>Score cumule : {_fmt4(r['score_total'])} "
        f"({r['part']:.1f} % du perimetre)<br>Pire anomalie : {_fmt4(r['score_max'])}"
        for _, r in h.iterrows()
    ]
    fig.data[0].hoverinfo = "text"
    fig.data[0].insidetextorientation = "radial"
    fig.data[0].maxdepth = len(axes)
    fig.data[0].marker = dict(
        colors=h["score_moyen"].tolist(), colorscale=ECHELLE, cmin=0, cmax=cmax,
        line=dict(color="white", width=1.6),
        colorbar=dict(title="Gravite<br>moyenne", thickness=16, len=.7, tickformat="~s")
    )
    fig.update_layout(
        height=620,
        title=dict(text=f"Repartition des anomalies  ·  {' › '.join(axes)}<br><sup>{titre}</sup>",
                   font=dict(size=15), x=.015, xanchor="left")
    )
    return fig


def _Socle_hierarchie_current(sub, axes):
    """Version autonome de hierarchie pour le callback Dash."""
    if not len(sub) or not axes:
        return pd.DataFrame()
    prof_max = len(axes)
    agg = {"score_total": (COL_SCORE, "sum"),
           "score_moyen": (COL_SCORE, "mean"),
           "score_max": (COL_SCORE, "max"), "n": (COL_SCORE, "size")}
    total = float(sub[COL_SCORE].sum()) or 1.0
    lignes = []
    for prof in range(1, prof_max + 1):
        cols = axes[:prof]
        g = sub.groupby(cols, observed=True).agg(**agg).reset_index()
        for _, r in g.iterrows():
            vals = [str(r[c]) for c in cols]
            st = float(r["score_total"])
            if not np.isfinite(st):
                continue
            lignes.append({
                "id": SEP.join(vals), "label": vals[-1],
                "parent": SEP.join(vals[:-1]) if prof > 1 else "",
                "profondeur": prof, "score_total": st,
                "valeur_secteur": st if prof == prof_max else 0.0,
                "score_moyen": float(r["score_moyen"]),
                "score_max": float(r["score_max"]), "n": int(r["n"]),
                "part": 100 * st / total
            })
    return pd.DataFrame(lignes)


def _dash_barres(sub, axes, titre, target):
    fig = _creer_barres()
    if not len(sub):
        fig.update_layout(title=dict(text=f"{titre}<br><sup>Aucune anomalie</sup>", font=dict(size=14), x=.015, xanchor="left"), height=260)
        return fig
    d = socle_global.preparer(sub.head(TOP_N_PANNEAUX), axes).iloc[::-1]
    montants = d[COL_OBS].astype(float).tolist()
    fig.data[0].x = montants
    fig.data[0].y = d["libelle_rang"].tolist()
    fig.data[0].marker.color = montants
    if montants:
        fig.data[0].marker.cmin, fig.data[0].marker.cmax = min(montants), max(montants)
    fig.data[0].text = [
        f"<b>{r['libelle_rang']}</b><br>{target} observe : {_fmt(r[COL_OBS])}<br>"
        f"Predit : {_fmt(r[COL_PRED])}<br>Intervalle : [{_fmt(r[COL_LO])} ; {_fmt(r[COL_HI])}]<br>"
        f"Score : {_fmt4(r[COL_SCORE])}"
        for _, r in d.iterrows()
    ]
    fig.data[0].hovertemplate = "%{text}<extra></extra>"
    fig.update_layout(
        height=max(380, 38 * len(d) + 150),
        xaxis_title=f"<b>{target}</b>",
        title=dict(text=f"Les {len(d)} anomalies les plus graves<br><sup>{titre}  ·  montants bruts, aucune sommation</sup>",
                   font=dict(size=14), x=.015, xanchor="left")
    )
    return fig


def _dash_forest(sub, axes, titre, target):
    fig = _creer_forest(target)
    if not len(sub):
        fig.update_layout(title=dict(text=f"{titre}<br><sup>Aucune anomalie</sup>", font=dict(size=14), x=.015, xanchor="left"), height=260)
        return fig
    d = socle_global.preparer(sub.head(TOP_N_PANNEAUX), axes).iloc[::-1].reset_index(drop=True)
    y = list(range(len(d)))
    lo = d[COL_LO].astype(float).to_numpy()
    hi = d[COL_HI].astype(float).to_numpy()
    obs = d[COL_OBS].astype(float).to_numpy()
    pred = d[COL_PRED].astype(float).to_numpy()
    xs_band, ys_band, xs_over, ys_over = [], [], [], []
    for yi, l, h, o in zip(y, lo, hi, obs):
        xs_band += [l, h, None]
        ys_band += [yi, yi, None]
        cible = h if o > h else l
        xs_over += [cible, o, None]
        ys_over += [yi, yi, None]
    textes = [
        f"<b>{r['libelle_rang']}</b><br>{target} observe : {_fmt(o)}<br>Predit : {_fmt(p)}<br>"
        f"Intervalle : [{_fmt(l)} ; {_fmt(h)}]"
        for (_, r), o, p, l, h in zip(d.iterrows(), obs, pred, lo, hi)
    ]
    fig.data[0].x, fig.data[0].y = xs_band, ys_band
    fig.data[1].x, fig.data[1].y = xs_over, ys_over
    fig.data[2].x, fig.data[2].y = pred.tolist(), y
    fig.data[2].text, fig.data[2].hovertemplate = textes, "%{text}<extra></extra>"
    fig.data[3].x, fig.data[3].y = obs.tolist(), y
    fig.data[3].text, fig.data[3].hovertemplate = textes, "%{text}<extra></extra>"
    fig.update_layout(
        height=max(420, 44 * len(d) + 175),
        xaxis_title=f"<b>{target}</b>",
        yaxis=dict(tickmode="array", tickvals=y,
                   ticktext=d["libelle_rang"].str.slice(0, 40).tolist(), tickfont=dict(size=10)),
        title=dict(text="Intervalle conforme, prediction et valeur observee<br>"
                        f"<sup>{titre}  ·  valeurs brutes de df, aucune sommation</sup>",
                   font=dict(size=14), x=.015, xanchor="left")
    )
    return fig


def _dash_evolution(hist, per, ctx, cle, target, alpha):
    fig = _creer_evolution(target)
    if hist is None or not len(hist):
        fig.update_layout(title=dict(text="Aucun historique pour ce sous-portefeuille.", font=dict(size=14), x=.015, xanchor="left"), height=260)
        return fig
    val = hist[target].astype(float).to_numpy()
    p_valide = ctx["per"] if ctx and ctx.get("per") in per else per[-1]
    couvert = bool(ctx["couvert"]) if ctx else True
    coul = _OK if couvert else _ACCENT
    halo = "rgba(61,90,158,0.18)" if couvert else "rgba(192,57,43,0.20)"
    fig.data[0].x, fig.data[0].y = per, val
    if ctx:
        fig.data[1].x, fig.data[1].y = [p_valide, p_valide], [ctx["lo"], ctx["hi"]]
        fig.data[1].name = f"Intervalle conforme {100 * (1 - alpha):.0f} %"
        fig.data[2].x, fig.data[2].y = [p_valide], [ctx["pred"]]
        fig.data[4].x, fig.data[4].y = [p_valide], [ctx["obs"]]
        fig.data[4].marker.color = halo
        fig.data[5].x, fig.data[5].y = [p_valide], [ctx["obs"]]
        fig.data[5].marker.color = coul
        fig.data[5].name = "Couvert" if couvert else "Hors intervalle"
    fig.data[3].x, fig.data[3].y = per, val
    fig.data[3].text = [_fmt(v) for v in val]
    fig.data[3].hovertemplate = "<b>%{x}</b><br>" + target + " : <b>%{y:,.0f}</b><extra></extra>"
    alerte = ("  ·  <b style='color:" + _ACCENT + "'>un seul trimestre dans df : pas de courbe possible</b>"
              if len(hist) < 2 else "")
    rang = ctx["rang"] if ctx else "?"
    fig.update_layout(
        height=420,
        title=dict(text=f"<b>#{rang}  {' | '.join(cle)}</b><br>"
                        f"<span style='font-size:11px'>{target} · {len(hist)} trimestre(s) · {per[0]} → {per[-1]}"
                        + (f" · periode validee <b>{p_valide}</b>" if ctx else "") + alerte + "</span>",
                   font=dict(size=14), x=.015, xanchor="left")
    )
    return fig


def _dash_variables(hist, per, ctx):
    fig = _creer_variables()
    if hist is None or len(hist) < 3:
        n = 0 if hist is None else len(hist)
        fig.update_layout(title=dict(text=f"Classement des variables indisponible : {n} trimestre(s) dans df, il en faut au moins 3.", font=dict(size=14), x=.015, xanchor="left"), height=260)
        return fig
    t = socle_global.variables_explicatives(hist)
    if not len(t):
        fig.update_layout(title=dict(text="Aucune variable explicative numerique exploitable.", font=dict(size=14), x=.015, xanchor="left"), height=260)
        return fig
    p_valide = ctx["per"] if ctx and ctx.get("per") in per else per[-1]
    k = per.index(p_valide)
    for i in range(N_VARS_EXPLIC):
        curve, point = fig.data[2*i], fig.data[2*i+1]
        if i < len(t):
            v = t.iloc[i]["variable"]
            vals = hist[v].astype(float).to_numpy()
            curve.x, curve.y = per, vals
            curve.hovertemplate = f"<b>%{{x}}</b><br>{v} : <b>%{{y:,.0f}}</b><extra></extra>"
            point.x, point.y = [per[k]], [vals[k]]
            fig.layout.annotations[i].text = f"<b>{v}</b>"
            fig.layout.annotations[i].font.color = _DOUX[i % len(_DOUX)]
        else:
            curve.x, curve.y = [], []
            point.x, point.y = [], []
            fig.layout.annotations[i].text = " "
    fig.update_layout(
        height=150 * max(len(t), 1) + 120,
        title=dict(text="<b>Les variables qui expliquent ce comportement</b>", font=dict(size=14), x=.015, xanchor="left")
    )
    return fig


def tableau_de_bord_unifie_dash(df, id_cols=None, target=None, alpha=None,
                                top_n=TOP_N_PANNEAUX):
    """Construit le dashboard Dash a partir de la meme base df.

    L'application est destinee a etre executee directement dans Jupyter :

        app, controle = tableau_de_bord_unifie_dash(df, ID_COLS, TARGET)
        app.run(jupyter_mode="inline", jupyter_width="100%", jupyter_height=1200)
    """
    global socle_global
    id_cols = id_cols if id_cols is not None else _session("ID_COLS")[1]
    target = target if target is not None else _session("TARGET")[1]
    if alpha is None:
        ok, a = _session("ALPHA")
        alpha = a if ok else .10
    if id_cols is None or target is None:
        raise NameError("ID_COLS et TARGET doivent exister dans la session.")

    socle_global = _Socle(df, id_cols, target, alpha)
    socle = socle_global
    cles = socle.cles
    defaut_maille = next((c for c in ("Lob", "Partner", "Companies", "Risk") if c in cles), cles[0])
    axes_defaut = cles[:2] if len(cles) >= 2 else cles[:1]

    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = "Tableau de bord des anomalies"

    app.layout = html.Div([
        html.H2("Tableau de bord unifie — anomalies, evolution, priorisation",
                style={"marginBottom": "4px", "color": _ENCRE}),
        html.Div(f"Source unique : df · {len(df):,} lignes · seules les anomalies sont affichees.",
                 style={"fontSize": "12px", "color": _GRIS, "marginBottom": "12px"}),

        _dash_bandeau(
            "Axes — ils structurent le cercle et composent les libelles. "
            "Ils ne modifient aucun montant : chaque ligne garde les valeurs de df. "
            "Un clic sur une part du cercle filtre le perimetre.",
            fond="#e3f2fd", coul="#0d47a1"),
        dcc.Checklist(
            id="dash-axes", options=[{"label": c, "value": c} for c in cles],
            value=axes_defaut, inline=True,
            style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "10px"},
            inputStyle={"marginRight": "4px"}
        ),
        dcc.Graph(id="dash-cercle", figure=_creer_cercle(),
                  style={"width": "100%"}, config={"displaylogo": False}),

        _dash_bandeau(
            f"Perimetre — la maille et la valeur filtrent l'ensemble du tableau de bord. "
            f"Seules les anomalies ({COL_SCORE} different de zero) sont affichees.",
            fond="#e8f5e9", coul="#1b5e20"),
        html.Div([
            html.Div([
                html.Label("Maille :", style={"fontWeight": "600", "fontSize": "12px"}),
                dcc.Dropdown(id="dash-maille", options=[{"label": c, "value": c} for c in cles],
                             value=defaut_maille, clearable=False)
            ], style={"flex": "0 0 300px"}),
            html.Div([
                html.Label("Valeur :", style={"fontWeight": "600", "fontSize": "12px"}),
                dcc.Dropdown(id="dash-valeur", options=[{"label": "— vue generale —", "value": VUE_GENERALE}],
                             value=VUE_GENERALE, clearable=False)
            ], style={"flex": "1", "minWidth": "320px"}),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),
        html.Div(id="dash-cartes"),
        dcc.Graph(id="dash-barres", figure=_creer_barres(), config={"displaylogo": False}),
        dcc.Graph(id="dash-forest", figure=_creer_forest(target), config={"displaylogo": False}),

        _dash_bandeau(
            "L'anomalie en detail — evolution de la cible sur son propre historique, "
            "avec l'intervalle conforme et la prediction de la ligne, puis les variables "
            "qui expliquent son comportement.",
            fond="#fff3e0", coul="#e65100"),
        html.Label("Anomalie :", style={"fontWeight": "600", "fontSize": "12px"}),
        dcc.Dropdown(id="dash-unite", options=[], value=None, clearable=False,
                     placeholder="Selectionner une anomalie"),
        dcc.Graph(id="dash-evolution", figure=_creer_evolution(target), config={"displaylogo": False}),
        dcc.Graph(id="dash-vars", figure=_creer_variables(), config={"displaylogo": False}),

        _dash_bandeau(f"Tableau de priorisation — {N_LIGNES_TABLE} anomalies les plus graves."),
        html.Div(id="dash-tableau"),
        html.Div(id="dash-message", style={"fontSize": "11px", "color": _GRIS, "marginTop": "8px"})
    ], style=_dash_css())

    @app.callback(
        Output("dash-valeur", "options"),
        Output("dash-valeur", "value"),
        Input("dash-maille", "value"),
        State("dash-valeur", "value")
    )
    def maj_valeurs(maille, ancienne):
        if not maille or maille not in socle.ano.columns:
            return [{"label": "— vue generale —", "value": VUE_GENERALE}], VUE_GENERALE
        g = (socle.ano.groupby(maille, observed=True)[COL_SCORE]
             .agg(["size", "sum"]).reset_index().sort_values("sum", ascending=False))
        opts = [{"label": "— vue generale —", "value": VUE_GENERALE}]
        opts += [{"label": f"{r[maille]}   ({int(r['size'])} anomalies)", "value": str(r[maille])}
                 for _, r in g.iterrows()]
        values = {x["value"] for x in opts}
        return opts, ancienne if ancienne in values else VUE_GENERALE

    @app.callback(
        Output("dash-cercle", "figure"),
        Output("dash-cartes", "children"),
        Output("dash-barres", "figure"),
        Output("dash-forest", "figure"),
        Output("dash-unite", "options"),
        Output("dash-unite", "value"),
        Output("dash-evolution", "figure"),
        Output("dash-vars", "figure"),
        Output("dash-tableau", "children"),
        Output("dash-message", "children"),
        Input("dash-axes", "value"),
        Input("dash-maille", "value"),
        Input("dash-valeur", "value"),
        Input("dash-unite", "value"),
        Input("dash-cercle", "clickData"),
    )
    def maj_dashboard(axes, maille, valeur, unite, click_data):
        ctx = callback_context
        triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
        axes = [a for a in (axes or []) if a in cles] or [cles[0]]

        # Clic sur le sunburst : on transforme le chemin clique en filtre
        # (meme comportement que fw_cercle.on_click dans la version ipywidgets).
        if triggered == "dash-cercle" and click_data and click_data.get("points"):
            point = click_data["points"][0]
            ident = point.get("id", "")
            parts = str(ident).split(SEP) if ident else []
            if parts and len(parts) <= len(axes):
                maille = axes[len(parts) - 1]
                valeur = parts[-1]

        sub, titre = socle.filtrer(maille, valeur)
        options_unites = []
        for _, r in socle.preparer(sub.head(N_UNITES_LISTE), axes).iterrows():
            cle = tuple(str(r[c]) for c in cles)
            options_unites.append({
                "label": f"{r['libelle_rang']}   ({_fmt(r[COL_OBS])})",
                "value": list(cle)
            })
        dispo = [o["value"] for o in options_unites]
        unite_courante = list(unite) if unite is not None else None
        if triggered != "dash-unite" or unite_courante not in dispo:
            unite_courante = dispo[0] if dispo else None

        cercle = _dash_cercle(sub, axes, titre)
        cartes = _dash_cartes(sub, titre, socle.score_global)
        barres = _dash_barres(sub, axes, titre, target)
        forest = _dash_forest(sub, axes, titre, target)

        r = socle.ligne(sub, tuple(unite_courante) if unite_courante is not None else None)
        contexte = socle.contexte(r)
        if r is None:
            evolution = _creer_evolution(target)
            variables = _creer_variables()
            evolution.update_layout(title=dict(text="Aucune anomalie dans ce perimetre.", font=dict(size=14), x=.015, xanchor="left"), height=260)
            variables.update_layout(title=dict(text="Aucune anomalie dans ce perimetre.", font=dict(size=14), x=.015, xanchor="left"), height=260)
        else:
            cle = tuple(str(r[c]) for c in cles)
            hist, per = socle.historique(cle)
            evolution = _dash_evolution(hist, per, contexte, cle, target, alpha)
            variables = _dash_variables(hist, per, contexte)

        tableau = _dash_table(sub, titre, socle, axes)
        message = (f"Perimetre courant : {titre} · {len(sub)} anomalie(s) · "
                   f"periode validee : {socle.periode[0]}-T{socle.periode[1]}" if socle.periode
                   else f"Perimetre courant : {titre} · {len(sub)} anomalie(s)")
        return cercle, cartes, barres, forest, options_unites, unite_courante, evolution, variables, tableau, message

    return app, {
        "socle": socle,
        "app": app,
        "cles": cles,
        "target": target,
        "alpha": alpha,
    }


# =============================================================================
#  4. EXECUTION AUTOMATIQUE DANS JUPYTER
# =============================================================================
#
#  Prerequis identiques a la version originale :
#      df, ID_COLS, TARGET       (ALPHA facultatif)
#
#  Le mode INLINE est volontaire : aucun mode external/tab n'est utilise.
# =============================================================================

_PREREQUIS_DASH = ["df", "ID_COLS", "TARGET"]
_trouve_dash = {n: _session(n) for n in _PREREQUIS_DASH}
_manquants_dash = [n for n, (ok, _) in _trouve_dash.items() if not ok]

if _manquants_dash:
    print("=" * 74)
    print("TABLEAU DE BORD DASH NON AFFICHE : variables absentes de la session")
    print("=" * 74)
    for n in _manquants_dash:
        print(f"  - {n}")
    print("\nExecutez d'abord les cellules qui creent ces variables, puis relancez celle-ci.")
else:
    try:
        app_dash, controles_dash = tableau_de_bord_unifie_dash(
            _trouve_dash["df"][1],
            id_cols=_trouve_dash["ID_COLS"][1],
            target=_trouve_dash["TARGET"][1]
        )
        # Dash 2.11+ : affichage directement dans la cellule Jupyter.
        app_dash.run(
            jupyter_mode="inline",
            jupyter_width="100%",
            jupyter_height=1250,
            debug=False,
            dev_tools_hot_reload=False
        )
    except ValueError as _e:
        print("=" * 74)
        print("TABLEAU DE BORD DASH NON AFFICHE")
        print("=" * 74)
        print(f"  {_e}")
