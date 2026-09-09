

# -*- coding: utf-8 -*-
# =============================================================================
#  TABLEAU DE BORD UNIFIE - anomalies, evolution, priorisation
#
#  UN SEUL BLOC, AUTONOME. Rien a executer avant, hormis d'avoir en session :
#      anomalies_prio, expl, df, ID_COLS, TARGET      (ALPHA facultatif)
#
#  CONTENU
#  -------
#    - cercle hierarchique (sunburst) pilote par 4 selecteurs de segment
#    - cartes de synthese
#    - barres des anomalies les plus critiques, agregees sur les axes actifs
#    - forest plot intervalle / prediction / valeur observee
#    - evolution de la cible, AVEC intervalle conforme et prediction
#    - les 5 variables qui expliquent le comportement de l'anomalie
#    - tableau de priorisation
#
#  MODIFICATIONS DE CETTE VERSION
#  -------------------------------
#  1. L'EVOLUTION SUIT MAINTENANT LES AXES D'AGREGATION. Avant, elle etait
#     figee sur la cle COMPLETE (toutes les dimensions de ID_COLS) : bouger
#     les boutons d'axes ne changeait rien pour elle. Desormais le selecteur
#     d'unite, l'historique, la prediction et l'intervalle conforme sont
#     construits sur les seuls axes actifs. Desactivez Risk, et les lignes qui
#     n'en differaient que par lui sont regroupees, historique compris.
#  2. LES 5 VARIABLES EXPLICATIVES, sous la courbe. Elles ne sont pas prises
#     au hasard : chaque variable numerique est notee par l'ampleur de sa
#     RUPTURE au trimestre valide, mesuree en ecarts inter-quartiles par
#     rapport a son propre passe. Les cinq plus fortes ruptures sont tracees
#     avec leur montant. C'est ce qui permet de dire "le probleme vient d'ici".
#  3. LES BARRES SONT COLOREES PAR LE MONTANT, plus par le rang normalise.
#  4. La prediction et l'intervalle conforme de la courbe d'evolution sont
#     conserves tels quels, comme convenu.
#
#  UN POINT DE RIGUEUR A CONNAITRE
#  --------------------------------
#  Quand on agrege sur moins d'axes que la cle complete, les bornes conformes
#  affichees sont la SOMME des bornes individuelles. Cette somme n'a plus la
#  garantie de couverture a 90 % du conforme : la couverture d'une somme
#  d'intervalles n'est pas celle de leurs composantes. Le titre le signale des
#  qu'une agregation a lieu. Pour lire un intervalle rigoureux, gardez tous
#  les axes actifs.
# =============================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

# ┌──────────────────────────── PARAMETRES ────────────────────────────┐
TOP_N_PANNEAUX = 12       # barres et forest plot
N_TRIMESTRES   = 10       # historique affiche
N_UNITES_LISTE = 30       # unites proposees dans le selecteur
N_LIGNES_TABLE = 15       # lignes du tableau du bas
N_VARS_EXPLIC  = 5        # variables explicatives sous la courbe
COL_SCORE      = "score_composite"
COL_GWP        = "GWP"
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
#  1. SOCLE DE DONNEES
# =============================================================================
class _Socle:
    """Prepare une fois ce que les mises a jour reliront des dizaines de fois."""

    def __init__(self, anomalies_prio, expl, df, id_cols, target, alpha):
        self.df, self.target, self.alpha = df, target, alpha
        self.cles = [c for c in id_cols
                     if c in anomalies_prio.columns and c in expl.columns
                     and c in df.columns]
        if not self.cles:
            raise ValueError(
                "Aucune colonne d'identification commune entre anomalies_prio, "
                f"expl et df. ID_COLS fourni : {list(id_cols)}")

        self.dd = anomalies_prio.copy()
        self.ex = expl.copy()
        for c in self.cles:
            self.dd[c] = self.dd[c].astype(str)
            self.ex[c] = self.ex[c].astype(str)
        if COL_SCORE in self.dd.columns:
            self.dd = self.dd.dropna(subset=[COL_SCORE])
        self.score_global = float(self.dd[COL_SCORE].sum()) \
            if COL_SCORE in self.dd.columns else 0.0

        #  Cles textuelles de df et expl, par axe. Construites une fois : sans
        #  elles, chaque changement de selection relirait df en entier.
        self._df_txt = {c: df[c].astype(str).values for c in self.cles}
        self._ex_txt = {c: self.ex[c].astype(str).values for c in self.cles}

        #  Variables explicatives candidates : numeriques, hors cible et hors
        #  colonnes de resultat.
        self.vars_explic = [
            c for c in df.columns
            if c != target and c not in _EXCLURE_VARS
            and pd.api.types.is_numeric_dtype(df[c])]

    # ------------------------------------------------------------ filtrage
    def filtrer(self, colonne, valeur):
        """Filtre dd et ex. Se replie sur la vue generale si la valeur ne
        correspond pas a la colonne : c'est l'etat transitoire d'un changement
        de maille, et filtrer dessus viderait tous les panneaux."""
        if not colonne or colonne not in self.dd.columns:
            return self.dd, self.ex, "vue generale"
        if not valeur:
            return self.dd, self.ex, f"{colonne} — vue generale"
        m = self.dd[colonne].astype(str) == str(valeur)
        if not m.any():
            return self.dd, self.ex, f"{colonne} — vue generale"
        return (self.dd[m],
                self.ex[self.ex[colonne].astype(str) == str(valeur)],
                f"{colonne} = {valeur}")

    # ------------------------------------------------------- agregation
    def agreger(self, sub, axes, top_n):
        """Agrege les anomalies sur les seules dimensions actives."""
        axes = [a for a in axes if a in sub.columns] or self.cles[:1]
        if not len(sub):
            return pd.DataFrame(columns=axes + ["score_total", "n", "libelle"])
        agg = {"score_total": (COL_SCORE, "sum"), "score_max": (COL_SCORE, "max"),
               "n": (COL_SCORE, "size")}
        if COL_GWP in sub.columns:
            agg["gwp"] = (COL_GWP, "sum")
        g = sub.groupby(axes, observed=True).agg(**agg).reset_index()
        g["libelle"] = g[axes].astype(str).agg(" | ".join, axis=1).str.slice(0, 38)
        return g.nlargest(min(top_n, len(g)), "score_total")

    # ------------------------------------------------------------ unites
    def unites(self, sub, axes, n=N_UNITES_LISTE):
        """Unites du perimetre, definies par les AXES ACTIFS et non plus par
        la cle complete. C'est ce qui rend le panneau d'evolution solidaire
        des boutons d'axes."""
        axes = [a for a in axes if a in sub.columns] or self.cles[:1]
        if not len(sub):
            return []
        g = self.agreger(sub, axes, n)
        options = []
        for _, r in g.iterrows():
            valeurs = tuple(str(r[a]) for a in axes)
            options.append((f"{' | '.join(valeurs)}   "
                            f"({_fmt4(r['score_total'])}"
                            + (f", {int(r['n'])} anomalies)" if r["n"] > 1
                               else ")"),
                            valeurs))
        return options

    # ------------------------------------------------- masques par axes
    def _masque(self, textes, axes, valeurs):
        m = np.ones(len(next(iter(textes.values()))), dtype=bool)
        for a, v in zip(axes, valeurs):
            m &= (textes[a] == str(v))
        return m

    # -------------------------------------------------------- historique
    def historique(self, valeurs, axes, n=N_TRIMESTRES):
        """Historique du groupe defini par (axes, valeurs).

        Quand les axes ne couvrent pas toute la cle, plusieurs lignes tombent
        dans le meme trimestre : on les somme. La somme est le bon agregat
        pour des montants, ce qui est le cas de la cible et des variables de
        volume. Elle serait fausse pour un taux, d'ou l'affichage explicite de
        "somme de N lignes" dans le titre.
        """
        axes = [a for a in axes if a in self._df_txt] or self.cles[:1]
        d = self.df[self._masque(self._df_txt, axes, valeurs)]
        if not len(d):
            return None, [], 0

        colonnes = [self.target] + [v for v in self.vars_explic if v in d.columns]
        cle_periode = ["year", "quarter"]
        g = d.groupby(cle_periode, observed=True)[colonnes].sum().reset_index()
        n_lignes = int(len(d) / max(len(g), 1))
        g = g.sort_values(cle_periode).tail(n)
        per = (g["year"].astype(int).astype(str) + "-T"
               + g["quarter"].astype(int).astype(str)).tolist()
        return g, per, n_lignes

    def contexte(self, valeurs, axes):
        """Prediction, bornes et statut du groupe, sommes sur ses lignes."""
        axes = [a for a in axes if a in self._ex_txt] or self.cles[:1]
        t = self.ex[self._masque(self._ex_txt, axes, valeurs)]
        if not len(t):
            return None
        lo, hi = float(t["borne_basse"].sum()), float(t["borne_haute"].sum())
        obs = float(t["y_obs"].sum())
        r = t.iloc[0]
        return dict(per=f"{int(r['year'])}-T{int(r['quarter'])}",
                    pred=float(t["y_pred"].sum()), lo=lo, hi=hi, obs=obs,
                    couvert=bool(lo <= obs <= hi), n_lignes=int(len(t)))

    # ------------------------------------ variables explicatives, notees
    def variables_explicatives(self, hist, n=N_VARS_EXPLIC):
        """Les n variables dont la RUPTURE au dernier trimestre est la plus
        forte, mesuree contre leur propre passe.

        Le critere n'est pas l'importance dans le modele mais l'ampleur du
        decrochage : (valeur du trimestre valide − mediane des trimestres
        precedents) rapportee a l'ecart inter-quartile de ces memes
        trimestres. Une variable stable qui saute brutalement remonte donc en
        tete, une variable naturellement volatile ne remonte que si elle sort
        franchement de son regime habituel. C'est ce qui permet de designer
        l'origine de l'anomalie plutot qu'une correlation generale.
        """
        if hist is None or len(hist) < 3:
            return pd.DataFrame(columns=["variable", "valeur", "mediane_passe",
                                         "variation", "z"])
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
            lignes.append({"variable": v, "valeur": courant,
                           "mediane_passe": med, "variation": courant - med,
                           "z": z})
        if not lignes:
            return pd.DataFrame(columns=["variable", "valeur", "mediane_passe",
                                         "variation", "z"])
        t = pd.DataFrame(lignes)
        return t.reindex(t["z"].abs().sort_values(ascending=False).index) \
                .head(n).reset_index(drop=True)

    # ------------------------------------------------------- hierarchie
    def hierarchie(self, chemin):
        """Noeuds du cercle, un par combinaison de chaque niveau.

        `valeur_secteur` vaut le score reel sur les FEUILLES et zero sur les
        parents. Avec branchvalues="remainder", Plotly additionne lui-meme les
        enfants : plus de contrainte "parent >= somme des enfants", donc plus
        de cercle blanc a partir de trois ou quatre niveaux.
        """
        prof_max = len(chemin)
        agg = {"score_total": (COL_SCORE, "sum"),
               "score_moyen": (COL_SCORE, "mean"),
               "score_max": (COL_SCORE, "max"), "n": (COL_SCORE, "size")}
        if COL_GWP in self.dd.columns:
            agg["gwp"] = (COL_GWP, "sum")

        lignes = []
        for prof in range(1, prof_max + 1):
            cols = chemin[:prof]
            g = self.dd.groupby(cols, observed=True).agg(**agg).reset_index()
            for _, r in g.iterrows():
                vals = [str(r[c]) for c in cols]
                total = float(r["score_total"])
                if not np.isfinite(total):
                    continue
                lignes.append({
                    "id": SEP.join(vals), "label": vals[-1],
                    "parent": SEP.join(vals[:-1]) if prof > 1 else "",
                    "profondeur": prof, "score_total": total,
                    "valeur_secteur": total if prof == prof_max else 0.0,
                    "score_moyen": float(r["score_moyen"]),
                    "score_max": float(r["score_max"]), "n": int(r["n"]),
                    "gwp": float(r["gwp"]) if "gwp" in g.columns else np.nan})
        return pd.DataFrame(lignes)


# =============================================================================
#  2. CREATION DES FIGURES (structure figee, seules les donnees changent)
# =============================================================================
def _creer_cercle():
    fig = go.Figure(go.Sunburst(ids=[], labels=[], parents=[], values=[],
                                branchvalues="remainder"))
    _mise_en_forme(fig, 620, dict(l=10, r=10, t=100, b=15))
    return go.FigureWidget(fig)


def _creer_barres():
    #  Couleur portee par le MONTANT lui-meme, plus par un rang normalise :
    #  la barre la plus longue est aussi la plus foncee, sans intermediaire.
    fig = go.Figure(go.Bar(x=[], y=[], orientation="h", showlegend=False,
                           marker=dict(colorscale=ECHELLE,
                                       line=dict(width=.5, color="white"),
                                       colorbar=dict(title="Montant",
                                                     thickness=14, len=.7,
                                                     tickformat="~s"))))
    fig.update_layout(xaxis_title="Score cumule",
                      xaxis=dict(tickformat="~s"),
                      yaxis=dict(tickfont=dict(size=10)))
    _mise_en_forme(fig, 520, dict(l=10, r=40, t=95, b=50))
    return go.FigureWidget(fig)


def _creer_forest():
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
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="center", x=.5))
    _mise_en_forme(fig, 520, dict(l=10, r=50, t=115, b=55))
    return go.FigureWidget(fig)


def _creer_evolution(target):
    """Courbe d'evolution. Les traces 1 et 2 portent l'intervalle conforme et
    la prediction : elles sont conservees telles quelles, comme convenu."""
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
    return go.FigureWidget(fig)


def _creer_variables(n=N_VARS_EXPLIC):
    """Un panneau par variable explicative, sous la courbe d'evolution."""
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
        a.update(x=0, xanchor="left", font=dict(size=11, color=_GRIS))
    return go.FigureWidget(fig)


def _vider(fw, message, hauteur=260):
    """Vide vraiment : x, y, text et annotations."""
    with fw.batch_update():
        for t in fw.data:
            t.x, t.y = [], []
            if "text" in t:
                t.text = []
            if t.type == "sunburst":
                t.ids, t.labels, t.parents, t.values = [], [], [], []
        for a in fw.layout.annotations:
            a.text = " "
        fw.layout.title = dict(text=message, font=dict(size=14), x=.015,
                               xanchor="left")
        fw.layout.height = hauteur


# =============================================================================
#  3. LE TABLEAU DE BORD
# =============================================================================
def tableau_de_bord_unifie(anomalies_prio, expl, df, id_cols=None, target=None,
                           alpha=None, top_n=TOP_N_PANNEAUX):
    id_cols = id_cols if id_cols is not None else _session("ID_COLS")[1]
    target = target if target is not None else _session("TARGET")[1]
    if alpha is None:
        ok, a = _session("ALPHA")
        alpha = a if ok else .10
    if id_cols is None or target is None:
        raise NameError("ID_COLS et TARGET doivent exister dans la session.")

    socle = _Socle(anomalies_prio, expl, df, id_cols, target, alpha)
    cles = socle.cles

    # ------------------------------------------------------------- widgets
    prefs = [c for c in ("Lob", "Partner", "Companies", "Risk") if c in cles]
    defauts = [(prefs[0] if prefs else cles[0]), None, None, None]
    niveaux = [widgets.Dropdown(
        options=[("— aucun —", None)] + [(c, c) for c in cles],
        value=defauts[i], description=f"Segment {i + 1} :",
        layout=widgets.Layout(width="250px"),
        style={"description_width": "72px"}) for i in range(4)]

    sel_maille = widgets.Dropdown(
        options=[(c, c) for c in cles], value=defauts[0],
        description="Maille :", layout=widgets.Layout(width="330px"),
        style={"description_width": "72px"})
    sel_valeur = widgets.Dropdown(
        options=[("— vue generale —", VUE_GENERALE)], value=VUE_GENERALE,
        description="Valeur :", layout=widgets.Layout(width="470px"),
        style={"description_width": "72px"})
    sel_unite = widgets.Dropdown(
        options=[], description="Unite :",
        layout=widgets.Layout(width="700px"),
        style={"description_width": "72px"})

    axes_btns = [widgets.ToggleButton(
        value=True, description=c, layout=widgets.Layout(width="auto"),
        button_style="info") for c in cles]

    sel_echelle = widgets.ToggleButtons(
        options=[("Ecart normalise", "z"), ("Montants", "lin")], value="z",
        style={"button_width": "auto"}, layout=widgets.Layout(width="auto"))

    # ------------------------------------------------------------ figures
    fw_cercle = _creer_cercle()
    fw_barres = _creer_barres()
    fw_forest = _creer_forest()
    fw_evol = _creer_evolution(target)
    fw_vars = _creer_variables()
    z_cartes, z_table = widgets.Output(), widgets.Output()
    verrou = {"actif": False}

    def _axes_actifs():
        actifs = [c for c, b in zip(cles, axes_btns) if b.value]
        return actifs or [cles[0]]        # jamais zero axe

    def _chemin():
        vus, out = set(), []
        for w in niveaux:
            if w.value and w.value not in vus:
                out.append(w.value)
                vus.add(w.value)
        return out

    # --------------------------------------------------------- le cercle
    def _maj_cercle(*_):
        chemin = _chemin()
        if not chemin:
            chemin = [defauts[0]]
        try:
            h = socle.hierarchie(chemin)
            if not len(h):
                _vider(fw_cercle, "Aucune anomalie a representer.", 300)
                return
            cmax = float(np.nanpercentile(h["score_moyen"], 95))
            if not np.isfinite(cmax) or cmax <= 0:
                cmax = float(h["score_moyen"].max()) or 1.0
            survol = [
                f"<b>{r['label']}</b><br>"
                f"Gravite moyenne : {_fmt4(r['score_moyen'])}<br>"
                f"Anomalies : {int(r['n'])}<br>"
                f"Score cumule : {_fmt4(r['score_total'])} "
                f"({100 * r['score_total'] / max(socle.score_global, 1e-12):.1f} %"
                " du total)<br>"
                f"Pire anomalie : {_fmt4(r['score_max'])}"
                + (f"<br>{COL_GWP} : {_fmt(r['gwp'])}"
                   if pd.notna(r.get("gwp")) else "")
                for _, r in h.iterrows()]
            with fw_cercle.batch_update():
                t = fw_cercle.data[0]
                t.ids, t.labels = h["id"].tolist(), h["label"].tolist()
                t.parents = h["parent"].tolist()
                t.values = h["valeur_secteur"].tolist()
                t.branchvalues = "remainder"
                t.text = [f"{100 * v / max(socle.score_global, 1e-12):.0f} %"
                          for v in h["score_total"]]
                t.texttemplate = "%{label}<br>%{text}"
                t.hovertext, t.hoverinfo = survol, "text"
                t.insidetextorientation = "radial"
                t.maxdepth = len(chemin)
                t.marker = dict(
                    colors=h["score_moyen"].tolist(), colorscale=ECHELLE,
                    cmin=0, cmax=cmax, line=dict(color="white", width=1.6),
                    colorbar=dict(title="Gravite<br>moyenne", thickness=16,
                                  len=.7, tickformat="~s"))
                fw_cercle.layout.height = 620
                fw_cercle.layout.title = dict(
                    text=f"Repartition des anomalies  ·  {' › '.join(chemin)}",
                    font=dict(size=15), x=.015, xanchor="left")
        except Exception as e:
            print(f"Cercle non mis a jour : {type(e).__name__} : {str(e)[:120]}")

    # --------------------------------------------------------- les cartes
    def _cartes(sub, titre, sub_ex):
        part = (sub[COL_SCORE].sum() / max(socle.score_global, 1e-12)
                if COL_SCORE in sub.columns else 0)
        gwp = (_fmt(sub[COL_GWP].sum()) if COL_GWP in sub.columns
               and sub[COL_GWP].notna().any() else "n/a")
        grav = _fmt4(sub[COL_SCORE].mean()) if len(sub) else "—"
        pire = "—"
        if len(sub) and "rank" in sub.columns:
            rk = sub.loc[sub[COL_SCORE].idxmax(), "rank"]
            if pd.notna(rk):
                pire = f"#{int(rk)}"
        couv = (f"{100 * sub_ex['dans_intervalle'].mean():.1f} %"
                if sub_ex is not None and len(sub_ex) else "n/a")
        cartes = [("Anomalies", f"{len(sub):,}".replace(",", " "), "#37474f"),
                  ("Part du score global", f"{100 * part:.1f} %", "#ad1457"),
                  (f"{COL_GWP} total", gwp, "#2e7d32"),
                  ("Couverture CQR", couv, "#00838f"),
                  ("Gravite moyenne", grav, "#5e35b1"),
                  ("Pire anomalie", pire, "#6a1b9a")]
        blocs = "".join(
            f"<div style='flex:1;min-width:130px;background:#fff;"
            f"border:1px solid #e0e0e0;border-left:5px solid {c};"
            f"border-radius:7px;padding:11px 13px;"
            f"box-shadow:0 1px 3px rgba(0,0,0,.07)'>"
            f"<div style='font-size:10.5px;color:#78909c;"
            f"text-transform:uppercase;letter-spacing:.6px'>{t}</div>"
            f"<div style='font-size:19px;font-weight:600;color:{c};"
            f"margin-top:4px'>{v}</div></div>" for t, v, c in cartes)
        return HTML(
            f"<div style='font-family:system-ui,sans-serif;margin:6px 0 14px 0'>"
            f"<div style='font-size:15px;font-weight:600;color:#263238;"
            f"margin-bottom:10px'>{titre}</div>"
            f"<div style='display:flex;gap:9px;flex-wrap:wrap'>{blocs}</div></div>")

    # ---------------------------------------------------------- les barres
    def _maj_barres(sub, titre):
        axes = _axes_actifs()
        g = socle.agreger(sub, axes, top_n)
        if not len(g):
            _vider(fw_barres, f"{titre}<br><sup>Aucune anomalie</sup>", 260)
            return
        g = g.iloc[::-1]                       # plus grave en haut
        survol = [
            f"<b>{r['libelle']}</b><br>Score cumule : {_fmt4(r['score_total'])}"
            f"<br>Anomalies regroupees : {int(r['n'])}"
            f"<br>Pire : {_fmt4(r['score_max'])}"
            + (f"<br>{COL_GWP} : {_fmt(r['gwp'])}"
               if "gwp" in g.columns and pd.notna(r.get("gwp")) else "")
            for _, r in g.iterrows()]
        montants = g["score_total"].tolist()
        with fw_barres.batch_update():
            t = fw_barres.data[0]
            t.x, t.y = montants, g["libelle"].tolist()
            #  Le montant sert directement d'echelle de couleur. Le rang
            #  normalise a ete retire : il n'apportait rien de lisible.
            t.marker.color = montants
            t.marker.cmin, t.marker.cmax = min(montants), max(montants)
            t.text, t.hovertemplate = survol, "%{text}<extra></extra>"
            fw_barres.layout.height = max(380, 38 * len(g) + 150)
            fw_barres.layout.title = dict(
                text=f"Les {len(g)} plus critiques  ·  agrege sur "
                     f"{' + '.join(axes)}<br><sup>{titre}</sup>",
                font=dict(size=14), x=.015, xanchor="left")

    # ---------------------------------------------------------- le forest
    def _maj_forest(sub, titre):
        if not len(sub):
            _vider(fw_forest, f"{titre}<br><sup>Aucune anomalie</sup>", 260)
            return
        axes = _axes_actifs()
        d = sub.nlargest(min(top_n, len(sub)), COL_SCORE).iloc[::-1] \
            .reset_index(drop=True)
        y = list(range(len(d)))
        lo = d["borne_basse"].values.astype(float)
        hi = d["borne_haute"].values.astype(float)
        obs = d["y_obs"].values.astype(float)
        pred = d["y_pred"].values.astype(float)

        if sel_echelle.value == "z":
            demi = np.maximum((hi - lo) / 2, 1e-12)
            centre = (hi + lo) / 2
            lo_t, hi_t = np.full(len(d), -1.0), np.full(len(d), 1.0)
            obs_t, pred_t = (obs - centre) / demi, (pred - centre) / demi
            titre_x = "Ecart normalise  z = (observe − centre) / demi-largeur"
        else:
            lo_t, hi_t, obs_t, pred_t = lo, hi, obs, pred
            titre_x = target

        xs_band, ys_band, xs_over, ys_over = [], [], [], []
        for yi, l, h, o in zip(y, lo_t, hi_t, obs_t):
            xs_band += [l, h, None]
            ys_band += [yi, yi, None]
            cible = h if o > h else l
            xs_over += [cible, o, None]
            ys_over += [yi, yi, None]

        libelles = d[[c for c in axes if c in d.columns]].astype(str) \
            .agg(" | ".join, axis=1).str.slice(0, 34).tolist()
        ticks = [f"#{int(r)}  {lab}" if pd.notna(r) else lab
                 for r, lab in zip(d.get("rank", pd.Series([np.nan] * len(d))),
                                   libelles)]
        textes = [
            f"<b>{lab}</b><br>Observe : {_fmt(o)}<br>Predit : {_fmt(p)}"
            f"<br>Intervalle : [{_fmt(l)} ; {_fmt(h)}]"
            f"<br>Ecart normalise : {(o - (h + l) / 2) / max((h - l) / 2, 1e-12):.2f}"
            for lab, o, p, l, h in zip(libelles, obs, pred, lo, hi)]

        with fw_forest.batch_update():
            fw_forest.data[0].x, fw_forest.data[0].y = xs_band, ys_band
            fw_forest.data[1].x, fw_forest.data[1].y = xs_over, ys_over
            fw_forest.data[2].x, fw_forest.data[2].y = pred_t, y
            fw_forest.data[2].text = textes
            fw_forest.data[2].hovertemplate = "%{text}<extra></extra>"
            fw_forest.data[3].x, fw_forest.data[3].y = obs_t, y
            fw_forest.data[3].text = textes
            fw_forest.data[3].hovertemplate = "%{text}<extra></extra>"
            fw_forest.layout.xaxis.type = "linear"
            fw_forest.layout.xaxis.title.text = titre_x
            fw_forest.layout.xaxis.tickformat = "~s" if sel_echelle.value == "lin" \
                else ""
            fw_forest.layout.yaxis = dict(tickmode="array", tickvals=y,
                                          ticktext=ticks, tickfont=dict(size=10))
            fw_forest.layout.height = max(420, 44 * len(d) + 175)
            fw_forest.layout.title = dict(
                text=f"Intervalle conforme, prediction et valeur observee"
                     f"<br><sup>{titre}</sup>",
                font=dict(size=14), x=.015, xanchor="left")

    # ------------------------------------- evolution + variables explicatives
    def _maj_unite(*_):
        valeurs = sel_unite.value
        axes = _axes_actifs()
        if valeurs is None:
            _vider(fw_evol, "Aucun sous-portefeuille dans ce perimetre.")
            _vider(fw_vars, "Aucun sous-portefeuille dans ce perimetre.")
            return
        try:
            hist, per, n_lignes = socle.historique(valeurs, axes)
            ctx = socle.contexte(valeurs, axes)
            _maj_evolution(hist, per, ctx, valeurs, axes, n_lignes)
            _maj_variables(hist, per, ctx, valeurs)
        except Exception as e:
            msg = f"Mise a jour impossible : {type(e).__name__} : {str(e)[:110]}"
            _vider(fw_evol, msg)
            _vider(fw_vars, msg)

    def _maj_evolution(hist, per, ctx, valeurs, axes, n_lignes):
        if hist is None or not len(hist):
            _vider(fw_evol, "Aucun historique pour ce sous-portefeuille.")
            return
        val = hist[target].to_numpy(dtype="float64")
        xb = yb = xp = yp = xh = yh = []
        if ctx and ctx["per"] in per:
            xb, yb = [ctx["per"], ctx["per"]], [ctx["lo"], ctx["hi"]]
            xp, yp = [ctx["per"]], [ctx["pred"]]
            xh, yh = [ctx["per"]], [ctx["obs"]]
        couvert = bool(ctx["couvert"]) if ctx else True
        coul = _OK if couvert else _ACCENT
        halo = "rgba(61,90,158,0.18)" if couvert else "rgba(192,57,43,0.20)"
        delta = 100 * (val[-1] - val[0]) / abs(val[0]) if val[0] else np.nan
        fleche = "▲" if (np.isfinite(delta) and delta >= 0) else "▼"
        agrege = len(axes) < len(cles)
        with fw_evol.batch_update():
            fw_evol.data[0].x, fw_evol.data[0].y = per, val
            fw_evol.data[1].x, fw_evol.data[1].y = xb, yb
            fw_evol.data[1].name = f"Intervalle conforme {100 * (1 - alpha):.0f} %"
            fw_evol.data[2].x, fw_evol.data[2].y = xp, yp
            fw_evol.data[3].x, fw_evol.data[3].y = per, val
            fw_evol.data[3].text = [_fmt(v) for v in val]
            fw_evol.data[3].hovertemplate = ("<b>%{x}</b><br>" + target
                                             + " : <b>%{y:,.0f}</b><extra></extra>")
            fw_evol.data[4].x, fw_evol.data[4].y = xh, yh
            fw_evol.data[4].marker.color = halo
            fw_evol.data[5].x, fw_evol.data[5].y = xh, yh
            fw_evol.data[5].marker.color = coul
            fw_evol.data[5].name = "Couvert" if couvert else "Hors intervalle"
            fw_evol.layout.height = 420
            fw_evol.layout.title = dict(
                text=f"<b style='color:{_ENCRE}'>{' | '.join(valeurs)}</b>"
                     f"<br><span style='font-size:11px'>{target} · {len(hist)} "
                     f"trimestres · {per[0]} → {per[-1]} · "
                     f"<span style='color:{_ACCENT if delta < 0 else _OK}'>"
                     f"{fleche} {abs(delta):.1f} %</span>"
                     f" · axes {' + '.join(axes)}"
                     + (f" · <b style='color:{_ACCENT}'>somme de "
                        f"{ctx['n_lignes'] if ctx else n_lignes} lignes, "
                        "bornes conformes additionnees</b>" if agrege else "")
                     + (f" · periode validee <b>{ctx['per']}</b>" if ctx else "")
                     + "</span>", font=dict(size=14), x=.015, xanchor="left")

    def _maj_variables(hist, per, ctx, valeurs):
        if hist is None or len(hist) < 3:
            _vider(fw_vars, "Historique trop court pour classer les variables.")
            return
        t = socle.variables_explicatives(hist)
        if not len(t):
            _vider(fw_vars, "Aucune variable explicative numerique exploitable "
                            "dans l'historique.")
            return
        k = per.index(ctx["per"]) if ctx and ctx["per"] in per else len(per) - 1
        with fw_vars.batch_update():
            for i in range(N_VARS_EXPLIC):
                t_l, t_p = fw_vars.data[2 * i], fw_vars.data[2 * i + 1]
                ann = fw_vars.layout.annotations[i]
                if i < len(t):
                    r = t.iloc[i]
                    v = r["variable"]
                    vals = hist[v].to_numpy(dtype="float64")
                    t_l.x, t_l.y = per, vals
                    t_l.hovertemplate = (f"<b>%{{x}}</b><br>{v} : "
                                         "<b>%{y:,.0f}</b><extra></extra>")
                    t_p.x, t_p.y = [per[k]], [vals[k]]
                    sens = "au-dessus de" if r["variation"] >= 0 else "sous"
                    ann.text = (
                        f"<b>{v}</b>   {_fmt(r['valeur'])}   ·   "
                        f"{_fmt(abs(r['variation']))} {sens} sa mediane "
                        f"({_fmt(r['mediane_passe'])})   ·   "
                        f"rupture {abs(r['z']):.1f} ecarts")
                    ann.font.color = _DOUX[i % len(_DOUX)]
                else:
                    t_l.x, t_l.y = [], []
                    t_p.x, t_p.y = [], []
                    ann.text = " "
            fw_vars.layout.height = 150 * max(len(t), 1) + 120
            fw_vars.layout.title = dict(
                text="<b>Ce qui explique le comportement de cette anomalie</b>"
                     f"<br><span style='font-size:11px'>les {len(t)} variables "
                     "dont la valeur du trimestre valide s'ecarte le plus de "
                     "leur propre passe, mesure en ecarts inter-quartiles · "
                     f"periode <b>{per[k]}</b></span>",
                font=dict(size=14), x=.015, xanchor="left")

    # ----------------------------------------------------------- le tableau
    def _maj_tableau(sub, titre):
        with z_table:
            clear_output(wait=True)
            try:
                if not len(sub):
                    print(f"Aucune anomalie dans le perimetre : {titre}")
                    return
                axes = _axes_actifs()
                d = sub.nlargest(min(N_LIGNES_TABLE, len(sub)), COL_SCORE).copy()
                t = pd.DataFrame({
                    "Rang": (d["rank"].astype("Int64") if "rank" in d
                             else range(1, len(d) + 1)),
                    "Maille": d[[c for c in axes if c in d.columns]]
                    .astype(str).agg(" | ".join, axis=1),
                    "Y_obs": d["y_obs"], "Y_pred": d["y_pred"],
                    "CP_bas": d["borne_basse"], "CP_haut": d["borne_haute"],
                    "Couvert": d["dans_intervalle"], "Score": d[COL_SCORE]})
                if COL_GWP in d.columns:
                    t[COL_GWP] = d[COL_GWP]
                t = t.reset_index(drop=True)
                fmt_col = {c: (lambda v: _fmt(v))
                           for c in ("Y_obs", "Y_pred", "CP_bas", "CP_haut")}
                fmt_col["Score"] = lambda v: _fmt4(v)
                if COL_GWP in t.columns:
                    fmt_col[COL_GWP] = lambda v: _fmt(v)
                try:
                    display(t.style
                            .background_gradient(subset=["Score"], cmap="Reds")
                            .format(fmt_col)
                            .set_caption(f"Tableau de priorisation — {titre}"))
                except Exception:
                    display(t)
            except Exception as e:
                print(f"Tableau non genere : {type(e).__name__} : {str(e)[:150]}")

    # -------------------------------------------------------- orchestration
    def _options_valeurs(colonne):
        g = (socle.dd.groupby(colonne, observed=True)[COL_SCORE]
             .agg(["size", "sum"]).reset_index()
             .sort_values("sum", ascending=False))
        return [("— vue generale —", VUE_GENERALE)] + [
            (f"{r[colonne]}   ({int(r['size'])} anomalies)", str(r[colonne]))
            for _, r in g.iterrows()]

    def _maj_panneaux(*_):
        if verrou["actif"]:
            return
        axes = _axes_actifs()
        sub, sub_ex, titre = socle.filtrer(sel_maille.value, sel_valeur.value)
        with z_cartes:
            clear_output(wait=True)
            display(_cartes(sub, titre, sub_ex))
        _maj_barres(sub, titre)
        _maj_forest(sub, titre)

        #  Le selecteur d'unite est reconstruit sur les AXES ACTIFS : c'est ce
        #  qui rend l'evolution solidaire des boutons d'axes.
        options = socle.unites(sub, axes)
        verrou["actif"] = True
        try:
            ancienne = sel_unite.value
            sel_unite.options = options
            dispo = [v for _, v in options]
            sel_unite.value = (ancienne if ancienne in dispo
                               else (dispo[0] if dispo else None))
        finally:
            verrou["actif"] = False
        _maj_unite()
        _maj_tableau(sub, titre)

    def _maj_valeurs(*_):
        if verrou["actif"]:
            return
        verrou["actif"] = True
        try:
            sel_valeur.options = _options_valeurs(sel_maille.value)
            sel_valeur.value = VUE_GENERALE
        finally:
            verrou["actif"] = False
        _maj_panneaux()

    def _au_clic_cercle(trace, points, state):
        if not points.point_inds:
            return
        parts = trace.ids[points.point_inds[0]].split(SEP)
        chemin = _chemin() or [defauts[0]]
        if not parts or len(parts) > len(chemin):
            return
        colonne, valeur = chemin[len(parts) - 1], parts[-1]
        verrou["actif"] = True
        try:
            sel_maille.value = colonne
            sel_valeur.options = _options_valeurs(colonne)
            dispo = [v for _, v in sel_valeur.options]
            sel_valeur.value = valeur if valeur in dispo else VUE_GENERALE
        finally:
            verrou["actif"] = False
        _maj_panneaux()

    # ------------------------------------------------------- branchements
    for w in niveaux:
        w.observe(lambda c: _maj_cercle() if c["name"] == "value" else None,
                  names="value")
    sel_maille.observe(lambda c: _maj_valeurs() if c["name"] == "value" else None,
                       names="value")
    sel_valeur.observe(lambda c: _maj_panneaux() if c["name"] == "value" else None,
                       names="value")
    sel_unite.observe(
        lambda c: (_maj_unite() if not verrou["actif"] else None)
        if c["name"] == "value" else None, names="value")
    for b in axes_btns:
        b.observe(lambda c: _maj_panneaux() if c["name"] == "value" else None,
                  names="value")
    sel_echelle.observe(lambda c: _maj_panneaux() if c["name"] == "value" else None,
                        names="value")
    try:
        fw_cercle.data[0].on_click(_au_clic_cercle)
        clic = True
    except Exception:
        clic = False

    # ---------------------------------------------------------- affichage
    display(_bandeau(
        "<b>Structure du cercle</b> — choisissez jusqu'a quatre niveaux "
        "d'emboitement." + ("  Le clic sur une part filtre les panneaux."
                            if clic else ""),
        fond="#e3f2fd", coul="#0d47a1"))
    display(widgets.HBox(niveaux[:2]))
    display(widgets.HBox(niveaux[2:]))
    display(fw_cercle)

    display(_bandeau(
        "<b>Perimetre et granularite</b> — la maille et la valeur filtrent. "
        "Les boutons d'axes commandent l'agregation des barres, du forest plot, "
        "du tableau <b>et de la courbe d'evolution</b> : desactivez-en un pour "
        "regrouper les anomalies qui n'en differaient que par lui.",
        fond="#e8f5e9", coul="#1b5e20"))
    display(widgets.HBox([sel_maille, sel_valeur]))
    display(widgets.HBox(
        [widgets.HTML("<b style='font-size:12.5px;color:#37474f;"
                      "padding-right:8px'>Axes d'agregation :</b>")] + axes_btns,
        layout=widgets.Layout(flex_flow="row wrap", align_items="center")))
    display(widgets.HBox(
        [widgets.HTML("<b style='font-size:12.5px;color:#37474f;"
                      "padding-right:8px'>Axe du forest plot :</b>"),
         sel_echelle], layout=widgets.Layout(align_items="center")))
    display(z_cartes)
    display(fw_barres)
    display(fw_forest)

    display(_bandeau(
        "<b>Le sous-portefeuille en detail</b> — evolution de la cible sur "
        f"{N_TRIMESTRES} trimestres avec l'intervalle conforme et la "
        "prediction, puis les variables qui expliquent son comportement. "
        "Cette section suit les axes d'agregation.",
        fond="#fff3e0", coul="#e65100"))
    display(sel_unite)
    display(fw_evol)
    display(fw_vars)

    display(_bandeau("<b>Tableau de priorisation</b> — perimetre courant, "
                     f"{N_LIGNES_TABLE} lignes les plus graves."))
    display(z_table)

    _maj_cercle()
    _maj_valeurs()
    return {"cercle": fw_cercle, "barres": fw_barres, "forest": fw_forest,
            "evolution": fw_evol, "variables": fw_vars, "maille": sel_maille,
            "valeur": sel_valeur, "unite": sel_unite, "axes": axes_btns,
            "echelle": sel_echelle, "niveaux": niveaux, "cartes": z_cartes,
            "tableau": z_table, "socle": socle, "rafraichir": _maj_panneaux}


# =============================================================================
#  4. EXECUTION AUTOMATIQUE
# =============================================================================
_PREREQUIS = ["anomalies_prio", "expl", "df", "ID_COLS", "TARGET"]
_trouve = {n: _session(n) for n in _PREREQUIS}
_manquants = [n for n, (ok, _) in _trouve.items() if not ok]

if _manquants:
    print("=" * 74)
    print("TABLEAU DE BORD NON AFFICHE : variables absentes de la session")
    print("=" * 74)
    for n in _manquants:
        print(f"  - {n}")
    print("\nExecutez d'abord les cellules qui les creent, puis relancez "
          "celle-ci.")
else:
    controles = tableau_de_bord_unifie(
        _trouve["anomalies_prio"][1], _trouve["expl"][1], _trouve["df"][1],
        id_cols=_trouve["ID_COLS"][1], target=_trouve["TARGET"][1])
















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
