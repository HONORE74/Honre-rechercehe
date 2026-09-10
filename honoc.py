# -*- coding: utf-8 -*-
# =============================================================================
#  TABLEAU DE BORD UNIFIE - anomalies, evolution, priorisation
#
#  UN SEUL BLOC, AUTONOME. Rien a executer avant, hormis d'avoir en session :
#      anomalies_prio, expl, df, ID_COLS, TARGET      (ALPHA facultatif)
#
#  PILOTAGE
#  --------
#  Les AXES D'AGREGATION commandent tout : le cercle, les barres, le forest
#  plot, la courbe d'evolution et le tableau. Activer ou desactiver une
#  dimension regroupe ou eclate l'ensemble du tableau de bord d'un seul geste.
#  La maille et la valeur filtrent ce peritmetre, et elles rafraichissent elles
#  aussi tous les panneaux, cercle compris.
#
#  MODIFICATIONS DE CETTE VERSION
#  -------------------------------
#  1. UNE SEULE SOURCE POUR LES TROIS PANNEAUX. C'etait le vrai defaut : la
#     courbe d'evolution lisait `expl` AGREGE sur les axes actifs, alors que le
#     forest plot et le tableau lisaient `anomalies_prio` LIGNE PAR LIGNE sur
#     la cle complete. Trois panneaux, trois `y_obs` differents pour le meme
#     sous-portefeuille. Desormais une methode unique, `table_axes()`, agrege
#     `expl` sur les axes actifs, et les trois panneaux la partagent. Le point
#     observe de la courbe, celui du forest plot et la colonne Y_obs du
#     tableau portent maintenant exactement la meme valeur.
#  2. Les selecteurs de segment sont supprimes : ce sont les axes d'agregation
#     qui structurent le cercle.
#  3. Le ruban "Ecart normalise / Montants" est supprime. Le forest plot est
#     toujours en montants, en notation courte.
#  4. La carte "GWP total" est retiree des cartes de synthese.
#  5. Les variables explicatives n'affichent plus que leur nom.
#  6. Toutes les fleches d'evolution sont supprimees.
#
#  UN POINT DE RIGUEUR A CONNAITRE
#  --------------------------------
#  Quand les axes actifs ne couvrent pas toute la cle, les bornes conformes
#  affichees sont la SOMME des bornes individuelles. Cette somme n'a plus la
#  garantie de couverture a 90 % du conforme : la couverture d'une somme
#  d'intervalles n'est pas celle de ses composantes. Le titre le signale des
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

        #  Cles textuelles de df, par axe. Construites une fois : sans elles,
        #  chaque changement de selection relirait df en entier.
        self._df_txt = {c: df[c].astype(str).values for c in self.cles}

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

    # ================================================================
    #  LA SOURCE UNIQUE : tous les panneaux passent par ici
    # ================================================================
    def table_axes(self, sub, sub_ex, axes):
        """Agrege `expl` sur les axes actifs et y joint le score des anomalies.

        C'est la methode centrale de ce fichier. Le forest plot, le tableau de
        priorisation et le point observe de la courbe d'evolution en sortent
        tous les trois, donc ils ne peuvent plus se contredire : un meme
        sous-portefeuille affiche le meme y_obs partout.

        Les montants sont sommes, y compris les bornes conformes. La somme est
        le bon agregat pour des montants, mais elle fait perdre la garantie de
        couverture du conforme, ce que le titre des figures signale.
        """
        axes = [a for a in axes if a in sub_ex.columns] or self.cles[:1]
        if not len(sub_ex):
            return pd.DataFrame(columns=axes + ["y_obs", "y_pred", "lo", "hi",
                                                "score", "n", "libelle",
                                                "couvert"])
        montants = {"y_obs": ("y_obs", "sum"), "y_pred": ("y_pred", "sum"),
                    "lo": ("borne_basse", "sum"), "hi": ("borne_haute", "sum"),
                    "n_lignes": ("y_obs", "size")}
        t = sub_ex.groupby(axes, observed=True).agg(**montants).reset_index()

        if len(sub) and COL_SCORE in sub.columns:
            s = (sub.groupby(axes, observed=True)
                 .agg(score=(COL_SCORE, "sum"), score_max=(COL_SCORE, "max"),
                      n=(COL_SCORE, "size")).reset_index())
            t = t.merge(s, on=axes, how="inner")   # on ne garde que les anomalies
        else:
            t["score"], t["score_max"], t["n"] = 0.0, 0.0, 0

        t["couvert"] = (t["y_obs"] >= t["lo"]) & (t["y_obs"] <= t["hi"])
        t["libelle"] = t[axes].astype(str).agg(" | ".join, axis=1).str.slice(0, 38)
        return t.sort_values("score", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------ unites
    def unites(self, table, n=N_UNITES_LISTE, axes=None):
        """Options du selecteur d'unite, tirees de la MEME table agregee."""
        if not len(table):
            return []
        axes = axes or self.cles[:1]
        options = []
        for _, r in table.head(n).iterrows():
            valeurs = tuple(str(r[a]) for a in axes)
            options.append((f"{' | '.join(valeurs)}   ({_fmt(r['y_obs'])})",
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
        """Historique du groupe defini par (axes, valeurs), somme par trimestre."""
        axes = [a for a in axes if a in self._df_txt] or self.cles[:1]
        d = self.df[self._masque(self._df_txt, axes, valeurs)]
        if not len(d):
            return None, [], 0

        colonnes = [self.target] + [v for v in self.vars_explic if v in d.columns]
        g = d.groupby(["year", "quarter"], observed=True)[colonnes] \
             .sum().reset_index()
        n_lignes = int(len(d) / max(len(g), 1))
        g = g.sort_values(["year", "quarter"]).tail(n)
        per = (g["year"].astype(int).astype(str) + "-T"
               + g["quarter"].astype(int).astype(str)).tolist()
        return g, per, n_lignes

    def contexte(self, table, valeurs, axes):
        """Le contexte conforme du groupe, lu dans la MEME table agregee que
        le forest plot et le tableau. Aucun recalcul separe, donc aucune
        divergence possible entre les trois panneaux."""
        axes = [a for a in axes if a in table.columns] or self.cles[:1]
        if not len(table):
            return None
        m = np.ones(len(table), dtype=bool)
        for a, v in zip(axes, valeurs):
            m &= (table[a].astype(str).values == str(v))
        t = table[m]
        if not len(t):
            return None
        r = t.iloc[0]
        per = None
        if {"year", "quarter"} <= set(self.ex.columns):
            e = self.ex.iloc[0]
            per = f"{int(e['year'])}-T{int(e['quarter'])}"
        return dict(per=per, pred=float(r["y_pred"]), lo=float(r["lo"]),
                    hi=float(r["hi"]), obs=float(r["y_obs"]),
                    couvert=bool(r["couvert"]),
                    n_lignes=int(r.get("n_lignes", 1)))

    # ------------------------------------ variables explicatives, notees
    def variables_explicatives(self, hist, n=N_VARS_EXPLIC):
        """Les n variables dont la RUPTURE au dernier trimestre est la plus
        forte, mesuree contre leur propre passe.

        Critere : (valeur du trimestre valide − mediane des trimestres
        precedents) rapportee a l'ecart inter-quartile de ces memes trimestres.
        Une variable stable qui saute remonte en tete ; une variable
        naturellement volatile ne remonte que si elle sort de son regime.
        """
        vide = pd.DataFrame(columns=["variable", "valeur", "mediane_passe",
                                     "variation", "z"])
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
            lignes.append({"variable": v, "valeur": courant,
                           "mediane_passe": med, "variation": courant - med,
                           "z": z})
        if not lignes:
            return vide
        t = pd.DataFrame(lignes)
        return t.reindex(t["z"].abs().sort_values(ascending=False).index) \
                .head(n).reset_index(drop=True)

    # ------------------------------------------------------- hierarchie
    def hierarchie(self, sub, chemin):
        """Noeuds du cercle, construits sur le PERIMETRE COURANT et sur les
        axes actifs. Le cercle suit donc a la fois les boutons d'axes et le
        filtre maille/valeur.

        `valeur_secteur` vaut le score reel sur les feuilles et zero sur les
        parents. Avec branchvalues="remainder", Plotly additionne lui-meme les
        enfants : plus de contrainte "parent >= somme des enfants", donc plus
        de cercle blanc a partir de trois ou quatre niveaux.
        """
        if not len(sub) or not chemin:
            return pd.DataFrame()
        prof_max = len(chemin)
        agg = {"score_total": (COL_SCORE, "sum"),
               "score_moyen": (COL_SCORE, "mean"),
               "score_max": (COL_SCORE, "max"), "n": (COL_SCORE, "size")}
        total_perimetre = float(sub[COL_SCORE].sum()) or 1.0

        lignes = []
        for prof in range(1, prof_max + 1):
            cols = chemin[:prof]
            g = sub.groupby(cols, observed=True).agg(**agg).reset_index()
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
                    "part": 100 * total / total_perimetre})
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
    #  Couleur portee par le MONTANT lui-meme : la barre la plus longue est
    #  aussi la plus foncee, sans rang normalise intermediaire.
    fig = go.Figure(go.Bar(x=[], y=[], orientation="h", showlegend=False,
                           marker=dict(colorscale=ECHELLE,
                                       line=dict(width=.5, color="white"),
                                       colorbar=dict(title="Montant",
                                                     thickness=14, len=.7,
                                                     tickformat="~s"))))
    fig.update_layout(xaxis=dict(tickformat="~s"),
                      yaxis=dict(tickfont=dict(size=10)))
    _mise_en_forme(fig, 520, dict(l=10, r=40, t=95, b=50))
    return go.FigureWidget(fig)


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
    return go.FigureWidget(fig)


def _creer_evolution(target):
    """Courbe d'evolution. Les traces 1 et 2 portent l'intervalle conforme et
    la prediction : elles sont conservees, comme convenu."""
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
        a.update(x=0, xanchor="left", font=dict(size=12, color=_GRIS))
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
    defaut_maille = next((c for c in ("Lob", "Partner", "Companies", "Risk")
                          if c in cles), cles[0])

    # ------------------------------------------------------------- widgets
    #  Les axes d'agregation remplacent les anciens selecteurs de segment :
    #  ils structurent le cercle ET commandent tous les autres panneaux.
    axes_btns = [widgets.ToggleButton(
        value=(c in cles[:2]), description=c,
        layout=widgets.Layout(width="auto"),
        button_style="info") for c in cles]

    sel_maille = widgets.Dropdown(
        options=[(c, c) for c in cles], value=defaut_maille,
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

    # ------------------------------------------------------------ figures
    fw_cercle = _creer_cercle()
    fw_barres = _creer_barres()
    fw_forest = _creer_forest(target)
    fw_evol = _creer_evolution(target)
    fw_vars = _creer_variables()
    z_cartes, z_table = widgets.Output(), widgets.Output()
    verrou = {"actif": False}
    etat = {"table": pd.DataFrame()}      # la table agregee courante, partagee

    def _axes_actifs():
        actifs = [c for c, b in zip(cles, axes_btns) if b.value]
        return actifs or [cles[0]]        # jamais zero axe

    # --------------------------------------------------------- le cercle
    def _maj_cercle(sub, titre):
        axes = _axes_actifs()
        try:
            h = socle.hierarchie(sub, axes)
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
                f"({r['part']:.1f} % du perimetre)<br>"
                f"Pire anomalie : {_fmt4(r['score_max'])}"
                for _, r in h.iterrows()]
            with fw_cercle.batch_update():
                t = fw_cercle.data[0]
                t.ids, t.labels = h["id"].tolist(), h["label"].tolist()
                t.parents = h["parent"].tolist()
                t.values = h["valeur_secteur"].tolist()
                t.branchvalues = "remainder"
                t.text = [f"{p:.0f} %" for p in h["part"]]
                t.texttemplate = "%{label}<br>%{text}"
                t.hovertext, t.hoverinfo = survol, "text"
                t.insidetextorientation = "radial"
                t.maxdepth = len(axes)
                t.marker = dict(
                    colors=h["score_moyen"].tolist(), colorscale=ECHELLE,
                    cmin=0, cmax=cmax, line=dict(color="white", width=1.6),
                    colorbar=dict(title="Gravite<br>moyenne", thickness=16,
                                  len=.7, tickformat="~s"))
                fw_cercle.layout.height = 620
                fw_cercle.layout.title = dict(
                    text=f"Repartition des anomalies  ·  {' › '.join(axes)}"
                         f"<br><sup>{titre}</sup>",
                    font=dict(size=15), x=.015, xanchor="left")
        except Exception as e:
            #  Une exception avalee par ipywidgets laisserait un cercle blanc
            #  sans explication. On garde le dessin precedent et on le dit.
            print(f"Cercle non mis a jour : {type(e).__name__} : {str(e)[:120]}")

    # --------------------------------------------------------- les cartes
    def _cartes(sub, titre, table):
        part = (sub[COL_SCORE].sum() / max(socle.score_global, 1e-12)
                if COL_SCORE in sub.columns and len(sub) else 0)
        grav = _fmt4(sub[COL_SCORE].mean()) if len(sub) else "—"
        pire = "—"
        if len(sub) and "rank" in sub.columns:
            rk = sub.loc[sub[COL_SCORE].idxmax(), "rank"]
            if pd.notna(rk):
                pire = f"#{int(rk)}"
        couv = (f"{100 * table['couvert'].mean():.1f} %" if len(table) else "n/a")
        cartes = [("Anomalies", f"{len(sub):,}".replace(",", " "), "#37474f"),
                  ("Part du score global", f"{100 * part:.1f} %", "#ad1457"),
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
    def _maj_barres(table, titre):
        axes = _axes_actifs()
        if not len(table):
            _vider(fw_barres, f"{titre}<br><sup>Aucune anomalie</sup>", 260)
            return
        g = table.head(top_n).iloc[::-1]        # plus grave en haut
        survol = [
            f"<b>{r['libelle']}</b><br>{target} observe : {_fmt(r['y_obs'])}"
            f"<br>Predit : {_fmt(r['y_pred'])}"
            f"<br>Intervalle : [{_fmt(r['lo'])} ; {_fmt(r['hi'])}]"
            f"<br>Score cumule : {_fmt4(r['score'])}"
            f"<br>Anomalies regroupees : {int(r['n'])}"
            for _, r in g.iterrows()]
        montants = g["y_obs"].tolist()
        with fw_barres.batch_update():
            t = fw_barres.data[0]
            t.x, t.y = montants, g["libelle"].tolist()
            t.marker.color = montants
            t.marker.cmin, t.marker.cmax = min(montants), max(montants)
            t.text, t.hovertemplate = survol, "%{text}<extra></extra>"
            fw_barres.layout.xaxis.title.text = f"<b>{target}</b>"
            fw_barres.layout.height = max(380, 38 * len(g) + 150)
            fw_barres.layout.title = dict(
                text=f"Les {len(g)} plus critiques  ·  agrege sur "
                     f"{' + '.join(axes)}<br><sup>{titre}</sup>",
                font=dict(size=14), x=.015, xanchor="left")

    # ---------------------------------------------------------- le forest
    def _maj_forest(table, titre):
        axes = _axes_actifs()
        if not len(table):
            _vider(fw_forest, f"{titre}<br><sup>Aucune anomalie</sup>", 260)
            return
        d = table.head(top_n).iloc[::-1].reset_index(drop=True)
        y = list(range(len(d)))
        lo = d["lo"].to_numpy(dtype="float64")
        hi = d["hi"].to_numpy(dtype="float64")
        obs = d["y_obs"].to_numpy(dtype="float64")
        pred = d["y_pred"].to_numpy(dtype="float64")

        xs_band, ys_band, xs_over, ys_over = [], [], [], []
        for yi, l, h, o in zip(y, lo, hi, obs):
            xs_band += [l, h, None]
            ys_band += [yi, yi, None]
            cible = h if o > h else l
            xs_over += [cible, o, None]
            ys_over += [yi, yi, None]

        libelles = d["libelle"].str.slice(0, 34).tolist()
        ticks = [f"#{i + 1}  {lab}"
                 for i, lab in zip(range(len(d) - 1, -1, -1), libelles)]
        textes = [
            f"<b>{lab}</b><br>{target} observe : {_fmt(o)}<br>Predit : {_fmt(p)}"
            f"<br>Intervalle : [{_fmt(l)} ; {_fmt(h)}]"
            for lab, o, p, l, h in zip(libelles, obs, pred, lo, hi)]

        with fw_forest.batch_update():
            fw_forest.data[0].x, fw_forest.data[0].y = xs_band, ys_band
            fw_forest.data[1].x, fw_forest.data[1].y = xs_over, ys_over
            fw_forest.data[2].x, fw_forest.data[2].y = pred, y
            fw_forest.data[2].text = textes
            fw_forest.data[2].hovertemplate = "%{text}<extra></extra>"
            fw_forest.data[3].x, fw_forest.data[3].y = obs, y
            fw_forest.data[3].text = textes
            fw_forest.data[3].hovertemplate = "%{text}<extra></extra>"
            fw_forest.layout.xaxis.title.text = f"<b>{target}</b>"
            fw_forest.layout.yaxis = dict(tickmode="array", tickvals=y,
                                          ticktext=ticks, tickfont=dict(size=10))
            fw_forest.layout.height = max(420, 44 * len(d) + 175)
            fw_forest.layout.title = dict(
                text="Intervalle conforme, prediction et valeur observee"
                     f"<br><sup>{titre}  ·  agrege sur {' + '.join(axes)}, "
                     "memes montants que la courbe d'evolution</sup>",
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
            ctx = socle.contexte(etat["table"], valeurs, axes)
            _maj_evolution(hist, per, ctx, valeurs, axes)
            _maj_variables(hist, per, ctx)
        except Exception as e:
            msg = f"Mise a jour impossible : {type(e).__name__} : {str(e)[:110]}"
            _vider(fw_evol, msg)
            _vider(fw_vars, msg)

    def _maj_evolution(hist, per, ctx, valeurs, axes):
        if hist is None or not len(hist):
            _vider(fw_evol, "Aucun historique pour ce sous-portefeuille.")
            return
        val = hist[target].to_numpy(dtype="float64")
        #  Le trimestre valide est le dernier de l'historique quand `expl` ne
        #  porte pas d'annee/trimestre exploitable.
        p_valide = ctx["per"] if ctx and ctx.get("per") in per else per[-1]
        xb = yb = xp = yp = xh = yh = []
        if ctx:
            xb, yb = [p_valide, p_valide], [ctx["lo"], ctx["hi"]]
            xp, yp = [p_valide], [ctx["pred"]]
            xh, yh = [p_valide], [ctx["obs"]]
        couvert = bool(ctx["couvert"]) if ctx else True
        coul = _OK if couvert else _ACCENT
        halo = "rgba(61,90,158,0.18)" if couvert else "rgba(192,57,43,0.20)"
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
                     f"axes {' + '.join(axes)}"
                     + (f" · <b style='color:{_ACCENT}'>somme de "
                        f"{ctx['n_lignes'] if ctx else 1} lignes, bornes "
                        "conformes additionnees</b>" if agrege else "")
                     + f" · periode validee <b>{p_valide}</b></span>",
                font=dict(size=14), x=.015, xanchor="left")

    def _maj_variables(hist, per, ctx):
        if hist is None or len(hist) < 3:
            _vider(fw_vars, "Historique trop court pour classer les variables.")
            return
        t = socle.variables_explicatives(hist)
        if not len(t):
            _vider(fw_vars, "Aucune variable explicative numerique exploitable "
                            "dans l'historique.")
            return
        p_valide = ctx["per"] if ctx and ctx.get("per") in per else per[-1]
        k = per.index(p_valide)
        with fw_vars.batch_update():
            for i in range(N_VARS_EXPLIC):
                t_l, t_p = fw_vars.data[2 * i], fw_vars.data[2 * i + 1]
                ann = fw_vars.layout.annotations[i]
                if i < len(t):
                    v = t.iloc[i]["variable"]
                    vals = hist[v].to_numpy(dtype="float64")
                    t_l.x, t_l.y = per, vals
                    t_l.hovertemplate = (f"<b>%{{x}}</b><br>{v} : "
                                         "<b>%{y:,.0f}</b><extra></extra>")
                    t_p.x, t_p.y = [per[k]], [vals[k]]
                    ann.text = f"<b>{v}</b>"      # le nom seul, rien d'autre
                    ann.font.color = _DOUX[i % len(_DOUX)]
                else:
                    t_l.x, t_l.y = [], []
                    t_p.x, t_p.y = [], []
                    ann.text = " "
            fw_vars.layout.height = 150 * max(len(t), 1) + 120
            fw_vars.layout.title = dict(
                text="<b>Les variables qui expliquent ce comportement</b>",
                font=dict(size=14), x=.015, xanchor="left")

    # ----------------------------------------------------------- le tableau
    def _maj_tableau(table, titre):
        with z_table:
            #  clear_output(wait=True) n'efface qu'a l'arrivee du contenu
            #  suivant : sans le try, une erreur laisserait l'ANCIEN tableau.
            clear_output(wait=True)
            try:
                if not len(table):
                    print(f"Aucune anomalie dans le perimetre : {titre}")
                    return
                d = table.head(N_LIGNES_TABLE)
                #  Memes colonnes, memes montants que le forest plot et que le
                #  point observe de la courbe : une seule source pour tous.
                t = pd.DataFrame({
                    "Rang": range(1, len(d) + 1),
                    "Maille": d["libelle"].values,
                    "Y_obs": d["y_obs"].values, "Y_pred": d["y_pred"].values,
                    "CP_bas": d["lo"].values, "CP_haut": d["hi"].values,
                    "Couvert": d["couvert"].values, "Score": d["score"].values})
                fmt_col = {c: (lambda v: _fmt(v))
                           for c in ("Y_obs", "Y_pred", "CP_bas", "CP_haut")}
                fmt_col["Score"] = lambda v: _fmt4(v)
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
        """Point d'entree unique : tout passe par ici, donc tout reste
        coherent. Les axes comme la maille et la valeur y aboutissent."""
        if verrou["actif"]:
            return
        axes = _axes_actifs()
        sub, sub_ex, titre = socle.filtrer(sel_maille.value, sel_valeur.value)

        #  La table agregee est calculee UNE fois et partagee par le forest
        #  plot, les barres, le tableau et le contexte de la courbe.
        table = socle.table_axes(sub, sub_ex, axes)
        etat["table"] = table

        _maj_cercle(sub, titre)
        with z_cartes:
            clear_output(wait=True)
            display(_cartes(sub, titre, table))
        _maj_barres(table, titre)
        _maj_forest(table, titre)

        options = socle.unites(table, axes=axes)
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
        _maj_tableau(table, titre)

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
        axes = _axes_actifs()
        if not parts or len(parts) > len(axes):
            return
        colonne, valeur = axes[len(parts) - 1], parts[-1]
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
    for b in axes_btns:
        b.observe(lambda c: _maj_panneaux() if c["name"] == "value" else None,
                  names="value")
    sel_maille.observe(lambda c: _maj_valeurs() if c["name"] == "value" else None,
                       names="value")
    sel_valeur.observe(lambda c: _maj_panneaux() if c["name"] == "value" else None,
                       names="value")
    sel_unite.observe(
        lambda c: (_maj_unite() if not verrou["actif"] else None)
        if c["name"] == "value" else None, names="value")
    try:
        fw_cercle.data[0].on_click(_au_clic_cercle)
        clic = True
    except Exception:
        clic = False

    # ---------------------------------------------------------- affichage
    display(_bandeau(
        "<b>Axes d'agregation</b> — ils structurent le cercle et commandent "
        "tous les panneaux. Desactivez une dimension pour regrouper les "
        "anomalies qui n'en differaient que par elle."
        + ("  Le clic sur une part du cercle filtre le perimetre."
           if clic else ""),
        fond="#e3f2fd", coul="#0d47a1"))
    display(widgets.HBox(
        axes_btns, layout=widgets.Layout(flex_flow="row wrap",
                                         align_items="center")))
    display(fw_cercle)

    display(_bandeau(
        "<b>Perimetre</b> — la maille et la valeur filtrent l'ensemble du "
        "tableau de bord, cercle compris.", fond="#e8f5e9", coul="#1b5e20"))
    display(widgets.HBox([sel_maille, sel_valeur]))
    display(z_cartes)
    display(fw_barres)
    display(fw_forest)

    display(_bandeau(
        "<b>Le sous-portefeuille en detail</b> — evolution de la cible sur "
        f"{N_TRIMESTRES} trimestres avec l'intervalle conforme et la "
        "prediction, puis les variables qui expliquent son comportement.",
        fond="#fff3e0", coul="#e65100"))
    display(sel_unite)
    display(fw_evol)
    display(fw_vars)

    display(_bandeau("<b>Tableau de priorisation</b> — perimetre courant, "
                     f"{N_LIGNES_TABLE} lignes les plus graves."))
    display(z_table)

    _maj_valeurs()
    return {"cercle": fw_cercle, "barres": fw_barres, "forest": fw_forest,
            "evolution": fw_evol, "variables": fw_vars, "maille": sel_maille,
            "valeur": sel_valeur, "unite": sel_unite, "axes": axes_btns,
            "cartes": z_cartes, "tableau": z_table, "socle": socle,
            "etat": etat, "rafraichir": _maj_panneaux}


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
