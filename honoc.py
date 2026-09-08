# -*- coding: utf-8 -*-
# =============================================================================
#  TABLEAU DE BORD UNIFIE - anomalies, evolution, SHAP, priorisation
#
#  UN SEUL BLOC, AUTONOME. Rien a executer avant, hormis d'avoir en session :
#      anomalies_prio, expl, df, ID_COLS, TARGET      (ALPHA facultatif)
#  Aucune dependance a dashboard_complet(), tableau_priorisation() ou aux
#  helpers _creer_fw_* de vos cellules precedentes : tout est redefini ici.
#
#  CE QUE VOUS AVIEZ DEJA, CONSERVE
#  --------------------------------
#    - cercle hierarchique (sunburst) pilote par 4 selecteurs de segment,
#      cliquable, qui synchronise le filtre
#    - barres des anomalies les plus critiques
#    - forest plot intervalle / prediction / valeur observee
#    - cartes de synthese
#    - selecteurs maille + valeur
#
#  CE QUI EST AJOUTE
#  -----------------
#    1. AXES D'AGREGATION (ruban de boutons, un par dimension). C'est la
#       demande "pouvoir regarder juste 2 dimensions au lieu de 3" : les barres
#       s'agregent sur les seules dimensions actives, et les libelles suivent.
#    2. EVOLUTION DE LA CIBLE sur les 10 derniers trimestres (mission 1)
#    3. DECOMPOSITION SHAP en cascade (mission 2)
#    4. EVOLUTION DES 5 VARIABLES DETERMINANTES (mission 2)
#    5. TABLEAU DE PRIORISATION en bas, filtre sur le perimetre courant
#  Tout se met a jour ensemble, sur les memes selecteurs.
#
#  DEUX CORRECTIONS PAR RAPPORT A VOTRE VERSION
#  ---------------------------------------------
#    a) Le "None" qui s'affiche sous vos selecteurs vient d'un display()
#       imbrique dans _bandeau() : _bandeau recoit None et l'imprime. Corrige.
#    b) Le forest plot passait l'axe en echelle logarithmique
#       (LOG_FOREST = True). La consigne du projet interdit tout logarithme,
#       axes compris. Remplace par une VUE NORMALISEE z = (obs - centre) /
#       demi-largeur, qui est une division et non une transformation non
#       lineaire : tous les intervalles deviennent [-1, +1] et la position de
#       l'observation se lit directement en nombre de demi-largeurs. C'est la
#       meme solution que celle deja retenue dans graphiques_conformal.py.
#       L'echelle en montants reste disponible dans le ruban.
# =============================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

# ┌──────────────────────────── PARAMETRES ────────────────────────────┐
TOP_N_PANNEAUX = 12       # barres et forest plot
N_TRIMESTRES   = 10       # enonce mission 1
N_VARS_SHAP    = 5        # enonce mission 2
N_UNITES_LISTE = 30       # unites proposees dans le selecteur
N_LIGNES_TABLE = 15       # lignes du tableau du bas
COL_SCORE      = "score_composite"
COL_GWP        = "GWP"
ECHELLE        = "Bluered"
CHEMIN_MODELE  = "artefacts_modele/modele_final.joblib"
VUE_GENERALE   = ""
# └─────────────────────────────────────────────────────────────────────┘

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
        height=hauteur, separators=", ",
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

        #  Index de cles, construits une fois. Sans eux chaque changement de
        #  selection relirait df en entier pour chaque sous-portefeuille.
        self._cle_df = df[self.cles].astype(str).agg("\x1f".join, axis=1).values
        self._cle_ex = self.ex[self.cles].agg("\x1f".join, axis=1).values

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
        """Agrege les anomalies sur les seules dimensions actives.

        C'est ici que se joue la demande "regarder 2 dimensions au lieu de 3" :
        avec moins d'axes, les anomalies d'un meme partenaire se cumulent au
        lieu d'apparaitre eclatees, et le classement change de sens.
        """
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
    def unites(self, sub, n=N_UNITES_LISTE):
        if not len(sub):
            return []
        d = (sub.nlargest(min(n, len(sub)), COL_SCORE)
             if COL_SCORE in sub.columns else sub.head(n))
        options = []
        for _, r in d.iterrows():
            cle = tuple(str(r[c]) for c in self.cles)
            rang = (f"#{int(r['rank'])} " if "rank" in d.columns
                    and pd.notna(r.get("rank")) else "")
            sc = f"   ({_fmt4(r[COL_SCORE])})" if COL_SCORE in d.columns else ""
            options.append((f"{rang}{' | '.join(cle)}{sc}", cle))
        return options

    # -------------------------------------------------------- historique
    def historique(self, cle, n=N_TRIMESTRES):
        h = self.df[self._cle_df == "\x1f".join(cle)]
        if not len(h):
            return None, []
        h = (h.sort_values("time_idx") if "time_idx" in h.columns
             else h.sort_values(["year", "quarter"])).tail(n)
        per = (h["year"].astype(int).astype(str) + "-T"
               + h["quarter"].astype(int).astype(str)).tolist()
        return h, per

    def contexte(self, cle):
        t = self.ex[self._cle_ex == "\x1f".join(cle)]
        if not len(t):
            return None
        r = t.iloc[0]
        return dict(per=f"{int(r['year'])}-T{int(r['quarter'])}",
                    pred=float(r["y_pred"]), lo=float(r["borne_basse"]),
                    hi=float(r["borne_haute"]), obs=float(r["y_obs"]),
                    couvert=bool(r["dans_intervalle"]))

    # ------------------------------------------------------- hierarchie
    def hierarchie(self, chemin):
        lignes = []
        for prof in range(1, len(chemin) + 1):
            cols = chemin[:prof]
            agg = {"score_total": (COL_SCORE, "sum"),
                   "score_moyen": (COL_SCORE, "mean"),
                   "score_max": (COL_SCORE, "max"), "n": (COL_SCORE, "size")}
            if COL_GWP in self.dd.columns:
                agg["gwp"] = (COL_GWP, "sum")
            g = self.dd.groupby(cols, observed=True).agg(**agg).reset_index()
            for _, r in g.iterrows():
                vals = [str(r[c]) for c in cols]
                lignes.append({
                    "id": "/".join(vals), "label": vals[-1],
                    "parent": "/".join(vals[:-1]) if prof > 1 else "",
                    "profondeur": prof, "score_total": r["score_total"],
                    "score_moyen": r["score_moyen"], "score_max": r["score_max"],
                    "n": int(r["n"]), "gwp": r.get("gwp", np.nan)})
        return pd.DataFrame(lignes)


# =============================================================================
#  2. SHAP LOCAL, avec cache
# =============================================================================
class _Shap:
    def __init__(self, modele=None, x_model=None, chemin=CHEMIN_MODELE):
        self.cache, self.explainer, self.raison = {}, None, None
        self.x_model, self.feats, self.mdl = x_model, [], None
        try:
            import shap                                      # noqa: F401
        except ImportError:
            self.raison = "Paquet `shap` absent (pip install shap)."
            return
        try:
            if modele is None:
                import joblib
                art = joblib.load(chemin)
                modele = (next(v for v in art.values() if hasattr(v, "predict"))
                          if isinstance(art, dict) else art)
            mdl = (modele.named_steps["model"]
                   if hasattr(modele, "named_steps") else modele)
            self.feats = (list(mdl.feature_name_) if hasattr(mdl, "feature_name_")
                          else list(mdl.feature_names_in_))
            self.mdl = mdl
            self.explainer = shap.TreeExplainer(mdl)
        except Exception as e:
            self.raison = f"Modele indisponible pour SHAP : {str(e)[:90]}"

    @property
    def actif(self):
        return self.explainer is not None

    def decomposer(self, cle, ligne):
        if not self.actif or ligne is None or not len(ligne):
            return None
        if cle in self.cache:
            return self.cache[cle]
        try:
            if self.x_model is not None:
                x = self.x_model.loc[ligne.index, self.feats]
            else:
                absentes = [f for f in self.feats if f not in ligne.columns]
                if absentes:
                    raise KeyError(f"{len(absentes)} variable(s) du modele "
                                   f"absente(s) de df (ex. {absentes[:2]})")
                x = ligne[self.feats].copy()
                for c in x.columns:
                    if x[c].dtype == object:
                        x[c] = x[c].astype("category")
            sv = np.asarray(self.explainer.shap_values(x)).ravel()
            base = float(np.ravel(self.explainer.expected_value)[0])
            contrib = pd.Series(sv, index=self.feats)
            pred = float(self.mdl.predict(x)[0])
            res = dict(base=base, contrib=contrib, pred=pred,
                       ecart=abs(base + contrib.sum() - pred))
        except Exception as e:
            res = dict(erreur=f"{type(e).__name__} : {str(e)[:100]}")
        self.cache[cle] = res
        return res


# =============================================================================
#  3. CREATION DES FIGURES (structure figee, seules les donnees changent)
# =============================================================================
def _creer_cercle():
    fig = go.Figure(go.Sunburst(ids=[], labels=[], parents=[], values=[],
                                branchvalues="total"))
    _mise_en_forme(fig, 600, dict(l=10, r=10, t=100, b=15))
    return go.FigureWidget(fig)


def _creer_barres():
    fig = go.Figure(go.Bar(x=[], y=[], orientation="h", showlegend=False,
                           marker=dict(colorscale=ECHELLE, cmin=0, cmax=1,
                                       line=dict(width=.5, color="white"))))
    fig.update_layout(xaxis_title="Score cumule", yaxis=dict(tickfont=dict(size=9)))
    _mise_en_forme(fig, 520, dict(l=10, r=30, t=95, b=45))
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
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.03,
                                  xanchor="center", x=.5))
    _mise_en_forme(fig, 520, dict(l=10, r=40, t=110, b=50))
    return go.FigureWidget(fig)


def _creer_evolution(target):
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
    fig.update_yaxes(gridcolor=_GRILLE, zeroline=False, tickformat=",.0f",
                     title_text=f"<b>{target}</b>")
    _mise_en_forme(fig, 400, dict(l=85, r=45, t=95, b=45))
    fig.update_layout(hovermode="x unified",
                      legend=dict(orientation="h", y=1.06, x=1,
                                  xanchor="right", font=dict(size=11)))
    return go.FigureWidget(fig)


def _creer_shap():
    fig = go.Figure(go.Waterfall(
        orientation="v", x=[], y=[], measure=[], text=[],
        textposition="outside", textfont=dict(size=10),
        connector=dict(line=dict(color=_GRILLE, width=1)),
        increasing=dict(marker=dict(color=_ACCENT)),
        decreasing=dict(marker=dict(color=_OK)),
        totals=dict(marker=dict(color=_ENCRE))))
    fig.update_xaxes(showgrid=False, linecolor=_GRILLE, tickangle=-30,
                     tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=_GRILLE, zeroline=False, tickformat=",.0f")
    _mise_en_forme(fig, 430, dict(l=70, r=30, t=95, b=110))
    return go.FigureWidget(fig)


def _creer_variables(n=N_VARS_SHAP):
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        vertical_spacing=.045, subplot_titles=[" "] * n)
    for i in range(n):
        c = _DOUX[i % len(_DOUX)]
        fig.add_scatter(x=[], y=[], mode="lines+markers", showlegend=False,
                        line=dict(color=c, width=2.3, shape="spline",
                                  smoothing=.55),
                        marker=dict(size=6, color="white",
                                    line=dict(color=c, width=1.8)),
                        row=i + 1, col=1)
        fig.add_scatter(x=[], y=[], mode="markers", showlegend=False,
                        hoverinfo="skip",
                        marker=dict(size=12, color=c,
                                    line=dict(color="white", width=2.2)),
                        row=i + 1, col=1)
    fig.update_xaxes(showgrid=False, showticklabels=False, linecolor=_GRILLE)
    fig.update_xaxes(showticklabels=True, title_text="<b>Trimestre</b>",
                     row=n, col=1)
    fig.update_yaxes(gridcolor=_GRILLE, zeroline=False, tickfont=dict(size=9))
    _mise_en_forme(fig, 150 * n + 120, dict(l=70, r=30, t=95, b=45))
    fig.update_layout(hovermode="x unified")
    for a in fig.layout.annotations:
        a.update(x=0, xanchor="left", font=dict(size=11, color=_GRIS))
    return go.FigureWidget(fig)


def _vider(fw, message, hauteur=260):
    """Vide vraiment : x, y, text, mesures et annotations."""
    with fw.batch_update():
        for t in fw.data:
            t.x, t.y = [], []
            if "text" in t:
                t.text = []
            if t.type == "waterfall":
                t.measure = []
            if t.type == "sunburst":
                t.ids, t.labels, t.parents, t.values = [], [], [], []
        for a in fw.layout.annotations:
            a.text = " "
        fw.layout.title = dict(text=message, font=dict(size=14), x=.015,
                               xanchor="left")
        fw.layout.height = hauteur


# =============================================================================
#  4. LE TABLEAU DE BORD
# =============================================================================
def tableau_de_bord_unifie(anomalies_prio, expl, df, id_cols=None, target=None,
                           alpha=None, modele=None, x_model=None,
                           top_n=TOP_N_PANNEAUX):
    id_cols = id_cols if id_cols is not None else _session("ID_COLS")[1]
    target = target if target is not None else _session("TARGET")[1]
    if alpha is None:
        ok, a = _session("ALPHA")
        alpha = a if ok else .10
    if modele is None:
        modele = _session("MODELE_TE")[1]
    if id_cols is None or target is None:
        raise NameError("ID_COLS et TARGET doivent exister dans la session.")

    socle = _Socle(anomalies_prio, expl, df, id_cols, target, alpha)
    shap_moteur = _Shap(modele=modele, x_model=x_model)
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
        description="1 · Maille :", layout=widgets.Layout(width="330px"),
        style={"description_width": "90px"})
    sel_valeur = widgets.Dropdown(
        options=[("— vue generale —", VUE_GENERALE)], value=VUE_GENERALE,
        description="2 · Valeur :", layout=widgets.Layout(width="470px"),
        style={"description_width": "90px"})
    sel_unite = widgets.Dropdown(
        options=[], description="3 · Unite :",
        layout=widgets.Layout(width="640px"),
        style={"description_width": "90px"})

    #  RUBAN DES AXES D'AGREGATION : c'est la nouveaute demandee. Un bouton par
    #  dimension ; desactiver une dimension fait fusionner les anomalies qui
    #  n'en differaient que par elle.
    axes_btns = [widgets.ToggleButton(
        value=True, description=c, layout=widgets.Layout(width="auto"),
        button_style="info") for c in cles]

    sel_echelle = widgets.ToggleButtons(
        options=[("Ecart normalise", "z"), ("Montants", "lin")], value="z",
        style={"button_width": "auto"},
        layout=widgets.Layout(width="auto"))

    # ------------------------------------------------------------ figures
    fw_cercle = _creer_cercle()
    fw_barres = _creer_barres()
    fw_forest = _creer_forest()
    fw_evol = _creer_evolution(target)
    fw_shap = _creer_shap()
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
            #  On revient a la couche par defaut au lieu de vider le cercle.
            #  Un cercle blanc laisse croire a un plantage.
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
                t.parents, t.values = h["parent"].tolist(), h["score_total"].tolist()
                t.text = [f"{100 * v / max(socle.score_global, 1e-12):.0f} %"
                          for v in h["score_total"]]
                t.texttemplate = "%{label}<br>%{text}"
                t.hovertext, t.hoverinfo = survol, "text"
                t.insidetextorientation = "radial"
                t.maxdepth = len(chemin)
                t.marker = dict(colors=h["score_moyen"].tolist(),
                                colorscale=ECHELLE, cmin=0, cmax=cmax,
                                line=dict(color="white", width=1.6),
                                colorbar=dict(title="Gravite<br>moyenne",
                                              thickness=16, len=.7,
                                              tickformat=".2g"))
                fw_cercle.layout.height = 600
                fw_cercle.layout.title = dict(
                    text=f"Repartition des anomalies  ·  {' › '.join(chemin)}",
                    font=dict(size=15), x=.015, xanchor="left")
        except Exception as e:
            #  Une exception avalee par ipywidgets laisserait un cercle blanc
            #  sans explication. On garde le dessin precedent et on le dit.
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
        with fw_barres.batch_update():
            t = fw_barres.data[0]
            t.x, t.y = g["score_total"].tolist(), g["libelle"].tolist()
            t.marker.color = g["score_total"].rank(pct=True).tolist()
            t.text, t.hovertemplate = survol, "%{text}<extra></extra>"
            fw_barres.layout.height = max(360, 34 * len(g) + 150)
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
            #  Vue normalisee : division par la demi-largeur, pas de logarithme.
            #  Tous les intervalles se ramenent a [-1, +1], donc des montants de
            #  0,7 EUR et de 317 MEUR deviennent comparables sur un meme axe.
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
            fw_forest.layout.yaxis = dict(tickmode="array", tickvals=y,
                                          ticktext=ticks, tickfont=dict(size=9))
            fw_forest.layout.height = max(400, 40 * len(d) + 170)
            fw_forest.layout.title = dict(
                text=f"Intervalle conforme, prediction et valeur observee"
                     f"<br><sup>{titre}</sup>",
                font=dict(size=14), x=.015, xanchor="left")

    # ------------------------------------------------- evolution et SHAP
    def _maj_unite(*_):
        cle = sel_unite.value
        if cle is None:
            for fw in (fw_evol, fw_shap, fw_vars):
                _vider(fw, "Aucun sous-portefeuille dans ce perimetre.")
            return
        try:
            h, per = socle.historique(cle)
            ctx = socle.contexte(cle)
            _maj_evolution(h, per, ctx, cle)
            if not shap_moteur.actif:
                raison = shap_moteur.raison or "SHAP indisponible."
                _vider(fw_shap, raison)
                _vider(fw_vars, raison)
                return
            dec = shap_moteur.decomposer(cle, h.iloc[[-1]] if h is not None
                                         and len(h) else None)
            _maj_shap(dec)
            _maj_variables(dec, h, per, ctx)
        except Exception as e:
            msg = f"Mise a jour impossible : {type(e).__name__} : {str(e)[:110]}"
            for fw in (fw_evol, fw_shap, fw_vars):
                _vider(fw, msg)

    def _maj_evolution(h, per, ctx, cle):
        if h is None or not len(h):
            _vider(fw_evol, "Aucun historique pour ce sous-portefeuille.")
            return
        val = h[target].values.astype(float)
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
            fw_evol.layout.height = 400
            fw_evol.layout.title = dict(
                text=f"<b style='color:{_ENCRE}'>{' | '.join(cle)}</b>"
                     f"<br><span style='font-size:11px'>{target} · {len(h)} "
                     f"trimestres · {per[0]} → {per[-1]} · "
                     f"<span style='color:{_ACCENT if delta < 0 else _OK}'>"
                     f"{fleche} {abs(delta):.1f} %</span>"
                     + (f" · periode validee <b>{ctx['per']}</b>" if ctx else "")
                     + "</span>", font=dict(size=14), x=.015, xanchor="left")

    def _maj_shap(dec):
        if dec is None:
            _vider(fw_shap, "Decomposition SHAP indisponible.")
            return
        if "erreur" in dec:
            _vider(fw_shap, f"SHAP : {dec['erreur']}")
            return
        contrib, base, pred = dec["contrib"], dec["base"], dec["pred"]
        top = contrib.reindex(
            contrib.abs().sort_values(ascending=False).index).head(N_VARS_SHAP)
        reste = contrib.sum() - top.sum()
        #  "Autres variables" ferme la cascade sur la prediction. Sans ce
        #  terme, le graphique n'atterrirait pas sur la bonne valeur.
        x = ["Base"] + list(top.index) + ["Autres variables", "Prediction"]
        y = [base] + list(top.values) + [reste, 0]
        mesure = ["absolute"] + ["relative"] * (len(top) + 1) + ["total"]
        txt = ([_fmt(base)]
               + [("+" if v >= 0 else "−") + _fmt(abs(v)) for v in top.values]
               + [("+" if reste >= 0 else "−") + _fmt(abs(reste)), _fmt(pred)])
        exact = dec["ecart"] < 1e-6 * max(abs(pred), 1.0)
        with fw_shap.batch_update():
            t = fw_shap.data[0]
            t.x, t.y, t.measure, t.text = x, y, mesure, txt
            t.hovertemplate = "<b>%{x}</b><br>%{text}<extra></extra>"
            fw_shap.layout.height = 430
            fw_shap.layout.title = dict(
                text="<b>Decomposition SHAP locale</b>"
                     f"<br><span style='font-size:11px'>base {_fmt(base)} + "
                     f"contributions = prediction {_fmt(pred)} · "
                     + (f"<span style='color:{_OK}'>reconstitution exacte</span>"
                        if exact else
                        f"<span style='color:{_ACCENT}'>ecart "
                        f"{dec['ecart']:.3g}, A VERIFIER</span>")
                     + "</span>", font=dict(size=14), x=.015, xanchor="left")

    def _maj_variables(dec, h, per, ctx):
        if dec is None or "erreur" in dec or h is None or not len(h):
            _vider(fw_vars, "Evolution des variables indisponible.")
            return
        contrib = dec["contrib"]
        top = contrib.reindex(
            contrib.abs().sort_values(ascending=False).index).head(N_VARS_SHAP)
        #  On ne remplace pas une variable non tracable par la 6e : l'enonce
        #  demande les 5 plus determinantes, pas les 5 que l'on sait dessiner.
        tracables = [v for v in top.index
                     if v in h.columns and pd.api.types.is_numeric_dtype(h[v])]
        ecartees = [v for v in top.index if v not in tracables]
        k = per.index(ctx["per"]) if ctx and ctx["per"] in per else len(per) - 1
        total = contrib.abs().sum() or 1.0
        with fw_vars.batch_update():
            for i in range(N_VARS_SHAP):
                t_l, t_p = fw_vars.data[2 * i], fw_vars.data[2 * i + 1]
                ann = fw_vars.layout.annotations[i]
                if i < len(tracables):
                    v = tracables[i]
                    vv = h[v].values.astype(float)
                    t_l.x, t_l.y = per, vv
                    t_l.hovertemplate = (f"<b>%{{x}}</b><br>{v} : "
                                         "<b>%{y:,.4g}</b><extra></extra>")
                    t_p.x, t_p.y = [per[k]], [vv[k]]
                    signe = "+" if contrib[v] >= 0 else "−"
                    ann.text = (f"<b>{v}</b>   SHAP {signe}{_fmt(abs(contrib[v]))}"
                                f"   ({100 * abs(contrib[v]) / total:.0f} %)")
                    ann.font.color = _DOUX[i % len(_DOUX)]
                else:
                    t_l.x, t_l.y = [], []
                    t_p.x, t_p.y = [], []
                    ann.text = " "
            fw_vars.layout.height = 150 * max(len(tracables), 1) + 120
            fw_vars.layout.title = dict(
                text=f"<b>{len(tracables)} variables les plus determinantes</b>"
                     f"<br><span style='font-size:11px'>memes trimestres · "
                     f"periode validee <b>{per[k]}</b>"
                     + (f" · ecartees, non numeriques : {', '.join(ecartees)}"
                        if ecartees else "") + "</span>",
                font=dict(size=14), x=.015, xanchor="left")

    # ----------------------------------------------------------- le tableau
    def _maj_tableau(sub, titre):
        with z_table:
            #  clear_output(wait=True) n'efface qu'a l'arrivee du contenu
            #  suivant : sans le try, une erreur laisserait l'ANCIEN tableau.
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
        sub, sub_ex, titre = socle.filtrer(sel_maille.value, sel_valeur.value)
        with z_cartes:
            clear_output(wait=True)
            display(_cartes(sub, titre, sub_ex))
        _maj_barres(sub, titre)
        _maj_forest(sub, titre)

        options = socle.unites(sub)
        verrou["actif"] = True
        try:
            ancienne = sel_unite.value
            sel_unite.options = options
            dispo = [v for _, v in options]
            #  On conserve l'unite courante si elle survit au filtre, sinon les
            #  panneaux du bas sauteraient a chaque changement de perimetre.
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
        parts = trace.ids[points.point_inds[0]].split("/")
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
        "<b>1 · Structure du cercle</b> — choisissez jusqu'a quatre niveaux "
        "d'emboitement." + ("  Le clic sur une part filtre les panneaux."
                            if clic else ""),
        fond="#e3f2fd", coul="#0d47a1"))
    display(widgets.HBox(niveaux[:2]))
    display(widgets.HBox(niveaux[2:]))
    display(fw_cercle)

    display(_bandeau(
        "<b>2 · Perimetre et granularite</b> — la maille et la valeur filtrent. "
        "Les boutons d'axes commandent l'agregation des barres, du forest plot "
        "et du tableau : desactivez-en un pour regrouper les anomalies qui n'en "
        "differaient que par lui.", fond="#e8f5e9", coul="#1b5e20"))
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
    display(widgets.HBox([fw_barres, fw_forest],
                         layout=widgets.Layout(width="100%")))

    display(_bandeau(
        "<b>3 · Le sous-portefeuille en detail</b> — evolution de la cible sur "
        f"{N_TRIMESTRES} trimestres, puis decomposition SHAP de la prediction "
        "validee et evolution des variables qui la determinent.",
        fond="#fff3e0", coul="#e65100"))
    display(sel_unite)
    display(fw_evol)
    display(widgets.HBox([fw_shap, fw_vars],
                         layout=widgets.Layout(width="100%")))

    display(_bandeau("<b>4 · Tableau de priorisation</b> — perimetre courant, "
                     f"{N_LIGNES_TABLE} lignes les plus graves."))
    display(z_table)

    _maj_cercle()
    _maj_valeurs()
    return {"cercle": fw_cercle, "barres": fw_barres, "forest": fw_forest,
            "evolution": fw_evol, "shap": fw_shap, "variables": fw_vars,
            "maille": sel_maille, "valeur": sel_valeur, "unite": sel_unite,
            "axes": axes_btns, "echelle": sel_echelle, "cartes": z_cartes,
            "tableau": z_table, "socle": socle, "shap_moteur": shap_moteur,
            "rafraichir": _maj_panneaux}


# =============================================================================
#  5. EXECUTION AUTOMATIQUE
# =============================================================================
#  Le tableau de bord se lance tout seul si vos donnees sont en session. Sinon,
#  un message dit exactement ce qui manque, plutot que de laisser un ecran vide.
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
