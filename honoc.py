# -*- coding: utf-8 -*-
# =============================================================================
#  EXTENSION DU TABLEAU DE BORD - MISSIONS 1 ET 2
#
#  Ce fichier NE MODIFIE PAS votre code. Il se branche par-dessus le tableau de
#  bord existant et lui ajoute quatre panneaux, tous synchronises avec vos
#  selecteurs :
#
#    1. Selecteur d'unite      - les sous-portefeuilles du perimetre courant,
#                                 tries du plus grave au moins grave
#    2. Evolution de la target - mission 1 (historique + intervalle conforme +
#                                 prediction + statut de couverture)
#    3. Decomposition SHAP     - mission 2, en cascade : base + contributions
#                                 = prediction, avec le reste regroupe pour que
#                                 la cascade se referme exactement
#    4. Evolution des variables- mission 2, les 5 variables les plus
#                                 determinantes sur les memes trimestres
#    5. Tableau de priorisation- votre tableau_priorisation(), en bas, filtre
#                                 sur le perimetre courant
#
#  USAGE
#  -----
#  ORDRE OBLIGATOIRE, sur deux cellules distinctes :
#
#    Cellule 1 (la votre, deja ecrite) :
#        controles = dashboard_complet(anomalies_prio, expl)
#
#    Cellule 2 (ce fichier, colle tel quel) :
#        s'auto-branche TOUT SEUL en derniere ligne, A CONDITION que
#        `controles`, `anomalies_prio`, `expl` et `df` existent DEJA dans la
#        session au moment ou vous executez CETTE cellule. Sinon, un message
#        clair vous dit ce qui manque -- rien ne reste silencieux.
#
#  Si vous collez ce fichier AVANT d'avoir execute votre `dashboard_complet()`,
#  ou dans la MEME cellule que lui, `controles` n'existe pas encore : c'est la
#  cause la plus frequente d'un dashboard qui ne s'affiche pas.
#
#  Prerequis : votre dashboard_complet() et tableau_priorisation() deja definis,
#  ainsi que ID_COLS, TARGET, ALPHA. `shap` est optionnel : sans lui, les deux
#  panneaux SHAP affichent un message au lieu de faire echouer le reste.
# =============================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output

# ┌──────────────────────────── PARAMETRES ────────────────────────────┐
N_TRIMESTRES   = 10      # enonce mission 1 : 10 derniers trimestres
N_VARS_SHAP    = 5       # enonce mission 2 : 5 variables
N_UNITES_LISTE = 30      # unites proposees dans le selecteur
N_LIGNES_TABLE = 15      # lignes du tableau de priorisation en bas
CHEMIN_MODELE  = "artefacts_modele/modele_final.joblib"
# └─────────────────────────────────────────────────────────────────────┘

_ENCRE, _ACCENT, _OK = "#141B34", "#FF5A5F", "#3D5A9E"
_BLEU, _GRILLE, _GRIS = "#636EFA", "#EDF1F7", "#8A93A5"
_DOUX = ["#6C8EBF", "#82B366", "#C08552", "#9673A6", "#5F9EA0"]
_VIDE = "rgba(0,0,0,0)"


def _fmt(v):
    """Format francais des milliers. 1249441.0 -> '1 249 441'."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:,.0f}".replace(",", " ")


def _mise_en_forme(fig, hauteur, titre=""):
    fig.update_layout(
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=_GRIS),
        title=dict(text=titre, x=.015, xanchor="left", font=dict(size=14)),
        height=hauteur, separators=", ",
        hoverlabel=dict(bgcolor="white", bordercolor=_GRILLE, align="left",
                        font=dict(size=12.5, color=_ENCRE)),
        margin=dict(l=80, r=150, t=70, b=45))
    return fig


# =============================================================================
#  1. SOCLE DE DONNEES : filtrage, unites, historique
# =============================================================================
class _Socle:
    """Prepare une fois pour toutes ce que les mises a jour reliront souvent."""

    def __init__(self, anomalies_prio, expl, df, id_cols, target, alpha):
        self.df, self.target, self.alpha = df, target, alpha
        self.cles = [c for c in id_cols
                     if c in anomalies_prio.columns and c in expl.columns
                     and c in df.columns]
        if not self.cles:
            raise ValueError("Aucune colonne d'identification commune entre "
                             "anomalies_prio, expl et df.")

        self.dd = anomalies_prio.copy()
        self.ex = expl.copy()
        for c in self.cles:
            self.dd[c] = self.dd[c].astype(str)
            self.ex[c] = self.ex[c].astype(str)
        if "score_composite" in self.dd.columns:
            self.dd = self.dd.dropna(subset=["score_composite"])

        #  Index des cles cote df, construit une seule fois. Sans lui, chaque
        #  changement de selection relirait df en entier pour chaque unite.
        self._cle_df = df[self.cles].astype(str).agg("\x1f".join, axis=1)
        self._cle_ex = self.ex[self.cles].agg("\x1f".join, axis=1)

    # ----------------------------------------------------------- filtrage
    def filtrer(self, colonne, valeur):
        """Meme logique que votre _filtrer_maille, sur dd et ex a la fois.

        Une garde en plus, et c'est elle qui empeche les panneaux de se vider.
        Quand la maille change, le selecteur de valeur porte encore, le temps
        d'un evenement, une valeur de l'ANCIENNE colonne. Filtrer dessus
        donnerait un perimetre vide et blanchirait tout l'affichage. On revient
        alors a la vue generale, qui est l'etat correct a cet instant.
        """
        if not colonne or colonne not in self.dd.columns:
            return self.dd, self.ex, "vue generale"
        if not valeur:
            return self.dd, self.ex, f"{colonne} — vue generale"
        m_dd = self.dd[colonne].astype(str) == str(valeur)
        if not m_dd.any():
            return self.dd, self.ex, f"{colonne} — vue generale"
        m_ex = self.ex[colonne].astype(str) == str(valeur)
        return self.dd[m_dd], self.ex[m_ex], f"{colonne} = {valeur}"

    # ------------------------------------------------------------ unites
    def unites(self, sub, n=N_UNITES_LISTE):
        """(label, cle) des n sous-portefeuilles les plus graves du perimetre."""
        if not len(sub):
            return []
        col = "score_composite" if "score_composite" in sub.columns else None
        d = sub.nlargest(min(n, len(sub)), col) if col else sub.head(n)
        options = []
        for _, r in d.iterrows():
            cle = tuple(str(r[c]) for c in self.cles)
            rang = (f"#{int(r['rank'])} " if "rank" in d.columns
                    and pd.notna(r.get("rank")) else "")
            score = f"  ({r[col]:,.4g})".replace(",", " ") if col else ""
            options.append((f"{rang}{' | '.join(cle)}{score}", cle))
        return options

    # -------------------------------------------------------- historique
    def historique(self, cle, n=N_TRIMESTRES):
        k = "\x1f".join(cle)
        h = self.df[self._cle_df.values == k]
        if not len(h):
            return None, []
        h = (h.sort_values("time_idx") if "time_idx" in h.columns
             else h.sort_values(["year", "quarter"])).tail(n)
        per = (h["year"].astype(int).astype(str) + "-T"
               + h["quarter"].astype(int).astype(str)).tolist()
        return h, per

    def contexte(self, cle):
        """Prediction, bornes et statut sur la periode validee."""
        k = "\x1f".join(cle)
        t = self.ex[self._cle_ex.values == k]
        if not len(t):
            return None
        r = t.iloc[0]
        return dict(per=f"{int(r['year'])}-T{int(r['quarter'])}",
                    pred=float(r["y_pred"]), lo=float(r["borne_basse"]),
                    hi=float(r["borne_haute"]), obs=float(r["y_obs"]),
                    couvert=bool(r["dans_intervalle"]))


# =============================================================================
#  2. SHAP LOCAL, avec cache
# =============================================================================
class _Shap:
    """Decomposition SHAP locale d'une prediction, calculee au plus une fois
    par sous-portefeuille. Degrade proprement si shap ou le modele manquent."""

    def __init__(self, modele=None, x_model=None, chemin=CHEMIN_MODELE):
        self.cache, self.explainer, self.raison = {}, None, None
        self.x_model = x_model
        try:
            import shap                                   # noqa: F401
        except ImportError:
            self.raison = "Le paquet `shap` n'est pas installe (pip install shap)."
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
        """-> dict(base, contrib, pred, ecart) ou None."""
        if not self.actif or ligne is None:
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
            res = dict(erreur=str(e)[:110])
        self.cache[cle] = res
        return res


# =============================================================================
#  3. CREATION DES FIGURES (structure fixe, seules les donnees changent)
# =============================================================================
#  Meme parti pris que vos _creer_fw_* / _maj_fw_* : on cree une fois la
#  structure des traces, puis on ne touche qu'a leurs donnees dans un
#  batch_update. C'est ce qui evite le clignotement et les figures qui
#  disparaissent quand une mise a jour echoue.
# =============================================================================
def _creer_fw_evolution(target):
    fig = go.Figure()
    fig.add_scatter(x=[], y=[], mode="lines", line=dict(color=_VIDE, width=0),
                    fill="tozeroy", fillcolor="rgba(140,147,165,0.10)",
                    showlegend=False, hoverinfo="skip")                  # 0 aire
    fig.add_scatter(x=[], y=[], mode="lines", line=dict(color=_BLEU, width=15),
                    opacity=.26, name="Intervalle conforme")             # 1 bande
    fig.add_scatter(x=[], y=[], mode="markers", name="Prediction",
                    marker=dict(symbol="diamond", size=11, color="white",
                                line=dict(color=_ENCRE, width=1.8)))     # 2 pred
    fig.add_scatter(x=[], y=[], mode="lines+markers+text", name=target,
                    line=dict(color=_ENCRE, width=2.8, shape="spline",
                              smoothing=.55),
                    marker=dict(size=9, color="white",
                                line=dict(color=_ENCRE, width=2.2)),
                    textposition="top center",
                    textfont=dict(size=9.5, color=_GRIS))                # 3 cible
    fig.add_scatter(x=[], y=[], mode="markers", showlegend=False,
                    hoverinfo="skip",
                    marker=dict(size=34, color="rgba(255,90,95,0.20)"))  # 4 halo
    fig.add_scatter(x=[], y=[], mode="markers", name="Statut",
                    marker=dict(size=13, color=_ACCENT,
                                line=dict(color="white", width=2.4)))    # 5 statut
    fig.update_xaxes(showgrid=False, linecolor=_GRILLE, showspikes=True,
                     spikemode="across", spikethickness=1.2, spikedash="dot",
                     spikecolor=_GRIS, title_text="<b>Trimestre</b>")
    fig.update_yaxes(gridcolor=_GRILLE, zeroline=False, tickformat=",.0f",
                     title_text=f"<b>{target}</b>")
    _mise_en_forme(fig, 400)
    fig.update_layout(hovermode="x unified",
                      legend=dict(orientation="h", y=1.06, x=1,
                                  xanchor="right", font=dict(size=11)))
    return go.FigureWidget(fig)


def _creer_fw_shap():
    fig = go.Figure(go.Waterfall(
        orientation="v", x=[], y=[], measure=[], text=[],
        textposition="outside", textfont=dict(size=10),
        connector=dict(line=dict(color=_GRILLE, width=1)),
        increasing=dict(marker=dict(color="#C0392B")),
        decreasing=dict(marker=dict(color=_OK)),
        totals=dict(marker=dict(color=_ENCRE))))
    fig.update_xaxes(showgrid=False, linecolor=_GRILLE, tickangle=-30,
                     tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=_GRILLE, zeroline=False, tickformat=",.0f")
    _mise_en_forme(fig, 430)
    return go.FigureWidget(fig)


def _creer_fw_variables(n=N_VARS_SHAP):
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        vertical_spacing=.045, subplot_titles=[" "] * n)
    for i in range(n):
        fig.add_scatter(x=[], y=[], mode="lines+markers", showlegend=False,
                        line=dict(color=_DOUX[i % len(_DOUX)], width=2.3,
                                  shape="spline", smoothing=.55),
                        marker=dict(size=6, color="white",
                                    line=dict(color=_DOUX[i % len(_DOUX)],
                                              width=1.8)),
                        row=i + 1, col=1)
        fig.add_scatter(x=[], y=[], mode="markers", showlegend=False,
                        hoverinfo="skip",
                        marker=dict(size=12, color=_DOUX[i % len(_DOUX)],
                                    line=dict(color="white", width=2.2)),
                        row=i + 1, col=1)
    fig.update_xaxes(showgrid=False, showticklabels=False, linecolor=_GRILLE)
    fig.update_xaxes(showticklabels=True, title_text="<b>Trimestre</b>",
                     row=n, col=1)
    fig.update_yaxes(gridcolor=_GRILLE, zeroline=False, tickfont=dict(size=9))
    _mise_en_forme(fig, 150 * n + 120)
    fig.update_layout(hovermode="x unified", margin=dict(l=80, r=60, t=70, b=45))
    for a in fig.layout.annotations:
        a.update(x=0, xanchor="left", font=dict(size=11, color=_GRIS))
    return go.FigureWidget(fig)


# =============================================================================
#  4. MISES A JOUR
# =============================================================================
def _titre_vide(fw, message, hauteur=260):
    """Vide reellement une figure. Le `text` compte autant que x et y : une
    etiquette laissee derriere resterait affichee sur une figure sans donnees."""
    with fw.batch_update():
        for t in fw.data:
            t.x, t.y = [], []
            if "text" in t:
                t.text = []
            if t.type == "waterfall":
                t.measure = []
        for a in fw.layout.annotations:
            a.text = " "
        fw.layout.title.text = message
        fw.layout.height = hauteur


def _maj_evolution(fw, socle, cle, h, per, ctx):
    if h is None or not len(h):
        _titre_vide(fw, "Aucun historique pour ce sous-portefeuille.")
        return
    val = h[socle.target].values.astype(float)
    nom = " | ".join(cle)

    xb = yb = xp = yp = xh = yh = []
    if ctx and ctx["per"] in per:
        xb, yb = [ctx["per"], ctx["per"]], [ctx["lo"], ctx["hi"]]
        xp, yp = [ctx["per"]], [ctx["pred"]]
        xh, yh = [ctx["per"]], [ctx["obs"]]
    couvert = bool(ctx["couvert"]) if ctx else True
    coul = _OK if couvert else _ACCENT
    halo = "rgba(61,90,158,0.18)" if couvert else "rgba(255,90,95,0.20)"

    delta = 100 * (val[-1] - val[0]) / abs(val[0]) if val[0] else np.nan
    fleche = "▲" if (np.isfinite(delta) and delta >= 0) else "▼"

    with fw.batch_update():
        fw.data[0].x, fw.data[0].y = per, val
        fw.data[1].x, fw.data[1].y = xb, yb
        fw.data[1].name = f"Intervalle conforme {100 * (1 - socle.alpha):.0f} %"
        fw.data[2].x, fw.data[2].y = xp, yp
        fw.data[3].x, fw.data[3].y = per, val
        fw.data[3].text = [_fmt(v) for v in val]
        fw.data[3].hovertemplate = ("<b>%{x}</b><br>" + socle.target
                                    + " : <b>%{y:,.0f}</b><extra></extra>")
        fw.data[4].x, fw.data[4].y = xh, yh
        fw.data[4].marker.color = halo
        fw.data[5].x, fw.data[5].y = xh, yh
        fw.data[5].marker.color = coul
        fw.data[5].name = "Couvert" if couvert else "Hors intervalle"
        fw.layout.height = 400
        fw.layout.title.text = (
            f"<b style='color:{_ENCRE}'>{nom}</b>"
            f"<br><span style='font-size:11px'>{socle.target} · {len(h)} "
            f"trimestres · {per[0]} → {per[-1]} · "
            f"<span style='color:{_ACCENT if delta < 0 else _OK}'>{fleche} "
            f"{abs(delta):.1f} %</span>"
            + (f" · periode validee <b>{ctx['per']}</b>" if ctx else "")
            + "</span>")


def _maj_shap(fw, dec, n=N_VARS_SHAP):
    if dec is None:
        _titre_vide(fw, "Decomposition SHAP indisponible.")
        return
    if "erreur" in dec:
        _titre_vide(fw, f"SHAP : {dec['erreur']}")
        return

    contrib, base, pred = dec["contrib"], dec["base"], dec["pred"]
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(n)
    reste = contrib.sum() - top.sum()

    #  "Autres variables" est indispensable : sans ce terme, la cascade ne se
    #  refermerait pas sur la prediction et le graphique serait faux.
    x = ["Base"] + list(top.index) + ["Autres variables", "Prediction"]
    y = [base] + list(top.values) + [reste, 0]
    mesure = ["absolute"] + ["relative"] * (len(top) + 1) + ["total"]
    txt = ([_fmt(base)]
           + [("+" if v >= 0 else "−") + _fmt(abs(v)) for v in top.values]
           + [("+" if reste >= 0 else "−") + _fmt(abs(reste)), _fmt(pred)])

    exact = dec["ecart"] < 1e-6 * max(abs(pred), 1.0)
    with fw.batch_update():
        t = fw.data[0]
        t.x, t.y, t.measure, t.text = x, y, mesure, txt
        t.hovertemplate = "<b>%{x}</b><br>%{text}<extra></extra>"
        fw.layout.height = 430
        fw.layout.title.text = (
            "<b>Decomposition SHAP locale</b>"
            f"<br><span style='font-size:11px'>base {_fmt(base)} + "
            f"contributions = prediction {_fmt(pred)} · "
            + (f"<span style='color:{_OK}'>reconstitution exacte</span>" if exact
               else f"<span style='color:{_ACCENT}'>ecart {dec['ecart']:.3g}, "
                    "A VERIFIER</span>")
            + "</span>")


def _maj_variables(fw, socle, dec, h, per, ctx, n=N_VARS_SHAP):
    if dec is None or "erreur" in dec or h is None or not len(h):
        _titre_vide(fw, "Evolution des variables indisponible.", hauteur=260)
        return

    contrib = dec["contrib"]
    #  On prend les n plus determinantes, PUIS on retire celles qu'on ne peut
    #  pas tracer. On ne les remplace pas par la 6e : l'enonce demande les 5
    #  plus determinantes, pas les 5 plus determinantes que l'on sait dessiner.
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(n)
    tracables = [v for v in top.index
                 if v in h.columns and pd.api.types.is_numeric_dtype(h[v])]
    ecartees = [v for v in top.index if v not in tracables]

    k = per.index(ctx["per"]) if ctx and ctx["per"] in per else len(per) - 1
    total_abs = contrib.abs().sum() or 1.0

    with fw.batch_update():
        for i in range(n):
            t_ligne, t_point = fw.data[2 * i], fw.data[2 * i + 1]
            ann = fw.layout.annotations[i]
            if i < len(tracables):
                v = tracables[i]
                vv = h[v].values.astype(float)
                t_ligne.x, t_ligne.y = per, vv
                t_ligne.hovertemplate = (f"<b>%{{x}}</b><br>{v} : "
                                         "<b>%{y:,.4g}</b><extra></extra>")
                t_point.x, t_point.y = [per[k]], [vv[k]]
                signe = "+" if contrib[v] >= 0 else "−"
                ann.text = (f"<b>{v}</b>   SHAP {signe}{_fmt(abs(contrib[v]))}"
                            f"   ({100 * abs(contrib[v]) / total_abs:.0f} %)")
                ann.font.color = _DOUX[i % len(_DOUX)]
            else:
                t_ligne.x, t_ligne.y = [], []
                t_point.x, t_point.y = [], []
                ann.text = " "
        fw.layout.height = 150 * max(len(tracables), 1) + 120
        fw.layout.title.text = (
            f"<b>{len(tracables)} variables les plus determinantes</b>"
            f"<br><span style='font-size:11px'>memes trimestres que la cible · "
            f"periode validee <b>{per[k]}</b>"
            + (f" · ecartees car non numeriques : {', '.join(ecartees)}"
               if ecartees else "") + "</span>")


# =============================================================================
#  5. BRANCHEMENT SUR LE TABLEAU DE BORD EXISTANT
# =============================================================================
#  Registre des observateurs poses. Sans lui, relancer la cellule empilerait un
#  SECOND jeu d'observateurs sur les MEMES selecteurs : chaque changement
#  mettrait a jour l'ancien affichage en plus du nouveau, l'ancien resterait
#  visible, et les figures se dedoubleraient a chaque relance.
_BRANCHEMENTS = {}


def _debrancher(*cibles):
    """Retire les observateurs poses par un appel precedent sur ces widgets."""
    retires = 0
    for w in cibles:
        for handler in _BRANCHEMENTS.pop(getattr(w, "model_id", id(w)), []):
            try:
                w.unobserve(handler, names="value")
                retires += 1
            except Exception:
                pass
    return retires


def _observer(w, handler):
    w.observe(handler, names="value")
    _BRANCHEMENTS.setdefault(getattr(w, "model_id", id(w)), []).append(handler)


def brancher_extensions(controles, anomalies_prio, expl, df,
                        id_cols=None, target=None, alpha=None,
                        modele=None, x_model=None,
                        fonction_tableau=None, top_n_table=N_LIGNES_TABLE):
    """Ajoute les panneaux des missions 1 et 2 sous votre tableau de bord.

    controles : le dict retourne par votre dashboard_complet().
    Retourne un dict avec les nouveaux widgets, pour pouvoir les piloter.
    """
    g = globals()
    id_cols = id_cols if id_cols is not None else g.get("ID_COLS")
    target = target if target is not None else g.get("TARGET")
    alpha = alpha if alpha is not None else g.get("ALPHA", .10)
    modele = modele if modele is not None else g.get("MODELE_TE")
    fonction_tableau = fonction_tableau or g.get("tableau_priorisation")
    if id_cols is None or target is None:
        raise NameError("ID_COLS et TARGET doivent exister dans la session.")

    sel_maille = controles["maille"]
    sel_valeur = controles.get("Observation") or controles.get("valeur")
    if sel_valeur is None:
        raise KeyError("Le selecteur de valeur est introuvable dans `controles`.")

    #  Relance de la cellule : on retire d'abord les observateurs precedents.
    n_retires = _debrancher(sel_maille, sel_valeur)
    if n_retires:
        print(f"Branchement precedent retire ({n_retires} observateurs) : "
              "les anciens panneaux ne se mettront plus a jour.")

    socle = _Socle(anomalies_prio, expl, df, id_cols, target, alpha)
    moteur_shap = _Shap(modele=modele, x_model=x_model)

    sel_unite = widgets.Dropdown(
        options=[], description="3 · Unite :",
        layout=widgets.Layout(width="620px"),
        style={"description_width": "90px"})
    fw_evol = _creer_fw_evolution(target)
    fw_shap = _creer_fw_shap()
    fw_vars = _creer_fw_variables()
    z_table = widgets.Output()
    verrou = {"actif": False}

    # ------------------------------------------------------- mises a jour
    def _maj_unite(*_):
        """Les trois panneaux lies au sous-portefeuille selectionne.

        Tout est enveloppe : si une mise a jour echoue, les panneaux sont vides
        avec le message d'erreur. Une exception laissee filer dans un callback
        ipywidgets est avalee silencieusement, et l'affichage garderait alors
        l'unite PRECEDENTE en donnant l'impression que rien n'a change.
        """
        cle = sel_unite.value
        vide = "Aucun sous-portefeuille dans ce perimetre."
        if cle is None:
            for fw in (fw_evol, fw_shap, fw_vars):
                _titre_vide(fw, vide)
            return
        try:
            h, per = socle.historique(cle)
            ctx = socle.contexte(cle)
            _maj_evolution(fw_evol, socle, cle, h, per, ctx)
            if not moteur_shap.actif:
                raison = moteur_shap.raison or "SHAP indisponible."
                _titre_vide(fw_shap, raison)
                _titre_vide(fw_vars, raison)
                return
            dec = moteur_shap.decomposer(
                cle, h.iloc[[-1]] if h is not None and len(h) else None)
            _maj_shap(fw_shap, dec)
            _maj_variables(fw_vars, socle, dec, h, per, ctx)
        except Exception as e:
            msg = f"Mise a jour impossible : {type(e).__name__} : {str(e)[:120]}"
            for fw in (fw_evol, fw_shap, fw_vars):
                _titre_vide(fw, msg)

    def _maj_perimetre(*_):
        """Le perimetre a change : on repeuple la liste d'unites et le tableau."""
        if verrou["actif"]:
            return
        sub, sub_ex, titre = socle.filtrer(sel_maille.value, sel_valeur.value)
        options = socle.unites(sub)

        verrou["actif"] = True
        try:
            ancienne = sel_unite.value
            sel_unite.options = options
            dispo = [v for _, v in options]
            #  On garde l'unite courante si elle survit au nouveau filtre. Sans
            #  cela, chaque clic de selection ferait sauter le panneau ailleurs.
            sel_unite.value = (ancienne if ancienne in dispo
                               else (dispo[0] if dispo else None))
        finally:
            verrou["actif"] = False

        _maj_unite()
        with z_table:
            #  clear_output(wait=True) n'efface qu'a l'arrivee du contenu
            #  suivant. Si le rendu du tableau echouait sans etre rattrape,
            #  aucun contenu n'arriverait et l'ANCIEN tableau resterait affiche.
            #  Le try garantit qu'il arrive toujours quelque chose.
            clear_output(wait=True)
            try:
                if fonction_tableau is None:
                    print("tableau_priorisation() introuvable dans la session.")
                elif not len(sub):
                    print(f"Aucune anomalie dans le perimetre : {titre}")
                else:
                    print(f"Perimetre : {titre}   ({len(sub)} anomalies)")
                    display(fonction_tableau(sub, top_n=top_n_table))
            except Exception as e:
                print(f"Tableau non genere : {type(e).__name__} : {str(e)[:150]}")

    def _sur_unite(c):
        if c["name"] == "value" and not verrou["actif"]:
            _maj_unite()

    def _sur_perimetre(c):
        if c["name"] == "value":
            _maj_perimetre()

    _observer(sel_maille, _sur_perimetre)
    _observer(sel_valeur, _sur_perimetre)
    sel_unite.observe(_sur_unite, names="value")

    #  Clic sur une barre du panneau existant -> selectionne l'unite ici.
    #  Enveloppe dans un try : si le clic n'est pas supporte, les menus suffisent.
    def _au_clic_barre(trace, points, state):
        if not points.point_inds:
            return
        libelle = trace.y[points.point_inds[0]]
        for lab, cle in sel_unite.options:
            if libelle in lab or " | ".join(cle).startswith(str(libelle)[:20]):
                sel_unite.value = cle
                return
    try:
        controles["bar"].data[0].on_click(_au_clic_barre)
        clic_barre = True
    except Exception:
        clic_barre = False

    # ---------------------------------------------------------- affichage
    def _bandeau(txt, fond="#eceff1", coul="#37474f"):
        from IPython.display import HTML
        return HTML(f"<div style='font-family:system-ui,sans-serif;"
                    f"font-size:12.5px;color:{coul};background:{fond};"
                    f"padding:9px 13px;border-radius:6px;margin:18px 0 8px 0'>"
                    f"{txt}</div>")

    display(_bandeau(
        "<b>Missions 1 et 2</b> — evolution de la cible, decomposition SHAP et "
        "evolution des variables determinantes. Ces panneaux suivent les "
        "selecteurs ci-dessus"
        + (", et le clic sur une barre." if clic_barre else "."),
        fond="#e3f2fd", coul="#0d47a1"))
    display(sel_unite)
    display(fw_evol)
    display(widgets.HBox([fw_shap, fw_vars],
                         layout=widgets.Layout(width="100%")))
    display(_bandeau("<b>Tableau de priorisation</b> — perimetre courant, "
                     f"{top_n_table} lignes les plus graves."))
    display(z_table)

    _maj_perimetre()
    return {"unite": sel_unite, "evolution": fw_evol, "shap": fw_shap,
            "variables": fw_vars, "tableau": z_table, "socle": socle,
            "moteur_shap": moteur_shap, "rafraichir": _maj_perimetre,
            "debrancher": lambda: _debrancher(sel_maille, sel_valeur)}


# =============================================================================
#  EXECUTION AUTOMATIQUE
# =============================================================================
#  Contrairement a un simple ruban de selection, ce fichier ne peut pas se
#  brancher a l'aveugle : il lui faut `controles`, `anomalies_prio`, `expl` et
#  `df`, qui viennent de VOTRE cellule dashboard_complet(), executee AVANT
#  celle-ci. On regarde donc si ces variables existent deja dans la session.
#  Si oui, le branchement se fait tout seul. Si non, un message precis dit ce
#  qui manque, au lieu de laisser un ecran vide sans explication -- c'est
#  exactement le symptome "rien ne s'affiche" qui se reglait en silence.
def _variable_session(nom):
    """Cherche `nom` dans la session Jupyter, pas seulement dans ce module."""
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


_PREREQUIS = ["controles", "anomalies_prio", "expl", "df"]
_trouvees = {n: _variable_session(n) for n in _PREREQUIS}
_manquantes = [n for n, (ok, _) in _trouvees.items() if not ok]

if _manquantes:
    print("=" * 74)
    print("DASHBOARD NON AFFICHE : variables manquantes dans la session")
    print("=" * 74)
    for n in _manquantes:
        print(f"  - {n}")
    print()
    print("Executez D'ABORD, dans une cellule PRECEDENTE, votre code qui cree")
    print("ces variables -- typiquement :")
    print("    controles = dashboard_complet(anomalies_prio, expl)")
    print("Puis relancez CETTE cellule (celle de l'extension) une seconde fois.")
else:
    extras = brancher_extensions(
        _trouvees["controles"][1], _trouvees["anomalies_prio"][1],
        _trouvees["expl"][1], _trouvees["df"][1])
