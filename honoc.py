
import os, sys, json, subprocess
from pathlib import Path
from datetime import datetime
from urllib.parse import quote
import numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, no_update, ctx, dash_table

PORT = 8888
runurl = os.environ.get("DOMINO_RUN_HOST_PATH")
app = Dash(__name__,
           routes_pathname_prefix='/',
           requests_pathname_prefix=runurl)

# =============================================================================
#  CHARGEMENT DES DONNEES  (adapter les chemins a votre environnement Domino)
# =============================================================================
# chemin_fichier = "/domino/datasets/local/conformal_pred_actuariat/DataSet/Doss..."
# CHEMIN_MODELE  = Path("/domino/datasets/local/conformal_pred_actuariat/DataSet/...")
# base_path      = Path("/domino/datasets/local/conformal_pred_actuariat/DataSet")
# df_model = pd.read_parquet(chemin_fichier)
# df = df_model.copy()
# target_folder = base_path / "Dossier_concatener"
# results_v2    = pd.read_pickle(target_folder / "results_v2.pkl1")
# df_anomalies  = pd.read_pickle(target_folder / "anomalies_prio.pkl1")
# df.drop(columns=["annee"], inplace=True)
# results_v2.drop(columns=["annee"], inplace=True)
# print(results_v2.shape, df_anomalies.shape)
# df = results_v2.copy()

# =============================================================================
#  PARAMETRES
# =============================================================================
# anomalies_prio = df_anomalies.copy()
# TARGET = "Claims_incurred"
# GWP = "avg_dec_Claims_incurred"
# LEAKS = ["Conso", "Currencies", "period", "Reinsurance"]
# ID_COLS = [c for c in ["Partner", "Companies", "Lob", "Activity",
#            "Periodicity", "Risk"] if c in anomalies_prio.columns]
# EXCLUDE_COLS = [TARGET, "time_idx", "year", "quarter", "annee"] + LEAKS
# FEATURE_COLS = [c for c in anomalies_prio.columns if c not in EXCLUDE_COLS]
# CATEGORIELLE = [c for c in FEATURE_COLS
#                 if anomalies_prio[c].dtype == object
#                 or str(anomalies_prio[c].dtype) == "category"]
# for c in CATEGORIELLE:
#     anomalies_prio[c] = anomalies_prio[c].astype("category")
# ALPHA = 0.10
# N_CALIB = 1
# N_TEST = 1

TOP_N_PANNEAUX   = 12
N_TRIMESTRES     = 10
N_UNITES_LISTE   = 30
N_LIGNES_TABLE   = 15
N_VARS_EXPLIC    = 5
COL_SCORE        = "score_composite"
ECHELLE          = "Bluered"
VUE_GENERALE     = ""
SEP              = "\x1f"

_EXCLURE_VARS = {"time_idx", "year", "quarter", "annee", "y_obs", "y_pred",
                 "borne_basse", "borne_haute", "dans_intervalle", "largeur",
                 "score_composite", "rank", "ecart_intervalle", "severite",
                 "A_ecart_borne", "B_erreur_modele", "sigma", "step"}

PALETTE = ["#4C72B0", "#0DB452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#0ABBC3", "#8C8C8C", "#CCB974", "#64B5CD"]

_DOUX = ["#6C8EBF", "#82B366", "#C08552", "#9673A6", "#5F9EA0"]

_ENCRE, _ACCENT, _OK = "#141B34", "#c0392b", "#3D5A9E"
_BLEU, _GRILLE, _GRIS = "#636EFA", "#EDF1F7", "#8A93A5"
_VIDE = "rgba(0,0,0,0)"

# -- Page et boites du corps (presentation d'origine) --
_FOND_PAGE      = "#f4f6f9"
_OMBRE_BOITE    = "0 2px 12px rgba(20,27,52,.08)"
_STYLE_BOITE    = {"background": "#fff", "borderRadius": "10px",
                   "boxShadow": _OMBRE_BOITE, "padding": "8px",
                   "marginBottom": "20px"}

# =====================================================================
#  SYSTEME DE DESIGN  (en-tete, cartes KPI, etat des sous-portefeuilles)
# =====================================================================
_FOND_CARTE     = "#ffffff"
_BORDURE        = "#e6eaf2"
_TITRE_PRINC    = "#0f172a"
_TEXTE_DOUX     = "#64748b"
_OMBRE_CARTE    = ("0 1px 2px rgba(15,23,42,.04), "
                   "0 8px 24px -14px rgba(15,23,42,.16)")
_RAYON          = "14px"

_ACCENT_BLEU    = "#1a56db"
_ACCENT_VERT    = "#059669"
_ACCENT_ORANGE  = "#d97706"
_ACCENT_VIOLET  = "#7c3aed"
_ACCENT_ROSE    = "#e11d48"
_ACCENT_TEAL    = "#0d9488"
_ACCENT_INDIGO  = "#4f46e5"

# Teintes tres claires, pour les fonds de bandeaux et pastilles d'icone
_TEINTE = {_ACCENT_BLEU: "#eff5ff", _ACCENT_VERT: "#ecfdf5",
           _ACCENT_ORANGE: "#fff8ed", _ACCENT_VIOLET: "#f5f2ff",
           _ACCENT_ROSE: "#fff1f4", _ACCENT_TEAL: "#effcfa",
           _ACCENT_INDIGO: "#f0f0ff"}

# Trace SVG des icones (viewBox 24x24)
_IC_ALERTE = ("M12 2L1 21h22L12 2zm1 15h-2v-2h2v2zm0-4h-2V8h2v5z")
_IC_LIGNES = ("M3 5h18v2.2H3V5zm0 5.9h18v2.2H3v-2.2zM3 16.8h18V19H3v-2.2z")
_IC_CAL    = ("M19 3h-1V1h-2v2H8V1H6v2H5a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 "
              "0 002-2V5a2 2 0 00-2-2zm0 16H5V8h14v11z")
_IC_STATS  = ("M3.5 18.5l6-6 4 4L22 6.9l-1.4-1.4-7.1 8-4-4L2 17l1.5 1.5z")
_IC_RANG   = ("M7.5 21H2V9h5.5v12zm7.25-18h-5.5v18h5.5V3zM22 11h-5.5v10H22V11z")
_IC_OK     = ("M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4L9 16.2z")
_IC_CIBLE  = ("M12 2a10 10 0 100 20 10 10 0 000-20zm0 18a8 8 0 110-16 8 8 0 "
              "010 16zm0-14a6 6 0 100 12 6 6 0 000-12zm0 10a4 4 0 110-8 4 4 "
              "0 010 8zm0-6a2 2 0 100 4 2 2 0 000-4z")


# =====================================================================
#  Fonctions utilitaires
# =====================================================================
def _fmt(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:,.0f}".replace(",", " ")      # espace insecable

def _fmt4(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:,.4g}".replace(",", " ")

def _session(nom):
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
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=_GRIS),
        height=hauteur, autosize=True, separators=", ",
        hoverlabel=dict(bgcolor="white", bordercolor=_GRILLE, align="left",
                        font=dict(size=13, color=_ENCRE)),
        margin=marges or dict(l=10, r=40, t=95, b=45))
    return fig


def _bandeau(txt, fond="#eceff1", coul="#37474f"):
    return html.Div(html.B(txt), style={
        "fontFamily": "Inter, system-ui, sans-serif", "fontSize": "14px",
        "fontWeight": "600", "color": coul, "background": fond,
        "padding": "12px 18px", "borderRadius": "8px",
        "margin": "18px 0 10px 0", "borderLeft": f"4px solid {coul}",
        "letterSpacing": ".3px"})


# =====================================================================
#  Briques d'interface (composants Dash natifs)
# =====================================================================
def _icone(trace, couleur, taille=17):
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
           f"width='{taille}' height='{taille}' fill='{couleur}'>"
           f"<path d='{trace}'/></svg>")
    return html.Img(src="data:image/svg+xml;charset=utf-8," + quote(svg),
                    style={"width": f"{taille}px", "height": f"{taille}px",
                           "display": "block"})


def _pastille(trace, couleur, taille=36):
    return html.Div(
        _icone(trace, couleur, int(taille * .52)),
        style={"width": f"{taille}px", "height": f"{taille}px",
               "borderRadius": "10px", "flex": "0 0 auto",
               "background": _TEINTE.get(couleur, "#f1f5f9"),
               "display": "flex", "alignItems": "center",
               "justifyContent": "center"})


def _badge(texte, couleur):
    return html.Span(
        [html.Span(style={"width": "6px", "height": "6px",
                          "borderRadius": "50%", "background": couleur,
                          "display": "inline-block", "marginRight": "6px"}),
         texte],
        style={"display": "inline-flex", "alignItems": "center",
               "fontSize": "11px", "fontWeight": "650", "color": couleur,
               "background": _TEINTE.get(couleur, "#f1f5f9"),
               "border": f"1px solid {couleur}22", "borderRadius": "999px",
               "padding": "3px 9px", "whiteSpace": "nowrap"})


def _barre_progres(part, couleur, cible=None):
    """Barre de progression (0..1) avec repere de cible optionnel."""
    part = float(np.clip(part if np.isfinite(part) else 0.0, 0, 1))
    enfants = [html.Div(style={"width": f"{100 * part:.1f}%", "height": "100%",
                               "background": couleur, "borderRadius": "999px",
                               "transition": "width .5s ease"})]
    if cible is not None:
        enfants.append(html.Div(
            title=f"Cible {100 * cible:.0f} %",
            style={"position": "absolute", "left": f"{100 * cible:.1f}%",
                   "top": "-3px", "bottom": "-3px", "width": "2px",
                   "background": _TITRE_PRINC, "borderRadius": "2px",
                   "opacity": ".55"}))
    return html.Div(enfants, style={
        "position": "relative", "height": "6px", "background": "#eef1f6",
        "borderRadius": "999px", "marginTop": "11px"})


def _puce(couleur, texte, forme="rond"):
    rayon = "50%" if forme == "rond" else "2px"
    return html.Span([
        html.Span(style={"width": "9px", "height": "9px", "borderRadius": rayon,
                         "background": couleur, "display": "inline-block",
                         "marginRight": "6px"}),
        texte], style={"display": "inline-flex", "alignItems": "center",
                       "marginRight": "16px", "whiteSpace": "nowrap"})


def _tracker(socle, filtre):
    """Une barre par sous-portefeuille du perimetre, dans l'ordre des cles :
    rouge s'il porte une anomalie (score_composite > 0), vert sinon."""
    ex = socle._ex_perimetre(filtre)
    if not len(ex):
        return None
    anomalies = set(map(tuple, socle.dd[socle.cles].to_numpy()))
    ex = ex.sort_values(socle.cles)
    cles = [tuple(r) for r in ex[socle.cles].to_numpy()]
    statut = np.array([c in anomalies for c in cles])
    n, n_anom = len(cles), int(statut.sum())
    VERT, ROUGE, AMBRE = "#34d399", "#f43f5e", "#fbbf24"

    max_barres = 160
    if n <= max_barres:
        barres = [(ROUGE if s else VERT,
                   f"{' | '.join(c)} : {'anomalie' if s else 'sans anomalie'}")
                  for c, s in zip(cles, statut)]
    else:                       # regroupement en tranches consecutives
        bornes = np.linspace(0, n, max_barres + 1).astype(int)
        barres = []
        for a, b in zip(bornes[:-1], bornes[1:]):
            k = int(statut[a:b].sum())
            part = k / max(b - a, 1)
            coul = VERT if k == 0 else (AMBRE if part < .5 else ROUGE)
            barres.append((coul, f"{k} anomalie(s) sur {b - a} : "
                                 f"{' | '.join(cles[a])} … {' | '.join(cles[b - 1])}"))

    part = n_anom / n
    return html.Div(className="carte-kpi", style={
        "background": _FOND_CARTE, "borderRadius": _RAYON,
        "border": f"1px solid {_BORDURE}", "boxShadow": _OMBRE_CARTE,
        "padding": "16px 18px 14px 18px", "marginTop": "14px"}, children=[
        html.Div(style={"display": "flex", "justifyContent": "space-between",
                        "alignItems": "center", "gap": "12px",
                        "flexWrap": "wrap", "marginBottom": "12px"}, children=[
            html.Div([
                html.Div("État des sous-portefeuilles",
                         style={"fontSize": "13.5px", "fontWeight": "700",
                                "color": _TITRE_PRINC}),
                html.Div(f"Période validée {socle.libelle_periode('carte')}  ·  "
                         "survolez une barre pour identifier le sous-portefeuille",
                         style={"fontSize": "11.5px", "color": _TEXTE_DOUX,
                                "marginTop": "2px"}),
            ]),
            _badge(f"{n_anom} anomalie(s) sur {n}  ·  "
                   f"{100 * part:.1f} %".replace(".", ","),
                   _ACCENT_ROSE if n_anom else _ACCENT_VERT),
        ]),
        html.Div([html.Div(title=t, className="barre-tracker",
                           style={"flex": "1 1 0", "minWidth": "2px",
                                  "height": "30px", "borderRadius": "3px",
                                  "background": c})
                  for c, t in barres],
                 style={"display": "flex", "gap": "2px"}),
        html.Div(style={"display": "flex", "justifyContent": "space-between",
                        "marginTop": "9px", "fontSize": "11px",
                        "color": _TEXTE_DOUX, "flexWrap": "wrap",
                        "gap": "8px"}, children=[
            html.Span(f"Ordre : {' › '.join(socle.cles)}"),
            html.Span([_puce(VERT, "Sans anomalie", "carre"),
                       _puce(ROUGE, "Anomalie (score_composite > 0)", "carre")]
                      + ([_puce(AMBRE, "Tranche partiellement anomale", "carre")]
                         if n > max_barres else [])),
        ]),
    ])


# =====================================================================
#  Gabarit HTML : police et decor de l'en-tete, survol du tracker
# =====================================================================
_GABARIT_HTML = """<!DOCTYPE html>
<html lang="fr">
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;450;500;600;650;700;750;800&display=swap" rel="stylesheet">
    <style>
      html, body { margin:0; padding:0; background:#f4f6f9; }

      /* ---- en-tete ---- */
      .entete-grille { position:absolute; inset:0; pointer-events:none;
        background-image:
          linear-gradient(rgba(255,255,255,.055) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255,255,255,.055) 1px, transparent 1px);
        background-size:30px 30px;
        -webkit-mask-image:linear-gradient(100deg, transparent 20%, #000 75%);
        mask-image:linear-gradient(100deg, transparent 20%, #000 75%); }
      .point-vivant { width:7px; height:7px; border-radius:50%;
        background:#4ade80; display:inline-block; margin-right:10px;
        box-shadow:0 0 0 0 rgba(74,222,128,.55);
        animation:pouls 2.2s ease-out infinite; }
      @keyframes pouls {
        0%   { box-shadow:0 0 0 0 rgba(74,222,128,.55); }
        70%  { box-shadow:0 0 0 9px rgba(74,222,128,0); }
        100% { box-shadow:0 0 0 0 rgba(74,222,128,0); } }

      /* ---- etat des sous-portefeuilles ---- */
      .barre-tracker:hover { filter:brightness(.9); }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
  </body>
</html>"""


def _norm_txt(valeurs):
    """Ecriture canonique d'une cle : espaces retires, '2024.0' -> '2024'.
    Sans elle, un float d'un cote et un int de l'autre ne se joignent jamais."""
    t = pd.Series(valeurs).astype(str).str.strip()
    return t.str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True).to_numpy()


def _encode_cle(axes, valeurs):
    return json.dumps([list(axes), list(valeurs)])

def _decode_cle(s):
    if not s:
        return None
    axes, valeurs = json.loads(s)
    return tuple(axes), tuple(valeurs)


# =====================================================================
#  Classe _Socle  (inchangee)
# =====================================================================
class _Socle:
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
        for c in self.cles:                  # une seule ecriture des cles partout
            self.dd[c] = _norm_txt(self.dd[c])
            self.ex[c] = _norm_txt(self.ex[c])
        if COL_SCORE in self.dd.columns:
            self.dd = self.dd.dropna(subset=[COL_SCORE])
            self.dd = self.dd[self.dd[COL_SCORE] > 0]
        self.score_global = float(self.dd[COL_SCORE].sum()) \
            if COL_SCORE in self.dd.columns else 0.0
        self._df_txt = {c: _norm_txt(df[c]) for c in self.cles}
        self.vars_explic = [
            c for c in df.columns
            if c != target and c not in _EXCLURE_VARS
            and pd.api.types.is_numeric_dtype(df[c])]

        # --- y_obs, y_pred, bornes, dans_intervalle depuis anomalies_prio ---
        # (a defaut ceux d'expl) : c'est ce qu'affichent le graphique sous le
        # cercle, le forest plot et la Prioritization Table, comme a l'origine.
        self.source_y_obs = ("anomalies_prio" if "y_obs" in self.dd.columns
                             else "expl (anomalies_prio n'a pas de colonne y_obs)")
        for _col in ("y_obs", "y_pred", "borne_basse", "borne_haute",
                     "dans_intervalle"):
            if _col in self.dd.columns and _col in self.ex.columns:
                _map = (self.dd.groupby(self.cles, observed=True)[_col]
                        .first().reset_index()
                        .rename(columns={_col: f"_{_col}_p"}))
                self.ex = self.ex.merge(_map, on=self.cles, how="left")
                _m2 = self.ex[f"_{_col}_p"].notna()
                self.ex.loc[_m2, _col] = self.ex.loc[_m2, f"_{_col}_p"]
                self.ex.drop(columns=[f"_{_col}_p"], inplace=True)

        # Periode validee : celle des predictions (expl), a defaut la plus
        # recente de df_model.
        self.periode = self._periode_validee()
        self._m_periode = self._masque_periode(df)
        self._m_periode_ex = self._masque_periode(self.ex)
        self.diagnostic = [f"periode validee : {self.libelle_periode()}",
                           f"y_obs des graphiques et de la Prioritization "
                           f"Table : {self.source_y_obs}"]

        # --- y_obs_df : vraie valeur de df_model de chaque ligne (carte Coverage) ---
        self.ex["y_obs_df"] = np.nan
        if target in df.columns:
            obs = self._obs_df(self.cles).rename(columns={"y_obs": "_obs_df"})
            res = self.ex[self.cles].merge(obs, on=self.cles, how="left")
            trouve = res["_obs_df"].notna().to_numpy() & self._m_periode_ex
            self.ex["y_obs_df"] = np.where(trouve, res["_obs_df"].to_numpy(),
                                           np.nan)
            self.diagnostic.append(
                f"df_model : {int(trouve.sum())}/{int(self._m_periode_ex.sum())} "
                f"lignes d'expl retrouvees a la periode {self.libelle_periode()}")
            if trouve.sum() < self._m_periode_ex.sum():
                exemple = " | ".join(
                    self.ex.loc[self._m_periode_ex & ~trouve, self.cles].iloc[0])
                self.diagnostic.append(
                    f"  sans correspondance (valeur laissee vide), ex. : {exemple}")

        for ligne in self.diagnostic:
            print(f"[controle] {ligne}")

    # ---------------------------------------------------------------
    #  Periode validee et perimetre : une seule definition partout
    # ---------------------------------------------------------------
    def _periode_validee(self):
        for src in (self.ex, self.df):
            if {"year", "quarter"} <= set(src.columns):
                yq = src[["year", "quarter"]].apply(
                    pd.to_numeric, errors="coerce").dropna()
                if len(yq):
                    an = int(yq["year"].max())
                    tr = int(yq.loc[yq["year"] == an, "quarter"].max())
                    return an, tr
        return None

    def _masque_periode(self, frame):
        if self.periode is None or not {"year", "quarter"} <= set(frame.columns):
            return np.ones(len(frame), dtype=bool)
        an, tr = self.periode
        y = pd.to_numeric(frame["year"], errors="coerce").to_numpy()
        q = pd.to_numeric(frame["quarter"], errors="coerce").to_numpy()
        return (y == an) & (q == tr)

    def libelle_periode(self, style="tiret"):
        if self.periode is None:
            return "—"
        an, tr = self.periode
        return f"Q{tr} {an}" if style == "carte" else f"{an}-Q{tr}"

    def perimetre(self, colonne, valeur):
        """Filtre reellement applique par filtrer() ; None en vue generale."""
        if not colonne or colonne not in self.dd.columns or not valeur:
            return None
        v = _norm_txt([valeur])[0]
        return (colonne, v) if (self.dd[colonne] == v).any() else None

    def _masque_filtre(self, filtre):
        m = np.ones(len(self.df), dtype=bool)
        if filtre and filtre[0] in self._df_txt:
            m &= self._df_txt[filtre[0]] == filtre[1]
        return m

    def _obs_df(self, axes, filtre=None):
        """Valeur observee de la cible dans df_model, par groupe d'axes, a la
        periode validee et dans le perimetre : meme masque que historique(),
        donc egale au dernier point de la courbe du meme groupe."""
        m = self._m_periode & self._masque_filtre(filtre)
        d = pd.DataFrame({a: self._df_txt[a][m] for a in axes})
        d["y_obs"] = pd.to_numeric(self.df[self.target],
                                   errors="coerce").to_numpy()[m]
        return d.groupby(axes, observed=True)["y_obs"].sum().reset_index()

    def _ex_perimetre(self, filtre):
        ex = self.ex[self._m_periode_ex]
        if filtre and filtre[0] in ex.columns:
            ex = ex[ex[filtre[0]] == filtre[1]]
        return ex

    def _ex_pour(self, sub):
        if not len(sub):
            return self.ex.iloc[0:0]
        cles_sub = set(map(tuple, sub[self.cles].astype(str).values))
        m = np.array([tuple(row) in cles_sub
                     for row in self.ex[self.cles].astype(str).values])
        return self.ex[m]

    def filtrer(self, colonne, valeur):
        filtre = self.perimetre(colonne, valeur)
        if filtre is not None:
            sub = self.dd[self.dd[colonne] == filtre[1]]
            titre = f"{colonne} = {valeur}"
        elif colonne and colonne in self.dd.columns:
            sub, titre = self.dd, f"{colonne} — General view"
        else:
            sub, titre = self.dd, " General view"
        return sub, self._ex_pour(sub), titre

    def table_axes(self, sub, sub_ex, axes):
        """Une ligne par groupe d'axes, sommee sur les anomalies du groupe
        (valeurs d'anomalies_prio, comme a l'origine)."""
        axes = [a for a in axes if a in sub_ex.columns] or self.cles[:1]
        if not len(sub_ex):
            return pd.DataFrame(columns=axes + [
                "y_obs", "y_pred", "lo", "hi", COL_SCORE, "n",
                "libelle", "couvert"])
        montants = {"y_obs": ("y_obs", "sum"), "y_pred": ("y_pred", "sum"),
                    "lo": ("borne_basse", "sum"), "hi": ("borne_haute", "sum"),
                    "n_lignes": ("y_obs", "size")}
        t = sub_ex.groupby(axes, observed=True).agg(**montants).reset_index()

        # score_composite repris tel quel de anomalies_prio (aucun cumul) :
        # pour un groupe, celui de son anomalie la plus grave.
        if len(sub) and COL_SCORE in sub.columns:
            s = (sub.groupby(axes, observed=True)
                    .agg(**{COL_SCORE: (COL_SCORE, "max"),
                            "n": (COL_SCORE, "size")}).reset_index())
            t = t.merge(s, on=axes, how="left")
            t[COL_SCORE] = t[COL_SCORE].fillna(0.0)
            t["n"] = t["n"].fillna(0).astype(int)
        else:
            t[COL_SCORE], t["n"] = 0.0, 0

        t["couvert"] = (t["y_obs"] >= t["lo"]) & (t["y_obs"] <= t["hi"])
        t["libelle"] = t[axes].astype(str).agg(" | ".join, axis=1).str.slice(0, 38)
        t = t.sort_values(COL_SCORE, ascending=False).reset_index(drop=True)
        return t

    def table_complete(self, sub, axes, filtre=None):
        """Pour le graphique a la maille la plus fine, dont la courbe vient de
        df_model : groupes contenant au moins une anomalie, montants sommes sur
        TOUS leurs sous-portefeuilles du perimetre (observe df_model,
        prediction et bornes d'expl), pour que le point observe et
        l'intervalle portent sur les memes lignes."""
        axes = [a for a in axes if a in self.cles] or self.cles[:1]
        colonnes = axes + ["y_obs", "y_pred", "lo", "hi", COL_SCORE, "n",
                           "n_lignes", "libelle", "couvert"]
        if not len(sub) or COL_SCORE not in sub.columns:
            return pd.DataFrame(columns=colonnes)

        t = (sub.groupby(axes, observed=True)
                .agg(**{COL_SCORE: (COL_SCORE, "max"),
                        "n": (COL_SCORE, "size")}).reset_index())
        t = t.merge(self._obs_df(axes, filtre), on=axes, how="left")

        ex = self._ex_perimetre(filtre)
        prev = (ex.groupby(axes, observed=True)
                  .agg(y_pred=("y_pred", "sum"), lo=("borne_basse", "sum"),
                       hi=("borne_haute", "sum"),
                       n_lignes=("y_pred", "size")).reset_index())
        t = t.merge(prev, on=axes, how="left")
        t["n_lignes"] = t["n_lignes"].fillna(0).astype(int)

        t["couvert"] = (t["y_obs"] >= t["lo"]) & (t["y_obs"] <= t["hi"])
        t["libelle"] = t[axes].astype(str).agg(" | ".join, axis=1).str.slice(0, 38)
        t = t.sort_values(COL_SCORE, ascending=False).reset_index(drop=True)
        return t

    def unites(self, table, axes, n=N_UNITES_LISTE):
        if not len(table):
            return []
        d = table.head(n)
        return [(f"{r['libelle']}   ({_fmt(r['y_obs'])})",
                 tuple(str(r[a]) for a in axes))
                for _, r in d.iterrows()]

    def historique(self, valeurs, axes, n=N_TRIMESTRES, filtre=None):
        m = self._masque_filtre(filtre)       # meme perimetre que les barres
        for a, v in zip(axes, valeurs):
            m &= (self._df_txt[a] == str(v))
        d = self.df[m]
        n_lignes = len(d)
        if not n_lignes:
            return None, [], 0
        colonnes = [self.target] + [v for v in self.vars_explic if v in d.columns]
        if {"year", "quarter"} <= set(d.columns):
            g = d.groupby(["year", "quarter"], observed=True)[colonnes].sum().reset_index()
            g = g.sort_values(["year", "quarter"]).tail(n)
            per = (g["year"].astype(int).astype(str) + "-Q"
                   + g["quarter"].astype(int).astype(str)).tolist()
            return g, per, n_lignes
        return (d[colonnes].tail(n).reset_index(drop=True),
                [str(i) for i in range(min(n, n_lignes))], n_lignes)

    def contexte(self, table, valeurs, axes):
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
        per = self.libelle_periode() if self.periode else None
        return dict(per=per, pred=float(r["y_pred"]), lo=float(r["lo"]),
                    hi=float(r["hi"]), obs=float(r["y_obs"]),
                    couvert=bool(r["couvert"]),
                    n_lignes=int(r.get("n_lignes", 1)))

    def variables_explicatives(self, hist, n=N_VARS_EXPLIC):
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

    def hierarchie(self, sub, chemin):
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


# =====================================================================
#  Figures vides (templates)
# =====================================================================
def _creer_cercle():
    fig = go.Figure(go.Sunburst(ids=[], labels=[], parents=[], values=[],
                                branchvalues="remainder"))
    _mise_en_forme(fig, 620, dict(l=10, r=10, t=100, b=15))
    return fig

def _creer_barres():
    # textposition="none" : sans lui, le texte de survol s'imprime sur les barres
    fig = go.Figure(go.Bar(x=[], y=[], orientation="h", showlegend=False,
                           textposition="none",
                           marker=dict(colorscale=ECHELLE,
                                       line=dict(width=.5, color="white"),
                                       colorbar=dict(title="Montant",
                                                     thickness=14, len=.7,
                                                     tickformat="~s"))))
    fig.update_layout(xaxis=dict(tickformat="~s"),
                      yaxis=dict(tickfont=dict(size=11)))
    _mise_en_forme(fig, 520, dict(l=10, r=40, t=95, b=50))
    return fig

def _creer_forest(target):
    fig = go.Figure()
    fig.add_scatter(x=[], y=[], mode="lines", hoverinfo="skip", opacity=.35,
                    line=dict(color="#3a6bbf", width=10),
                    name="Intervalle conforme")
    fig.add_scatter(x=[], y=[], mode="lines", showlegend=False, hoverinfo="skip",
                    line=dict(color=_ACCENT, width=2, dash="dot"))
    fig.add_scatter(x=[], y=[], mode="markers", name="Prediction",
                    marker=dict(symbol="diamond", size=10, color="white",
                                line=dict(color="black", width=1.5)))
    fig.add_scatter(x=[], y=[], mode="markers", name="Valeur observee",
                    marker=dict(size=13, color=_ACCENT,
                                line=dict(color="#7b241c", width=1.3)))
    fig.update_xaxes(tickformat="~s", title_text=f"<b>{target}</b>")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="center", x=.5))
    _mise_en_forme(fig, 520, dict(l=10, r=50, t=115, b=55))
    return fig

def _creer_evolution(target):
    fig = go.Figure()
    fig.add_scatter(x=[], y=[], mode="lines", line=dict(color=_VIDE, width=0),
                    fill="tozeroy", fillcolor="rgba(140,147,165,0.10)",
                    showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=[], y=[], mode="lines", line=dict(color=_BLEU, width=15),
                    opacity=.26, name="Intervalle conforme")
    fig.add_scatter(x=[], y=[], mode="markers", name="Prediction",
                    marker=dict(symbol="diamond", size=11, color="white",
                                line=dict(color=_ENCRE, width=1.8)))
    fig.add_scatter(x=[], y=[], mode="lines+markers+text", name=target,
                    line=dict(color=_ENCRE, width=2.8, shape="linear"),
                    marker=dict(size=9, color="white",
                                line=dict(color=_ENCRE, width=2.2)),
                    textposition="top center",
                    textfont=dict(size=9.5, color=_GRIS))
    fig.add_scatter(x=[], y=[], mode="markers", showlegend=False,
                    hoverinfo="skip",
                    marker=dict(size=34, color="rgba(192,57,43,0.20)"))
    fig.add_scatter(x=[], y=[], mode="markers", name="Statut",
                    marker=dict(size=13, color=_ACCENT,
                                line=dict(color="white", width=2.4)))
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
                        row=i + 1, col=1)
        fig.add_scatter(x=[], y=[], mode="markers", showlegend=False,
                        hoverinfo="skip",
                        marker=dict(size=13, color=c,
                                    line=dict(color="white", width=2.2)),
                        row=i + 1, col=1)
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


def _vider(fig, message, hauteur=260):
    for t in fig.data:
        t.x, t.y = [], []
        if "text" in t:
            t.text = []
        if t.type == "sunburst":
            t.ids, t.labels, t.parents, t.values = [], [], [], []
    for a in fig.layout.annotations:
        a.text = " "
    fig.layout.title = dict(text=message, font=dict(size=14), x=.015,
                            xanchor="left")
    fig.layout.height = hauteur
    return fig


# =====================================================================
#  Figures remplies
# =====================================================================
def _fig_cercle(socle, sub, titre, axes):
    fig = _creer_cercle()
    h = socle.hierarchie(sub, axes)
    if not len(h):
        return _vider(fig, "Aucune anomalie a representer.", 300)
    cmax = float(np.nanpercentile(h["score_moyen"], 95))
    if not np.isfinite(cmax) or cmax <= 0:
        cmax = float(h["score_moyen"].max()) or 1.0
    survol = [
        f"<b>{r['label']}</b><br>"
        f"Anomalies : {int(r['n'])}<br>"
        f"score_composite max : {_fmt4(r['score_max'])}<br>"
        f"score_composite moyen : {_fmt4(r['score_moyen'])}<br>"
        f"Part du perimetre : {r['part']:.1f} %"
        for _, r in h.iterrows()]
    t = fig.data[0]
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
        colorbar=dict(title="score_composite<br>moyen", thickness=16,
                      len=.7, tickformat="~s"))
    fig.layout.height = 620
    fig.layout.title = dict(
        text=f"Distribution of anomalies : {' > '.join(axes)}"
             f"<br><sup>{titre}</sup>",
        font=dict(size=22), x=.015, xanchor="left")
    return fig


# =====================================================================
#  _cartes  — CORRIGE : retourne les 5 bandeaux KPI
# =====================================================================
def _stats_df_model(socle):
    n_lignes = 0
    moyenne = vmin = vmax = None
    df = socle.df
    if socle.target in df.columns:
        col = df[socle.target].dropna()
        n_lignes = int(len(df))
        if len(col):
            moyenne = float(col.mean())
            vmin = float(col.min())
            vmax = float(col.max())
    return n_lignes, socle.libelle_periode("carte"), moyenne, vmin, vmax


def _carte(libelle, valeur, coul, icone, detail=None, extra=None):
    return html.Div(
        className="carte-kpi",
        style={"background": _FOND_CARTE, "borderRadius": _RAYON,
               "border": f"1px solid {_BORDURE}", "boxShadow": _OMBRE_CARTE,
               "padding": "15px 17px 16px 17px", "position": "relative",
               "overflow": "hidden"},
        children=[
            html.Div(style={"position": "absolute", "top": 0, "left": 0,
                            "right": 0, "height": "3px", "background": coul,
                            "opacity": ".85"}),
            html.Div(
                style={"display": "flex", "alignItems": "flex-start",
                       "justifyContent": "space-between", "gap": "10px"},
                children=[
                    html.Div(libelle,
                             style={"fontSize": "10.5px", "fontWeight": "650",
                                    "color": _TEXTE_DOUX,
                                    "textTransform": "uppercase",
                                    "letterSpacing": ".7px",
                                    "lineHeight": "1.45", "paddingTop": "3px"}),
                    _pastille(icone, coul, 34),
                ]),
            html.Div(valeur,
                     style={"fontSize": "25px", "fontWeight": "750",
                            "color": _TITRE_PRINC, "marginTop": "8px",
                            "lineHeight": "1.15",
                            "letterSpacing": "-.4px",
                            "fontVariantNumeric": "tabular-nums"}),
            extra,
            html.Div(detail,
                     style={"fontSize": "11.5px", "color": _TEXTE_DOUX,
                            "marginTop": "8px", "fontWeight": "550",
                            "lineHeight": "1.45",
                            "fontVariantNumeric": "tabular-nums",
                            "display": "-webkit-box", "WebkitLineClamp": 2,
                            "WebkitBoxOrient": "vertical", "overflow": "hidden"},
                     title=detail if isinstance(detail, str) else None)
            if detail else None,
        ])


def _pct(x):
    return f"{100 * x:.1f} %".replace(".", ",")


def _cartes(socle, sub, sub_ex, titre, filtre=None):
    n_lignes, periode, moyenne, vmin, vmax = _stats_df_model(socle)
    df = socle.df
    n_sp_df = int(len(pd.DataFrame(socle._df_txt).drop_duplicates()))
    n_trim = (int(df[["year", "quarter"]].drop_duplicates().shape[0])
              if {"year", "quarter"} <= set(df.columns) else None)

    # Perimetre : tous les sous-portefeuilles predits a la periode validee
    ex = socle._ex_perimetre(filtre)
    n_perim = len(ex)
    part_anom = len(sub) / n_perim if n_perim else float("nan")

    # Couverture empirique : observe (df_model) dans son intervalle conforme,
    # sur tous les sous-portefeuilles du perimetre, comparee a la cible 1 - alpha
    cible = 1 - socle.alpha
    obs_ok = ex["y_obs_df"].notna()
    if n_perim and obs_ok.any():
        e = ex[obs_ok]
        couverture = float(((e["y_obs_df"] >= e["borne_basse"])
                            & (e["y_obs_df"] <= e["borne_haute"])).mean())
    else:
        couverture = float("nan")

    pire_rang, pire_nom = None, "—"
    if len(sub):
        pire = sub.loc[sub[COL_SCORE].idxmax()]
        if "rank" in sub.columns and pd.notna(pire.get("rank")):
            pire_rang = int(pire["rank"])
        pire_nom = " | ".join(str(pire[c]) for c in socle.cles)

    if np.isfinite(couverture):
        ecart = couverture - cible
        badge = (_badge("conforme a la cible", _ACCENT_VERT) if ecart >= -.02
                 else _badge(f"sous la cible de {cible:.0%}".replace("%", " %"),
                             _ACCENT_ORANGE))
    else:
        badge = None

    cartes = [
        _carte("Anomalies", f"{len(sub):,}".replace(",", " "),
               _ACCENT_ROSE, _IC_ALERTE,
               f"sur {n_perim} sous-portefeuilles  ·  {_pct(part_anom)}"
               if n_perim else "dans le périmètre courant",
               _barre_progres(part_anom, _ACCENT_ROSE) if n_perim else None),
        _carte("Number of lines", f"{n_lignes:,}".replace(",", " "),
               _ACCENT_INDIGO, _IC_LIGNES,
               f"{n_sp_df} sous-portefeuilles"
               + (f"  ·  {n_trim} trimestres" if n_trim else "")),
        _carte("Période concernée", periode,
               _ACCENT_ORANGE, _IC_CAL, "période validée par le modèle"),
        _carte("Moyenne", _fmt(moyenne), _ACCENT_TEAL, _IC_STATS,
               f"Min {_fmt(vmin)}  ·  Max {_fmt(vmax)}"),
        _carte("Rank", f"#{pire_rang}" if pire_rang else "—",
               _ACCENT_VIOLET, _IC_RANG, pire_nom),
        _carte("Coverage", _pct(couverture) if np.isfinite(couverture) else "—",
               _ACCENT_VERT, _IC_OK,
               f"cible {cible:.0%} (1 − α)  ·  observé df_model".replace("%", " %"),
               html.Div([
                   html.Div(badge, style={"marginTop": "8px"}) if badge else None,
                   _barre_progres(couverture, _ACCENT_VERT, cible),
               ])),
    ]

    return html.Div([
        html.Div([html.Span("Périmètre", style={"fontWeight": "700",
                                                 "color": _TITRE_PRINC,
                                                 "marginRight": "8px"}),
                  html.Span(titre.strip())],
                 className="titre-perimetre",
                 style={"fontSize": "12.5px", "fontWeight": "500",
                        "color": _TEXTE_DOUX, "margin": "0 0 12px 2px"}),
        html.Div(cartes, className="grille-kpi",
                 style={"display": "grid", "gap": "14px",
                        "gridTemplateColumns":
                            "repeat(auto-fit, minmax(208px, 1fr))"}),
        _tracker(socle, filtre),
    ], style={"margin": "4px 0 6px 0"})


def _fig_barres(table, titre, axes, target, top_n):
    fig = _creer_barres()
    if not len(table):
        return _vider(fig, f"{titre}<br><sup>Aucune anomalie</sup>", 260)
    g = table.head(top_n).iloc[::-1]
    survol = [
        f"<b>{r['libelle']}</b><br>{target} observe : {_fmt(r['y_obs'])}"
        f"<br>Predit : {_fmt(r['y_pred'])}"
        f"<br>Intervalle : [{_fmt(r['lo'])} ; {_fmt(r['hi'])}]"
        f"<br>score_composite : {_fmt4(r[COL_SCORE])}"
        f"<br>Anomalies regroupees : {int(r['n'])}"
        for _, r in g.iterrows()]
    montants = g["y_obs"].tolist()
    t = fig.data[0]
    t.x, t.y = montants, g["libelle"].tolist()
    t.marker.color = montants
    t.marker.cmin, t.marker.cmax = min(montants), max(montants)
    t.text, t.hovertemplate = survol, "%{text}<extra></extra>"
    fig.layout.xaxis.title.text = f"<b>{target}</b>"
    fig.layout.height = max(380, 38 * len(g) + 150)
    fig.layout.title = dict(
        text=f"<br><sup>{titre}</sup>",
        font=dict(size=24), x=.015, xanchor="left")
    return fig


def _fig_forest(table, titre, axes, target, top_n):
    fig = _creer_forest(target)
    if not len(table):
        return _vider(fig, f"{titre}<br><sup>Aucune anomalie</sup>", 260)
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
        if o > h or o < l:          # depassement seulement si hors intervalle
            xs_over += [h if o > h else l, o, None]
            ys_over += [yi, yi, None]
    libelles = d["libelle"].str.slice(0, 34).tolist()
    ticks = [f"#{i + 1}  {lab}"
             for i, lab in zip(range(len(d) - 1, -1, -1), libelles)]
    textes = [
        f"<b>{lab}</b><br>{target} observe : {_fmt(o)}<br>Predit : {_fmt(p)}"
        f"<br>Intervalle : [{_fmt(l)} ; {_fmt(h)}]"
        for lab, o, p, l, h in zip(libelles, obs, pred, lo, hi)]
    fig.data[0].x, fig.data[0].y = xs_band, ys_band
    fig.data[1].x, fig.data[1].y = xs_over, ys_over
    fig.data[2].x, fig.data[2].y = pred, y
    fig.data[2].text = textes
    fig.data[2].hovertemplate = "%{text}<extra></extra>"
    fig.data[3].x, fig.data[3].y = obs, y
    fig.data[3].text = textes
    fig.data[3].hovertemplate = "%{text}<extra></extra>"
    fig.layout.xaxis.title.text = f"<b>{target}</b>"
    fig.layout.yaxis = dict(tickmode="array", tickvals=y,
                            ticktext=ticks, tickfont=dict(size=11))
    fig.layout.height = max(420, 44 * len(d) + 175)
    fig.layout.title = dict(
        text="Conformal Prediction Intervals: Observed vs. Predicted Values",
        font=dict(size=18), x=.015, xanchor="left")
    return fig


def _remplir_evolution(fig, hist, per, ctx_, valeurs, axes, target, alpha,
                       conforme):
    if hist is None or not len(hist):
        return _vider(fig, "Aucun historique pour ce sous-portefeuille.")
    val = hist[target].to_numpy(dtype="float64")
    p_valide = ctx_["per"] if ctx_ and ctx_.get("per") in per else per[-1]
    idx_valide = per.index(p_valide) if p_valide in per else len(per) - 1
    obs_vraie = float(val[idx_valide])
    xh, yh = [p_valide], [obs_vraie]
    if conforme and ctx_:
        xb, yb = [p_valide, p_valide], [ctx_["lo"], ctx_["hi"]]
        xp, yp = [p_valide], [ctx_["pred"]]
        couvert = obs_vraie >= ctx_["lo"] and obs_vraie <= ctx_["hi"]
    else:
        xb = yb = xp = yp = []
        couvert = True
    coul = _OK if couvert else _ACCENT
    halo = "rgba(61,90,158,0.18)" if couvert else "rgba(192,57,43,0.20)"
    fig.data[0].x, fig.data[0].y = per, val
    fig.data[1].x, fig.data[1].y = xb, yb
    fig.data[1].name = f"Intervalle conforme {100 * (1 - alpha):.0f} %"
    fig.data[1].showlegend = bool(conforme)
    fig.data[2].x, fig.data[2].y = xp, yp
    fig.data[2].showlegend = bool(conforme)
    fig.data[3].x, fig.data[3].y = per, val
    fig.data[3].text = [_fmt(v) for v in val]
    fig.data[3].hovertemplate = ("<b>%{x}</b><br>" + target
                                 + " : <b>%{y:,.0f}</b><extra></extra>")
    fig.data[4].x, fig.data[4].y = xh, yh
    fig.data[4].marker.color = halo
    fig.data[5].x, fig.data[5].y = xh, yh
    fig.data[5].marker.color = coul
    fig.data[5].name = (("Couvert" if couvert else "Hors intervalle")
                        if conforme else "Valeur observee")
    fig.layout.height = 420
    fig.layout.title = dict(
        text=f"<b style='color:{_ENCRE}'>{' | '.join(valeurs)}</b>"
             f"<br><span style='font-size:11px'>{target} · "
             f"{len(hist)} trimestres · {per[0]} -> {per[-1]} · "
             f"axes {' + '.join(axes)}"
             + (f" · periode validee <b>{p_valide}</b>" if (conforme and ctx_)
                else "")
             + "</span>",
        font=dict(size=14), x=.015, xanchor="left")
    return fig


def _fig_evolution(hist, per, ctx_, valeurs, axes, target, alpha):
    """Graphe 1 : evolution observee seule (sans prediction ni intervalle)."""
    fig = _creer_evolution(target)
    return _remplir_evolution(fig, hist, per, ctx_, valeurs, axes, target,
                              alpha, conforme=False)


def _fig_evolution_fine(socle, sub, target, alpha, axes, valeurs=None,
                        filtre=None):
    """Graphe 2 : prediction et intervalle conforme, sur les axes propres a ce
    graphique (par defaut tous, c'est-a-dire la maille la plus fine). Ne
    depend jamais des axes d'agregation du haut, seulement du perimetre
    (maille/valeur) et de ses propres reglages. Sans selection explicite,
    retombe sur le groupe le plus grave du perimetre."""
    fig = _creer_evolution(target)
    if sub is None or not len(sub):
        return _vider(fig, "Aucune anomalie a la maille la plus fine.")
    table = socle.table_complete(sub, axes, filtre)
    if not len(table):
        return _vider(fig, "Aucune anomalie a la maille la plus fine.")
    if not valeurs:
        valeurs = [str(table.iloc[0][a]) for a in axes]
    valeurs = list(valeurs)
    hist, per, _ = socle.historique(valeurs, axes, filtre=filtre)
    ctx_ = socle.contexte(table, valeurs, axes)
    return _remplir_evolution(fig, hist, per, ctx_, valeurs, axes,
                              target, alpha, conforme=True)


def _fig_variables(socle, hist, per, ctx_):
    fig = _creer_variables()
    if hist is None or len(hist) < 3:
        n = 0 if hist is None else len(hist)
        return _vider(fig,
                      f"Classement des variables indisponible : {n} trimestre(s) "
                      "dans df, il en faut au moins 3.")
    t = socle.variables_explicatives(hist)
    if not len(t):
        return _vider(fig, "Aucune variable explicative numerique exploitable.")
    p_valide = ctx_["per"] if ctx_ and ctx_.get("per") in per else per[-1]
    k = per.index(p_valide)
    for i in range(N_VARS_EXPLIC):
        t_l, t_p = fig.data[2 * i], fig.data[2 * i + 1]
        ann = fig.layout.annotations[i]
        if i < len(t):
            v = t.iloc[i]["variable"]
            vals = hist[v].to_numpy(dtype="float64")
            t_l.x, t_l.y = per, vals
            t_l.hovertemplate = (f"<b>%{{x}}</b><br>{v} : "
                                 "<b>%{y:,.0f}</b><extra></extra>")
            t_p.x, t_p.y = [per[k]], [vals[k]]
            ann.text = f"<b>{v}</b>"
            ann.font.color = _DOUX[i % len(_DOUX)]
        else:
            t_l.x, t_l.y = [], []
            t_p.x, t_p.y = [], []
            ann.text = " "
    fig.layout.height = 150 * max(len(t), 1) + 120
    fig.layout.title = dict(
        text="<b>Les variables explicatives qui expliquent ce comportement</b>",
        font=dict(size=14), x=.015, xanchor="left")
    return fig


def _fig_variable_unique(socle, hist, per, ctx_, var_name):
    """Sparkline pour une variable explicative selectionnee via le dropdown."""
    if hist is None or len(hist) < 3 or var_name not in hist.columns:
        return html.P(f"Variable '{var_name}' indisponible.",
                      style={"color": _GRIS, "padding": "12px"})
    vals = hist[var_name].to_numpy(dtype="float64", na_value=np.nan)
    if not np.isfinite(vals).all():
        return html.P(f"Variable '{var_name}' contient des valeurs non finies.",
                      style={"color": _GRIS, "padding": "12px"})
    p_valide = ctx_["per"] if ctx_ and ctx_.get("per") in per else per[-1]
    k = per.index(p_valide)
    c = _DOUX[0]
    fig = go.Figure()
    fig.add_scatter(x=per, y=vals, mode="lines+markers", showlegend=False,
                    line=dict(color=c, width=2.3, shape="linear"),
                    marker=dict(size=6, color="white",
                                line=dict(color=c, width=1.8)),
                    hovertemplate=(f"<b>%{{x}}</b><br>{var_name} : "
                                   "<b>%{y:,.0f}</b><extra></extra>"))
    fig.add_scatter(x=[per[k]], y=[vals[k]], mode="markers", showlegend=False,
                    hoverinfo="skip",
                    marker=dict(size=13, color=c,
                                line=dict(color="white", width=2.2)))
    _mise_en_forme(fig, 200, dict(l=85, r=45, t=50, b=35))
    fig.update_layout(title=dict(text=f"<b>{var_name}</b>",
                                 font=dict(size=13, color=c),
                                 x=.015, xanchor="left"),
                      hovermode="x unified")
    fig.update_xaxes(showgrid=False, linecolor=_GRILLE,
                     title_text="<b>Trimestre</b>")
    fig.update_yaxes(gridcolor=_GRILLE, zeroline=False, tickformat="~s",
                     tickfont=dict(size=9))
    return dcc.Graph(figure=fig)


def _tableau(table, titre):
    if not len(table):
        return html.P(f"Aucune anomalie dans le perimetre : {titre}")
    try:
        d = table.head(N_LIGNES_TABLE)
        n = len(d)
        t = pd.DataFrame({
            "Rang": range(1, n + 1),
            "Maille": d["libelle"].values,
            "Observe": [_fmt(v) for v in d["y_obs"]],
            "Prediction": [_fmt(v) for v in d["y_pred"]],
            "Borne inferieure": [_fmt(v) for v in d["lo"]],
            "Borne superieure": [_fmt(v) for v in d["hi"]],
            COL_SCORE: [_fmt4(v) for v in d[COL_SCORE]],
        })
        scores = d[COL_SCORE].to_numpy(dtype="float64")
        smin, smax = float(scores.min()), float(scores.max())
        etendue = (smax - smin) or 1.0
        degrade = [{"if": {"row_index": i},
                    "backgroundColor": f"rgba(198,40,40,{0.12 + 0.55 * (scores[i]-smin)/etendue})"}
                   for i in range(n)]
        return html.Div([
            html.Div(f"Prioritization Table — {titre}",
                     style={"fontWeight": "700", "fontSize": "20px",
                            "margin": "4px 0 14px 0", "color": _TITRE_PRINC}),
            dash_table.DataTable(
                data=t.to_dict("records"),
                columns=[{"name": c, "id": c} for c in t.columns],
                style_as_list_view=True,
                style_table={"overflowX": "auto", "borderRadius": "8px",
                             "boxShadow": _OMBRE_BOITE},
                style_cell={"fontFamily": "Inter, system-ui, sans-serif",
                            "fontSize": "13px",
                            "padding": "10px 14px", "border": "none",
                            "borderBottom": "1px solid #e5e7eb"},
                style_cell_conditional=[
                    {"if": {"column_id": "Maille"}, "textAlign": "left",
                     "fontWeight": "600"}],
                style_header={"backgroundColor": "#f8fafc", "fontWeight": "700",
                              "color": _TITRE_PRINC, "fontSize": "13px",
                              "border": "none",
                              "borderBottom": f"2px solid {_ACCENT_BLEU}"},
                style_data_conditional=degrade,
            ),
        ])
    except Exception as e:
        return html.P(f"Tableau non genere : {type(e).__name__} : {str(e)[:150]}")


# =====================================================================
#  Grand titre du dashboard
# =====================================================================
def _titre_dashboard(socle):
    target = socle.target
    n_anomalies = len(socle.dd)
    n_sp = len(socle.ex[socle._m_periode_ex])

    def _meta(libelle, valeur):
        return html.Div([
            html.Div(libelle,
                     style={"fontSize": "10px", "fontWeight": "600",
                            "letterSpacing": "1.1px", "textTransform": "uppercase",
                            "color": "rgba(255,255,255,.55)"}),
            html.Div(valeur,
                     style={"fontSize": "19px", "fontWeight": "750",
                            "color": "#fff", "marginTop": "3px",
                            "fontVariantNumeric": "tabular-nums"}),
        ], style={"padding": "0 22px",
                  "borderLeft": "1px solid rgba(255,255,255,.16)"})

    def _pastille_entete(icone, texte):
        return html.Span([_icone(icone, "rgba(255,255,255,.85)", 12),
                          html.Span(texte)],
                         style={"display": "inline-flex", "alignItems": "center",
                                "gap": "7px", "fontSize": "11.5px",
                                "fontWeight": "550", "color": "rgba(255,255,255,.9)",
                                "background": "rgba(255,255,255,.09)",
                                "border": "1px solid rgba(255,255,255,.16)",
                                "borderRadius": "999px", "padding": "5px 12px",
                                "backdropFilter": "blur(4px)"})

    return html.Div(
        className="entete",
        style={"position": "relative", "overflow": "hidden",
               "background": "linear-gradient(115deg,#0b1437 0%,#17307a 52%,"
                             "#2f4fd0 100%)",
               "borderRadius": "20px", "padding": "32px 36px 28px 36px",
               "margin": "0 0 16px 0",
               "boxShadow": "0 18px 40px -22px rgba(15,32,90,.65)"},
        children=[
            html.Div(className="entete-grille"),        # trame decorative
            html.Div(style={  # halo decoratif
                "position": "absolute", "top": "-140px", "right": "-80px",
                "width": "460px", "height": "460px", "borderRadius": "50%",
                "background": "radial-gradient(circle,rgba(129,140,248,.50) 0%,"
                              "rgba(129,140,248,0) 66%)", "pointerEvents": "none"}),
            html.Div(
                style={"display": "flex", "alignItems": "flex-end",
                       "justifyContent": "space-between", "flexWrap": "wrap",
                       "gap": "22px", "position": "relative"},
                children=[
                    html.Div([
                        html.Div([
                            html.Span(className="point-vivant"),
                            "Anomaly Detection Dashboard",
                        ], style={"fontSize": "11px", "fontWeight": "650",
                                  "color": "rgba(255,255,255,.72)",
                                  "letterSpacing": "1.8px",
                                  "textTransform": "uppercase",
                                  "display": "flex", "alignItems": "center"}),
                        html.Div(target,
                                 style={"fontSize": "36px", "fontWeight": "800",
                                        "color": "#fff", "lineHeight": "1.1",
                                        "letterSpacing": "-1px",
                                        "margin": "12px 0 8px 0"}),
                        html.Div([
                            "Tableau de bord de visualisation de ",
                            html.B(target, style={"color": "#fff",
                                                  "fontWeight": "650"}),
                            " : détection et priorisation des observations "
                            "atypiques par Conformal Prediction, pour la "
                            "validation des indicateurs IFRS 17 et Solvabilité II.",
                        ], style={"fontSize": "13.5px", "fontWeight": "450",
                                  "color": "rgba(255,255,255,.72)",
                                  "maxWidth": "760px", "lineHeight": "1.55"}),
                        html.Div([
                            _pastille_entete(_IC_CAL, "Période validée "
                                             + socle.libelle_periode("carte")),
                            _pastille_entete(_IC_CIBLE, "Couverture cible "
                                             f"{100 * (1 - socle.alpha):.0f} %"),
                        ], style={"display": "flex", "flexWrap": "wrap",
                                  "gap": "8px", "marginTop": "18px"}),
                    ]),
                    html.Div([
                        _meta("Période", socle.libelle_periode("carte")),
                        _meta("Sous-portefeuilles",
                              f"{n_sp:,}".replace(",", " ")),
                        _meta("Anomalies",
                              f"{n_anomalies:,}".replace(",", " ")),
                    ], style={"display": "flex", "alignItems": "center",
                              "paddingBottom": "4px"}),
                ]),
        ])


# =====================================================================
#  configurer_app
# =====================================================================
def configurer_app(app, anomalies_prio, expl, df, id_cols, target,
                   alpha=.10, top_n=TOP_N_PANNEAUX):
    socle = _Socle(anomalies_prio, expl, df, id_cols, target, alpha)
    cles = socle.cles
    defaut_maille = next((c for c in ("Lob", "Partner", "Companies", "Risk")
                          if c in cles), cles[0])

    etat = {"table": pd.DataFrame(), "hist": None, "per": None, "ctx_": None}

    def _axes_actifs(axes_value):
        return axes_value or [cles[0]]

    def _options_valeurs(colonne):
        g = (socle.dd.groupby(colonne, observed=True)[COL_SCORE]
             .agg(["size", "sum"]).reset_index()
             .sort_values("sum", ascending=False))
        return [{"label": "— General view —", "value": VUE_GENERALE}] + [
            {"label": f"{r[colonne]}   ({int(r['size'])} anomalies)",
             "value": str(r[colonne])}
            for _, r in g.iterrows()]

    def _axes_fins(axes_value):
        choisis = [c for c in cles if c in (axes_value or [])]
        return choisis or [cles[0]]

    # -- Widgets --
    _style_puce = {"display": "inline-block", "padding": "6px 14px",
                   "margin": "0 6px 6px 0", "border": "1px solid #c7d2fe",
                   "borderRadius": "6px", "background": "#eef2ff",
                   "cursor": "pointer", "fontSize": "13px",
                   "fontWeight": "500", "color": _ACCENT_BLEU,
                   "transition": "all .15s"}

    axes_checklist = dcc.Checklist(
        id="axes-checklist",
        options=[{"label": f" {c} ", "value": c} for c in cles],
        value=[c for c in cles[:2]],
        inline=True,
        inputStyle={"marginRight": "4px"},
        labelStyle=_style_puce)

    sel_maille = dcc.Dropdown(
        id="sel-maille",
        options=[{"label": c, "value": c} for c in cles],
        value=defaut_maille, clearable=False,
        style={"width": "330px"})
    sel_valeur = dcc.Dropdown(
        id="sel-valeur",
        options=[{"label": "— General view —", "value": VUE_GENERALE}],
        value=VUE_GENERALE, clearable=False,
        style={"width": "470px"})
    sel_unite = dcc.Dropdown(
        id="sel-unite", options=[], value=None, clearable=False,
        style={"width": "720px"})

    store_clic_valeur = dcc.Store(id="store-clic-valeur", data=None)

    # Reglages propres au graphique a la maille la plus fine : ils ne pilotent
    # que lui. Tous les axes coches par defaut = maille la plus fine.
    axes_fine = dcc.Checklist(
        id="axes-fine",
        options=[{"label": f" {c} ", "value": c} for c in cles],
        value=list(cles),
        inline=True,
        inputStyle={"marginRight": "4px"},
        labelStyle=_style_puce)
    sel_unite_fine = dcc.Dropdown(
        id="sel-unite-fine", options=[], value=None, clearable=False,
        style={"width": "720px"})
    sel_tri_fine = dcc.Dropdown(
        id="sel-tri-fine", clearable=False, value="score",
        options=[{"label": "Trier par score_composite", "value": "score"},
                 {"label": "Trier par écart à l'intervalle", "value": "ecart"},
                 {"label": "Trier par montant observé", "value": "montant"}],
        style={"width": "330px"})

    sel_var_extra = dcc.Dropdown(
        id="sel-var-extra",
        options=[],
        value=None, clearable=True, multi=True,
        placeholder="Selectionner une ou plusieurs variables explicatives...",
        style={"width": "700px", "marginBottom": "10px"})

    fig_cercle = dcc.Graph(id="fig-cercle", figure=_creer_cercle())
    fig_barres = dcc.Graph(id="fig-barres", figure=_creer_barres())
    fig_forest = dcc.Graph(id="fig-forest", figure=_creer_forest(target))
    fig_evol   = dcc.Graph(id="fig-evol",   figure=_creer_evolution(target))
    fig_evol_fine = dcc.Graph(id="fig-evol-fine",
                              figure=_creer_evolution(target))
    fig_vars   = dcc.Graph(id="fig-vars",   figure=_creer_variables())
    cartes_div  = html.Div(id="cartes-div")
    tableau_div = html.Div(id="tableau-div")

    app.index_string = _GABARIT_HTML
    app.title = f"{target} · Anomaly Detection Dashboard"

    # -- Layout : en-tete, cartes et etat des sous-portefeuilles en haut ;
    #    tout le reste dans la presentation d'origine --
    app.layout = html.Div(
        style={"fontFamily": "Inter, system-ui, sans-serif",
               "background": _FOND_PAGE, "minHeight": "100vh",
               "padding": "24px 28px"},
        children=[
            store_clic_valeur,

            _titre_dashboard(socle),
            cartes_div,

            _bandeau(" Select the desired granularity for agregation ",
                     fond="#e3f2fd", coul="#0d47a1"),
            html.Div(axes_checklist, style={"marginBottom": "8px"}),
            html.Div(fig_cercle, style=_STYLE_BOITE),

            _bandeau("Filter scope", fond="#e8f5e9", coul="#1b5e20"),
            html.Div([html.Div(sel_maille, style={"display": "inline-block",
                                                   "marginRight": "47px"}),
                      html.Div(sel_valeur, style={"display": "inline-block"})]),

            _bandeau("Identifications des observations Atypiques",
                     fond="#fff3e0", coul="#e65100"),
            html.Div(fig_barres, style=_STYLE_BOITE),

            _bandeau("Conformal Prediction Intervalle",
                     fond="#ede7f6", coul="#4527a0"),
            html.Div(fig_forest, style=_STYLE_BOITE),

            _bandeau("Historique de l'observation",
                     fond="#e0f2f1", coul="#004d40"),
            sel_unite,
            html.Div(fig_evol, style=_STYLE_BOITE),

            _bandeau("Anomalie a la maille la plus fine — prediction et "
                     "intervalle conforme", fond="#ede7f6", coul="#4527a0"),
            html.Div("Granularité de ce graphique (par défaut, la maille la "
                     "plus fine) :",
                     style={"fontSize": "13px", "color": _GRIS,
                            "margin": "0 0 6px 2px"}),
            html.Div(axes_fine, style={"marginBottom": "8px"}),
            html.Div([html.Div(sel_unite_fine, style={"display": "inline-block",
                                                       "marginRight": "47px"}),
                      html.Div(sel_tri_fine, style={"display": "inline-block"})]),
            html.Div(fig_evol_fine, style=_STYLE_BOITE),

            _bandeau("Variables explicatives", fond="#fce4ec", coul="#880e4f"),
            html.Div(fig_vars, style={**_STYLE_BOITE, "marginBottom": "10px"}),
            html.Div(sel_var_extra, style={"marginBottom": "10px"}),
            html.Div(id="fig-vars-extra-div", style=_STYLE_BOITE),

            _bandeau("Prioritization Table"),
            tableau_div,
        ]
    )

    # -- Callbacks --
    @app.callback(
        Output("sel-maille", "value"),
        Output("store-clic-valeur", "data"),
        Input("fig-cercle", "clickData"),
        State("axes-checklist", "value"),
        prevent_initial_call=True,
    )
    def au_clic_cercle(clickData, axes_value):
        if not clickData or not clickData.get("points"):
            return no_update, no_update
        pid = clickData["points"][0].get("id", "")
        parts = pid.split(SEP) if pid else []
        axes = _axes_actifs(axes_value)
        if not parts or len(parts) > len(axes):
            return no_update, no_update
        colonne, valeur = axes[len(parts) - 1], parts[-1]
        return colonne, valeur

    @app.callback(
        Output("sel-valeur", "options"),
        Output("sel-valeur", "value"),
        Input("sel-maille", "value"),
        Input("store-clic-valeur", "data"),
    )
    def maj_valeurs(colonne, valeur_cliquee):
        options = _options_valeurs(colonne)
        dispo = [o["value"] for o in options]
        declencheurs = {t["prop_id"].split(".")[0] for t in ctx.triggered}
        if "store-clic-valeur" in declencheurs and valeur_cliquee in dispo:
            valeur = valeur_cliquee
        else:
            valeur = VUE_GENERALE
        return options, valeur

    @app.callback(
        Output("fig-cercle", "figure"),
        Output("cartes-div", "children"),
        Output("fig-barres", "figure"),
        Output("fig-forest", "figure"),
        Output("sel-unite", "options"),
        Output("sel-unite", "value"),
        Output("tableau-div", "children"),
        Input("axes-checklist", "value"),
        Input("sel-valeur", "value"),
        State("sel-maille", "value"),
        State("sel-unite", "value"),
    )
    def maj_panneaux(axes_value, valeur, maille, unite_ancienne):
        axes = _axes_actifs(axes_value)
        sub, sub_ex, titre = socle.filtrer(maille, valeur)
        filtre = socle.perimetre(maille, valeur)
        table = socle.table_axes(sub, sub_ex, axes)
        etat["table"] = table

        try:
            fc = _fig_cercle(socle, sub, titre, axes)
        except Exception as e:
            print(f"Cercle non mis a jour : {type(e).__name__} : {str(e)[:120]}")
            fc = no_update

        cartes = _cartes(socle, sub, sub_ex, titre, filtre)
        barres = _fig_barres(table, titre, axes, target, top_n)
        forest = _fig_forest(table, titre, axes, target, top_n)

        options = [{"label": lbl, "value": _encode_cle(axes, cle)}
                   for lbl, cle in socle.unites(table, axes)]
        dispo = [o["value"] for o in options]
        unite_valeur = (unite_ancienne if unite_ancienne in dispo
                        else (dispo[0] if dispo else None))

        tableau = _tableau(table, titre)

        return fc, cartes, barres, forest, options, unite_valeur, tableau

    # --- Reglages propres au graphique a la maille la plus fine ---
    @app.callback(
        Output("sel-unite-fine", "options"),
        Output("sel-unite-fine", "value"),
        Input("sel-valeur", "value"),
        Input("sel-tri-fine", "value"),
        Input("axes-fine", "value"),
        State("sel-maille", "value"),
        State("sel-unite-fine", "value"),
    )
    def maj_options_fine(valeur, tri, axes_value, maille, ancienne):
        axes = _axes_fins(axes_value)
        sub, _, _ = socle.filtrer(maille, valeur)
        if not len(sub):
            return [], None
        t = socle.table_complete(sub, axes, socle.perimetre(maille, valeur))
        if not len(t):
            return [], None
        if tri == "montant":
            t = t.assign(_tri=t["y_obs"])
        elif tri == "ecart":
            larg = (t["hi"] - t["lo"]).replace(0, np.nan)
            depass = np.maximum(t["lo"] - t["y_obs"],
                                t["y_obs"] - t["hi"]).clip(lower=0)
            t = t.assign(_tri=(depass / larg).fillna(0.0))
        else:
            t = t.assign(_tri=t[COL_SCORE])
        t = t.sort_values("_tri", ascending=False).head(N_UNITES_LISTE)
        options = []
        for _, r in t.iterrows():
            cle = tuple(str(r[a]) for a in axes)
            options.append({"label": f"{' | '.join(cle)}   ({_fmt(r['y_obs'])})",
                            "value": _encode_cle(axes, cle)})
        dispo = [o["value"] for o in options]
        return options, (ancienne if ancienne in dispo
                         else (dispo[0] if dispo else None))

    @app.callback(
        Output("fig-evol-fine", "figure"),
        Input("sel-unite-fine", "value"),
        State("sel-maille", "value"),
        State("sel-valeur", "value"),
        State("axes-fine", "value"),
    )
    def maj_evol_fine(cle_encodee, maille, valeur, axes_value):
        try:
            sub, _, _ = socle.filtrer(maille, valeur)
            axes, valeurs = _axes_fins(axes_value), None
            if cle_encodee:
                axes, valeurs = _decode_cle(cle_encodee)
            return _fig_evolution_fine(socle, sub, target, alpha, list(axes),
                                       valeurs, socle.perimetre(maille, valeur))
        except Exception as e:
            return _vider(_creer_evolution(target),
                          f"Maille fine indisponible : {type(e).__name__} : "
                          f"{str(e)[:110]}")

    @app.callback(
        Output("fig-evol", "figure"),
        Output("fig-vars", "figure"),
        Output("sel-var-extra", "options"),
        Output("sel-var-extra", "value"),
        Input("sel-unite", "value"),
        State("sel-maille", "value"),
        State("sel-valeur", "value"),
    )
    def maj_unite(cle_encodee, maille, valeur):
        vide_options = []
        if cle_encodee is None:
            return (_vider(_creer_evolution(target),
                           "Aucun sous-portefeuille dans ce perimetre."),
                    _vider(_creer_variables(),
                           "Aucun sous-portefeuille dans ce perimetre."),
                    vide_options, None)
        try:
            axes, valeurs = _decode_cle(cle_encodee)
            hist, per, n_lignes = socle.historique(
                valeurs, axes, filtre=socle.perimetre(maille, valeur))
            ctx_ = socle.contexte(etat["table"], valeurs, axes)
            etat["hist"] = hist
            etat["per"] = per
            etat["ctx_"] = ctx_
            top_vars = socle.variables_explicatives(hist)
            top_names = set(top_vars["variable"].tolist()) if len(top_vars) else set()
            all_vars = [v for v in socle.vars_explic
                        if v not in top_names and hist is not None and v in hist.columns]
            var_options = [{"label": v, "value": v} for v in sorted(all_vars)]
            return (_fig_evolution(hist, per, ctx_, valeurs, axes, target, alpha),
                    _fig_variables(socle, hist, per, ctx_),
                    var_options, None)
        except Exception as e:
            msg = f"Mise a jour impossible : {type(e).__name__} : {str(e)[:110]}"
            return (_vider(_creer_evolution(target), msg),
                    _vider(_creer_variables(), msg),
                    vide_options, None)

    @app.callback(
        Output("fig-vars-extra-div", "children"),
        Input("sel-var-extra", "value"),
        State("sel-unite", "value"),
    )
    def maj_var_extra(noms_vars, cle_encodee):
        if not noms_vars or cle_encodee is None:
            return ""
        if isinstance(noms_vars, str):
            noms_vars = [noms_vars]
        hist = etat.get("hist")
        per = etat.get("per")
        ctx_ = etat.get("ctx_")
        if hist is None or per is None:
            return ""
        return [html.Div(_fig_variable_unique(socle, hist, per, ctx_, v),
                         style={"marginBottom": "6px"})
                for v in noms_vars]


# =====================================================================
#  DEMARRAGE
# =====================================================================
_PREREQUIS = ["df_model", "anomalies_prio", "expl", "ID_COLS", "TARGET"]
_trouve = {n: _session(n) for n in _PREREQUIS}
_manquants = [n for n, (ok, _) in _trouve.items() if not ok]

if _manquants:
    print("APP NON DEMARREE : variables absentes")
    for n in _manquants:
        print(f"  - {n}")
    app.layout = html.Div("Donnees manquantes : voir la console du run.")
else:
    try:
        configurer_app(app, _trouve["anomalies_prio"][1], _trouve["expl"][1],
                        _trouve["df_model"][1], id_cols=_trouve["ID_COLS"][1],
                        target=_trouve["TARGET"][1])
    except ValueError as _e:
        app.layout = html.Div(f"TABLEAU DE BORD NON AFFICHE : {_e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)
