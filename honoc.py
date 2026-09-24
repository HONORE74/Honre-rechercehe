
import os, sys, json, subprocess
from pathlib import Path
from datetime import datetime
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
# GWP = "Earned_Premium"
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

_EXCLURE_VARS = {"time_idx", "year", "quarter", "y_obs", "y_pred",
                 "borne_basse", "borne_haute", "dans_intervalle", "largeur",
                 "score_composite", "rank", "ecart_intervalle", "severite",
                 "A_ecart_borne", "B_erreur_modele", "sigma", "step"}

PALETTE = ["#4C72B0", "#0DB452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#0ABBC3", "#8C8C8C", "#CCB974", "#64B5CD"]

_DOUX = ["#6C8EBF", "#82B366", "#C08552", "#9673A6", "#5F9EA0"]

_ENCRE, _ACCENT, _OK = "#141B34", "#c0392b", "#3D5A9E"
_BLEU, _GRILLE, _GRIS = "#636EFA", "#EDF1F7", "#8A93A5"
_VIDE = "rgba(0,0,0,0)"

# -- Couleurs du theme ameliore --
_FOND_PAGE      = "#f4f6f9"
_FOND_CARTE     = "#ffffff"
_OMBRE_CARTE    = "0 2px 12px rgba(20,27,52,.08)"
_TITRE_PRINC    = "#0d1b3e"
_ACCENT_BLEU    = "#1a56db"
_ACCENT_VERT    = "#059669"
_ACCENT_ORANGE  = "#d97706"
_ACCENT_VIOLET  = "#7c3aed"
_ACCENT_ROSE    = "#db2777"


# =====================================================================
#  Fonctions utilitaires
# =====================================================================
def _fmt(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:,.0f}".replace(",", " ")

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


# =====================================================================
#  _bandeau  — CORRIGE : retourne le HTML complet
# =====================================================================
def _bandeau(txt, fond="#eceff1", coul="#37474f"):
    return (
        f"<div style='font-family:Inter,system-ui,sans-serif;font-size:14px;"
        f"font-weight:600;color:{coul};background:{fond};"
        f"padding:12px 18px;border-radius:8px;margin:18px 0 10px 0;"
        f"border-left:4px solid {coul};letter-spacing:.3px'>"
        f"{txt}</div>"
    )


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
        for c in self.cles:
            self.dd[c] = self.dd[c].astype(str)
            self.ex[c] = self.ex[c].astype(str)
        if COL_SCORE in self.dd.columns:
            self.dd = self.dd.dropna(subset=[COL_SCORE])
            self.dd = self.dd[self.dd[COL_SCORE] > 0]
        self.score_global = float(self.dd[COL_SCORE].sum()) \
            if COL_SCORE in self.dd.columns else 0.0
        self._df_txt = {c: df[c].astype(str).values for c in self.cles}
        self.vars_explic = [
            c for c in df.columns
            if c != target and c not in _EXCLURE_VARS
            and pd.api.types.is_numeric_dtype(df[c])]

    def _ex_pour(self, sub):
        if not len(sub):
            return self.ex.iloc[0:0]
        cles_sub = set(map(tuple, sub[self.cles].astype(str).values))
        m = np.array([tuple(row) in cles_sub
                     for row in self.ex[self.cles].astype(str).values])
        return self.ex[m]

    def filtrer(self, colonne, valeur):
        if not colonne or colonne not in self.dd.columns:
            sub, titre = self.dd, " General view"
        elif not valeur:
            sub, titre = self.dd, f"{colonne} — General view"
        else:
            m = self.dd[colonne].astype(str) == str(valeur)
            if not m.any():
                sub, titre = self.dd, f"{colonne} — General view"
            else:
                sub, titre = self.dd[m], f"{colonne} = {valeur}"
        return sub, self._ex_pour(sub), titre

    def table_axes(self, sub, sub_ex, axes):
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
            t = t.merge(s, on=axes, how="left")
            t["score"] = t["score"].fillna(0.0)
            t["score_max"] = t["score_max"].fillna(0.0)
            t["n"] = t["n"].fillna(0).astype(int)
        else:
            t["score"], t["score_max"], t["n"] = 0.0, 0.0, 0

        t["couvert"] = (t["y_obs"] >= t["lo"]) & (t["y_obs"] <= t["hi"])
        t["libelle"] = t[axes].astype(str).agg(" | ".join, axis=1).str.slice(0, 38)
        t = t.sort_values("score", ascending=False).reset_index(drop=True)
        return t

    def unites(self, table, axes, n=N_UNITES_LISTE):
        if not len(table):
            return []
        d = table.head(n)
        return [(f"{r['libelle']}   ({_fmt(r['y_obs'])})",
                 tuple(str(r[a]) for a in axes))
                for _, r in d.iterrows()]

    def historique(self, valeurs, axes, n=N_TRIMESTRES):
        m = np.ones(len(self.df), dtype=bool)
        for a, v in zip(axes, valeurs):
            m &= (self.df[a].astype(str).values == str(v))
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
        per = None
        if {"year", "quarter"} <= set(self.ex.columns):
            e = self.ex.iloc[0]
            per = f"{int(e['year'])}-Q{int(e['quarter'])}"
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
    fig = go.Figure(go.Bar(x=[], y=[], orientation="h", showlegend=False,
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
                    line=dict(color=_ENCRE, width=2.8, shape="spline",
                              smoothing=.55),
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
        f"Gravite moyenne : {_fmt4(r['score_moyen'])}<br>"
        f"Anomalies : {int(r['n'])}<br>"
        f"Score cumule : {_fmt4(r['score_total'])} "
        f"({r['part']:.1f} % du perimetre)<br>"
        f"Pire anomalie : {_fmt4(r['score_max'])}"
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
        colorbar=dict(title="Gravite<br>moyenne", thickness=16,
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
def _cartes(socle, sub, sub_ex, titre):
    pire_rang, obs_pire, interv_pire = None, "—", "—"
    score_total = 0.0
    couverture_pct = "—"

    if len(sub):
        idx = sub[COL_SCORE].idxmax()
        pire = sub.loc[idx]
        if "rank" in sub.columns and pd.notna(pire.get("rank")):
            pire_rang = int(pire["rank"])

        m = np.ones(len(sub_ex), dtype=bool)
        for c in socle.cles:
            m &= (sub_ex[c].astype(str).values == str(pire[c]))
        ligne_ex = sub_ex[m]
        if len(ligne_ex):
            obs_pire = _fmt(ligne_ex["y_obs"].sum())
            interv_pire = (f"[{_fmt(ligne_ex['borne_basse'].sum())} ; "
                           f"{_fmt(ligne_ex['borne_haute'].sum())}]")

        score_total = float(sub[COL_SCORE].sum())

        if "dans_intervalle" in sub_ex.columns and len(sub_ex):
            n_couvert = int(sub_ex["dans_intervalle"].sum())
            couverture_pct = f"{100 * n_couvert / len(sub_ex):.1f} %"

    cartes = [
        ("Anomalies",
         f"{len(sub):,}".replace(",", " "),
         _ACCENT_BLEU,
         "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 "
         "2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"),
        ("Observation — Worst anomaly",
         f"{obs_pire}  {interv_pire}",
         _ACCENT_ROSE,
         "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 "
         "2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"),
        ("Rank",
         f"#{pire_rang}" if pire_rang else "—",
         _ACCENT_VIOLET,
         "M7.5 21H2V9h5.5v12zm7.25-18h-5.5v18h5.5V3zM22 11h-5.5v10H22V11z"),
        ("Score total",
         _fmt4(score_total),
         _ACCENT_ORANGE,
         "M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 "
         "2-2V5c0-1.1-.9-2-2-2zm-7 14H5v-2h7v2zm5-4H5v-2h12v2zm0-4H5V7h12v2z"),
        ("Coverage",
         couverture_pct,
         _ACCENT_VERT,
         "M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"),
    ]

    blocs = "".join(
        f"<div style='flex:1;min-width:175px;background:{_FOND_CARTE};"
        f"border:1px solid #e5e7eb;border-radius:10px;"
        f"padding:14px 16px;position:relative;overflow:hidden;"
        f"box-shadow:{_OMBRE_CARTE}'>"

        f"<div style='position:absolute;top:10px;right:12px;opacity:.12'>"
        f"<svg width='36' height='36' viewBox='0 0 24 24' fill='{couleur}'>"
        f"<path d='{icone}'/></svg></div>"

        f"<div style='font-size:11px;color:#6b7280;"
        f"text-transform:uppercase;letter-spacing:.8px;font-weight:500'>"
        f"{titre_carte}</div>"

        f"<div style='font-size:20px;font-weight:700;color:{couleur};"
        f"margin-top:6px;line-height:1.3'>{valeur}</div>"

        f"</div>"
        for titre_carte, valeur, couleur, icone in cartes
    )

    return (
        f"<div style='font-family:Inter,system-ui,sans-serif;"
        f"margin:10px 0 18px 0'>"
        f"<div style='font-size:16px;font-weight:700;color:{_TITRE_PRINC};"
        f"margin-bottom:12px'>{titre}</div>"
        f"<div style='display:flex;gap:12px;flex-wrap:wrap'>{blocs}</div>"
        f"</div>"
    )


def _fig_barres(table, titre, axes, target, top_n):
    fig = _creer_barres()
    if not len(table):
        return _vider(fig, f"{titre}<br><sup>Aucune anomalie</sup>", 260)
    g = table.head(top_n).iloc[::-1]
    survol = [
        f"<b>{r['libelle']}</b><br>{target} observe : {_fmt(r['y_obs'])}"
        f"<br>Predit : {_fmt(r['y_pred'])}"
        f"<br>Intervalle : [{_fmt(r['lo'])} ; {_fmt(r['hi'])}]"
        f"<br>Score cumule : {_fmt4(r['score'])}"
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


def _fig_evolution(hist, per, ctx_, valeurs, axes, target, alpha):
    fig = _creer_evolution(target)
    if hist is None or not len(hist):
        return _vider(fig, "Aucun historique pour ce sous-portefeuille.")
    val = hist[target].to_numpy(dtype="float64")
    p_valide = ctx_["per"] if ctx_ and ctx_.get("per") in per else per[-1]
    xb = yb = xp = yp = xh = yh = []
    if ctx_:
        xb, yb = [p_valide, p_valide], [ctx_["lo"], ctx_["hi"]]
        xp, yp = [p_valide], [ctx_["pred"]]
        xh, yh = [p_valide], [ctx_["obs"]]
    couvert = bool(ctx_["couvert"]) if ctx_ else True
    coul = _OK if couvert else _ACCENT
    halo = "rgba(61,90,158,0.18)" if couvert else "rgba(192,57,43,0.20)"
    fig.data[0].x, fig.data[0].y = per, val
    fig.data[1].x, fig.data[1].y = xb, yb
    fig.data[1].name = f"Intervalle conforme {100 * (1 - alpha):.0f} %"
    fig.data[2].x, fig.data[2].y = xp, yp
    fig.data[3].x, fig.data[3].y = per, val
    fig.data[3].text = [_fmt(v) for v in val]
    fig.data[3].hovertemplate = ("<b>%{x}</b><br>" + target
                                 + " : <b>%{y:,.0f}</b><extra></extra>")
    fig.data[4].x, fig.data[4].y = xh, yh
    fig.data[4].marker.color = halo
    fig.data[5].x, fig.data[5].y = xh, yh
    fig.data[5].marker.color = coul
    fig.data[5].name = "Couvert" if couvert else "Hors intervalle"
    fig.layout.height = 420
    fig.layout.title = dict(
        text=f"<b style='color:{_ENCRE}'>{' | '.join(valeurs)}</b>"
             f"<br><span style='font-size:11px'>{target} · "
             f"{len(hist)} trimestres · {per[0]} -> {per[-1]} · "
             f"axes {' + '.join(axes)}"
             + (f" · periode validee <b>{p_valide}</b>" if ctx_ else "")
             + "</span>",
        font=dict(size=14), x=.015, xanchor="left")
    return fig


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
        text="<b>Les variables qui expliquent ce comportement</b>",
        font=dict(size=14), x=.015, xanchor="left")
    return fig


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
            "Score": [_fmt4(v) for v in d["score"]],
        })
        scores = d["score"].to_numpy(dtype="float64")
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
                             "boxShadow": _OMBRE_CARTE},
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
def _titre_dashboard(target):
    return html.Div(
        style={"background": f"linear-gradient(135deg, {_TITRE_PRINC} 0%, "
                              f"{_ACCENT_BLEU} 100%)",
               "borderRadius": "12px", "padding": "28px 32px",
               "margin": "0 0 24px 0",
               "boxShadow": "0 4px 20px rgba(13,27,62,.18)"},
        children=[
            html.Div("ANOMALY DETECTION DASHBOARD",
                     style={"fontSize": "13px", "fontWeight": "600",
                            "color": "rgba(255,255,255,.6)",
                            "letterSpacing": "2.5px",
                            "textTransform": "uppercase",
                            "marginBottom": "6px"}),
            html.Div(target,
                     style={"fontSize": "32px", "fontWeight": "800",
                            "color": "#ffffff",
                            "letterSpacing": ".5px",
                            "lineHeight": "1.2"}),
            html.Div("Conformal Prediction — Prioritization & Monitoring",
                     style={"fontSize": "14px", "fontWeight": "400",
                            "color": "rgba(255,255,255,.55)",
                            "marginTop": "8px"}),
        ]
    )


# =====================================================================
#  configurer_app
# =====================================================================
def configurer_app(app, anomalies_prio, expl, df, id_cols, target,
                   alpha=.10, top_n=TOP_N_PANNEAUX):
    socle = _Socle(anomalies_prio, expl, df, id_cols, target, alpha)
    cles = socle.cles
    defaut_maille = next((c for c in ("Lob", "Partner", "Companies", "Risk")
                          if c in cles), cles[0])

    etat = {"table": pd.DataFrame()}

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

    # -- Widgets --
    axes_checklist = dcc.Checklist(
        id="axes-checklist",
        options=[{"label": f" {c} ", "value": c} for c in cles],
        value=[c for c in cles[:2]],
        inline=True,
        inputStyle={"marginRight": "4px"},
        labelStyle={"display": "inline-block", "padding": "6px 14px",
                     "margin": "0 6px 6px 0", "border": "1px solid #c7d2fe",
                     "borderRadius": "6px", "background": "#eef2ff",
                     "cursor": "pointer", "fontSize": "13px",
                     "fontWeight": "500", "color": _ACCENT_BLEU,
                     "transition": "all .15s"})

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

    fig_cercle = dcc.Graph(id="fig-cercle", figure=_creer_cercle())
    fig_barres = dcc.Graph(id="fig-barres", figure=_creer_barres())
    fig_forest = dcc.Graph(id="fig-forest", figure=_creer_forest(target))
    fig_evol   = dcc.Graph(id="fig-evol",   figure=_creer_evolution(target))
    fig_vars   = dcc.Graph(id="fig-vars",   figure=_creer_variables())
    cartes_div  = html.Div(id="cartes-div")
    tableau_div = html.Div(id="tableau-div")

    # -- Layout --
    app.layout = html.Div(
        style={"fontFamily": "Inter, system-ui, sans-serif",
               "background": _FOND_PAGE, "minHeight": "100vh",
               "padding": "24px 28px"},
        children=[
            store_clic_valeur,

            _titre_dashboard(target),

            dcc.Markdown(_bandeau(
                "<b> Select the desired granularity for agregation </b> ",
                fond="#e3f2fd", coul="#0d47a1"), dangerously_allow_html=True),
            html.Div(axes_checklist, style={"marginBottom": "8px"}),
            html.Div(fig_cercle,
                     style={"background": "#fff", "borderRadius": "10px",
                            "boxShadow": _OMBRE_CARTE, "padding": "8px",
                            "marginBottom": "20px"}),

            dcc.Markdown(_bandeau(
                "<b>Filter scope</b>",
                fond="#e8f5e9", coul="#1b5e20"), dangerously_allow_html=True),
            html.Div([html.Div(sel_maille, style={"display": "inline-block",
                                                   "marginRight": "47px"}),
                      html.Div(sel_valeur, style={"display": "inline-block"})]),
            cartes_div,
            html.Div(fig_barres,
                     style={"background": "#fff", "borderRadius": "10px",
                            "boxShadow": _OMBRE_CARTE, "padding": "8px",
                            "marginBottom": "20px"}),
            html.Div(fig_forest,
                     style={"background": "#fff", "borderRadius": "10px",
                            "boxShadow": _OMBRE_CARTE, "padding": "8px",
                            "marginBottom": "20px"}),

            dcc.Markdown(
                f"<div style='font-family:Inter,system-ui,sans-serif;"
                f"font-size:14px;font-weight:600;"
                f"color:#e65100;background:#fff3e0;"
                f"padding:12px 18px;border-radius:8px;"
                f"margin:18px 0 10px 0;border-left:4px solid #e65100;"
                f"letter-spacing:.3px'>"
                f"<b>History of the worst anomaly's evolution</b> "
                f"</div>",
                dangerously_allow_html=True),
            sel_unite,
            html.Div(fig_evol,
                     style={"background": "#fff", "borderRadius": "10px",
                            "boxShadow": _OMBRE_CARTE, "padding": "8px",
                            "marginBottom": "20px"}),
            html.Div(fig_vars,
                     style={"background": "#fff", "borderRadius": "10px",
                            "boxShadow": _OMBRE_CARTE, "padding": "8px",
                            "marginBottom": "20px"}),

            dcc.Markdown(_bandeau("<b>Prioritization Table</b>"),
                         dangerously_allow_html=True),
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
        table = socle.table_axes(sub, sub_ex, axes)
        etat["table"] = table

        try:
            fc = _fig_cercle(socle, sub, titre, axes)
        except Exception as e:
            print(f"Cercle non mis a jour : {type(e).__name__} : {str(e)[:120]}")
            fc = no_update

        cartes = dcc.Markdown(_cartes(socle, sub, sub_ex, titre),
                              dangerously_allow_html=True)
        barres = _fig_barres(table, titre, axes, target, top_n)
        forest = _fig_forest(table, titre, axes, target, top_n)

        options = [{"label": lbl, "value": _encode_cle(axes, cle)}
                   for lbl, cle in socle.unites(table, axes)]
        dispo = [o["value"] for o in options]
        unite_valeur = (unite_ancienne if unite_ancienne in dispo
                        else (dispo[0] if dispo else None))

        tableau = _tableau(table, titre)

        return fc, cartes, barres, forest, options, unite_valeur, tableau

    @app.callback(
        Output("fig-evol", "figure"),
        Output("fig-vars", "figure"),
        Input("sel-unite", "value"),
    )
    def maj_unite(cle_encodee):
        if cle_encodee is None:
            return (_vider(_creer_evolution(target),
                           "Aucun sous-portefeuille dans ce perimetre."),
                    _vider(_creer_variables(),
                           "Aucun sous-portefeuille dans ce perimetre."))
        try:
            axes, valeurs = _decode_cle(cle_encodee)
            hist, per, n_lignes = socle.historique(valeurs, axes)
            ctx_ = socle.contexte(etat["table"], valeurs, axes)
            return (_fig_evolution(hist, per, ctx_, valeurs, axes, target, alpha),
                    _fig_variables(socle, hist, per, ctx_))
        except Exception as e:
            msg = f"Mise a jour impossible : {type(e).__name__} : {str(e)[:110]}"
            return _vider(_creer_evolution(target), msg), \
                   _vider(_creer_variables(), msg)


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
