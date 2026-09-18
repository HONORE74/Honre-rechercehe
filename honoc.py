import os
import plotly.express as px
from dash import Dash, dcc, html

PORT = 8824

def get_request_prefix(port=PORT):
    """
    Generate the request prefix based on environment variables.

    Parameters
    ----------
    port : int, optional
        The port number to include in the prefix. Default is 8887.

    Returns
    -------
    str or None
        The constructed prefix if 'DOMINO_RUN_ID' is set, otherwise None.
    """
    if "DOMINO_RUN_ID" in os.environ:
        owner = os.environ.get("DOMINO_PROJECT_OWNER")
        project = os.environ.get("DOMINO_PROJECT_NAME")
        run_id = os.environ.get("DOMINO_RUN_ID")
        port_str = str(port)  # Ensure port is a string
        vscode_proxy = ("VSCODE_PROXY_URI" in os.environ) and (
            "JUPYTER_SERVER_URL" not in os.environ
        )  # detect if vscode is used to proxy the dash app
        prefix = f"/{owner}/{project}/{'r/' if vscode_proxy else ''}notebookSession/{run_id}/proxy/{port_str}/"
    elif "JUPYTER_BASE_PATH" in os.environ:
        # Case for Datacamp
        prefix = os.environ.get("JUPYTER_BASE_PATH").replace("/web/", "/web/app/")
    else:
        prefix = None
    return prefix


get_request_prefix()

# Create app with routing config
app = Dash(__name__,
           routes_pathname_prefix='/',
           requests_pathname_prefix=get_request_prefix())


# =============================================================================
#  Mon code (migration du tableau de bord ipywidgets -> Dash)
# =============================================================================
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, no_update, ctx

TOP_N_PANNEAUX = 12
N_TRIMESTRES   = 10
N_UNITES_LISTE = 30
N_LIGNES_TABLE = 15
N_VARS_EXPLIC  = 5
COL_SCORE      = "score_composite"
COL_OBS        = "y_obs"
COL_PRED       = "y_pred"
COL_LO, COL_HI = "borne_basse", "borne_haute"
PERIODE_VALIDEE = "auto"
ECHELLE        = "Bluered"
VUE_GENERALE   = ""

SEP = "\x1f"

_EXCLURE_VARS = {"time_idx", "year", "quarter", "y_obs", "y_pred",
                 "borne_basse", "borne_haute", "dans_intervalle", "largeur",
                 "score_composite", "rank", "ecart_intervalle", "severite",
                 "A_ecart_borne", "B_erreur_modele", "sigma", "step"}

_ENCRE, _ACCENT, _OK = "#141B34", "#c0392b", "#3D5A9E"
_BLEU, _GRILLE, _GRIS = "#636EFA", "#EDF1F7", "#8A93A5"
_DOUX = ["#6C8EBF", "#82B366", "#C08552", "#9673A6", "#5F9EA0"]
_VIDE = "rgba(0,0,0,0)"


def _fmt(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:,.0f}".replace(",", " ")


def _fmt4(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:,.4g}".replace(",", " ")


def _session(nom):
    """Retrouve une variable dans une session Jupyter (%run -i) ou dans le
    module. Sur un lancement en script/app Domino pur, ne trouvera rien tant
    que tu n'as pas ajoute ton propre chargement de donnees plus bas."""
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
    return (f"<div style='font-family:system-ui,sans-serif;font-size:12.5px;"
            f"color:{coul};background:{fond};padding:9px 13px;"
            f"border-radius:6px;margin:14px 0 8px 0'>{txt}</div>")


def _encode_cle(cle):
    return json.dumps(list(cle))


def _decode_cle(s):
    return tuple(json.loads(s)) if s else None


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

        score = pd.to_numeric(valide[COL_SCORE], errors="coerce").fillna(0)
        self.ano = valide[score != 0].copy()
        self.ano[COL_SCORE] = score[score != 0].values
        if not len(self.ano):
            raise ValueError(
                f"Aucune anomalie : toutes les lignes de la periode validee "
                f"ont un {COL_SCORE} nul.")

        if COL_OBS not in self.ano.columns:
            self.ano[COL_OBS] = self.ano[target].values

        for c in self.cles:
            self.ano[c] = self.ano[c].astype(str)
        self.ano = self.ano.sort_values(COL_SCORE, ascending=False) \
                           .reset_index(drop=True)
        self.ano["rang"] = np.arange(1, len(self.ano) + 1)
        self.score_global = float(self.ano[COL_SCORE].sum()) or 1.0

        self._df_txt = {c: df[c].astype(str).values for c in self.cles}

        self.vars_explic = [
            c for c in df.columns
            if c != target and c not in _EXCLURE_VARS
            and pd.api.types.is_numeric_dtype(df[c])]

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
                    print("      ATTENTION : un seul trimestre dans df.")
                elif len(self.trimestres) < 3:
                    print("      NOTE : deux trimestres seulement.")
            print(f"   periode validee    : {per_txt}   "
                  f"({int(porte_cp.sum()):,} lignes avec prediction)"
                  .replace(",", " "))
            print(f"   anomalies retenues : {len(self.ano):,} "
                  f"({COL_SCORE} != 0)".replace(",", " "))
            print(f"   axes disponibles   : {', '.join(self.cles)}")
            print(f"   variables suivies  : {len(self.vars_explic)}")
            print("=" * 74)

    def filtrer(self, colonne, valeur):
        if not colonne or colonne not in self.ano.columns:
            return self.ano, "vue generale"
        if not valeur:
            return self.ano, f"{colonne} — vue generale"
        m = self.ano[colonne].astype(str) == str(valeur)
        if not m.any():
            return self.ano, f"{colonne} — vue generale"
        return self.ano[m], f"{colonne} = {valeur}"

    def preparer(self, sub, axes):
        axes = [a for a in axes if a in sub.columns] or self.cles[:1]
        if not len(sub):
            return sub.assign(libelle=pd.Series(dtype=str))
        t = sub.copy()
        t["libelle"] = t[axes].astype(str).agg(" | ".join, axis=1).str.slice(0, 38)
        t["libelle_rang"] = ("#" + t["rang"].astype(str) + "  " + t["libelle"])
        return t

    def unites(self, sub, axes, n=N_UNITES_LISTE):
        if not len(sub):
            return []
        t = self.preparer(sub.head(n), axes)
        return [(f"{r['libelle_rang']}   ({_fmt(r[COL_OBS])})",
                 tuple(str(r[c]) for c in self.cles))
                for _, r in t.iterrows()]

    def ligne(self, sub, cle):
        if not len(sub) or cle is None:
            return None
        m = np.ones(len(sub), dtype=bool)
        for c, v in zip(self.cles, cle):
            m &= (sub[c].astype(str).values == str(v))
        t = sub[m]
        return None if not len(t) else t.iloc[0]

    def contexte(self, r):
        if r is None:
            return None
        per = (f"{self.periode[0]}-T{self.periode[1]}" if self.periode else None)
        obs, lo, hi = float(r[COL_OBS]), float(r[COL_LO]), float(r[COL_HI])
        return dict(per=per, pred=float(r[COL_PRED]), lo=lo, hi=hi, obs=obs,
                    couvert=bool(lo <= obs <= hi), score=float(r[COL_SCORE]),
                    rang=int(r["rang"]))

    def _masque(self, textes, cles, valeurs):
        m = np.ones(len(next(iter(textes.values()))), dtype=bool)
        for a, v in zip(cles, valeurs):
            m &= (textes[a] == str(v))
        return m

    def historique(self, cle, n=N_TRIMESTRES):
        d = self.df[self._masque(self._df_txt, self.cles, cle)]
        if not len(d):
            return None, []
        colonnes = [self.target] + [v for v in self.vars_explic if v in d.columns]
        if {"year", "quarter"} <= set(d.columns):
            g = d.groupby(["year", "quarter"], observed=True)[colonnes] \
                 .sum().reset_index().sort_values(["year", "quarter"]).tail(n)
            per = (g["year"].astype(int).astype(str) + "-T"
                   + g["quarter"].astype(int).astype(str)).tolist()
            return g, per
        return d[colonnes].tail(n).reset_index(drop=True), \
            [str(i) for i in range(min(n, len(d)))]

    def variables_explicatives(self, hist, n=N_VARS_EXPLIC):
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

    def hierarchie(self, sub, chemin):
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
        text=f"Repartition des anomalies  ·  {' > '.join(axes)}"
             f"<br><sup>{titre}</sup>",
        font=dict(size=15), x=.015, xanchor="left")
    return fig


def _cartes(socle, sub, titre):
    part = sub[COL_SCORE].sum() / socle.score_global if len(sub) else 0
    hors = int((~((sub[COL_OBS] >= sub[COL_LO])
                  & (sub[COL_OBS] <= sub[COL_HI]))).sum()) if len(sub) else 0
    cartes = [("Anomalies", f"{len(sub):,}".replace(",", " "), "#37474f"),
              ("Part du score global", f"{100 * part:.1f} %", "#ad1457"),
              ("Hors intervalle", f"{hors:,}".replace(",", " "), "#00838f"),
              ("Gravite moyenne",
               _fmt4(sub[COL_SCORE].mean()) if len(sub) else "—", "#5e35b1"),
              ("Pire anomalie",
               _fmt4(sub[COL_SCORE].max()) if len(sub) else "—", "#6a1b9a")]
    blocs = "".join(
        f"<div style='flex:1;min-width:130px;background:#fff;"
        f"border:1px solid #e0e0e0;border-left:5px solid {c};"
        f"border-radius:7px;padding:11px 13px;"
        f"box-shadow:0 1px 3px rgba(0,0,0,.07)'>"
        f"<div style='font-size:10.5px;color:#78909c;"
        f"text-transform:uppercase;letter-spacing:.6px'>{t}</div>"
        f"<div style='font-size:19px;font-weight:600;color:{c};"
        f"margin-top:4px'>{v}</div></div>" for t, v, c in cartes)
    return (
        f"<div style='font-family:system-ui,sans-serif;margin:6px 0 14px 0'>"
        f"<div style='font-size:15px;font-weight:600;color:#263238;"
        f"margin-bottom:10px'>{titre}</div>"
        f"<div style='display:flex;gap:9px;flex-wrap:wrap'>{blocs}</div></div>")


def _fig_barres(socle, sub, titre, axes, target, top_n):
    fig = _creer_barres()
    if not len(sub):
        return _vider(fig, f"{titre}<br><sup>Aucune anomalie</sup>", 260)
    d = socle.preparer(sub.head(top_n), axes).iloc[::-1]
    survol = [
        f"<b>{r['libelle_rang']}</b><br>{target} observe : {_fmt(r[COL_OBS])}"
        f"<br>Predit : {_fmt(r[COL_PRED])}"
        f"<br>Intervalle : [{_fmt(r[COL_LO])} ; {_fmt(r[COL_HI])}]"
        f"<br>Score : {_fmt4(r[COL_SCORE])}"
        for _, r in d.iterrows()]
    montants = d[COL_OBS].tolist()
    t = fig.data[0]
    t.x, t.y = montants, d["libelle_rang"].tolist()
    t.marker.color = montants
    t.marker.cmin, t.marker.cmax = min(montants), max(montants)
    t.text, t.hovertemplate = survol, "%{text}<extra></extra>"
    fig.layout.xaxis.title.text = f"<b>{target}</b>"
    fig.layout.height = max(380, 38 * len(d) + 150)
    fig.layout.title = dict(
        text=f"Les {len(d)} anomalies les plus graves"
             f"<br><sup>{titre}  ·  montants bruts, aucune sommation</sup>",
        font=dict(size=14), x=.015, xanchor="left")
    return fig


def _fig_forest(socle, sub, titre, axes, target, top_n):
    fig = _creer_forest(target)
    if not len(sub):
        return _vider(fig, f"{titre}<br><sup>Aucune anomalie</sup>", 260)
    d = socle.preparer(sub.head(top_n), axes).iloc[::-1].reset_index(drop=True)
    y = list(range(len(d)))
    lo = d[COL_LO].to_numpy(dtype="float64")
    hi = d[COL_HI].to_numpy(dtype="float64")
    obs = d[COL_OBS].to_numpy(dtype="float64")
    pred = d[COL_PRED].to_numpy(dtype="float64")

    xs_band, ys_band, xs_over, ys_over = [], [], [], []
    for yi, l, h, o in zip(y, lo, hi, obs):
        xs_band += [l, h, None]
        ys_band += [yi, yi, None]
        cible = h if o > h else l
        xs_over += [cible, o, None]
        ys_over += [yi, yi, None]

    textes = [
        f"<b>{r['libelle_rang']}</b><br>{target} observe : {_fmt(o)}"
        f"<br>Predit : {_fmt(p)}<br>Intervalle : [{_fmt(l)} ; {_fmt(h)}]"
        for (_, r), o, p, l, h in zip(d.iterrows(), obs, pred, lo, hi)]

    fig.data[0].x, fig.data[0].y = xs_band, ys_band
    fig.data[1].x, fig.data[1].y = xs_over, ys_over
    fig.data[2].x, fig.data[2].y = pred, y
    fig.data[2].text = textes
    fig.data[2].hovertemplate = "%{text}<extra></extra>"
    fig.data[3].x, fig.data[3].y = obs, y
    fig.data[3].text = textes
    fig.data[3].hovertemplate = "%{text}<extra></extra>"
    fig.layout.xaxis.title.text = f"<b>{target}</b>"
    fig.layout.yaxis = dict(
        tickmode="array", tickvals=y,
        ticktext=d["libelle_rang"].str.slice(0, 40).tolist(),
        tickfont=dict(size=10))
    fig.layout.height = max(420, 44 * len(d) + 175)
    fig.layout.title = dict(
        text="Intervalle conforme, prediction et valeur observee"
             f"<br><sup>{titre}  ·  valeurs brutes de df, aucune "
             "sommation</sup>",
        font=dict(size=14), x=.015, xanchor="left")
    return fig


def _fig_evolution(hist, per, ctx_, cle, target, alpha):
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

    alerte = ("  ·  <b style='color:" + _ACCENT + "'>un seul trimestre "
              "dans df : pas de courbe possible</b>" if len(hist) < 2 else "")
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
        text=f"<b style='color:{_ENCRE}'>#{ctx_['rang'] if ctx_ else '?'}"
             f"  {' | '.join(cle)}</b>"
             f"<br><span style='font-size:11px'>{target} · "
             f"{len(hist)} trimestre(s) · {per[0]} -> {per[-1]}"
             + (f" · periode validee <b>{p_valide}</b>" if ctx_ else "")
             + alerte + "</span>",
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


def _tableau(socle, sub, titre, axes, target):
    if not len(sub):
        return html.P(f"Aucune anomalie dans le perimetre : {titre}")
    try:
        d = socle.preparer(sub.head(N_LIGNES_TABLE), axes)
        t = pd.DataFrame({
            "Rang": d["rang"].values,
            "Maille": d["libelle"].values,
            "Y_obs": d[COL_OBS].values, "Y_pred": d[COL_PRED].values,
            "CP_bas": d[COL_LO].values, "CP_haut": d[COL_HI].values,
            "Couvert": ((d[COL_OBS] >= d[COL_LO])
                        & (d[COL_OBS] <= d[COL_HI])).values,
            "Score": d[COL_SCORE].values})
        fmt_col = {c: (lambda v: _fmt(v))
                   for c in ("Y_obs", "Y_pred", "CP_bas", "CP_haut")}
        fmt_col["Score"] = lambda v: _fmt4(v)
        try:
            html_str = (t.style
                        .background_gradient(subset=["Score"], cmap="Reds")
                        .format(fmt_col)
                        .set_caption(f"Tableau de priorisation — {titre}")
                        .to_html())
        except Exception:
            html_str = t.to_html()
        return dcc.Markdown(html_str, dangerously_allow_html=True)
    except Exception as e:
        return html.P(f"Tableau non genere : {type(e).__name__} : {str(e)[:150]}")


def configurer_app(app, df, id_cols, target, alpha=.10, top_n=TOP_N_PANNEAUX):
    """Configure app.layout et les callbacks sur l'app Domino deja creee
    (elle n'est pas recreee ici : c'est celle du gabarit, avec son prefixe
    de proxy)."""
    socle = _Socle(df, id_cols, target, alpha)
    cles = socle.cles
    defaut_maille = next((c for c in ("Lob", "Partner", "Companies", "Risk")
                          if c in cles), cles[0])

    etat = {"sub": pd.DataFrame()}

    def _axes_actifs(axes_value):
        return axes_value or [cles[0]]

    def _options_valeurs(colonne):
        g = (socle.ano.groupby(colonne, observed=True)[COL_SCORE]
             .agg(["size", "sum"]).reset_index()
             .sort_values("sum", ascending=False))
        return [{"label": "— vue generale —", "value": VUE_GENERALE}] + [
            {"label": f"{r[colonne]}   ({int(r['size'])} anomalies)",
             "value": str(r[colonne])}
            for _, r in g.iterrows()]

    axes_checklist = dcc.Checklist(
        id="axes-checklist",
        options=[{"label": f" {c} ", "value": c} for c in cles],
        value=[c for c in cles[:2]],
        inline=True,
        inputStyle={"marginRight": "4px"},
        labelStyle={"display": "inline-block", "padding": "4px 10px",
                    "margin": "0 4px 4px 0", "border": "1px solid #90a4ae",
                    "borderRadius": "4px", "background": "#eef3fb",
                    "cursor": "pointer"})

    sel_maille = dcc.Dropdown(
        id="sel-maille",
        options=[{"label": c, "value": c} for c in cles],
        value=defaut_maille, clearable=False,
        style={"width": "330px"})
    sel_valeur = dcc.Dropdown(
        id="sel-valeur",
        options=[{"label": "— vue generale —", "value": VUE_GENERALE}],
        value=VUE_GENERALE, clearable=False,
        style={"width": "470px"})
    sel_unite = dcc.Dropdown(
        id="sel-unite", options=[], value=None, clearable=False,
        style={"width": "720px"})

    store_clic_valeur = dcc.Store(id="store-clic-valeur", data=None)

    fig_cercle = dcc.Graph(id="fig-cercle", figure=_creer_cercle())
    fig_barres = dcc.Graph(id="fig-barres", figure=_creer_barres())
    fig_forest = dcc.Graph(id="fig-forest", figure=_creer_forest(target))
    fig_evol = dcc.Graph(id="fig-evol", figure=_creer_evolution(target))
    fig_vars = dcc.Graph(id="fig-vars", figure=_creer_variables())
    cartes_div = html.Div(id="cartes-div")
    tableau_div = html.Div(id="tableau-div")

    app.layout = html.Div([
        store_clic_valeur,
        dcc.Markdown(_bandeau(
            "<b>Axes</b> — ils structurent le cercle et composent les libelles. "
            "Ils ne modifient aucun montant. Le clic sur une part du cercle "
            "filtre le perimetre.",
            fond="#e3f2fd", coul="#0d47a1"), dangerously_allow_html=True),
        html.Div(axes_checklist, style={"marginBottom": "8px"}),
        fig_cercle,

        dcc.Markdown(_bandeau(
            "<b>Perimetre</b> — la maille et la valeur filtrent l'ensemble du "
            "tableau de bord. Seules les anomalies "
            f"({COL_SCORE} different de zero) sont affichees.",
            fond="#e8f5e9", coul="#1b5e20"), dangerously_allow_html=True),
        html.Div([html.Div(sel_maille, style={"display": "inline-block",
                                              "marginRight": "12px"}),
                 html.Div(sel_valeur, style={"display": "inline-block"})]),
        cartes_div,
        fig_barres,
        fig_forest,

        dcc.Markdown(_bandeau(
            "<b>L'anomalie en detail</b> — evolution de la cible, intervalle "
            "conforme, prediction, puis variables explicatives.",
            fond="#fff3e0", coul="#e65100"), dangerously_allow_html=True),
        sel_unite,
        fig_evol,
        fig_vars,

        dcc.Markdown(_bandeau("<b>Tableau de priorisation</b> — perimetre "
                             f"courant, {N_LIGNES_TABLE} anomalies les plus "
                             "graves.")),
        tableau_div,
    ])

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
        sub, titre = socle.filtrer(maille, valeur)
        etat["sub"] = sub

        try:
            fc = _fig_cercle(socle, sub, titre, axes)
        except Exception as e:
            print(f"Cercle non mis a jour : {type(e).__name__} : {str(e)[:120]}")
            fc = no_update

        cartes = dcc.Markdown(_cartes(socle, sub, titre),
                              dangerously_allow_html=True)
        barres = _fig_barres(socle, sub, titre, axes, target, top_n)
        forest = _fig_forest(socle, sub, titre, axes, target, top_n)

        options = [{"label": lbl, "value": _encode_cle(cle)}
                   for lbl, cle in socle.unites(sub, axes)]
        dispo = [o["value"] for o in options]
        unite_valeur = (unite_ancienne if unite_ancienne in dispo
                        else (dispo[0] if dispo else None))

        tableau = _tableau(socle, sub, titre, axes, target)

        return fc, cartes, barres, forest, options, unite_valeur, tableau

    @app.callback(
        Output("fig-evol", "figure"),
        Output("fig-vars", "figure"),
        Input("sel-unite", "value"),
    )
    def maj_unite(cle_encodee):
        if cle_encodee is None:
            return (_vider(_creer_evolution(target),
                           "Aucune anomalie dans ce perimetre."),
                    _vider(_creer_variables(),
                           "Aucune anomalie dans ce perimetre."))
        try:
            cle = _decode_cle(cle_encodee)
            r = socle.ligne(etat["sub"], cle)
            ctx_ = socle.contexte(r)
            hist, per = socle.historique(cle)
            return (_fig_evolution(hist, per, ctx_, cle, target, alpha),
                    _fig_variables(socle, hist, per, ctx_))
        except Exception as e:
            msg = f"Mise a jour impossible : {type(e).__name__} : {str(e)[:110]}"
            return _vider(_creer_evolution(target), msg), \
                   _vider(_creer_variables(), msg)


# =============================================================================
#  CHARGEMENT DES DONNEES
#  Ce fichier tourne comme une app Domino independante : il n'a plus acces
#  aux variables de ta session Jupyter. _session() reste par securite (si tu
#  executes ce fichier via %run -i depuis un notebook), mais pour un vrai
#  lancement d'app, REMPLACE ce bloc par ton propre chargement de df,
#  ID_COLS, TARGET.
# =============================================================================
_PREREQUIS = ["df", "ID_COLS", "TARGET"]
_trouve = {n: _session(n) for n in _PREREQUIS}
_manquants = [n for n, (ok, _) in _trouve.items() if not ok]

if _manquants:
    print("APP NON DEMARREE : variables absentes")
    for n in _manquants:
        print(f"  - {n}")
    app.layout = html.Div("Donnees manquantes : voir la console du run.")
else:
    try:
        configurer_app(app, _trouve["df"][1], id_cols=_trouve["ID_COLS"][1],
                       target=_trouve["TARGET"][1])
    except ValueError as _e:
        app.layout = html.Div(f"TABLEAU DE BORD NON AFFICHE : {_e}")


#Cette partie par mon code

app.run(jupyter_mode="external", debug=True, port=PORT)
