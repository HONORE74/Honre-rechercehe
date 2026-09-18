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
#  Mon code (migration du tableau de bord ipywidgets -> Dash, base sur
#  l'architecture a 3 bases : anomalies_prio / expl / df_model)
# =============================================================================
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, no_update, ctx, dash_table

TOP_N_PANNEAUX = 12
N_TRIMESTRES   = 10
N_UNITES_LISTE = 30
N_LIGNES_TABLE = 15
N_VARS_EXPLIC  = 5
COL_SCORE      = "score_composite"
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


def _encode_cle(axes, valeurs):
    return json.dumps([list(axes), list(valeurs)])


def _decode_cle(s):
    if not s:
        return None
    axes, valeurs = json.loads(s)
    return tuple(axes), tuple(valeurs)


class _Socle:
    """Trois bases : anomalies_prio (dd, scores), expl (ex, vraies valeurs
    ligne a ligne), df_model (df, historique complet). table_axes() agrege
    les vraies valeurs de ex par les axes actifs et y greffe les scores de dd."""

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
        self._df_txt = {c: df[c].astype(str).values for c in self.cles}
        self.vars_explic = [
            c for c in df.columns
            if c != target and c not in _EXCLURE_VARS
            and pd.api.types.is_numeric_dtype(df[c])]

    def filtrer(self, colonne, valeur):
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
            # RECONSTRUIT (capture coupee juste apres le bloc ci-dessus) :
            # fusion des deux agregats sur les axes, puis defauts a 0 pour les
            # groupes sans anomalie.
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
        # RECONSTRUIT (methode non couverte par les captures) : coherent avec
        # les colonnes de table_axes() et avec l'usage qui en est fait plus
        # bas (sel_unite.value = un tuple de valeurs, une par axe actif).
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
        text=f"Repartition des anomalies · {' > '.join(axes)}"
             f"<br><sup>{titre}</sup>",
        font=dict(size=15), x=.015, xanchor="left")
    return fig


def _cartes(socle, sub, titre, table):
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
    return (
        f"<div style='font-family:system-ui,sans-serif;margin:6px 0 14px 0'>"
        f"<div style='font-size:15px;font-weight:600;color:#263238;"
        f"margin-bottom:10px'>{titre}</div>"
        f"<div style='display:flex;gap:9px;flex-wrap:wrap'>{blocs}</div></div>")


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
        text=f"Les {len(g)} plus critiques · agrege sur {' + '.join(axes)}"
             f"<br><sup>{titre}</sup>",
        font=dict(size=14), x=.015, xanchor="left")
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
                            ticktext=ticks, tickfont=dict(size=10))
    fig.layout.height = max(420, 44 * len(d) + 175)
    fig.layout.title = dict(
        text="Intervalle conforme, prediction et valeur observee"
             f"<br><sup>{titre} · agrege sur {' + '.join(axes)}, "
             "memes montants que la courbe d'evolution</sup>",
        font=dict(size=14), x=.015, xanchor="left")
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
            "Y_obs": [_fmt(v) for v in d["y_obs"]],
            "Y_pred": [_fmt(v) for v in d["y_pred"]],
            "CP_bas": [_fmt(v) for v in d["lo"]],
            "CP_haut": [_fmt(v) for v in d["hi"]],
            "Couvert": np.where(d["couvert"], "Oui", "Non"),
            "Score": [_fmt4(v) for v in d["score"]],
        })
        scores = d["score"].to_numpy(dtype="float64")
        smin, smax = float(scores.min()), float(scores.max())
        etendue = (smax - smin) or 1.0
        degrade = [{"if": {"row_index": i},
                    "backgroundColor": f"rgba(198,40,40,{0.12 + 0.55 * (scores[i]-smin)/etendue})"}
                   for i in range(n)]
        return html.Div([
            html.Div(f"Tableau de priorisation — {titre}",
                     style={"fontWeight": "600", "fontSize": "14px",
                            "margin": "4px 0 10px 0", "color": "#263238"}),
            dash_table.DataTable(
                data=t.to_dict("records"),
                columns=[{"name": c, "id": c} for c in t.columns],
                style_as_list_view=True,
                style_table={"overflowX": "auto"},
                style_cell={"fontFamily": "system-ui, sans-serif", "fontSize": "12.5px",
                            "padding": "7px 12px", "border": "none",
                            "borderBottom": "1px solid #eceff1"},
                style_cell_conditional=[{"if": {"column_id": "Maille"}, "textAlign": "left"}],
                style_header={"backgroundColor": "#f5f7fa", "fontWeight": "600",
                              "border": "none", "borderBottom": "2px solid #cfd8dc"},
                style_data_conditional=degrade,
            ),
        ])
    except Exception as e:
        return html.P(f"Tableau non genere : {type(e).__name__} : {str(e)[:150]}")


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
            "<b>Axes d'agregation</b> — ils controlent le regroupement des "
            "vraies valeurs (montants sommes a ce niveau) et le clic sur une "
            "part du cercle filtre le perimetre.",
            fond="#e3f2fd", coul="#0d47a1"), dangerously_allow_html=True),
        html.Div(axes_checklist, style={"marginBottom": "8px"}),
        fig_cercle,

        dcc.Markdown(_bandeau(
            "<b>Perimetre</b> — la maille et la valeur filtrent l'ensemble du "
            "tableau de bord.",
            fond="#e8f5e9", coul="#1b5e20"), dangerously_allow_html=True),
        html.Div([html.Div(sel_maille, style={"display": "inline-block",
                                              "marginRight": "12px"}),
                 html.Div(sel_valeur, style={"display": "inline-block"})]),
        cartes_div,
        fig_barres,
        fig_forest,

        dcc.Markdown(_bandeau(
            "<b>Le sous-portefeuille en detail</b> — evolution de la cible sur "
            "son historique complet (df_model), intervalle conforme, "
            "prediction, puis variables explicatives.",
            fond="#fff3e0", coul="#e65100"), dangerously_allow_html=True),
        sel_unite,
        fig_evol,
        fig_vars,

        dcc.Markdown(_bandeau("<b>Tableau de priorisation</b>")),
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
        sub, sub_ex, titre = socle.filtrer(maille, valeur)
        table = socle.table_axes(sub, sub_ex, axes)
        etat["table"] = table

        try:
            fc = _fig_cercle(socle, sub, titre, axes)
        except Exception as e:
            print(f"Cercle non mis a jour : {type(e).__name__} : {str(e)[:120]}")
            fc = no_update

        cartes = dcc.Markdown(_cartes(socle, sub, titre, table),
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


# =============================================================================
#  CHARGEMENT DES DONNEES
#  Ce fichier tourne comme une app Domino independante : il n'a plus acces
#  aux variables de ta session Jupyter. _session() reste par securite (si tu
#  executes ce fichier via %run -i depuis un notebook), mais pour un vrai
#  lancement d'app, REMPLACE ce bloc par ton propre chargement de df_model,
#  anomalies_prio, expl, ID_COLS, TARGET.
# =============================================================================
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


#Cette partie par mon code

app.run(jupyter_mode="external", debug=True, port=PORT)
