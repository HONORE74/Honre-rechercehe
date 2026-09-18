# -*- coding: utf-8 -*-
# =============================================================================
#  TABLEAU DE BORD UNIFIE — version Dash (compatible Domino Web App)
#
#  Remplace la version ipywidgets. Memes regles, meme logique metier :
#    1. SEULES LES ANOMALIES (score_composite != 0) SONT AFFICHEES.
#    2. AUCUNE SOMMATION sur les montants. Valeurs brutes de df.
#
#  USAGE
#  -----
#  Option A — dans un notebook (cellule unique) :
#      from tableau_de_bord_dash import lancer
#      lancer(df, ID_COLS, TARGET)          # ouvre dans le navigateur
#
#  Option B — Domino Web App (app.py) :
#      from tableau_de_bord_dash import creer_app
#      app = creer_app(df, ID_COLS, TARGET)
#      app.run(host="0.0.0.0", port=8888)
#
#  Option C — standalone :
#      Renseigner df, ID_COLS, TARGET en bas de ce fichier, puis :
#      python tableau_de_bord_dash.py
# =============================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import json

# ┌──────────────────────────── PARAMETRES ────────────────────────────┐
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
# └─────────────────────────────────────────────────────────────────────┘

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


def _mise_en_forme(fig, hauteur, marges=None):
    fig.update_layout(
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=_GRIS),
        height=hauteur, autosize=True, separators=", ",
        hoverlabel=dict(bgcolor="white", bordercolor=_GRILLE, align="left",
                        font=dict(size=12.5, color=_ENCRE)),
        margin=marges or dict(l=10, r=40, t=95, b=45))
    return fig


# =============================================================================
#  SOCLE DE DONNEES
# =============================================================================
class Socle:

    def __init__(self, df, id_cols, target, alpha=0.10):
        self.df = df
        self.target = target
        self.alpha = alpha
        self.cles = [c for c in id_cols if c in df.columns]
        if not self.cles:
            raise ValueError(f"Aucune colonne de ID_COLS dans df : {list(id_cols)}")
        if target not in df.columns:
            raise ValueError(f"TARGET '{target}' absent de df.")
        if COL_SCORE not in df.columns:
            raise ValueError(f"Colonne '{COL_SCORE}' absente de df.")

        cp = [c for c in (COL_LO, COL_HI, COL_PRED) if c in df.columns]
        if not cp:
            raise ValueError(f"Colonnes {COL_LO}, {COL_HI}, {COL_PRED} absentes.")
        porte_cp = df[cp].notna().all(axis=1)
        if not porte_cp.any():
            raise ValueError("Aucune ligne avec prediction conforme.")

        valide = df[porte_cp]
        self.periode = None
        if {"year", "quarter"} <= set(df.columns):
            couples = sorted(set(zip(valide["year"].astype(int),
                                     valide["quarter"].astype(int))))
            self.periode = (tuple(PERIODE_VALIDEE)
                            if PERIODE_VALIDEE != "auto" else couples[-1])
            valide = valide[(valide["year"].astype(int) == self.periode[0])
                            & (valide["quarter"].astype(int) == self.periode[1])]

        score = pd.to_numeric(valide[COL_SCORE], errors="coerce").fillna(0)
        self.ano = valide[score != 0].copy()
        self.ano[COL_SCORE] = score[score != 0].values
        if not len(self.ano):
            raise ValueError(f"Aucune anomalie ({COL_SCORE} tous nuls).")

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
            return sub.assign(libelle=pd.Series(dtype=str),
                              libelle_rang=pd.Series(dtype=str))
        t = sub.copy()
        # --- FIX du bug '.str' : forcer en Series via apply, jamais .agg sur DataFrame ---
        if len(axes) == 1:
            libelle_series = t[axes[0]].astype(str)
        else:
            libelle_series = t[axes].apply(
                lambda row: " | ".join(row.astype(str)), axis=1)
        t["libelle"] = libelle_series.str.slice(0, 38)
        t["libelle_rang"] = "#" + t["rang"].astype(str) + "  " + t["libelle"]
        return t

    def unites_options(self, sub, axes, n=N_UNITES_LISTE):
        if not len(sub):
            return []
        t = self.preparer(sub.head(n), axes)
        options = []
        for _, r in t.iterrows():
            cle_json = json.dumps([str(r[c]) for c in self.cles])
            label = f"{r['libelle_rang']}   ({_fmt(r[COL_OBS])})"
            options.append({"label": label, "value": cle_json})
        return options

    def ligne(self, sub, cle_list):
        if not len(sub) or cle_list is None:
            return None
        m = np.ones(len(sub), dtype=bool)
        for c, v in zip(self.cles, cle_list):
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

    def historique(self, cle_list, n=N_TRIMESTRES):
        m = np.ones(len(self.df), dtype=bool)
        for c, v in zip(self.cles, cle_list):
            m &= (self._df_txt[c] == str(v))
        d = self.df[m]
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

    def options_valeurs(self, colonne):
        if colonne not in self.ano.columns:
            return [{"label": "— vue generale —", "value": ""}]
        g = (self.ano.groupby(colonne, observed=True)[COL_SCORE]
             .agg(["size", "sum"]).reset_index()
             .sort_values("sum", ascending=False))
        opts = [{"label": "— vue generale —", "value": ""}]
        for _, r in g.iterrows():
            opts.append({
                "label": f"{r[colonne]}   ({int(r['size'])} anomalies)",
                "value": str(r[colonne])})
        return opts


# =============================================================================
#  FIGURES — fonctions pures, pas de FigureWidget
# =============================================================================
def _fig_vide(message, hauteur=260):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=message, font=dict(size=14), x=0.015, xanchor="left"),
        height=hauteur, template="plotly_white",
        xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def fig_cercle(socle, sub, titre, axes):
    h = socle.hierarchie(sub, axes)
    if not len(h):
        return _fig_vide("Aucune anomalie a representer.", 300)
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
    fig = go.Figure(go.Sunburst(
        ids=h["id"].tolist(), labels=h["label"].tolist(),
        parents=h["parent"].tolist(), values=h["valeur_secteur"].tolist(),
        branchvalues="remainder",
        text=[f"{p:.0f} %" for p in h["part"]],
        texttemplate="%{label}<br>%{text}",
        hovertext=survol, hoverinfo="text",
        insidetextorientation="radial",
        maxdepth=len(axes),
        marker=dict(colors=h["score_moyen"].tolist(), colorscale=ECHELLE,
                    cmin=0, cmax=cmax,
                    line=dict(color="white", width=1.6),
                    colorbar=dict(title="Gravite<br>moyenne", thickness=16,
                                  len=.7, tickformat="~s"))))
    _mise_en_forme(fig, 620, dict(l=10, r=10, t=100, b=15))
    fig.update_layout(title=dict(
        text=f"Repartition des anomalies  ·  {' › '.join(axes)}"
             f"<br><sup>{titre}</sup>",
        font=dict(size=15), x=.015, xanchor="left"))
    return fig


def fig_barres(socle, sub, titre, axes, target, top_n=TOP_N_PANNEAUX):
    if not len(sub):
        return _fig_vide(f"{titre} — Aucune anomalie", 260)
    d = socle.preparer(sub.head(top_n), axes).iloc[::-1]
    survol = [
        f"<b>{r['libelle_rang']}</b><br>{target} observe : {_fmt(r[COL_OBS])}"
        f"<br>Predit : {_fmt(r[COL_PRED])}"
        f"<br>Intervalle : [{_fmt(r[COL_LO])} ; {_fmt(r[COL_HI])}]"
        f"<br>Score : {_fmt4(r[COL_SCORE])}"
        for _, r in d.iterrows()]
    montants = d[COL_OBS].tolist()
    fig = go.Figure(go.Bar(
        x=montants, y=d["libelle_rang"].tolist(), orientation="h",
        showlegend=False, text=survol, hovertemplate="%{text}<extra></extra>",
        marker=dict(color=montants, colorscale=ECHELLE,
                    cmin=min(montants), cmax=max(montants),
                    line=dict(width=.5, color="white"),
                    colorbar=dict(title="Montant", thickness=14, len=.7,
                                  tickformat="~s"))))
    fig.update_layout(xaxis=dict(tickformat="~s", title_text=f"<b>{target}</b>"),
                      yaxis=dict(tickfont=dict(size=10)))
    _mise_en_forme(fig, max(380, 38 * len(d) + 150), dict(l=10, r=40, t=95, b=50))
    fig.update_layout(title=dict(
        text=f"Les {len(d)} anomalies les plus graves"
             f"<br><sup>{titre}  ·  montants bruts, aucune sommation</sup>",
        font=dict(size=14), x=.015, xanchor="left"))
    return fig


def fig_forest(socle, sub, titre, axes, target, top_n=TOP_N_PANNEAUX):
    if not len(sub):
        return _fig_vide(f"{titre} — Aucune anomalie", 260)
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

    fig = go.Figure()
    fig.add_scatter(x=xs_band, y=ys_band, mode="lines", hoverinfo="skip",
                    opacity=.35, line=dict(color="#3a6bbf", width=10),
                    name="Intervalle conforme")
    fig.add_scatter(x=xs_over, y=ys_over, mode="lines", showlegend=False,
                    hoverinfo="skip", line=dict(color=_ACCENT, width=2, dash="dot"))
    fig.add_scatter(x=pred.tolist(), y=y, mode="markers", name="Prediction",
                    text=textes, hovertemplate="%{text}<extra></extra>",
                    marker=dict(symbol="diamond", size=10, color="white",
                                line=dict(color="black", width=1.5)))
    fig.add_scatter(x=obs.tolist(), y=y, mode="markers", name="Valeur observee",
                    text=textes, hovertemplate="%{text}<extra></extra>",
                    marker=dict(size=13, color=_ACCENT,
                                line=dict(color="#7b241c", width=1.3)))
    fig.update_xaxes(tickformat="~s", title_text=f"<b>{target}</b>")
    fig.update_yaxes(tickmode="array", tickvals=y,
                     ticktext=d["libelle_rang"].str.slice(0, 40).tolist(),
                     tickfont=dict(size=10))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="center", x=.5))
    _mise_en_forme(fig, max(420, 44 * len(d) + 175), dict(l=10, r=50, t=115, b=55))
    fig.update_layout(title=dict(
        text="Intervalle conforme, prediction et valeur observee"
             f"<br><sup>{titre}  ·  valeurs brutes, aucune sommation</sup>",
        font=dict(size=14), x=.015, xanchor="left"))
    return fig


def fig_evolution(socle, cle_list, target, alpha):
    hist, per = socle.historique(cle_list)
    if hist is None or not len(hist):
        return _fig_vide("Aucun historique pour ce sous-portefeuille.")

    sub_line = socle.ano
    r = socle.ligne(sub_line, cle_list)
    ctx = socle.contexte(r)
    val = hist[target].to_numpy(dtype="float64")
    p_valide = ctx["per"] if ctx and ctx.get("per") in per else per[-1]

    fig = go.Figure()
    fig.add_scatter(x=per, y=val.tolist(), mode="lines",
                    line=dict(color=_VIDE, width=0),
                    fill="tozeroy", fillcolor="rgba(140,147,165,0.10)",
                    showlegend=False, hoverinfo="skip")
    if ctx:
        fig.add_scatter(x=[p_valide, p_valide], y=[ctx["lo"], ctx["hi"]],
                        mode="lines", line=dict(color=_BLEU, width=15),
                        opacity=.26,
                        name=f"Intervalle conforme {100*(1-alpha):.0f} %")
        fig.add_scatter(x=[p_valide], y=[ctx["pred"]], mode="markers",
                        name="Prediction",
                        marker=dict(symbol="diamond", size=11, color="white",
                                    line=dict(color=_ENCRE, width=1.8)))
    fig.add_scatter(x=per, y=val.tolist(), mode="lines+markers+text",
                    name=target,
                    line=dict(color=_ENCRE, width=2.8, shape="spline",
                              smoothing=.55),
                    marker=dict(size=9, color="white",
                                line=dict(color=_ENCRE, width=2.2)),
                    text=[_fmt(v) for v in val], textposition="top center",
                    textfont=dict(size=9.5, color=_GRIS),
                    hovertemplate=("<b>%{x}</b><br>" + target
                                  + " : <b>%{y:,.0f}</b><extra></extra>"))
    if ctx:
        couvert = ctx["couvert"]
        coul = _OK if couvert else _ACCENT
        halo = "rgba(61,90,158,0.18)" if couvert else "rgba(192,57,43,0.20)"
        fig.add_scatter(x=[p_valide], y=[ctx["obs"]], mode="markers",
                        showlegend=False, hoverinfo="skip",
                        marker=dict(size=34, color=halo))
        fig.add_scatter(x=[p_valide], y=[ctx["obs"]], mode="markers",
                        name="Couvert" if couvert else "Hors intervalle",
                        marker=dict(size=13, color=coul,
                                    line=dict(color="white", width=2.4)))

    alerte = ""
    if len(hist) < 2:
        alerte = (f"  ·  <b style='color:{_ACCENT}'>un seul trimestre "
                  "dans df : pas de courbe possible</b>")
    rang = ctx["rang"] if ctx else "?"
    fig.update_xaxes(showgrid=False, linecolor=_GRILLE, showspikes=True,
                     spikemode="across", spikethickness=1.2, spikedash="dot",
                     spikecolor=_GRIS, title_text="<b>Trimestre</b>")
    fig.update_yaxes(gridcolor=_GRILLE, zeroline=False, tickformat="~s",
                     title_text=f"<b>{target}</b>")
    _mise_en_forme(fig, 420, dict(l=85, r=45, t=95, b=45))
    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", y=1.06, x=1, xanchor="right",
                    font=dict(size=11)),
        title=dict(
            text=f"<b style='color:{_ENCRE}'>#{rang}"
                 f"  {' | '.join(cle_list)}</b>"
                 f"<br><span style='font-size:11px'>{target} · "
                 f"{len(hist)} trimestre(s) · {per[0]} → {per[-1]}"
                 + (f" · periode validee <b>{p_valide}</b>" if ctx else "")
                 + alerte + "</span>",
            font=dict(size=14), x=.015, xanchor="left"))
    return fig


def fig_variables(socle, cle_list, target):
    hist, per = socle.historique(cle_list)
    if hist is None or len(hist) < 3:
        n = 0 if hist is None else len(hist)
        return _fig_vide(
            f"Classement des variables indisponible : {n} trimestre(s), "
            "il en faut au moins 3.")
    t = socle.variables_explicatives(hist)
    if not len(t):
        return _fig_vide("Aucune variable explicative numerique exploitable.")

    r = socle.ligne(socle.ano, cle_list)
    ctx = socle.contexte(r)
    p_valide = ctx["per"] if ctx and ctx.get("per") in per else per[-1]
    k = per.index(p_valide) if p_valide in per else len(per) - 1
    n_vars = min(len(t), N_VARS_EXPLIC)

    fig = make_subplots(rows=n_vars, cols=1, shared_xaxes=True,
                        vertical_spacing=.055,
                        subplot_titles=[f"<b>{t.iloc[i]['variable']}</b>"
                                        for i in range(n_vars)])
    for i in range(n_vars):
        c = _DOUX[i % len(_DOUX)]
        v = t.iloc[i]["variable"]
        vals = hist[v].to_numpy(dtype="float64")
        fig.add_scatter(x=per, y=vals.tolist(), mode="lines+markers",
                        showlegend=False,
                        line=dict(color=c, width=2.3, shape="spline",
                                  smoothing=.55),
                        marker=dict(size=6, color="white",
                                    line=dict(color=c, width=1.8)),
                        hovertemplate=(f"<b>%{{x}}</b><br>{v} : "
                                       "<b>%{y:,.0f}</b><extra></extra>"),
                        row=i + 1, col=1)
        fig.add_scatter(x=[per[k]], y=[float(vals[k])], mode="markers",
                        showlegend=False, hoverinfo="skip",
                        marker=dict(size=13, color=c,
                                    line=dict(color="white", width=2.2)),
                        row=i + 1, col=1)
    fig.update_xaxes(showgrid=False, showticklabels=False, linecolor=_GRILLE)
    fig.update_xaxes(showticklabels=True, title_text="<b>Trimestre</b>",
                     row=n_vars, col=1)
    fig.update_yaxes(gridcolor=_GRILLE, zeroline=False, tickformat="~s",
                     tickfont=dict(size=9))
    _mise_en_forme(fig, 150 * n_vars + 120, dict(l=85, r=45, t=95, b=45))
    fig.update_layout(
        hovermode="x unified",
        title=dict(text="<b>Les variables qui expliquent ce comportement</b>",
                   font=dict(size=14), x=.015, xanchor="left"))
    for a in fig.layout.annotations:
        a.update(x=0, xanchor="left", font=dict(size=12, color=_GRIS))
    return fig


# =============================================================================
#  CARTES KPI — HTML pur
# =============================================================================
def composant_cartes(socle, sub, titre):
    part = sub[COL_SCORE].sum() / socle.score_global if len(sub) else 0
    hors = 0
    if len(sub) and COL_OBS in sub.columns and COL_LO in sub.columns:
        hors = int((~((sub[COL_OBS] >= sub[COL_LO])
                      & (sub[COL_OBS] <= sub[COL_HI]))).sum())
    cartes = [("Anomalies", f"{len(sub):,}".replace(",", " "), "#37474f"),
              ("Part du score global", f"{100 * part:.1f} %", "#ad1457"),
              ("Hors intervalle", f"{hors:,}".replace(",", " "), "#00838f"),
              ("Gravite moyenne",
               _fmt4(sub[COL_SCORE].mean()) if len(sub) else "—", "#5e35b1"),
              ("Pire anomalie",
               _fmt4(sub[COL_SCORE].max()) if len(sub) else "—", "#6a1b9a")]
    blocs = [
        html.Div(style={
            "flex": "1", "minWidth": "130px", "background": "#fff",
            "border": "1px solid #e0e0e0", "borderLeft": f"5px solid {c}",
            "borderRadius": "7px", "padding": "11px 13px",
            "boxShadow": "0 1px 3px rgba(0,0,0,.07)"
        }, children=[
            html.Div(t, style={
                "fontSize": "10.5px", "color": "#78909c",
                "textTransform": "uppercase", "letterSpacing": ".6px"}),
            html.Div(v, style={
                "fontSize": "19px", "fontWeight": "600", "color": c,
                "marginTop": "4px"})
        ]) for t, v, c in cartes]
    return html.Div(style={
        "fontFamily": "system-ui, sans-serif", "margin": "6px 0 14px 0"
    }, children=[
        html.Div(titre, style={
            "fontSize": "15px", "fontWeight": "600", "color": "#263238",
            "marginBottom": "10px"}),
        html.Div(style={"display": "flex", "gap": "9px", "flexWrap": "wrap"},
                 children=blocs)
    ])


# =============================================================================
#  TABLEAU — donnees pour dash_table
# =============================================================================
def donnees_tableau(socle, sub, axes):
    if not len(sub):
        return [], []
    d = socle.preparer(sub.head(N_LIGNES_TABLE), axes)
    rows = []
    for _, r in d.iterrows():
        rows.append({
            "Rang": int(r["rang"]),
            "Maille": r["libelle"],
            "Y_obs": _fmt(r[COL_OBS]),
            "Y_pred": _fmt(r[COL_PRED]),
            "CP_bas": _fmt(r[COL_LO]),
            "CP_haut": _fmt(r[COL_HI]),
            "Couvert": "Oui" if r[COL_OBS] >= r[COL_LO]
                                and r[COL_OBS] <= r[COL_HI] else "Non",
            "Score": _fmt4(r[COL_SCORE])})
    cols = [{"name": c, "id": c} for c in
            ["Rang", "Maille", "Y_obs", "Y_pred", "CP_bas", "CP_haut",
             "Couvert", "Score"]]
    return rows, cols


# =============================================================================
#  BANDEAU HTML
# =============================================================================
def _bandeau(txt, fond="#eceff1", coul="#37474f"):
    return html.Div(txt, style={
        "fontFamily": "system-ui, sans-serif", "fontSize": "12.5px",
        "color": coul, "background": fond, "padding": "9px 13px",
        "borderRadius": "6px", "margin": "14px 0 8px 0"})


# =============================================================================
#  APP DASH — LAYOUT + CALLBACKS
# =============================================================================
def creer_app(df, id_cols, target, alpha=0.10):
    socle = Socle(df, id_cols, target, alpha)
    cles = socle.cles
    defaut_maille = next((c for c in ("Lob", "Partner", "Companies", "Risk")
                          if c in cles), cles[0])
    defaut_axes = cles[:2]

    app = Dash(__name__)
    app.title = "Tableau de bord — Anomalies conformes"

    app.layout = html.Div(style={
        "fontFamily": "Inter, system-ui, sans-serif",
        "maxWidth": "1400px", "margin": "0 auto", "padding": "20px"
    }, children=[

        html.H1("Tableau de bord — Anomalies conformes",
                style={"fontSize": "22px", "color": _ENCRE, "marginBottom": "4px"}),
        html.P(f"Source : {len(df):,} lignes  ·  "
               f"{len(socle.ano):,} anomalies  ·  "
               f"Periode : {socle.periode[0]}-T{socle.periode[1]}"
               if socle.periode else f"Source : {len(df):,} lignes",
               style={"fontSize": "13px", "color": _GRIS, "marginBottom": "20px"}),

        # ---------- SECTION AXES ----------
        _bandeau(
            "Axes — ils structurent le cercle et composent les libelles. "
            "Ils ne modifient aucun montant.",
            fond="#e3f2fd", coul="#0d47a1"),

        dcc.Checklist(
            id="axes-checklist",
            options=[{"label": f"  {c}", "value": c} for c in cles],
            value=defaut_axes,
            inline=True,
            style={"marginBottom": "12px", "fontSize": "14px"},
            inputStyle={"marginRight": "5px"},
            labelStyle={"marginRight": "18px", "cursor": "pointer"}),

        dcc.Graph(id="graph-cercle", config={"displayModeBar": False}),

        # ---------- SECTION PERIMETRE ----------
        _bandeau(
            "Perimetre — la maille et la valeur filtrent l'ensemble du "
            f"tableau de bord. Seules les anomalies ({COL_SCORE} != 0) "
            "sont affichees.",
            fond="#e8f5e9", coul="#1b5e20"),

        html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                         "marginBottom": "12px"}, children=[
            html.Div([
                html.Label("Maille :", style={"fontWeight": "600",
                                               "fontSize": "13px",
                                               "marginBottom": "4px"}),
                dcc.Dropdown(
                    id="sel-maille",
                    options=[{"label": c, "value": c} for c in cles],
                    value=defaut_maille, clearable=False,
                    style={"width": "260px"})
            ]),
            html.Div([
                html.Label("Valeur :", style={"fontWeight": "600",
                                               "fontSize": "13px",
                                               "marginBottom": "4px"}),
                dcc.Dropdown(
                    id="sel-valeur",
                    options=socle.options_valeurs(defaut_maille),
                    value="", clearable=False,
                    style={"width": "420px"})
            ]),
        ]),

        html.Div(id="zone-cartes"),

        dcc.Graph(id="graph-barres", config={"displayModeBar": False}),
        dcc.Graph(id="graph-forest", config={"displayModeBar": False}),

        # ---------- SECTION DETAIL ----------
        _bandeau(
            "L'anomalie en detail — evolution de la cible sur son historique, "
            "puis les variables qui expliquent son comportement.",
            fond="#fff3e0", coul="#e65100"),

        html.Div([
            html.Label("Anomalie :", style={"fontWeight": "600",
                                             "fontSize": "13px",
                                             "marginBottom": "4px"}),
            dcc.Dropdown(id="sel-unite", options=[], value=None,
                         clearable=False,
                         style={"width": "100%", "maxWidth": "720px"})
        ], style={"marginBottom": "12px"}),

        dcc.Graph(id="graph-evolution", config={"displayModeBar": False}),
        dcc.Graph(id="graph-variables", config={"displayModeBar": False}),

        # ---------- TABLEAU ----------
        _bandeau(f"Tableau de priorisation — {N_LIGNES_TABLE} anomalies "
                 "les plus graves du perimetre courant."),

        dash_table.DataTable(
            id="table-priorisation",
            columns=[], data=[],
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#f5f5f5", "fontWeight": "600",
                           "fontSize": "12px", "color": _ENCRE},
            style_cell={"textAlign": "right", "fontSize": "12px",
                         "padding": "6px 10px", "fontFamily": "monospace"},
            style_cell_conditional=[
                {"if": {"column_id": "Maille"}, "textAlign": "left",
                 "maxWidth": "220px", "overflow": "hidden",
                 "textOverflow": "ellipsis"},
                {"if": {"column_id": "Couvert"}, "textAlign": "center"}],
            style_data_conditional=[
                {"if": {"filter_query": '{Couvert} = "Non"'},
                 "backgroundColor": "rgba(192,57,43,0.08)",
                 "color": _ACCENT}],
            page_size=N_LIGNES_TABLE),

    ])

    # ================================================================
    #  CALLBACK 1 : maille change → options de valeur recalculees
    # ================================================================
    @app.callback(
        Output("sel-valeur", "options"),
        Output("sel-valeur", "value"),
        Input("sel-maille", "value"))
    def maj_options_valeur(maille):
        opts = socle.options_valeurs(maille)
        return opts, ""

    # ================================================================
    #  CALLBACK 2 : CENTRAL — tout se met a jour ensemble
    #
    #  Inputs : axes, maille, valeur (les 3 selecteurs du perimetre)
    #  Outputs : TOUS les graphiques + cartes + tableau + options unite
    #
    #  C'est l'equivalent exact de _maj_panneaux() dans la version
    #  ipywidgets. Un seul callback, donc tout reste coherent.
    # ================================================================
    @app.callback(
        Output("graph-cercle", "figure"),
        Output("zone-cartes", "children"),
        Output("graph-barres", "figure"),
        Output("graph-forest", "figure"),
        Output("sel-unite", "options"),
        Output("sel-unite", "value"),
        Output("table-priorisation", "data"),
        Output("table-priorisation", "columns"),
        Input("axes-checklist", "value"),
        Input("sel-maille", "value"),
        Input("sel-valeur", "value"))
    def maj_panneaux(axes_actifs, maille, valeur):
        axes = axes_actifs if axes_actifs else [cles[0]]
        sub, titre = socle.filtrer(maille, valeur)

        f_cercle = fig_cercle(socle, sub, titre, axes)
        cartes = composant_cartes(socle, sub, titre)
        f_barres = fig_barres(socle, sub, titre, axes, target)
        f_forest = fig_forest(socle, sub, titre, axes, target)

        opts_unite = socle.unites_options(sub, axes)
        val_unite = opts_unite[0]["value"] if opts_unite else None

        rows, cols = donnees_tableau(socle, sub, axes)

        return (f_cercle, cartes, f_barres, f_forest,
                opts_unite, val_unite, rows, cols)

    # ================================================================
    #  CALLBACK 3 : selection d'une anomalie → evolution + variables
    # ================================================================
    @app.callback(
        Output("graph-evolution", "figure"),
        Output("graph-variables", "figure"),
        Input("sel-unite", "value"))
    def maj_detail(cle_json):
        if not cle_json:
            return (_fig_vide("Selectionnez une anomalie ci-dessus."),
                    _fig_vide("Selectionnez une anomalie ci-dessus."))
        cle_list = json.loads(cle_json)
        f_evol = fig_evolution(socle, cle_list, target, alpha)
        f_vars = fig_variables(socle, cle_list, target)
        return f_evol, f_vars

    return app


# =============================================================================
#  RACCOURCI NOTEBOOK
# =============================================================================
def lancer(df, id_cols, target, alpha=0.10, port=8050):
    app = creer_app(df, id_cols, target, alpha)
    app.run(debug=False, port=port)


# =============================================================================
#  EXECUTION STANDALONE (Domino Web App ou local)
#
#  Pour Domino Web App : ce fichier EST l'app.py.
#  Il charge les donnees depuis des fichiers CSV/Parquet poses dans le meme
#  dossier, puis lance le serveur Dash sur le port 8888.
#
#  Fichiers attendus (a deposer a cote de ce script) :
#    - donnees_dashboard.csv   ou   donnees_dashboard.parquet
#
#  Le CSV/Parquet doit contenir AU MINIMUM les colonnes :
#    - les colonnes d'identifiants (Partner, Companies, Lob, Activity, etc.)
#    - la cible (Claims_incurred ou autre)
#    - year, quarter
#    - y_pred, borne_basse, borne_haute, score_composite
#
#  Variables a ajuster ci-dessous si les noms de colonnes different.
# =============================================================================
if __name__ == "__main__":
    import sys
    import os

    # ─── CONFIGURATION (a ajuster selon vos donnees) ───
    ID_COLS = ["Partner", "Companies", "Lob", "Activity", "Periodicity", "Risk"]
    TARGET  = "Claims_incurred"
    ALPHA   = 0.10

    # ─── CHARGEMENT DES DONNEES ───
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_path = os.path.join(script_dir, "donnees_dashboard.parquet")
    csv_path = os.path.join(script_dir, "donnees_dashboard.csv")

    if os.path.exists(parquet_path):
        print(f"Chargement de {parquet_path} ...")
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        print(f"Chargement de {csv_path} ...")
        df = pd.read_csv(csv_path)
    else:
        print("=" * 74)
        print("TABLEAU DE BORD NON LANCE")
        print("=" * 74)
        print()
        print("Aucun fichier de donnees trouve. Deposez l'un de ces fichiers")
        print(f"dans le dossier {script_dir} :")
        print("  - donnees_dashboard.parquet   (recommande)")
        print("  - donnees_dashboard.csv")
        print()
        print("Ou bien, depuis un notebook :")
        print("  from tableau_de_bord_dash import creer_app")
        print("  app = creer_app(df, ID_COLS, TARGET)")
        print('  app.run(host="0.0.0.0", port=8888)')
        sys.exit(1)

    print(f"  {len(df):,} lignes chargees.".replace(",", " "))

    try:
        app = creer_app(df, ID_COLS, TARGET, ALPHA)
    except ValueError as e:
        print("=" * 74)
        print("TABLEAU DE BORD NON LANCE")
        print("=" * 74)
        print(f"  {e}")
        sys.exit(1)

    print("=" * 74)
    print("Tableau de bord pret — ouvrez http://localhost:8888")
    print("=" * 74)
    app.run(host="0.0.0.0", port=8888, debug=False)
