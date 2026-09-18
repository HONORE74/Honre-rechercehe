# -*- coding: utf-8 -*-
# =============================================================================
# TABLEAU DE BORD UNIFIE - anomalies, evolution, priorisation
# VERSION DASH (remplace ipywidgets — compatible Domino WebApp)
#
# Memes regles que l'original :
# 1. SEULES LES ANOMALIES SONT AFFICHEES (score_composite != 0)
# 2. AUCUNE SOMMATION — valeurs brutes de la ligne
# =============================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from dash import Dash, dcc, html, Input, Output, State, callback_context, dash_table, no_update
import dash_bootstrap_components as dbc

# ┌──────────────────────────── PARAMETRES ────────────────────────────┐
TOP_N_PANNEAUX  = 12
N_TRIMESTRES    = 10
N_UNITES_LISTE  = 30
N_LIGNES_TABLE  = 15
N_VARS_EXPLIC   = 5
COL_SCORE       = "score_composite"
COL_OBS         = "y_obs"
COL_PRED        = "y_pred"
COL_LO, COL_HI = "borne_basse", "borne_haute"
PERIODE_VALIDEE = "auto"
ECHELLE         = "Bluered"
VUE_GENERALE    = ""
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
# SOCLE — identique a l'original, zero modification
# =============================================================================
class _Socle:

    def __init__(self, df, id_cols, target, alpha, verbeux=True):
        self.df, self.target, self.alpha = df, target, alpha
        self.cles = [c for c in id_cols if c in df.columns]
        if not self.cles:
            raise ValueError(f"Aucune colonne de ID_COLS n'est dans df : {list(id_cols)}")
        if target not in df.columns:
            raise ValueError(f"TARGET '{target}' absent de df.")
        if COL_SCORE not in df.columns:
            raise ValueError(f"Colonne de score '{COL_SCORE}' absente de df.")

        cp = [c for c in (COL_LO, COL_HI, COL_PRED) if c in df.columns]
        if not cp:
            raise ValueError(f"df ne contient aucune des colonnes {COL_LO}, {COL_HI}, {COL_PRED}.")
        porte_cp = df[cp].notna().all(axis=1)
        if not porte_cp.any():
            raise ValueError(f"Aucune ligne de df ne porte de prediction conforme ({cp} tous renseignes).")

        valide = df[porte_cp]
        self.periode = None
        if {"year", "quarter"} <= set(df.columns):
            couples = sorted(set(zip(valide["year"].astype(int), valide["quarter"].astype(int))))
            self.periode = (tuple(PERIODE_VALIDEE) if PERIODE_VALIDEE != "auto" else couples[-1])
            if len(couples) > 1 and verbeux:
                print(f"Note : {len(couples)} trimestres portent une prediction. "
                      f"Le plus recent est retenu ({self.periode[0]}-T{self.periode[1]}).")
            valide = valide[(valide["year"].astype(int) == self.periode[0])
                            & (valide["quarter"].astype(int) == self.periode[1])]

        score = pd.to_numeric(valide[COL_SCORE], errors="coerce").fillna(0)
        self.ano = valide[score != 0].copy()
        self.ano[COL_SCORE] = score[score != 0].values
        if not len(self.ano):
            raise ValueError(f"Aucune anomalie : toutes les lignes ont un {COL_SCORE} nul.")

        if COL_OBS not in self.ano.columns:
            self.ano[COL_OBS] = self.ano[target].values

        for c in self.cles:
            self.ano[c] = self.ano[c].astype(str)
        self.ano = self.ano.sort_values(COL_SCORE, ascending=False).reset_index(drop=True)
        self.ano["rang"] = np.arange(1, len(self.ano) + 1)
        self.score_global = float(self.ano[COL_SCORE].sum()) or 1.0
        self._df_txt = {c: df[c].astype(str).values for c in self.cles}
        self.vars_explic = [c for c in df.columns
                            if c != target and c not in _EXCLURE_VARS
                            and pd.api.types.is_numeric_dtype(df[c])]
        self.trimestres = []
        if {"year", "quarter"} <= set(df.columns):
            self.trimestres = sorted(set(zip(df["year"].astype(int), df["quarter"].astype(int))))

        if verbeux:
            per_txt = f"{self.periode[0]}-T{self.periode[1]}" if self.periode else "non datee"
            print("=" * 74)
            print(f"SOURCE : df   ·   {len(df):,} lignes".replace(",", " "))
            print("=" * 74)
            if self.trimestres:
                t0, t1 = self.trimestres[0], self.trimestres[-1]
                print(f"   trimestres : {t0[0]}-T{t0[1]} -> {t1[0]}-T{t1[1]}   ({len(self.trimestres)} au total)")
            print(f"   periode validee    : {per_txt}   ({int(porte_cp.sum()):,} lignes)".replace(",", " "))
            print(f"   anomalies retenues : {len(self.ano):,} ({COL_SCORE} != 0)".replace(",", " "))
            print(f"   axes disponibles   : {', '.join(self.cles)}")
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
        # Correction bug .str : traitement separé selon len(axes)
        if len(axes) == 1:
            t["libelle"] = t[axes[0]].astype(str).str.slice(0, 38)
        else:
            t["libelle"] = t[axes].apply(
                lambda row: " | ".join(row.astype(str)), axis=1
            ).str.slice(0, 38)
        t["libelle_rang"] = "#" + t["rang"].astype(str) + "  " + t["libelle"]
        return t

    def unites(self, sub, axes, n=N_UNITES_LISTE):
        if not len(sub):
            return []
        t = self.preparer(sub.head(n), axes)
        return [{"label": f"{r['libelle_rang']}   ({_fmt(r[COL_OBS])})",
                 "value": SEP.join(str(r[c]) for c in self.cles)}
                for _, r in t.iterrows()]

    def ligne(self, sub, cle_str):
        if not len(sub) or cle_str is None:
            return None
        cle = cle_str.split(SEP)
        m = np.ones(len(sub), dtype=bool)
        for c, v in zip(self.cles, cle):
            m &= (sub[c].astype(str).values == str(v))
        t = sub[m]
        return None if not len(t) else t.iloc[0]

    def contexte(self, r):
        if r is None:
            return None
        per = f"{self.periode[0]}-T{self.periode[1]}" if self.periode else None
        obs, lo, hi = float(r[COL_OBS]), float(r[COL_LO]), float(r[COL_HI])
        return dict(per=per, pred=float(r[COL_PRED]), lo=lo, hi=hi, obs=obs,
                    couvert=bool(lo <= obs <= hi), score=float(r[COL_SCORE]),
                    rang=int(r["rang"]))

    def _masque(self, textes, cles, valeurs):
        m = np.ones(len(next(iter(textes.values()))), dtype=bool)
        for a, v in zip(cles, valeurs):
            m &= (textes[a] == str(v))
        return m

    def historique(self, cle_str, n=N_TRIMESTRES):
        cle = cle_str.split(SEP)
        d = self.df[self._masque(self._df_txt, self.cles, cle)]
        if not len(d):
            return None, []
        colonnes = [self.target] + [v for v in self.vars_explic if v in d.columns]
        if {"year", "quarter"} <= set(d.columns):
            g = (d.groupby(["year", "quarter"], observed=True)[colonnes]
                 .sum().reset_index()
                 .sort_values(["year", "quarter"]).tail(n))
            per = (g["year"].astype(int).astype(str) + "-T"
                   + g["quarter"].astype(int).astype(str)).tolist()
            return g, per
        return d[colonnes].tail(n).reset_index(drop=True), [str(i) for i in range(min(n, len(d)))]

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
        return t.reindex(t["z"].abs().sort_values(ascending=False).index).head(n).reset_index(drop=True)

    def hierarchie(self, sub, chemin):
        if not len(sub) or not chemin:
            return pd.DataFrame()
        prof_max = len(chemin)
        agg_spec = {"score_total": (COL_SCORE, "sum"),
                    "score_moyen": (COL_SCORE, "mean"),
                    "score_max":   (COL_SCORE, "max"),
                    "n":           (COL_SCORE, "size")}
        total = float(sub[COL_SCORE].sum()) or 1.0
        lignes = []
        for prof in range(1, prof_max + 1):
            cols = chemin[:prof]
            g = sub.groupby(cols, observed=True).agg(**agg_spec).reset_index()
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
                    "score_max":   float(r["score_max"]),
                    "n": int(r["n"]),
                    "part": 100 * st / total})
        return pd.DataFrame(lignes)

    def options_valeurs(self, colonne):
        g = (self.ano.groupby(colonne, observed=True)[COL_SCORE]
             .agg(["size", "sum"]).reset_index()
             .sort_values("sum", ascending=False))
        return [{"label": "— vue generale —", "value": VUE_GENERALE}] + [
            {"label": f"{r[colonne]}   ({int(r['size'])} anomalies)",
             "value": str(r[colonne])}
            for _, r in g.iterrows()]


# =============================================================================
# FIGURES — meme logique que l'original, retournent go.Figure
# =============================================================================

def _fig_vide(message, hauteur=260):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=message, font=dict(size=14), x=0.015, xanchor="left"),
        height=hauteur, template="plotly_white",
        paper_bgcolor="white", plot_bgcolor="white")
    return fig


def _fig_cercle(socle, sub, titre, axes):
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
        f"Score cumule : {_fmt4(r['score_total'])} ({r['part']:.1f} %)<br>"
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
        marker=dict(
            colors=h["score_moyen"].tolist(), colorscale=ECHELLE,
            cmin=0, cmax=cmax, line=dict(color="white", width=1.6),
            colorbar=dict(title="Gravite<br>moyenne", thickness=16, len=.7, tickformat="~s")),
    ))
    _mise_en_forme(fig, 620, dict(l=10, r=10, t=100, b=15))
    fig.update_layout(title=dict(
        text=f"Repartition des anomalies  ·  {' › '.join(axes)}<br><sup>{titre}</sup>",
        font=dict(size=15), x=.015, xanchor="left"))
    return fig


def _fig_barres(socle, sub, titre, target, axes, top_n=TOP_N_PANNEAUX):
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
        showlegend=False,
        marker=dict(color=montants, colorscale=ECHELLE,
                    cmin=min(montants), cmax=max(montants),
                    line=dict(width=.5, color="white"),
                    colorbar=dict(title="Montant", thickness=14, len=.7, tickformat="~s")),
        text=survol, hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(tickformat="~s", title_text=f"<b>{target}</b>"),
        yaxis=dict(tickfont=dict(size=10)),
        height=max(380, 38 * len(d) + 150),
        title=dict(
            text=f"Les {len(d)} anomalies les plus graves<br>"
                 f"<sup>{titre}  ·  montants bruts, aucune sommation</sup>",
            font=dict(size=14), x=.015, xanchor="left"),
    )
    _mise_en_forme(fig, max(380, 38 * len(d) + 150), dict(l=10, r=40, t=95, b=50))
    return fig


def _fig_forest(socle, sub, titre, target, axes, top_n=TOP_N_PANNEAUX):
    if not len(sub):
        return _fig_vide(f"{titre} — Aucune anomalie", 260)
    d = socle.preparer(sub.head(top_n), axes).iloc[::-1].reset_index(drop=True)
    y   = list(range(len(d)))
    lo  = d[COL_LO].to_numpy(dtype="float64")
    hi  = d[COL_HI].to_numpy(dtype="float64")
    obs = d[COL_OBS].to_numpy(dtype="float64")
    pred= d[COL_PRED].to_numpy(dtype="float64")

    xs_band, ys_band, xs_over, ys_over = [], [], [], []
    for yi, l, h, o in zip(y, lo, hi, obs):
        xs_band += [l, h, None]; ys_band += [yi, yi, None]
        cible = h if o > h else l
        xs_over += [cible, o, None]; ys_over += [yi, yi, None]

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
    fig.add_scatter(x=pred, y=y, mode="markers", name="Prediction",
                    marker=dict(symbol="diamond", size=10, color="white",
                                line=dict(color="black", width=1.5)),
                    text=textes, hovertemplate="%{text}<extra></extra>")
    fig.add_scatter(x=obs, y=y, mode="markers", name="Valeur observee",
                    marker=dict(size=13, color=_ACCENT,
                                line=dict(color="#7b241c", width=1.3)),
                    text=textes, hovertemplate="%{text}<extra></extra>")
    fig.update_xaxes(tickformat="~s", title_text=f"<b>{target}</b>")
    fig.update_layout(
        yaxis=dict(tickmode="array", tickvals=y,
                   ticktext=d["libelle_rang"].str.slice(0, 40).tolist(),
                   tickfont=dict(size=10)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=.5),
        height=max(420, 44 * len(d) + 175),
        title=dict(
            text="Intervalle conforme, prediction et valeur observee<br>"
                 f"<sup>{titre}  ·  valeurs brutes de df, aucune sommation</sup>",
            font=dict(size=14), x=.015, xanchor="left"),
    )
    _mise_en_forme(fig, max(420, 44 * len(d) + 175), dict(l=10, r=50, t=115, b=55))
    return fig


def _fig_evolution(socle, cle_str, target, alpha):
    if cle_str is None:
        return _fig_vide("Aucune anomalie dans ce perimetre.")
    r = socle.ligne(socle.ano, cle_str)
    ctx = socle.contexte(r)
    hist, per = socle.historique(cle_str)
    if hist is None or not len(hist):
        return _fig_vide("Aucun historique pour ce sous-portefeuille.")

    val = hist[target].to_numpy(dtype="float64")
    p_valide = (ctx["per"] if ctx and ctx.get("per") in per else per[-1])
    xb = yb = xp = yp = xh = yh = []
    if ctx:
        xb, yb = [p_valide, p_valide], [ctx["lo"], ctx["hi"]]
        xp, yp = [p_valide], [ctx["pred"]]
        xh, yh = [p_valide], [ctx["obs"]]
    couvert = bool(ctx["couvert"]) if ctx else True
    coul = _OK if couvert else _ACCENT
    halo = "rgba(61,90,158,0.18)" if couvert else "rgba(192,57,43,0.20)"
    alerte = (f"  ·  <b>un seul trimestre dans df : pas de courbe possible</b>"
              if len(hist) < 2 else "")
    cle_parts = cle_str.split(SEP)

    fig = go.Figure()
    fig.add_scatter(x=per, y=val, mode="lines", line=dict(color=_VIDE, width=0),
                    fill="tozeroy", fillcolor="rgba(140,147,165,0.10)",
                    showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=xb, y=yb, mode="lines", line=dict(color=_BLEU, width=15),
                    opacity=.26, name=f"Intervalle conforme {100*(1-alpha):.0f} %")
    fig.add_scatter(x=xp, y=yp, mode="markers", name="Prediction",
                    marker=dict(symbol="diamond", size=11, color="white",
                                line=dict(color=_ENCRE, width=1.8)))
    fig.add_scatter(x=per, y=val, mode="lines+markers+text", name=target,
                    line=dict(color=_ENCRE, width=2.8, shape="spline", smoothing=.55),
                    marker=dict(size=9, color="white", line=dict(color=_ENCRE, width=2.2)),
                    text=[_fmt(v) for v in val], textposition="top center",
                    textfont=dict(size=9.5, color=_GRIS),
                    hovertemplate=f"<b>%{{x}}</b><br>{target} : <b>%{{y:,.0f}}</b><extra></extra>")
    fig.add_scatter(x=xh, y=yh, mode="markers", showlegend=False, hoverinfo="skip",
                    marker=dict(size=34, color=halo))
    fig.add_scatter(x=xh, y=yh, mode="markers",
                    name="Couvert" if couvert else "Hors intervalle",
                    marker=dict(size=13, color=coul, line=dict(color="white", width=2.4)))
    fig.update_xaxes(showgrid=False, linecolor=_GRILLE, showspikes=True,
                     spikemode="across", spikethickness=1.2, spikedash="dot",
                     spikecolor=_GRIS, title_text="<b>Trimestre</b>")
    fig.update_yaxes(gridcolor=_GRILLE, zeroline=False, tickformat="~s",
                     title_text=f"<b>{target}</b>")
    _mise_en_forme(fig, 420, dict(l=85, r=45, t=95, b=45))
    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", y=1.06, x=1, xanchor="right", font=dict(size=11)),
        title=dict(
            text=(f"<b>#{ctx['rang'] if ctx else '?'}  {' | '.join(cle_parts)}</b><br>"
                  f"<span style='font-size:11px'>{target} · {len(hist)} trimestre(s) · "
                  f"{per[0]} → {per[-1]}"
                  + (f" · periode validee <b>{p_valide}</b>" if ctx else "")
                  + alerte + "</span>"),
            font=dict(size=14), x=.015, xanchor="left"),
    )
    return fig


def _fig_variables(socle, cle_str, target, alpha, n=N_VARS_EXPLIC):
    if cle_str is None:
        return _fig_vide("Aucune anomalie dans ce perimetre.", 260)
    r = socle.ligne(socle.ano, cle_str)
    ctx = socle.contexte(r)
    hist, per = socle.historique(cle_str)
    if hist is None or len(hist) < 3:
        n_t = 0 if hist is None else len(hist)
        return _fig_vide(f"Classement indisponible : {n_t} trimestre(s) (min 3).", 260)
    t = socle.variables_explicatives(hist)
    if not len(t):
        return _fig_vide("Aucune variable explicative numerique exploitable.", 260)

    p_valide = ctx["per"] if ctx and ctx.get("per") in per else per[-1]
    k = per.index(p_valide)
    n_reel = min(n, len(t))
    fig = make_subplots(rows=n_reel, cols=1, shared_xaxes=True,
                        vertical_spacing=.055,
                        subplot_titles=[t.iloc[i]["variable"] for i in range(n_reel)])
    for i in range(n_reel):
        c = _DOUX[i % len(_DOUX)]
        v = t.iloc[i]["variable"]
        vals = hist[v].to_numpy(dtype="float64")
        fig.add_scatter(x=per, y=vals, mode="lines+markers", showlegend=False,
                        line=dict(color=c, width=2.3, shape="spline", smoothing=.55),
                        marker=dict(size=6, color="white", line=dict(color=c, width=1.8)),
                        hovertemplate=f"<b>%{{x}}</b><br>{v} : <b>%{{y:,.0f}}</b><extra></extra>",
                        row=i+1, col=1)
        fig.add_scatter(x=[per[k]], y=[vals[k]], mode="markers", showlegend=False,
                        marker=dict(size=13, color=c, line=dict(color="white", width=2.2)),
                        hoverinfo="skip", row=i+1, col=1)
    fig.update_xaxes(showgrid=False, showticklabels=False, linecolor=_GRILLE)
    fig.update_xaxes(showticklabels=True, title_text="<b>Trimestre</b>", row=n_reel, col=1)
    fig.update_yaxes(gridcolor=_GRILLE, zeroline=False, tickformat="~s", tickfont=dict(size=9))
    _mise_en_forme(fig, 150 * n_reel + 120, dict(l=85, r=45, t=95, b=45))
    fig.update_layout(
        hovermode="x unified",
        title=dict(text="<b>Les variables qui expliquent ce comportement</b>",
                   font=dict(size=14), x=.015, xanchor="left"))
    for a in fig.layout.annotations:
        a.update(x=0, xanchor="left", font=dict(size=12, color=_GRIS))
    return fig


# =============================================================================
# CARTES KPI — html.Div au lieu de HTML(...)
# =============================================================================
def _cartes_dash(sub, titre, score_global):
    if not len(sub):
        return html.Div("Aucune anomalie dans ce perimetre.", className="text-muted p-3")
    part = sub[COL_SCORE].sum() / score_global if len(sub) else 0
    hors = int((~((sub[COL_OBS] >= sub[COL_LO]) & (sub[COL_OBS] <= sub[COL_HI]))).sum())
    cartes = [
        ("Anomalies",         f"{len(sub):,}".replace(",", " "), "#37474f"),
        ("Part du score",     f"{100 * part:.1f} %",             "#ad1457"),
        ("Hors intervalle",   f"{hors:,}".replace(",", " "),     "#00838f"),
        ("Gravite moyenne",   _fmt4(sub[COL_SCORE].mean()),       "#5e35b1"),
        ("Pire anomalie",     _fmt4(sub[COL_SCORE].max()),        "#6a1b9a"),
    ]
    return html.Div([
        html.Div(titre, style={"fontSize": "15px", "fontWeight": "600",
                               "color": "#263238", "marginBottom": "10px",
                               "fontFamily": "system-ui,sans-serif"}),
        html.Div([
            html.Div([
                html.Div(t, style={"fontSize": "10.5px", "color": "#78909c",
                                   "textTransform": "uppercase", "letterSpacing": ".6px"}),
                html.Div(v, style={"fontSize": "19px", "fontWeight": "600",
                                   "color": c, "marginTop": "4px"}),
            ], style={"flex": "1", "minWidth": "130px", "background": "#fff",
                      "border": "1px solid #e0e0e0",
                      "borderLeft": f"5px solid {c}",
                      "borderRadius": "7px", "padding": "11px 13px",
                      "boxShadow": "0 1px 3px rgba(0,0,0,.07)"})
            for t, v, c in cartes
        ], style={"display": "flex", "gap": "9px", "flexWrap": "wrap"}),
    ], style={"fontFamily": "system-ui,sans-serif", "margin": "6px 0 14px 0"})


# =============================================================================
# TABLEAU — dash_table.DataTable au lieu de display(df.style...)
# =============================================================================
def _tableau_dash(socle, sub, titre, axes):
    if not len(sub):
        return html.P(f"Aucune anomalie dans le perimetre : {titre}", className="text-muted")
    d = socle.preparer(sub.head(N_LIGNES_TABLE), axes)
    t = pd.DataFrame({
        "Rang":    d["rang"].values,
        "Maille":  d["libelle"].values,
        "Y_obs":   d[COL_OBS].values,
        "Y_pred":  d[COL_PRED].values,
        "CP_bas":  d[COL_LO].values,
        "CP_haut": d[COL_HI].values,
        "Couvert": ((d[COL_OBS] >= d[COL_LO]) & (d[COL_OBS] <= d[COL_HI])).values,
        "Score":   d[COL_SCORE].values,
    })
    for c in ["Y_obs", "Y_pred", "CP_bas", "CP_haut"]:
        t[c] = t[c].round(0).astype(int)
    t["Score"] = t["Score"].round(4)
    return dash_table.DataTable(
        data=t.to_dict("records"),
        columns=[{"name": c, "id": c} for c in t.columns],
        page_size=N_LIGNES_TABLE,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": _ENCRE, "color": "white",
                      "fontWeight": "bold", "fontSize": "12px"},
        style_data_conditional=[
            {"if": {"filter_query": "{Couvert} = False"},
             "backgroundColor": "#FFEBEE", "color": _ACCENT},
            {"if": {"column_id": "Score"},
             "background": "linear-gradient(90deg,#fff 0%,#ffcdd2 100%)"},
        ],
        style_cell={"fontSize": "12px", "padding": "6px",
                    "fontFamily": "system-ui,sans-serif"},
        caption=f"Tableau de priorisation — {titre}",
    )


# =============================================================================
# CHARGEMENT DES DONNEES
# =============================================================================
TARGET  = "Claims_incurred"
ALPHA   = 0.10
ID_COLS_CANDIDATS = ["Partner", "Companies", "Lob", "Activity", "Periodicity", "Risk"]

base_path     = Path("/domino/datasets/local/conformal_pred_actuariat/DataSet")
target_folder = base_path / "Dossier_concatenee"

# Chercher le parquet principal (df complet avec toutes les periodes)
_parquets = sorted(target_folder.glob("*.parquet")) if target_folder.exists() else []
if _parquets:
    print(f"Chargement : {_parquets[0].name}")
    df = pd.read_parquet(_parquets[0])
elif (target_folder / "anomalies_prio.pkl").exists():
    print("Chargement : anomalies_prio.pkl")
    df = pd.read_pickle(target_folder / "anomalies_prio.pkl")
else:
    raise FileNotFoundError(
        f"Aucun fichier de donnees trouve dans {target_folder}\n"
        "Verifier le chemin ou deposer le fichier parquet.")

df.drop(columns=["annee"], inplace=True, errors="ignore")
ID_COLS = [c for c in ID_COLS_CANDIDATS if c in df.columns]
print(f"Donnees : {len(df):,} lignes · TARGET={TARGET} · ID_COLS={ID_COLS}".replace(",", " "))

socle = _Socle(df, ID_COLS, TARGET, ALPHA)
defaut_maille = next((c for c in ("Lob", "Partner", "Companies", "Risk") if c in socle.cles),
                     socle.cles[0])

# =============================================================================
# APPLICATION DASH
# =============================================================================
app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True,
           title="Dashboard Conformal Prediction")

_BANDEAU_STYLE = {"fontFamily": "system-ui,sans-serif", "fontSize": "12.5px",
                  "borderRadius": "6px", "padding": "9px 13px", "margin": "14px 0 8px 0"}

app.layout = dbc.Container(fluid=True, style={"padding": "16px"}, children=[

    # ── En-tête ───────────────────────────────────────────────────────────
    html.Div([
        html.Span("Tableau de bord — Anomalies conformes",
                  style={"fontWeight": "600", "fontSize": "16px", "color": "#0d47a1"}),
        html.Span("  |  IFRS 17 / Solvabilite II · Conformal Prediction",
                  style={"fontSize": "12px", "color": "#546e7a"}),
    ], style={"background": "#e3f2fd", "padding": "12px 18px",
              "borderRadius": "8px", "marginBottom": "14px",
              "borderLeft": "5px solid #1565C0"}),

    # ── Axes (ToggleButtons → Checklist) ──────────────────────────────────
    html.Div([
        html.Div("Axes — structurent le cercle et composent les libelles. "
                 "Ne modifient aucun montant.",
                 style={**_BANDEAU_STYLE, "background": "#e3f2fd", "color": "#0d47a1"}),
        dcc.Checklist(
            id="axes-check",
            options=[{"label": f"  {c}", "value": c} for c in socle.cles],
            value=socle.cles[:2],
            inline=True,
            inputStyle={"marginRight": "4px"},
            labelStyle={"marginRight": "16px", "fontSize": "13px",
                        "background": "#1565C0", "color": "white",
                        "padding": "5px 12px", "borderRadius": "4px",
                        "cursor": "pointer"},
        ),
    ]),

    # ── Cercle ────────────────────────────────────────────────────────────
    dcc.Graph(id="g-cercle", config={"displayModeBar": False}),

    # ── Perimetre ─────────────────────────────────────────────────────────
    html.Div("Perimetre — la maille et la valeur filtrent tout le tableau de bord.",
             style={**_BANDEAU_STYLE, "background": "#e8f5e9", "color": "#1b5e20"}),
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id="dd-maille",
            options=[{"label": c, "value": c} for c in socle.cles],
            value=defaut_maille, clearable=False,
            placeholder="Maille…",
        ), md=4),
        dbc.Col(dcc.Dropdown(
            id="dd-valeur",
            options=socle.options_valeurs(defaut_maille),
            value=VUE_GENERALE,
            placeholder="— vue generale —",
        ), md=8),
    ], className="mb-2"),

    # ── Cartes KPI ────────────────────────────────────────────────────────
    html.Div(id="zone-cartes"),

    # ── Barres + Forest ───────────────────────────────────────────────────
    dcc.Graph(id="g-barres"),
    dcc.Graph(id="g-forest"),

    # ── Anomalie en detail ────────────────────────────────────────────────
    html.Div("L'anomalie en detail — evolution et variables explicatives.",
             style={**_BANDEAU_STYLE, "background": "#fff3e0", "color": "#e65100"}),
    dcc.Dropdown(id="dd-unite", options=[], value=None,
                 placeholder="Anomalie…",
                 style={"marginBottom": "8px"}),
    dcc.Graph(id="g-evol"),
    dcc.Graph(id="g-vars"),

    # ── Tableau ───────────────────────────────────────────────────────────
    html.Div("Tableau de priorisation — perimetre courant.",
             style={**_BANDEAU_STYLE, "background": "#eceff1", "color": "#37474f"}),
    html.Div(id="zone-tableau"),

    html.Div(style={"height": "40px"}),
])


# =============================================================================
# CALLBACKS
# =============================================================================

# 1 — maille change → options valeur
@app.callback(
    Output("dd-valeur", "options"),
    Output("dd-valeur", "value"),
    Input("dd-maille", "value"),
)
def cb_valeur_options(maille):
    return socle.options_valeurs(maille or defaut_maille), VUE_GENERALE


# 2 — panneaux principaux (cercle, barres, forest, cartes, unite)
@app.callback(
    Output("zone-cartes", "children"),
    Output("g-cercle",    "figure"),
    Output("g-barres",    "figure"),
    Output("g-forest",    "figure"),
    Output("dd-unite",    "options"),
    Output("dd-unite",    "value"),
    Input("axes-check",   "value"),
    Input("dd-maille",    "value"),
    Input("dd-valeur",    "value"),
    Input("g-cercle",     "clickData"),
)
def cb_panneaux(axes, maille, valeur, click_data):
    ctx = callback_context
    triggered = ctx.triggered_id if ctx.triggered_id else "dd-maille"

    # Clic sur le cercle → derive maille + valeur
    if triggered == "g-cercle" and click_data:
        try:
            point_id = click_data["points"][0]["id"]
            parts = point_id.split(SEP)
            axes_actifs = axes or [socle.cles[0]]
            if parts and len(parts) <= len(axes_actifs):
                maille = axes_actifs[len(parts) - 1]
                valeur = parts[-1]
        except Exception:
            pass

    axes_actifs = axes or [socle.cles[0]]
    sub, titre = socle.filtrer(maille or defaut_maille, valeur or VUE_GENERALE)

    cartes  = _cartes_dash(sub, titre, socle.score_global)
    f_cerc  = _fig_cercle(socle, sub, titre, axes_actifs)
    f_barr  = _fig_barres(socle, sub, titre, TARGET, axes_actifs)
    f_for   = _fig_forest(socle, sub, titre, TARGET, axes_actifs)
    options = socle.unites(sub, axes_actifs)
    valeur_unite = options[0]["value"] if options else None

    return cartes, f_cerc, f_barr, f_for, options, valeur_unite


# 3 — detail evolution + variables
@app.callback(
    Output("g-evol", "figure"),
    Output("g-vars", "figure"),
    Input("dd-unite", "value"),
)
def cb_detail(unite):
    f_evol = _fig_evolution(socle, unite, TARGET, ALPHA)
    f_vars = _fig_variables(socle, unite, TARGET, ALPHA)
    return f_evol, f_vars


# 4 — tableau de priorisation
@app.callback(
    Output("zone-tableau", "children"),
    Input("axes-check", "value"),
    Input("dd-maille",  "value"),
    Input("dd-valeur",  "value"),
)
def cb_tableau(axes, maille, valeur):
    axes_actifs = axes or [socle.cles[0]]
    sub, titre = socle.filtrer(maille or defaut_maille, valeur or VUE_GENERALE)
    return _tableau_dash(socle, sub, titre, axes_actifs)


# =============================================================================
# LANCEMENT
# =============================================================================
if __name__ == "__main__":
    print("Dashboard sur : http://localhost:8888")
    app.run(debug=False, port=8888, host="0.0.0.0")
