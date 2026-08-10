bLOUQUE 1

import numpy as np
import pandas as pd
import plotly.graph_objects as go

TOP_N      = 15
N_REF      = 8
MAXLEN_LAB = 28


def _labels(df, id_cols=None, maxlen=MAXLEN_LAB):
    id_cols = id_cols or [c for c in ID_COLS if c in df.columns]
    lab = df[id_cols].astype(str).agg("_".join, axis=1)
    return lab.str.slice(0, maxlen)


def _preparer_donnees(anomalies_prio, results_v2, top_n=TOP_N, n_ref=N_REF):
    top = anomalies_prio.head(top_n).copy()
    top["_bloc"] = "anomalie"

    normales = results_v2[results_v2["dans_intervalle"]].copy()
    if GWP_COL in normales.columns and len(normales):
        ref = normales.nlargest(min(n_ref, len(normales)), GWP_COL).copy()
    else:
        ref = normales.head(n_ref).copy()
    ref["_bloc"] = "reference"

    for c in ["A_ecart_borne", "B_erreur_modele", "score_composite", "rank"]:
        if c not in ref.columns:
            ref[c] = np.nan

    cols = [c for c in ID_COLS if c in top.columns] + [
        "y_obs", "y_pred", "borne_basse", "borne_haute", "dans_intervalle",
        "A_ecart_borne", "B_erreur_modele", "score_composite", "rank", "_bloc"]
    if GWP_COL in top.columns and GWP_COL in ref.columns:
        cols.append(GWP_COL)
    cols = [c for c in cols if c in top.columns and c in ref.columns]

    d = pd.concat([top[cols], ref[cols]], ignore_index=True)
    d["label"] = _labels(d)
    return d


def _hover(row, id_cols):
    ident = " | ".join(f"{c}={row[c]}" for c in id_cols if c in row.index)
    txt = (f"<b>{ident}</b><br>"
           f"y_obs: {row['y_obs']:,.0f}<br>"
           f"y_pred: {row['y_pred']:,.0f}<br>"
           f"Intervalle: [{row['borne_basse']:,.0f} ; {row['borne_haute']:,.0f}]")
    if pd.notna(row.get("score_composite", np.nan)):
        txt += (f"<br>Rang: {int(row['rank'])}<br>"
                f"Score: {row['score_composite']:.4f}<br>"
                f"A_i: {row['A_ecart_borne']:.3f}  B_i: {row['B_erreur_modele']:.3f}")
        if GWP_COL in row.index:
            txt += f"<br>{GWP_COL}: {row[GWP_COL]:,.0f}"
    return txt

print("Socle pret.")


bLOQUE 2




def figure_absolue_interactive(anomalies_prio, results_v2, top_n=TOP_N, n_ref=N_REF, log_y=True):
    d = _preparer_donnees(anomalies_prio, results_v2, top_n, n_ref)
    x = list(range(len(d)))
    n_ano = int((d["_bloc"] == "anomalie").sum())
    dedans = d["dans_intervalle"].values.astype(bool)
    lo = d["borne_basse"].values.astype(float)
    hi = d["borne_haute"].values.astype(float)
    obs = d["y_obs"].values.astype(float)
    pred = d["y_pred"].values.astype(float)
    id_cols = [c for c in ID_COLS if c in d.columns]
    log_ok = log_y and (obs > 0).all() and (pred > 0).all()

    fig = go.Figure()

    xs_band, ys_band = [], []
    for xi, l, h in zip(x, lo, hi):
        xs_band += [xi, xi, None]
        ys_band += [l, h, None]
    fig.add_trace(go.Scatter(x=xs_band, y=ys_band, mode="lines",
                              line=dict(color="#3a6bbf", width=10), opacity=0.28,
                              name=f"Intervalle conforme ({100*(1-ALPHA):.0f}%)", hoverinfo="skip"))

    xs_over, ys_over = [], []
    for xi, o, l, h, dd in zip(x, obs, lo, hi, dedans):
        if not dd:
            cible = h if o > h else l
            xs_over += [xi, xi, None]
            ys_over += [cible, o, None]
    if xs_over:
        fig.add_trace(go.Scatter(x=xs_over, y=ys_over, mode="lines",
                                  line=dict(color="#c0392b", width=2, dash="dot"),
                                  showlegend=False, hoverinfo="skip"))

    fig.add_trace(go.Scatter(x=x, y=pred, mode="markers", name="Prediction",
                              marker=dict(symbol="diamond", size=10, color="white",
                                          line=dict(color="black", width=1.5)),
                              text=[_hover(r, id_cols) for _, r in d.iterrows()],
                              hovertemplate="%{text}<extra></extra>"))

    for mask, color, dark, nom in [(dedans, "#27ae60", "#145a32", "Dans l'intervalle"),
                                    (~dedans, "#c0392b", "#7b241c", "HORS intervalle")]:
        xi_m = np.array(x)[mask]
        if len(xi_m) == 0:
            continue
        sub = d[mask]
        fig.add_trace(go.Scatter(
            x=xi_m, y=obs[mask], mode="markers", name=nom,
            marker=dict(size=13, color=color, line=dict(color=dark, width=1.3)),
            text=[_hover(r, id_cols) for _, r in sub.iterrows()],
            hovertemplate="%{text}<extra></extra>"))

    for i in range(n_ano):
        fig.add_annotation(x=x[i], y=obs[i], text=f"#{int(d['rank'].iloc[i])}",
                            showarrow=False, yshift=16, font=dict(size=10, color="#7b241c"))

    if n_ano < len(d):
        fig.add_vline(x=n_ano - 0.5, line=dict(color="gray", dash="dash", width=1.5))
        fig.add_annotation(x=n_ano - 0.5, y=1, yref="paper", text="unites normales (reference)",
                            showarrow=False, xanchor="left", font=dict(size=10, color="gray"))

    fig.update_layout(
        title="Intervalles conformes et valeurs observees - top anomalies prioritaires",
        xaxis=dict(tickmode="array", tickvals=x, ticktext=d["label"].tolist(), tickangle=45),
        yaxis=dict(title=TARGET, type="log" if log_ok else "linear"),
        template="plotly_white", height=650, hovermode="closest")
    fig.show()


figure_absolue_interactive(anomalies_prio, results_v2)


Bloque 3

def figure_normalisee_interactive(anomalies_prio, results_v2, top_n=TOP_N, n_ref=N_REF):
    d = _preparer_donnees(anomalies_prio, results_v2, top_n, n_ref)
    centre = (d["borne_haute"].values + d["borne_basse"].values) / 2
    demi = np.maximum((d["borne_haute"].values - d["borne_basse"].values) / 2, 1e-9)
    z_obs = (d["y_obs"].values - centre) / demi
    z_pred = (d["y_pred"].values - centre) / demi
    x = list(range(len(d)))
    dedans = d["dans_intervalle"].values.astype(bool)
    n_ano = int((d["_bloc"] == "anomalie").sum())
    id_cols = [c for c in ID_COLS if c in d.columns]

    fig = go.Figure()
    fig.add_hrect(y0=-1, y1=1, fillcolor="#a9c4ea", opacity=0.35, line_width=0,
                 annotation_text=f"Zone conforme ({100*(1-ALPHA):.0f}%)", annotation_position="top left")
    fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=1))

    xs_over, ys_over = [], []
    for xi, z, dd in zip(x, z_obs, dedans):
        if not dd:
            xs_over += [xi, xi, None]
            ys_over += [np.sign(z), z, None]
    if xs_over:
        fig.add_trace(go.Scatter(x=xs_over, y=ys_over, mode="lines",
                                  line=dict(color="#c0392b", width=2, dash="dot"),
                                  showlegend=False, hoverinfo="skip"))

    fig.add_trace(go.Scatter(x=x, y=z_pred, mode="markers", name="Prediction",
                              marker=dict(symbol="diamond", size=9, color="white",
                                          line=dict(color="black", width=1.3)),
                              hoverinfo="skip"))

    for mask, color, dark, nom in [(dedans, "#27ae60", "#145a32", "Dans l'intervalle"),
                                    (~dedans, "#c0392b", "#7b241c", "HORS intervalle")]:
        xi_m = np.array(x)[mask]
        if len(xi_m) == 0:
            continue
        sub = d[mask]
        z_sub = z_obs[mask]
        fig.add_trace(go.Scatter(
            x=xi_m, y=z_sub, mode="markers", name=nom,
            marker=dict(size=13, color=color, line=dict(color=dark, width=1.3)),
            text=[_hover(r, id_cols) + f"<br>z = {zz:.2f}" for (_, r), zz in zip(sub.iterrows(), z_sub)],
            hovertemplate="%{text}<extra></extra>"))

    if n_ano < len(d):
        fig.add_vline(x=n_ano - 0.5, line=dict(color="gray", dash="dash", width=1.5))

    fig.update_layout(
        title="Vue normalisee - tous les portefeuilles a la meme echelle (z = ecart / demi-largeur)",
        xaxis=dict(tickmode="array", tickvals=x, ticktext=d["label"].tolist(), tickangle=45),
        yaxis=dict(title="Position normalisee (z)"),
        template="plotly_white", height=600, hovermode="closest")
    fig.show()


figure_normalisee_interactive(anomalies_prio, results_v2)



Bloque 4 


def figure_forest_interactive(anomalies_prio, top_n=TOP_N, log_x=True):
    d = anomalies_prio.head(top_n).copy()
    d["label"] = _labels(d)
    d = d.iloc[::-1].reset_index(drop=True)
    y = list(range(len(d)))
    id_cols = [c for c in ID_COLS if c in d.columns]

    lo = d["borne_basse"].values.astype(float)
    hi = d["borne_haute"].values.astype(float)
    obs = d["y_obs"].values.astype(float)
    pred = d["y_pred"].values.astype(float)
    log_ok = log_x and (obs > 0).all() and (pred > 0).all()

    fig = go.Figure()
    xs_band, ys_band = [], []
    for yi, l, h in zip(y, lo, hi):
        xs_band += [l, h, None]
        ys_band += [yi, yi, None]
    fig.add_trace(go.Scatter(x=xs_band, y=ys_band, mode="lines",
                              line=dict(color="#3a6bbf", width=10), opacity=0.3,
                              name=f"Intervalle conforme ({100*(1-ALPHA):.0f}%)", hoverinfo="skip"))

    xs_over, ys_over = [], []
    for yi, o, l, h in zip(y, obs, lo, hi):
        cible = h if o > h else l
        xs_over += [cible, o, None]
        ys_over += [yi, yi, None]
    fig.add_trace(go.Scatter(x=xs_over, y=ys_over, mode="lines",
                              line=dict(color="#c0392b", width=2, dash="dot"),
                              showlegend=False, hoverinfo="skip"))

    fig.add_trace(go.Scatter(x=pred, y=y, mode="markers", name="Prediction",
                              marker=dict(symbol="diamond", size=10, color="white",
                                          line=dict(color="black", width=1.5)),
                              text=[_hover(r, id_cols) for _, r in d.iterrows()],
                              hovertemplate="%{text}<extra></extra>"))
    fig.add_trace(go.Scatter(x=obs, y=y, mode="markers", name="Valeur comptabilisee",
                              marker=dict(size=13, color="#c0392b", line=dict(color="#7b241c", width=1.3)),
                              text=[_hover(r, id_cols) for _, r in d.iterrows()],
                              hovertemplate="%{text}<extra></extra>"))

    fig.update_layout(
        title=f"Top {top_n} anomalies - intervalle conforme, prediction et valeur observee",
        xaxis=dict(title=TARGET, type="log" if log_ok else "linear"),
        yaxis=dict(tickmode="array", tickvals=y,
                  ticktext=[f"#{int(r)}  {l}" for r, l in zip(d["rank"], d["label"])]),
        template="plotly_white", height=max(450, 40 * len(d) + 150))
    fig.show()



Bloque 5

def figure_decomposition_interactive(anomalies_prio, top_n=TOP_N):
    facteurs = {"A (ecart borne)": "A_ecart_borne",
                "B (erreur modele)": "B_erreur_modele",
                f"{GWP_COL} (exposition)": GWP_COL}
    facteurs = {k: v for k, v in facteurs.items() if v in anomalies_prio.columns}

    pct = pd.DataFrame({k: anomalies_prio[v].rank(pct=True) * 100
                        for k, v in facteurs.items()})
    pct["Score final"] = anomalies_prio["score_composite"].rank(pct=True) * 100
    d = pct.head(top_n).iloc[::-1]
    labels = _labels(anomalies_prio.head(top_n)).iloc[::-1]
    rangs = anomalies_prio["rank"].head(top_n).astype(int).iloc[::-1]
    yticks = [f"#{r}  {l}" for r, l in zip(rangs, labels)]

    fig = go.Figure(data=go.Heatmap(
        z=d.values, x=list(d.columns), y=yticks,
        colorscale="RdYlGn_r", zmin=0, zmax=100,
        text=np.round(d.values, 0), texttemplate="%{text}",
        colorbar=dict(title="Rang percentile")))
    fig.update_layout(
        title="Quel facteur explique la priorite ? (100 = le plus eleve du lot)",
        template="plotly_white", height=max(450, 32 * len(d) + 150))
    fig.show()

    corr = anomalies_prio[["score_composite"] + list(facteurs.values())].corr(
        method="spearman")["score_composite"].drop("score_composite")
    print("Correlation de Spearman entre le score final et chaque facteur :")
    print(corr.round(3).to_string())
    dom = corr.idxmax()
    print(f"\nFacteur dominant : {dom}  (rho = {corr.max():.3f})")
    if corr.max() > 0.9:
        print("/!\\ Correlation > 0.9 : le classement est quasi entierement porte")
        print("    par ce seul facteur.")


figure_decomposition_interactive(anomalies_prio)


figure_forest_interactive(anomalies_prio)
