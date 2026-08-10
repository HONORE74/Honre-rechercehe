# =========================================================================
# RUBAN CONFORME INTERACTIF
#   Axe X : unites statistiques (ID_COLS, 5 champs) triees par prediction
#   Axe Y : RBNS_eop
#   Ruban : borne basse -> borne haute du CP (large et continu)
#   Points: bleu = dans l'intervalle / rouge = hors intervalle
# =========================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def ruban_conforme(df, n_unites=80, echelle="log", label_cols=None,
                   taille_point=5, n_ticks=25, random_state=42):
    """
    df          : results_v2 (ou results_test)
    n_unites    : nombre d'unites statistiques affichees (60-120 = bon compromis)
    echelle     : "log" -> ruban de largeur constante (recommande)
                  "lineaire" -> fidele aux montants, mais ruban ecrase a gauche
    label_cols  : colonnes composant l'etiquette X (5 par defaut)
    """
    label_cols = label_cols or [c for c in ID_COLS if c in df.columns][:5]

    d = df.sample(min(n_unites, len(df)), random_state=random_state).copy()
    d = d.sort_values("y_pred").reset_index(drop=True)

    x = np.arange(len(d))
    lo = d["borne_basse"].values.astype(float).copy()
    hi = d["borne_haute"].values.astype(float)
    pred = d["y_pred"].values.astype(float)
    obs = d["y_obs"].values.astype(float)
    dedans = d["dans_intervalle"].values.astype(bool)

    label = d[label_cols].astype(str).agg(" | ".join, axis=1)

    log_ok = (echelle == "log") and (obs > 0).all() and (pred > 0).all()
    if log_ok:
        lo = np.maximum(lo, np.nanmin(obs[obs > 0]) * 1e-2)

    # Diagnostic : le ruban sera-t-il visible ?
    largeur_rel = np.median((hi - lo) / np.maximum(pred, 1e-9))
    print(f"Unites affichees          : {len(d)}")
    print(f"Largeur mediane du ruban  : {100*largeur_rel:.1f} % de la prediction")
    print(f"Observations hors ruban   : {(~dedans).sum()} / {len(d)}  ({100*(~dedans).mean():.1f} %)")
    if largeur_rel < 0.05:
        print("  -> ruban naturellement etroit : zoomer dans la figure (molette) pour le detailler")

    hover = [(f"<b>{l}</b><br>"
              f"Periode : {r['year']}-T{int(r['quarter'])}<br>"
              f"RBNS observe  : {r['y_obs']:,.0f}<br>"
              f"Prediction    : {r['y_pred']:,.0f}<br>"
              f"Intervalle CP : [{r['borne_basse']:,.0f} ; {r['borne_haute']:,.0f}]<br>"
              f"Statut : {'DANS' if r['dans_intervalle'] else 'HORS'} l'intervalle")
             for l, (_, r) in zip(label, d.iterrows())]

    fig = go.Figure()

    # --- Ruban conforme : borne basse puis borne haute avec remplissage ---
    fig.add_trace(go.Scatter(
        x=x, y=lo, mode="lines",
        line=dict(color="#1f4e9c", width=2.8),
        name="Borne inferieure CP", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=x, y=hi, mode="lines",
        line=dict(color="#1f4e9c", width=2.8),
        fill="tonexty", fillcolor="rgba(110,160,225,0.32)",
        name="Borne superieure CP", hoverinfo="skip"))

    # --- Courbe de prediction, au coeur du ruban ---
    fig.add_trace(go.Scatter(
        x=x, y=pred, mode="lines",
        line=dict(color="black", width=2.2),
        name="Prediction (CQR)", hoverinfo="skip"))

    # --- Observations : petits points ---
    fig.add_trace(go.Scatter(
        x=x[dedans], y=obs[dedans], mode="markers",
        marker=dict(size=taille_point, color="#1565c0",
                    line=dict(width=0.4, color="#0d3c78")),
        name=f"Dans l'intervalle ({dedans.sum()})",
        text=[h for h, k in zip(hover, dedans) if k],
        hovertemplate="%{text}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=x[~dedans], y=obs[~dedans], mode="markers",
        marker=dict(size=taille_point + 2, color="#e53935",
                    line=dict(width=0.6, color="#8e0000")),
        name=f"HORS intervalle ({(~dedans).sum()})",
        text=[h for h, k in zip(hover, dedans) if not k],
        hovertemplate="%{text}<extra></extra>"))

    pas = max(1, len(d) // n_ticks)
    fig.update_layout(
        title=dict(text=f"Intervalles de Conformal Prediction et valeurs observees "
                        f"— {len(d)} unites statistiques (couverture cible {100*(1-ALPHA):.0f} %)",
                   font=dict(size=15)),
        xaxis=dict(title="Unites statistiques (triees par prediction croissante)",
                   tickmode="array", tickvals=x[::pas],
                   ticktext=label.iloc[::pas].tolist(),
                   tickangle=60, tickfont=dict(size=8)),
        yaxis=dict(title=TARGET + ("  (echelle log)" if log_ok else ""),
                   type="log" if log_ok else "linear"),
        template="plotly_white", height=680, hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    fig.show()


ruban_conforme(results_v2, n_unites=80, echelle="log")
