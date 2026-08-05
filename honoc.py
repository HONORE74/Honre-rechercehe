import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

def preparer_trois_variantes(df_calib, y_calib, y_lo_calib, y_hi_calib,
                             df_test, y_lo_test, y_hi_test, y_test,
                             segment_col, alpha=0.10, n_min=40):
    """Construit les 3 scenarios : sans conformalisation / marginale / conditionnelle."""
    scores = np.maximum(y_lo_calib - y_calib, y_calib - y_hi_calib)
    n = len(scores)
    Q_glob = np.quantile(scores, min(np.ceil((n+1)*(1-alpha))/n, 1.0), method="higher")

    seg_cal = df_calib[segment_col].astype(str).values
    Q_seg = {}
    for s in np.unique(seg_cal):
        m = seg_cal == s; k = int(m.sum())
        if k < n_min:
            Q_seg[s] = Q_glob
        else:
            Q_seg[s] = float(np.quantile(scores[m],
                        min(np.ceil((k+1)*(1-alpha))/k, 1.0), method="higher"))
    seg_te = df_test[segment_col].astype(str).values
    Q_te = np.array([Q_seg.get(s, Q_glob) for s in seg_te])

    variantes = {
        "Sans conformalisation": (np.clip(y_lo_test, 0, None),          y_hi_test),
        "Couverture marginale":  (np.clip(y_lo_test - Q_glob, 0, None), y_hi_test + Q_glob),
        "Couverture conditionnelle": (np.clip(y_lo_test - Q_te, 0, None), y_hi_test + Q_te),
    }
    out = {}
    for nom, (lo, hi) in variantes.items():
        out[nom] = pd.DataFrame({"y_obs": np.asarray(y_test, float),
                                 "borne_basse": lo, "borne_haute": hi,
                                 "largeur": hi - lo, "segment": seg_te,
                                 "couvert": (np.asarray(y_test, float) >= lo) &
                                            (np.asarray(y_test, float) <= hi)})
    return out, Q_glob, Q_seg


def figure_reference(variantes, alpha=0.10, n_points=180, n_groupes=2, seed=42):
    """Reproduction de la figure canonique : 3 colonnes x 2 lignes."""
    rng = np.random.default_rng(seed)
    ref = list(variantes.values())[0]
    groupes = ref["segment"].value_counts().head(n_groupes).index.tolist()

    fig = plt.figure(figsize=(16.5, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25], hspace=0.28, wspace=0.16)
    VERT, ROUGE, JAUNE = "#5CB85C", "#D9534F", "#F0C419"

    for j, (nom, d) in enumerate(variantes.items()):
        # ---------- Ligne 1 : disques de points par groupe ----------
        ax = fig.add_subplot(gs[0, j]); ax.set_xlim(-1.15, 2.35); ax.set_ylim(-1.5, 1.5)
        ax.axis("off"); ax.set_aspect("equal")
        for g, cx in zip(groupes, [0.0, 2.0] if n_groupes == 2 else np.arange(n_groupes)*2.0):
            sub = d[d["segment"] == g]
            if not len(sub): continue
            ech = sub.sample(min(n_points, len(sub)), random_state=seed)
            r = np.sqrt(rng.random(len(ech))); th = rng.random(len(ech)) * 2*np.pi
            ax.add_patch(Circle((cx, 0), 1.0, facecolor="none",
                                edgecolor="#999999", lw=1.3))
            ax.scatter(cx + r*np.cos(th)*0.93, r*np.sin(th)*0.93, s=13,
                       c=np.where(ech["couvert"], VERT, ROUGE), alpha=0.85)
            ax.text(cx, 1.16, str(g)[:16], ha="center", fontsize=10, fontweight="bold")
            ax.text(cx, -1.28, f"{sub['couvert'].mean():.0%} de couverture",
                    ha="center", fontsize=10.5, fontweight="bold")
        ax.set_title(nom, fontsize=14, fontweight="bold", pad=14)
        if j == 0:
            ax.text(-1.45, 0, "Erreurs par groupe", rotation=90, va="center",
                    fontsize=11.5, fontweight="bold")

        # ---------- Ligne 2 : bandes d'intervalles ----------
        ax2 = fig.add_subplot(gs[1, j])
        s = d.sort_values("borne_haute").reset_index(drop=True)
        p95 = np.percentile(s["y_obs"], 95)
        s = s[s["y_obs"] <= p95].reset_index(drop=True)
        x = np.arange(len(s))
        ax2.fill_between(x, s["borne_basse"], s["borne_haute"],
                         color=JAUNE, alpha=0.45, zorder=1)
        ax2.plot(x, s["borne_basse"], color="#C9A200", lw=1.6, zorder=2)
        ax2.plot(x, s["borne_haute"], color="#C9A200", lw=1.6, zorder=2)
        ok = s["couvert"].values
        ax2.scatter(x[ok],  s.loc[ok, "y_obs"],  s=7, color="#333333", alpha=0.55, zorder=3)
        ax2.scatter(x[~ok], s.loc[~ok, "y_obs"], s=16, color=ROUGE, zorder=4)
        ax2.set_xlabel("Observations triees", fontsize=10)
        ax2.set_xticks([])
        cov = d["couvert"].mean()
        larg = d["largeur"].median()
        ax2.set_title(f"Couverture globale {cov:.1%}   |   largeur mediane {larg:,.0f} EUR",
                      fontsize=10.5)
        if j == 0:
            ax2.set_ylabel("Intervalles de prediction", fontsize=11.5, fontweight="bold")

    fig.legend(handles=[
        Line2D([0],[0], marker="o", color="w", markerfacecolor=VERT, markersize=11,
               label="Observation correctement couverte"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=ROUGE, markersize=11,
               label="Observation NON couverte"),
        Line2D([0],[0], color=JAUNE, lw=9, alpha=0.6, label="Intervalle de prediction")],
        loc="upper center", ncol=3, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle(f"Sans couverture  /  marginale  /  conditionnelle   "
                 f"(cible {1-alpha:.0%})", fontsize=15, fontweight="bold", y=1.06)
    plt.savefig("figure_reference_couverture.png", dpi=220, bbox_inches="tight")
    plt.show()


variantes, Q_glob, Q_seg = preparer_trois_variantes(
    df_calib, y_calib.values, y_lo_calib, y_hi_calib,
    df_test, y_lo_test, y_hi_test, y_test.values,
    segment_col="Lob", alpha=ALPHA)
figure_reference(variantes, alpha=ALPHA)










import plotly.graph_objects as go

def interactif_trois_scenarios(variantes, alpha=0.10, n_min=25,
                               fichier="cp_trois_scenarios.html"):
    cible = 1 - alpha
    fig = go.Figure()
    boutons, noms = [], list(variantes)

    for k, nom in enumerate(noms):
        d = variantes[nom]
        t = (d.groupby("segment")["couvert"].agg(k_="sum", n="size").reset_index())
        t = t[t["n"] >= n_min]
        t["cov"] = t["k_"] / t["n"]
        t = t.sort_values("cov").reset_index(drop=True)
        vis = (k == 0)

        fig.add_trace(go.Scatter(x=t["segment"], y=t["cov"], mode="lines",
                                 line=dict(color="#2F2F2F", width=2),
                                 showlegend=False, visible=vis, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=t["segment"], y=t["cov"], mode="markers+text",
            marker=dict(size=np.clip(14 + 26*t["n"]/t["n"].max(), 14, 38),
                        color=np.where(np.abs(t["cov"]-cible) <= 0.05, "#5CB85C", "#D9534F"),
                        line=dict(color="#2F2F2F", width=1.5)),
            text=[f"{v:.0%}" for v in t["cov"]], textposition="top center",
            customdata=np.column_stack([t["n"], t["cov"]-cible]),
            hovertemplate="<b>%{x}</b><br>Couverture : %{y:.2%}<br>"
                          "n : %{customdata[0]:,.0f}<br>"
                          "Ecart cible : %{customdata[1]:+.2%}<extra></extra>",
            showlegend=False, visible=vis))

    for k, nom in enumerate(noms):
        v = []
        for j in range(len(noms)):
            v.extend([j == k] * 2)
        cov_g = variantes[nom]["couvert"].mean()
        lg = variantes[nom]["largeur"].median()
        boutons.append(dict(label=nom, method="update", args=[
            {"visible": v},
            {"title": f"{nom}<br><sub>Couverture globale {cov_g:.1%} | "
                      f"largeur mediane {lg:,.0f} EUR</sub>"}]))

    fig.add_hrect(y0=cible-0.05, y1=cible+0.05, fillcolor="#5CB85C",
                  opacity=0.12, line_width=0)
    fig.add_hline(y=cible, line_dash="dash", line_color="#333333", line_width=2,
                  annotation_text=f"Cible {cible:.0%}", annotation_position="right")
    fig.update_layout(
        title=f"{noms[0]}<br><sub>Couverture globale "
              f"{variantes[noms[0]]['couvert'].mean():.1%}</sub>",
        updatemenus=[dict(buttons=boutons, direction="down", x=1.0, xanchor="right",
                          y=1.18, yanchor="top", showactive=True)],
        yaxis=dict(title="Couverture par segment", tickformat=".0%", range=[0, 1.06]),
        xaxis=dict(title="Segment", tickangle=-40),
        template="plotly_white", height=620, margin=dict(t=120, b=120))
    fig.write_html(fichier); fig.show()
    print(f"Sauvegarde : {fichier}")

interactif_trois_scenarios(variantes, ALPHA)













def interactif_intervalles(variantes, nom="Couverture conditionnelle",
                           max_pts=2500, fichier="cp_intervalles.html"):
    d = variantes[nom].copy()
    p95 = np.percentile(d["y_obs"], 95)
    d = d[d["y_obs"] <= p95].sort_values("borne_haute").reset_index(drop=True)
    if len(d) > max_pts:
        d = d.iloc[np.linspace(0, len(d)-1, max_pts).astype(int)].reset_index(drop=True)
    x = np.arange(len(d))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([d["borne_haute"], d["borne_basse"][::-1]]),
        fill="toself", fillcolor="rgba(240,196,25,0.35)", line=dict(width=0),
        name="Intervalle conforme", hoverinfo="skip"))
    for masque, coul, lab, taille in [(d["couvert"], "#333333", "Couvert", 5),
                                      (~d["couvert"], "#D9534F", "NON couvert", 9)]:
        s = d[masque]
        fig.add_trace(go.Scatter(x=x[masque.values], y=s["y_obs"], mode="markers",
            marker=dict(size=taille, color=coul, opacity=0.75), name=lab,
            customdata=np.column_stack([s["borne_basse"], s["borne_haute"],
                                        s["largeur"], s["segment"]]),
            hovertemplate="Observe : %{y:,.0f} EUR<br>"
                          "Intervalle : [%{customdata[0]:,.0f} ; %{customdata[1]:,.0f}]<br>"
                          "Largeur : %{customdata[2]:,.0f} EUR<br>"
                          "Segment : %{customdata[3]}<extra></extra>"))
    fig.update_layout(
        title=f"{nom} -- couverture {variantes[nom]['couvert'].mean():.1%}"
              f"<br><sub>Survol pour le detail. Zoom a la souris.</sub>",
        xaxis=dict(title="Observations triees par borne haute", showticklabels=False),
        yaxis=dict(title="Valeur (EUR)"),
        template="plotly_white", height=640, hovermode="closest")
    fig.write_html(fichier); fig.show()
    print(f"Sauvegarde : {fichier}")

interactif_intervalles(variantes)













import plotly.express as px

def interactif_heatmap(variantes, nom="Couverture conditionnelle",
                       alpha=0.10, n_strates=5, n_min=8,
                       fichier="cp_heatmap.html"):
    cible = 1 - alpha
    d = variantes[nom].copy()
    d["strate"] = pd.qcut(d["y_obs"], n_strates,
                          labels=[f"S{i+1}" for i in range(n_strates)], duplicates="drop")
    g = (d.groupby(["segment","strate"], observed=True)["couvert"]
         .agg(cov="mean", n="size").reset_index())
    g = g[g["n"] >= n_min]
    mat = g.pivot(index="segment", columns="strate", values="cov")
    cnt = g.pivot(index="segment", columns="strate", values="n")
    mat = mat.loc[mat.mean(axis=1).sort_values().index]

    fig = px.imshow(mat, color_continuous_scale="RdYlGn",
                    zmin=cible-0.25, zmax=cible+0.06, aspect="auto",
                    labels=dict(x="Strate de magnitude", y="Segment", color="Couverture"))
    fig.update_traces(
        customdata=cnt.loc[mat.index].values,
        hovertemplate="Segment : %{y}<br>Strate : %{x}<br>"
                      "Couverture : %{z:.1%}<br>n : %{customdata:,.0f}<extra></extra>")
    fig.update_layout(
        title=f"{nom} -- couverture croisee segment x magnitude"
              f"<br><sub>Rouge = sous-couverture. Cible {cible:.0%}.</sub>",
        coloraxis_colorbar=dict(tickformat=".0%"),
        template="plotly_white", height=680)
    fig.write_html(fichier); fig.show()
    print(f"Sauvegarde : {fichier}")

interactif_heatmap(variantes, alpha=ALPHA)
