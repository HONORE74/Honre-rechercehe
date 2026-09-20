import os
chemin_db = f"artefacts_modele/{NOM_ETUDE}.db"
if os.path.exists(chemin_db):
    os.remove(chemin_db)
    print(f"Base supprimée : {chemin_db}")







d = pd.DataFrame({"y": Y, "a": P0, "b": P})
d["r"] = d["b"] / np.maximum(d["a"], 1e-9)
n_groups = 15
d["q"] = pd.qcut(d["r"].rank(method="first"), n_groups, labels=range(1, n_groups + 1))
g = d.groupby("q", observed=True).agg(obs=("y", "mean"), a=("a", "mean"), b=("b", "mean")).reset_index()

fig = go.Figure()
for col, nom, coul, w, dash in [("obs", "Observé", C["ardoise"], 4, None),
                                 ("a", "Avant tuning", C["gris"], 2.5, "dash"),
                                 ("b", "Après tuning", C["corail"], 3, "dot")]:
    fig.add_scatter(x=g["q"].astype(str), y=g[col], name=nom, mode="lines+markers",
                    line=dict(color=coul, width=w, dash=dash), marker=dict(size=9))
fig.update_layout(title=f"<b>Observe vs predit ({n_groups} groupes)</b>",
                  xaxis_title="Groupe (trie par rapport de prediction croissant)",
                  yaxis_title="Valeur moyenne du groupe (IBNR)",
                  template=TPL, height=500,
                  legend=dict(orientation="h", y=1.06, x=.5, xanchor="center"))
fig.show()
