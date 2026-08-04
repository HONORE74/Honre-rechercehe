def conformal_conditionnel(df_calib, y_calib, y_lo_calib, y_hi_calib,
                           df_test, y_lo_test, y_hi_test,
                           segment_col, alpha, n_min=50):
    """Le Q_hat n'est plus global : il est calcule DANS chaque segment."""
    scores = np.maximum(y_lo_calib - y_calib, y_calib - y_hi_calib)

    n_g = len(scores)
    Q_global = np.quantile(scores, min(np.ceil((n_g+1)*(1-alpha))/n_g, 1.0), method="higher")

    seg_cal = df_calib[segment_col].astype(str).values
    Q_par_seg = {}
    for s in np.unique(seg_cal):
        m = seg_cal == s
        n = int(m.sum())
        if n < n_min:
            Q_par_seg[s] = (Q_global, n, "repli global (n trop faible)")
        else:
            q_lvl = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
            Q_par_seg[s] = (float(np.quantile(scores[m], q_lvl, method="higher")), n, "conditionnel")

    seg_test = df_test[segment_col].astype(str).values
    Q_test = np.array([Q_par_seg.get(s, (Q_global, 0, "absent calib"))[0] for s in seg_test])

    print("=" * 76)
    print(f"  Q_hat CONDITIONNEL par '{segment_col}'   (global de reference : {Q_global:,.0f})")
    print(f"\n  {'segment':<26}{'n_calib':>9}{'Q_hat':>15}   statut")
    for s, (q, n, st) in sorted(Q_par_seg.items(), key=lambda kv: -kv[1][0]):
        ratio = q / Q_global
        print(f"  {s:<26}{n:>9}{q:>15,.0f}  (x{ratio:.2f})  {st}")
    print("=" * 76)

    return np.clip(y_lo_test - Q_test, 0, None), y_hi_test + Q_test, Q_par_seg, Q_global







import lightgbm as lgb
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

def tester_conditionnalite(X_test, results, alpha, n_splits=5):
    """Test formel de couverture conditionnelle, sans segmentation arbitraire.
    Principe : on essaie de PREDIRE la non-couverture a partir de X.
    Si c'est possible (AUC > 0.5), la couverture n'est PAS conditionnelle."""
    y_miss = (~results["dans_intervalle"]).astype(int).values

    if y_miss.sum() < 20:
        print("Trop peu de non-couvertures pour tester."); return None

    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             min_child_samples=40, reg_lambda=5.0,
                             random_state=42, n_jobs=-1, verbose=-1)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    proba = cross_val_predict(clf, X_test, y_miss, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y_miss, proba)

    # Test de permutation : l'AUC est-il significativement > 0.5 ?
    rng = np.random.default_rng(42)
    aucs_h0 = [roc_auc_score(rng.permutation(y_miss), proba) for _ in range(200)]
    p_val = np.mean(np.array(aucs_h0) >= auc)

    print("=" * 76)
    print("  TEST DE COUVERTURE CONDITIONNELLE (sans segmentation arbitraire)")
    print("=" * 76)
    print(f"  Taux de non-couverture : {y_miss.mean():.2%}  (attendu {alpha:.0%})")
    print(f"  AUC de prediction de la non-couverture : {auc:.4f}")
    print(f"  p-value (permutation)                  : {p_val:.4f}")
    print()
    if auc > 0.60 and p_val < 0.05:
        print("  --> COUVERTURE NON CONDITIONNELLE.")
        print("      La non-couverture est PREDICTIBLE a partir de X : certaines")
        print("      regions de l'espace sont systematiquement mal couvertes.")
    elif auc > 0.55:
        print("  --> Conditionnalite PARTIELLEMENT violee (signal faible mais present).")
    else:
        print("  --> Couverture approximativement conditionnelle : la non-couverture")
        print("      est imprevisible a partir de X, ce qui est le comportement attendu.")

    # Quelles variables expliquent la non-couverture ?
    clf.fit(X_test, y_miss)
    imp = (pd.Series(clf.booster_.feature_importance("gain"), index=X_test.columns)
           .sort_values(ascending=False).head(10))
    print(f"\n  Variables qui expliquent le mieux la non-couverture :")
    print((imp / imp.sum() * 100).round(2).to_string())
    print("=" * 76)
    return {"auc": auc, "p_value": p_val, "importances": imp}

diag_cond = tester_conditionnalite(X_test, results_test, ALPHA)







import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def intervalle_wilson(k, n, z=1.96):
    """Intervalle de Wilson : plus fiable que l'approximation normale
    quand n est petit ou la proportion proche de 0/1."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / d
    demi = (z / d) * np.sqrt(p*(1-p)/n + z**2/(4*n**2))
    return max(0, centre - demi), min(1, centre + demi)


def plot_couverture_conditionnelle(results, segment_col="Lob", alpha=0.10,
                                   n_min=25, max_segments=30, tolerance=0.05,
                                   trier=True, fichier=None):
    """Couverture conditionnelle par segment.
    x = segments | y = couverture | ligne horizontale a 1-alpha
    Chaque point est relie a la ligne cible par une tige : la longueur = l'ecart."""
    cible = 1 - alpha

    t = (results.groupby(segment_col, observed=True)["dans_intervalle"]
         .agg(couverts="sum", n="size").reset_index())
    t = t[t["n"] >= n_min].copy()
    t["couverture"] = t["couverts"] / t["n"]
    t[["ic_bas", "ic_haut"]] = t.apply(
        lambda r: pd.Series(intervalle_wilson(r["couverts"], r["n"])), axis=1)
    t["ecart"] = t["couverture"] - cible

    if trier:
        t = t.sort_values("couverture")
    t = t.head(max_segments).reset_index(drop=True)

    # --- Statut statistique : l'IC contient-il la cible ? ---
    def statut(r):
        if r["ic_haut"] < cible:   return "sous_signif"
        if r["ic_bas"]  > cible:   return "sur_signif"
        if abs(r["ecart"]) <= tolerance: return "conforme"
        return "ecart_non_signif"

    t["statut"] = t.apply(statut, axis=1)

    COULEURS = {"conforme":         "#2E8B57",   # vert  : dans la tolerance
                "ecart_non_signif": "#DAA520",   # ambre : ecart non significatif
                "sous_signif":      "#B22222",   # rouge : sous-couverture prouvee
                "sur_signif":       "#4169E1"}   # bleu  : sur-couverture (gaspillage)
    LABELS = {"conforme":         f"Conforme (ecart <= {tolerance:.0%})",
              "ecart_non_signif": "Ecart non significatif",
              "sous_signif":      "SOUS-couverture significative",
              "sur_signif":       "SUR-couverture significative"}

    fig, ax = plt.subplots(figsize=(max(11, 0.55*len(t)), 7))
    x = np.arange(len(t))

    # --- Bande de tolerance + ligne cible ---
    ax.axhspan(cible - tolerance, cible + tolerance, color="#2E8B57", alpha=0.07, zorder=0)
    ax.axhline(cible, color="black", ls="--", lw=2, zorder=2)
    ax.text(len(t) - 0.4, cible + 0.006, f"Cible {cible:.0%}",
            ha="right", va="bottom", fontsize=11, fontweight="bold")

    # --- Tiges : ecart a la cible ---
    for xi, row in zip(x, t.itertuples()):
        ax.vlines(xi, cible, row.couverture,
                  color=COULEURS[row.statut], lw=2.2, alpha=0.55, zorder=3)

    # --- Barres d'incertitude (Wilson) ---
    ax.vlines(x, t["ic_bas"], t["ic_haut"], color="gray", lw=1.1, alpha=0.75, zorder=4)
    for xi, lo, hi in zip(x, t["ic_bas"], t["ic_haut"]):
        ax.hlines([lo, hi], xi - 0.12, xi + 0.12, color="gray", lw=1.1, alpha=0.75, zorder=4)

    # --- Points : taille proportionnelle a l'effectif ---
    tailles = 60 + 340 * (t["n"] - t["n"].min()) / max(t["n"].max() - t["n"].min(), 1)
    ax.scatter(x, t["couverture"], s=tailles,
               c=[COULEURS[s] for s in t["statut"]],
               edgecolor="white", linewidth=1.6, zorder=6)

    # --- Effectifs sous l'axe ---
    for xi, row in zip(x, t.itertuples()):
        ax.annotate(f"n={row.n}", (xi, 0.005), ha="center", va="bottom",
                    fontsize=7, color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels(t[segment_col].astype(str), rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Taux de couverture observe", fontsize=12)
    ax.set_xlabel(f"Segment  ({segment_col})", fontsize=12)
    ax.set_ylim(0, 1.04)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, 0.1)])
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    glob = results["dans_intervalle"].mean()
    pire = t.loc[t["couverture"].idxmin()]
    ax.set_title(f"Couverture conditionnelle par {segment_col}\n"
                 f"Marginale globale : {glob:.1%}   |   "
                 f"Pire segment : {pire[segment_col]} a {pire['couverture']:.1%}",
                 fontsize=13, fontweight="bold", pad=14)

    presents = [s for s in COULEURS if s in set(t["statut"])]
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COULEURS[s],
                      markersize=11, label=LABELS[s]) for s in presents]
    handles.append(Line2D([0], [0], color="gray", lw=1.2, label="IC 95% (Wilson)"))
    ax.legend(handles=handles, fontsize=9, loc="lower right", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(fichier or f"couverture_conditionnelle_{segment_col}.png", dpi=220)
    plt.show()

    print(f"\n{'='*72}")
    print(f"  Segments analyses (n >= {n_min}) : {len(t)}")
    for st in ["sous_signif", "sur_signif", "ecart_non_signif", "conforme"]:
        c = (t["statut"] == st).sum()
        if c:
            print(f"    {LABELS[st]:<42} : {c}")
    print(f"  Ecart absolu moyen a la cible : {t['ecart'].abs().mean():.2%}")
    print(f"{'='*72}")
    return t

tab_lob = plot_couverture_conditionnelle(results_test, "Lob", ALPHA)



















def plot_multi_segments(results, segment_cols=("Lob","Risk","Activity","Periodicity"),
                        alpha=0.10, n_min=25, max_seg=15, tolerance=0.05):
    cible = 1 - alpha
    cols = [c for c in segment_cols if c in results.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(5.2*len(cols), 6.5), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, col in zip(axes, cols):
        t = (results.groupby(col, observed=True)["dans_intervalle"]
             .agg(k="sum", n="size").reset_index())
        t = t[t["n"] >= n_min].copy()
        if not len(t):
            ax.set_visible(False); continue
        t["cov"] = t["k"] / t["n"]
        t[["lo","hi"]] = t.apply(lambda r: pd.Series(intervalle_wilson(r["k"], r["n"])), axis=1)
        t = t.sort_values("cov").head(max_seg).reset_index(drop=True)
        x = np.arange(len(t))
        coul = ["#B22222" if h < cible else "#4169E1" if l > cible
                else "#2E8B57" if abs(c-cible) <= tolerance else "#DAA520"
                for c, l, h in zip(t["cov"], t["lo"], t["hi"])]

        ax.axhspan(cible-tolerance, cible+tolerance, color="#2E8B57", alpha=0.07)
        ax.axhline(cible, color="black", ls="--", lw=1.8)
        ax.vlines(x, cible, t["cov"], color=coul, lw=2, alpha=0.55)
        ax.vlines(x, t["lo"], t["hi"], color="gray", lw=1, alpha=0.7)
        ax.scatter(x, t["cov"], s=90, c=coul, edgecolor="white", linewidth=1.3, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(t[col].astype(str), rotation=60, ha="right", fontsize=8)
        ax.set_title(f"{col}  ({len(t)} modalites)", fontsize=11, fontweight="bold")
        ax.grid(axis="y", ls=":", alpha=0.4); ax.set_axisbelow(True)
        for s in ("top","right"): ax.spines[s].set_visible(False)

    axes[0].set_ylabel("Couverture observee", fontsize=12)
    axes[0].set_ylim(0, 1.04)
    axes[0].set_yticks(np.arange(0, 1.01, 0.1))
    axes[0].set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, 0.1)])
    fig.suptitle(f"Couverture conditionnelle -- toutes dimensions  "
                 f"(cible {cible:.0%}, marginale {results['dans_intervalle'].mean():.1%})",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig("couverture_multi_segments.png", dpi=220); plt.show()

plot_multi_segments(results_test, alpha=ALPHA)






















def plot_avant_apres(res_marg, res_mond, segment_col="Lob", alpha=0.10,
                     n_min=25, max_seg=20):
    cible = 1 - alpha
    a = res_marg.groupby(segment_col, observed=True)["dans_intervalle"].agg(k="sum", n="size")
    b = res_mond.groupby(segment_col, observed=True)["dans_intervalle"].agg(k="sum", n="size")
    t = a.join(b, lsuffix="_m", rsuffix="_c")
    t = t[t["n_m"] >= n_min].copy()
    t["cov_marg"] = t["k_m"] / t["n_m"]
    t["cov_mond"] = t["k_c"] / t["n_c"]
    t = t.sort_values("cov_marg").head(max_seg).reset_index()
    x = np.arange(len(t))

    fig, ax = plt.subplots(figsize=(max(11, 0.6*len(t)), 7))
    ax.axhspan(cible-0.05, cible+0.05, color="#2E8B57", alpha=0.07)
    ax.axhline(cible, color="black", ls="--", lw=2)
    ax.text(len(t)-0.4, cible+0.006, f"Cible {cible:.0%}", ha="right", fontsize=11,
            fontweight="bold")

    # fleches du marginal vers le Mondrian
    for xi, r in zip(x, t.itertuples()):
        ax.annotate("", xy=(xi, r.cov_mond), xytext=(xi, r.cov_marg),
                    arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.6,
                                    shrinkA=4, shrinkB=4))
    ax.scatter(x, t["cov_marg"], s=130, c="#DAA520", edgecolor="white", lw=1.5,
               zorder=5, label="Avant : Q_hat global (marginal)")
    ax.scatter(x, t["cov_mond"], s=130, c="#2E8B57", edgecolor="white", lw=1.5,
               zorder=5, marker="D", label="Apres : Q_hat par segment (conditionnel)")

    ax.set_xticks(x)
    ax.set_xticklabels(t[segment_col].astype(str), rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Couverture observee", fontsize=12)
    ax.set_xlabel(f"Segment  ({segment_col})", fontsize=12)
    ax.set_ylim(0, 1.04); ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, 0.1)])
    ax.grid(axis="y", ls=":", alpha=0.4); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)

    e_m = (t["cov_marg"] - cible).abs().mean()
    e_c = (t["cov_mond"] - cible).abs().mean()
    ax.set_title(f"Effet du conditionnement du Q_hat sur la couverture\n"
                 f"Ecart absolu moyen : {e_m:.2%} -> {e_c:.2%}  "
                 f"({(1-e_c/e_m)*100:+.0f} %)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.95)
    plt.tight_layout(); plt.savefig("avant_apres_mondrian.png", dpi=220); plt.show()
    return t

# a lancer apres avoir construit results_mond avec le Q_hat conditionnel
comp = plot_avant_apres(results_test, results_mond, "Lob", ALPHA)














import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_funnel(results, segment_cols=("Lob","Risk","Activity","Periodicity","Partner"),
                alpha=0.10, n_min=8):
    """Funnel plot : couverture vs effectif, avec l'entonnoir de variation attendue.
    Hors entonnoir = decrochage REEL. Dans l'entonnoir = bruit d'echantillonnage."""
    cible = 1 - alpha
    pts = []
    for col in [c for c in segment_cols if c in results.columns]:
        g = results.groupby(col, observed=True)["dans_intervalle"].agg(k="sum", n="size")
        for mod, r in g[g["n"] >= n_min].iterrows():
            pts.append({"dim": col, "mod": str(mod)[:22],
                        "n": int(r["n"]), "cov": r["k"]/r["n"]})
    p = pd.DataFrame(pts)

    n_grille = np.linspace(p["n"].min(), p["n"].max(), 400)
    b95 = 1.96 * np.sqrt(cible*(1-cible)/n_grille)
    b99 = 2.58 * np.sqrt(cible*(1-cible)/n_grille)

    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.fill_between(n_grille, cible-b99, cible+b99, color="#B0C4DE", alpha=0.25,
                    label="Entonnoir 99 %")
    ax.fill_between(n_grille, cible-b95, cible+b95, color="#4682B4", alpha=0.22,
                    label="Entonnoir 95 %")
    ax.axhline(cible, color="black", ls="--", lw=2, label=f"Cible {cible:.0%}")

    seuil = 1.96*np.sqrt(cible*(1-cible)/p["n"])
    p["hors"] = (p["cov"] < cible - seuil) | (p["cov"] > cible + seuil)
    p["sens"] = np.where(p["cov"] < cible, "sous", "sur")

    dedans = p[~p["hors"]]
    ax.scatter(dedans["n"], dedans["cov"], s=45, c="#9E9E9E", alpha=0.55,
               edgecolor="white", lw=0.8, zorder=4)
    for sens, coul, lab in [("sous", "#B22222", "Sous-couverture REELLE"),
                            ("sur",  "#4169E1", "Sur-couverture REELLE")]:
        d = p[p["hors"] & (p["sens"] == sens)]
        ax.scatter(d["n"], d["cov"], s=130, c=coul, edgecolor="white", lw=1.5,
                   zorder=6, label=lab)
        for r in d.nlargest(min(6, len(d)), "n").itertuples():
            ax.annotate(f"{r.dim}={r.mod}", (r.n, r.cov), fontsize=7.5,
                        xytext=(7, 0), textcoords="offset points", va="center")

    ax.scatter([], [], s=45, c="#9E9E9E", label="Ecart compatible avec le hasard")
    ax.set_xlabel("Effectif du segment (n)", fontsize=12)
    ax.set_ylabel("Couverture observee", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, 0.1)])
    ax.grid(ls=":", alpha=0.35); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.set_title(f"Funnel plot -- {p['hors'].sum()} segments hors entonnoir sur {len(p)}\n"
                 f"L'entonnoir se resserre quand n augmente : un petit segment PEUT devier sans alerte",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
    plt.tight_layout(); plt.savefig("funnel_couverture.png", dpi=220); plt.show()
    return p.sort_values("cov")

funnel = plot_funnel(results_test, alpha=ALPHA)















from sklearn.tree import DecisionTreeClassifier, plot_tree

def plot_arbre_non_couverture(results, X_test, alpha=0.10, max_depth=3, min_leaf=40):
    """Un arbre apprend OU la couverture echoue. Aucune segmentation choisie a l'avance."""
    y_miss = (~results["dans_intervalle"]).astype(int).values

    X = X_test.copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "category":
            X[c] = X[c].astype("category").cat.codes
    X = X.fillna(-999)

    arbre = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf,
                                   random_state=42)
    arbre.fit(X, y_miss)

    fig, ax = plt.subplots(1, 2, figsize=(19, 8),
                           gridspec_kw={"width_ratios": [1.5, 1]})
    plot_tree(arbre, feature_names=list(X.columns), class_names=["couvert","MANQUE"],
              filled=True, rounded=True, fontsize=7.5, ax=ax[0], impurity=False,
              proportion=True)
    ax[0].set_title("Regions ou la couverture echoue (decouvertes automatiquement)",
                    fontsize=12, fontweight="bold")

    feuille = arbre.apply(X)
    res = (pd.DataFrame({"feuille": feuille, "manque": y_miss})
           .groupby("feuille")["manque"].agg(taux="mean", n="size")
           .sort_values("taux"))
    coul = ["#B22222" if t > alpha*2 else "#DAA520" if t > alpha*1.3 else "#2E8B57"
            for t in res["taux"]]
    ax[1].barh(range(len(res)), res["taux"], color=coul)
    ax[1].axvline(alpha, color="black", ls="--", lw=2, label=f"Taux attendu {alpha:.0%}")
    ax[1].set_yticks(range(len(res)))
    ax[1].set_yticklabels([f"Feuille {f} (n={n})" for f, n in zip(res.index, res["n"])],
                          fontsize=9)
    ax[1].set_xlabel("Taux de NON-couverture", fontsize=11)
    ax[1].set_title("Taux de manque par region decouverte", fontsize=12, fontweight="bold")
    ax[1].legend(fontsize=9)
    for i, (t, n) in enumerate(zip(res["taux"], res["n"])):
        ax[1].text(t + 0.005, i, f"{t:.1%}", va="center", fontsize=8)
    for s in ("top","right"): ax[1].spines[s].set_visible(False)

    plt.tight_layout(); plt.savefig("arbre_non_couverture.png", dpi=220); plt.show()

    print("=" * 76)
    print(f"  Taux de non-couverture global : {y_miss.mean():.2%}  (attendu {alpha:.0%})")
    print(f"  Pire region  : {res['taux'].iloc[-1]:.2%}  (n={res['n'].iloc[-1]})")
    print(f"  Meilleure    : {res['taux'].iloc[0]:.2%}  (n={res['n'].iloc[0]})")
    print(f"  Amplitude    : {res['taux'].iloc[-1] - res['taux'].iloc[0]:.2%}")
    print(f"\n  Si l'amplitude est large, la couverture n'est PAS conditionnelle.")
    print("=" * 76)
    return arbre, res

arbre, feuilles = plot_arbre_non_couverture(results_test, X_test, ALPHA)









def plot_bubble_couverture_largeur(results, segment_col="Lob", alpha=0.10, n_min=25):
    cible = 1 - alpha
    t = (results.groupby(segment_col, observed=True)
         .agg(n=("dans_intervalle","size"), cov=("dans_intervalle","mean"),
              larg=("largeur_intervalle","median"),
              pred=("y_pred","median")).reset_index())
    t = t[t["n"] >= n_min].copy()
    t["larg_rel"] = t["larg"] / t["pred"].abs().clip(lower=1e-6)
    med_rel = t["larg_rel"].median()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axvline(cible, color="black", ls="--", lw=2)
    ax.axhline(med_rel, color="gray", ls=":", lw=1.5)

    quad = {"ideal": ("#2E8B57", "Ideal : couvert et etroit"),
            "conserv": ("#4169E1", "Conservateur : couvert mais large"),
            "danger": ("#B22222", "DANGER : etroit ET mal couvert"),
            "casse": ("#8B0000", "Casse : large ET mal couvert")}
    def q(r):
        ok, etroit = r["cov"] >= cible - 0.02, r["larg_rel"] <= med_rel
        return "ideal" if (ok and etroit) else "conserv" if ok else \
               "danger" if etroit else "casse"
    t["quad"] = t.apply(q, axis=1)

    tailles = 100 + 900 * (t["n"] - t["n"].min()) / max(t["n"].max()-t["n"].min(), 1)
    ax.scatter(t["cov"], t["larg_rel"], s=tailles,
               c=[quad[k][0] for k in t["quad"]], alpha=0.72,
               edgecolor="white", linewidth=1.8, zorder=5)
    for r in t.itertuples():
        ax.annotate(str(getattr(r, segment_col))[:16], (r.cov, r.larg_rel),
                    fontsize=8, ha="center", va="center", zorder=6)

    ax.text(cible+0.005, ax.get_ylim()[1]*0.97, f"Cible {cible:.0%}", fontsize=10,
            fontweight="bold", va="top")
    ax.set_xlabel("Couverture observee", fontsize=12)
    ax.set_ylabel("Largeur relative mediane  (largeur / prediction)", fontsize=12)
    ax.set_xlim(min(0.4, t["cov"].min()-0.05), 1.02)
    ax.grid(ls=":", alpha=0.35); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.set_title(f"Arbitrage couverture / finesse par {segment_col}\n"
                 "Taille du disque = effectif du segment",
                 fontsize=13, fontweight="bold", pad=14)
    handles = [Line2D([0],[0], marker="o", color="w", markerfacecolor=c, markersize=12,
                      label=l) for k,(c,l) in quad.items() if k in set(t["quad"])]
    ax.legend(handles=handles, fontsize=9, loc="upper left", framealpha=0.95)
    plt.tight_layout(); plt.savefig("bubble_couverture_largeur.png", dpi=220); plt.show()
    return t

bulles = plot_bubble_couverture_largeur(results_test, "Lob", ALPHA)








def plot_carte_couverture(results, alpha=0.10, n_bins=6, n_min_cell=15):
    cible = 1 - alpha
    d = results.copy()
    d["bx"] = pd.qcut(d["y_pred"], n_bins, labels=False, duplicates="drop")
    d["by"] = pd.qcut(d["largeur_intervalle"], n_bins, labels=False, duplicates="drop")
    g = d.groupby(["by","bx"], observed=True)["dans_intervalle"].agg(["mean","size"])
    mat = g["mean"].unstack().reindex(index=range(n_bins), columns=range(n_bins))
    cnt = g["size"].unstack().reindex(index=range(n_bins), columns=range(n_bins))

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    m = mat.copy(); m[cnt < n_min_cell] = np.nan
    im = ax.imshow(m.values, cmap="RdYlGn", vmin=cible-0.30, vmax=cible+0.08,
                   origin="lower", aspect="auto")
    for i in range(n_bins):
        for j in range(n_bins):
            v, c = mat.values[i,j], cnt.values[i,j]
            if not np.isnan(v) and c >= n_min_cell:
                ax.text(j, i, f"{v:.0%}\nn={int(c)}", ha="center", va="center",
                        fontsize=8, fontweight="bold" if v < cible-0.08 else "normal")
    ax.set_xticks(range(n_bins)); ax.set_xticklabels([f"D{i+1}" for i in range(n_bins)])
    ax.set_yticks(range(n_bins)); ax.set_yticklabels([f"D{i+1}" for i in range(n_bins)])
    ax.set_xlabel("Decile de PREDICTION (croissante)", fontsize=12)
    ax.set_ylabel("Decile de LARGEUR d'intervalle (croissante)", fontsize=12)
    ax.set_title(f"Carte de couverture conditionnelle 2D  (cible {cible:.0%})\n"
                 "Chaque case = une region de l'espace, sans segmentation metier",
                 fontsize=13, fontweight="bold", pad=14)
    cb = plt.colorbar(im, ax=ax); cb.set_label("Couverture", fontsize=11)
    plt.tight_layout(); plt.savefig("carte_couverture_2d.png", dpi=220); plt.show()

    plat = mat.values[~np.isnan(mat.values)]
    print(f"  Couverture min sur la carte : {plat.min():.1%}")
    print(f"  Couverture max sur la carte : {plat.max():.1%}")
    print(f"  Amplitude : {plat.max()-plat.min():.1%}  "
          f"(si > 20 pts, la couverture n'est clairement pas conditionnelle)")
    return mat, cnt

carte, effectifs = plot_carte_couverture(results_test, ALPHA)



