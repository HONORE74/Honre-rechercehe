def conformal_score_normalise(y_calib, y_lo_calib, y_hi_calib,
                              y_lo_test, y_hi_test, y_test,
                              n_bins=5, alpha=0.10, n_min_bin=100, par_groupe=True):
    """Score RELATIF : l'ecart est divise par la largeur de l'intervalle.
    Le Q_hat devient un MULTIPLICATEUR, pas une constante en euros."""
    larg_calib = np.maximum(y_hi_calib - y_lo_calib, 1e-6)
    larg_test  = np.maximum(y_hi_test  - y_lo_test,  1e-6)

    # Score normalise : sans unite, comparable entre petits et gros contrats
    scores = np.maximum(y_lo_calib - y_calib, y_calib - y_hi_calib) / larg_calib

    def qhat(s, a=alpha):
        n = len(s)
        return float(np.quantile(s, min(np.ceil((n+1)*(1-a))/n, 1.0), method="higher"))

    q_glob = qhat(scores)
    print(f"Q_hat global normalise : {q_glob:.4f}  "
          f"(= elargissement de {q_glob*100:.1f} % de la largeur de chaque intervalle)")

    if not par_groupe:
        lo = np.clip(y_lo_test - q_glob*larg_test, 0, None)
        hi = y_hi_test + q_glob*larg_test
        table = None
    else:
        bornes = np.unique(np.quantile(larg_calib, np.linspace(0, 1, n_bins+1)))
        bornes[0], bornes[-1] = -np.inf, np.inf
        gc = pd.cut(larg_calib, bins=bornes, labels=False, include_lowest=True)
        gt = pd.cut(larg_test,  bins=bornes, labels=False, include_lowest=True)

        lignes = []
        for g in range(len(bornes)-1):
            m = gc == g; n_g = int(m.sum())
            q_g = q_glob if n_g < n_min_bin else qhat(scores[m])
            lignes.append({"id_groupe": g, "n_calib": n_g, "Q_hat_norm": q_g})
        table = pd.DataFrame(lignes)
        print("\n", table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

        q_ind = table.set_index("id_groupe")["Q_hat_norm"].reindex(gt).values
        lo = np.clip(y_lo_test - q_ind*larg_test, 0, None)
        hi = y_hi_test + q_ind*larg_test

    res = pd.DataFrame({"y_obs": np.asarray(y_test, float),
                        "borne_basse": lo, "borne_haute": hi,
                        "largeur_finale": hi - lo,
                        "id_groupe": (gt if par_groupe else 0)})
    res["dans_intervalle"] = (res["y_obs"] >= res["borne_basse"]) & (res["y_obs"] <= res["borne_haute"])
    return res, table


# --- Comparaison des trois approches ---
for nom, params in [("Score ABSOLU, 10 groupes",   dict(n_bins=10, par_groupe=True)),
                    ("Score NORMALISE, global",     dict(par_groupe=False)),
                    ("Score NORMALISE, 5 groupes",  dict(n_bins=5,  par_groupe=True))]:
    print("\n" + "="*70); print(f"  {nom}"); print("="*70)
    if "ABSOLU" in nom:
        r, v = resultats, verif
        cov, amp = r["dans_intervalle"].mean(), v["couverture"].max()-v["couverture"].min()
        larg = r["largeur_finale"].median()
    else:
        r, _ = conformal_score_normalise(y_calib.values, y_lo_calib, y_hi_calib,
                                         y_lo_test, y_hi_test, y_test.values,
                                         alpha=ALPHA, **params)
        g = r.groupby("id_groupe")["dans_intervalle"].mean()
        cov, amp, larg = r["dans_intervalle"].mean(), g.max()-g.min(), r["largeur_finale"].median()
    print(f"  Couverture : {cov:.2%}   Amplitude : {amp:.2%}   Largeur mediane : {larg:,.0f}")



























import matplotlib.pyplot as plt

COULEUR_OK, COULEUR_ALERTE = "#0ca30c", "#d03b3b"
COULEUR_PRIM, COULEUR_ACC  = "#2a78d6", "#eb6834"
COULEUR_INK, COULEUR_GRILLE = "#0b0b0b", "#e1e0d9"


def visualiser_conformal_groupe(resultats, table_qhat, verif, calib_scores,
                                q_global, alpha=0.10, fichier="conformal_par_groupe.png"):
    cible = 1 - alpha
    fig, ax = plt.subplots(2, 2, figsize=(16, 11))

    # ---- (1) Scores conformes avec TOUS les Q_hat, un par groupe ----
    p99 = np.percentile(calib_scores, 99)
    ax[0,0].hist(calib_scores[calib_scores <= p99], bins=50, color=COULEUR_PRIM,
                 alpha=0.6, edgecolor="white", density=True)
    palette = plt.cm.viridis(np.linspace(0, 0.9, len(table_qhat)))
    for (_, r), c in zip(table_qhat.iterrows(), palette):
        if r["Q_hat"] <= p99:
            ax[0,0].axvline(r["Q_hat"], color=c, lw=1.8, alpha=0.85)
    ax[0,0].axvline(q_global, color=COULEUR_ACC, ls="--", lw=2.6,
                    label=f"Q_hat global = {q_global:,.0f}")
    ax[0,0].set_xlabel("Score conforme (calibration)")
    ax[0,0].set_ylabel("Densite")
    ax[0,0].set_title(f"1. Un seuil par groupe, au lieu d'un seul\n"
                      f"{len(table_qhat)} lignes colorees = les {len(table_qhat)} Q_hat",
                      fontsize=11, fontweight="bold")
    ax[0,0].legend(fontsize=9)

    # ---- (2) Q_hat en fonction du groupe : la correction croit-elle ? ----
    ax[0,1].plot(table_qhat["id_groupe"], table_qhat["Q_hat"], marker="o",
                 color=COULEUR_PRIM, lw=2.2, markersize=9, label="Q_hat par groupe")
    ax[0,1].axhline(q_global, color=COULEUR_ACC, ls="--", lw=2,
                    label=f"Q_hat global = {q_global:,.0f}")
    ax[0,1].fill_between(table_qhat["id_groupe"], 0, table_qhat["Q_hat"],
                         color=COULEUR_PRIM, alpha=0.12)
    ax[0,1].set_xlabel("Groupe (0 = fourchettes etroites -> 9 = fourchettes larges)")
    ax[0,1].set_ylabel("Q_hat (correction appliquee)")
    ax[0,1].set_title("2. La correction s'adapte-t-elle a la difficulte ?\n"
                      "Une courbe croissante = comportement attendu",
                      fontsize=11, fontweight="bold")
    ax[0,1].legend(fontsize=9)
    for _, r in table_qhat.iterrows():
        ax[0,1].annotate(f"{r['Q_hat']:,.0f}", (r["id_groupe"], r["Q_hat"]),
                         fontsize=7, xytext=(0, 7), textcoords="offset points",
                         ha="center")

    # ---- (3) Couverture par groupe : le test qui valide la methode ----
    coul = [COULEUR_ALERTE if abs(e) > 0.05 else COULEUR_OK for e in verif["ecart_cible"]]
    ax[1,0].bar(verif["id_groupe"], verif["couverture"], color=coul)
    ax[1,0].axhline(cible, color=COULEUR_INK, ls="--", lw=2, label=f"Cible {cible:.0%}")
    ax[1,0].axhspan(cible-0.05, cible+0.05, color=COULEUR_OK, alpha=0.10,
                    label="Tolerance +/- 5 pts")
    ax[1,0].set_xlabel("Groupe"); ax[1,0].set_ylabel("Couverture observee")
    ax[1,0].set_ylim(0, 1.05)
    amp = verif["couverture"].max() - verif["couverture"].min()
    ax[1,0].set_title(f"3. Couverture par groupe -- amplitude {amp:.2%}\n"
                      "Toutes les barres proches de la ligne = methode reussie",
                      fontsize=11, fontweight="bold")
    ax[1,0].legend(fontsize=9)
    for _, r in verif.iterrows():
        ax[1,0].text(r["id_groupe"], r["couverture"],
                     f"{r['couverture']:.0%}\nn={int(r['n'])}",
                     ha="center", va="bottom", fontsize=7)

    # ---- (4) Largeur avant / apres correction, par groupe ----
    x = verif["id_groupe"]; w = 0.38
    ax[1,1].bar(x - w/2, verif["largeur_brute_med"], w, color="#9ec5f4",
                label="Avant correction (q95 - q05)")
    ax[1,1].bar(x + w/2, verif["largeur_finale_med"], w, color=COULEUR_PRIM,
                label="Apres correction")
    ax[1,1].set_xlabel("Groupe"); ax[1,1].set_ylabel("Largeur mediane")
    ax[1,1].set_title("4. Le prix a payer : elargissement par groupe\n"
                      "L'ecart montre ou la correction agit le plus",
                      fontsize=11, fontweight="bold")
    ax[1,1].legend(fontsize=9)

    for a in ax.ravel():
        a.grid(axis="y", ls=":", alpha=0.4, color=COULEUR_GRILLE)
        a.set_axisbelow(True)
        for s in ("top","right"): a.spines[s].set_visible(False)

    plt.tight_layout(); plt.savefig(fichier, dpi=220, bbox_inches="tight"); plt.show()

    print(f"\n  Elargissement median global : "
          f"{(verif['largeur_finale_med']/verif['largeur_brute_med']).median():.2f}x")


visualiser_conformal_groupe(resultats, table_qhat, verif, calib_scores, q_global, ALPHA)
