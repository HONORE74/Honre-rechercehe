import numpy as np, pandas as pd

def conformal_par_largeur(df_calib, y_calib, y_lo_calib, y_hi_calib,
                          df_test, y_lo_test, y_hi_test, y_test,
                          n_bins=10, alpha=0.10, n_min_bin=30):
    """Les 5 etapes de la methode du tuteur, avec jointure explicite."""

    # ---------- ETAPE 1 : largeur brute, sans conformalisation ----------
    largeur_calib = y_hi_calib - y_lo_calib
    largeur_test  = y_hi_test  - y_lo_test

    # ---------- ETAPE 2 : bornes des groupes, definies sur la CALIBRATION ----------
    bornes = np.unique(np.quantile(largeur_calib, np.linspace(0, 1, n_bins + 1)))
    bornes[0], bornes[-1] = -np.inf, np.inf     # capture toute valeur extreme du test
    n_reel = len(bornes) - 1

    # <<< C'EST ICI que chaque ligne recoit son id_groupe >>>
    id_groupe_calib = pd.cut(largeur_calib, bins=bornes, labels=False, include_lowest=True)
    id_groupe_test  = pd.cut(largeur_test,  bins=bornes, labels=False, include_lowest=True)

    # ---------- ETAPE 3 : un Q_hat par groupe ----------
    scores = np.maximum(y_lo_calib - y_calib, y_calib - y_hi_calib)
    n_tot  = len(scores)
    q_global = float(np.quantile(scores,
                     min(np.ceil((n_tot+1)*(1-alpha))/n_tot, 1.0), method="higher"))

    lignes = []
    for g in range(n_reel):
        m = id_groupe_calib == g
        n_g = int(m.sum())
        if n_g < n_min_bin:
            q_g, statut = q_global, "repli global"
        else:
            niveau = min(np.ceil((n_g + 1) * (1 - alpha)) / n_g, 1.0)
            q_g, statut = float(np.quantile(scores[m], niveau, method="higher")), "propre"
        lignes.append({"id_groupe": g,
                       "largeur_min": bornes[g], "largeur_max": bornes[g+1],
                       "n_calib": n_g, "Q_hat": q_g, "statut": statut})
    table_qhat = pd.DataFrame(lignes)

    print("=" * 88); print("  ETAPE 3 -- UN Q_hat PAR GROUPE"); print("=" * 88)
    print(table_qhat.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print(f"\n  Q_hat global (reference) : {q_global:,.2f}")
    print(f"  Rapport max/min entre groupes : "
          f"{table_qhat['Q_hat'].max()/max(table_qhat['Q_hat'].min(), 1e-9):,.1f}x")

    # ---------- ETAPE 4 : JOINTURE ----------
    res = pd.DataFrame({"id_groupe": id_groupe_test,
                        "y_obs": np.asarray(y_test, float),
                        "q05": y_lo_test, "q95": y_hi_test,
                        "largeur_brute": largeur_test})
    n_avant = len(res)
    res = res.merge(table_qhat[["id_groupe", "Q_hat"]], on="id_groupe",
                    how="left", validate="many_to_one")

    print(f"\n  ETAPE 4 -- JOINTURE")
    print(f"    Lignes avant / apres : {n_avant:,} / {len(res):,}  "
          f"{'OK' if len(res)==n_avant else 'DUPLICATION !'}")
    print(f"    Lignes sans Q_hat    : {res['Q_hat'].isna().sum():,}")
    print(f"    Q_hat distincts      : {res['Q_hat'].nunique()}")

    # ---------- ETAPE 5 : bornes finales avec le Q_hat INDIVIDUEL ----------
    res["borne_basse"] = np.clip(res["q05"] - res["Q_hat"], 0, None)
    res["borne_haute"] = res["q95"] + res["Q_hat"]
    res["largeur_finale"] = res["borne_haute"] - res["borne_basse"]
    res["dans_intervalle"] = ((res["y_obs"] >= res["borne_basse"]) &
                              (res["y_obs"] <= res["borne_haute"]))

    verif = (res.groupby("id_groupe")
             .agg(n=("y_obs","size"), Q_hat=("Q_hat","first"),
                  couverture=("dans_intervalle","mean"),
                  largeur_brute_med=("largeur_brute","median"),
                  largeur_finale_med=("largeur_finale","median")).reset_index())
    verif["ecart_cible"] = verif["couverture"] - (1 - alpha)

    print("\n" + "=" * 88); print("  VERIFICATION PAR GROUPE"); print("=" * 88)
    print(verif.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print(f"\n  Couverture globale      : {res['dans_intervalle'].mean():.2%}  (cible {1-alpha:.0%})")
    print(f"  Amplitude entre groupes : {verif['couverture'].max()-verif['couverture'].min():.2%}")
    print("=" * 88)
    return res, table_qhat, bornes, verif, q_global


resultats, table_qhat, bornes, verif, q_global = conformal_par_largeur(
    df_calib, y_calib.values, y_lo_calib, y_hi_calib,
    df_test, y_lo_test, y_hi_test, y_test.values,
    n_bins=10, alpha=ALPHA)



































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
