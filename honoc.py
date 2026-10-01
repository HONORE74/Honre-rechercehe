# ============ STANDARDISATION DES COLONNES : a coller en haut du notebook ============
# Remplace les vraies valeurs par des valeurs sans unite : les courbes, les ecarts et
# le classement restent les memes, mais aucun montant reel n'apparait.
# A appliquer sur des COPIES, juste avant les graphiques / le tableau de bord,
# jamais avant le filtre des sinistres, l'entrainement du modele ou le calcul du score.
import pandas as pd

CIBLE = "Claims_incurred"
CENTRER = False    # False : x / ecart-type  (conserve le 0, les signes et le score)
                   # True  : (x - moyenne) / ecart-type  (standardisation "complete")
NE_PAS_TOUCHER = {"year", "quarter", "annee", "time_idx", "rank", "p_value",
                  "A_ecart_borne", "B_erreur_modele", "score_prediction"}
MEME_UNITE = {"y_obs", "y_pred", "borne_basse", "borne_haute", "y_obs_df"}
ECARTS = {"largeur_intervalle", "ecart_intervalle", "score_nonconformite", "score_composite"}


def parametres_standardisation(ref, cible=CIBLE):
    """Moyenne et ecart-type de chaque colonne numerique, calcules UNE fois sur la
    base de reference (df_model), puis reutilises pour toutes les tables."""
    if cible not in ref.columns:
        raise KeyError(f"La cible '{cible}' est absente de la base de reference.")
    p = {}
    for c in ref.select_dtypes("number").columns:
        s = ref[c].std()
        p[c] = (ref[c].mean(), s if s and s == s else 1.0)
    return p


def standardiser(df, params, cible=CIBLE, centrer=None):
    """Copie de `df` standardisee. Toutes les colonnes exprimees en montant de la
    cible (observe, prediction, bornes, lags, avg_dec...) utilisent la MEME echelle,
    celle de la cible : l'observe reste dedans ou dehors de l'intervalle, comme avant.
    Identifiants, temps, rangs, p-values et 0/1 ne sont pas modifies."""
    centrer = CENTRER if centrer is None else centrer
    m_c, s_c = params[cible]
    out = df.copy()
    for c in out.select_dtypes("number").columns:
        if c in NE_PAS_TOUCHER:
            continue
        if c in ECARTS:                                   # differences : jamais centrees
            out[c] = out[c] / s_c
            continue
        m, s = (m_c, s_c) if (c in MEME_UNITE or cible in c) else params.get(c, (None, None))
        if s is None:                                     # colonne inconnue de la reference
            m, s = out[c].mean(), (out[c].std() or 1.0)
        out[c] = (out[c] - (m if centrer else 0)) / s
    return out
