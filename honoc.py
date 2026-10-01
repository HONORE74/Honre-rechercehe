import os

def exporter_pdf(fig, nom, dossier="figures", largeur_cm=16, hauteur_cm=None):
    """Enregistre une figure matplotlib ou plotly en PDF, a la largeur du texte LaTeX."""
    os.makedirs(dossier, exist_ok=True)
    chemin = os.path.join(dossier, nom + ".pdf")
    if type(fig).__module__.startswith("plotly"):
        import plotly.io as pio
        try:                                    # supprime le cadre "Loading [MathJax]"
            if pio.kaleido.scope.mathjax:
                pio.kaleido.scope.mathjax = None
        except Exception:
            pass
        px = lambda cm: round(cm / 2.54 * 96)
        hauteur = px(hauteur_cm) if hauteur_cm else (fig.layout.height or px(largeur_cm * 0.62))
        fig.write_image(chemin, width=px(largeur_cm), height=hauteur)
    else:
        fig = getattr(fig, "figure", fig)       # accepte aussi un ax
        fig.set_size_inches(largeur_cm / 2.54, (hauteur_cm or largeur_cm * 0.62) / 2.54)
        fig.savefig(chemin, bbox_inches="tight")
    print("PDF enregistre :", chemin)
    return chemin












exporter_pdf(fig, "fig_03_couverture")                       # matplotlib ou plotly
exporter_pdf(fig_forest, "dash_forest", "figures_dashboard") # une figure du tableau de bord
