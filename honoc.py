# ================== EXPORT D'IMAGES : a coller en haut du notebook ==================
import io, os, base64
from IPython.display import HTML, display

DOSSIER_IMAGES = "images"   # copie des images matplotlib dans Domino
LARGEUR_CM = 16             # largeur de l'image dans le memoire

try:   # tous les graphiques plotly : l'appareil photo telecharge une image 4x plus nette
    import plotly.io as pio
    for _r in list(pio.renderers):
        pio.renderers[_r].config = {"toImageButtonOptions": {"format": "png", "scale": 4}}
except Exception:
    pass


def enregistrer(fig, nom, largeur_cm=LARGEUR_CM, hauteur_cm=None):
    """Image PNG grande et nette, telechargee directement sur votre PC par le
    navigateur (ni kaleido, ni PDF). Plotly : cliquez sur l'appareil photo du
    graphique affiche. Matplotlib : cliquez sur le lien affiche."""
    px = round(largeur_cm / 2.54 * 96)              # 16 cm -> 605 px : texte ~9 pt
    if type(fig).__module__.startswith("plotly"):
        import plotly.graph_objects as go
        f = go.Figure(fig)                          # copie : l'original ne bouge pas
        haut = round(hauteur_cm / 2.54 * 96) if hauteur_cm else (f.layout.height or round(px * .62))
        f.update_layout(width=px, height=haut)
        f.show(config={"displaylogo": False, "toImageButtonOptions": {
            "format": "png", "filename": nom, "width": px, "height": haut, "scale": 4}})
        print(f"-> cliquez sur l'appareil photo (en haut a droite du graphique) : {nom}.png")
    else:
        fig = getattr(fig, "figure", fig)            # accepte aussi un ax
        fig.set_size_inches(largeur_cm / 2.54, (hauteur_cm or largeur_cm * .62) / 2.54)
        tampon = io.BytesIO()
        fig.savefig(tampon, format="png", dpi=300, bbox_inches="tight", facecolor="white")
        os.makedirs(DOSSIER_IMAGES, exist_ok=True)
        with open(os.path.join(DOSSIER_IMAGES, nom + ".png"), "wb") as fichier:
            fichier.write(tampon.getvalue())
        b64 = base64.b64encode(tampon.getvalue()).decode()
        display(HTML(f'<a download="{nom}.png" href="data:image/png;base64,{b64}" '
                     f'style="font-size:16px">&#11015; Telecharger {nom}.png</a>'))





















enregistrer(fig, "fig_03_couverture")
enregistrer(_fig_forest(table, titre, axes, TARGET, 12), "dash_forest")
