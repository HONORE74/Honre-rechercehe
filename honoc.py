# ============ MASQUER LES VALEURS DANS TOUS LES GRAPHIQUES : en haut du notebook ============
# Agit seulement a l'affichage et a l'enregistrement : les donnees et les calculs ne
# changent pas. Pour revoir les valeurs : MASQUER = False (sans relancer cette cellule).
import re, numbers

MASQUER = True
_CHIFFRE = re.compile(r"\d")
_AVEC_ECHELLE = {"heatmap", "contour", "histogram2d", "histogram2dcontour", "surface",
                 "choropleth", "densitymapbox", "mesh3d", "cone", "streamtube", "volume"}


def _numerique(v):
    if v is None:
        return False
    if isinstance(v, dict):                          # tableau encode (plotly >= 6)
        return str(v.get("dtype", "x"))[:1] in "fiu"
    if hasattr(v, "dtype"):
        return v.dtype.kind in "fiu"
    v = [x for x in list(v)[:50] if x is not None]
    return bool(v) and all(isinstance(x, numbers.Number) and not isinstance(x, bool) for x in v)


def _masquer_plotly(d):
    lay = d.setdefault("layout", {})
    num = {}                                         # axe -> porte-t-il des valeurs ?
    for t in d.get("data", []):
        t.pop("text", None); t.pop("hovertext", None); t.pop("hovertemplate", None)
        t["hoverinfo"] = "none"                      # plus d'infobulle (le clic marche encore)
        if t.get("type") in ("pie", "sunburst", "treemap", "icicle", "funnelarea"):
            t["textinfo"], t["texttemplate"] = "label", "%{label}"   # noms, sans % ni montant
        else:
            t.pop("texttemplate", None)
            if "text" in str(t.get("mode", "")):
                t["mode"] = "+".join(m for m in t["mode"].split("+") if m != "text") or "markers"
        if t.get("type") in _AVEC_ECHELLE:            # echelle de couleur (score, niveau...)
            t["showscale"] = False
        m = t.get("marker")
        if isinstance(m, dict) and any(k in m for k in ("colorscale", "colorbar", "cmin", "showscale")):
            m["showscale"] = False
        for a in ("x", "y"):
            cle = a + "axis" + str(t.get(a + "axis", a))[1:]
            num[cle] = num.get(cle, False) or _numerique(t.get(a))
    for cle, ax in list(lay.items()):
        if cle.startswith("coloraxis") and isinstance(ax, dict):
            ax["showscale"] = False
        if not re.fullmatch(r"[xy]axis\d*", cle) or not isinstance(ax, dict) or ax.get("ticktext") is not None:
            continue                                 # libelles poses a la main = des noms
        if ax.get("type") in ("linear", "log") or (ax.get("type") is None and num.get(cle)):
            ax["showticklabels"] = False
    for cle in num:
        if num[cle] and cle not in lay:              # axe par defaut, absent du layout
            lay[cle] = {"showticklabels": False}
    for a in lay.get("annotations", []) or []:
        if _CHIFFRE.search(str(a.get("text", ""))):
            a["visible"] = False
    return d


try:   # plotly : notebook, enregistrement ET tableau de bord Dash passent par to_dict()
    import plotly.basedatatypes as _pb
    if not hasattr(_pb.BaseFigure, "_to_dict_brut"):
        _pb.BaseFigure._to_dict_brut = _pb.BaseFigure.to_dict
    _pb.BaseFigure.to_dict = lambda self: (_masquer_plotly(self._to_dict_brut())
                                           if MASQUER else self._to_dict_brut())
except ImportError:
    pass


def _masquer_mpl(fig):
    from matplotlib.ticker import NullFormatter, FixedFormatter
    for ax in fig.axes:
        for axe in (ax.xaxis, ax.yaxis):
            conv = axe.get_converter() if hasattr(axe, "get_converter") else axe.converter
            f = axe.get_major_formatter()                  # libelles poses a la main ?
            a_la_main = isinstance(f, FixedFormatter) or getattr(
                getattr(getattr(f, "func", None), "func", None), "__name__", "") == "_format_with_dict"
            if conv is None and not a_la_main:
                axe.set_major_formatter(NullFormatter())   # valeurs, pas noms ni dates
                axe.set_minor_formatter(NullFormatter())
        for t in ax.texts:                                 # etiquettes chiffrees
            if _CHIFFRE.search(t.get_text()):
                t.set_visible(False)


try:   # matplotlib : chaque dessin (affichage, savefig) passe par Figure.draw
    import matplotlib.figure as _mf
    if not hasattr(_mf.Figure, "_draw_brut"):
        _mf.Figure._draw_brut = _mf.Figure.draw
    def _draw_masque(self, renderer):
        if MASQUER:
            _masquer_mpl(self)
        return self._draw_brut(renderer)
    _mf.Figure.draw = _draw_masque
except ImportError:
    pass
