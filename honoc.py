# ===== EXPORT DU TABLEAU DE BORD EN IMAGES : a coller n'importe ou APRES  app = Dash(...) =====
# Ajoute 3 boutons en haut a droite du tableau de bord. Tout se passe dans VOTRE
# navigateur : les fichiers arrivent dans vos Telechargements (rien a installer).
_EXPORT_JS = r"""
<div id="export-barre" style="position:fixed;top:8px;right:8px;z-index:99999;display:flex;
     gap:6px;align-items:center;font:13px Arial;background:#fff;padding:5px 7px;
     border:1px solid #ccc;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.15)">
  <button onclick="exporterPage()">Page entiere (PNG)</button>
  <button onclick="exporterPDF()">Page entiere (PDF)</button>
  <button onclick="exporterGraphes()">Chaque graphique (PNG)</button>
  <span id="export-msg" style="color:#555"></span>
</div>
<script>
(function () {
  const NETTETE = 2;                                   // 2 = deux fois plus net que l'ecran
  const msg = t => { document.getElementById('export-msg').textContent = t; };
  const pause = ms => new Promise(r => setTimeout(r, ms));
  const graphes = () => [...document.querySelectorAll('.js-plotly-plot')]
                          .filter(g => g.offsetWidth > 0 && g.offsetHeight > 0);
  function telecharger(url, nom) {
    const a = document.createElement('a'); a.href = url; a.download = nom;
    document.body.appendChild(a); a.click(); a.remove();
  }
  function chargerHtml2canvas() {
    return new Promise((ok, ko) => {
      if (window.html2canvas) return ok();
      const s = document.createElement('script');
      s.src = 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js';
      s.onload = ok; s.onerror = ko; document.head.appendChild(s); setTimeout(ko, 15000);
    });
  }
  async function graphesEnImages() {                   // rendu exact de plotly
    const remis = [];
    for (const g of graphes()) {
      const w = g.offsetWidth, h = g.offsetHeight;
      const img = new Image();
      img.src = await Plotly.toImage(g, {format: 'png', width: w, height: h, scale: NETTETE});
      await img.decode();
      img.style.cssText = `width:${w}px;height:${h}px;display:block`;
      g.parentNode.insertBefore(img, g); g.style.display = 'none'; remis.push([g, img]);
    }
    return {imgs: remis.map(r => r[1]),
            remettre: () => remis.forEach(([g, img]) => { img.remove(); g.style.display = ''; })};
  }
  window.capturerPage = async function () {
    window.scrollTo(0, 0);
    const H = document.documentElement.scrollHeight, W = document.documentElement.scrollWidth;
    const echelle = Math.max(1, Math.min(NETTETE, 30000 / H, 16000 / W));
    const {imgs, remettre} = await graphesEnImages();
    try {
      try {
        await chargerHtml2canvas();
        return await html2canvas(document.body, {scale: echelle, backgroundColor: '#ffffff',
          windowWidth: W, windowHeight: H, logging: false,
          ignoreElements: el => el.id === 'export-barre'});
      } catch (e) {                                    // sans internet : graphiques empiles
        const larg = Math.max(...imgs.map(i => i.naturalWidth));
        const c = document.createElement('canvas');
        c.width = larg; c.height = imgs.reduce((s, i) => s + i.naturalHeight + 30, 0);
        const ctx = c.getContext('2d'); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, c.width, c.height);
        let y = 0; for (const i of imgs) { ctx.drawImage(i, 0, y); y += i.naturalHeight + 30; }
        c.seulementGraphiques = true; return c;
      }
    } finally { remettre(); }
  };
  window.exporterPage = async function () {
    msg('capture en cours...');
    try {
      const c = await capturerPage();
      c.toBlob(b => telecharger(URL.createObjectURL(b), 'tableau_de_bord_complet.png'));
      msg(c.seulementGraphiques ? 'fait (graphiques seuls : bibliotheque de capture inaccessible)' : 'fait');
    } catch (e) { msg('erreur : ' + e); }
  };
  window.exporterGraphes = async function () {
    const gs = graphes();
    for (let i = 0; i < gs.length; i++) {
      msg(`graphique ${i + 1} / ${gs.length}`);
      const nom = (gs[i].closest('.dash-graph') || gs[i]).id || 'graphique';
      await Plotly.downloadImage(gs[i], {format: 'png', scale: 3, width: gs[i].offsetWidth,
        height: gs[i].offsetHeight, filename: `tdb_${String(i + 1).padStart(2, '0')}_${nom}`});
      await pause(500);
    }
    msg(`fait : ${gs.length} graphiques`);
  };
  window.preparerPDF = function () {                   // une seule longue page
    window.scrollTo(0, 0);
    const vh = [...document.querySelectorAll('[style*="vh"]')]   // 100vh = hauteur de PAGE a
      .filter(el => el.style.minHeight.includes('vh'));          // l'impression : a neutraliser
    vh.forEach(el => { el.dataset.minh = el.style.minHeight; el.style.minHeight = '0'; });
    window.addEventListener('afterprint', () =>
      vh.forEach(el => { el.style.minHeight = el.dataset.minh; }), {once: true});
    const W = document.documentElement.scrollWidth, H = document.documentElement.scrollHeight;
    let st = document.getElementById('export-page');
    if (!st) { st = document.createElement('style'); st.id = 'export-page'; document.head.appendChild(st); }
    st.textContent = `@page { size: ${W}px ${Math.ceil(H * 1.01) + 150}px; margin: 0 }
      @media print { #export-barre { display: none !important }
      html, body { height: auto !important; overflow: hidden !important;
                   -webkit-print-color-adjust: exact; print-color-adjust: exact } }`;
  };
  window.exporterPDF = function () {
    preparerPDF(); msg('choisissez « Enregistrer au format PDF »'); window.print();
  };
})();
</script>
"""
if not getattr(app, "_export_ajoute", False):      # ajoute a chaque page servie, meme si
    _interp = app.interpolate_index                  # le gabarit HTML change ensuite
    app.interpolate_index = lambda **k: _interp(**k).replace("</body>", _EXPORT_JS + "</body>", 1)
    app._export_ajoute = True
