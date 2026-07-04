/* BA Crime Predictor — frontend del MVP.
 * Carga predictions.json (salida de src/train.py) + barrios.geojson y arma:
 *  - choropleth de riesgo por barrio para el (día, franja) seleccionado
 *  - panel de detalle por barrio
 *  - gráfico citywide (historia + forecast)
 *  - tablas de métricas baseline vs modelo
 */

const DOW_SHORT = ["LUN", "MAR", "MIÉ", "JUE", "VIE", "SÁB", "DOM"];
const DOW_FULL = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"];
const RAMP = [[34, 197, 94], [250, 204, 21], [239, 68, 68]]; // verde -> amarillo -> rojo

const state = { dow: 0, band: 0, normalize: true, selected: null };
let DATA = null, GEO = null, METRICS = null, geoLayer = null, map = null;

/* ---------- helpers ---------- */
const $ = (sel) => document.querySelector(sel);

function nowInBuenosAires() {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Argentina/Buenos_Aires",
    weekday: "short", hour: "numeric", hour12: false,
  }).formatToParts(new Date());
  const wd = parts.find(p => p.type === "weekday").value;
  const hour = parseInt(parts.find(p => p.type === "hour").value, 10) % 24;
  const dowMap = { Mon: 0, Tue: 1, Wed: 2, Thu: 3, Fri: 4, Sat: 5, Sun: 6 };
  return { dow: dowMap[wd] ?? 0, band: Math.floor(hour / 4) };
}

function rampColor(t) {
  t = Math.max(0, Math.min(1, t));
  const seg = t < 0.5 ? [RAMP[0], RAMP[1], t * 2] : [RAMP[1], RAMP[2], (t - 0.5) * 2];
  const [a, b, u] = seg;
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * u));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function slotValue(barrio) {
  const b = DATA.barrios[barrio];
  if (!b) return 0;
  const v = b.esperado[state.dow][state.band];
  return state.normalize && b.area_km2 ? v / b.area_km2 : v;
}

function currentMax() {
  const vals = Object.keys(DATA.barrios).map(slotValue).sort((a, b) => a - b);
  return vals[Math.floor(vals.length * 0.95)] || 1; // p95 para que un outlier no lave la escala
}

/* ---------- mapa ---------- */
function initMap() {
  map = L.map("map", { zoomControl: true, attributionControl: false })
    .setView([-34.615, -58.445], 12);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    maxZoom: 18, subdomains: "abcd",
  }).addTo(map);

  geoLayer = L.geoJSON(GEO, {
    style: styleFeature,
    onEachFeature: (feat, layer) => {
      layer.on({
        mouseover: (e) => { e.target.setStyle({ weight: 2.5, color: "#f59e0b" }); e.target.bringToFront(); },
        mouseout: (e) => geoLayer.resetStyle(e.target),
        click: (e) => { state.selected = feat.properties.name; renderDetail(); },
      });
      layer.bindTooltip(() => {
        const name = feat.properties.name;
        const b = DATA.barrios[name];
        if (!b) return `<b>${name}</b>sin datos`;
        const exp = b.esperado[state.dow][state.band];
        const risk = b.riesgo[state.dow][state.band];
        return `<b>${name}</b>` +
          `<span class="tip-val">${exp.toFixed(1)}</span> incidentes esperados / semana<br>` +
          `índice de riesgo: <span class="tip-val">${Math.round(risk)}</span>/100`;
      }, { sticky: true, className: "barrio-tip" });
    },
  }).addTo(map);
}

function styleFeature(feat) {
  const vmax = styleFeature._vmax;
  const v = slotValue(feat.properties.name);
  const t = Math.sqrt(Math.min(v / vmax, 1)); // escala raíz: mejor spread visual en distribución sesgada
  return {
    fillColor: rampColor(t), fillOpacity: 0.62,
    color: "#0b1020", weight: 1,
  };
}

function repaint() {
  styleFeature._vmax = currentMax();
  geoLayer.setStyle(styleFeature);
  const unit = state.normalize ? "incid./sem·km²" : "incid./semana";
  $("#legend-title").textContent = `Riesgo esperado (${unit})`;
  $("#legend-max").textContent = `≥ ${styleFeature._vmax.toFixed(1)}`;
  $("#selection-pill").innerHTML =
    `${DOW_FULL[state.dow]} · ${DATA.meta.bandas[state.band]} h` +
    `<span class="pill-sub">semana del ${DATA.meta.semana_predicha}</span>`;
  renderDetail();
}

/* ---------- controles ---------- */
function buildControls() {
  const dowRow = $("#dow-buttons"), bandRow = $("#band-buttons");
  DOW_SHORT.forEach((d, i) => {
    const btn = document.createElement("button");
    btn.textContent = d;
    btn.onclick = () => { state.dow = i; syncButtons(); repaint(); };
    dowRow.appendChild(btn);
  });
  DATA.meta.bandas.forEach((b, i) => {
    const btn = document.createElement("button");
    btn.textContent = b;
    btn.onclick = () => { state.band = i; syncButtons(); repaint(); };
    bandRow.appendChild(btn);
  });
  $("#normalize").onchange = (e) => { state.normalize = e.target.checked; repaint(); };
  syncButtons();
}

function syncButtons() {
  [...$("#dow-buttons").children].forEach((b, i) => b.classList.toggle("active", i === state.dow));
  [...$("#band-buttons").children].forEach((b, i) => b.classList.toggle("active", i === state.band));
}

/* ---------- panel de detalle ---------- */
function renderDetail() {
  const el = $("#barrio-detail");
  if (!state.selected || !DATA.barrios[state.selected]) return;
  const name = state.selected;
  const b = DATA.barrios[name];
  const exp = b.esperado[state.dow][state.band];
  const risk = Math.round(b.riesgo[state.dow][state.band]);
  const riskClass = risk >= 75 ? "risk-high" : risk >= 40 ? "risk-mid" : "risk-low";
  const maxBand = Math.max(...b.banda_dist, 1);

  el.innerHTML = `
    <h3>${name}</h3>
    <div class="comuna">Comuna ${b.comuna ?? "—"} · ${b.area_km2 ?? "?"} km² · ${b.total_12m.toLocaleString("es-AR")} incidentes en los últimos 12 meses</div>
    <div class="chip-grid">
      <div class="chip ${riskClass}"><div class="chip-val">${risk}<small>/100</small></div><div class="chip-label">índice de riesgo del slot</div></div>
      <div class="chip"><div class="chip-val">${exp.toFixed(1)}</div><div class="chip-label">incid. esperados/semana</div></div>
      <div class="chip"><div class="chip-val">${b.uso_arma_pct ?? "—"}%</div><div class="chip-label">con uso de arma (12m)</div></div>
      <div class="chip"><div class="chip-val">${(state.normalize && b.area_km2 ? exp / b.area_km2 : exp).toFixed(1)}</div><div class="chip-label">${state.normalize ? "incid./sem·km²" : "incid./semana"}</div></div>
    </div>
    <h4>Tipos de delito predominantes (12m)</h4>
    ${b.top_tipos.map(t => `
      <div class="tipo-row">
        <div class="tipo-head"><span>${t.tipo}</span><span>${t.pct}%</span></div>
        <div class="tipo-bar"><span style="width:${t.pct}%"></span></div>
      </div>`).join("")}
    <h4>Distribución por franja horaria (12m)</h4>
    <div class="band-chart">
      ${b.banda_dist.map((v, i) => `
        <div class="band-col ${i === state.band ? "current" : ""}">
          <div class="bar" style="height:${Math.round(100 * v / maxBand)}%"></div>
          <div class="band-label">${DATA.meta.bandas[i].split("-")[0]}h</div>
        </div>`).join("")}
    </div>`;
}

/* ---------- gráfico citywide ---------- */
function buildChart() {
  const hist = DATA.citywide.historia, fc = DATA.citywide.forecast;
  const labels = [...hist.map(d => d.fecha), ...fc.map(d => d.fecha)];
  const histData = [...hist.map(d => d.n), ...Array(fc.length).fill(null)];
  const fcData = [...Array(hist.length - 1).fill(null), hist[hist.length - 1].n, ...fc.map(d => d.pred)];

  new Chart($("#citywide-chart"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Observado", data: histData, borderColor: "#60a5fa", backgroundColor: "#60a5fa22",
          fill: true, tension: .3, pointRadius: 0, borderWidth: 2 },
        { label: "Pronóstico (14 días)", data: fcData, borderColor: "#f59e0b", borderDash: [6, 4],
          fill: false, tension: .3, pointRadius: 2, borderWidth: 2 },
      ],
    },
    options: {
      responsive: true, interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { color: "#93a1bd", font: { family: "Inter" } } } },
      scales: {
        x: { ticks: { color: "#93a1bd", maxTicksLimit: 12 }, grid: { color: "#24304f44" } },
        y: { ticks: { color: "#93a1bd" }, grid: { color: "#24304f44" }, title: { display: true, text: "incidentes/día", color: "#93a1bd" } },
      },
    },
  });
}

/* ---------- métricas ---------- */
function buildMetrics() {
  const cards = [
    {
      title: "Modelo espacial — riesgo por slot",
      sub: `Conteo semanal por barrio × día × franja (${METRICS.modelo_slots.n_slots.toLocaleString("es-AR")} slots). ` +
           `Test: ${METRICS.modelo_slots.test_semanas} semanas desde ${METRICS.modelo_slots.test_desde}.`,
      baseline: "Media histórica del slot", m: METRICS.modelo_slots,
    },
    {
      title: "Modelo temporal — incidentes diarios CABA",
      sub: `Total diario de la ciudad. Test: ${METRICS.modelo_diario.test_dias} días desde ${METRICS.modelo_diario.test_desde}.`,
      baseline: "Naive estacional (t−7)", m: METRICS.modelo_diario,
    },
  ];
  $("#metrics-grid").innerHTML = cards.map(c => {
    const impMae = c.m.mejora_mae_pct, impRmse = c.m.mejora_rmse_pct;
    const cls = (v) => v >= 0 ? "pos" : "neg";
    const sign = (v) => (v >= 0 ? "−" : "+") + Math.abs(v).toFixed(1) + "% error";
    return `
    <div class="card metric-card">
      <h3>${c.title}</h3>
      <div class="metric-sub">${c.sub}</div>
      <table class="metric-table">
        <tr><th></th><th>MAE</th><th>RMSE</th></tr>
        <tr><td>Baseline (${c.baseline})</td><td>${c.m.baseline_mae}</td><td>${c.m.baseline_rmse}</td></tr>
        <tr><td>Gradient Boosting (Poisson)</td><td><strong>${c.m.model_mae}</strong></td><td><strong>${c.m.model_rmse}</strong></td></tr>
        <tr><td>Mejora vs baseline</td>
            <td class="improvement ${cls(impMae)}">${sign(impMae)}</td>
            <td class="improvement ${cls(impRmse)}">${sign(impRmse)}</td></tr>
      </table>
    </div>`;
  }).join("");
}

/* ---------- init ---------- */
async function init() {
  const [pred, geo, metrics] = await Promise.all([
    fetch("data/predictions.json").then(r => r.json()),
    fetch("data/barrios.geojson").then(r => r.json()),
    fetch("data/metrics.json").then(r => r.json()),
  ]);
  DATA = pred; GEO = geo; METRICS = metrics;

  const now = nowInBuenosAires();
  state.dow = now.dow; state.band = now.band;

  $("#topbar-meta").textContent =
    `Datos: BA Data 2022–2025 (${(576410).toLocaleString("es-AR")} incidentes) · ` +
    `Predicción para la semana del ${DATA.meta.semana_predicha} · Actualizado ${DATA.meta.generado}`;

  initMap();
  buildControls();
  styleFeature._vmax = currentMax();
  repaint();
  buildChart();
  buildMetrics();
}

init().catch(err => {
  console.error(err);
  $("#topbar-meta").textContent = "Error cargando datos — ¿corriste src/prepare.py y src/train.py?";
});
