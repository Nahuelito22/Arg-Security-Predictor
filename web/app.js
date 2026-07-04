/* BA Crime Predictor v0.2 — frontend.
 * Consume predictions.json / metrics.json (salida de src/train.py) + geojsons y arma:
 *  - choropleth por barrio o comuna, filtrable por tipo de delito
 *  - panel de detalle con huella semanal (heatmap 7x6 clickeable)
 *  - planificación operativa: top slots + CSV + reporte imprimible + alertas
 *  - forecast citywide con banda de incertidumbre calibrada (CQR)
 */

const DOW_SHORT = ["LUN", "MAR", "MIÉ", "JUE", "VIE", "SÁB", "DOM"];
const DOW_FULL = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"];
const RAMP = [[34, 197, 94], [250, 204, 21], [239, 68, 68]]; // verde -> amarillo -> rojo

const state = { dow: 0, band: 0, normalize: true, tipo: "Todos", vista: "barrios", selected: null };
let DATA = null, METRICS = null, TIPOS = [];
let map = null;
const layers = { barrios: null, comunas: null };
const keyToLayer = { barrios: {}, comunas: {} };

/* ---------- helpers ---------- */
const $ = (sel) => document.querySelector(sel);
const fmt = (n, d = 1) => Number(n).toLocaleString("es-AR", { maximumFractionDigits: d, minimumFractionDigits: 0 });

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

const zonesData = () => state.vista === "barrios" ? DATA.barrios : DATA.comunas;
const zoneLabel = (key) => state.vista === "barrios" ? key : `Comuna ${key}`;
const featKey = (feat) => state.vista === "barrios" ? feat.properties.name : String(feat.properties.comuna);

function rawExpected(z, dow, band) {
  let v = z.esperado[dow][band];
  if (state.tipo !== "Todos") v *= (z.tipo_share[state.tipo] || 0);
  return v;
}

function slotValue(key) {
  const z = zonesData()[key];
  if (!z) return 0;
  const v = rawExpected(z, state.dow, state.band);
  return state.normalize && z.area_km2 ? v / z.area_km2 : v;
}

function currentMax() {
  const vals = Object.keys(zonesData()).map(slotValue).sort((a, b) => a - b);
  return vals[Math.floor(vals.length * 0.95)] || 1; // p95 para que un outlier no lave la escala
}

/* ---------- mapa ---------- */
function makeLayer(geojson, vista) {
  return L.geoJSON(geojson, {
    style: (feat) => styleFor(featKeyOf(feat, vista)),
    onEachFeature: (feat, layer) => {
      const key = featKeyOf(feat, vista);
      keyToLayer[vista][key] = layer;
      layer.on({
        mouseover: (e) => { e.target.setStyle({ weight: 2.5, color: "#f59e0b" }); e.target.bringToFront(); },
        mouseout: (e) => layers[vista].resetStyle(e.target),
        click: () => { state.selected = key; renderDetail(); },
      });
      layer.bindTooltip(() => tooltipHtml(key), { sticky: true, className: "barrio-tip" });
    },
  });
}
const featKeyOf = (feat, vista) => vista === "barrios" ? feat.properties.name : String(feat.properties.comuna);

function styleFor(key) {
  const vmax = styleFor._vmax || 1;
  const t = Math.sqrt(Math.min(slotValue(key) / vmax, 1)); // escala raíz: mejor spread visual
  return { fillColor: rampColor(t), fillOpacity: 0.62, color: "#0b1020", weight: 1 };
}

function tooltipHtml(key) {
  const z = zonesData()[key];
  if (!z) return `<b>${zoneLabel(key)}</b>sin datos`;
  const exp = rawExpected(z, state.dow, state.band);
  let html = `<b>${zoneLabel(key)}</b>` +
    `<span class="tip-val">${fmt(exp)}</span> incidentes esperados / semana` +
    (state.tipo !== "Todos" ? ` <small>(${state.tipo})</small>` : "");
  if (state.tipo === "Todos") {
    html += `<br>índice de riesgo: <span class="tip-val">${Math.round(z.riesgo[state.dow][state.band])}</span>/100`;
  }
  return html;
}

function initMap(geoB, geoC) {
  map = L.map("map", { zoomControl: true, attributionControl: false })
    .setView([-34.615, -58.445], 12);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    maxZoom: 18, subdomains: "abcd",
  }).addTo(map);
  layers.barrios = makeLayer(geoB, "barrios");
  layers.comunas = makeLayer(geoC, "comunas");
  layers.barrios.addTo(map);
}

function setVista(vista) {
  if (vista === state.vista) return;
  map.removeLayer(layers[state.vista]);
  state.vista = vista;
  state.selected = null;
  layers[vista].addTo(map);
  buildZoneSelect();
  syncButtons();
  renderAll();
  $("#zone-detail").innerHTML = `<div class="detail-empty"><div class="detail-empty-icon">🗺️</div><p>Elegí una ${vista === "barrios" ? "zona" : "comuna"} del mapa para ver su detalle.</p></div>`;
}

function repaint() {
  styleFor._vmax = currentMax();
  layers[state.vista].setStyle((feat) => styleFor(featKeyOf(feat, state.vista)));
  const unit = state.normalize ? "incid./sem·km²" : "incid./semana";
  const tipoTag = state.tipo === "Todos" ? "" : ` · ${state.tipo}`;
  $("#legend-title").textContent = `Riesgo esperado (${unit})${tipoTag}`;
  $("#legend-max").textContent = `≥ ${fmt(styleFor._vmax)}`;
  $("#selection-pill").innerHTML =
    `${DOW_FULL[state.dow]} · ${DATA.meta.bandas[state.band]} h${tipoTag}` +
    `<span class="pill-sub">semana del ${DATA.meta.semana_predicha} · vista ${state.vista}</span>`;
}

/* ---------- controles ---------- */
function buildControls() {
  DOW_SHORT.forEach((d, i) => {
    const btn = document.createElement("button");
    btn.textContent = d;
    btn.onclick = () => { state.dow = i; syncButtons(); renderAll(); };
    $("#dow-buttons").appendChild(btn);
  });
  DATA.meta.bandas.forEach((b, i) => {
    const btn = document.createElement("button");
    btn.textContent = b;
    btn.onclick = () => { state.band = i; syncButtons(); renderAll(); };
    $("#band-buttons").appendChild(btn);
  });
  ["Todos", ...TIPOS].forEach((t) => {
    const btn = document.createElement("button");
    btn.textContent = t;
    btn.onclick = () => { state.tipo = t; syncButtons(); renderAll(); };
    $("#tipo-buttons").appendChild(btn);
  });
  [...$("#vista-buttons").children].forEach((btn) => {
    btn.onclick = () => setVista(btn.dataset.vista);
  });
  $("#normalize").onchange = (e) => { state.normalize = e.target.checked; repaint(); renderDetail(); };
  $("#zone-select").onchange = (e) => {
    if (!e.target.value) return;
    state.selected = e.target.value;
    renderDetail();
    const layer = keyToLayer[state.vista][state.selected];
    if (layer) map.fitBounds(layer.getBounds(), { padding: [30, 30], maxZoom: 14 });
  };
  $("#csv-btn").onclick = downloadCsv;
  $("#print-btn").onclick = () => window.print();
  buildZoneSelect();
  syncButtons();
}

function buildZoneSelect() {
  const sel = $("#zone-select");
  const keys = Object.keys(zonesData());
  keys.sort(state.vista === "comunas" ? (a, b) => +a - +b : undefined);
  sel.innerHTML = `<option value="">Elegir…</option>` +
    keys.map(k => `<option value="${k}">${zoneLabel(k)}</option>`).join("");
}

function syncButtons() {
  [...$("#dow-buttons").children].forEach((b, i) => b.classList.toggle("active", i === state.dow));
  [...$("#band-buttons").children].forEach((b, i) => b.classList.toggle("active", i === state.band));
  [...$("#tipo-buttons").children].forEach((b) => b.classList.toggle("active", b.textContent === state.tipo));
  [...$("#vista-buttons").children].forEach((b) => b.classList.toggle("active", b.dataset.vista === state.vista));
}

/* ---------- panel de detalle ---------- */
function renderDetail() {
  const z = state.selected ? zonesData()[state.selected] : null;
  if (!z) return;
  const el = $("#zone-detail");
  const exp = rawExpected(z, state.dow, state.band);
  const risk = Math.round(z.riesgo[state.dow][state.band]);
  const riskClass = risk >= 75 ? "risk-high" : risk >= 40 ? "risk-mid" : "risk-low";
  const maxBand = Math.max(...z.banda_dist, 1);
  const heatMax = Math.max(...z.esperado.flat(), 0.001);
  const sub = state.vista === "barrios"
    ? `Comuna ${z.comuna ?? "—"} · ${z.area_km2 ?? "?"} km² · ${fmt(z.total_12m, 0)} incidentes en 12 meses`
    : `${z.barrios.length} barrios · ${z.area_km2 ?? "?"} km² · ${fmt(z.total_12m, 0)} incidentes en 12 meses`;

  el.innerHTML = `
    <h3>${zoneLabel(state.selected)}</h3>
    <div class="comuna">${sub}</div>
    <div class="chip-grid">
      <div class="chip ${riskClass}"><div class="chip-val">${risk}<small>/100</small></div><div class="chip-label">índice de riesgo del slot</div></div>
      <div class="chip"><div class="chip-val">${fmt(exp)}</div><div class="chip-label">incid. esperados/semana${state.tipo !== "Todos" ? ` (${state.tipo})` : ""}</div></div>
      <div class="chip"><div class="chip-val">${z.uso_arma_pct ?? "—"}%</div><div class="chip-label">con uso de arma (12m)</div></div>
      <div class="chip"><div class="chip-val">${z.uso_moto_pct ?? "—"}%</div><div class="chip-label">con uso de moto (12m)</div></div>
    </div>
    <h4>Huella semanal — esperado por slot (clic para explorar)</h4>
    <table class="heatmap">
      <tr><th></th>${DATA.meta.bandas.map(b => `<th>${b.split("-")[0]}h</th>`).join("")}</tr>
      ${z.esperado.map((row, d) => `<tr><th>${DOW_SHORT[d]}</th>${row.map((v, b) => {
        const t = Math.sqrt(v / heatMax);
        const cur = d === state.dow && b === state.band ? " current" : "";
        return `<td class="hm-cell${cur}" data-d="${d}" data-b="${b}" style="background:${rampColor(t)}" title="${DOW_FULL[d]} ${DATA.meta.bandas[b]}: ${fmt(v)}"></td>`;
      }).join("")}</tr>`).join("")}
    </table>
    ${state.vista === "comunas" ? `<h4>Barrios</h4><p class="barrios-list">${z.barrios.join(" · ")}</p>` : ""}
    <h4>Tipos de delito predominantes (12m)</h4>
    ${z.top_tipos.map(t => `
      <div class="tipo-row">
        <div class="tipo-head"><span>${t.tipo}</span><span>${t.pct}%</span></div>
        <div class="tipo-bar"><span style="width:${t.pct}%"></span></div>
      </div>`).join("")}
    <h4>Distribución por franja horaria (12m)</h4>
    <div class="band-chart">
      ${z.banda_dist.map((v, i) => `
        <div class="band-col ${i === state.band ? "current" : ""}">
          <div class="bar" style="height:${Math.round(100 * v / maxBand)}%"></div>
          <div class="band-label">${DATA.meta.bandas[i].split("-")[0]}h</div>
        </div>`).join("")}
    </div>`;

  el.querySelectorAll(".hm-cell").forEach(cell => {
    cell.onclick = () => {
      state.dow = +cell.dataset.d; state.band = +cell.dataset.b;
      syncButtons(); renderAll();
    };
  });
}

/* ---------- planificación operativa ---------- */
function renderTopSlots() {
  // Slot pico de cada zona (si no, el ranking absoluto es todo Palermo y no aporta)
  const rows = [];
  for (const [key, z] of Object.entries(zonesData())) {
    let best = null;
    for (let d = 0; d < 7; d++) for (let b = 0; b < 6; b++) {
      const exp = rawExpected(z, d, b);
      if (!best || exp > best.exp) best = { key, d, b, exp, area: z.area_km2, riesgo: z.riesgo[d][b] };
    }
    if (best) rows.push(best);
  }
  rows.sort((a, b) => b.exp - a.exp);
  const top = rows.slice(0, 10);
  const showIdx = state.tipo === "Todos";
  $("#top-slots-title").textContent = `Slot pico por zona — top 10 (vista ${state.vista})` +
    (state.tipo !== "Todos" ? ` · ${state.tipo}` : "");
  $("#top-slots-sub").textContent = `El momento de mayor demanda esperada de cada zona, para la semana del ${DATA.meta.semana_predicha}. Clic en una fila para verla en el mapa. El CSV incluye los ${state.vista === "barrios" ? "2.016" : "630"} slots completos.`;
  $("#top-slots-table").innerHTML = `
    <tr><th>#</th><th>Zona</th><th>Día</th><th>Franja</th><th>Esperado</th>${showIdx ? "<th>Índice</th>" : ""}<th>/km²</th></tr>` +
    top.map((r, i) => `
      <tr class="slot-row" data-key="${r.key}" data-d="${r.d}" data-b="${r.b}">
        <td>${i + 1}</td><td class="zone-cell">${zoneLabel(r.key)}</td>
        <td>${DOW_SHORT[r.d]}</td><td>${DATA.meta.bandas[r.b]}</td>
        <td class="num">${fmt(r.exp)}</td>
        ${showIdx ? `<td class="num">${Math.round(r.riesgo)}</td>` : ""}
        <td class="num">${r.area ? fmt(r.exp / r.area) : "—"}</td>
      </tr>`).join("");
  $("#top-slots-table").querySelectorAll(".slot-row").forEach(tr => {
    tr.onclick = () => {
      state.selected = tr.dataset.key; state.dow = +tr.dataset.d; state.band = +tr.dataset.b;
      syncButtons(); renderAll();
      const layer = keyToLayer[state.vista][state.selected];
      if (layer) map.fitBounds(layer.getBounds(), { padding: [30, 30], maxZoom: 14 });
      window.scrollTo({ top: 0, behavior: "smooth" });
    };
  });
}

function sparkline(serie) {
  const w = 84, h = 22, min = Math.min(...serie), max = Math.max(...serie);
  const pts = serie.map((v, i) =>
    `${(i / (serie.length - 1)) * w},${h - 2 - (h - 4) * ((v - min) / Math.max(max - min, 1))}`
  ).join(" ");
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}"><polyline points="${pts}"/></svg>`;
}

function renderAlerts() {
  const a = DATA.alertas;
  $("#alerts-sub").textContent =
    `Últimas 4 semanas (${a.ventana.reciente[0]} → ${a.ventana.reciente[1]}) vs las 12 previas de cada barrio. ` +
    `Se listan desvíos con significancia |z| ≥ 2 (test de Poisson).`;
  const row = (r, cls, arrow) => `
    <div class="alert-row ${cls}" data-barrio="${r.barrio}">
      <div class="alert-main"><span class="alert-arrow">${arrow}</span>
        <div><div class="alert-name">${r.barrio}</div>
        <div class="alert-detail">${r.obs_4sem} obs. vs ${fmt(r.esperado_4sem)} esperados · z=${r.z}</div></div>
      </div>
      <div class="alert-side">${sparkline(r.serie_16sem)}<span class="alert-pct">${r.cambio_pct > 0 ? "+" : ""}${fmt(r.cambio_pct)}%</span></div>
    </div>`;
  $("#alerts-list").innerHTML =
    (a.subiendo.length ? `<div class="alert-group-label">En aumento</div>` + a.subiendo.map(r => row(r, "up", "▲")).join("") : "") +
    (a.bajando.length ? `<div class="alert-group-label">En descenso</div>` + a.bajando.map(r => row(r, "down", "▼")).join("") : "") +
    (!a.subiendo.length && !a.bajando.length ? `<p class="card-sub">Sin desvíos significativos en la ventana actual.</p>` : "");
  $("#alerts-list").querySelectorAll(".alert-row").forEach(el => {
    el.onclick = () => {
      if (state.vista !== "barrios") setVista("barrios");
      state.selected = el.dataset.barrio;
      renderDetail();
      const layer = keyToLayer.barrios[state.selected];
      if (layer) map.fitBounds(layer.getBounds(), { padding: [30, 30], maxZoom: 14 });
      window.scrollTo({ top: 0, behavior: "smooth" });
    };
  });
}

function downloadCsv() {
  const showIdx = state.tipo === "Todos";
  const header = ["zona", "dia", "franja", "esperado_semanal", "esperado_por_km2"].concat(showIdx ? ["indice_riesgo"] : []);
  const lines = [header.join(";")];
  for (const [key, z] of Object.entries(zonesData())) {
    for (let d = 0; d < 7; d++) for (let b = 0; b < 6; b++) {
      const exp = rawExpected(z, d, b);
      const cols = [zoneLabel(key), DOW_FULL[d], DATA.meta.bandas[b],
                    exp.toFixed(2), z.area_km2 ? (exp / z.area_km2).toFixed(3) : ""];
      if (showIdx) cols.push(Math.round(z.riesgo[d][b]));
      lines.push(cols.join(";"));
    }
  }
  const blob = new Blob(["﻿" + lines.join("\r\n")], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `predicciones_${state.vista}${state.tipo !== "Todos" ? "_" + state.tipo.toLowerCase() : ""}_${DATA.meta.semana_predicha}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ---------- gráfico citywide ---------- */
function buildChart() {
  const hist = DATA.citywide.historia, fc = DATA.citywide.forecast;
  const labels = [...hist.map(d => d.fecha), ...fc.map(d => d.fecha)];
  const nH = hist.length;
  const histData = [...hist.map(d => d.n), ...Array(fc.length).fill(null)];
  const fcData = [...Array(nH - 1).fill(null), hist[nH - 1].n, ...fc.map(d => d.pred)];
  const p10 = [...Array(nH).fill(null), ...fc.map(d => d.p10)];
  const p90 = [...Array(nH).fill(null), ...fc.map(d => d.p90)];

  new Chart($("#citywide-chart"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Observado", data: histData, borderColor: "#60a5fa", backgroundColor: "#60a5fa22",
          fill: true, tension: .3, pointRadius: 0, borderWidth: 2 },
        { label: "Pronóstico (14 días)", data: fcData, borderColor: "#f59e0b", borderDash: [6, 4],
          fill: false, tension: .3, pointRadius: 2, borderWidth: 2 },
        { label: "_p10", data: p10, borderColor: "transparent", pointRadius: 0, fill: false },
        { label: "Banda 80% (calibrada)", data: p90, borderColor: "transparent", pointRadius: 0,
          backgroundColor: "#f59e0b1f", fill: "-1" },
      ],
    },
    options: {
      responsive: true, interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { color: "#93a1bd", font: { family: "Inter" },
                                     filter: (item) => item.text !== "_p10" } } },
      scales: {
        x: { ticks: { color: "#93a1bd", maxTicksLimit: 12 }, grid: { color: "#24304f44" } },
        y: { ticks: { color: "#93a1bd" }, grid: { color: "#24304f44" },
             title: { display: true, text: "incidentes/día", color: "#93a1bd" } },
      },
    },
  });
}

/* ---------- métricas ---------- */
function buildMetrics() {
  const ms = METRICS.modelo_slots, md = METRICS.modelo_diario;
  const cards = [
    {
      title: "Modelo espacial — riesgo por slot",
      sub: `Conteo semanal por barrio × día × franja (${fmt(ms.n_slots, 0)} slots). Test: ${ms.test_semanas} semanas desde ${ms.test_desde}.`,
      baseline: "Media histórica del slot", m: ms, extra: "",
    },
    {
      title: "Modelo temporal — incidentes diarios CABA",
      sub: `Total diario de la ciudad, con feriados nacionales como feature. Test: ${md.test_dias} días desde ${md.test_desde}.`,
      baseline: "Naive estacional (t−7)", m: md,
      extra: `<div class="coverage-note">Banda q10–q90 calibrada por conformal prediction: cobertura real en test
              <strong>${md.cobertura_q10_q90_pct}%</strong> (objetivo 80%; sin calibrar era ${md.cobertura_sin_calibrar_pct}%).</div>`,
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
      ${c.extra}
    </div>`;
  }).join("");
}

const renderAll = () => { repaint(); renderDetail(); renderTopSlots(); };

/* ---------- init ---------- */
async function init() {
  const [pred, geoB, geoC, metrics] = await Promise.all([
    fetch("data/predictions.json").then(r => r.json()),
    fetch("data/barrios.geojson").then(r => r.json()),
    fetch("data/comunas.geojson").then(r => r.json()),
    fetch("data/metrics.json").then(r => r.json()),
  ]);
  DATA = pred; METRICS = metrics;

  // tipos ordenados por volumen citywide (ponderado por total de cada barrio)
  const weight = {};
  for (const z of Object.values(DATA.barrios)) {
    for (const [t, s] of Object.entries(z.tipo_share)) weight[t] = (weight[t] || 0) + s * z.total_12m;
  }
  TIPOS = Object.entries(weight).sort((a, b) => b[1] - a[1]).map(([t]) => t);

  const now = nowInBuenosAires();
  state.dow = now.dow; state.band = now.band;

  $("#topbar-meta").textContent =
    `Datos: BA Data 2022–2025 (${fmt(576410, 0)} incidentes) · ` +
    `Predicción para la semana del ${DATA.meta.semana_predicha} · Actualizado ${DATA.meta.generado}`;

  initMap(geoB, geoC);
  buildControls();
  styleFor._vmax = currentMax();
  repaint();
  renderTopSlots();
  renderAlerts();
  buildChart();
  buildMetrics();
}

init().catch(err => {
  console.error(err);
  $("#topbar-meta").textContent = "Error cargando datos — ¿corriste src/prepare.py y src/train.py?";
});
