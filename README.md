# Arg-Security Predictor

> **Plataforma funcionando:** mapa interactivo de riesgo delictivo por barrio y comuna, día y franja horaria para la Ciudad de Buenos Aires, con modelos de Machine Learning evaluados honestamente sobre 576.410 incidentes reales (2022–2025), alertas de cambio de patrón y herramientas de planificación operativa.

![status](https://img.shields.io/badge/estado-v0.2-f59e0b) ![python](https://img.shields.io/badge/python-3.10%2B-3776AB) ![license](https://img.shields.io/badge/licencia-MIT-green)

## ¿Qué hace?

La página (`web/`) responde la pregunta del proyecto original — *"¿qué probabilidad de incidente tiene esta zona un viernes a la noche?"* — con datos reales, y suma las herramientas que un planificador operativo usaría de verdad:

- 🗺️ **Mapa coroplético de CABA** por **barrio (48) o comuna (15)** — la comuna es la unidad territorial de las Comisarías Vecinales. Riesgo esperado para cualquier día × franja de 4h, filtrable por **tipo de delito** y normalizable por km². Abre mostrando el riesgo de *ahora*.
- 🎯 **Índice de riesgo 0–100 por slot** (percentil entre los 2.016 slots barrio×día×franja) + incidentes esperados por semana + **huella semanal** de cada zona (heatmap 7×6 clickeable).
- 🚨 **Alertas de cambio de patrón**: barrios cuya actividad de las últimas 4 semanas se desvía significativamente de sus 12 previas (test de Poisson, |z| ≥ 2) — señal temprana de nuevas modalidades. Con sparklines de 16 semanas.
- 📋 **Planificación operativa**: slot pico por zona (top 10), **export CSV** de los 2.016 slots y **reporte imprimible**.
- 📈 **Forecast de la ciudad** a 14 días con **banda de incertidumbre calibrada por conformal prediction** (cobertura real medida: 78,9% vs objetivo 80%).
- 📏 **Métricas publicadas**: cada modelo se compara contra un baseline ingenuo con split temporal — los números reales están en la página, incluidas las mejoras modestas.

## Resultados (test = datos futuros nunca vistos por el modelo)

| Modelo | Baseline | MAE | RMSE | Mejora MAE |
|---|---|---|---|---|
| **Espacial** — conteo semanal por barrio×día×franja | Media histórica del slot | 0.898 → **0.863** | 1.363 → **1.337** | −3.9% |
| **Temporal** — incidentes diarios CABA | Naive estacional (t−7) | 48.81 → **33.57** | 72.23 → **49.62** | −31.2% |

Ambos son Gradient Boosting con **pérdida de Poisson** (los delitos son conteos, no valores continuos), con calendario, tendencia, lags, contexto de barrio y **feriados nacionales** (solo agregar feriados bajó el MAE diario de 40.4 a 33.6). Banda q10–q90 calibrada con **CQR (Conformalized Quantile Regression)**: cobertura real en test 78,9% (sin calibrar: 62,2%). Split estrictamente temporal: 26 semanas / 90 días finales como test. Los valores exactos de cada corrida quedan en `web/data/metrics.json`; el detalle de decisiones está en [PLAN.md](PLAN.md).

## Cómo correrlo

```bash
git clone https://github.com/Nahuelito22/Arg-Security-Predictor.git
cd Arg-Security-Predictor
python -m venv .venv && .venv\Scripts\activate    # Windows (en Linux: source .venv/bin/activate)
pip install -r requirements.txt

# 1) Descargar datos abiertos a data/raw/  (delitos_2022.csv ... delitos_2025.csv + barrios.geojson)
#    https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-justicia-y-seguridad/delitos/delitos_2022.csv  (ídem 2023/2024/2025)
#    https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/barrios/barrios.geojson

# 2) Pipeline completo
python src/prepare.py     # limpieza + control de calidad + agregaciones (~40 s)
python src/train.py       # entrenamiento + evaluación + export a web/data/ (~2 min)

# 3) Ver la página
python -m http.server 8765 --directory web
# → http://localhost:8765
```

## Estructura

```
├── data/raw/          CSVs oficiales de BA Data (no se commitean, ~90 MB)
├── data/processed/    incidentes limpios + agregaciones (regenerable)
├── src/
│   ├── prepare.py     ETL: limpieza, calidad, slots semanales, perfiles de barrio, geojson
│   └── train.py       modelos A (espacial) y B (temporal) + métricas + export JSON
└── web/               página estática (Leaflet + Chart.js, sin backend)
    └── data/          artefactos que consume la página (sí se commitean)
```

## Datos

**Fuente:** [Dataset de delitos — BA Data](https://data.buenosaires.gob.ar/dataset/delitos) (Ministerio de Justicia y Seguridad, GCBA). 588.856 filas crudas 2022–2025; se usa el 97,9% (se descartan filas sin barrio o sin franja horaria válida, reportado en `data/processed/meta.json`).

**Por qué solo CABA (por ahora):** es la única jurisdicción argentina que publica datos a nivel incidente con fecha, franja, barrio y tipo. El plan original contemplaba Mendoza y Córdoba; se federarán si aparecen datasets granulares equivalentes (ver backlog en PLAN.md).

## ⚖️ Limitaciones y ética

- Los datos son **denuncias registradas**, no delitos ocurridos: hay subregistro y el sesgo de denuncia varía por zona y tipo de delito.
- Las predicciones son **agregadas a nivel barrio** con fines de análisis y planificación. Esto **no es** una herramienta de vigilancia, patrullaje dirigido ni perfilamiento individual.
- Un índice de riesgo alto describe un *slot espacio-temporal*, no define a un barrio ni a su gente.
- Proyecto educativo de portfolio, sin uso operativo.

## Visión original (roadmap completo)

<details>
<summary>Ver la propuesta federal completa (Mendoza + Córdoba + BA, 4 módulos analíticos)</summary>

El plan original del proyecto contempla: federación de datos de las 3 provincias más pobladas,
análisis espacial con DBSCAN (hotspots dinámicos), forecasting con LSTM/Prophet, scoring de
riesgo con XGBoost enriquecido con clima/calendario/infraestructura, y detección de anomalías
con Isolation Forest. El estado de cada punto y las decisiones de simplificación del MVP están
documentados en [PLAN.md](PLAN.md).

</details>

## Autores

- [Nahuelito22](https://github.com/Nahuelito22) — idea original y arquitectura del proyecto
- Joaquín Rao — MVP: pipeline de datos, modelos y dashboard

Licencia MIT.
