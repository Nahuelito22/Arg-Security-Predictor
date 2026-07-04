# Plan vivo — Arg-Security-Predictor

> Este archivo es la fuente de verdad del estado del proyecto. Se actualiza en cada sesión de trabajo.
> Última actualización: 2026-07-04 (v0.2)

## Estado actual: v0.2 — plataforma operativa ✅

Página estática con mapa de riesgo por **barrio y comuna**, filtro por tipo de delito,
**alertas de cambio de patrón** (test de Poisson sobre las últimas 4 semanas de cada barrio),
slot pico por zona con **export CSV** y reporte imprimible, forecast citywide con **banda de
incertidumbre calibrada (CQR)** y feriados nacionales como feature. Todo corre local y gratis.

### Qué se agregó en v0.2 (sobre el MVP v0.1)

| Mejora | Detalle | Impacto medido |
|---|---|---|
| Feriados nacionales (feature) | Lista curada 2022–2026 en el modelo diario | MAE 40.4 → **33.6** (mejora total vs baseline: 31,2%) |
| Contexto de barrio (feature) | Rolling 4 semanas del total del barrio en el modelo de slots | MAE 0.8658 → **0.8629** |
| Banda de incertidumbre | Cuantiles q10/q90 + calibración conformal (CQR) en tramo separado | Cobertura real en test: 62,2% → **78,9%** (objetivo 80%) |
| Alertas de cambio de patrón | z-score de Poisson, últimas 4 semanas vs 12 previas, umbral \|z\|≥2 | Módulo D del roadmap original, versión estadística explicable |
| Vista por comuna | Agregación a las 15 comunas (unidad de las Comisarías Vecinales) | Uso operativo directo |
| Filtro por tipo de delito | Descriptivo: mezcla histórica 12m por zona (documentado como tal) | Robo ≠ Hurto en estrategia de prevención |
| Planificación operativa | Slot pico por zona, CSV de 2.016 slots, reporte imprimible | Lo que se lleva a una reunión de turnos |

## Decisiones tomadas (y por qué)

| Decisión | Razón |
|---|---|
| **Solo CABA** para el MVP (no Mendoza/Córdoba) | Es la única jurisdicción con dataset abierto a nivel incidente (fecha + franja + barrio + tipo). Federar 3 provincias sin datos granulares comparables era el mayor riesgo del plan original. |
| **2022–2025** (se excluye 2021 y anteriores) | 2020–2021 están distorsionados por la pandemia; 4 años completos alcanzan para estacionalidad + tendencia. |
| **Gradient Boosting con pérdida Poisson** en vez de LSTM | Los conteos por slot son ralos (media ~0.9/semana): un LSTM no tiene señal suficiente y complica el stack (TensorFlow). HistGradientBoosting de sklearn modela conteos correctamente (Poisson), corre en segundos y es interpretable. El LSTM queda en backlog para el forecast citywide. |
| **Slots de 4 horas** (6 franjas/día) en vez de hora a hora | Con 48 barrios × 7 días × 24 horas los conteos quedan casi todos en 0/1: puro ruido. 4h equilibra granularidad y señal. |
| **Baseline explícito en cada modelo** | Regla del proyecto: un modelo que no supera a la media histórica / naive estacional no se publica. Las métricas van en la página, incluidas las mejoras modestas. |
| **Predicciones precomputadas → página estática** | El MVP no necesita backend: `train.py` exporta JSON y la página los consume. Deploy gratis (GitHub Pages/Vercel) y cero costo de operación. |
| **Índice de riesgo = percentil (0–100)** | Comunicar "riesgo 90/100" es más honesto que inventar probabilidades: es riesgo *relativo* entre los 2.016 slots de la ciudad. |

## Resultados actuales (test temporal, datos nunca vistos)

| Modelo | Baseline | MAE baseline | MAE modelo | Mejora |
|---|---|---|---|---|
| Espacial (slot semanal) | Media histórica del slot | 0.898 | 0.863 | **−3.9%** |
| Temporal (diario citywide) | Naive estacional t−7 | 48.81 | 33.57 | **−31.2%** |

Banda q10–q90 del forecast: cobertura real en test **78,9%** (objetivo 80%) tras calibración CQR.

Lectura honesta: en el modelo espacial la media histórica es un baseline muy fuerte (los patrones
por barrio son estables); la ganancia viene de capturar tendencia, estacionalidad y contexto de
barrio. En el diario la mejora es grande y los feriados explican casi la mitad. Todos los números
están publicados en la página.

## Roadmap original → estado

1. ~~Diccionario de datos unificado~~ → ✅ hecho para CABA (`src/prepare.py` + reporte de calidad)
2. ~~EDA y mapas estáticos~~ → ✅ superado: mapa interactivo barrio/comuna con filtros
3. Modelo espacial (DBSCAN) y temporal (LSTM) → ⚠️ reemplazados por GBM Poisson (ver decisiones); DBSCAN de hotspots por coordenadas queda en backlog (los datos ya traen lat/lon limpias en `incidents_clean.csv`)
4. ~~Scoring de riesgo~~ → ✅ índice 0–100 por barrio × día × franja + banda de incertidumbre calibrada
5. ~~Dashboard~~ → ✅ `web/` (Leaflet + Chart.js, estático) con planificación operativa
6. ~~Detección de anomalías (módulo D)~~ → ✅ v0.2: alertas por z-score de Poisson (más explicable que Isolation Forest para este caso; IF queda como upgrade posible)

## Backlog priorizado (siguiente sesión)

- [x] ~~Deploy GitHub Pages~~ → workflow en `.github/workflows/pages.yml`
- [x] ~~Intervalos de predicción~~ → v0.2: CQR con cobertura medida
- [ ] **DBSCAN hotspots**: clusters por lat/lon dentro de barrio, capa opcional en el mapa
- [ ] **Variables exógenas**: clima histórico (Open-Meteo, gratis), eventos masivos
- [ ] **Pool de modelos** (idea original del proyecto): sumar Prophet/SARIMA al forecast diario y ensamblar; comparar en la tabla de métricas
- [ ] **Automatizar refresh**: script que baje el CSV nuevo del año en curso y regenere todo (cron o Action mensual)
- [ ] **Backtesting de alertas**: medir precisión/recall de las alertas contra ventanas históricas
- [ ] **Mendoza/Córdoba**: evaluar si existen datasets granulares equivalentes; si no, documentar por qué no se federó
- [ ] **Paper/informe técnico** corto (formato notebook o PDF) para el CV

## Cómo correr todo

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
# 1. descargar CSVs 2022-2025 de BA Data a data/raw/ (ver README)
python src/prepare.py    # limpieza + agregaciones + geojson (~40s)
python src/train.py      # modelos + evaluación + export a web/data/ (~2min)
# servir la página:
python -m http.server 8765 --directory web
```

## Reglas del proyecto

- Ningún número de negocio se inventa: todo sale de los datos o se marca como supuesto.
- Todo modelo se compara contra un baseline y se publica la comparación, gane o pierda.
- Costo de operación objetivo: $0 (datos abiertos, cómputo local, hosting estático).
- Los datos crudos no se commitean (se regeneran con el pipeline).
