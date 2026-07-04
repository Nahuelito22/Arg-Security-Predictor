# Plan vivo — Arg-Security-Predictor

> Este archivo es la fuente de verdad del estado del proyecto. Se actualiza en cada sesión de trabajo.
> Última actualización: 2026-07-04

## Estado actual: MVP v0.1 funcionando ✅

Página estática con mapa de riesgo por barrio de CABA + dos modelos entrenados y evaluados
honestamente contra baselines. Todo corre local y gratis (sin APIs pagas, sin servidores).

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
| Espacial (slot semanal) | Media histórica del slot | 0.898 | 0.866 | **−3.6%** |
| Temporal (diario citywide) | Naive estacional t−7 | 48.81 | 40.38 | **−17.3%** |

Lectura honesta: en el modelo espacial la media histórica es un baseline muy fuerte (los patrones
por barrio son estables); la ganancia viene de capturar tendencia y estacionalidad. En el diario
la mejora es clara. Ambos números están publicados en la página.

## Roadmap original → estado

1. ~~Diccionario de datos unificado~~ → ✅ hecho para CABA (`src/prepare.py` + reporte de calidad)
2. ~~EDA y mapas estáticos~~ → ✅ superado: mapa interactivo con selector día/franja
3. Modelo espacial (DBSCAN) y temporal (LSTM) → ⚠️ reemplazados por GBM Poisson (ver decisiones); DBSCAN de hotspots por coordenadas queda en backlog (los datos ya traen lat/lon limpias en `incidents_clean.csv`)
4. ~~Scoring de riesgo~~ → ✅ índice 0–100 por barrio × día × franja
5. ~~Dashboard~~ → ✅ `web/` (Leaflet + Chart.js, estático)

## Backlog priorizado (siguiente sesión)

- [ ] **Deploy**: GitHub Pages (gratis; `web/` ya es autocontenida)
- [ ] **DBSCAN hotspots**: clusters por lat/lon dentro de barrio, capa opcional en el mapa
- [ ] **Variables exógenas**: clima histórico (Open-Meteo, gratis), feriados (`workalendar`), eventos
- [ ] **Pool de modelos** (idea original del proyecto): sumar Prophet/SARIMA al forecast diario y ensamblar; comparar en la tabla de métricas
- [ ] **Intervalos de predicción** (cuantiles del GBM) en vez de punto único
- [ ] **Automatizar refresh**: script que baje el CSV nuevo del año en curso y regenere todo
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
