# Arg-Security Predictor: Plataforma Federal de Inteligencia Criminal

> **Sistema integral de análisis, clustering y predicción de seguridad ciudadana para Mendoza, Córdoba y Buenos Aires, potenciado por Deep Learning.**

## Resumen del Proyecto
Este proyecto tiene como objetivo desarrollar una solución de Policía Predictiva basada en datos. A diferencia de los enfoques tradicionales que solo muestran "dónde robaron ayer", nuestra plataforma busca responder preguntas complejas mediante Machine Learning: 
* ¿Qué probabilidad de incidente tiene esta esquina un viernes por la noche? 

* ¿Cómo influyen factores externos (clima, iluminación, festividades) en la tasa delictiva?

## Objetivos Estratégicos
1.  **Federación de Datos:** Homologar estructuras de datos dispares de las tres provincias más pobladas de Argentina.
2.  **Modelado Predictivo:** Anticipar la demanda de seguridad (cantidad de incidentes) y el riesgo específico por zona.
3.  **Detección de Patrones Ocultos:** Descubrir correlaciones no obvias (ej: relación entre fases lunares/clima y tipos de delito).
4.  **Ética del Dato:** Implementar algoritmos que mitiguen el sesgo de "zona roja" mediante un enfoque de corredores seguros.



## Arquitectura del Pipeline de Datos

### 1. Ingesta y Enriquecimiento (ETL Avanzado)
No solo consumimos datos de delitos, sino que los enriquecemos para dar contexto al modelo (Feature Engineering).

* **Fuentes Primarias (El "Qué"):**
    * *SNIC (Sistema Nacional de Información Criminal):* Estadísticas macro para validación de tendencias.
    * *Portales de Datos Abiertos (BA Data, Gobierno de CBA, Mendoza):* Datasets de contravenciones y llamadas al 911 (si disponibles).
    * *Scraping de Noticias (NLP):* Bot personalizado para extraer incidentes geolocalizados desde medios digitales (Mdzol, La Voz, Clarín) usando reconocimiento de entidades (NER).

* **Fuentes Secundarias (El "Contexto" - Variables Exógenas):**
    * *APIs Meteorológicas:* Temperatura, precipitaciones y nubosidad histórica (¿Aumenta el robo en días de lluvia?).
    * *Datos de Calendario:* Feriados, fechas de cobro (bancos), fases lunares (iluminación nocturna natural).
    * *Infraestructura:* Ubicación de comisarías, paradas de colectivo y cámaras de seguridad.

* **Procesamiento:**
    * **Geocoding Inverso:** Transformación de direcciones textuales ("Av. San Martín y Las Heras") a coordenadas (Lat/Long) usando *Nominatim/OpenStreetMap*.
    * **Taxonomía Unificada:** Mapeo de categorías provinciales a un estándar único (ej: "Arrebato" y "Sustracción en vía pública" -> `ROBO_VIA_PUBLICA`).

---

### 2. Modelado y Algoritmos (Core Intelligence)
Implementaremos cuatro módulos analíticos para cubrir distintas necesidades:

#### A. Análisis Espacial (Clustering)
* **Algoritmo:** **DBSCAN** (Density-Based Spatial Clustering).
* **Objetivo:** Identificar **Hotspots Dinámicos**. A diferencia de K-Means, DBSCAN detecta zonas de forma irregular y descarta el "ruido" (delitos aislados que no constituyen tendencia), permitiendo a las fuerzas focalizar patrullaje en clusters reales de alta densidad.

#### B. Predicción Temporal (Forecasting)
* **Algoritmo:** **LSTM** (Long Short-Term Memory) / **Prophet**.
* **Objetivo:** Predecir la **Carga Delictiva**. Estimar el volumen de incidentes esperados para la próxima semana/mes. Esto permite la planificación eficiente de recursos humanos (turnos policiales).

#### C. Scoring de Riesgo (Clasificación/Regresión)
* **Algoritmo:** **XGBoost** o **Random Forest**.
* **Objetivo:** Calcular el **Índice de Seguridad (0-100)**.
    * *Input:* Coordenadas, Día de la semana, Hora, Clima, Cercanía a comisaría.
    * *Output:* Probabilidad de ocurrencia de un incidente en ese momento y lugar específico.
    * *Uso:* Permitir al usuario consultar: "¿Qué tan segura es esta zona un sábado a las 3 AM?".

#### D. Detección de Anomalías (Unsupervised)
* **Algoritmo:** **Isolation Forest**.
* **Objetivo:** Detectar **Cambios de Patrón**. El sistema alertará cuando una zona tradicionalmente tranquila experimente un pico inusual de actividad (outliers), lo cual suele indicar una nueva modalidad delictiva o una banda operando temporalmente.

---

### 3. Stack Tecnológico
* **Lenguaje:** Python 3.10+.
* **Data Engineering:** Pandas, GeoPandas, SQLAlchemy.
* **NLP & Scraping:** BeautifulSoup, Spacy (para extraer direcciones de noticias).
* **Machine Learning:** Scikit-Learn (Random Forest, DBSCAN, Isolation Forest), TensorFlow/Keras (LSTM), XGBoost.
* **Visualización:** Folium (Mapas de calor interactivos), Seaborn, Plotly.

### 4. Métricas de Evaluación
Para asegurar la fiabilidad de los modelos, utilizaremos:
* *Para Clustering:* **Silhouette Score** (cohesión de los clusters).
* *Para Predicción (Regresión):* **RMSE** (Error Cuadrático Medio) y **MAE**.
* *Para Clasificación:* **F1-Score** y **Matriz de Confusión** (crucial para equilibrar falsos positivos/negativos).

---

## 📅 Roadmap del Proyecto
1. Definición del Diccionario de Datos unificado y construcción de scrapers.
2. EDA (Análisis Exploratorio) y generación de mapas de calor estáticos.
3. Entrenamiento del modelo espacial (DBSCAN) y temporal (LSTM).
4. Desarrollo del modelo de Scoring de Riesgo (XGBoost) con variables climáticas.
5. Integración y Dashboard final.

---
