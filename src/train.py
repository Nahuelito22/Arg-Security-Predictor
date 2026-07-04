"""Entrena y evalua los modelos del MVP y exporta los artefactos para la web.

Modelo A (espacial-temporal): conteo semanal esperado por slot (barrio, dia, franja 4h).
  - Baseline: media historica del slot en train.
  - Modelo:   HistGradientBoostingRegressor con loss="poisson" + calendario,
              tendencia, lags por slot y contexto del barrio (barrio_roll4).
Modelo B (temporal citywide): total diario de incidentes de la ciudad.
  - Baseline: naive estacional (mismo dia de la semana anterior).
  - Modelo:   HistGradientBoostingRegressor + lags/rolling + feriados nacionales.
  - Incertidumbre: modelos cuantilicos q10/q90 con cobertura medida en test.
Alertas de cambio de patron (modulo D del roadmap original, version estadistica):
  - Ultimas 4 semanas de cada barrio vs 12 semanas previas, z-score de Poisson.

Evaluacion con split temporal (nada de shuffle): test = ultimas 26 semanas (A)
y ultimos 90 dias (B). Toda metrica se publica comparada contra su baseline.

Salidas: web/data/predictions.json, web/data/metrics.json
Uso:  python src/train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
WEB_DATA = ROOT / "web" / "data"

TEST_WEEKS = 26
TEST_DAYS = 90
BAND_LABELS = ["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"]
DOW_LABELS = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]

# Feriados nacionales principales 2022-2026 (fijos + carnaval + semana santa + puentes).
# Lista curada a mano; una omision puntual degrada poco (es una feature, no una regla).
FERIADOS = [
    "2022-01-01", "2022-02-28", "2022-03-01", "2022-03-24", "2022-04-02", "2022-04-15",
    "2022-05-01", "2022-05-18", "2022-05-25", "2022-06-17", "2022-06-20", "2022-07-09",
    "2022-08-15", "2022-10-07", "2022-10-10", "2022-11-20", "2022-11-21", "2022-12-08",
    "2022-12-09", "2022-12-25",
    "2023-01-01", "2023-02-20", "2023-02-21", "2023-03-24", "2023-04-02", "2023-04-07",
    "2023-05-01", "2023-05-25", "2023-05-26", "2023-06-17", "2023-06-19", "2023-06-20",
    "2023-07-09", "2023-08-21", "2023-10-13", "2023-10-16", "2023-11-20", "2023-12-08",
    "2023-12-25",
    "2024-01-01", "2024-02-12", "2024-02-13", "2024-03-24", "2024-03-29", "2024-04-01",
    "2024-04-02", "2024-05-01", "2024-05-25", "2024-06-17", "2024-06-20", "2024-06-21",
    "2024-07-09", "2024-08-17", "2024-10-11", "2024-10-12", "2024-11-18", "2024-12-08",
    "2024-12-25",
    "2025-01-01", "2025-03-03", "2025-03-04", "2025-03-24", "2025-04-02", "2025-04-18",
    "2025-05-01", "2025-05-02", "2025-05-25", "2025-06-16", "2025-06-20", "2025-07-09",
    "2025-08-15", "2025-08-17", "2025-10-12", "2025-11-21", "2025-11-24", "2025-12-08",
    "2025-12-25",
    "2026-01-01", "2026-02-16", "2026-02-17", "2026-03-24", "2026-04-02", "2026-04-03",
    "2026-05-01", "2026-05-25", "2026-06-17", "2026-06-20", "2026-07-09",
]
FERIADOS_TS = set(pd.to_datetime(FERIADOS))


def rmse(y, p) -> float:
    return float(np.sqrt(mean_squared_error(y, p)))


# ---------------------------------------------------------------- Modelo A --
def build_slot_grid() -> pd.DataFrame:
    """Grid completo semana x barrio x dow x band con ceros explicitos."""
    slots = pd.read_csv(PROC / "slots_weekly.csv", parse_dates=["week_start"])
    weeks = pd.date_range(slots["week_start"].min(), slots["week_start"].max(), freq="7D")
    barrios = sorted(slots["barrio"].unique())
    grid = pd.MultiIndex.from_product(
        [weeks, barrios, range(7), range(6)], names=["week_start", "barrio", "dow", "band"]
    ).to_frame(index=False)
    grid = grid.merge(slots, on=["week_start", "barrio", "dow", "band"], how="left")
    grid["n"] = grid["n"].fillna(0).astype(float)
    return grid


def add_slot_features(grid: pd.DataFrame) -> pd.DataFrame:
    grid = grid.sort_values(["barrio", "dow", "band", "week_start"]).reset_index(drop=True)
    weeks_sorted = np.sort(grid["week_start"].unique())
    week_index = {w: i for i, w in enumerate(weeks_sorted)}
    grid["trend"] = grid["week_start"].map(week_index).astype(int)
    month = grid["week_start"].dt.month
    grid["month_sin"] = np.sin(2 * np.pi * month / 12)
    grid["month_cos"] = np.cos(2 * np.pi * month / 12)
    g = grid.groupby(["barrio", "dow", "band"], observed=True)["n"]
    grid["lag_1"] = g.shift(1)
    grid["roll_4"] = g.shift(1).rolling(4).mean().reset_index(drop=True)
    grid["roll_8"] = g.shift(1).rolling(8).mean().reset_index(drop=True)

    # Contexto del barrio: total semanal del barrio (todas las franjas), rolling 4.
    # Captura olas a nivel barrio que un slot ralo no ve por si solo.
    bt = grid.groupby(["barrio", "week_start"], as_index=False)["n"].sum().rename(columns={"n": "bt_n"})
    bt = bt.sort_values(["barrio", "week_start"]).reset_index(drop=True)
    bt["barrio_roll4"] = bt.groupby("barrio")["bt_n"].shift(1).rolling(4).mean().reset_index(drop=True)
    grid = grid.merge(bt[["barrio", "week_start", "barrio_roll4"]], on=["barrio", "week_start"], how="left")

    grid["barrio_code"] = grid["barrio"].astype("category").cat.codes
    return grid.dropna(subset=["lag_1", "roll_4", "roll_8", "barrio_roll4"]).reset_index(drop=True)


FEATURES_A = ["barrio_code", "dow", "band", "trend", "month_sin", "month_cos",
              "lag_1", "roll_4", "roll_8", "barrio_roll4"]


def train_model_a(grid: pd.DataFrame):
    weeks_sorted = np.sort(grid["week_start"].unique())
    split_week = weeks_sorted[-TEST_WEEKS]
    train = grid[grid["week_start"] < split_week]
    test = grid[grid["week_start"] >= split_week]

    # Baseline: media del slot en train
    slot_mean = train.groupby(["barrio", "dow", "band"], observed=True)["n"].mean().rename("slot_mean")
    test_b = test.join(slot_mean, on=["barrio", "dow", "band"])
    test_b["slot_mean"] = test_b["slot_mean"].fillna(train["n"].mean())

    model = HistGradientBoostingRegressor(
        loss="poisson", max_iter=300, learning_rate=0.07,
        max_leaf_nodes=63, min_samples_leaf=50,
        categorical_features=[0, 1, 2],  # barrio_code, dow, band
        random_state=42, early_stopping=False,
    )
    model.fit(train[FEATURES_A], train["n"])
    pred = model.predict(test[FEATURES_A]).clip(min=0)

    metrics = {
        "train_semanas": int(train["week_start"].nunique()),
        "test_semanas": int(test["week_start"].nunique()),
        "test_desde": str(pd.Timestamp(split_week).date()),
        "n_slots": int(grid.groupby(["barrio", "dow", "band"], observed=True).ngroups),
        "baseline_mae": round(float(mean_absolute_error(test_b["n"], test_b["slot_mean"])), 4),
        "baseline_rmse": round(rmse(test_b["n"], test_b["slot_mean"]), 4),
        "model_mae": round(float(mean_absolute_error(test["n"], pred)), 4),
        "model_rmse": round(rmse(test["n"], pred), 4),
    }
    metrics["mejora_mae_pct"] = round(100 * (1 - metrics["model_mae"] / metrics["baseline_mae"]), 2)
    metrics["mejora_rmse_pct"] = round(100 * (1 - metrics["model_rmse"] / metrics["baseline_rmse"]), 2)
    return model, metrics


def predict_next_week(grid: pd.DataFrame, model) -> pd.DataFrame:
    """Prediccion 1 semana hacia adelante para cada slot, con lags reales."""
    last_week = grid["week_start"].max()
    next_week = last_week + pd.Timedelta(days=7)
    hist = grid[grid["week_start"] > last_week - pd.Timedelta(days=7 * 9)]
    rows = []
    for (barrio, dow, band), g in hist.groupby(["barrio", "dow", "band"], observed=True):
        g = g.sort_values("week_start")
        n = g["n"].to_numpy()
        rows.append({
            "barrio": barrio, "dow": dow, "band": band,
            "lag_1": n[-1], "roll_4": n[-4:].mean(), "roll_8": n[-8:].mean(),
        })
    fut = pd.DataFrame(rows)
    # contexto barrio: media semanal del total del barrio en las ultimas 4 semanas
    bt_last = (grid[grid["week_start"] > last_week - pd.Timedelta(days=28)]
               .groupby("barrio")["n"].sum() / 4).rename("barrio_roll4")
    fut = fut.join(bt_last, on="barrio")
    fut["trend"] = grid["trend"].max() + 1
    month = next_week.month
    fut["month_sin"] = np.sin(2 * np.pi * month / 12)
    fut["month_cos"] = np.cos(2 * np.pi * month / 12)
    cat = grid.drop_duplicates("barrio")[["barrio", "barrio_code"]]
    fut = fut.merge(cat, on="barrio")
    fut["pred"] = model.predict(fut[FEATURES_A]).clip(min=0)
    fut["next_week"] = next_week
    return fut


# ---------------------------------------------------------------- Modelo B --
FEATS_B = ["dow", "month_sin", "month_cos", "trend", "es_feriado", "es_vispera",
           "lag_1", "lag_7", "lag_14", "roll_7", "roll_28"]


def daily_features(df: pd.DataFrame) -> pd.DataFrame:
    df["dow"] = df["fecha"].dt.weekday
    df["month_sin"] = np.sin(2 * np.pi * df["fecha"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["fecha"].dt.month / 12)
    df["trend"] = np.arange(len(df))
    df["es_feriado"] = df["fecha"].isin(FERIADOS_TS).astype(int)
    df["es_vispera"] = (df["fecha"] + pd.Timedelta(days=1)).isin(FERIADOS_TS).astype(int)
    for lag in (1, 7, 14):
        df[f"lag_{lag}"] = df["n"].shift(lag)
    df["roll_7"] = df["n"].shift(1).rolling(7).mean()
    df["roll_28"] = df["n"].shift(1).rolling(28).mean()
    return df


def train_model_b():
    daily = pd.read_csv(PROC / "daily_city.csv", parse_dates=["fecha"]).sort_values("fecha")
    daily = daily.set_index("fecha").asfreq("D")
    daily["n"] = daily["n"].fillna(0)
    df = daily_features(daily.reset_index()).dropna().reset_index(drop=True)

    train, test = df.iloc[:-TEST_DAYS], df.iloc[-TEST_DAYS:]

    def make(loss, data, q=None):
        m = HistGradientBoostingRegressor(
            loss=loss, quantile=q, max_iter=400, learning_rate=0.05,
            categorical_features=[0], random_state=42, early_stopping=False,
        )
        m.fit(data[FEATS_B], data["n"])
        return m

    model = make("poisson", train)

    # Banda de incertidumbre con CQR (Conformalized Quantile Regression):
    # los cuantiles crudos del GBM subestiman la dispersion, asi que se
    # calibra un ajuste aditivo en un tramo de calibracion separado y la
    # cobertura se mide en test, que nunca participo de la calibracion.
    train_fit, cal = train.iloc[:-90], train.iloc[-90:]
    m_q10 = make("quantile", train_fit, 0.1)
    m_q90 = make("quantile", train_fit, 0.9)
    c10 = m_q10.predict(cal[FEATS_B])
    c90 = m_q90.predict(cal[FEATS_B])
    scores = np.maximum(c10 - cal["n"], cal["n"] - c90)
    n_cal = len(cal)
    q_hat = float(np.quantile(scores, min(1.0, np.ceil((n_cal + 1) * 0.8) / n_cal)))

    pred = model.predict(test[FEATS_B]).clip(min=0)
    p10 = (m_q10.predict(test[FEATS_B]) - q_hat).clip(min=0)
    p90 = m_q90.predict(test[FEATS_B]) + q_hat
    cov_raw = float(((test["n"] >= m_q10.predict(test[FEATS_B])) &
                     (test["n"] <= m_q90.predict(test[FEATS_B]))).mean())
    naive = test["lag_7"]  # mismo dia de la semana pasada

    metrics = {
        "train_dias": len(train), "test_dias": len(test),
        "test_desde": str(test["fecha"].iloc[0].date()),
        "baseline_mae": round(float(mean_absolute_error(test["n"], naive)), 2),
        "baseline_rmse": round(rmse(test["n"], naive), 2),
        "model_mae": round(float(mean_absolute_error(test["n"], pred)), 2),
        "model_rmse": round(rmse(test["n"], pred), 2),
        "cobertura_sin_calibrar_pct": round(100 * cov_raw, 1),
        "cobertura_q10_q90_pct": round(100 * float(((test["n"] >= p10) & (test["n"] <= p90)).mean()), 1),
        "ajuste_conformal": round(q_hat, 1),
    }
    metrics["mejora_mae_pct"] = round(100 * (1 - metrics["model_mae"] / metrics["baseline_mae"]), 2)
    metrics["mejora_rmse_pct"] = round(100 * (1 - metrics["model_rmse"] / metrics["baseline_rmse"]), 2)

    # Forecast recursivo 14 dias (los cuantiles usan los lags del camino mediano)
    hist = df.copy()
    forecasts = []
    for _ in range(14):
        last = hist.iloc[-1]
        next_date = last["fecha"] + pd.Timedelta(days=1)
        row = {
            "fecha": next_date, "dow": next_date.weekday(),
            "month_sin": np.sin(2 * np.pi * next_date.month / 12),
            "month_cos": np.cos(2 * np.pi * next_date.month / 12),
            "trend": last["trend"] + 1,
            "es_feriado": int(pd.Timestamp(next_date) in FERIADOS_TS),
            "es_vispera": int(pd.Timestamp(next_date + pd.Timedelta(days=1)) in FERIADOS_TS),
            "lag_1": last["n"],
            "lag_7": hist.iloc[-7]["n"] if len(hist) >= 7 else last["n"],
            "lag_14": hist.iloc[-14]["n"] if len(hist) >= 14 else last["n"],
            "roll_7": hist["n"].iloc[-7:].mean(),
            "roll_28": hist["n"].iloc[-28:].mean(),
        }
        X = pd.DataFrame([row])[FEATS_B]
        p = float(model.predict(X).clip(min=0)[0])
        row["n"] = p
        forecasts.append({
            "fecha": str(next_date.date()), "pred": round(p, 1),
            "p10": round(max(0.0, float(m_q10.predict(X)[0]) - q_hat), 1),
            "p90": round(float(m_q90.predict(X)[0]) + q_hat, 1),
        })
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)

    history = [{"fecha": str(d.date()), "n": int(v)}
               for d, v in zip(df["fecha"].iloc[-60:], df["n"].iloc[-60:])]
    return metrics, forecasts, history


# ----------------------------------------------------------------- Alertas --
def build_alerts():
    """Cambio de patron por barrio: ultimas 4 semanas vs 12 previas (z de Poisson)."""
    slots = pd.read_csv(PROC / "slots_weekly.csv", parse_dates=["week_start"])
    bw = slots.groupby(["week_start", "barrio"], as_index=False)["n"].sum()
    weeks = np.sort(bw["week_start"].unique())
    last4, prev12 = weeks[-4:], weeks[-16:-4]

    obs = bw[bw["week_start"].isin(last4)].groupby("barrio")["n"].sum()
    ref = bw[bw["week_start"].isin(prev12)].groupby("barrio")["n"].sum() * (4 / 12)
    serie = (bw[bw["week_start"].isin(weeks[-16:])]
             .pivot(index="barrio", columns="week_start", values="n").fillna(0))

    out = []
    for barrio in obs.index:
        e, o = float(ref.get(barrio, 0)), float(obs[barrio])
        if e < 20:  # barrios/series demasiado chicos: puro ruido
            continue
        z = (o - e) / np.sqrt(e)
        out.append({
            "barrio": barrio, "obs_4sem": int(o), "esperado_4sem": round(e, 1),
            "cambio_pct": round(100 * (o - e) / e, 1), "z": round(float(z), 2),
            "serie_16sem": [int(x) for x in serie.loc[barrio].values],
        })
    out.sort(key=lambda r: r["z"], reverse=True)
    subiendo = [r for r in out if r["z"] >= 2][:6]
    bajando = sorted([r for r in out if r["z"] <= -2], key=lambda r: r["z"])[:6]
    ventana = {
        "reciente": [str(pd.Timestamp(last4[0]).date()), str(pd.Timestamp(last4[-1]).date())],
        "referencia": [str(pd.Timestamp(prev12[0]).date()), str(pd.Timestamp(prev12[-1]).date())],
    }
    return {"subiendo": subiendo, "bajando": bajando, "ventana": ventana}


# ------------------------------------------------------------------- main --
def main() -> None:
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    meta = json.loads((PROC / "meta.json").read_text(encoding="utf-8"))
    profile = json.loads((PROC / "barrio_profile.json").read_text(encoding="utf-8"))
    comuna_profile = json.loads((PROC / "comuna_profile.json").read_text(encoding="utf-8"))

    print("== Modelo A: riesgo por slot (barrio x dia x franja) ==")
    grid = build_slot_grid()
    grid = add_slot_features(grid)
    model_a, metrics_a = train_model_a(grid)
    print(f"  baseline MAE={metrics_a['baseline_mae']}  modelo MAE={metrics_a['model_mae']}"
          f"  (mejora {metrics_a['mejora_mae_pct']}%)  [v0.1: 0.8658]")

    fut = predict_next_week(grid, model_a)

    print("== Modelo B: forecast diario citywide ==")
    metrics_b, forecasts, history = train_model_b()
    print(f"  baseline MAE={metrics_b['baseline_mae']}  modelo MAE={metrics_b['model_mae']}"
          f"  (mejora {metrics_b['mejora_mae_pct']}%)  [v0.1: 40.38]")
    print(f"  cobertura banda q10-q90 en test: {metrics_b['cobertura_q10_q90_pct']}% (ideal ~80%)")

    print("== Alertas de cambio de patron ==")
    alertas = build_alerts()
    print(f"  subiendo: {[a['barrio'] for a in alertas['subiendo']]}")
    print(f"  bajando:  {[a['barrio'] for a in alertas['bajando']]}")

    # --- Ensamblado del JSON para la web ------------------------------------
    geo = json.loads((WEB_DATA / "barrios.geojson").read_text(encoding="utf-8"))
    areas = {f["properties"]["name"]: f["properties"]["area_km2"] for f in geo["features"]}
    geo_c = json.loads((WEB_DATA / "comunas.geojson").read_text(encoding="utf-8"))
    areas_c = {str(f["properties"]["comuna"]): f["properties"]["area_km2"] for f in geo_c["features"]}

    # indice de riesgo 0-100 = percentil del valor esperado entre todos los slots
    all_preds = fut["pred"].to_numpy()
    order = all_preds.argsort().argsort()
    fut["risk_idx"] = (100 * order / (len(all_preds) - 1)).round(1)

    barrios_out = {}
    for barrio, g in fut.groupby("barrio"):
        mat = np.zeros((7, 6))
        idx = np.zeros((7, 6))
        for _, r in g.iterrows():
            mat[int(r["dow"]), int(r["band"])] = round(float(r["pred"]), 2)
            idx[int(r["dow"]), int(r["band"])] = r["risk_idx"]
        p = profile.get(barrio, {})
        barrios_out[barrio] = {
            "comuna": p.get("comuna"),
            "area_km2": areas.get(barrio),
            "total_12m": p.get("total_12m", 0),
            "top_tipos": p.get("top_tipos", []),
            "tipo_share": p.get("tipo_share", {}),
            "uso_arma_pct": p.get("uso_arma_pct"),
            "uso_moto_pct": p.get("uso_moto_pct"),
            "banda_dist": p.get("banda_dist", [0] * 6),
            "esperado": mat.tolist(),
            "riesgo": idx.tolist(),
        }

    # --- Agregacion por comuna (unidad operativa de la Policia de la Ciudad) --
    comunas_out = {}
    for barrio, b in barrios_out.items():
        c = str(b["comuna"])
        if c not in comunas_out:
            comunas_out[c] = {"esperado": np.zeros((7, 6)), "barrios": []}
        comunas_out[c]["esperado"] += np.array(b["esperado"])
        comunas_out[c]["barrios"].append(barrio)

    flat = np.concatenate([v["esperado"].ravel() for v in comunas_out.values()])
    orden_c = flat.argsort().argsort().reshape(len(comunas_out), 7, 6)
    for i, (c, v) in enumerate(comunas_out.items()):
        cp = comuna_profile.get(c, {})
        comunas_out[c] = {
            "area_km2": areas_c.get(c),
            "barrios": sorted(v["barrios"]),
            "total_12m": cp.get("total_12m", 0),
            "top_tipos": cp.get("top_tipos", []),
            "tipo_share": cp.get("tipo_share", {}),
            "uso_arma_pct": cp.get("uso_arma_pct"),
            "uso_moto_pct": cp.get("uso_moto_pct"),
            "banda_dist": cp.get("banda_dist", [0] * 6),
            "esperado": np.round(v["esperado"], 2).tolist(),
            "riesgo": np.round(100 * orden_c[i] / (flat.size - 1), 1).tolist(),
        }

    predictions = {
        "meta": {
            "version": "0.2",
            "generado": str(pd.Timestamp.now().date()),
            "datos_desde": meta["fecha_min"], "datos_hasta": meta["fecha_max"],
            "semana_predicha": str(fut["next_week"].iloc[0].date()),
            "bandas": BAND_LABELS, "dias": DOW_LABELS,
            "fuente": "BA Data - Ministerio de Justicia y Seguridad GCBA",
        },
        "barrios": barrios_out,
        "comunas": comunas_out,
        "alertas": alertas,
        "citywide": {"historia": history, "forecast": forecasts},
    }
    (WEB_DATA / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False), encoding="utf-8")

    metrics = {
        "modelo_slots": metrics_a,
        "modelo_diario": metrics_b,
        "alertas": {"subiendo": len(alertas["subiendo"]), "bajando": len(alertas["bajando"])},
        "calidad_datos": meta["calidad"],
        "rango_datos": {"desde": meta["fecha_min"], "hasta": meta["fecha_max"]},
    }
    (WEB_DATA / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")

    print("== train.py OK ==")


if __name__ == "__main__":
    main()
