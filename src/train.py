"""Entrena y evalua los modelos del MVP y exporta los artefactos para la web.

Modelo A (espacial-temporal): conteo semanal esperado por slot (barrio, dia, franja 4h).
  - Baseline: media historica del slot en train.
  - Modelo:   HistGradientBoostingRegressor con loss="poisson" + features de
              calendario, tendencia y lags por slot.
Modelo B (temporal citywide): total diario de incidentes de la ciudad.
  - Baseline: naive estacional (mismo dia de la semana anterior).
  - Modelo:   HistGradientBoostingRegressor + lags/rolling.

Evaluacion con split temporal (nada de shuffle): test = ultimas 26 semanas (A)
y ultimos 90 dias (B). Metricas: MAE y RMSE contra baseline.

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

rng = np.random.RandomState(42)


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
    grid["barrio_code"] = grid["barrio"].astype("category").cat.codes
    return grid.dropna(subset=["lag_1", "roll_4", "roll_8"]).reset_index(drop=True)


FEATURES_A = ["barrio_code", "dow", "band", "trend", "month_sin", "month_cos",
              "lag_1", "roll_4", "roll_8"]


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
def train_model_b():
    daily = pd.read_csv(PROC / "daily_city.csv", parse_dates=["fecha"]).sort_values("fecha")
    daily = daily.set_index("fecha").asfreq("D")
    daily["n"] = daily["n"].fillna(0)
    df = daily.reset_index()
    df["dow"] = df["fecha"].dt.weekday
    df["month_sin"] = np.sin(2 * np.pi * df["fecha"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["fecha"].dt.month / 12)
    df["trend"] = np.arange(len(df))
    for lag in (1, 7, 14):
        df[f"lag_{lag}"] = df["n"].shift(lag)
    df["roll_7"] = df["n"].shift(1).rolling(7).mean()
    df["roll_28"] = df["n"].shift(1).rolling(28).mean()
    df = df.dropna().reset_index(drop=True)

    feats = ["dow", "month_sin", "month_cos", "trend", "lag_1", "lag_7", "lag_14", "roll_7", "roll_28"]
    train, test = df.iloc[:-TEST_DAYS], df.iloc[-TEST_DAYS:]

    model = HistGradientBoostingRegressor(
        loss="poisson", max_iter=400, learning_rate=0.05,
        categorical_features=[0], random_state=42, early_stopping=False,
    )
    model.fit(train[feats], train["n"])
    pred = model.predict(test[feats]).clip(min=0)
    naive = test["lag_7"]  # mismo dia de la semana pasada

    metrics = {
        "train_dias": len(train), "test_dias": len(test),
        "test_desde": str(test["fecha"].iloc[0].date()),
        "baseline_mae": round(float(mean_absolute_error(test["n"], naive)), 2),
        "baseline_rmse": round(rmse(test["n"], naive), 2),
        "model_mae": round(float(mean_absolute_error(test["n"], pred)), 2),
        "model_rmse": round(rmse(test["n"], pred), 2),
    }
    metrics["mejora_mae_pct"] = round(100 * (1 - metrics["model_mae"] / metrics["baseline_mae"]), 2)
    metrics["mejora_rmse_pct"] = round(100 * (1 - metrics["model_rmse"] / metrics["baseline_rmse"]), 2)

    # Forecast recursivo 14 dias
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
            "lag_1": last["n"],
            "lag_7": hist.iloc[-7]["n"] if len(hist) >= 7 else last["n"],
            "lag_14": hist.iloc[-14]["n"] if len(hist) >= 14 else last["n"],
            "roll_7": hist["n"].iloc[-7:].mean(),
            "roll_28": hist["n"].iloc[-28:].mean(),
        }
        p = float(model.predict(pd.DataFrame([row])[feats]).clip(min=0)[0])
        row["n"] = p
        forecasts.append({"fecha": str(next_date.date()), "pred": round(p, 1)})
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)

    history = [{"fecha": str(d.date()), "n": int(v)}
               for d, v in zip(df["fecha"].iloc[-60:], df["n"].iloc[-60:])]
    return metrics, forecasts, history


# ------------------------------------------------------------------- main --
def main() -> None:
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    meta = json.loads((PROC / "meta.json").read_text(encoding="utf-8"))
    profile = json.loads((PROC / "barrio_profile.json").read_text(encoding="utf-8"))

    print("== Modelo A: riesgo por slot (barrio x dia x franja) ==")
    grid = build_slot_grid()
    grid = add_slot_features(grid)
    model_a, metrics_a = train_model_a(grid)
    print(f"  baseline MAE={metrics_a['baseline_mae']}  modelo MAE={metrics_a['model_mae']}"
          f"  (mejora {metrics_a['mejora_mae_pct']}%)")

    fut = predict_next_week(grid, model_a)

    print("== Modelo B: forecast diario citywide ==")
    metrics_b, forecasts, history = train_model_b()
    print(f"  baseline MAE={metrics_b['baseline_mae']}  modelo MAE={metrics_b['model_mae']}"
          f"  (mejora {metrics_b['mejora_mae_pct']}%)")

    # --- Ensamblado del JSON para la web ------------------------------------
    geo = json.loads((WEB_DATA / "barrios.geojson").read_text(encoding="utf-8"))
    areas = {f["properties"]["name"]: f["properties"]["area_km2"] for f in geo["features"]}

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
            "uso_arma_pct": p.get("uso_arma_pct"),
            "banda_dist": p.get("banda_dist", [0] * 6),
            "esperado": mat.tolist(),
            "riesgo": idx.tolist(),
        }

    predictions = {
        "meta": {
            "generado": str(pd.Timestamp.now().date()),
            "datos_desde": meta["fecha_min"], "datos_hasta": meta["fecha_max"],
            "semana_predicha": str(fut["next_week"].iloc[0].date()),
            "bandas": BAND_LABELS, "dias": DOW_LABELS,
            "fuente": "BA Data - Ministerio de Justicia y Seguridad GCBA",
        },
        "barrios": barrios_out,
        "citywide": {"historia": history, "forecast": forecasts},
    }
    (WEB_DATA / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False), encoding="utf-8")

    metrics = {
        "modelo_slots": metrics_a,
        "modelo_diario": metrics_b,
        "calidad_datos": meta["calidad"],
        "rango_datos": {"desde": meta["fecha_min"], "hasta": meta["fecha_max"]},
    }
    (WEB_DATA / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")

    print("== train.py OK ==")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
