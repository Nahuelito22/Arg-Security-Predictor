"""Prepara los datos crudos de delitos de CABA (BA Data) para modelado.

Entradas:  data/raw/delitos_{2022..2025}.csv, data/raw/barrios.geojson
Salidas:   data/processed/incidents_clean.csv   (incidente a incidente, slim)
           data/processed/slots_weekly.csv      (week_start, barrio, dow, band, n)
           data/processed/daily_city.csv        (fecha, n)
           data/processed/barrio_profile.json   (perfil descriptivo por barrio)
           data/processed/meta.json             (rango de fechas, calidad de datos)
           web/data/barrios.geojson             (geometrías con nombre normalizado + area_km2)

Uso:  python src/prepare.py
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
WEB_DATA = ROOT / "web" / "data"

YEARS = [2022, 2023, 2024, 2025]
NA_VALUES = ["NULL", "null", "S/D", "SD", "", "NA", "sd"]

# Franjas de 4 horas: 0 -> 00-04, 1 -> 04-08, ..., 5 -> 20-24
BAND_LABELS = ["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"]
DOW_LABELS = ["LUN", "MAR", "MIE", "JUE", "VIE", "SAB", "DOM"]

# Correcciones manuales de nombres CSV -> geojson (detectadas por el reporte
# de cruces al final de este script).
MANUAL_MAP: dict[str, str] = {"BOCA": "LA BOCA"}


def norm_name(s: str) -> str:
    """Normaliza nombre de barrio: mayúsculas, sin acentos (conserva Ñ), espacios simples."""
    s = str(s).strip().upper()
    s = s.replace("Ñ", "\x00")  # proteger la Ñ antes de quitar acentos
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.replace("\x00", "Ñ")
    s = re.sub(r"\s+", " ", s)
    return s


def polygon_area_km2(coords: list) -> float:
    """Área aproximada (shoelace sobre grados escalados) para polígonos chicos como CABA."""
    lat0 = math.radians(-34.61)
    kx = 111.32 * math.cos(lat0)  # km por grado de longitud
    ky = 110.57                   # km por grado de latitud
    area = 0.0
    ring = coords[0]  # anillo exterior
    for i in range(len(ring) - 1):
        x1, y1 = ring[i][0] * kx, ring[i][1] * ky
        x2, y2 = ring[i + 1][0] * kx, ring[i + 1][1] * ky
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def load_raw() -> pd.DataFrame:
    frames = []
    for y in YEARS:
        f = RAW / f"delitos_{y}.csv"
        df = pd.read_csv(f, na_values=NA_VALUES, keep_default_na=True, dtype=str)
        df["__source_year"] = y
        frames.append(df)
        print(f"  leido {f.name}: {len(df):,} filas")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    WEB_DATA.mkdir(parents=True, exist_ok=True)

    print("== Carga de CSVs crudos ==")
    df = load_raw()
    n_total = len(df)

    # --- Tipado y limpieza -------------------------------------------------
    df["fecha"] = pd.to_datetime(df["fecha"], format="%Y-%m-%d", errors="coerce")
    df["franja"] = pd.to_numeric(df["franja"], errors="coerce")
    df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce").fillna(1).astype(int)
    df["latitud"] = pd.to_numeric(df["latitud"], errors="coerce").replace(0, pd.NA)
    df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce").replace(0, pd.NA)

    drop_fecha = df["fecha"].isna()
    drop_barrio = df["barrio"].isna()
    drop_franja = df["franja"].isna() | ~df["franja"].between(0, 23)

    quality = {
        "filas_totales": int(n_total),
        "sin_fecha": int(drop_fecha.sum()),
        "sin_barrio": int(drop_barrio.sum()),
        "sin_franja_valida": int(drop_franja.sum()),
    }
    keep = ~(drop_fecha | drop_barrio | drop_franja)
    df = df.loc[keep].copy()
    quality["filas_usadas"] = int(len(df))
    quality["pct_descartado"] = round(100 * (1 - len(df) / n_total), 2)

    df["barrio"] = df["barrio"].map(norm_name)
    df["barrio"] = df["barrio"].replace(MANUAL_MAP)
    df["dow"] = df["fecha"].dt.weekday  # 0=lunes
    df["band"] = (df["franja"] // 4).astype(int)
    df["week_start"] = df["fecha"] - pd.to_timedelta(df["fecha"].dt.weekday, unit="D")

    # Semana final incompleta distorsiona los conteos semanales: se recorta.
    max_fecha = df["fecha"].max()
    last_complete_week = (max_fecha - pd.Timedelta(days=6)) - pd.to_timedelta(
        (max_fecha - pd.Timedelta(days=6)).weekday(), unit="D"
    )
    df_weekly = df[df["week_start"] <= last_complete_week]

    print(f"  rango de fechas: {df['fecha'].min().date()} -> {max_fecha.date()}")
    print(f"  ultima semana completa: {last_complete_week.date()}")
    print(f"  calidad: {quality}")

    # --- Salidas -----------------------------------------------------------
    slim_cols = ["fecha", "franja", "band", "dow", "week_start", "tipo", "subtipo",
                 "uso_arma", "uso_moto", "barrio", "comuna", "latitud", "longitud", "cantidad"]
    df[slim_cols].to_csv(OUT / "incidents_clean.csv", index=False)

    slots = (df_weekly.groupby(["week_start", "barrio", "dow", "band"], observed=True)["cantidad"]
             .sum().reset_index(name="n"))
    slots.to_csv(OUT / "slots_weekly.csv", index=False)
    print(f"  slots_weekly: {len(slots):,} filas (solo slots con >=1 incidente; el grid completo se arma en train)")

    daily = df.groupby("fecha")["cantidad"].sum().reset_index(name="n")
    daily.to_csv(OUT / "daily_city.csv", index=False)

    # --- Perfil por barrio y por comuna (para el panel de la web) -----------
    # Se calcula sobre los ultimos 12 meses para reflejar composicion actual.
    cutoff = max_fecha - pd.Timedelta(days=365)
    recent = df[df["fecha"] >= cutoff]

    def profile_of(g: pd.DataFrame) -> dict:
        total = int(g["cantidad"].sum())
        tipos = g.groupby("tipo")["cantidad"].sum().sort_values(ascending=False)
        return {
            "total_12m": total,
            "top_tipos": [{"tipo": t, "pct": round(100 * v / total, 1)} for t, v in tipos.head(4).items()],
            "tipo_share": {t: round(float(v) / total, 4) for t, v in tipos.items()},
            "uso_arma_pct": round(100 * float((g["uso_arma"] == "SI").mul(g["cantidad"]).sum()) / total, 1),
            "uso_moto_pct": round(100 * float((g["uso_moto"] == "SI").mul(g["cantidad"]).sum()) / total, 1),
            "banda_dist": [int(x) for x in g.groupby("band")["cantidad"].sum().reindex(range(6), fill_value=0).values],
        }

    profile: dict[str, dict] = {}
    for barrio, g in recent.groupby("barrio"):
        p = profile_of(g)
        p["comuna"] = (int(pd.to_numeric(g["comuna"], errors="coerce").dropna().mode().iloc[0])
                       if g["comuna"].notna().any() else None)
        profile[barrio] = p
    (OUT / "barrio_profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  perfiles de barrio: {len(profile)}")

    recent_c = recent.assign(comuna_n=pd.to_numeric(recent["comuna"], errors="coerce"))
    comuna_profile = {str(int(c)): profile_of(g) for c, g in recent_c.dropna(subset=["comuna_n"]).groupby("comuna_n")}
    (OUT / "comuna_profile.json").write_text(json.dumps(comuna_profile, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  perfiles de comuna: {len(comuna_profile)}")

    # --- GeoJSON de barrios -------------------------------------------------
    gj = json.loads((RAW / "barrios.geojson").read_text(encoding="utf-8"))
    name_key = None
    sample_props = gj["features"][0]["properties"]
    for k in sample_props:
        if k.lower() in ("barrio", "nombre", "name"):
            name_key = k
            break
    if name_key is None:
        raise SystemExit(f"No encuentro campo de nombre en geojson. Props: {list(sample_props)}")

    geo_names = set()
    out_features = []
    for feat in gj["features"]:
        props = feat["properties"]
        name = norm_name(props[name_key])
        geo_names.add(name)
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            area = polygon_area_km2(geom["coordinates"])
        else:  # MultiPolygon
            area = sum(polygon_area_km2(p) for p in geom["coordinates"])
        area_prop = None
        for k in props:
            if k.lower() == "area":
                area_prop = pd.to_numeric(props[k], errors="coerce")
                break
        if area_prop and not pd.isna(area_prop) and area_prop > 1000:
            area = float(area_prop) / 1e6  # m2 -> km2
        out_features.append({
            "type": "Feature",
            "properties": {"name": name, "comuna": props.get("COMUNA") or props.get("comuna"), "area_km2": round(area, 2)},
            "geometry": geom,
        })
    (WEB_DATA / "barrios.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": out_features}, ensure_ascii=False),
        encoding="utf-8")

    # --- GeoJSON de comunas ---------------------------------------------------
    cj = json.loads((RAW / "comunas.geojson").read_text(encoding="utf-8"))
    comuna_features = []
    for feat in cj["features"]:
        props = feat["properties"]
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            area = polygon_area_km2(geom["coordinates"])
        else:
            area = sum(polygon_area_km2(p) for p in geom["coordinates"])
        area_prop = pd.to_numeric(props.get("area"), errors="coerce")
        if area_prop and not pd.isna(area_prop) and area_prop > 1000:
            area = float(area_prop) / 1e6
        comuna_features.append({
            "type": "Feature",
            "properties": {"comuna": int(props["comuna"]), "barrios": props.get("barrios", ""),
                           "area_km2": round(area, 2)},
            "geometry": geom,
        })
    (WEB_DATA / "comunas.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": comuna_features}, ensure_ascii=False),
        encoding="utf-8")
    print(f"  comunas geojson: {len(comuna_features)} features")

    # --- Cruce de nombres CSV vs geojson -------------------------------------
    csv_names = set(df["barrio"].unique())
    only_csv = sorted(csv_names - geo_names)
    only_geo = sorted(geo_names - csv_names)
    if only_csv or only_geo:
        print("  [AVISO] nombres sin cruzar ->")
        print(f"    solo en CSV:     {only_csv}")
        print(f"    solo en geojson: {only_geo}")
    else:
        print("  cruce de nombres CSV/geojson: 100% OK")

    meta = {
        "fecha_min": str(df["fecha"].min().date()),
        "fecha_max": str(max_fecha.date()),
        "ultima_semana_completa": str(last_complete_week.date()),
        "calidad": quality,
        "barrios": len(csv_names),
        "nombres_sin_cruzar_csv": only_csv,
        "nombres_sin_cruzar_geojson": only_geo,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("== prepare.py OK ==")


if __name__ == "__main__":
    main()
