from __future__ import annotations

from io import BytesIO
import zipfile

from pathlib import Path

import pandas as pd
import numpy as np
import requests

from bvg_core.data import aggregate_trade_day, load_master_dataset
from .config import (
    BVG_COLUMN_MAP,
    BVG_NUMERIC_COLUMNS,
    BVG_REQUIRED_COLUMNS,
)


def get_company_history(
    df: pd.DataFrame, company: str, tail_rows: int = 30
) -> pd.DataFrame:
    history = df[df["empresa"] == company].sort_values("fecha").copy()
    if history.empty:
        raise ValueError(f"No se encontraron filas para la empresa: {company}")
    if len(history) < tail_rows:
        raise ValueError(
            f"La empresa {company} tiene {len(history)} filas (<{tail_rows})."
        )
    return history.tail(tail_rows).copy()


def build_input_row(
    fecha: str,
    close_last: float,
    close_vwap: float,
    volume_shares_day: float,
    turnover_value_day: float,
    n_trades_day: int,
    company: str,
) -> pd.DataFrame:
    row = {
        "fecha": pd.to_datetime(fecha, errors="coerce"),
        "empresa": company,
        "close_last": close_last,
        "close_vwap": close_vwap,
        "volume_shares_day": volume_shares_day,
        "turnover_value_day": turnover_value_day,
        "n_trades_day": n_trades_day,
    }
    if pd.isna(row["fecha"]):
        raise ValueError("La fecha ingresada es inválida.")
    return pd.DataFrame([row])


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[^0-9.,-]", "", regex=True)
    has_dot = cleaned.str.contains(r"\.", regex=True)
    has_comma = cleaned.str.contains(",", regex=False)
    both_mask = has_dot & has_comma
    cleaned = cleaned.where(~both_mask, cleaned.str.replace(",", "", regex=False))
    comma_only = (~has_dot) & has_comma
    cleaned = cleaned.where(~comma_only, cleaned.str.replace(",", ".", regex=False))
    return pd.to_numeric(cleaned, errors="coerce")


def download_and_clean_bvg(
    url: str,
    target_issuers: list[str],
    *,
    timeout_s: int = 30,
) -> pd.DataFrame:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(
            url,
            timeout=timeout_s,
            headers=headers,
            allow_redirects=True,
        )
        if not response.ok:
            raise ConnectionError(
                "Fallo de comunicación con BVG. "
                f"HTTP {response.status_code} al descargar el archivo."
            )
    except requests.RequestException as exc:
        raise ConnectionError(
            "Fallo de comunicación con BVG. "
            f"No se pudo descargar el archivo: {exc}"
        ) from exc

    raw_bytes = response.content
    try:
        raw_df = pd.read_excel(BytesIO(raw_bytes), engine="openpyxl", skiprows=2)
    except Exception as exc:  # noqa: BLE001
        error_msg = str(exc).lower()
        if "stylesheet" in error_msg or "style" in error_msg:
            try:
                cleaned_bytes = _strip_styles_xml(raw_bytes)
                raw_df = pd.read_excel(
                    BytesIO(cleaned_bytes), engine="openpyxl", skiprows=2
                )
            except Exception as retry_exc:  # noqa: BLE001
                raise ValueError(
                    "Error al parsear el XLSX de BVG tras limpiar styles.xml: "
                    f"{retry_exc}"
                ) from retry_exc
        else:
            raise ValueError(
                "Error al parsear el XLSX de BVG: "
                f"{exc}"
            ) from exc

    raw_df = raw_df.copy()
    missing = BVG_REQUIRED_COLUMNS.difference(raw_df.columns)
    if missing:
        raise ValueError(
            "El archivo BVG no contiene columnas requeridas: "
            f"{sorted(missing)}"
        )

    df = raw_df.loc[:, list(BVG_REQUIRED_COLUMNS)].copy()
    df = df.rename(columns=BVG_COLUMN_MAP)
    df["fecha"] = pd.to_datetime(df["fecha"], origin="1899-12-30", unit="D", errors="coerce")
    df["empresa"] = df["empresa"].astype(str).str.strip()

    if df["empresa"].isna().any() or df["fecha"].isna().any():
        df = df.loc[df["empresa"].notna() & df["fecha"].notna()].copy()

    for col in BVG_NUMERIC_COLUMNS:
        df.loc[:, col] = _clean_numeric_series(df[col])
    df = df.loc[df["empresa"].isin(target_issuers)].copy()
    required_fields = ["fecha", "empresa", *BVG_NUMERIC_COLUMNS]
    df = df.loc[df[required_fields].notna().all(axis=1)].copy()

    if df.empty:
        raise ValueError("No se encontraron filas válidas para emisores objetivo.")

    return df.copy()


def _strip_styles_xml(raw_bytes: bytes) -> bytes:
    buffer_in = BytesIO(raw_bytes)
    buffer_out = BytesIO()
    with zipfile.ZipFile(buffer_in, "r") as zin, zipfile.ZipFile(
        buffer_out, "w", compression=zipfile.ZIP_DEFLATED
    ) as zout:
        for item in zin.infolist():
            if item.filename == "xl/styles.xml":
                continue
            with zin.open(item) as src:
                zout.writestr(item, src.read())
    return buffer_out.getvalue()


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("No hay datos BVG para agregar.")

    work_df = df.copy()
    work_df["fecha"] = pd.to_datetime(work_df["fecha"], errors="coerce")
    work_df = work_df.loc[work_df["fecha"].notna()].copy()
    work_df["fecha"] = work_df["fecha"].dt.normalize()
    work_df = work_df.sort_values(["empresa", "fecha"])

    aggregated = (
        work_df.groupby(["empresa", "fecha"])
        .apply(aggregate_trade_day)
        .reset_index()
    )

    if aggregated.empty:
        raise ValueError("La agregación diaria no produjo resultados.")

    return aggregated


__all__ = [
    "load_master_dataset",
    "get_company_history",
    "build_input_row",
    "download_and_clean_bvg",
    "aggregate_daily",
    "aggregate_trade_day",
]
