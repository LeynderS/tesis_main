from __future__ import annotations

from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np

from .config import REQUIRED_RAW_COLUMNS


def load_master_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset procesado en {path.as_posix()}"
        )

    df = pd.read_csv(path)
    if "fecha" not in df.columns:
        raise ValueError("El dataset no contiene la columna 'fecha'.")

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    if df["fecha"].isna().any():
        raise ValueError("Se detectaron fechas inválidas en el dataset base.")

    missing = REQUIRED_RAW_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en dataset base: {sorted(missing)}")

    return df.copy()


def aggregate_trade_day(g: pd.DataFrame) -> pd.Series:
    shares = float(g["numero_acciones"].sum()) if g["numero_acciones"].notna().any() else np.nan
    turnover = float(g["valor_efecto"].sum()) if g["valor_efecto"].notna().any() else np.nan

    if pd.notna(shares) and shares > 0 and pd.notna(turnover):
        vwap = turnover / shares
    else:
        vwap = float(g["precio"].iloc[-1])

    return pd.Series({
        'close_last': float(g["precio"].iloc[-1]),
        'close_vwap': float(vwap),
        'volume_shares_day': shares,
        'turnover_value_day': turnover,
        'n_trades_day': int(len(g))
    })


def get_friday_data_with_fallback(
    price_df: pd.DataFrame,
    company: str,
    target_friday: pd.Timestamp,
    *,
    max_back_days: int = 4,
) -> tuple[pd.Timestamp | None, float | None, str]:
    if price_df.empty:
        return None, None, "Datos insuficientes"

    company_df = price_df.loc[price_df["empresa"] == company].copy()
    if company_df.empty:
        return None, None, "Datos insuficientes"

    company_df["fecha"] = pd.to_datetime(company_df["fecha"], errors="coerce").dt.normalize()
    company_df = company_df.loc[company_df["fecha"].notna()].copy()
    company_df = company_df.loc[company_df["fecha"] <= target_friday].copy()
    if company_df.empty:
        return None, None, "Datos insuficientes"

    close_by_date = (
        company_df.sort_values("fecha")
        .groupby("fecha", as_index=True)["close_last"]
        .last()
    )

    target_friday = pd.to_datetime(target_friday).normalize()
    for delta in range(0, max_back_days + 1):
        candidate = target_friday - timedelta(days=delta)
        if candidate in close_by_date.index:
            close_val = float(close_by_date.loc[candidate])
            weekday = candidate.strftime("%A").lower()
            weekday_map = {
                "monday": "lunes",
                "tuesday": "martes",
                "wednesday": "miércoles",
                "thursday": "jueves",
                "friday": "viernes",
                "saturday": "sábado",
                "sunday": "domingo",
            }
            weekday_es = weekday_map.get(weekday, weekday)
            if delta == 0:
                obs = f"Datos del día {weekday_es} ({candidate.date().isoformat()})."
            else:
                obs = (
                    "Usando datos del día "
                    f"{weekday_es} ({candidate.date().isoformat()}) "
                    "por ausencia en viernes."
                )
            return candidate, close_val, obs

    if not close_by_date.empty:
        last_available = close_by_date.index.max()
        close_val = float(close_by_date.loc[last_available])
        return (
            last_available,
            close_val,
            f"Reutilizando último dato disponible ({last_available.date().isoformat()}) "
            "por ausencia total en semana objetivo.",
        )

    return None, None, "Datos insuficientes"


def compute_target_date(trading_dates: pd.Series, fecha_t: pd.Timestamp) -> pd.Timestamp:
    """Regla t+5: viernes inmediato siguiente; si no hay transacciones, jueves previo."""
    if trading_dates.empty:
        raise ValueError("No hay fechas de trading disponibles para calcular t+5.")
    trading_dates = pd.to_datetime(trading_dates).dt.normalize()
    fecha_t = pd.to_datetime(fecha_t).normalize()
    weekday = fecha_t.weekday()  # Monday=0 ... Friday=4
    days_to_next_friday = (4 - weekday) % 7
    if days_to_next_friday == 0:
        days_to_next_friday = 7
    target_friday = fecha_t + timedelta(days=days_to_next_friday)
    if target_friday in set(trading_dates):
        return target_friday

    thursday = target_friday - timedelta(days=1)
    if thursday in set(trading_dates):
        return thursday

    previous_dates = trading_dates[trading_dates < target_friday]
    if previous_dates.empty:
        raise ValueError("No hay fecha previa disponible para resolver t+5.")
    return previous_dates.max()


__all__ = [
    "load_master_dataset",
    "aggregate_trade_day",
    "get_friday_data_with_fallback",
    "compute_target_date",
]
