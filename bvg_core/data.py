from __future__ import annotations

from pathlib import Path

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


__all__ = [
    "load_master_dataset",
    "aggregate_trade_day",
]
