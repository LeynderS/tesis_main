from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from bvg_core.config import (
    FECHA_COL_2,
    CLOSE_LAST_COL,
    VOLUME_SHARES_DAY_COL,
    TURNOVER_VALUE_DAY_COL,
    N_TRADES_DAY_COL,
)


def build_features_for_company(
    d: pd.DataFrame, *, horizons: Sequence[int] | None = None
) -> pd.DataFrame:
    HORIZONS = horizons or [5]

    d = d.sort_values(FECHA_COL_2).reset_index(drop=True).copy()

    d["ret_1d"] = np.log(d[CLOSE_LAST_COL] / d[CLOSE_LAST_COL].shift(1))

    # Memoria corta (solo info hasta t-1)
    d["ret_lag_1"] = d["ret_1d"].shift(1)
    d["ret_lag_2"] = d["ret_1d"].shift(2)
    d["ret_lag_3"] = d["ret_1d"].shift(3)

    d["mom_3"] = d["ret_1d"].shift(1).rolling(3).sum()
    d["mom_5"] = d["ret_1d"].shift(1).rolling(5).sum()
    d["mom_10"] = d["ret_1d"].shift(1).rolling(10).sum()

    # Volatilidad y regimen
    d["vol_5"] = d["ret_1d"].shift(1).rolling(5).std()
    d["vol_10"] = d["ret_1d"].shift(1).rolling(10).std()
    d["regime_vol_ratio"] = d["vol_5"] / (d["vol_10"] + 1e-12)

    # Tendencia de precio
    d["ma_5"] = d[CLOSE_LAST_COL].shift(1).rolling(5).mean()
    d["ma_10"] = d[CLOSE_LAST_COL].shift(1).rolling(10).mean()
    d["ma_gap"] = d["ma_5"] - d["ma_10"]
    d["price_vs_ma10"] = d[CLOSE_LAST_COL].shift(1) / (d["ma_10"] + 1e-12) - 1.0

    # RSI 14 sobre retornos previos
    ret_prev = d["ret_1d"].shift(1)
    gain = ret_prev.clip(lower=0)
    loss = -ret_prev.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    d["rsi_14"] = 100 - (100 / (1 + rs))

    # Liquidez/microestructura
    d["turnover_log1p"] = np.log1p(d[TURNOVER_VALUE_DAY_COL])
    d["volume_log1p"] = np.log1p(d[VOLUME_SHARES_DAY_COL])
    d["avg_trade_size"] = d[VOLUME_SHARES_DAY_COL] / (d[N_TRADES_DAY_COL] + 1e-12)
    d["avg_trade_size_log1p"] = np.log1p(d["avg_trade_size"])

    illiq_raw = d["ret_1d"].abs() / (d[TURNOVER_VALUE_DAY_COL] + 1.0)
    d["amihud_5"] = illiq_raw.shift(1).rolling(5).mean()

    gap_days = d[FECHA_COL_2].diff().dt.days
    fallback_gap = gap_days.median() if gap_days.notna().any() else 1
    d["days_since_trade"] = gap_days.fillna(fallback_gap).clip(lower=1)

    # Targets multihorizonte base para clasificacion direccional
    for h in HORIZONS:
        d[f"ret_fwd_h{h}"] = np.log(d[CLOSE_LAST_COL].shift(-h) / d[CLOSE_LAST_COL])
        d[f"target_up_h{h}"] = np.where(
            d[f"ret_fwd_h{h}"].notna(),
            (d[f"ret_fwd_h{h}"] > 0).astype(int),
            np.nan,
        )

    return d


__all__ = ["build_features_for_company"]
