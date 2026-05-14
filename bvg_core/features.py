from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def build_features_for_company(
    d: pd.DataFrame, *, horizons: Sequence[int] | None = None
) -> pd.DataFrame:
    HORIZONS = horizons or [5]

    d = d.sort_values("fecha").reset_index(drop=True).copy()

    d["ret_1d"] = np.log(d["close_last"] / d["close_last"].shift(1))

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
    d["ma_5"] = d["close_last"].shift(1).rolling(5).mean()
    d["ma_10"] = d["close_last"].shift(1).rolling(10).mean()
    d["ma_gap"] = d["ma_5"] - d["ma_10"]
    d["price_vs_ma10"] = d["close_last"].shift(1) / (d["ma_10"] + 1e-12) - 1.0

    # RSI 14 sobre retornos previos
    ret_prev = d["ret_1d"].shift(1)
    gain = ret_prev.clip(lower=0)
    loss = -ret_prev.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    d["rsi_14"] = 100 - (100 / (1 + rs))

    # Liquidez/microestructura
    d["turnover_log1p"] = np.log1p(d["turnover_value_day"])
    d["volume_log1p"] = np.log1p(d["volume_shares_day"])
    d["avg_trade_size"] = d["volume_shares_day"] / (d["n_trades_day"] + 1e-12)
    d["avg_trade_size_log1p"] = np.log1p(d["avg_trade_size"])

    illiq_raw = d["ret_1d"].abs() / (d["turnover_value_day"] + 1.0)
    d["amihud_5"] = illiq_raw.shift(1).rolling(5).mean()

    gap_days = d["fecha"].diff().dt.days
    fallback_gap = gap_days.median() if gap_days.notna().any() else 1
    d["days_since_trade"] = gap_days.fillna(fallback_gap).clip(lower=1)

    # Targets multihorizonte base para clasificacion direccional
    for h in HORIZONS:
        d[f"ret_fwd_h{h}"] = np.log(d["close_last"].shift(-h) / d["close_last"])
        d[f"target_up_h{h}"] = np.where(
            d[f"ret_fwd_h{h}"].notna(),
            (d[f"ret_fwd_h{h}"] > 0).astype(int),
            np.nan,
        )

    return d


__all__ = ["build_features_for_company"]
