from __future__ import annotations

from typing import Sequence

import pandas as pd


def temporal_split_company(
    df_in: pd.DataFrame,
    company: str,
    *,
    test_size: int = 30,
    company_col: str = "empresa",
    date_col: str = "fecha",
    target_col: str = "target",
    feature_cols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if feature_cols is None:
        raise ValueError("feature_cols es requerido para el split temporal.")

    d = (
        df_in.loc[df_in[company_col] == company]
        .sort_values(date_col, kind="stable")
        .reset_index(drop=True)
        .copy()
    )
    if len(d) <= test_size:
        raise ValueError(
            f"Datos insuficientes para {company}: n={len(d)} test_size={test_size}"
        )

    cutoff = len(d) - test_size
    tr = d.iloc[:cutoff].copy()
    te = d.iloc[cutoff:].copy()

    X_train = tr.loc[:, list(feature_cols)].copy()
    y_train = tr.loc[:, target_col].astype(int).copy()
    X_test = te.loc[:, list(feature_cols)].copy()
    y_test = te.loc[:, target_col].astype(int).copy()

    return tr, te, X_train, y_train, X_test, y_test


__all__ = ["temporal_split_company"]
