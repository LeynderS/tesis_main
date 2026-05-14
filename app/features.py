from __future__ import annotations

import pandas as pd

from .config import HORIZONS
from bvg_core.features import build_features_for_company as _build_features_for_company


def build_features_for_company(d: pd.DataFrame) -> pd.DataFrame:
    return _build_features_for_company(d, horizons=HORIZONS)


__all__ = ["build_features_for_company"]
