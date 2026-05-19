from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from bvg_core.config import (
    EMPRESA_COL,
    FECHA_COL_2,
    PROCESSED_DIR,
    REQUIRED_RAW_COLUMNS,
)
from bvg_core.features import build_features_for_company


def merge_master_with_live(master_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """Merge historical master with live daily data at aggregated-daily level.

    Extracts only the base columns from ``master_df``, concatenates with
    ``daily_df``, deduplicates by ``(empresa, fecha)`` keeping the live
    row on conflict, and sorts by ``fecha`` ascending.
    """
    base_cols = list(REQUIRED_RAW_COLUMNS)

    master_base = master_df.loc[
        :, [c for c in base_cols if c in master_df.columns]
    ].copy()
    daily_base = daily_df.loc[
        :, [c for c in base_cols if c in daily_df.columns]
    ].copy()

    for df in (master_base, daily_base):
        df[FECHA_COL_2] = pd.to_datetime(df[FECHA_COL_2], errors="coerce")

    merged = pd.concat([master_base, daily_base], ignore_index=True)
    merged = merged.drop_duplicates(
        subset=[EMPRESA_COL, FECHA_COL_2], keep="last"
    )
    merged = merged.sort_values(FECHA_COL_2).reset_index(drop=True)
    return merged


def build_versioned_features(
    merged_df: pd.DataFrame, *, horizons: list[int] | None = None
) -> pd.DataFrame:
    """Rebuild features on merged data, h=5 only by default.

    Calls ``build_features_for_company`` per company, then drops rows
    where **all** feature columns are NaN (warm-up window).
    """
    if horizons is None:
        horizons = [5]

    featured_frames: list[pd.DataFrame] = []
    for company in sorted(merged_df[EMPRESA_COL].unique()):
        company_df = merged_df.loc[merged_df[EMPRESA_COL] == company].copy()
        if company_df.empty:
            continue
        featured = build_features_for_company(company_df, horizons=horizons)
        featured_frames.append(featured)

    if not featured_frames:
        return pd.DataFrame()

    result = pd.concat(featured_frames, ignore_index=True)
    result[FECHA_COL_2] = pd.to_datetime(result[FECHA_COL_2], errors="coerce")
    result = result.sort_values([EMPRESA_COL, FECHA_COL_2]).reset_index(drop=True)

    # Identify feature columns (exclude identity, intermediate, and target cols)
    exclude = {FECHA_COL_2, EMPRESA_COL, "ret_1d"}
    for h in horizons:
        exclude.add(f"ret_fwd_h{h}")
        exclude.add(f"target_up_h{h}")

    feature_cols = [c for c in result.columns if c not in exclude]
    if feature_cols:
        mask = result[feature_cols].notna().any(axis=1)
        result = result.loc[mask].copy()

    return result


def get_latest_dataset_version(
    data_dir: Path | str = PROCESSED_DIR,
) -> Path | None:
    """Return the most recent versioned dataset file, or ``None``.

    Looks for files matching ``BVG_features_svc_master_v*.csv`` in
    ``data_dir`` and picks the one with the latest date suffix.
    """
    data_dir = Path(data_dir)
    files = list(data_dir.glob("BVG_features_svc_master_v*.csv"))
    if not files:
        return None

    def _extract_date(path: Path) -> datetime:
        stem = path.stem
        prefix = "BVG_features_svc_master_v"
        if stem.startswith(prefix):
            date_str = stem[len(prefix) :]
            try:
                return datetime.strptime(date_str, "%Y%m%d")
            except ValueError:
                pass
        return datetime.min

    files.sort(key=_extract_date, reverse=True)
    return files[0]


def update_latest_marker(
    versioned_path: Path | str,
    marker_path: Path | str = PROCESSED_DIR / "BVG_features_svc_master.csv",
) -> None:
    """Copy ``versioned_path`` to the canonical master path.

    Uses ``shutil.copy2`` for Windows compatibility (no symlinks).
    """
    versioned_path = Path(versioned_path)
    marker_path = Path(marker_path)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(versioned_path), str(marker_path))
