from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BVG_FALLBACK_MAX_DAYS, LOG_COLUMNS, TARGET_H, COMPANY_ABBR_MAP
from .features import build_features_for_company
from .inference import (
    build_quantum_train_matrix,
    infer_classical,
    infer_quantum,
    load_classical_artifacts,
    load_quantum_artifacts,
    validate_feature_row,
)
from bvg_core.data import compute_target_date, get_friday_data_with_fallback

def reset_log_file(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=LOG_COLUMNS).to_csv(log_path, index=False)


def ensure_log_file(log_path: Path, *, reset: bool = False) -> None:
    if reset or not log_path.exists():
        reset_log_file(log_path)
        return

    df_header = pd.read_csv(log_path, nrows=0)
    if list(df_header.columns) != LOG_COLUMNS:
        reset_log_file(log_path)


def read_log(log_path: Path) -> pd.DataFrame:
    ensure_log_file(log_path)
    df = pd.read_csv(log_path)
    missing = [c for c in LOG_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Log corrupto. Columnas faltantes: {missing}")
    return df


def should_run_catchup(log_df: pd.DataFrame, price_df: pd.DataFrame) -> bool:
    if price_df.empty:
        return False
    price_dates = pd.to_datetime(price_df["fecha"], errors="coerce").dt.normalize()
    price_dates = price_dates.loc[price_dates.notna()].copy()
    if price_dates.empty:
        return False

    max_date = price_dates.max()

    # Próximo viernes objetivo después del último dato disponible
    days_to_next_friday = (4 - max_date.weekday()) % 7
    if days_to_next_friday == 0:
        days_to_next_friday = 7
    last_target_friday = max_date + timedelta(days=days_to_next_friday)

    if log_df.empty:
        return True

    log_target_dates = pd.to_datetime(log_df["fecha_t5"], errors="coerce").dt.normalize()
    log_target_dates = log_target_dates.loc[log_target_dates.notna()].copy()
    if log_target_dates.empty:
        return True
    return last_target_friday > log_target_dates.max()


def append_log(log_path: Path, record: dict) -> None:
    ensure_log_file(log_path)
    row_df = pd.DataFrame([record], columns=LOG_COLUMNS)
    row_df.to_csv(log_path, mode="a", header=False, index=False)




def resolve_t_plus_5(
    log_path: Path,
    price_df: pd.DataFrame,
    company: str,
    pending_id: str,
    close_override: float | None = None,
) -> pd.DataFrame:
    log_df = read_log(log_path)
    if log_df.empty:
        raise ValueError("No existen registros en la bitácora.")

    if not pending_id:
        raise ValueError("Debe seleccionar un registro pendiente.")

    pending_mask = (log_df["status"] == "PENDIENTE") & (log_df["company"] == company)
    if pending_id:
        pending_mask &= log_df["id"] == pending_id

    if not pending_mask.any():
        raise ValueError("No se encontró un registro pendiente para resolver.")

    idx = log_df.loc[pending_mask].index[0]
    fecha_objetivo = pd.to_datetime(log_df.loc[idx, "fecha_t"], errors="coerce")
    close_t = float(log_df.loc[idx, "close_t"])

    close_t5 = close_override
    if close_t5 is None:
        match = price_df.loc[
            (price_df["empresa"] == company) & (price_df["fecha"].dt.normalize() == fecha_objetivo)
        ]
        if match.empty:
            raise ValueError(
                "No se encontró precio de cierre para la fecha objetivo. "
                "Ingrese el precio manualmente."
            )
        close_t5 = float(match["close_last"].iloc[0])

    ret_fwd = float(np.log(close_t5 / close_t))
    y_pred = int(log_df.loc[idx, "y_pred"])
    status = "ACIERTO" if (ret_fwd > 0) == (y_pred == 1) else "FALLO"

    log_df.loc[idx, "close_t5"] = close_t5
    log_df.loc[idx, "ret_fwd_h5"] = ret_fwd
    log_df.loc[idx, "status"] = status

    log_df.to_csv(log_path, index=False)
    return log_df


class BundleCache:
    def __init__(self) -> None:
        self._bundles: dict[str, dict] = {}
        self._xtrain: dict[str, np.ndarray] = {}

    def get_bundle(self, company: str, family: str, model_name: str) -> dict:
        key = f"{company}:{family}:{model_name}"
        if key not in self._bundles:
            if family == "classical":
                self._bundles[key] = load_classical_artifacts(company, model_name)
            else:
                self._bundles[key] = load_quantum_artifacts(company, model_name)
        return self._bundles[key]

    def get_xtrain(self, key: str, *, compute_fn) -> np.ndarray:
        if key not in self._xtrain:
            self._xtrain[key] = compute_fn()
        return self._xtrain[key]


def generate_semantic_id(
    week_num: int,
    company: str,
    family: str,
    company_abbr_map: dict,
) -> str:
    abbr = company_abbr_map.get(company, company.replace(" ", "_")[:5])
    family_title = "Clasico" if family == "classical" else "Cuantico"
    return f"{week_num}_{abbr}_{family_title}"


def resolve_pending(
    log_df: pd.DataFrame,
    work_df: pd.DataFrame,
    current_friday: pd.Timestamp,
) -> pd.DataFrame:
    if log_df.empty:
        return log_df

    pending_mask = log_df["status"] == "PENDIENTE"
    if not pending_mask.any():
        return log_df

    pending_rows = log_df.loc[pending_mask].copy()
    target_dates = pd.to_datetime(pending_rows["fecha_t5"], errors="coerce")
    resolvable_idx = pending_rows.loc[
        target_dates.notna() & (target_dates <= current_friday)
    ].index

    for idx in resolvable_idx:
        company = log_df.loc[idx, "company"]
        target_date = pd.to_datetime(log_df.loc[idx, "fecha_t5"], errors="coerce").normalize()
        fecha_usada, close_t5, observacion = get_friday_data_with_fallback(
            work_df,
            company,
            target_date,
            max_back_days=BVG_FALLBACK_MAX_DAYS,
        )
        if fecha_usada is None or close_t5 is None:
            log_df.loc[idx, "Observacion_Datos"] = observacion
            log_df.loc[idx, "Fecha_Datos_Usados"] = ""
            continue

        close_t = float(log_df.loc[idx, "close_t"])
        ret_fwd = float(np.log(close_t5 / close_t))
        y_pred = int(log_df.loc[idx, "y_pred"])
        status = "ACIERTO" if (ret_fwd > 0) == (y_pred == 1) else "FALLO"

        log_df.loc[idx, "close_t5"] = close_t5
        log_df.loc[idx, "ret_fwd_h5"] = ret_fwd
        log_df.loc[idx, "status"] = status
        log_df.loc[idx, "Fecha_Datos_Usados"] = fecha_usada.date().isoformat()
        log_df.loc[idx, "Observacion_Datos"] = observacion

    return log_df


def get_history(
    work_df: pd.DataFrame,
    company: str,
    cutoff_date: pd.Timestamp,
    history_window: int,
) -> pd.DataFrame:
    company_df = work_df.loc[
        (work_df["empresa"] == company) & (work_df["fecha"] <= cutoff_date)
    ].sort_values("fecha")
    if company_df.empty or len(company_df) < history_window:
        return pd.DataFrame()
    return company_df.tail(history_window).copy()


def prepare_work_df(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df.empty:
        return pd.DataFrame()
    work_df = price_df.copy()
    work_df["fecha"] = pd.to_datetime(work_df["fecha"], errors="coerce").dt.normalize()
    work_df = work_df.loc[work_df["fecha"].notna()].copy()
    return work_df


def get_last_valid_target_friday(work_df: pd.DataFrame) -> pd.Timestamp | None:
    max_trading_date = work_df["fecha"].max()
    if pd.isna(max_trading_date):
        return None
    max_trading_date = pd.to_datetime(max_trading_date).normalize()
    days_to_next_friday = (4 - max_trading_date.weekday()) % 7
    if days_to_next_friday == 0:
        days_to_next_friday = 7
    return max_trading_date + timedelta(days=days_to_next_friday)


def resolve_start_ts(
    log_df: pd.DataFrame,
    default_start_date: str,
) -> pd.Timestamp:
    last_inference = None
    if not log_df.empty and "fecha_t5" in log_df.columns:
        log_dates = pd.to_datetime(log_df["fecha_t5"], errors="coerce").dt.normalize()
        log_dates = log_dates.loc[log_dates.notna()]
        if not log_dates.empty:
            last_inference = log_dates.max()

    start_ts = pd.to_datetime(
        last_inference if pd.notna(last_inference) else default_start_date,
        errors="coerce",
    ).normalize()
    if pd.isna(start_ts):
        start_ts = pd.to_datetime(default_start_date).normalize()

    if not log_df.empty:
        pending_mask = log_df["status"] == "PENDIENTE"
        if pending_mask.any():
            pending_max = pd.to_datetime(
                log_df.loc[pending_mask, "fecha_t5"], errors="coerce"
            ).max()
            if pd.notna(pending_max):
                pending_max = pd.to_datetime(pending_max).normalize()
                if pending_max < start_ts:
                    start_ts = pending_max

    if start_ts.weekday() != 4:
        days_ahead = (4 - start_ts.weekday()) % 7
        start_ts = start_ts + timedelta(days=days_ahead)
    return start_ts


def predict_for_friday(
    *,
    log_df: pd.DataFrame,
    work_df: pd.DataFrame,
    master_df: pd.DataFrame,
    model_name: str,
    model_families: tuple[str, ...],
    history_window: int,
    cache: BundleCache,
    companies: list[str],
    current_friday: pd.Timestamp,
    week_num: int,
) -> list[dict]:
    data_friday = current_friday - timedelta(days=7)
    target_date = compute_target_date(work_df["fecha"], data_friday)
    new_records: list[dict] = []

    for company in companies:
        fecha_usada, close_t, observacion = get_friday_data_with_fallback(
            work_df,
            company,
            data_friday,
            max_back_days=BVG_FALLBACK_MAX_DAYS,
        )
        if fecha_usada is None or close_t is None:
            continue

        history = get_history(work_df, company, fecha_usada, history_window)
        if history.empty:
            continue

        row = work_df.loc[
            (work_df["empresa"] == company) & (work_df["fecha"] == fecha_usada)
        ].copy()
        if row.empty:
            continue

        combined = history.copy()
        if not (combined["fecha"] == fecha_usada).any():
            combined = pd.concat([combined, row], ignore_index=True)

        feature_df = build_features_for_company(combined)
        feature_row = feature_df.tail(1).copy()

        for family in model_families:
            existing_mask = (
                (log_df["company"] == company)
                & (log_df["model_family"] == family)
                & (
                    pd.to_datetime(log_df["fecha_t"], errors="coerce")
                    .dt.normalize()
                    == current_friday
                )
            )
            if not log_df.empty and existing_mask.any():
                continue

            bundle_key = f"{company}:{family}:{model_name}"
            bundle = cache.get_bundle(company, family, model_name)

            feature_columns = list(bundle["manifest"]["feature_columns"])
            X_row = validate_feature_row(feature_row, feature_columns)

            if family == "classical":
                inference = infer_classical(bundle["pipeline"], X_row)
            else:
                try:
                    X_train_q = cache.get_xtrain(
                        bundle_key,
                        compute_fn=lambda: build_quantum_train_matrix(
                            master_df,
                            company,
                            feature_columns,
                            bundle["scaler"],
                            bundle["pca"],
                            train_end_date=bundle["manifest"].get("train_end_date"),
                            test_size=0,
                        ),
                    )
                except ValueError:
                    continue

                inference = infer_quantum(
                    bundle,
                    X_row,
                    master_df,
                    company,
                    X_train_q_cached=X_train_q,
                )

            semantic_id = generate_semantic_id(
                week_num,
                company,
                family,
                COMPANY_ABBR_MAP,
            )

            record = {
                "id": semantic_id,
                "company": company,
                "model_family": family,
                "model_name": bundle["model_name"],
                "horizonte": TARGET_H,
                "fecha_t": data_friday.date().isoformat(),
                "close_t": float(close_t),
                "y_pred": inference["y_pred"],
                "proba_up": inference["proba_up"],
                "status": "PENDIENTE",
                "fecha_t5": target_date.date().isoformat(),
                "close_t5": np.nan,
                "ret_fwd_h5": np.nan,
                "Fecha_Datos_Usados": fecha_usada.date().isoformat(),
                "Observacion_Datos": observacion,
            }
            new_records.append(record)

    return new_records


def catchup_loop(
    log_path: Path,
    price_df: pd.DataFrame,
    model_name: str,
    master_df: pd.DataFrame,
    *,
    model_families: tuple[str, ...] = ("classical", "quantum"),
    default_start_date: str = "2026-03-27",
    history_window: int = 30,
) -> pd.DataFrame:
    log_df = read_log(log_path)
    work_df = prepare_work_df(price_df)
    if work_df.empty:
        return log_df

    last_valid_target_friday = get_last_valid_target_friday(work_df)
    if last_valid_target_friday is None:
        return log_df

    start_ts = resolve_start_ts(log_df, default_start_date)
    if start_ts > last_valid_target_friday:
        return log_df

    cache = BundleCache()
    companies = sorted(work_df["empresa"].dropna().unique().tolist())

    current_friday = start_ts
    week_num = 1
    while current_friday <= last_valid_target_friday:
        log_df = resolve_pending(log_df, work_df, current_friday)
        new_records = predict_for_friday(
            log_df=log_df,
            work_df=work_df,
            master_df=master_df,
            model_name=model_name,
            model_families=model_families,
            history_window=history_window,
            cache=cache,
            companies=companies,
            current_friday=current_friday,
            week_num=week_num,
        )

        if new_records:
            log_df = pd.concat(
                [log_df, pd.DataFrame(new_records, columns=LOG_COLUMNS)], ignore_index=True
            )

        current_friday = current_friday + timedelta(days=7)
        week_num += 1

    log_df.to_csv(log_path, index=False)
    return log_df


def run_global_catchup(
    log_path: Path,
    price_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    model_name: str = "h5",
) -> pd.DataFrame:
    return catchup_loop(
        log_path,
        price_df,
        model_name,
        master_df,
        model_families=("classical", "quantum"),
    )
