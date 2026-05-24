from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import BVG_FALLBACK_MAX_DAYS, LOG_COLUMNS, TARGET_H, COMPANY_ABBR_MAP, COMPANY_FILE_MAP, MODELS_DIR

from app.inference import (
    build_quantum_train_matrix,
    infer_classical,
    infer_quantum,
    load_classical_artifacts,
    load_quantum_artifacts,
    validate_feature_row,
)
from bvg_core.data import get_friday_data_with_fallback
from bvg_core.features import build_features_for_company


@dataclass
class CatchupResult:
    log_df: pd.DataFrame
    retrain_due: bool = False
    retrain_week: int | None = None
    retrain_friday: pd.Timestamp | None = None


def reset_log_file(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=LOG_COLUMNS).to_csv(log_path, index=False)


def ensure_log_file(log_path: Path, *, reset: bool = False) -> None:
    if reset or not log_path.exists():
        reset_log_file(log_path)
        return

    df_header = pd.read_csv(log_path, nrows=0)
    current_cols = list(df_header.columns)

    if current_cols != LOG_COLUMNS:
        reset_log_file(log_path)


def read_log(log_path: Path) -> pd.DataFrame:
    ensure_log_file(log_path)
    df = pd.read_csv(log_path)
    missing = [c for c in LOG_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Log corrupto. Columnas faltantes: {missing}")
    return df


def detect_model_version(company: str, model_family: str, model_name: str) -> str:
    """Detect if a retrained model artifact exists and return its version string.

    Returns 'original' if no retrained artifact exists.
    Returns 'retrained_{YYYYMMDD}' if a retrained artifact exists.
    """
    tag = COMPANY_FILE_MAP[company]
    retrained_dir = MODELS_DIR / model_family / "retrained"

    if not retrained_dir.exists():
        return "original"

    if model_family == "classical":
        patterns = [
            f"{tag}_{model_name}_retrained_*.joblib",
        ]
        matches = sorted({p for pattern in patterns for p in retrained_dir.glob(pattern)})
    else:
        pattern = f"{tag}_{model_name}_svc_retrained_*.joblib"
        matches = []
        for subdir in sorted(retrained_dir.iterdir()):
            if subdir.is_dir():
                matches.extend(sorted(subdir.glob(pattern)))

    if not matches:
        return "original"

    latest = matches[-1]
    date_str = latest.stem.split("_retrained_")[-1]
    return f"retrained_{date_str}"


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

    def get_bundle(self, company: str, family: str, model_name: str, version: str = "original") -> dict:
        key = f"{company}:{family}:{model_name}:{version}"
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

    def clear(self) -> None:
        self._bundles.clear()
        self._xtrain.clear()


def generate_semantic_id(
    week_num: int,
    company: str,
    family: str,
    model_version: str,
    company_abbr_map: dict,
) -> str:
    abbr = company_abbr_map.get(company, company.replace(" ", "_")[:5])
    family_title = "Clasico" if family == "classical" else "Cuantico"
    return f"{week_num}_{abbr}_{family_title}_{model_version}"


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
            last_inference = log_dates.max() + timedelta(days=7)

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
    model_version: str = "original",
    model_versions: dict[str, str] | None = None,
) -> list[dict]:
    data_friday = current_friday - timedelta(days=7)
    target_date = pd.to_datetime(current_friday).normalize()
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
            version = model_version
            if model_versions is not None:
                version = model_versions.get(f"{company}:{family}", model_version)

            # Deduplication: skip if already predicted for this company/family/version/date
            if not log_df.empty:
                existing_mask = (
                    (log_df["company"] == company)
                    & (log_df["model_family"] == family)
                    & (log_df["model_version"] == version)
                    & (
                        pd.to_datetime(log_df["fecha_t"], errors="coerce")
                        .dt.normalize()
                        == data_friday
                    )
                )
                if existing_mask.any():
                    continue

            bundle_key = f"{company}:{family}:{model_name}:{version}"
            bundle = cache.get_bundle(company, family, model_name, version=version)

            feature_columns = list(bundle["manifest"]["feature_columns"])
            X_row = validate_feature_row(feature_row, feature_columns)

            if family == "classical":
                inference = infer_classical(bundle["pipeline"], X_row)
            else:
                try:
                    train_end_date = (
                        bundle.get("training_cutoff")
                        or bundle.get("train_end_date_for_inference")
                        or bundle["manifest"].get("train_end_date")
                    )
                    X_train_q = cache.get_xtrain(
                        bundle_key,
                        compute_fn=lambda: build_quantum_train_matrix(
                            master_df,
                            company,
                            feature_columns,
                            bundle["scaler"],
                            bundle["pca"],
                            train_end_date=train_end_date,
                            test_size=bundle.get("train_matrix_test_size", 0),
                            training_policy=bundle.get("training_policy"),
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
                version,
                COMPANY_ABBR_MAP,
            )

            record = {
                "id": semantic_id,
                "company": company,
                "model_family": family,
                "model_name": bundle["model_name"],
                "model_version": bundle.get("model_version", version),
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


def count_predictions(log_df: pd.DataFrame, company: str, model_family: str, model_version: str) -> int:
    """Count rows matching company, family, and version in log_df."""
    mask = (
        (log_df["company"] == company)
        & (log_df["model_family"] == model_family)
        & (log_df["model_version"] == model_version)
    )
    return len(log_df.loc[mask])


def is_gate_active(
    log_df: pd.DataFrame,
    companies: list[str],
    model_families: tuple[str, ...] = ("classical", "quantum"),
    model_name: str = "h5",
) -> bool:
    """Return True if any company+family has reached the 4-prediction limit
    for its currently active 'original' version.
    """
    if log_df.empty:
        return False
    for company in companies:
        for family in model_families:
            original_count = count_predictions(log_df, company, family, "original")
            retrained_available = detect_model_version(company, family, model_name) != "original"
            if original_count >= 4 and not retrained_available:
                return True
    return False


def resolve_active_model_version(
    log_df: pd.DataFrame,
    company: str,
    model_family: str,
    model_name: str = "h5",
) -> str:
    """Return the phase-correct model version for RFC6 catch-up.

    Phase 1 always uses the original artifacts for the first four predictions,
    even if retrained artifacts already exist on disk. After four original rows
    exist for the company/family, the retrained artifact may be used.
    """
    if log_df.empty:
        return "original"
    if count_predictions(log_df, company, model_family, "original") < 4:
        return "original"
    return detect_model_version(company, model_family, model_name)


def resolve_week_num(current_friday: pd.Timestamp, default_start_date: str) -> int:
    """Return global RFC6 week number from the nominal target Friday."""
    start = pd.to_datetime(default_start_date).normalize()
    current = pd.to_datetime(current_friday).normalize()
    delta_days = max((current - start).days, 0)
    return delta_days // 7 + 1


def derive_training_cutoff(
    log_df: pd.DataFrame,
    *,
    model_version: str = "original",
) -> pd.Timestamp | None:
    """Derive the retraining cutoff date from the log.

    Returns the maximum fecha_t5 among rows with the given model_version.
    This is the latest known target date at the gate and prevents
    lookahead bias when used to truncate training data.
    """
    if log_df.empty or "fecha_t5" not in log_df.columns or "model_version" not in log_df.columns:
        return None
    mask = log_df["model_version"] == model_version
    if not mask.any():
        return None
    dates = pd.to_datetime(log_df.loc[mask, "fecha_t5"], errors="coerce")
    dates = dates.loc[dates.notna()]
    if dates.empty:
        return None
    return dates.max().normalize()


def catchup_loop(
    log_path: Path,
    price_df: pd.DataFrame,
    model_name: str,
    master_df: pd.DataFrame,
    *,
    model_families: tuple[str, ...] = ("classical", "quantum"),
    default_start_date: str = "2026-03-27",
    history_window: int = 30,
    cache: BundleCache | None = None,
) -> CatchupResult:
    log_df = read_log(log_path)
    work_df = prepare_work_df(price_df)
    if work_df.empty:
        return CatchupResult(log_df=log_df)

    last_valid_target_friday = get_last_valid_target_friday(work_df)
    if last_valid_target_friday is None:
        return CatchupResult(log_df=log_df)

    start_ts = resolve_start_ts(log_df, default_start_date)
    if start_ts > last_valid_target_friday:
        return CatchupResult(log_df=log_df)

    if cache is None:
        cache = BundleCache()
    companies = sorted(work_df["empresa"].dropna().unique().tolist())

    current_friday = start_ts
    while current_friday <= last_valid_target_friday:
        log_df = resolve_pending(log_df, work_df, current_friday)

        model_versions = {}
        for company in companies:
            for family in model_families:
                model_versions[f"{company}:{family}"] = resolve_active_model_version(
                    log_df,
                    company,
                    family,
                    model_name,
                )

        week_num = resolve_week_num(current_friday, default_start_date)

        # Row-count gate: require retrain after 4 original predictions per company+family
        for company in companies:
            for family in model_families:
                version = model_versions.get(f"{company}:{family}", "original")
                if version == "original" and count_predictions(log_df, company, family, version) >= 4:
                    log_df.to_csv(log_path, index=False)
                    return CatchupResult(
                        log_df=log_df,
                        retrain_due=True,
                        retrain_week=week_num,
                        retrain_friday=current_friday,
                    )

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
            model_versions=model_versions,
        )

        if new_records:
            log_df = pd.concat(
                [log_df, pd.DataFrame(new_records, columns=LOG_COLUMNS)], ignore_index=True
            )

        current_friday = current_friday + timedelta(days=7)

    log_df.to_csv(log_path, index=False)
    return CatchupResult(log_df=log_df)


def run_global_catchup(
    log_path: Path,
    price_df: pd.DataFrame,
    master_df: pd.DataFrame,
    *,
    model_name: str = "h5",
    cache: BundleCache | None = None,
) -> CatchupResult:
    """Run the global catch-up loop for all companies and model families.

    Returns a CatchupResult containing the updated log DataFrame and any
    retrain scheduling information.
    """
    return catchup_loop(
        log_path,
        price_df,
        model_name,
        master_df,
        model_families=("classical", "quantum"),
        cache=cache,
    )
