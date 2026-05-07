from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .config import BVG_FALLBACK_MAX_DAYS, LOG_COLUMNS, TARGET_H, COMPANY_ABBR_MAP
from .features import build_features_for_company
from .inference import (
    build_quantum_train_matrix,
    infer_classical,
    load_classical_artifacts,
    load_quantum_artifacts,
    validate_feature_row,
)


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


def get_friday_data_with_fallback(
    price_df: pd.DataFrame,
    company: str,
    target_friday: pd.Timestamp,
    *,
    max_back_days: int = BVG_FALLBACK_MAX_DAYS,
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

    # Fallback extendido: reutilizar último dato disponible anterior
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
    if price_df.empty:
        return log_df

    work_df = price_df.copy()
    work_df["fecha"] = pd.to_datetime(work_df["fecha"], errors="coerce").dt.normalize()
    work_df = work_df.loc[work_df["fecha"].notna()].copy()
    if work_df.empty:
        return log_df

    max_trading_date = work_df["fecha"].max()
    if pd.isna(max_trading_date):
        return log_df

    max_trading_date = pd.to_datetime(max_trading_date).normalize()

    # Próximo viernes objetivo después del último dato disponible
    days_to_next_friday = (4 - max_trading_date.weekday()) % 7
    if days_to_next_friday == 0:
        days_to_next_friday = 7
    last_valid_target_friday = max_trading_date + timedelta(days=days_to_next_friday)

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
    if start_ts > last_valid_target_friday:
        return log_df

    def resolve_pending(current_friday: pd.Timestamp) -> None:
        if log_df.empty:
            return
        pending_mask = log_df["status"] == "PENDIENTE"
        if not pending_mask.any():
            return

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

    def get_history(company: str, cutoff_date: pd.Timestamp) -> pd.DataFrame:
        company_df = work_df.loc[
            (work_df["empresa"] == company) & (work_df["fecha"] <= cutoff_date)
        ].sort_values("fecha")
        if company_df.empty or len(company_df) < history_window:
            return pd.DataFrame()
        return company_df.tail(history_window).copy()

    def load_bundle(company: str, family: str) -> dict:
        if family == "classical":
            return load_classical_artifacts(company, model_name)
        return load_quantum_artifacts(company, model_name)

    def infer_quantum_cached(
        bundle: dict,
        X_row: pd.DataFrame,
        X_train_q: np.ndarray,
    ) -> dict:
        scaler = bundle["scaler"]
        pca = bundle["pca"]
        svc = bundle["svc"]
        qkernel = bundle["qkernel"]
        Xq = pca.transform(scaler.transform(X_row))
        K = qkernel.evaluate(x_vec=Xq, y_vec=X_train_q)
        y_pred = int(svc.predict(K)[0])
        proba_up = np.nan
        if hasattr(svc, "predict_proba"):
            proba_up = float(svc.predict_proba(K)[0, 1])
        return {"y_pred": y_pred, "proba_up": proba_up}

    bundle_cache: dict[str, dict] = {}
    xtrain_cache: dict[str, np.ndarray] = {}
    new_records: list[dict] = []
    companies = sorted(work_df["empresa"].dropna().unique().tolist())

    # current_friday is the PREDICTION TARGET date (Friday we predict for)
    # data is looked up from the previous Friday (current_friday - 7 days)
    current_friday = start_ts
    week_num = 1
    while current_friday <= last_valid_target_friday:
        resolve_pending(current_friday)
        data_friday = current_friday - timedelta(days=7)

        for company in companies:
            fecha_usada, close_t, observacion = get_friday_data_with_fallback(
                work_df, company, data_friday
            )
            if fecha_usada is None or close_t is None:
                continue

            history = get_history(company, fecha_usada)
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
                if bundle_key not in bundle_cache:
                    bundle_cache[bundle_key] = load_bundle(company, family)
                bundle = bundle_cache[bundle_key]

                feature_columns = list(bundle["manifest"]["feature_columns"])
                
                X_row = validate_feature_row(feature_row, feature_columns)

                if family == "classical":
                    inference = infer_classical(bundle["pipeline"], X_row)
                else:
                    try:
                        if bundle_key not in xtrain_cache:
                            train_end_date = bundle["manifest"].get("train_end_date")
                            xtrain_cache[bundle_key] = build_quantum_train_matrix(
                                master_df,
                                company,
                                feature_columns,
                                bundle["scaler"],
                                bundle["pca"],
                                train_end_date=train_end_date,
                                test_size=0,
                            )
                    except ValueError:
                        continue
                    inference = infer_quantum_cached(
                        bundle,
                        X_row,
                        xtrain_cache[bundle_key],
                    )

                abbr = COMPANY_ABBR_MAP.get(company, company.replace(" ", "_")[:5])
                family_title = "Clasico" if family == "classical" else "Cuantico"
                semantic_id = f"{week_num}_{abbr}_{family_title}"

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
                    "fecha_t5": current_friday.date().isoformat(),
                    "close_t5": np.nan,
                    "ret_fwd_h5": np.nan,
                    "Fecha_Datos_Usados": fecha_usada.date().isoformat(),
                    "Observacion_Datos": observacion,
                }
                new_records.append(record)

        # Merge new records into log_df at end of each Friday iteration
        # so resolve_pending in the next iteration can see them
        if new_records:
            log_df = pd.concat(
                [log_df, pd.DataFrame(new_records, columns=LOG_COLUMNS)], ignore_index=True
            )
            new_records = []

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
