from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from app.config import DATA_PATH, MODELS_DIR, COMPANY_FILE_MAP
from bvg_core.data import load_master_dataset
from bvg_core.features import build_features_for_company
from bvg_core.splits import temporal_split_company


def _get_manifest_path(company: str, model_name: str = "h5") -> Path:
    tag = COMPANY_FILE_MAP.get(company)
    if not tag:
        raise FileNotFoundError(f"Empresa no mapeada: {company}")
    return MODELS_DIR / "classical" / f"{tag}_{model_name}_manifest.json"


def _load_manifest(company: str, model_name: str = "h5") -> dict[str, Any]:
    path = _get_manifest_path(company, model_name)
    if not path.exists():
        raise FileNotFoundError(f"Manifest no encontrado: {path.as_posix()}")
    return json.loads(path.read_text(encoding="utf-8"))


def _last_retrain_date(manifest: dict[str, Any]) -> datetime | None:
    history = manifest.get("retrain_history", [])
    if not history:
        return None
    last = history[-1]
    ts_str = last.get("retrained_at_utc") or last.get("utc_timestamp")
    if not ts_str:
        return None
    try:
        return pd.to_datetime(ts_str).to_pydatetime()
    except Exception:  # noqa: BLE001
        return None


def _rolling_accuracy(
    log_df: pd.DataFrame,
    company: str,
    model_family: str,
    min_resolved: int = 4,
) -> float | None:
    resolved = log_df.loc[
        (log_df["company"] == company)
        & (log_df["model_family"] == model_family)
        & (log_df["status"].isin(["ACIERTO", "FALLO"]))
    ].copy()
    if len(resolved) < min_resolved:
        return None
    return float((resolved.tail(min_resolved)["status"] == "ACIERTO").mean())


def should_retrain(
    log_df: pd.DataFrame,
    company: str,
    model_family: str,
    min_resolved: int = 4,
    threshold: float = 0.40,
    cooldown_days: int = 7,
) -> bool:
    """Return True if rolling accuracy is below threshold and cooldown has passed."""
    acc = _rolling_accuracy(log_df, company, model_family, min_resolved)
    if acc is None:
        return False
    if acc >= threshold:
        return False

    if model_family == "classical":
        try:
            manifest = _load_manifest(company)
        except FileNotFoundError:
            return False
        last_retrain = _last_retrain_date(manifest)
        if last_retrain is not None:
            days_since = (datetime.now(timezone.utc) - last_retrain).days
            if days_since < cooldown_days:
                return False

    return True


def should_retrain_quantum(
    log_df: pd.DataFrame,
    company: str,
    min_resolved: int = 4,
    threshold: float = 0.40,
) -> str | None:
    """Return a warning message if quantum drift is detected, otherwise None."""
    drift = should_retrain(log_df, company, "quantum", min_resolved, threshold)
    if drift:
        return (
            f"El modelo cuántico para **{company}** muestra deriva "
            f"(rolling accuracy < {threshold}). El reentrenamiento cuántico "
            "debe ejecutarse manualmente desde el notebook correspondiente."
        )
    return None


def retrain_classical_model(
    company: str,
    master_df: pd.DataFrame | None,
    trigger_accuracy: float,
    *,
    model_name: str = "h5",
    dataset_version: str | None = None,
) -> dict[str, Any]:
    """Retrain a classical SVC model using original manifest hyperparameters."""
    manifest = _load_manifest(company, model_name)
    params_fixed = manifest.get("params_fixed")
    feature_columns = manifest.get("feature_columns")
    if not params_fixed or not feature_columns:
        raise ValueError(
            f"El manifest de {company} no contiene 'params_fixed' o 'feature_columns'."
        )

    df = master_df
    if df is None or df.empty:
        df = load_master_dataset(DATA_PATH)

    company_df = df.loc[df["empresa"] == company].copy()
    if company_df.empty:
        raise ValueError(f"No se encontraron datos para la empresa: {company}")

    featured = build_features_for_company(company_df)
    featured = featured.dropna(
        subset=["target_up_h5"] + list(feature_columns)
    ).copy()

    if featured.empty:
        raise ValueError("No hay filas válidas tras construir features.")

    _, _, X_train, y_train, X_test, y_test = temporal_split_company(
        featured,
        company,
        test_size=30,
        feature_cols=feature_columns,
        target_col="target_up_h5",
    )

    params = dict(params_fixed)
    params.setdefault("random_state", 42)
    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("svc", SVC(**params)),
    ])
    pipeline.fit(X_train, y_train)

    # Smoke test: predict on the first 5 test rows
    _ = pipeline.predict(X_test.iloc[:5])

    # Accuracy gate on the full test set
    y_pred = pipeline.predict(X_test)
    accuracy = float(np.mean(y_pred == y_test.values))
    if accuracy <= 0.3:
        raise ValueError(
            f"Smoke test falló: accuracy={accuracy:.3f} en test (<=0.3)."
        )

    # Serialize
    tag = COMPANY_FILE_MAP[company]
    retrained_dir = MODELS_DIR / "classical" / "retrained"
    retrained_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = retrained_dir / f"{tag}_{model_name}_retrained_{date_str}.joblib"
    joblib.dump(pipeline, out_path)

    # Update manifest retrain_history
    history_entry = {
        "retrained_at_utc": datetime.now(timezone.utc).isoformat(),
        "trigger_accuracy": float(trigger_accuracy),
        "test_accuracy": accuracy,
        "smoke_test_passed": True,
        "dataset_version": dataset_version or "v1.0.0-legacy",
    }
    manifest.setdefault("retrain_history", []).append(history_entry)
    manifest_path = _get_manifest_path(company, model_name)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "model_path": out_path,
        "test_accuracy": accuracy,
        "manifest_path": manifest_path,
    }
