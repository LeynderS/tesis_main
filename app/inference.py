from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from .config import MODELS_DIR, TARGET_H, TEST_SIZE, COMPANY_FILE_MAP
from bvg_core.quantum import build_qkernel, build_quantum_train_matrix as _build_quantum_train_matrix

def get_company_tag(company: str) -> str:
    if company not in COMPANY_FILE_MAP:
        raise ValueError(f"Empresa no mapeada: {company}")
    return COMPANY_FILE_MAP[company]

def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Manifest no encontrado: {path.as_posix()}")
    return json.loads(path.read_text(encoding="utf-8"))

def load_classical_artifacts(company: str, model_name: str) -> dict:
    tag = get_company_tag(company)
    base = MODELS_DIR / "classical"
    
    manifest_path = base / f"{tag}_{model_name}_manifest.json"
    pipeline_path = base / f"{tag}_{model_name}_pipeline.joblib"

    manifest = load_manifest(manifest_path)
    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Pipeline clásico no encontrado: {pipeline_path.as_posix()}"
        )
        
    pipeline = joblib.load(pipeline_path)
    return {
        "manifest": manifest,
        "pipeline": pipeline,
        "model_name": f"{tag}_{model_name}",
    }
def load_quantum_artifacts(company: str, model_name: str) -> dict:
    tag = get_company_tag(company)
    base = MODELS_DIR / "quantum"
    
    manifest_path = base / f"{tag}_{model_name}_manifest.json"
    scaler_path = base / f"{tag}_{model_name}_scaler.joblib"
    pca_path = base / f"{tag}_{model_name}_pca.joblib"
    svc_path = base / f"{tag}_{model_name}_svc.joblib"
    kernel_config_path = base / f"{tag}_{model_name}_kernel_config.json"

    manifest = load_manifest(manifest_path)
    
    missing = [p for p in [scaler_path, pca_path, svc_path, kernel_config_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan artefactos cuánticos: " + ", ".join(p.as_posix() for p in missing)
        )

    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    svc = joblib.load(svc_path)
    
    kernel_config = json.loads(kernel_config_path.read_text(encoding="utf-8"))
    qkernel = build_qkernel(kernel_config)

    return {
        "manifest": manifest,
        "scaler": scaler,
        "pca": pca,
        "svc": svc,
        "qkernel": qkernel,
        "kernel_config": kernel_config,
        "model_name": f"{tag}_{model_name}",
    }


def validate_feature_row(
    feature_row: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    missing = [c for c in feature_columns if c not in feature_row.columns]
    if missing:
        raise ValueError(f"Feature row sin columnas requeridas: {missing}")
    X_row = feature_row.loc[:, feature_columns].copy()
    if X_row.isna().any().any():
        raise ValueError("La fila de features contiene NaN. Verifique los datos de entrada.")
    return X_row


def build_quantum_train_matrix(
    df: pd.DataFrame,
    company: str,
    feature_columns: list[str],
    scaler,
    pca,
    *,
    train_end_date: str | None = None,
    test_size: int | None = TEST_SIZE,
) -> np.ndarray:
    return _build_quantum_train_matrix(
        df,
        company,
        feature_columns,
        scaler,
        pca,
        train_end_date=train_end_date,
        test_size=test_size,
    )


def infer_classical(pipeline, X_row: pd.DataFrame) -> dict:
    y_pred = int(pipeline.predict(X_row)[0])
    proba_up = np.nan
    if hasattr(pipeline, "predict_proba"):
        proba_up = float(pipeline.predict_proba(X_row)[0, 1])
    return {"y_pred": y_pred, "proba_up": proba_up}


def infer_quantum(
    bundle: dict, X_row: pd.DataFrame, master_df: pd.DataFrame, company: str
) -> dict:
    scaler = bundle["scaler"]
    pca = bundle["pca"]
    svc = bundle["svc"]
    qkernel = bundle["qkernel"]
    feature_columns = list(bundle["manifest"]["feature_columns"])

    train_end_date = bundle["manifest"].get("train_end_date")
    X_train_q = build_quantum_train_matrix(
        master_df, company, feature_columns, scaler, pca,
        train_end_date=train_end_date, test_size=0,
    )
    Xq = pca.transform(scaler.transform(X_row))
    K = qkernel.evaluate(x_vec=Xq, y_vec=X_train_q)
    y_pred = int(svc.predict(K)[0])
    proba_up = np.nan
    if hasattr(svc, "predict_proba"):
        proba_up = float(svc.predict_proba(K)[0, 1])
    return {"y_pred": y_pred, "proba_up": proba_up}
