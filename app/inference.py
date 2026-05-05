from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute

from .config import MODELS_DIR, TARGET_H, TEST_SIZE


def safe_company(company: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in company.upper())
    return "_".join(part for part in cleaned.split("_") if part)


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Manifest no encontrado: {path.as_posix()}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_model_tag(base: Path, tag: str, model_name: str) -> str:
    model_tag = safe_company(model_name)
    manifest_path = base / f"{tag}_{model_tag}_manifest.json"
    if manifest_path.exists():
        return model_tag
    fallback_tag = "h5"
    fallback_path = base / f"{tag}_{fallback_tag}_manifest.json"
    if fallback_path.exists():
        return fallback_tag
    return model_tag


def load_classical_artifacts(company: str, model_name: str) -> dict:
    tag = safe_company(company)
    base = MODELS_DIR / "classical"
    model_tag = _resolve_model_tag(base, tag, model_name)
    manifest_path = base / f"{tag}_{model_tag}_manifest.json"
    pipeline_path = base / f"{tag}_{model_tag}_pipeline.joblib"
    manifest = load_manifest(manifest_path)
    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Pipeline clásico no encontrado: {pipeline_path.as_posix()}"
        )
    pipeline = joblib.load(pipeline_path)
    return {
        "manifest": manifest,
        "pipeline": pipeline,
        "model_name": f"{tag}_{model_tag}",
    }


def build_qkernel(config: dict) -> FidelityQuantumKernel:
    feature_dim = int(config.get("feature_dimension", 5))
    reps = int(config.get("reps", 2))
    entanglement = config.get("entanglement", "linear")
    feature_map = ZZFeatureMap(
        feature_dimension=feature_dim,
        reps=reps,
        entanglement=entanglement,
    )
    fidelity = ComputeUncompute(sampler=StatevectorSampler(seed=42))
    return FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)


def load_quantum_artifacts(company: str, model_name: str) -> dict:
    tag = safe_company(company)
    base = MODELS_DIR / "quantum"
    model_tag = _resolve_model_tag(base, tag, model_name)
    manifest_path = base / f"{tag}_{model_tag}_manifest.json"
    scaler_path = base / f"{tag}_{model_tag}_scaler.joblib"
    pca_path = base / f"{tag}_{model_tag}_pca.joblib"
    svc_path = base / f"{tag}_{model_tag}_svc.joblib"
    kernel_config_path = base / f"{tag}_{model_tag}_kernel_config.json"

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
        "model_name": f"{tag}_{model_tag}",
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
    test_size: int = TEST_SIZE,
) -> np.ndarray:
    if "horizonte" in df.columns:
        df = df.loc[df["horizonte"].astype(str) == TARGET_H].copy()
    company_df = df.loc[df["empresa"] == company].sort_values("fecha").copy()
    if company_df.empty:
        raise ValueError(f"No hay datos de entrenamiento para {company}.")
    company_df = company_df.dropna(subset=feature_columns + ["fecha", "empresa"]).copy()
    if len(company_df) <= test_size:
        raise ValueError(
            f"Datos insuficientes para reconstruir X_train_q (n={len(company_df)})."
        )
    train_df = company_df.iloc[: len(company_df) - test_size].copy()
    if train_df.empty:
        raise ValueError("No hay datos suficientes tras el split temporal.")
    X_train = train_df.loc[:, feature_columns].copy()
    if X_train.isna().any().any():
        raise ValueError("X_train contiene NaN luego de limpiar.")
    X_train_q = pca.transform(scaler.transform(X_train))
    return X_train_q


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

    X_train_q = build_quantum_train_matrix(master_df, company, feature_columns, scaler, pca)
    Xq = pca.transform(scaler.transform(X_row))
    K = qkernel.evaluate(x_vec=Xq, y_vec=X_train_q)
    y_pred = int(svc.predict(K)[0])
    proba_up = np.nan
    if hasattr(svc, "predict_proba"):
        proba_up = float(svc.predict_proba(K)[0, 1])
    return {"y_pred": y_pred, "proba_up": proba_up}
