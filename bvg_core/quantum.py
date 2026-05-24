from __future__ import annotations

import numpy as np
import pandas as pd
from qiskit.circuit.library import zz_feature_map
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute

from bvg_core.config import TEST_SIZE


def build_qkernel(config: dict) -> FidelityQuantumKernel:
    feature_dim = int(config.get("feature_dimension", 5))
    reps = int(config.get("reps", 2))
    entanglement = config.get("entanglement", "linear")

    feature_map = zz_feature_map(
        feature_dimension=feature_dim,
        reps=reps,
        entanglement=entanglement,
    )

    fidelity = ComputeUncompute(sampler=StatevectorSampler(seed=42))
    return FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)


def build_quantum_train_matrix(
    df: pd.DataFrame,
    company: str,
    feature_columns: list[str],
    scaler,
    pca,
    *,
    train_end_date: str | None = None,
    test_size: int | None = TEST_SIZE,
    training_policy: str | None = None,
    target_horizon: int = 5,
) -> np.ndarray:
    company_df = df.loc[df["empresa"] == company].sort_values("fecha").copy()
    if company_df.empty:
        raise ValueError(f"No hay datos de entrenamiento para {company}.")
    if train_end_date:
        train_end = pd.to_datetime(train_end_date, errors="coerce")
        if pd.notna(train_end):
            company_df = company_df.loc[company_df["fecha"] <= train_end].copy()
    company_df = company_df.dropna(subset=feature_columns + ["fecha", "empresa"]).copy()

    if training_policy == "rows_with_known_h5_target_at_cutoff" and train_end_date:
        if len(company_df) <= target_horizon:
            raise ValueError(
                "Datos insuficientes para aplicar rows_with_known_h5_target_at_cutoff."
            )
        company_df = company_df.iloc[:-target_horizon].copy()

    if test_size is not None and test_size > 0 and len(company_df) <= test_size:
        raise ValueError(
            f"Datos insuficientes para reconstruir X_train_q (n={len(company_df)})."
        )
    if test_size is None or test_size <= 0:
        train_df = company_df.copy()
    else:
        train_df = company_df.iloc[: len(company_df) - test_size].copy()

    if train_df.empty:
        raise ValueError("No hay datos suficientes tras el split temporal.")
    X_train = train_df.loc[:, feature_columns].copy()
    if X_train.isna().any().any():
        raise ValueError("X_train contiene NaN luego de limpiar.")
    X_train_q = pca.transform(scaler.transform(X_train))
    return X_train_q


__all__ = ["build_qkernel", "build_quantum_train_matrix"]
