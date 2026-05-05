from __future__ import annotations

from typing import NamedTuple

import pandas as pd
import streamlit as st


class SidebarState(NamedTuple):
    model_family: str
    company: str
    dataset_path: str


def setup_page() -> None:
    st.set_page_config(page_title="Validación Experimental", layout="wide")
    st.title("Validación Experimental de Modelos de Predicción Financiera")
    st.caption("Bitácora de inferencias t+5")


def sidebar_config(
    companies: list[str],
    *,
    dataset_path: str,
) -> SidebarState:
    with st.sidebar:
        st.header("Configuración")
        model_family = st.selectbox(
            "Familia de modelo", ["Clásico", "Cuántico"], index=0
        )
        company = st.selectbox("Empresa", companies, index=0)
        st.divider()
        
    return SidebarState(
        model_family=model_family,
        company=company,
        dataset_path=dataset_path,
    )


def action_button() -> bool:
    return st.button("Actualizar Datos y Calcular Inferencias", type="primary")


def show_quick_history(history_df: pd.DataFrame, company: str) -> None:
    st.subheader("Historial reciente")
    if history_df.empty:
        st.info("No hay historial disponible para la empresa seleccionada.")
        return
    st.caption(f"Últimas {len(history_df)} filas para {company}.")
    st.dataframe(
        history_df.tail(10).copy(),
        width="stretch",
    )


def show_traceability_table(log_df: pd.DataFrame) -> None:
    st.subheader("Bitácora de trazabilidad")
    if log_df.empty:
        st.info("La bitácora está vacía.")
        return

    def highlight_obs(val: object) -> str:
        if isinstance(val, str) and val.strip():
            return "background-color: #fff3cd"
        return ""

    styled = log_df.style.applymap(highlight_obs, subset=["Observacion_Datos"])
    st.dataframe(styled, width="stretch")


def show_warning_if_insufficient_data(daily_df: pd.DataFrame) -> bool:
    if daily_df.empty:
        st.warning("Datos insuficientes para ejecutar el catch-up.")
        return True
    max_date = pd.to_datetime(daily_df["fecha"], errors="coerce").max()
    if pd.isna(max_date):
        st.warning("Datos insuficientes para ejecutar el catch-up.")
        return True
    if max_date.normalize().weekday() != 4:
        st.warning(
            "No hay datos suficientes (semana en curso). "
            "Espere al cierre del viernes para ejecutar el catch-up."
        )
        return True
    return False


def show_metrics_header() -> None:
    st.divider()
    st.subheader("Métricas de la bitácora")


def show_live_accuracy(accuracy: float | None) -> None:
    if accuracy is None:
        st.info("Aún no hay predicciones resueltas para calcular accuracy.")
        return
    st.metric("Accuracy en vivo", f"{accuracy:.2%}")


def show_price_chart(fig: object) -> None:
    st.subheader("Gráfico de precios")
    st.pyplot(fig, clear_figure=True)
