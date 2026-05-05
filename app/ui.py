from __future__ import annotations

import pandas as pd
import streamlit as st

from app.config import (
    THEME_ACCENT_CLASSICAL,
    THEME_ACCENT_QUANTUM
)


def setup_page() -> None:
    st.set_page_config(page_title="Validación Experimental", layout="wide")
    st.markdown(
        f"""
        <style>
        .deep-card {{
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
        }}
        .deep-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
        }}
        .badge-classical {{
            background-color: {THEME_ACCENT_CLASSICAL};
            color: #ffffff;
        }}
        .badge-quantum {{
            background-color: {THEME_ACCENT_QUANTUM};
            color: #ffffff;
        }}
        .compact-table {{
            font-family: "JetBrains Mono", "Fira Mono", "Consolas", monospace;
            font-size: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Validación Experimental de Modelos de Predicción Financiera")
    st.caption("Bitácora de inferencias t+5")


def render_top_bar(last_run: str | None) -> bool:
    left, right = st.columns([3, 1])
    with left:
        label = "—" if not last_run else last_run
        st.caption(f"Última ejecución: {label}")
    with right:
        return st.button("Actualizar BVG y ejecutar catch-up", type="primary")


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

    styled = (
        log_df.style.applymap(highlight_obs, subset=["Observacion_Datos"])
        .set_table_attributes('class="compact-table"')
    )
    st.dataframe(styled, width="stretch", height=320)

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
    st.plotly_chart(fig, width="stretch")


def _compute_accuracy_f1(resolved_df: pd.DataFrame) -> tuple[float | None, float | None]:
    if resolved_df.empty:
        return None, None
    y_true = (resolved_df["ret_fwd_h5"] > 0).astype(int)
    y_pred = resolved_df["y_pred"].astype(int)
    accuracy = float((y_true == y_pred).mean())

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    if tp == 0:
        return accuracy, None
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return accuracy, None
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, float(f1)


def build_cards(log_df: pd.DataFrame, company: str) -> None:
    st.subheader("Comparativo de modelos")
    if log_df.empty:
        st.info("No hay registros para la empresa seleccionada.")
        return

    def render_family_card(family: str, label: str, badge_class: str) -> None:
        family_df = log_df.loc[
            (log_df["company"] == company) & (log_df["model_family"] == family)
        ].copy()
        if family_df.empty:
            st.markdown(
                f"<div class='deep-card'><span class='deep-badge {badge_class}'>"
                f"{label}</span><p>Sin datos disponibles.</p></div>",
                unsafe_allow_html=True,
            )
            return

        family_df["fecha_t"] = pd.to_datetime(family_df["fecha_t"], errors="coerce")
        latest = family_df.loc[family_df["fecha_t"].idxmax()].copy()
        direction = "UP" if int(latest["y_pred"]) == 1 else "DOWN"
        proba_up = latest.get("proba_up")
        proba_label = "—" if pd.isna(proba_up) else f"{float(proba_up):.2%}"

        resolved = family_df.loc[
            family_df["ret_fwd_h5"].notna() & family_df["y_pred"].notna()
        ].copy()
        accuracy, f1 = _compute_accuracy_f1(resolved)
        acc_label = "—" if accuracy is None else f"{accuracy:.2%}"
        f1_label = "—" if f1 is None else f"{f1:.2%}"

        st.markdown(
            """
            <div class="deep-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="deep-badge {badge_class}">{label}</span>
                    <span style="font-size:12px; opacity:0.7;">{fecha}</span>
                </div>
                <div style="display:grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-top:10px;">
                    <div>
                        <div style="font-size:12px; opacity:0.7;">Dirección</div>
                        <div style="font-size:20px; font-weight:700;">{direction}</div>
                    </div>
                    <div>
                        <div style="font-size:12px; opacity:0.7;">Proba Up</div>
                        <div style="font-size:20px; font-weight:700;">{proba}</div>
                    </div>
                    <div>
                        <div style="font-size:12px; opacity:0.7;">Accuracy</div>
                        <div style="font-size:20px; font-weight:700;">{accuracy}</div>
                    </div>
                    <div>
                        <div style="font-size:12px; opacity:0.7;">F1</div>
                        <div style="font-size:20px; font-weight:700;">{f1}</div>
                    </div>
                </div>
            </div>
            """.format(
                badge_class=badge_class,
                label=label,
                fecha=pd.to_datetime(latest["fecha_t"]).date().isoformat(),
                direction=direction,
                proba=proba_label,
                accuracy=acc_label,
                f1=f1_label,
            ),
            unsafe_allow_html=True,
        )

    left, right = st.columns(2)
    with left:
        render_family_card("classical", "Clásico", "badge-classical")
    with right:
        render_family_card("quantum", "Cuántico", "badge-quantum")
