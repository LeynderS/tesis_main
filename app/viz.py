from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from app.config import THEME_ACCENT_CLASSICAL, THEME_ACCENT_QUANTUM


def compute_accuracy(log_df: pd.DataFrame) -> float | None:
    resolved = log_df.loc[log_df["status"].isin(["ACIERTO", "FALLO"])].copy()
    if resolved.empty:
        return None
    return float((resolved["status"] == "ACIERTO").mean())


def build_plotly(history: pd.DataFrame, log_df: pd.DataFrame, company: str) -> go.Figure:
    history_plot = history.tail(60).copy()
    history_plot["fecha"] = pd.to_datetime(history_plot["fecha"], errors="coerce")
    history_plot = history_plot.loc[history_plot["fecha"].notna()].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_plot["fecha"],
            y=history_plot["close_last"],
            mode="lines",
            name="Cierre real",
            line=dict(width=2),
        )
    )

    if not log_df.empty:
        log_df = log_df.copy()
        log_df["fecha_t"] = pd.to_datetime(log_df["fecha_t"], errors="coerce")
        log_df = log_df.loc[log_df["fecha_t"].notna()].copy()
        if not log_df.empty:
            cutoff = history_plot["fecha"].min() if not history_plot.empty else None
            if cutoff is not None:
                log_df = log_df.loc[log_df["fecha_t"] >= cutoff].copy()

            for family, label, color in (
                ("classical", "Predicción Clásica", THEME_ACCENT_CLASSICAL),
                ("quantum", "Predicción Cuántica", THEME_ACCENT_QUANTUM),
            ):
                family_df = log_df.loc[
                    (log_df["company"] == company)
                    & (log_df["model_family"] == family)
                ].copy()
                if family_df.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=family_df["fecha_t"],
                        y=family_df["close_t"],
                        mode="markers",
                        name=label,
                        marker=dict(color=color, size=9, symbol="circle"),
                    )
                )

    fig.update_layout(
        title="Histórico + Predicciones",
        xaxis_title="Fecha",
        yaxis_title="Precio",
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=20, r=20, t=50, b=40),
    )
    return fig
