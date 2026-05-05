from __future__ import annotations

import pandas as pd


def compute_accuracy(log_df: pd.DataFrame) -> float | None:
    resolved = log_df.loc[log_df["status"].isin(["ACIERTO", "FALLO"])].copy()
    if resolved.empty:
        return None
    return float((resolved["status"] == "ACIERTO").mean())


def build_price_figure(
    history: pd.DataFrame, log_df: pd.DataFrame, company: str
) -> tuple:
    history_plot = history.tail(20).copy()
    history_plot["fecha"] = pd.to_datetime(history_plot["fecha"], errors="coerce")
    fig = history_plot.plot(
        x="fecha",
        y="close_last",
        figsize=(10, 4),
        title="Últimos 20 precios (con predicciones)",
    ).get_figure()

    if not log_df.empty:
        log_df = log_df.copy()
        log_df["fecha_t"] = pd.to_datetime(log_df["fecha_t"], errors="coerce")
        recent_preds = log_df[
            (log_df["company"] == company)
            & (log_df["fecha_t"].notna())
            & (log_df["fecha_t"] >= history_plot["fecha"].min())
        ].copy()
        if not recent_preds.empty:
            colors = recent_preds["y_pred"].apply(
                lambda v: "green" if int(v) == 1 else "red"
            )
            ax = fig.axes[0]
            ax.scatter(recent_preds["fecha_t"], recent_preds["close_t"], c=colors, marker="o")

    return fig, history_plot
