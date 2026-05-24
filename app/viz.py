from __future__ import annotations

from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go

from app.config import THEME_ACCENT_CLASSICAL, THEME_ACCENT_QUANTUM


def compute_accuracy(log_df: pd.DataFrame) -> float | None:
    resolved = log_df.loc[log_df["status"].isin(["ACIERTO", "FALLO"])].copy()
    if resolved.empty:
        return None
    return float((resolved["status"] == "ACIERTO").mean())


def build_plotly(history: pd.DataFrame, log_df: pd.DataFrame, company: str, gate_active: bool = False) -> go.Figure:
    history_plot = history.copy()
    history_plot["fecha"] = pd.to_datetime(history_plot["fecha"], errors="coerce")
    history_plot = history_plot.loc[history_plot["fecha"].notna()].copy()

    fig = go.Figure()

    # ── Base price line ──────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=history_plot["fecha"],
            y=history_plot["close_last"],
            mode="lines",
            name="Precio real",
            line=dict(width=2, color="#64748b"),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.2f}<extra>Precio real</extra>",
        )
    )

    if not log_df.empty:
        log_df = log_df.copy()
        log_df["fecha_t"] = pd.to_datetime(log_df["fecha_t"], errors="coerce")
        log_df["fecha_t5"] = pd.to_datetime(log_df["fecha_t5"], errors="coerce")
        log_df = log_df.loc[log_df["fecha_t5"].notna()].copy()

        if not log_df.empty:
            company_log = log_df.loc[log_df["company"] == company].copy()
            if not company_log.empty:

                # ── Prediction arrows (at target date) ───────────────────────
                for family, label, color, x_offset in (
                    ("classical", "Clásico ▲", THEME_ACCENT_CLASSICAL, -0.35),
                    ("quantum", "Cuántico ▲", THEME_ACCENT_QUANTUM, 0.35),
                ):
                    fam_df = company_log.loc[company_log["model_family"] == family].copy()
                    if fam_df.empty:
                        continue

                    up_df = fam_df.loc[fam_df["y_pred"] == 1].copy()
                    down_df = fam_df.loc[fam_df["y_pred"] == 0].copy()

                    if not up_df.empty:
                        up_df["x_plot"] = up_df["fecha_t5"] + timedelta(days=x_offset)
                        up_df["y_plot"] = up_df["close_t"] * 1.02
                        fig.add_trace(
                            go.Scatter(
                                x=up_df["x_plot"],
                                y=up_df["y_plot"],
                                mode="markers",
                                name=label,
                                marker=dict(color=color, size=13, symbol="arrow-up"),
                                hovertemplate=(
                                    "%{x|%Y-%m-%d}<br>"
                                    "Predicción: SUBE<br>"
                                    "Precio datos: $%{customdata[0]:.2f}<extra>"
                                    + label
                                    + "</extra>"
                                ),
                                customdata=up_df[["close_t"]].values,
                            )
                        )

                    if not down_df.empty:
                        down_label = label.replace("▲", "▼")
                        down_df["x_plot"] = down_df["fecha_t5"] + timedelta(days=x_offset)
                        down_df["y_plot"] = down_df["close_t"] * 0.98
                        fig.add_trace(
                            go.Scatter(
                                x=down_df["x_plot"],
                                y=down_df["y_plot"],
                                mode="markers",
                                name=down_label,
                                marker=dict(color=color, size=13, symbol="arrow-down"),
                                hovertemplate=(
                                    "%{x|%Y-%m-%d}<br>"
                                    "Predicción: BAJA<br>"
                                    "Precio datos: $%{customdata[0]:.2f}<extra>"
                                    + down_label
                                    + "</extra>"
                                ),
                                customdata=down_df[["close_t"]].values,
                            )
                        )

                # ── Real movement bars (resolved only) ───────────────────────
                resolved = company_log.loc[
                    company_log["status"].isin(["ACIERTO", "FALLO"])
                ].copy()
                if not resolved.empty:
                    resolved = resolved.loc[
                        resolved["close_t"].notna() & resolved["close_t5"].notna()
                    ].copy()

                    for _, row in resolved.iterrows():
                        # Vertical bar from close_t to close_t5 at fecha_t5
                        went_up = float(row["close_t5"]) > float(row["close_t"])
                        bar_color = "#22c55e" if went_up else "#ef4444"
                        status_label = row["status"]

                        fig.add_trace(
                            go.Scatter(
                                x=[row["fecha_t5"], row["fecha_t5"]],
                                y=[float(row["close_t"]), float(row["close_t5"])],
                                mode="lines",
                                line=dict(color=bar_color, width=4),
                                name=f"Mov. real ({status_label})",
                                showlegend=False,
                                hovertemplate=(
                                    "%{x|%Y-%m-%d}<br>"
                                    + f"Inicio: ${float(row['close_t']):.2f}<br>"
                                    + f"Fin: ${float(row['close_t5']):.2f}<br>"
                                    + f"Resultado: {status_label}<extra>Movimiento real</extra>"
                                ),
                            )
                        )

    # ── Layout ───────────────────────────────────────────────────────────────
    min_date = history_plot["fecha"].min() if not history_plot.empty else None
    max_date = history_plot["fecha"].max() if not history_plot.empty else None

    # ── Post-gate observed-data zone ─────────────────────────────────────────
    if gate_active and not company_log.empty:
        last_pred_t5 = company_log["fecha_t5"].max()
        if pd.notna(last_pred_t5) and max_date is not None and max_date > last_pred_t5:
            fig.add_vrect(
                x0=last_pred_t5,
                x1=max_date,
                fillcolor="rgba(128,128,128,0.12)",
                line_width=0,
                layer="below",
            )
            fig.add_annotation(
                x=last_pred_t5 + (max_date - last_pred_t5) / 2,
                y=0.95,
                yref="paper",
                text="Predicciones pausadas — reentrenamiento pendiente",
                showarrow=False,
                font=dict(size=12, color="#94a3b8"),
            )
    fig.update_xaxes(
        range=[min_date, max_date],
        minallowed=min_date,
        # maxallowed=max_date,
    )

    fig.update_layout(
        title="Precio histórico, predicciones y movimientos reales",
        xaxis_title="Fecha",
        yaxis_title="Precio ($)",
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
        margin=dict(l=20, r=20, t=60, b=60),
        hovermode="x unified",
    )
    return fig
